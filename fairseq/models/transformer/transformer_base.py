# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from functorch import make_functional_with_buffers, jvp, vjp, jacrev, jacfwd

import logging

from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqEncoderDecoderModel
from fairseq.models.transformer import (
    TransformerConfig,
    TransformerDecoderBase,
    TransformerEncoderBase,
)


logger = logging.getLogger(__name__)


class TransformerModelBase(FairseqEncoderDecoderModel):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    def __init__(self, cfg, encoder, decoder):
        super().__init__(encoder, decoder)
        self.cfg = cfg
        self.supports_align_args = True

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, TransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        # --  TODO T96535332
        #  bug caused by interaction between OmegaConf II and argparsing
        cfg.decoder.input_dim = int(cfg.decoder.input_dim)
        cfg.decoder.output_dim = int(cfg.decoder.output_dim)
        # --

        if cfg.encoder.layers_to_keep:
            cfg.encoder.layers = len(cfg.encoder.layers_to_keep.split(","))
        if cfg.decoder.layers_to_keep:
            cfg.decoder.layers = len(cfg.decoder.layers_to_keep.split(","))

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if cfg.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if cfg.encoder.embed_dim != cfg.decoder.embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if cfg.decoder.embed_path and (
                cfg.decoder.embed_path != cfg.encoder.embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        elif cfg.merge_src_tgt_embed:
            logger.info(f"source dict size: {len(src_dict)}")
            logger.info(f"target dict size: {len(tgt_dict)}")
            src_dict.update(tgt_dict)
            task.src_dict = src_dict
            task.tgt_dict = src_dict
            logger.info(f"merged dict size: {len(src_dict)}")
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim
            )
            decoder_embed_tokens = encoder_embed_tokens
            cfg.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                cfg, src_dict, cfg.encoder.embed_dim, cfg.encoder.embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                cfg, tgt_dict, cfg.decoder.embed_dim, cfg.decoder.embed_path
            )
        if cfg.offload_activations:
            cfg.checkpoint_activations = True  # offloading implies checkpointing
        encoder = cls.build_encoder(cfg, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(cfg, tgt_dict, decoder_embed_tokens)
        return cls(cfg, encoder, decoder)

    @classmethod
    def build_embedding(cls, cfg, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()

        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        # if provided, load from preloaded dictionaries
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, cfg, src_dict, embed_tokens):
        return TransformerEncoderBase(cfg, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, cfg, tgt_dict, embed_tokens):
        return TransformerDecoderBase(
            cfg,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=cfg.no_cross_attention,
        )

    # TorchScript doesn't support optional arguments with variable length (**kwargs).
    # Current workaround is to add union of all arguments in child classes.
    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Run the forward pass for an encoder-decoder model.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        """
        encoder_out = self.encoder(
            src_tokens, src_lengths=src_lengths, return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    # Since get_normalized_probs is in the Fairseq Model which is not scriptable,
    # I rewrite the get_normalized_probs from Base Class to call the
    # helper function in the Base Class.
    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)

    @classmethod
    def power_method(self, src_tokens, jvp, vjp, max_steps=60):
        embedding_grad = self.encoder.embed_tokens.weight
        v = torch.randn_like(src_tokens).type(embedding_grad.dtype)
        v /= torch.norm(v, p=2, dim=-1).unsqueeze(-1)
        v = torch.matmul(weight.T, v.T).T
        
        sigma_prev = torch.zeros_like(src_tokens[:,:]).type(embedding_grad.dtype)
        for _ in range(max_steps):
            u = jvp(v)
            u /= torch.norm(u, p=2, dim=-1).unsqueeze(-1)
            v = vjp(u)
            v /= torch.norm(v, p=2, dim=-1).unsqueeze(-1)
            sigma = torch.matmul(u.T, u)
            
            if torch.all(torch.abs(sigma - sigma_prev) < 1e-2):
                break
        
        return sigma

    def compute_embbedings(self, src_tokens):
        x, embeddings = self.encoder.forward_embedding(src_tokens)
        return x, embeddings

    def operator_norm_implicit(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        max_steps: int = 60,
    ):
        """
        Run the forward pass for an encoder-decoder model in embedding space.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        
        Steps:
        
        1. pick random v
        2. x = Jv
        3. x = W^T W x
        4. x = J^T x
        5. take grad w.r.t v
        6. x = W^T x W
        7. norm = sqrt(trace(x))
        
        """
        
        embedded_model = EmbeddedModel(self.encoder, self.decoder, return_all_hiddens)
        
        model_fn, params, buffers = make_functional_with_buffers(embedded_model)
        x, embedding  = self.encoder.forward_embedding(src_tokens)

        decoder_adj = torch.matmul(self.decoder.embed_tokens.weight.T, self.decoder.embed_tokens.weight)
        def model_forward(x):
            return  model_fn(params, buffers, src_tokens, src_lengths, prev_output_tokens, embedding, x)[0].reshape(-1, decoder_adj.shape[-1])

        logger.info(f"x: {x.shape}")
        logger.info(f"model_forward(x): {model_forward(x).shape}")

        logger.info(f"decoder_adj: {decoder_adj.shape}")
        encoder_w = self.encoder.embed_tokens.weight

        def jvp_fn(v):
            jv = jvp(model_forward, (x,), (v,))[1]
            logger.info(f"jv: {jv.shape}")
            return jv

        def jvp_embed_fn(v):
            jv = jvp_fn(v)
            embed_jv = torch.matmul(decoder_adj, jv.T).T
            logger.info(f"W_adj * jv: {embed_jv.shape}")
            return embed_jv

        def vjp_fn(v):
            jv = jvp_embed_fn(v)
            vj = vjp(model_forward, x)[1](jv)[0]
            logger.info(f"vj: {vj.shape}")
            return vj

        #jvp_fn = lambda v: jvp(model_forward, (x,), (v,))[1][0].reshape(-1,decoder_adj.shape[1])
        #jvp_embed_fn = lambda v: torch.matmul(decoder_adj.unsqueeze(0), jvp_fn(v))
        #vjp_fn = lambda u: vjp(model_forward, x)[1](jvp_embed_fn(u))[0]
        dummy = torch.ones_like(x)
        
        inner_product_fn = lambda delta: jacrev(vjp_fn)(delta)
        inner_product = inner_product_fn(dummy)
        
        inner_product = torch.matmul(encoder_w.T, inner_product.T).T
        outer_product = torch.matmul(encoder_w.T, inner_product)
        
        op_norm = torch.sqrt(outer_product.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1))
                
        #op_norm = self.power_method(src_tokens, jvp_fn, vjp_fn)
        
        return op_norm
    
    def operator_norm_implicit_hessian(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        max_steps: int = 60,
    ):
        """
        Run the forward pass for an encoder-decoder model in embedding space.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        
        Steps:
        
        1. pick random v
        2. x = Jv
        3. x = W^T W x
        4. x = v^TJ^T x
        5. take grad w.r.t. v
        6. take grad w.r.t. v
        7. x = 0.5 W^T x W
        7. norm = sqrt(trace(x))
        
        """
        
        embedded_model = EmbeddedModel(self.encoder, self.decoder, return_all_hiddens)
        
        model_fn, params, buffers = make_functional_with_buffers(embedded_model)
        x, embedding  = self.encoder.forward_embedding(src_tokens)

        decoder_adj = torch.matmul(self.decoder.embed_tokens.weight.T, self.decoder.embed_tokens.weight)
        def model_forward(x):
            return  model_fn(params, buffers, src_tokens, src_lengths, prev_output_tokens, embedding, x)[0].reshape(-1, decoder_adj.shape[-1])

        logger.info(f"x: {x.shape}")
        logger.info(f"model_forward(x): {model_forward(x).shape}")

        logger.info(f"decoder_adj: {decoder_adj.shape}")
        encoder_w = self.encoder.embed_tokens.weight

        def jvp_fn(v):
            jv = jvp(model_forward, (x,), (v,))[1]
            logger.info(f"jv: {jv.shape}")
            return jv

        def jvp_embed_fn(v):
            jv = jvp_fn(v)
            embed_jv = torch.matmul(decoder_adj, jv.T).T
            logger.info(f"W_adj * jv: {embed_jv.shape}")
            return embed_jv

        def vjvp_fn(v):
            jv = jvp_embed_fn(v)
            vj = torch.bmm(jv.unsqueeze(-2), jv.unsqueeze(-1)).squeeze()
            logger.info(f"vj: {vj.shape}")
            return vj

        #jvp_fn = lambda v: jvp(model_forward, (x,), (v,))[1][0].reshape(-1,decoder_adj.shape[1])
        #jvp_embed_fn = lambda v: torch.matmul(decoder_adj.unsqueeze(0), jvp_fn(v))
        #vjp_fn = lambda u: vjp(model_forward, x)[1](jvp_embed_fn(u))[0]
        dummy = torch.ones_like(x)
        
        inner_product_fn = lambda delta: 0.5 * jacfwd(jacrev(vjvp_fn))(delta)
        inner_product = inner_product_fn(dummy)
        
        inner_product = torch.matmul(encoder_w.T, inner_product.T).T
        outer_product = torch.matmul(encoder_w.T, inner_product)
        
        op_norm = torch.sqrt(outer_product.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1))
        return op_norm
        
    def operator_norm(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        max_steps: int = 60,
    ):
        """
        Run the forward pass for an encoder-decoder model in embedding space.

        Copied from the base class, but without ``**kwargs``,
        which are not supported by TorchScript.
        
        Steps:
        
        1. pick random v
        2. x = Jv
        3. x = W^T W x
        4. x = v^TJ^T x
        5. take grad w.r.t. v
        6. take grad w.r.t. v
        7. x = 0.5 W^T x W
        7. norm = sqrt(trace(x))
        
        """
        
        embedded_model = EmbeddedModel(self.encoder, self.decoder, return_all_hiddens)
        
        model_fn, params, buffers = make_functional_with_buffers(embedded_model)
        x, embedding  = self.encoder.forward_embedding(src_tokens)

        decoder_adj = torch.matmul(self.decoder.embed_tokens.weight.T, self.decoder.embed_tokens.weight)
        def model_forward(x):
            return  model_fn(params, buffers, src_tokens, src_lengths, prev_output_tokens, embedding, x)[0].reshape(-1, decoder_adj.shape[-1])

        logger.info(f"x: {x.shape}")
        logger.info(f"model_forward(x): {model_forward(x).shape}")

        def jvp_fn(v):
            jv = jvp(model_forward, (x,), (v,))[1]
            logger.info(f"jv: {jv.shape}")
            return jv

        def vjvp_fn(v):
            jv = jvp_fn(v)
            vjv = torch.bmm(jv.unsqueeze(-2), jv.unsqueeze(-1)).squeeze()
            logger.info(f"vjv: {vjv.shape}")
            return vjv

        def contract(v):
            return torch.mean(
                torch.sqrt(
                    0.5 * torch.trace(vjvp_fn(v).unsqueeze(-1))
                ), 
                dim=0,
            )

        #jvp_fn = lambda v: jvp(model_forward, (x,), (v,))[1][0].reshape(-1,decoder_adj.shape[1])
        #jvp_embed_fn = lambda v: torch.matmul(decoder_adj.unsqueeze(0), jvp_fn(v))
        #vjp_fn = lambda u: vjp(model_forward, x)[1](jvp_embed_fn(u))[0]
        dummy = torch.ones_like(x)
        
        inner_product_fn = lambda delta: jacfwd(jacrev(contract))(delta)
        inner_product = inner_product_fn(dummy)
        
        op_norm = inner_product
        return op_norm


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

    
class EmbeddedModel(nn.Module):
    def __init__(self, encoder, decoder, return_all_hiddens):
        super(EmbeddedModel, self).__init__()
        self.encoder = encoder.cuda()
        self.decoder = decoder.cuda()
        self.return_all_hiddens = return_all_hiddens
        
    def forward(self, src_tokens, src_lengths, prev_output_tokens, embedding, x):
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, return_all_hiddens=self.return_all_hiddens, encoder_embedding=embedding, x=x)
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=True,
            alignment_layer=None,
            alignment_heads=None,
            src_lengths=src_lengths,
            return_all_hiddens=self.return_all_hiddens,
        )
        return decoder_out
        
