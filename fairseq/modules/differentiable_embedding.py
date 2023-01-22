import torch
import torch.nn as nn

class DifferentiableEmbedding(nn.Embedding):
    """ Differentiable embedding module
    """
    retain_input_grad: bool = False
    one_hot: torch.Tensor = None
        
    def input_grad(self):
        if self.one_hot is not None:
            return self.one_hot.grad
        else:
            return None
        
    def unset_input_grad_(self):
        if self.one_hot is not None:
            self.one_hot.grad = None
    
    def retain_input_grad_(self, retain: bool = True):
        self.retain_input_grad = retain
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bsz, tsz, dim = input.shape, self.weight.shape[1]
        self.one_hot = nn.functional.one_hot(input, num_classes=self.weight.shape[0]).dtype(self.weight.dtype)
        
        if self.retain_input_grad:
            self.one_hot.requires_grad_(True)
            self.one_hot.retain_grad()
        
        return torch.bmm(self.one_hot.view(-1), self.weight).view(bsz, tsz, dim)
