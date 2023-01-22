#! /bin/sh

source scripts/setup_env

fairseq-eval-lm "$DATA_DIR/data-bin/iwslt14.tokenized.de-en" \
    --path "$SAVE_DIR/transformer_vaswani_8/iwslt14_de_en/default/seed-1/checkpoint_290.pt" \
    --seed 1 \
    --gen-subset train \
    --results-path "$SAVE_DIR/checkpoints_transformers/transformer_vaswani_8/iwslt14_de_en/default/seed-1/jacobian-train-checkpoint-290.json" \
    --task translation \
    --checkpoint 290 \
    --metric jacobian
