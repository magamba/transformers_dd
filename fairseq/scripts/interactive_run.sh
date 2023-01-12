#! /bin/sh

source scripts/setup_env

fairseq-train $DATA_DIR/data-bin/iwslt14.tokenized.de-en \
    --arch "transformer_vaswani_32" \
    --seed 1 \
    --log-format "simple" \
    --log-file "train.log" \
    --tensorboard-logdir "tensorboard" \
    --max-tokens=4096 \
    --share-decoder-input-output-embed \
    --max-update 80000 \
    --optimizer adam \
    --adam-betas '(0.9, 0.98)' \
    --clip-norm 0.0 \
    --lr 0.0005 \
    --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --warmup-init-lr 1e-07 \
    --dropout 0.0 \
    --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --save-interval 5 \
    --keep-interval-updates 40 \
    --keep-interval-updates-pattern 1

