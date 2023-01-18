#!/usr/bin/env bash

name="checkpoints_transformers"
workers=0
base_model="transformer_vaswani"
seed=3

# array size:
#     islwt'14: 0-126:2%16
#     wmt'14: 1-127:2%16

sbatch --export NAME=$name,BASE_MODEL=$base_model,WORKERS=$workers,SEED=$seed \
       --array=0-126:2%16 \
       exps/train-epoch_wise-alvis.sbatch

sbatch --export NAME=$name,BASE_MODEL=$base_model,WORKERS=$workers,SEED=$seed \
       --array=1-127:2%16 \
       exps/train-epoch_wise-alvis.sbatch
