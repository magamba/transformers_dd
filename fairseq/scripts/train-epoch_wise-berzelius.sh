#!/usr/bin/env bash

name="checkpoints_transformers"
workers=0
base_model="transformer_vaswani"
seed=1

# array size: 0-970:2%8

sbatch --export NAME=$name,BASE_MODEL=$base_model,WORKERS=$workers,SEED=$seed \
       --array=0-970:2%8 \
       ../exps/train-epoch_wise-berzelius.sbatch
