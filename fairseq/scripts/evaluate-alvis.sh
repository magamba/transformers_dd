#!/usr/bin/env bash

name="checkpoints_transformers"
workers=0
base_model="transformer_vaswani"
seeds=(
    1
    2
    3
)

# array size:
#     islwt'14: 0-126:2%16
#     wmt'14: 1-127:2%16

splits=(
    "train"
    "valid"
    "test"
)

checkpoints=(
    "last"
)

for seed in ${seeds[@]}; do
    for split in ${splits[@]}; do
        for checkpoint in ${checkpoints[@]}; do
            sbatch --export NAME=$name,BASE_MODEL=$base_model,WORKERS=$workers,SEED=$seed,SPLIT=$split,CHECKPOINT=$checkpoint \
                   --array=0-126:2%16 \
                   exps/evaluate-alvis.sbatch

            sbatch --export NAME=$name,BASE_MODEL=$base_model,WORKERS=$workers,SEED=$seed,SPLIT=$split,CHECKPOINT=$checkpoint \
                   --array=1-127:2%16 \
                   exps/evaluate-alvis.sbatch
        done
    done
done
