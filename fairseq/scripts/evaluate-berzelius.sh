#!/usr/bin/env bash

name="checkpoints_transformers"
workers=0
base_model="transformer_vaswani"
seeds=(
    1
#    2
#    3
)

splits=(
    "train"
#    "test"
#    "valid"
)

declare -A checkpoints=(
   ["iwslt14_de_en"]=290
   ["wmt14_en_fr"]=115
)

metrics=(
#    "perplexity"
    "jacobian"
)

# array size:
#     islwt'14: 0-126:2%16
#     wmt'14: 1-127:2%16

for seed in ${seeds[@]}; do
    for split in ${splits[@]}; do
        for metric in ${metrics[@]}; do
            for checkpoint in ${checkpoints["iwslt14_de_en"]}; do
                sbatch --export NAME=$name,BASE_MODEL=$base_model,WORKERS=$workers,SEED=$seed,SPLIT=$split,CHECKPOINT=$checkpoint,METRIC=$metric \
                       --array=0 \
                       exps/evaluate-berzelius.sbatch
            done
            for checkpoint in ${checkpoints["wmt14_en_fr"]}; do
                sbatch --export NAME=$name,BASE_MODEL=$base_model,WORKERS=$workers,SEED=$seed,SPLIT=$split,CHECKPOINT=$checkpoint,METRIC=$metric \
                       --array=1 \
                       exps/evaluate-berzelius.sbatch
            done
        done
    done
done
