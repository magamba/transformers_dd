#! /bin/bash

#SBATCH -A berzelius-2022-53
#SBATCH --gpus=8
#SBATCH -t 2-0:00:00
#SBATCH --mail-type END,FAIL
#SBATCH --mail-user mgamba@kth.se
#SBATCH --output /proj/memorization/logs/%A_%a.out
#SBATCH --error /proj/memorization/logs/%A_%a.err

source scripts/setup_env
workers=${WORKERS}

echo "Job starting at: $(date)"

#     SLURM_ARRAY_TASK_ID to TRAIN_CONFIG lookup table
#   +--------------+-----------+-----------+
#   |   DATASET    | IWSLT'14 |  WMT'14  |
#   | / EMBED DIM  |  De - En |  En - Fr |
#   +--------------+----------+----------+
#   |       14     |     0    |     1    |
#   +--------------+----------+----------+
#   |      ...     |   2i + j |    ...   |
#   +--------------+----------+----------+
#   |      500     |    970   |    971   |
#   +--------------+----------+----------+

###########################
###    MODEL  WIDTHS    ###
###########################

widths=(
    {14..500}
)

###########################
###        SEEDS        ###
###########################

seed=${SEED}

###########################
###      DATASETS       ###
###########################

datasets=(
     "iwslt14.tokenized.de-en"
     "wmt14_en_fr"
)

# To add more datasets, edit the array above.
#
# Don't forget to exted the SLURM_ARRAY_TASK_ID lookup table
# accordingly, by adding rows to it, and to increase the slurm job
# array size. Finally, edit line 128 to switch between datasets.
#


###########################
###    TRAINING ARGS    ###
###########################

declare -A batch_sizes=(
    ["iwslt14.tokenized.de-en"]=4096
    ["wmt14_en_fr"]=3000
)

declare -A training_args=(
    ["iwslt14.tokenized.de-en"]="--share-decoder-input-output-embed"
    ["wmt14_en_fr"]="--share-all-embeddings"
)

args=" --max-tokens=${batch_sizes[$dataset]} \
       ${training_args[$dataset]} \
     "
common_args=" --max-update 80000 \
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
              --keep-interval-updates-pattern 1 \
            "

###########################
###  DON'T EDIT BELOW   ###
###########################

job_array_max_size=$((${#widths[@]} * ${#datasets[@]}))
if [ ${SLURM_ARRAY_TASK_ID} -gt $job_array_max_size ]; then
    echo "It seems like your slurm job array is too large."
    echo "Ignoring SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}."
    echo "Job array max size: $job_array_max_size"
    exit 1
fi

width=${widths[$((${SLURM_ARRAY_TASK_ID} /  ${#datasets[@]}))]}
dataset=${datasets[$((${SLURM_ARRAY_TASK_ID} % ${#datasets[@]}))]}

data="$DATA_DIR/data-bin/$dataset"
model="${BASE_MODEL}""_"$width
save_dir="$SAVE_DIR/${NAME}/$model/$dataset/default/seed-$seed"
logdir=$save_dir
logfile="$logdir/train.log"
tensorboard_dir=$logdir

# compose arguments
args="$args $common_args"

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "model: $model"
echo "dataset: $dataset"
echo "seed: $seed"
echo "args: $args"

# If no seed is provided, then pick one at random
if [ -z "$seed" ]; then
    seed=$RANDOM
fi

fairseq-train "$data"                       \
    --arch="$model"                         \
    --seed="$seed"                          \
    --log-format "simple"                   \
    --log-file $logfile                     \
    --tensorboard-logdir "$tensorboard_dir" \
    $args                                   \
    "$@"

echo "Job ended at: $(date)"
