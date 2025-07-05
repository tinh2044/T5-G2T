#!/bin/bash

BATCH_SIZE=1
EPOCHS=80
WORLD_SIZE=1
DIST_URL="env://"
LOCAL_RANK=0
OPT="adamw"
OPT_EPS=1e-9
OPT_BETAS="0.9 0.98"
MOMENTUM=0.9
WEIGHT_DECAY=0.0
SCHED="cosine"
LR=1e-3
WARMUP_LR=1e-6
MIN_LR=1e-8
DECAY_EPOCHS=30
WARMUP_EPOCHS=0
COOLDOWN_EPOCHS=10
PATIENCE_EPOCHS=10
DECAY_RATE=0.1
OUTPUT_DIR="./outputs/clip/phoenix-2014-T"
DEVICE="cpu"
SEED=0
RESUME=""
FINETUNE=""
NUM_WORKERS=0
CONFIG="./configs/phoenix-2014-t.yaml"

# Weights & Biases Configuration
USE_WANDB="--use_wandb"
WANDB_PROJECT="T5-G2T"
WANDB_RUN_NAME="clip_phoenix_2014_T"

# Run the Python script with arguments
python -m train_clip \
    --config $CONFIG \
    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --world_size $WORLD_SIZE \
    --dist_url $DIST_URL \
    --local_rank $LOCAL_RANK \
    --opt $OPT \
    --opt-eps $OPT_EPS \
    --opt-betas $OPT_BETAS \
    --weight-decay $WEIGHT_DECAY \
    --sched $SCHED \
    --lr $LR \
    --warmup-lr $WARMUP_LR \
    --min-lr $MIN_LR \
    --decay-epochs $DECAY_EPOCHS \
    --warmup-epochs $WARMUP_EPOCHS \
    --cooldown-epochs $COOLDOWN_EPOCHS \
    --patience-epochs $PATIENCE_EPOCHS \
    --decay-rate $DECAY_RATE \
    --output_dir $OUTPUT_DIR \
    --device $DEVICE \
    --seed $SEED \
    --finetune "$FINETUNE" \
    --num_workers $NUM_WORKERS \
    $USE_WANDB \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name $WANDB_RUN_NAME \
