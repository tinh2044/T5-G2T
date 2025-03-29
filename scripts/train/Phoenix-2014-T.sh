#!/bin/bash

BATCH_SIZE=4
EPOCHS=80
WORLD_SIZE=1
DIST_URL="env://"
LOCAL_RANK=0
OPT="adamw"
OPT_EPS=1e-9
OPT_BETAS="0.9 0.98"
# CLIP_GRAD="norm"
MOMENTUM=0.9
WEIGHT_DECAY=0.0
SCHED="cosine"
LR=1e-3
LR_NOISE=""
LR_NOISE_PCT=0.67
LR_NOISE_STD=1.0
WARMUP_LR=1e-6
MIN_LR=1e-8
DECAY_EPOCHS=30
WARMUP_EPOCHS=0
COOLDOWN_EPOCHS=10
PATIENCE_EPOCHS=10
DECAY_RATE=0.1
OUTPUT_DIR=""
DEVICE="cpu"
SEED=0
RESUME=""
START_EPOCH=0
EVAL=""
NUM_WORKERS=0
PIN_MEM="--pin-mem"
CONFIG="./configs/phoenix-2014-t.yaml"
INPUT_SIZE=224
RESIZE=256

# Run the Python script with arguments
python ./train_clip.py \
    --config $CONFIG \

    --batch-size $BATCH_SIZE \
    --epochs $EPOCHS \
    --world_size $WORLD_SIZE \
    --dist_url $DIST_URL \
    --local_rank $LOCAL_RANK \
    --opt $OPT \
    --opt-eps $OPT_EPS \
    --opt-betas $OPT_BETAS \
    # --clip-grad $CLIP_GRAD \
    --momentum $MOMENTUM \
    --weight-decay $WEIGHT_DECAY \
    --sched $SCHED \
    --lr $LR \
    --lr-noise $LR_NOISE \
    --lr-noise-pct $LR_NOISE_PCT \
    --lr-noise-std $LR_NOISE_STD \
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
    --resume $RESUME \
    --num_workers $NUM_WORKERS \
