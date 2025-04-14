#!/bin/bash

BATCH_SIZE=4
LOCAL_RANK=0
OUTPUT_DIR="./outputs/clip/Phoenix-2014-T"
DEVICE="cpu"
SEED=0
FINE_TUNE=" "
START_EPOCH=0
NUM_WORKERS=0
CONFIG="./configs/phoenix-2014-t.yaml"

# Run the Python script with arguments
python ./train_clip.py \
    --config $CONFIG \
    --finetune $FINE_TUNE \
    --eval
