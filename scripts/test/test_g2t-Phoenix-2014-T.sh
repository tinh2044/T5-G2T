#!/bin/bash

BATCH_SIZE=4
LOCAL_RANK=0
OUTPUT_DIR="./outputs"
DEVICE="cpu"
SEED=0
FINE_TUNE="./outputs/phoenix-2014-T.pth"
START_EPOCH=0
NUM_WORKERS=0
CONFIG="./configs/phoenix-2014-t.yaml"

# Run the Python script with arguments
python ./train_g2t.py \
    --config $CONFIG \
    --finetune $FINE_TUNE \
    --eval \
