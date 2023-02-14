#!/usr/bin/env bash

NUM_GPUS=$1
# CONFIG=configs/det/retinanet/retinanet_r50_fpn_mot17.py
CONFIG=configs/reid/resnet50_b32x8_MOT17.py
OUTPUT=/home/results/resnet_mot17_reid_train_exp1

[ -d $OUTPUT ] && rm -rf $OUTPUT

ARGS="--work-dir $OUTPUT \
--seed 0"

if [[ $NUM_GPUS -gt 1 ]]
then
    bash ./tools/dist_train.sh $CONFIG $NUM_GPUS $ARGS
else
    python ./tools/train.py $CONFIG \
    $ARGS \
    --gpu-id 0
fi