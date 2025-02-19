#!/usr/bin/env bash

NUM_GPUS=$1
# CONFIG=configs/det/retinanet/retinanet_r50_fpn_bdd.py
CONFIG=configs/reid/resnet50_b32x8_bdd.py
# CONFIG=configs/det/faster-rcnn/faster-rcnn_r50_fpn_4e_bdd.py
# CONFIG=configs/det/faster-rcnn/faster-rcnn_r50_fpn_4e_mot17.py
OUTPUT=/home/results/resnet_bdd_reid_train_exp4

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