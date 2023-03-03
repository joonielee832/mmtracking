#!/usr/bin/env bash

NUM_GPUS=$1
EXP_DIR=prob_reid_test
# CONFIG=configs/mot/deepsort/deepsort_retina_bdd.py
# CONFIG=configs/mot/deepsort/deepsort_faster-rcnn_fpn_4e_mot17-private-half.py
CONFIG=configs/reid/resnet50_prob_b32x8_MOT17.py
# OUTPUT=/home/results/resnet_bdd_reid_train_exp4
#? Direct eval in test.py
RESULTS_DIR=/home/results/$EXP_DIR
EVAL_DIR=$RESULTS_DIR/eval

# CHECKPOINT=$(find $RESULTS_DIR -name 'latest.pth')
# ARGS="--work-dir $RESULTS_DIR/eval \
# --out $RESULTS_DIR/eval/results.pkl \
# --eval track"
ARGS="--work-dir $RESULTS_DIR/eval \
--out $RESULTS_DIR/eval/results.pkl"

[ -d $EVAL_DIR ] && rm -rf $EVAL_DIR
if [[ $NUM_GPUS -gt 1 ]]
then
    bash tools/dist_test.sh $CONFIG $NUM_GPUS $ARGS
else
    python tools/test.py $CONFIG \
    $ARGS \
    --gpu-id 0
fi

# [ -d $EVAL_DIR ] && rm -rf $EVAL_DIR
# CHECKPOINT=$(find $RESULTS_DIR -name 'latest.pth')
# python tools/test.py $CONFIG \
# --checkpoint $CHECKPOINT \
# --work-dir $RESULTS_DIR/eval \
# --out $RESULTS_DIR/eval/results.pkl \
# --gpu-id 0 \
# --eval "mAP"