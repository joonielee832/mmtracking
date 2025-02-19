#!/usr/bin/env bash

NUM_GPUS=$1
EXP_DIR=deepsort_bdd_eval_exp1
CONFIG=configs/mot/deepsort/deepsort_retina_bdd.py
# OUTPUT=/home/results/resnet_bdd_reid_train_exp4
#? Direct eval in test.py
RESULTS_DIR=/home/results/$EXP_DIR
EVAL_DIR=$RESULTS_DIR/eval

# CHECKPOINT=$(find $RESULTS_DIR -name 'latest.pth')
ARGS="--work-dir $RESULTS_DIR/eval \
--out $RESULTS_DIR/eval/results.pkl \
--eval track"

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