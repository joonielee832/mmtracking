#!/usr/bin/env bash

NUM_GPUS=$1
EXP_DIR=prob_reid_mot17_train_exp18
CONFIG=configs/reid/prob_MOT17_attenuated/exp18.py
OUTPUT=/home/results

#? Direct eval in test.py
RESULTS_DIR=$OUTPUT/$EXP_DIR
EVAL_DIR=$RESULTS_DIR/eval

CHECKPOINT=$(find $RESULTS_DIR -name 'latest.pth')
ARGS="--work-dir $RESULTS_DIR/eval \
--checkpoint $CHECKPOINT \
--out $RESULTS_DIR/eval/results.pkl \
--eval mAP"

[ -d $EVAL_DIR ] && rm -rf $EVAL_DIR
if [[ $NUM_GPUS -gt 1 ]]
then
    bash tools/dist_test.sh $CONFIG $NUM_GPUS $ARGS
else
    python tools/test.py $CONFIG \
    $ARGS \
    --gpu-id 0
fi