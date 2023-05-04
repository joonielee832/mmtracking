#!/usr/bin/env bash

EXP_DIR=$1
CONFIG=configs/det/retinanet/prob_MOT17/exp$EXP_DIR.py
OUTPUT=/home/results/bayesod_eval/

#? Direct eval in test.py
RESULTS_DIR=$OUTPUT/$EXP_DIR
CHECKPOINT_DIR=/home/results/bayesod_mot17det_train_exp$EXP_DIR
EVAL_DIR=$RESULTS_DIR/eval

CHECKPOINT=$(find $CHECKPOINT_DIR -name 'best_bbox_mAP_epoch_*.pth')
ARGS="--work-dir $RESULTS_DIR/eval \
--checkpoint $CHECKPOINT \
--out $RESULTS_DIR/eval/results.pkl \
--eval bbox"

[ -d $EVAL_DIR ] && rm -rf $EVAL_DIR
python3 tools/test.py $CONFIG \
    $ARGS