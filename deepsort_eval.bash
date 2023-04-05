#!/usr/bin/env bash

NUM_GPUS=$1
# REID_EXP=$2
# EXP_DIR=$3
ALPHA=$2
# CONFIG=configs/mot/deepsort/bayesod_prob_reid/reid_${REID_EXP}_exp_${EXP_DIR}.py
CONFIG=configs/mot/deepsort/bayesod_with_motion/alpha_0_$ALPHA.py
OUTPUT=/home/results/deepsort/motion

#? Direct eval in test.py
# RESULTS_DIR=$OUTPUT/reid_${REID_EXP}_exp_${EXP_DIR}
RESULTS_DIR=$OUTPUT/bayesod_alpha_0_$ALPHA
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
    --gpu-id 1
fi