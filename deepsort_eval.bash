#!/usr/bin/env bash
EXP=$1
GPU_ID=$2
# REID_EXP=$2
# EXP_DIR=$3

# CONFIG=configs/mot/deepsort/bayesod_prob_reid/reid_${REID_EXP}_exp_${EXP_DIR}.py
CONFIG=configs/mot/deepsort/bayesod_with_motion/alpha_0_${EXP}_${GPU_ID}.py
OUTPUT=/home/results/deepsort/bayesod_with_motion

#? Direct eval in test.py
# RESULTS_DIR=$OUTPUT/reid_${REID_EXP}_exp_${EXP_DIR}
RESULTS_DIR=$OUTPUT/alpha_0_${EXP}
EVAL_DIR=$RESULTS_DIR/eval

# CHECKPOINT=$(find $RESULTS_DIR -name 'latest.pth')
ARGS="--work-dir $EVAL_DIR \
--show-dir $EVAL_DIR/show \
--out $EVAL_DIR/results.pkl \
--eval track"

[ -d $EVAL_DIR ] && rm -rf $EVAL_DIR
python3 tools/test.py $CONFIG \
    $ARGS \
    --gpu-id $GPU_ID