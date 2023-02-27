#!/usr/bin/env bash

NUM_GPUS=$1
CONFIG=$2
OUTPUT=$3

# [ -d $OUTPUT ] && rm -rf $OUTPUT
[ -f "${OUTPUT}/latest.pth" ] && CONFIG+="_resume"
CONFIG+=".py"
echo "CONFIG... $CONFIG"

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