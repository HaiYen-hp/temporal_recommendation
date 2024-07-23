#!/bin/bash

set -e
# source activate env1

cd /home/yenlh/temporal_recommendation/ASReP/pipeline

LOGDIR="./log"
LOGFILE="$LOGDIR/$(date '+%Y-%m-%d_%H-%M-%S').log"

# Ensure the log file name is not empty
if [ -z "$LOGFILE" ]; then
    echo "Log file name is empty. Exiting."
    exit 1
fi

# Create the directory if it does not exist
if [ ! -d "$LOGDIR" ]; then
    mkdir -p "$LOGDIR"
    echo "Directory $LOGDIR created."
else
    echo "Directory $LOGDIR already exists."
fi

python hyperparameter_tuning.py --dataset=Beauty --train_dir=default --gpus=1,2,3 --evalnegsample 100 --reversed 1 --reversed_gen_num 26 --M 20 2>&1 | while IFS= read -r line; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $line"
done | tee -a "$LOGFILE"


python training_data.py --dataset=Beauty --train_dir=default --gpus=1,2,3 --evalnegsample 100 --reversed_pretrain 1 --aug_traindata 16 --M 20 2>&1 | while IFS= read -r line; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $line"
done | tee -a "$LOGFILE"
