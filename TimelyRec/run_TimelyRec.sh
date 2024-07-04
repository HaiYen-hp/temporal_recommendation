#!/bin/bash

set -e
# source activate env1

cd /home/yenlh/temporal_recommendation/TimelyRec

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

CUDA_VISIBLE_DEVICES=1,2 python train.py --train_dir movielens --lr=0.0026 --embedding_size=64 --dropout_rate=0.15 --num_epochs 5 2>&1 | while IFS= read -r line; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $line"
done | tee -a "$LOGFILE"
