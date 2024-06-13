#!/bin/bash

set -e
# source activate env1

cd /home/yenlh/temporal_recommendation/ASReP

LOG_FILE="ASReP.log"

python main.py --dataset=Beauty --train_dir=default --lr=0.001 --hidden_units=128 --maxlen=100 --dropout_rate=0.7 --num_blocks=2 --l2_emb=0.0 --num_heads=4 --evalnegsample 100 --reversed 1 --reversed_gen_num 20 --M 20 --num_epochs 2 2>&1 | while IFS= read -r line; do echo "$(date '+%Y-%m-%d %H:%M:%S')- $line" done | tee -a "$LOG_FILE"

python main.py --dataset=Beauty --train_dir=default --lr=0.001 --hidden_units=128 --maxlen=100 --dropout_rate=0.7 --num_blocks=2 --l2_emb=0.0 --num_heads=4 --evalnegsample 100 --reversed_pretrain 1 --aug_traindata 15 --M 18 --num_epochs 2 2>&1 | while IFS= read -r line; do echo "$(date '+%Y-%m-%d %H:%M:%S')- $line" done | tee -a "$LOG_FILE"
