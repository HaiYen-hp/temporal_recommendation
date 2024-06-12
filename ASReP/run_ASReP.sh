#!/bin/bash

set -e
# source activate env1

cd /home/yenlh/temporal_recommendation/ASReP

# LOG_FILE="ASReP.log"

# log() {
#     local message="$1"
#     local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
#     echo "[$timestamp] $message" >> "$LOG_FILE"
# }

# python_output1=$(python main.py --dataset=Beauty --train_dir=default --lr=0.001 --hidden_units=128 --maxlen=100 --dropout_rate=0.7 --num_blocks=2 --l2_emb=0.0 --num_heads=4 --evalnegsample 100 --reversed 1 --reversed_gen_num 20 --M 20 --num_epochs 1 2>&1)

# python_output2=$(python main.py --dataset=Beauty --train_dir=default --lr=0.001 --hidden_units=128 --maxlen=100 --dropout_rate=0.7 --num_blocks=2 --l2_emb=0.0 --num_heads=4 --evalnegsample 100 --reversed_pretrain 1 --aug_traindata 15 --M 18 --num_epochs 1 2>&1)

# log "$python_output1"
# log "$python_output2"

# echo "Logs have been written to $LOG_FILE"

python main.py --dataset=Beauty --train_dir=default --lr=0.001 --hidden_units=128 --maxlen=100 --dropout_rate=0.7 --num_blocks=2 --l2_emb=0.0 --num_heads=4 --evalnegsample 100 --reversed 1 --reversed_gen_num 20 --M 20 --num_epochs 1 | tee log_ASReP.txt

python main.py --dataset=Beauty --train_dir=default --lr=0.001 --hidden_units=128 --maxlen=100 --dropout_rate=0.7 --num_blocks=2 --l2_emb=0.0 --num_heads=4 --evalnegsample 100 --reversed_pretrain 1 --aug_traindata 15 --M 18 --num_epochs 1 | tee log_ASReP.txt
