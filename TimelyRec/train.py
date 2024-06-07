import sys
import os
import keras
import random as rn
import numpy as np
import tensorflow as tf
# from tensorflow.python.keras.optimizers import Adam
from keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import backend as K
from keras.models import Model
from evaluate import evaluate
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session(config = config)
K.set_session(sess)
import pandas as pd
import math
from sklearn.utils import shuffle
import model as M
import time
from generateNegatives import getNegativeSamples
from TimePreprocessor import timestamp_processor

embedding_size = 32
batch_size = 256
learning_rate = 0.001
patience = 10
n_epoch = 50
sequence_length = 5
width = 128
depth = 4
dropout_rate = 0.1



tr_dataset = pd.read_csv("movielens/train.txt",sep=',',names="user_id,item_id,rating,timestamp".split(",")) 
va_dataset = pd.read_csv("movielens/validation.txt",sep=',',names="user_id,item_id,rating,timestamp".split(","))
te_dataset = pd.read_csv("movielens/test.txt",sep=',',names="user_id,item_id,rating,timestamp".split(","))

userSortedTimestamp = {}
for uid in tr_dataset.user_id.unique().tolist():
    trPosInstance = tr_dataset.loc[tr_dataset['user_id'] == uid]
    temp = va_dataset.loc[va_dataset['user_id'] == uid]
    vaPosInstance = temp.loc[temp['rating'] == 1]

    temp = te_dataset.loc[te_dataset['user_id'] == uid]
    tePosInstance = temp.loc[temp['rating'] == 1]

    posInstance = pd.concat([trPosInstance, vaPosInstance, tePosInstance], ignore_index=True)
    userSortedTimestamp[uid] = posInstance.sort_values(by=['timestamp'])

tr_dataset = timestamp_processor(tr_dataset, userSortedTimestamp, sequence_length)
va_dataset = timestamp_processor(va_dataset, userSortedTimestamp, sequence_length)
te_dataset = timestamp_processor(te_dataset, userSortedTimestamp, sequence_length)

num_users = max(tr_dataset['user_id'])
num_items = max(max(tr_dataset['item_id']), max(va_dataset['item_id']), max(te_dataset['item_id']))

tr_dataset['timestamp_hour'] = (tr_dataset['timestamp'] / 3600).astype(int)

dataset = tr_dataset.groupby('user_id')

userUninteractedItems = {}
userUninteractedTimes = {}
for uid, user_data in dataset:
    userItem = list(user_data['item_id'].unique())
    userTime = list(user_data['timestamp_hour'].unique())
    max_th = max(user_data['timestamp_hour'])
    min_th = min(user_data['timestamp_hour'])

    userUninteractedItems[uid] = list(set(range(1, num_items + 1)) - set(userItem))
    userUninteractedTimes[uid] = list(set(range(min_th, max_th + 1)) - set(userTime))

model = M.TimelyRec([6], num_users, num_items, embedding_size, sequence_length, width, depth, dropout=dropout_rate)

model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=learning_rate))

best_hr30 = 0
best_hr50 = 0
best_recall30 = 0
best_recall50 = 0
best_ndcg50 = 0
best_ndcg30 = 0
best_hr10_i = 0

for epoch in range(n_epoch):
    print ("Epoch " + str(epoch))
    print ("Generating negative samples...")
    t0 = time.time()
    tr_neg_item_dataset, tr_neg_time_dataset, tr_neg_itemtime_dataset = getNegativeSamples(tr_dataset, userUninteractedItems, userUninteractedTimes, num_users, num_items)

    tr_neg_time_dataset = tr_neg_time_dataset.drop(['year', 'month', 'date','hour', 'day_of_week'], axis=1)
    for i in range(sequence_length):
        tr_neg_time_dataset = tr_neg_time_dataset.drop(['month' + str(i), 'date' + str(i), 'hour' + str(i), 'day_of_week' + str(i), 'timestamp' + str(i), 'item_id' + str(i)], axis=1)

    tr_neg_itemtime_dataset = tr_neg_itemtime_dataset.drop(['year', 'month', 'date', 'hour', 'day_of_week'], axis=1)
    for i in range(sequence_length):
        tr_neg_itemtime_dataset = tr_neg_itemtime_dataset.drop(['month' + str(i), 'date' + str(i), 'hour' + str(i), 'day_of_week' + str(i), 'timestamp' + str(i), 'item_id' + str(i)], axis=1)

    tr_neg_time_dataset = timestamp_processor(tr_neg_time_dataset, userSortedTimestamp, sequence_length)
    tr_neg_itemtime_dataset = timestamp_processor(tr_neg_itemtime_dataset, userSortedTimestamp, sequence_length)
    tr_neg_dataset = pd.concat([tr_neg_item_dataset, tr_neg_time_dataset, tr_neg_itemtime_dataset])
    
    tr_posneg_dataset = shuffle(pd.concat([tr_dataset, tr_neg_dataset], join='inner', ignore_index=True))
    print ("Training...")
    t1 = time.time()
    # Train
    for i in range(int(len(tr_posneg_dataset) / batch_size) + 1):
        if (i + 1) * batch_size > len(tr_posneg_dataset):
            tr_batch = tr_posneg_dataset.iloc[i * batch_size : ]
        else:    
            tr_batch = tr_posneg_dataset.iloc[i * batch_size : (i + 1) * batch_size]

        user_input = tr_batch.user_id
        item_input = tr_batch.item_id
        
        recent_month_inputs = []
        recent_day_inputs = []
        recent_date_inputs = []
        recent_hour_inputs = []
        recent_timestamp_inputs = []
        recent_itemid_inputs = []

        month_input = tr_batch.month
        day_input = tr_batch.day_of_week
        date_input = tr_batch.date
        hour_input = tr_batch.hour
        timestamp_input = tr_batch.timestamp

        for j in range(sequence_length):
            recent_month_inputs.append(tr_batch['month' + str(j)])
            recent_day_inputs.append(tr_batch['day_of_week' + str(j)])
            recent_date_inputs.append(tr_batch['date' + str(j)])
            recent_hour_inputs.append(tr_batch['hour' + str(j)])
            recent_timestamp_inputs.append(tr_batch['timestamp' + str(j)])
            recent_itemid_inputs.append(tr_batch['item_id' + str(j)])

        labels = tr_batch.rating
        
        hist = model.fit([user_input, item_input, month_input, day_input, date_input, hour_input, timestamp_input] + [recent_month_inputs[j] for j in range(sequence_length)]+ [recent_day_inputs[j] for j in range(sequence_length)]+ [recent_date_inputs[j] for j in range(sequence_length)]+ [recent_hour_inputs[j] for j in range(sequence_length)]+ [recent_timestamp_inputs[j] for j in range(sequence_length)] + [recent_itemid_inputs[j] for j in range(sequence_length)], labels,
                batch_size=len(tr_batch), epochs=1, verbose=0, shuffle=False)

    print ("Training time: " + str(round(time.time() - t1, 1)))

    print('Iteration %d: loss = %.4f' 
        % (epoch, hist.history['loss'][0]))
    
    print ("Evaluating...")
    t2 = time.time()
    # Evaluation
    HR30, HR50, Recall30, Recall50, NDCG30, NDCG50 = evaluate(model, va_dataset, num_candidates=301, sequence_length=sequence_length)

    print ("Test time: " + str(round(time.time() - t2, 1)))
    print ("Val")
    print ("HR@30   : " + str(round(HR30, 4)))
    print ("HR@50   : " + str(round(HR50, 4)))
    print ("Recall@30  : " + str(round(Recall30, 4)))
    print ("Recall@50  : " + str(round(Recall50, 4)))
    print ("NDCG@30 : " + str(round(NDCG30, 4)))
    print ("NDCG@50: " + str(round(NDCG50, 4)))
    print ("")


    if HR30 > best_hr30:
        best_hr30 = HR30
        best_hr50 = HR50
        best_recall30 = Recall30
        best_recall50 = Recall50
        best_ndcg30 = NDCG30
        best_ndcg50 = NDCG50
        best_hr10_i = epoch
        model.save_weights("saved_model.weights.h5")
        
        
    print ("Best HR@30  : " + str(round(best_hr30, 4)))
    print ("Best HR@50   : " + str(round(best_hr50, 4)))
    print ("Best RECALL@30  : " + str(round(best_recall30, 4)))
    print ("Best RECALL@50  : " + str(round(best_recall50, 4)))
    print ("Best NDCG@30 : " + str(round(best_ndcg30, 4)))
    print ("Best NDCG@50: " + str(round(best_ndcg50, 4)))
    print ('')
    
    if best_hr10_i + patience < epoch:
        exit(1)