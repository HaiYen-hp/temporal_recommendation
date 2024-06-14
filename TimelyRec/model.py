import os
import keras
import random as rn
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Activation, Embedding
from keras.layers import Input, Flatten, dot, concatenate, Dropout
from tensorflow.python.keras import backend as K
from keras.models import Model
from keras.layers import Layer
from keras import initializers

from TemporalPositionEncoding import PositionalEncoding

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session(config = config)
K.set_session(sess)


class SurroundingSlots(Layer):
    def __init__(self, window_length, max_range, trainable=True, name=None, **kwargs):
        super(SurroundingSlots, self).__init__(name=name, trainable=trainable, **kwargs)
        self.window_length = window_length
        self.max_range = max_range

    # def build(self, inshape):
    #     1

    def call(self, x):
        surr = K.cast(x, dtype=tf.int32) + K.arange(start=-self.window_length, stop=self.window_length + 1, step=1)
        surrUnderflow = K.cast(surr < 0, dtype=tf.int32)
        surrOverflow = K.cast(surr > self.max_range - 1, dtype=tf.int32)

        return surr * (-(surrUnderflow + surrOverflow) + 1) + surrUnderflow * (surr + self.max_range) + surrOverflow * (surr - self.max_range)

    def compute_output_shape(self, inshape):
        return (inshape[0], self.window_length * 2 + 1)


class MATE(Layer):
    def __init__(self, dimension, trainable=True, name=None, **kwargs):
        super(MATE, self).__init__(name=name, trainable=trainable, **kwargs)
        self.dimension = dimension

    def build(self, inshape):
        # for multiplicative attention
        self.W = self.add_weight(name="W", shape=(self.dimension, self.dimension), initializer=initializers.get("random_normal"))
     
        # for personalization
        self.Wmonth = self.add_weight(name="Wmonth", shape=(self.dimension, self.dimension), initializer=initializers.get("random_normal"))
        self.Wday = self.add_weight(name="Wday", shape=(self.dimension, self.dimension), initializer=initializers.get("random_normal"))
        self.Wdate = self.add_weight(name="Wdate", shape=(self.dimension, self.dimension), initializer=initializers.get("random_normal"))
        self.Whour = self.add_weight(name="Whour", shape=(self.dimension, self.dimension), initializer=initializers.get("random_normal"))
       
    def call(self, x):
        userEmbedding = x[0]

        curMonthEmbedding = K.reshape(x[1], shape=(-1, 1, self.dimension)) 
        curDayEmbedding = K.reshape(x[2], shape=(-1, 1, self.dimension)) 
        curDateEmbedding = K.reshape(x[3], shape=(-1, 1, self.dimension)) 
        curHourEmbedding = K.reshape(x[4], shape=(-1, 1, self.dimension)) 

        monthEmbeddings = x[5] 
        dayEmbeddings = x[6]
        dateEmbeddings = x[7] 
        hourEmbeddings = x[8] 

        # personalization
        curMonthEmbedding = curMonthEmbedding * (K.dot(userEmbedding, self.Wmonth))
        curDayEmbedding = curDayEmbedding * (K.dot(userEmbedding, self.Wday))
        curDateEmbedding = curDateEmbedding * (K.dot(userEmbedding, self.Wdate))
        curHourEmbedding = curHourEmbedding * (K.dot(userEmbedding, self.Whour))
        monthEmbeddings = monthEmbeddings * (K.dot(userEmbedding, self.Wmonth))
        dayEmbeddings = dayEmbeddings * (K.dot(userEmbedding, self.Wday))
        dateEmbeddings = dateEmbeddings * (K.dot(userEmbedding, self.Wdate))
        hourEmbeddings = hourEmbeddings * (K.dot(userEmbedding, self.Whour))

        # query for gradated attention
        monthQ = curMonthEmbedding 
        dayQ = curDayEmbedding
        dateQ = curDateEmbedding 
        hourQ = curHourEmbedding
        
        # key, value
        monthKV = concatenate([monthEmbeddings, curMonthEmbedding], axis=1) 
        dayKV = concatenate([dayEmbeddings, curDayEmbedding], axis=1) 
        dateKV = concatenate([dateEmbeddings, curDateEmbedding], axis=1) 
        hourKV = concatenate([hourEmbeddings, curHourEmbedding], axis=1) 

        # attention score
        monthQKV = K.softmax(K.batch_dot(monthQ, K.permute_dimensions(monthKV, pattern=(0, 2, 1))) / K.sqrt(K.cast(self.dimension, dtype=tf.float32)), axis=-1)
        dayQKV = K.softmax(K.batch_dot(dayQ, K.permute_dimensions(dayKV, pattern=(0, 2, 1))) / K.sqrt(K.cast(self.dimension, dtype=tf.float32)), axis=-1)
        dateQKV = K.softmax(K.batch_dot(dateQ, K.permute_dimensions(dateKV, pattern=(0, 2, 1))) / K.sqrt(K.cast(self.dimension, dtype=tf.float32)), axis=-1)
        hourQKV = K.softmax(K.batch_dot(hourQ, K.permute_dimensions(hourKV, pattern=(0, 2, 1))) / K.sqrt(K.cast(self.dimension, dtype=tf.float32)), axis=-1)

        # embedding for each granularity of period information
        monthEmbedding = K.batch_dot(monthQKV, monthKV)
        dayEmbedding = K.batch_dot(dayQKV, dayKV) 
        dateEmbedding = K.batch_dot(dateQKV, dateKV) 
        hourEmbedding = K.batch_dot(hourQKV, hourKV) 

        # multiplicative attention
        q = userEmbedding 
        kv = K.concatenate([monthEmbedding, dayEmbedding, dateEmbedding, hourEmbedding], axis=1) 
        qW = K.dot(q, self.W)
        a = K.sigmoid(K.batch_dot(qW, K.permute_dimensions(kv, pattern=(0, 2, 1))))
        timeRepresentation = K.batch_dot(a, kv) 
        return timeRepresentation

    def compute_output_shape(self, inshape):
        return (None, 1, self.dimension)


class TAHE(Layer):
    def __init__(self, dimension, trainable=True, name=None, **kwargs):
        super(TAHE, self).__init__(name=name, trainable=trainable, **kwargs)
        self.dimension = dimension

    def build(self, inshape):
        1
        
    def call(self, x):
        recentTimeRepresentations = x[0]
        curTimeRepresentation = x[1]
        recentTimestamps = x[2]
        recentItemEmbeddings = x[3] 

        # previous timestamp == 0 ==> no history
        mask = K.cast(recentTimestamps > 0, dtype=tf.float32) 

        # time-based attention
        similarity = K.batch_dot(K.l2_normalize(curTimeRepresentation, axis=-1), K.permute_dimensions(K.l2_normalize(recentTimeRepresentations, axis=-1), pattern=(0, 2, 1)))
        masked_similarity = mask * ((similarity + 1.0) / 2.0)
        weightedPrevItemEmbeddings = K.batch_dot(masked_similarity, recentItemEmbeddings)
        userHistoryRepresentation = weightedPrevItemEmbeddings

        return userHistoryRepresentation

    def compute_output_shape(self, inshape):
        return (None, self.dimension)


class meanLayer(Layer):
    def __init__(self, trainable=True, name=None, **kwargs):
        super(meanLayer, self).__init__(name=name, trainable=trainable, **kwargs)

    def build(self, inshape):
        1

    def call(self, x):
        return K.mean(x, axis=1, keepdims=True)
        
    def compute_output_shape(self, inshape):
        return (inshape[0], 1, inshape[2])


class Slice(Layer):
    def __init__(self, index, trainable=True, name=None, **kwargs):
        super(Slice, self).__init__(name=name, trainable=trainable, **kwargs)
        self.index = index

    # def build(self, inshape):
    #     1

    def call(self, x):
        return x[:, self.index, :]

    def compute_output_shape(self, inshape):
        return (inshape[0], inshape[2])


class TemporalPositionEncoding(Layer):
    def __init__(self, trainable=True, name=None, **kwargs):
        super(TemporalPositionEncoding, self).__init__(name=name, trainable=trainable, **kwargs)

    def build(self, inshape):
        self.a = self.add_weight(name="a", shape=(1, ), initializer=initializers.get("ones"))

    def call(self, x):
        item = x[0] 
        time = x[1] 

        return item + time * self.a

    def compute_output_shape(self, inshape):
        return inshape[0]

def create_sequence(sequence_length : int):
    _input = []
    for i in range(sequence_length):
        _input.append(Input(shape=[1], dtype=tf.int32))
    return _input

def create_prev_embedding(embedding, prevEmbeddings:list, recentInput:list, idx_seq:int, ratio:float, num_of_date:int):
    for j in range(1, max(int(num_of_date * ratio + 0.5), 1) + 1):
        surr = embedding(SurroundingSlots(window_length=j, max_range=num_of_date)(recentInput[idx_seq]))
        prevEmbeddings[idx_seq].append(meanLayer()(surr))

def create_embedding(embeddings: list, embedding, dateInput, ratio: float, num_of_date: int):
    for i in range(1, max(int(num_of_date * ratio + 0.5), 1) + 1):
        surr = embedding(SurroundingSlots(window_length=i, max_range=num_of_date)(dateInput))
        embeddings.append(meanLayer()(surr))

def concat_embedding(embedding:list, prevEmbeddings:list, sequence_length: int, ratio:float, num_of_date:int):
    if int(num_of_date * ratio + 0.5) <= 1:
        embedding = embedding[0]
        for i in range(sequence_length):
            prevEmbeddings[i] = prevEmbeddings[i][0]
    else:    
        embedding = concatenate(embedding, axis=1) 
        for i in range(sequence_length):
            prevEmbeddings[i] = concatenate(prevEmbeddings[i], axis=1)
    
    return embedding, prevEmbeddings

def TimelyRec(num_users, num_items, embedding_size, sequence_length, width, depth, dropout=None):
    userInput = Input(shape=[1], dtype=tf.int32)
    itemInput = Input(shape=[1], dtype=tf.int32)
    monthInput = Input(shape=[1], dtype=tf.int32)
    dayInput = Input(shape=[1], dtype=tf.int32)
    dateInput = Input(shape=[1], dtype=tf.int32)
    hourInput = Input(shape=[1], dtype=tf.int32)
    curTimestampInput = Input(shape=[1], dtype=tf.int32)

    recentMonthInput = create_sequence(sequence_length)
    recentDayInput = create_sequence(sequence_length)
    recentDateInput = create_sequence(sequence_length)
    recentHourInput = create_sequence(sequence_length)
    recentTimestampInput = create_sequence(sequence_length)
    recentItemidInput = create_sequence(sequence_length)

    userEmbedding = Embedding(num_users+1, embedding_size)(userInput)
    itemEmbeddingSet = Embedding(num_items+1, embedding_size)
    itemEmbedding = itemEmbeddingSet(itemInput)
    recentItemEmbeddings = itemEmbeddingSet(concatenate(recentItemidInput, axis=-1))
    recentTimestamps = concatenate(recentTimestampInput, axis=-1) 

    monthEmbedding = Embedding(12, embedding_size)
    dayEmbedding = Embedding(7, embedding_size)
    dateEmbedding = Embedding(31, embedding_size)
    hourEmbedding = Embedding(24, embedding_size)

    curMonthEmbedding = monthEmbedding(monthInput)
    curDayEmbedding = dayEmbedding(dayInput)
    curDateEmbedding = dateEmbedding(dateInput)
    curHourEmbedding = hourEmbedding(hourInput)

    recentMonthEmbeddings = monthEmbedding(concatenate(recentMonthInput, axis=-1))
    recentDayEmbeddings = dayEmbedding(concatenate(recentDayInput, axis=-1))
    recentDateEmbeddings = dateEmbedding(concatenate(recentDateInput, axis=-1))
    recentHourEmbeddings = hourEmbedding(concatenate(recentHourInput, axis=-1))

    monthEmbeddings = []
    dayEmbeddings = []
    dateEmbeddings = []
    hourEmbeddings = []

    prevMonthEmbeddings = []
    prevDayEmbeddings = []
    prevDateEmbeddings = []
    prevHourEmbeddings = []

    ratio = 0.2

    for i in range(sequence_length):
        prevMonthEmbeddings.append([])
        create_prev_embedding(monthEmbedding, prevMonthEmbeddings, recentMonthInput, i, ratio, num_of_date=12)

        prevDayEmbeddings.append([])
        create_prev_embedding(dayEmbedding, prevDayEmbeddings, recentDayInput, i, ratio, num_of_date=7) 

        prevDateEmbeddings.append([])
        create_prev_embedding(dateEmbedding, prevDateEmbeddings, recentDateInput, i, ratio, num_of_date=31)

        prevHourEmbeddings.append([])
        create_prev_embedding(hourEmbedding, prevHourEmbeddings, recentHourInput, i, ratio, num_of_date=24)

    create_embedding(monthEmbeddings, monthEmbedding, monthInput, ratio, num_of_date=12)

    create_embedding(dayEmbeddings, dayEmbedding, dayInput, ratio, num_of_date=7)  

    create_embedding(dateEmbeddings, dateEmbedding, dateInput, ratio, num_of_date=31)
    
    create_embedding(hourEmbeddings, hourEmbedding, hourInput, ratio, num_of_date=24)


    monthEmbeddings, prevMonthEmbeddings = concat_embedding(monthEmbeddings, prevMonthEmbeddings, sequence_length, ratio, num_of_date=12)

    dayEmbeddings, prevDayEmbeddings = concat_embedding(dayEmbeddings, prevDayEmbeddings, sequence_length, ratio, num_of_date=7)

    dateEmbeddings, prevDateEmbeddings = concat_embedding(dateEmbeddings, prevDateEmbeddings, sequence_length, ratio, num_of_date=31)

    hourEmbeddings, prevHourEmbeddings = concat_embedding(hourEmbeddings, prevHourEmbeddings, sequence_length, ratio, num_of_date=24)

    recentTimestampTEs = PositionalEncoding(output_dim=embedding_size)(recentTimestamps)
    curTimestampTE = PositionalEncoding(output_dim=embedding_size)(curTimestampInput)

    # temporal position encoding
    te = TemporalPositionEncoding()
    itemEmbedding = te([itemEmbedding, curTimestampTE])    
    recentItemEmbeddings = te([recentItemEmbeddings, recentTimestampTEs])

    userVector = Flatten()(userEmbedding)
    itemVector = Flatten()(itemEmbedding)
    curTimestampTE = Flatten()(curTimestampTE)
    
    # MATE
    curTimeRepresentation = Flatten()(MATE(embedding_size)([userEmbedding, curMonthEmbedding, curDayEmbedding, curDateEmbedding, curHourEmbedding, monthEmbeddings, dayEmbeddings, dateEmbeddings, hourEmbeddings])) # None * embedding_size
    prevTimeRepresentations = []
    for i in range(sequence_length):
        prevTimeRepresentations.append(MATE(embedding_size)([userEmbedding, Slice(i)(recentMonthEmbeddings), Slice(i)(recentDayEmbeddings), Slice(i)(recentDateEmbeddings), Slice(i)(recentHourEmbeddings), prevMonthEmbeddings[i], prevDayEmbeddings[i], prevDateEmbeddings[i], prevHourEmbeddings[i]])) # None * embedding_size)
    prevTimeRepresentations = concatenate(prevTimeRepresentations, axis=1)

    # TAHE
    userHistoryRepresentation = TAHE(embedding_size)([prevTimeRepresentations, curTimeRepresentation, recentTimestamps, recentItemEmbeddings])

    # combination
    x = concatenate([userVector, itemVector, curTimeRepresentation, userHistoryRepresentation])
    in_shape = embedding_size * 4

    for i in range(depth):
        if i == depth - 1:
            x = Dense(1, input_shape=(in_shape,))(x)
        else:
            x = Dense(width, input_shape=(in_shape,))(x)
            x = Activation('relu')(x)
            if dropout is not None:
                x = Dropout(dropout)(x)
        in_shape = width
        
    outputs = Activation('sigmoid')(x)

    model = Model(inputs=[userInput, itemInput, monthInput, dayInput, dateInput, hourInput, curTimestampInput] + [recentMonthInput[i] for i in range(sequence_length)] + [recentDayInput[i] for i in range(sequence_length)] + [recentDateInput[i] for i in range(sequence_length)] + [recentHourInput[i] for i in range(sequence_length)] + [recentTimestampInput[i] for i in range(sequence_length)] + [recentItemidInput[i] for i in range(sequence_length)], outputs=outputs)
    return model