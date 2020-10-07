import os
import re
import csv
import random
import pickle
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras

import logging
logging.getLogger('tensorflow').disabled = True

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("\nNum GPUs Available: {}\n".format(len(physical_devices)))
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import model_from_json, load_model, Sequential, Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Dropout

from variables import*
from util import*
from sklearn.model_selection import train_test_split
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

class myCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.9975):
            print("\nReached 99.5% train accuracy.So stop training!")
            self.model.stop_training = True

class SentimentAnalyser:
    def __init__(self):
        data, Ytrain,Ytest,Xtrain,Xtest = get_sentiment_data()
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        self.Xtest  = Xtest
        self.Ytest  = Ytest
        self.data = data

    def tokenizing_data(self):
        tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
        tokenizer.fit_on_texts(self.Xtrain)

        Xtrain_seq = tokenizer.texts_to_sequences(self.Xtrain)
        self.Xtrain_pad = pad_sequences(Xtrain_seq, maxlen=max_length, truncating=trunc_type)

        Xtest_seq  = tokenizer.texts_to_sequences(self.Xtest)
        self.Xtest_pad = pad_sequences(Xtest_seq, maxlen=max_length)
        self.tokenizer = tokenizer

    def embedding_model(self):
        inputs = Input(shape=(max_length,))
        x = Embedding(output_dim=embedding_dimS, input_dim=vocab_size, input_length=max_length, name='embedding')(inputs)
        x = Bidirectional(LSTM(size_lstm), name='bidirectional_lstm')(x)
        x = Dense(denseS, activation='relu', name='dense1')(x)
        x = Dense(denseS, activation='relu', name='dense2')(x)
        x = Dense(denseS, activation='relu', name='dense3')(x)
        outputs = Dense(size_output, activation='sigmoid', name='dense_out')(x)

        model = Model(inputs=inputs, outputs=outputs)
        self.model = model

    def load_model(self):
        loaded_model = load_model(sentiment_weights)
        loaded_model.compile(
                        loss='binary_crossentropy',
                        optimizer='adam',
                        metrics=['accuracy']
                        )
        self.model = loaded_model

    def train_model(self,bias):
        callbacks = myCallback()
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        class_weights = {1: 1 ,
                        0: 1.6/bias }
        self.model.fit(
            self.Xtrain_pad,
            self.Ytrain,
            batch_size=batch_size,
            epochs=num_epochs,
            validation_data=(self.Xtest_pad,self.Ytest),
            callbacks= [callbacks],
            class_weight=class_weights
            )

    def save_model(self):
        self.model.save(sentiment_weights)

    def run(self):
        self.tokenizing_data()
        if os.path.exists(sentiment_weights):
            self.load_model()
        else:
            self.embedding_model()
            self.train_model(bias)
            self.save_model()

    def predict(self,reviews,labels):
        sequence_data = self.tokenizer.texts_to_sequences(reviews)
        padded_data = pad_sequences(sequence_data, maxlen=max_length)
        P = (self.model.predict(padded_data) > 0.7)
        return P