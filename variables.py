import os

seed = 42
vocab_size = 15000
max_length = 120
embedding_dimS = 512
trunc_type = 'post'
oov_tok = "<OOV>"
num_epochs = 10
batch_size = 128
size_lstm  = 256
denseS = 64
size_output = 1
validation_split = 0.15
bias = 0.21600911256083669
sentiment_weights = "data/sentiment_model.h5"

#Data paths and weights
twitter_data = 'data/training.1600000.processed.noemoticon.csv'