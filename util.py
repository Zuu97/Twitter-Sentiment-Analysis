import re
import os
import pandas as pd
import numpy as np
import pickle as pkl
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.utils import shuffle
from variables import*
import matplotlib.pyplot as plt

def get_sentiment_data():
    df = pd.read_csv(twitter_data)
    df = df[['polarity', 'text']]
    data  = df.dropna(axis = 0, how ='any')

    Nval = int(len(data) * validation_split)
    train_data, test_data = data[:-Nval], data[-Nval:]
    train_labels  = train_data['review_label'].values
    test_labels   = test_data['review_label'].values

    train_reviews = preprocessed_data(train_data['review_text'].values)
    test_reviews  = preprocessed_data(test_data['review_text'].values)
    return data, train_labels,test_labels,train_reviews,test_reviews

def lemmatization(lemmatizer,sentence):
    lem = [lemmatizer.lemmatize(k) for k in sentence]
    lem = set(lem)
    return [k for k in lem]

def remove_stop_words(stopwords_list,sentence):
    return [k for k in sentence if k not in stopwords_list]

def preprocess_one(review):
    lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords_list = stopwords.words('english')
    review = review.lower()
    remove_punc = tokenizer.tokenize(review) # Remove puntuations
    remove_num = [re.sub('[0-9]', '', i) for i in remove_punc] # Remove Numbers
    remove_num = [i for i in remove_num if len(i)>0] # Remove empty strings
    lemmatized = lemmatization(lemmatizer,remove_num) # Word Lemmatization
    remove_stop = remove_stop_words(stopwords_list,lemmatized) # remove stop words
    updated_review = ' '.join(remove_stop)
    return updated_review

def preprocessed_data(reviews):
    updated_reviews = []
    if isinstance(reviews, np.ndarray) or isinstance(reviews, list):
        for review in reviews:
            updated_review = preprocess_one(review)
            updated_reviews.append(updated_review)
    elif isinstance(reviews, np.str_)  or isinstance(reviews, str):
        updated_reviews = [preprocess_one(reviews)]

    return np.array(updated_reviews)
