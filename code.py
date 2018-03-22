#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 11:57:33 2018

@author: KJ
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import speech_recognition as sr
from gtts import gTTS
import os
import http.client, urllib
import json
import numpy as np


# get audio from the microphone
r=sr.Recognizer()
with sr.Microphone() as source:
	print("Speak:")
	audio=r.listen(source)


print(r.recognize_google(audio))

print ('Please wait a moment for the results to appear.\n')



npr = r.recognize_google(audio)


from __future__ import print_function, division
from future.utils import iteritems
from builtins import range



import nltk

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup


# Prepocessing test data
wordnet_lemmatizer_test = WordNetLemmatizer()

# from http://www.lextek.com/manuals/onix/stopwords1.html
stopwords = set(w.rstrip() for w in open('stopwords.txt'))

# function to tokenize test data
def my_tokens_test(s):
    s = s.lower() 
    tokens = nltk.tokenize.word_tokenize(s) 
    tokens = [t for t in tokens if len(t) > 2] # removing short words 
    tokens = [wordnet_lemmatizer_test.lemmatize(t) for t in tokens] 
    tokens = [t for t in tokens if t not in stopwords] 
    return tokens


word_index_map_test = {}
current_index_test = 0
positive_tokenized_test = []


tokens = my_tokens_test(npr)
positive_tokenized_test.append(tokens)
for token in tokens:
    if token not in word_index_map_test:
        word_index_map_test[token] = current_index_test
        current_index_test += 1


# Preprocessing training data 

wordnet_lemmatizer = WordNetLemmatizer()

# from http://www.lextek.com/manuals/onix/stopwords1.html
stopwords = set(w.rstrip() for w in open('stopwords.txt'))

# load the reviews
# data courtesy of http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
positive_reviews = BeautifulSoup(open('electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text')

negative_reviews = BeautifulSoup(open('electronics/negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')

# making number of positive and negaitive reviews balanced 
np.random.shuffle(positive_reviews)
positive_reviews = positive_reviews[:len(negative_reviews)]


# Function to tokenize training data 
def my_tokens(s):
    s = s.lower() 
    tokens = nltk.tokenize.word_tokenize(s) 
    tokens = [t for t in tokens if len(t) > 2] # removing short words 
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] 
    tokens = [t for t in tokens if t not in stopwords] 
    return tokens


word_index_map = {}
current_index = 0
positive_tokenized = []
negative_tokenized = []

for review in positive_reviews:
    tokens = my_tokens(review.text)
    positive_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    tokens = my_tokens(review.text)
    negative_tokenized.append(tokens)
    for token in tokens:
        if token not in word_index_map:
            word_index_map[token] = current_index
            current_index += 1


# function to create the training input matrice 
def tokens_convert_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1) # last element is the label
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum() 
    x[-1] = label
    return x

N = len(positive_tokenized) + len(negative_tokenized)

data = np.zeros((N, len(word_index_map) + 1))
i = 0
for tokens in positive_tokenized:
    xy = tokens_convert_vector(tokens, 1)
    data[i,:] = xy
    i += 1

for tokens in negative_tokenized:
    xy = tokens_convert_vector(tokens, 0)
    data[i,:] = xy
    i += 1
    

# function to create the test input matrice 
def tokens_convert_vector_test(tokens):
    x = np.zeros(len(word_index_map)+1) # last element is the label
    for t in tokens:
        i = word_index_map[t]
        x[i] += 1
    x = x / x.sum() 
    
    return x


data_test = np.zeros((N, len(word_index_map) + 1))
i = 0
for tokens in positive_tokenized_test:
    xy = tokens_convert_vector_test(tokens)
    data_test[i,:] = xy
    i += 1

data_test2 = data_test[:100,:-1]
  

np.random.shuffle(data)

X = data[:,:-1]
Y = data[:,-1]

# last 100 rows will be the test data 
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

# Using Logistic Regression for Classification 
model = LogisticRegression()

# Training the model
model.fit(Xtrain, Ytrain)


print("Classification Rate:", model.score(data_test2, Ytest))


threshold = 0.5
for word, index in iteritems(word_index_map_test):
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, weight)
