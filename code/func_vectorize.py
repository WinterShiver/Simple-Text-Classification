# -*-coding:utf-8-*-
# provide a vectorizer
import os 
import openpyxl
import re
import numpy as np
import time
import math

# para
wv_dim = 300 # 300 or 128
word_least = 5 # 句子中最少保留词语个数，否则不能用
word_in_sen = 10 # 句子中保留词语个数

# w2v model
from gensim.models.word2vec import Word2Vec
if wv_dim == 300:
    model = Word2Vec.load("../word2vec/word2vec_300.model")
elif wv_dim == 128:
    model = Word2Vec.load("../word2vec/word2vec_128.model")

def vectorize_w2v_add(str):
    data_vector = [0 for i in range(wv_dim)]
    str_list= str.split(" ")
    count = 0
    for word in str_list:
        try:
            data_vector = list(map(lambda x, y: x+y, data_vector, model.wv[word]))
            count = count + 1
        except KeyError:
            pass
    if count >= word_least:
        return [data_vector, True, count]
    else:
        return [data_vector, False, count]

def vectorize_w2v_link(str):
    data_vector = [0 for i in range(wv_dim * word_in_sen)]
    str_list= str.split(" ")
    count = 0
    for word in str_list:
        try:
            try:
                begin_index = wv_dim * count
                data_vector[begin_index: begin_index + wv_dim] = list(model.wv[word])
            except ValueError:
                print(word)
                print(model.wv[word].shape)
                # print(data_vector[wv_dim * count: wv_dim * (count + 1)].size)
            count = count + 1
            if count == word_in_sen:
                break
        except KeyError:
            pass
    if count == word_in_sen:
        return [data_vector, True, count]
    else:
        return [data_vector, False, count]

# main focus
vectorize = vectorize_w2v_link # vectorize_w2v_add vectorize_w2v_link
# TF-IDF is not used: baseline method is not needed