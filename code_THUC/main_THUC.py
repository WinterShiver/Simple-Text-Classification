# -*-coding:utf-8-*-
import os
import re
import jieba
import openpyxl
import numpy as np
import time
import math
import pickle as pkl
import sys

sys.path.append('../code')

from func_cv_multi_thread import *
from func_evaluation import *


def saveToFile(obj, fileName):
    if len(obj) != len(fileName):
        input('saveToFile in liantong.py: Length do not match.')
    for i in range(len(obj)):
        tFile = open(fileName[i], 'wb')
        pkl.dump(obj[i], tFile)
        tFile.close()

def openFromFile(fileName):
    obj = []
    for i in fileName:
        tFile = open(i, 'rb')
        obj.append(pkl.load(tFile))
        tFile.close()
    return obj
      
# vectorizer = TfidfVectorizer or CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = TfidfVectorizer
# def vectorize(vectorizer, train_words, test_words):
def vectorize(vectorizer, train_sen_list): 
    # referred from https://blog.csdn.net/u011010851/article/details/78060906/
    v = vectorizer(tokenizer = lambda x: jieba.cut(x, cut_all = True), binary = False, decode_error = 'ignore',stop_words = 'english') 
    print("Before Fit Word Vec:", time.process_time())
    train_data = v.fit_transform(train_sen_list)
    print("After Fit Word Vec:", time.process_time()) 
    # test_data = v.transform(test_words) 
    return train_data

def isShort(sen_str):
    return len(sen_str) < 10

def batch_generate(label_num, batch_num):
    # generate random labels of batch_num types
    return [i % batch_num + 1 for i in range(label_num)]
    
work_dir = '../THUCNews'

[str_list, label_list] = openFromFile(['str_list_micro.pkl', 'label_list_micro.pkl'])
data_matrix = vectorize(vectorizer, str_list).toarray()
print(type(data_matrix), data_matrix.shape)

from sklearn.decomposition import PCA
pca = PCA(n_components=300)
data_matrix = pca.fit_transform(data_matrix)

print("PCA, contri:", sum(pca.explained_variance_ratio_))
print("Time:", time.process_time())

label_batch = batch_generate(len(label_list), 10)
print("Before CV:", time.process_time())
[test_confuse_matrix, train_confuse_matrix, cv_result] = cv(data_matrix, label_list, label_batch)

'''wb = openpyxl.load_workbook('../data.xlsx')
sheet = wb.active
count = 0
for i in range(sen_num):
    if not i in except_list:
        sheet.cell(row = i+2, column = 9).value = cv_result[count]
        count = count + 1
wb.save('../result/tmp_data_label_cleaned_linear+0.3.xlsx')'''

'''results

3-500
[[491.   4.   2.]
 [  6. 495.   4.]
 [  4.   2. 495.]]

 Train Confuse Matrix:
[[4456.   17.    9.]
 [  53. 4480.   10.]
 [   0.   12. 4490.]]

 all-500-linear-1
 Total Result
Current Process Time: 37m2s641ms
Local Time: Tue Apr 16 02:25:13 2019
Sample: 7014.0 Samples
Accuracy:  0.9224408326204734
Least Recall, Least Precision, F1:  [0.8483033932135728, 0.8276515151515151, 0.8378502132689128]
['../THUCNews\\体育', '../THUCNews\\娱乐', '../THUCNews\\家居', '../THUCNews\\彩票', '../THUCNews\\房产', '../THUC
教育', '../THUCNews\\时尚', '../THUCNews\\时政', '../THUCNews\\星座', '../THUCNews\\游戏', '../THUCNews\\社会', '.
News\\科技', '../THUCNews\\股票', '../THUCNews\\财经']
Test Confuse Matrix:
[[491.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.]
 [  2. 476.   3.   0.   2.   6.   3.   2.   0.   5.   6.   3.   0.   2.]
 [  0.   2. 474.   0.  10.   6.   9.   2.   1.   1.   2.   2.   2.   2.]
 [  0.   0.   0. 491.   0.   0.   0.   0.   0.   0.   4.   0.   0.   0.]
 [  0.   0.   7.   1. 468.   2.   0.   0.   0.   1.   3.   1.   6.   6.]
 [  2.   3.   1.   0.   3. 435.   7.   2.   3.   0.  14.   4.   5.   4.]
 [  1.   6.   7.   0.   1.   6. 470.   0.   0.   0.   3.   0.   3.   0.]
 [  0.   2.   0.   0.   0.   2.   0. 460.   0.   0.   4.   6.  20.   2.]
 [  1.   4.   1.   0.   0.   7.   4.   0. 487.   7.   3.   1.   0.   3.]
 [  0.   0.   0.   0.   0.   0.   0.   0.   0. 482.   2.   0.   0.   0.]
 [  4.   3.   3.   8.   5.   9.   3.   7.   1.   2. 425.  20.   5.  10.]
 [  0.   2.   1.   1.   3.  12.   2.   8.   5.   3.  18. 442.   7.   5.]
 [  0.   1.   1.   0.   5.   4.   3.  18.   3.   0.   3.  18. 437.  35.]
 [  0.   2.   3.   0.   4.  12.   0.   2.   1.   0.  14.   4.  16. 432.]]


'''