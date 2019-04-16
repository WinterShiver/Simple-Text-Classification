# -*-coding:utf-8-*-
import os
import re
import jieba
import openpyxl
import numpy as np
import time
import math
import pickle as pkl

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

def clean_onetype(linetext, restr, newStr):
    '''
    input: 
    linetext: a line, beginning p means parameter
    restr: the str gotta be replaced
    str: replacing str
    output: line with a certain type of label cleaned
    '''
    # continuously trying to find comparable restr, add them in list(to manage their index) and replace them
    replaceList = []
    while re.search(restr, linetext) != None: 
        toBeReplaceStr = linetext[re.search(restr, linetext).span()[0]: re.search(restr, linetext).span()[1]]
        if not toBeReplaceStr in replaceList:
            replaceList.append(toBeReplaceStr)
        index = replaceList.index(toBeReplaceStr)
        linetext = linetext.replace(toBeReplaceStr, newStr+str(index))
    return linetext
        
def clean_oneline(linetext):
    '''
    input: a line, beginning p means parameter
    output: line with cleaned tel, date, time, long number
    '''
    # preparation
    cleanDict = {}
    cleanDict['longNum'] = '[0-9]{13}[0-9]*' # long integer
    cleanDict['tel'] = '[0-1][0-9]{10}|[0-1][0-9]{11}'
    cleanDict['date'] = '(20[0-1][0-9]-[0-1][0-9]-[0-3][0-9])|([0-1]?[0-9]月[0-3]?[0-9]日)'
    cleanDict['time'] = '[0-2][0-9](:|：)[0-5][0-9](:|：)[0-5][0-9]'
    cleanDict['shortTime'] = '[0-5][0-9](:|：)[0-5][0-9]'
    cleanDict['money'] = '[0-9]+.?[0-9]*元'
    cleanDict['money_l'] = '[0-9]+万'
    cleanDict['URL'] = '(http|www)\S*.(com|cn)'
    cleanDict['year'] = '(19|20)[0-9][0-9]年'
    cleanDict['time_cn'] = '([0-9]*日|号)?((上|下)午)?[0-1]?[0-9](点|时)'
    cleanDict['time_cnn'] = '[0-9]+(小时|分钟)'
    cleanDict['date_s'] = '[0-9]+(日|年|岁|月)'
    cleanDict['score'] = '[0-9]+(比|-)[0-9]+'
    cleanDict['precent'] = '[0-9]+.*[0-9]*[%]' 
    # accurate number, causing strange endless loop
    # replace
    for replaceStr in cleanDict:
        linetext = clean_onetype(linetext, cleanDict[replaceStr], replaceStr)
    return linetext

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


count = 0
work_dir = '../THUCNews'
# label_valid = {'游戏': 0, '教育': 0, '房产': 0}
label_valid = {'财经': 0, '彩票': 0, '房产': 0, '股票': 0, '家居': 0, '教育': 0, '科技': 0, '社会': 0, '时尚': 0, '时政': 0, '体育': 0, '星座': 0, '游戏': 0, '娱乐': 0}
label_list = []
str_list = []
for parent, dirnames, filenames in os.walk(work_dir,  followlinks=True):
    for filename in filenames:
        label = parent.split('\\')[-1]
        if (not label in label_valid) or (label_valid[label] > 500) :
            continue
        file_path = os.path.join(parent, filename) # one file one label
        file_str = ''
        label = parent.split('\\')[-1]
        txt = open(file_path, mode = 'r', encoding = 'utf8')
        for lines in txt:
            lines = lines.strip()
            if isShort(lines):
                continue
            else:
                file_str = file_str + lines
        file_str = clean_oneline(file_str)
        str_list.append(file_str)
        label_list.append(label)
        label_valid[label] = label_valid[label] + 1
        

saveToFile([str_list, label_list], ['str_list_500_all.pkl', 'label_list_500_all.pkl'])