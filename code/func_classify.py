# input: train_data, train_label: binary or not
#        test_data, test_label: test should always be all dataset, to ensure index corresponding
# output: result, possibility

import os 
import re
import numpy as np
import math

# para
svm_c = 1
svm_kernel = "linear"
expand = True # expand train set or not

# svm build
from sklearn import svm
clf = svm.SVC(C=svm_c, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma='auto', kernel=svm_kernel, max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False) 

# functions
def train_expand(train_data, train_label):
    # when number of samples are not balanced in each class, expand those who have less samples
    labels = np.unique(train_label)
    count_labels = dict(zip(labels, [0 for elem in labels]))
    time_labels = dict()
    for elem in train_label:
        count_labels[elem] = count_labels[elem] + 1
    # how many times?
    maxnum_labels = max(list(count_labels.values()))
    for elem in labels:
        time_labels[elem] = int(maxnum_labels / count_labels[elem])
        assert time_labels[elem] > 0
    # make result
    [tmp_train_data, tmp_train_label] = [[], []]
    for i in range(len(train_label)):
        tmp_time = time_labels[train_label[i]]
        for j in range(tmp_time):
            tmp_train_data.append(train_data[i])
            tmp_train_label.append(train_label[i])
    return [tmp_train_data, tmp_train_label]

def svm_multi(train_data, train_label, test_data, expand):
    # 直接使用SVC多分类
    # expand training set: see implement
    if expand:
        [train_data, train_label] = train_expand(train_data, train_label)
    # labels
    labels = np.unique(train_label)
    # return format
    test_result = [None for i in range(len(test_data))]   
    test_proba = [0 for i in range(len(test_data))]
    # svm build
    clf.fit(train_data, np.asarray(train_label))
    test_proba_tmp = clf.predict_proba(test_data)
    # take max proba
    for j in range(len(test_proba_tmp)):
        test_proba[j] = max(list(test_proba_tmp[j]))
        test_result[j] = labels[np.argmax(test_proba_tmp[j])]
    # return
    return [test_result, test_proba]

def svm_binary(train_data, train_label, test_data):
    # assume binarized label
    # return possibility of "1"
    # labels
    labels = np.unique(train_label)
    # return format
    test_result = [None for i in range(len(test_data))]   
    test_proba = [0 for i in range(len(test_data))]
    # svm build
    test_proba_tmp = clf.predict_proba(test_data)
    # take max proba
    for j in range(len(test_proba_tmp)):
        test_proba[j] = max(list(test_proba_tmp[j]))
        test_result[j] = labels[np.argmax(test_proba_tmp[j])]
        if(test_result == 0):
            test_proba[j] = 1 - test_proba[j]
    # return
    return [test_result, test_proba]

def svm_1vN(train_data, train_label, test_data, expand):
    # 使用SVC二分类+1vsN，取概率最大的
    # labels
    labels = np.unique(train_label)
    test_result = [None for i in range(len(test_data))]   
    test_proba = [0 for i in range(len(test_data))]
    for tmp_label in labels:
        tmp_train_label = [1 if elem == tmp_label else 0 for elem in train_label]
        if expand:
            [train_data, tmp_train_label] = train_expand(train_data, tmp_train_label)
        [tmp_test_result, tmp_test_proba] = svm_binary(train_data, tmp_train_label, test_data)
        for i in range(len(tmp_test_proba)):
            if tmp_test_proba[i] > test_proba[i]:
                test_proba[i] = tmp_test_proba[i]
                test_result[i] = tmp_label
    # return
    return [test_result, test_proba]

# main focus
classify = lambda train_data, train_label, test_data: svm_multi(train_data, train_label, test_data, expand) # svm_1vsN