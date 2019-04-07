import os 
import re
import numpy as np
import math
import time
import threading

class MyThread(threading.Thread):
    def __init__(self, func, args, name=''):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.args = args
        self.result = self.func(*self.args)
 
    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

from func_classify import *
# from func_classify import classify
from func_evaluation import *
# from func_evaluation import show_evaluation


def show_time(sec_num):
    # tell sec_num seconds properly
    hh = int(sec_num / 3600)
    sec_num = sec_num - hh * 3600
    mm = int(sec_num / 60)
    sec_num = sec_num - mm * 60
    ss = int(sec_num)
    ms = round((sec_num - ss) * 1000)
    output_str = ""
    if(hh > 0):
        output_str = output_str + str(hh) + "h"
    if(mm > 0):
        output_str = output_str + str(mm) + "m"
    if(ss > 0):
        output_str = output_str + str(ss) + "s"
    if(ms > 0):
        output_str = output_str + str(ms) + "ms"
    return output_str

def cv(all_data, all_label, all_batch):
    assert len(all_data) == len(all_label)
    assert len(all_batch) == len(all_label)
    # divide train & validation due to batch
    # return final confuse matrix and classification result
    all_result = [None for elem in all_label]
    # prepare
    batches = list(np.unique(all_batch))
    labels = list(np.unique(all_label))
    labels.sort()
    # confuse_matrix[i][j]: predict as i-th, but actually j-th
    train_confuse_matrix = np.zeros((len(labels), len(labels)))
    test_confuse_matrix = np.zeros((len(labels), len(labels)))
    tmp_test_confuse_matrix = [np.zeros((len(labels), len(labels))) for batch in batches]
    tmp_train_confuse_matrix = [np.zeros((len(labels), len(labels))) for batch in batches]
    curr_time = time.process_time()
    threads = []
    for i in range(len(batches)):
        t = MyThread(cv_batch, (all_data, all_label, all_batch, batches[i]), cv_batch.__name__)
        threads.append(t)
    for i in range(len(batches)):
        threads[i].start()
    for i in range(len(batches)):
        threads[i].join()
    for i in range(len(batches)):
        [tmp_test_confuse_matrix[i], tmp_train_confuse_matrix[i], tmp_test_result] = threads[i].get_result()
        train_confuse_matrix = train_confuse_matrix + tmp_train_confuse_matrix[i]
        test_confuse_matrix = test_confuse_matrix + tmp_test_confuse_matrix[i]
        for j in range(len(all_result)):
            if all_batch[j] == batches[i]:
                all_result[j] = tmp_test_result[j]
        # curr_time = time.process_time()
    # final show
    print("Total Result")
    print("Current Process Time:", show_time(time.process_time()))
    print("Local Time:", time.asctime(time.localtime(time.time())))
    show_evaluation(test_confuse_matrix)
    print(labels)
    print("Test Confuse Matrix:")
    print(test_confuse_matrix)
    print("Train Confuse Matrix:")
    print(train_confuse_matrix)
    return [test_confuse_matrix, train_confuse_matrix, all_result]

def cv_batch(all_data, all_label, all_batch, batch):
    curr_time = time.process_time()
    # prepare
    batches = list(np.unique(all_batch))
    labels = list(np.unique(all_label))
    labels.sort()
    tmp_train_confuse_matrix = np.zeros((len(labels), len(labels)))
    tmp_test_confuse_matrix = np.zeros((len(labels), len(labels)))
    [tmp_train_data, tmp_train_label] = [[], []]
    # divide train & validation, classify
    for i in range(len(all_label)):
        if all_batch[i] != batch:
            tmp_train_label.append(all_label[i])
            tmp_train_data.append(all_data[i])
    [tmp_test_result, tmp_test_proba] = classify(tmp_train_data, tmp_train_label, all_data)
    # total result
    for i in range(len(all_label)):
        predict_index = labels.index(tmp_test_result[i])
        actual_index = labels.index(all_label[i])
        if all_batch[i] != batch:
            tmp_train_confuse_matrix[predict_index][actual_index] = tmp_train_confuse_matrix[predict_index][actual_index] + 1
        else:
            tmp_test_confuse_matrix[predict_index][actual_index] = tmp_test_confuse_matrix[predict_index][actual_index] + 1 
    # show
    print("Batch:", batches.index(batch) + 1, "of", len(batches))
    print("Time of this Batch:", show_time(time.process_time() - curr_time))
    print("Current Process Time:", show_time(time.process_time()))
    print("Local Time:", time.asctime(time.localtime(time.time())))
    show_evaluation(tmp_test_confuse_matrix)
    print("Test Confuse Matrix:")
    print(tmp_test_confuse_matrix)
    # train is super good
    # print("Train Confuse Matrix:")
    # print(tmp_train_confuse_matrix)
    return [tmp_test_confuse_matrix, tmp_train_confuse_matrix, tmp_test_result]
