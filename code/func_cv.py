import os 
import re
import numpy as np
import math
import time

from func_classify import *
# from func_classify import classify
from func_evaluation import *
# from func_evaluation import show_evaluation


def showTime(sec_num):
    # tell sec_num seconds properly
    hh = int(sec_num / 3600)
    sec_num = sec_num - hh * 3600
    mm = int(sec_num / 60)
    sec_num = sec_num - mm * 60
    ss = int(sec_num)
    ms = round((sec_num - ss) * 1000)
    output_str = ""
    if(hh != 0):
        output_str + str(hh)

def cv(all_data, all_label, all_batch):
    # divide train & validation due to batch
    # return final confuse matrix
    batches = list(np.unique(all_batch))
    labels = list(np.unique(all_label))
    labels.sort()
    # confuse_matrix[i][j]: predict as i-th, but actually j-th
    train_confuse_matrix = np.zeros((len(labels), len(labels)))
    test_confuse_matrix = np.zeros((len(labels), len(labels)))
    curr_time = time.process_time()
    for batch in batches:
        # prepare
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
        train_confuse_matrix = train_confuse_matrix + tmp_train_confuse_matrix
        test_confuse_matrix = test_confuse_matrix + tmp_test_confuse_matrix
        # show
        print("Batch:", batches.index(batch), "of", len(batches))
        print("Time of this Batch:", format((time.process_time() - curr_time) / 60, '.2f'), "minutes")
        curr_time = time.process_time()
        print("Current Process Time:", format(curr_time / 60,'.2f'), "minutes")
        print("Local Time:", time.asctime(time.localtime(time.time())))
        show_evaluation(tmp_test_confuse_matrix)
        print("Test Confuse Matrix:")
        print(tmp_test_confuse_matrix)
        # train is super good
        # print("Train Confuse Matrix:")
        # print(tmp_train_confuse_matrix)
    # final show
    print("Total Result")
    print("Current Process Time:", format(time.process_time() / 60, '.2f'), "minutes")
    print("Local Time:", time.asctime(time.localtime(time.time())))
    show_evaluation(test_confuse_matrix)
    print(labels)
    print("Test Confuse Matrix:")
    print(test_confuse_matrix)
    print("Train Confuse Matrix:")
    print(train_confuse_matrix)
