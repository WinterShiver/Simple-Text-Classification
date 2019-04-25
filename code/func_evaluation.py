# func_evaluation.py
import numpy as np

def recall(matrix):
    '''each type R=TP/P'''
    assert matrix.shape[0] == matrix.shape[1]
    return [matrix[i][i]/sum(matrix[:,i]) for i in range(matrix.shape[0])]

def precision(matrix):
    '''each type P=TP/T'''
    assert matrix.shape[0] == matrix.shape[1]
    return [matrix[i][i]/sum(matrix[i,:]) for i in range(matrix.shape[0])]

def accuracy(matrix):
    '''A=T/ALL'''
    assert matrix.shape[0] == matrix.shape[1]
    return sum([matrix[i][i] for i in range(matrix.shape[0])])/sum(sum(matrix))

def least_f1(matrix):
    # matrix.dtype = np.float16
    tmp_recall = min(recall(matrix))
    tmp_precision = min(precision(matrix))
    f1 = 2 * tmp_recall * tmp_precision / (tmp_recall + tmp_precision)
    return [tmp_recall, tmp_precision, f1]

def macro_f1(matrix):
    # matrix.dtype = np.float16
    tmp_recall = np.mean(recall(matrix))
    tmp_precision = np.mean(precision(matrix))
    f1 = 2 * tmp_recall * tmp_precision / (tmp_recall + tmp_precision)
    return [tmp_recall, tmp_precision, f1]

def micro_f1(matrix):
    [TP, FP, FN] = [0, 0, 0]
    for i in range(matrix.shape[0]):
        TP += matrix[i][i]
        FP += sum(matrix[i, :]) - matrix[i][i]
        FN += sum(matrix[:, i]) - matrix[i][i]
    tmp_recall = TP / (TP + FN)
    tmp_precision = TP / (TP + FP)
    f1 = 2 * tmp_recall * tmp_precision / (tmp_recall + tmp_precision)
    return [tmp_recall, tmp_precision, f1]

def show_evaluation(matrix):
    print("Sample:", np.sum(matrix), "Samples")
    print("Accuracy: ", accuracy(matrix))
    print("Least： tmp_recall, tmp_precision, F1: ", least_f1(matrix))
    print("Macro： tmp_recall, tmp_precision, F1: ", macro_f1(matrix))
    print("Micro： tmp_recall, tmp_precision, F1: ", micro_f1(matrix))
