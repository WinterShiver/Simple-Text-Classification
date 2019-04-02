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
    least_recall = min(recall(matrix))
    least_precision = min(precision(matrix))
    f1 = 2 * least_recall * least_precision / (least_recall + least_precision)
    return [least_recall, least_precision, f1]

def show_evaluation(matrix):
    print("Sample:", np.sum(matrix), "Samples")
    print("Accuracy: ", accuracy(matrix))
    print("Least Recall, Least Precision, F1: ", least_f1(matrix))