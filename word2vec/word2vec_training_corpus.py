# -*-coding:utf-8-*-

import os 
import openpyxl
import re
import win32com.client as win32

# training_corpus: text8 + extracted sentences + 
training_corpus = open('training_corpus', mode='w')
tx8 = open('text8', mode='r')
for lines in tx8:
    training_corpus.write(lines)
tx8.close()

# para
sen_num = 5042 # sen: index from 0 to 5042
rep_time = 5 # enhance the affect of our paper

wb = openpyxl.load_workbook('../tmp_data_label_cleaned.xlsx')
ws = wb.active
for time in range(rep_time):
    for i in range(sen_num):
        training_corpus.write(str(ws.cell(row = i+2, column = 1).value))
training_corpus.close()
