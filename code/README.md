# Code

This directory provides with the main process and all the functions implemented. This directory is seperated so search the files in it with Control-F will be easier.

## File Description

main.py: the main process

func_\*.py: provide relevant function to main.py

Data and w2v models used are from data and word2vec, correspondingly.

## Current Avaliable

Vectorization: gensim.models.word2vec.Word2Vec + [add words to form sentences, link words to form sentences]

Classification: sklearn.svm.SVC + [multi-class classification, binary-class classification with 1vsRest]

Cross Validation: divide train and validation by a certain label(esp.by file), this supports [n-fold, leave-one-out]

## Modification Instruction

To modify vectorizing methods provided, add methods to func_vectorize.py.

To modify classifying methods provided, add methods to func_classify.py.

## Future Work

Vectorization: apply Doc2Vec.

Classification: add deep learning methods, esp. those who deal with sequential data, like LSTM or RNN and so on.

Cross Validation: not planned.
