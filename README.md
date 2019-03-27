# Simple-Text-Classification

A simple framework for text classification, including the parts of text vectorization, classification and cross validation.

## Run

word2vec/word2vec_model.py must be run before running main.py, to generate w2v model. Then main.py could be run. Output information includes the confuse matrixs and relevant statistical data.

## Current Avaliable

Vectorization: gensim.models.word2vec.Word2Vec + [add words to form sentences, link words to form sentences]

Classification: sklearn.svm.SVC + [multi-class classification, binary-class classification with 1vsRest]

Cross Validation: divide train and validation by a certain label(esp.by file), this supports [n-fold, leave-one-out]

## Modify

To modify vectorizing methods provided, add methods to **func_vectorize.py**.

To modify classifying methods provided, add methods to **func_classify.py**.

## Future Work

Vectorization: apply Doc2Vec.

Classification: add deep learning methods, esp. those who deal with sequential data, like LSTM or RNN and so on.

Cross Validation: not planned.
