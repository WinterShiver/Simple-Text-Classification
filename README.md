# Simple-Text-Classification
A simple framework for text classification, including the parts of text vectorization, classification and cross validation.

## Current Avaliable
Vectorization: gensim.models.word2vec.Word2Vec + [add words to form sentences, link words to form sentences]
Classification: sklearn.svm.SVC + [multi-class classification, binary-class classification with 1vsRest]
Cross Validation: divide train and validation by a certain label(esp.by file), this supports [n-fold, leave-one-out]
