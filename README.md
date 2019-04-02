# Simple-Text-Classification

A simple framework for text classification, including the parts of text vectorization, classification and cross validation.

## Run

word2vec/word2vec_model.py must be run before running main.py, to generate w2v model. Then main.py could be run. Output information includes the confuse matrixs and relevant statistical data.

## Current Avaliable

See in code/README.md.

## File Description

pre_data_cleaning.py: convert tmp_data_label.xlsx into tmp_data_label_cleaned.xlsx, not used in main.py

\*.xlsx: text data used

## Modify

To modify vectorizing methods provided, add methods to **func_vectorize.py**.

To modify classifying methods provided, add methods to **func_classify.py**.

## Future Work

Vectorization: apply Doc2Vec.

Classification: add deep learning methods, esp. those who deal with sequential data, like LSTM or RNN and so on.

Cross Validation: not planned.
