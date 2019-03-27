# word2vec
This directory contains the training corpus *(too large in size to be uploaded, see below)* and codes too build w2v model.


**text8**

text8 is a commonly used corpus for w2v model training， which could be downloaded at http://mattmahoney.net/dc/textdata, where more information of text8 and corpus of larger scale could also be accessed.

*text8 must be added to this folder before running word2vec_training_corpus.py.*


**word2vec_training_corpus.py**

word2vec_training_corpus.py is used to generate training_corpus we used from text8 and ../tmp_data_label.xlsx.

*word2vec_training_corpus.py must be run before running word2vec_model.py.*


**word2vec_model.py**

word2vec_model.py is used to generate w2v model we used in the main program.

*word2vec_model.py must be run before running ../main.py.*
