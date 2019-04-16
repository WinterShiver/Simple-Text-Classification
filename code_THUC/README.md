# code for THUC news

Run classification code on THUC news.

THUC news is available at http://thuctc.thunlp.org/#%E4%B8%AD%E6%96%87%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB%E6%95%B0%E6%8D%AE%E9%9B%86THUCNews 

work_dir = '../THUCNews'

## Run

Run clean_THUC.py to get cleaned str_list & label_list, of the form of .pkl file.

Run main_THUC to load str_list & label_list and do classification.

## Current Work and Future Work

vectorize: only include TF-IDF. Vectorization part should be seperated later.

## Directory Description

**THUCNews** contains data used in learning. In the directory there are sub-dirs, whose names are types of news, and they contain massive .txt files of news text.

## Modification Instruction

clean_THUC.py is independent, but main_THUC.py is dependent to func_\*.py files in ../code.
