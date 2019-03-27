# vectorize_model.py
# -*-coding:utf-8-*-

# para
wv_dim = 300

# vectorize: word2vec + normal linear
# build model: irrelevant to our data
from gensim.models.word2vec import LineSentence
from gensim.models.word2vec import Text8Corpus
line_sentences = Text8Corpus
sentences = line_sentences("training_corpus")
from gensim.models.word2vec import Word2Vec
model = Word2Vec(sentences, size = 300, min_count = 0)
# size: dimension of vector
# min_count: word appearing in n sentences, only count if n >= min_count
# here min_count = 0 to avoid sheng_pi_zi
# model.build_vocab("text8") # 生成词表
model.save("word2vec_300.model")
# test
print(model.wv['brain'].shape)
print(model.wv['brain'])
print(model.most_similar(positive=['brain'], negative=['carbon'], topn=5))

model = Word2Vec(sentences, size = 128, min_count = 0)
model.save("word2vec_128.model")
# test
print(model.wv['brain'].shape)
print(model.wv['brain'])
print(model.most_similar(positive=['brain'], negative=['carbon'], topn=5))

