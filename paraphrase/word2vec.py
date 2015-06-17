from pyword2vec import MapReader
from .ngram import get_words
from .names import named
import numpy as np
from math import sqrt

WORD2VEC_FILEMAP = "filemap.w2v"

_word2vec = MapReader(open(WORD2VEC_FILEMAP, "rb"))

def _norm(v):
    return sqrt(np.dot(v, v))

def _get_words_vec(sentence):
    return sum(_word2vec[word] for word in get_words(sentence)\
        if word in _word2vec)

@named("word2vec_dot")
def word2vec_features(data_row):
    u, v = map(_get_words_vec, (data_row.sent_1, data_row.sent_2))
    return [np.dot(u, v) / _norm(u) / _norm(v)]
