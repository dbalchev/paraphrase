from pyword2vec import MapReader
from .ngram import get_words
from .names import named
import numpy as np
from math import sqrt

WORD2VEC_FILEMAP = "filemap.w2v"

_word2vec = None
def word2vec(word):
    global _word2vec
    if not _word2vec:
        filemap_io = open(WORD2VEC_FILEMAP, "rb")
        _word2vec = MapReader(filemap_io)
    if word not in _word2vec:
        rnd = np.random.ranf(next(iter(_word2vec.items()))[1].shape)
        _word2vec[word] = rnd
    return _word2vec[word]

def _norm(v):
    return sqrt(np.dot(v, v))

def _get_words_vec(sentence):
    return sum(word2vec(word) for word in get_words(sentence))

@named("word2vec_dot")
def word2vec_features(data_row):
    u, v = map(_get_words_vec, (data_row.sent_1, data_row.sent_2))
    return [np.dot(u, v) / _norm(u) / _norm(v)]
