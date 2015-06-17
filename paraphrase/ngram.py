import re
from .names import with_name
from functools import partial

NONWORD_REGEX = re.compile("\W")
def get_words(tweet):
    return [NONWORD_REGEX.sub("", word.lower()) for word in tweet.split()]

def _windows(arr_like, n):
    l = len(arr_like)
    for i in range(l - n + 1):
        yield tuple(arr_like[i:(i+n)])

def gen_word_ngrams(tweet, n):
    words = get_words(tweet)
    return set(_windows(words, n))

def gen_chars_ngrams(tweet, n):
    words = get_words(tweet)
    result = set()
    for word in words:
        result |= set(_windows(word, n))
    return result

def get_ngram_features(ngram_gen, data_row):
    s1, s2 = map(ngram_gen, (data_row.sent_1, data_row.sent_2))
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    return [len(s1), len(s2), len(s1 & s2), len(s1 | s2),
        len(s1 - s2), len(s2 - s1)]

C1= with_name(partial(get_ngram_features, partial(gen_chars_ngrams, n=1)), "C1")
C2= with_name(partial(get_ngram_features, partial(gen_chars_ngrams, n=2)), "C2")

V1 = with_name(partial(get_ngram_features, partial(gen_word_ngrams, n=1)), "V1")
V2 = with_name(partial(get_ngram_features, partial(gen_word_ngrams, n=2)), "V2")
