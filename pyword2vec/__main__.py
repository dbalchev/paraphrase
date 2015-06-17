


def get_word2vec_map():
    if WORD2VEC_PICKLE_FILE.exists():
        with gzip.open(str(WORD2VEC_PICKLE_FILE)) as pickle_reader:
            return pickle.load(pickle_reader)
    else:
    word2vec = {word:vec for (word,vec) in read_wordvec_database()}
        with gzip.open(str(WORD2VEC_PICKLE_FILE), "wb") as pickle_writer:
            pickle.dump(word2vec, pickle_writer)
        return word2vec

word2vec = get_word2vec_map()
print(len(word2vec), word2vec["company"].shape)
