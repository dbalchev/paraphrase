import numpy
import gzip
from pathlib import Path
import pickle
import io
np = numpy

DATABASE_FILE = "e:\\random-stuff\\downloads\\GoogleNews-vectors-negative300.bin.gz"
WORD2VEC_PICKLE_FILE = Path("word2vec.pickle.gz")

def read_wordvec_database(database_file = DATABASE_FILE):
    FLOAT_SIZE = 4
    FLOAT_TYPE = np.dtype("float32")
    with gzip.open(database_file) as db:
        db = io.BufferedReader(db)
        def read_char_generator():
            while True:
                c = db.read(1)
                if c is None or c == b"\n" or c == b" ":
                    break
                yield c

        def read_string():
            return b"".join(read_char_generator()).decode()

        n_words, vec_dim = map(int, db.readline().decode().split())
        for _ in range(n_words):
            word = read_string()
            vec  = np.frombuffer(db.read(FLOAT_SIZE * vec_dim), dtype=FLOAT_TYPE)
            yield word, vec
