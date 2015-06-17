import bz2
import pickle
from struct import Struct
from collections import namedtuple
from .reader import read_wordvec_database
from contextlib import contextmanager
from functools import lru_cache
"""
The struct representing the file header.
It is 'w2v\0' as a magic number, then the offset to the pickled
_offset_map.
"""
HEADER_STRUCT = Struct("<4cQ")
CHUNK_HEADER  = Struct("<L")
MAGIC = (b'w', b'2', b'v', b'\0')

class VectorOffset(namedtuple("VectorOffset", ["chunk_start", "chunk_offset"])):
    __slots__ = []

class MapBuilder:
    def __init__(self, io, vecs_per_chunk=8):
        self._io = io
        self._vecs_per_chunk = vecs_per_chunk
        self._offset_map = {}
        self._vector_buffer = []
        self._io.seek(HEADER_STRUCT.size)

    def append(self, w2v):
        word, vec = w2v
        self._append_vec(vec)
        self._offset_map[word] = \
            VectorOffset(self._io.tell(), len(self._vector_buffer) - 1)

    def flush(self):
        self._flush_vec_buffer()
        map_offset = self._io.tell()
        pickle.dump(self._offset_map, self._io)
        print("offset after dump", self._io.tell())
        self._io.seek(0)
        self._io.write(HEADER_STRUCT.pack(*(MAGIC + (map_offset,))))
        print("map_offset", map_offset)
        self._io.flush()

    def _append_vec(self, vec):
        if len(self._vector_buffer) == self._vecs_per_chunk:
            self._flush_vec_buffer()
        self._vector_buffer.append(vec)

    def _flush_vec_buffer(self):
        if len(self._vector_buffer) == 0:
            return
        serialized = pickle.dumps(self._vector_buffer)
        compressed = bz2.compress(serialized)
        self._io.write(CHUNK_HEADER.pack(len(compressed)))
        self._io.write(compressed)
        self._vector_buffer.clear()


def create_filemap(binary_file, filemap_file):
    with open(filemap_file, "wb") as out_file:
        map_builder = MapBuilder(out_file)
        for w2v in read_wordvec_database(binary_file):
            map_builder.append(w2v)
        map_builder.flush()

class MagicBytesError(Exception):
    pass

class MapReader:
    def _get_vec_chunk(self, chunk_start):
        self._io.seek(chunk_start)
        compressed_len, = CHUNK_HEADER.unpack(self._io.read(CHUNK_HEADER.size))
        serialized = bz2.decompress(self._io.read(compressed_len))
        vec_buff = pickle.loads(serialized)
        return vec_buff

    def __init__(self, io, cache_size = 128):
        self._io = io
        io.seek(0)
        header_fields = HEADER_STRUCT.unpack(io.read(HEADER_STRUCT.size))
        if header_fields[:4] != MAGIC:
            raise MagicBytesError("the first 4 bytes {} should be {}".format(header_fields[:4], MAGIC))
        map_offset = header_fields[4]
        io.seek(map_offset)
        self._offset_map = pickle.load(io)
        self.get_vec_chunk = lru_cache(cache_size)(self._get_vec_chunk)
        self._added_items = {}

    def items(self):
        for wv in self._added_items.items():
            yield wv
        for word in self._offset_map.keys():
            if word not in self._added_items:
                yield (word, self[word])

    def __contains__(self, word):
        return word in self._offset_map or word in self._added_items

    def __getitem__(self, word):
        if word in self._added_items:
            return self._added_items[word]
        if word in self._offset_map:
            vec_offset = self._offset_map[word]
            vec_buff = self.get_vec_chunk(vec_offset.chunk_start)
            return vec_buff[vec_offset.chunk_offset]
        raise KeyError("word \"{}\" is not in MapReader".format(word))

    def __setitem__(self, word, value):
        self._added_items[word] = value


@contextmanager
def word2vec_map(filemap_file):
    with open(filemap_file, "rb") as file_io:
        yield MapReader(file_io)


def test_filemap(binary_file, filemap_file):
    with word2vec_map(filemap_file) as w2v_map:
        for word, vec in read_wordvec_database(binary_file):
            if (w2v_map[word] != vec).any():
                raise Exception("error for word " + word)
