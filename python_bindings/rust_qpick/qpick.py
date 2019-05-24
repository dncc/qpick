import os
from .lib import ffi, lib

class QpickResults(object):
    def __init__(self, ptr, next_fn, free_fn, free_item_fn, autom_ptr=None,
                 autom_free_fn=None):
        self._free_fn = free_fn
        self._free_item_fn = free_item_fn
        self._ptr = ffi.gc(ptr, free_fn)

        self._next_fn = next_fn
        if autom_ptr:
            self._autom_ptr = ffi.gc(autom_ptr, autom_free_fn)
            self._autom_free_fn = autom_free_fn
        else:
            self._autom_ptr = None

    def _free(self):
        # TODO: We could safely free the structures before the GC does,
        #       but unfortunately removing GC-callbacks is only supported
        #       in cffi >= 1.7, which is not yet released.
        # self._free_fn(self._ptr)
        # # Clear GC hook to prevent double-free
        # ffi.gc(self._ptr, None)
        # if self._autom_ptr:
        #     self._autom_free_fn(self._autom_ptr)
        #     ffi.gc(self._autom_ptr, None)
        pass

    def __iter__(self):
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        raise NotImplementedError

class QpickSearchResults(QpickResults):
    def __next__(self):
        itm = self._next_fn(self._ptr)
        if itm == ffi.NULL:
            self._free()
            raise StopIteration

        qid = itm.qid
        sc = itm.sc
        query = ffi.string(itm.query).decode('utf8')
        self._free_item_fn(itm)

        return (qid, sc, query)

class QpickDistResults(QpickResults):
    def __next__(self):
        itm = self._next_fn(self._ptr)
        if itm == ffi.NULL:
            self._free()
            raise StopIteration

        query = ffi.string(itm.query).decode('utf8')
        dist = itm.dist
        self._free_item_fn(itm)

        return (query, dist)


class Qpick(object):
    def __init__(self, dir_path=None, start_shard=None, end_shard=None, _pointer=None):
        """Loads a query index from a given directory.

        :param dir_path:    Directory path to index on disk
        :param start_shard: Index of the first shard in the index to load
        :param end_shard:   Index of the last shard in the index to load
        """
        # self._ctx = ffi.gc(lib.qpick_context_new(), lib.qpick_context_free)

        if dir_path:
            if not os.path.isdir(dir_path):
                raise Exception("%s is not a directory!" % dir_path)

            if type(dir_path) == str:
                dir_path = dir_path.encode('utf-8')

            if start_shard is not None and end_shard is not None:
                if end_shard <= start_shard:
                    raise Exception("Index of the last shard has to be greater than the start index!")
                else:
                    # returns a pointer to rust Qpick struct
                    s = lib.qpick_init_with_shard_range(dir_path, start_shard, end_shard)

            else:
                # returns a pointer to rust Qpick struct
                s = lib.qpick_init(dir_path)
        else:
            s = _pointer

        self._ptr = ffi.gc(s, lib.qpick_free)

    # qpick.get('a')
    def get(self, query, count=100):
        if type(query) == str:
            query = query.encode('utf-8')

        res_ptr = lib.qpick_get(self._ptr, query, count)
        return QpickSearchResults(res_ptr,
                                lib.qpick_search_iter_next,
                                lib.qpick_search_results_free,
                                lib.qpick_search_item_free)

    # qpick.get_distances('q', ['a', 'b', 'c'])
    def get_distances(self, query, candidates):
        qvec = lib.string_vec_init()
        qvec_ptr = ffi.gc(qvec, lib.string_vec_free)

        if type(query) == str:
            query = query.encode('utf-8')

        for q in candidates:
            if type(q) == str:
                q = q.encode('utf-8')
            lib.string_vec_push(qvec_ptr, q)

        res_ptr = lib.qpick_get_distances(self._ptr, query, qvec_ptr)

        return QpickDistResults(res_ptr,
                                lib.qpick_dist_iter_next,
                                lib.qpick_dist_results_free,
                                lib.qpick_dist_item_free)


def shard(file_path, nr_shards, output_dir, prefixes=[]):
    if type(file_path) == str:
        file_path = file_path.encode()

    if type(output_dir) == str:
        output_dir = output_dir.encode()

    pref_vec = lib.string_vec_init()
    pref_vec_ptr = ffi.gc(pref_vec, lib.string_vec_free)
    for p in prefixes:
        if type(p) == str:
            p = p.encode('utf-8')
        lib.string_vec_push(pref_vec_ptr, p)

    lib.qpick_shard(file_path, nr_shards, output_dir, pref_vec_ptr)

def compile_i2q(file_path, output_dir):
    if type(file_path) == str:
        file_path = file_path.encode()

    if type(output_dir) == str:
        output_dir = output_dir.encode()

    lib.qpick_compile_i2q(file_path, output_dir)


def index(input_dir, first_shard, last_shard, output_dir):
    if type(input_dir) == str:
        input_dir = input_dir.encode('utf-8')

    if type(output_dir) == str:
        output_dir = output_dir.encode('utf-8')

    lib.qpick_index(input_dir, first_shard, last_shard, output_dir)
