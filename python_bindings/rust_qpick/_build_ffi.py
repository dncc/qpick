from cffi import FFI

ffi = FFI()
ffi.set_source('rust_qpick._ffi', None)
ffi.cdef("""
    typedef struct Qpick Qpick;

    Qpick* qpick_init(char*);
    Qpick* qpick_init_with_shard_range(char*, uint32_t, uint32_t);
    void qpick_free(Qpick*);
    void string_free(char*);

    /**
       Iterator
    **/
    typedef struct {
        uint64_t  qid;
        float     sc;
        char*     query;
    } QpickSearchItem;

    typedef struct SearchResults SearchResults;

    SearchResults* qpick_get(Qpick*, char*, uint32_t);
    QpickSearchItem* qpick_search_iter_next(SearchResults*);

    void qpick_search_results_free(SearchResults*);
    void qpick_search_item_free(QpickSearchItem*);

    /**
       string vec
    **/
    typedef struct StringVec StringVec;
    StringVec* string_vec_init();
    void string_vec_free(StringVec*);
    void string_vec_push(StringVec*, char*);

    /**
       Distance Iterator
    **/
    typedef struct {
        char*  query;
        float  dist;
    } QpickDistItem;

    typedef struct DistResults DistResults;

    DistResults* qpick_get_distances(Qpick*, char*, StringVec*);
    QpickDistItem* qpick_dist_iter_next(DistResults*);

    void qpick_dist_results_free(DistResults*);
    void qpick_dist_item_free(QpickDistItem*);

    /**
        shard, index, compile_i2q API
    **/
    void qpick_shard(char*, uint32_t, char*, StringVec*, uint8_t);
    void qpick_compile_i2q(char*, char*);
    void qpick_index(char*, uint32_t, uint32_t, char*);

""")

if __name__ == '__main__':
    ffi.compile()
