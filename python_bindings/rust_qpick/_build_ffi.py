from cffi import FFI

ffi = FFI()
ffi.set_source('rust_qpick._ffi', None)
ffi.cdef("""
    typedef struct Qpick Qpick;

    Qpick* qpick_init(char*);
    Qpick* qpick_init_with_shard_range(char*, uint32_t, uint32_t);
    void qpick_free(Qpick*);
    char* qpick_get_as_string(Qpick*, char*, uint32_t);
    char* qpick_nget_as_string(Qpick*, char*, uint32_t);
    void string_free(char*);

    /**
       Iterator
    **/
    typedef struct {
        uint64_t  qid;
        float     sc;
    } QpickSearchItem;

    typedef struct SearchResults SearchResults;

    SearchResults* qpick_get(Qpick*, char*, uint32_t, float);
    QpickSearchItem* qpick_search_iter_next(SearchResults*);

    void qpick_search_results_free(SearchResults*);
    void qpick_search_item_free(QpickSearchItem*);

    /**
       nget API
    **/
    typedef struct StringVec StringVec;
    StringVec* string_vec_init();
    void string_vec_free(StringVec*);
    void string_vec_push(StringVec*, char*);

    SearchResults* qpick_nget(Qpick*, StringVec*, uint32_t, float);

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
        shard and index API
    **/
    void qpick_shard(char*, uint32_t, char*, uint32_t, StringVec*);
    void qpick_index(char*, uint32_t, uint32_t, char*);

""")

if __name__ == '__main__':
    ffi.compile()
