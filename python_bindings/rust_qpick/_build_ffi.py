from cffi import FFI

ffi = FFI()
ffi.set_source('rust_qpick._ffi', None)
ffi.cdef("""
    typedef struct Qpick Qpick;

    Qpick* qpick_init(char*);
    Qpick* qpick_init_with_shard_range(char*, uint32_t, uint32_t);
    void qpick_free(Qpick*);
    char* qpick_get_as_string(Qpick*, char*);
    void string_free(char*);

    /**
       Iterator
    **/
    typedef struct {
        uint64_t  qid;
        float     sc;
    } QpickItem;

    typedef struct QpickResults QpickResults;

    QpickResults* qpick_get(Qpick*, char*, uint32_t);
    QpickItem* qpick_iter_next(QpickResults*);

    void qpick_results_free(QpickResults*);
    void qpick_item_free(QpickItem*);

    /**
       nget api
    **/
    typedef struct QpickQueryVec QpickQueryVec;
    QpickQueryVec* query_vec_init();
    void query_vec_free(QpickQueryVec*);
    void query_vec_push(QpickQueryVec*, char*);

    QpickResults* qpick_nget(Qpick*, QpickQueryVec*, uint32_t);

""")

if __name__ == '__main__':
    ffi.compile()
