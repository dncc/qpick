#include <stdint.h>

typedef struct Qpick Qpick;

Qpick* qpick_init(char*);
Qpick* qpick_init_with_shard_range(char*, uint32_t, uint32_t);
void qpick_free(Qpick*);
char* qpick_get_as_string(Qpick*, char*, uint32_t, uint8_t);
void string_free(char*);

/**
   Iterator
**/
typedef struct {
  uint64_t  qid;
  float     sc;
} QpickSearchItem;

typedef struct SearchResults SearchResults;

SearchResults* qpick_get(Qpick*, char*, uint32_t);
QpickSearchItem* qpick_search_iter_next(SearchResults*);

void qpick_search_results_free(SearchResults*);
void qpick_search_item_free(QpickSearchItem*);

/**
   nget api
**/
typedef struct QpickQueryVec QpickQueryVec;
QpickQueryVec* query_vec_init();
void query_vec_free(QpickQueryVec*);
void query_vec_push(QpickQueryVec*, char*);

