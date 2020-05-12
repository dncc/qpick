qpick
===

Search for similar short strings (queries in particular) based on word-vector similarity (cosine distance) and keyword matches (TF-IDF-like scoring function).

#### Install

If Rust is already installed, run:

```
git clone https://github.com/dncc/qpick.git && cd qpick
make install
```

Without Rust installation, run:

```
git clone https://github.com/dncc/qpick.git && cd qpick
make install/rust
make install
```

#### Indexing

An expected input to build a search index is a directory containing `*.gz` files. Each line found in files is indexed as a separate query. A unique ordinal number of the line represents its unique id.

The search index is built in 2 stages, sharding and compilation:

 - To shard a test data set from `./test/sample.gz` into 32 shards and store them in the `./index` directory, run:

```
./bin/target/release/qpick shard ./test/sample.gz 32 ./index ""
```

 - To compile shards into a search index and store it in the `./index` directory, run:

```
./bin/target/release/qpick index ./index 0 32 ./index
```

Each shard is compiled in parallel. Since it could take a lot of RAM to compile a large number of queries at once (e.g. 5 billion), it is possible to compile only a few shards at the time. The following command would compile just 5 shards (shards 0, 1, 2, 3 and 4):

```
./bin/target/release/qpick index ./index 0 5 ./index
```

#### Searching

Once indexing is completed, searching can be done from the command line:

For instance:

```
./bin/target/release/qpick get "changing mac os menu bar" 10
```

gives (with the `./test/sample.gz` data set):
```
#=> [(0, 0.39147103, "changing mac menu bar"), (1, 0.5766359, "emails menu bar mac os")]
```

where each result is a tuple, containing:

  - query id,

  - distance from the original query, in the range from 0.0 to 1.0, where zero is the best (the closest) and 1 is the worst (the furthest) result,

  - and query.

Or the same example from python:

```python

from rust_qpick import Qpick
qpick = Qpick("./index")

# lookup with one query
list(qpick.get('changing mac os menu bar', 10))

# => [(0, 0.39147108793258667, 'changing mac menu bar'),
#     (1, 0.5766359567642212, 'emails menu bar mac os')]

```
