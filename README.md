qpick
===

Search for similar short strings (queries in particular).

#### Install
```
git clone https://github.com/dncc/qpick.git && git checkout
cd qpick
cargo build --release --manifest-path ./bin/Cargo.toml --verbose
# build python bindings
cd python_bindings
python setup.py install

```

#### Build Index

The following command would partition the input data from `./parts/input.txt` into 32 shards and store them in `./index` directory:

```
./bin/target/release/qpick shard "./parts/input.txt" 32 "./index"
```
Input is a text file where each line represents one query. The query id in the index will be the ordinal number of the line.

After shards are created the following command will compile them and store the index file in the same `./index` directory:

```
./bin/target/release/qpick index "./index" 0 32 "./index"
```

Each shard is compiled in parallel. Since to compile a large number of queries (e.g. 5 billion) takes a lot of memory, it is possible to compile only a few shards at once. The following command would for instance compile just 5 shards (0, 1, 2, 3, 4):

```
./bin/target/release/qpick index "./index" 0 5 "./index"
```

#### Search

From command line:
```
./bin/target/release/qpick get "gdp per capita USA" 100
```

In python:

```python

from rust_qpick import Qpick
qpick = Qpick("/raid/qpick/index")

# lookup with one query
results = list(qpick.get('a formatter for python files', 10))

# lookup with multiple queries
results = list(qpick.nget(['a formatter for python files', 'formatting library for python'], 100))

```
