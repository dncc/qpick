extern crate byteorder;
extern crate fst;
#[macro_use]
extern crate serde_derive;
extern crate serde_json;

use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::sync::mpsc::{Receiver, Sender};

use std::fs::File;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::ops::Range;
use std::cmp::{Ordering, PartialOrd};
use std::collections::{HashMap, HashSet};

use byteorder::{ByteOrder, LittleEndian};
use fst::Map;
use std::io::SeekFrom;

use fst::Error;

#[macro_use]
pub mod util;
pub mod config;
pub mod ngrams;
pub mod merge;
pub mod shard;
pub mod builder;
pub mod stopwords;

use shard::QueryType;

extern crate scoped_threadpool;
use scoped_threadpool::Pool;

macro_rules! make_static_var_and_getter {
    ($fn_name:ident, $var_name:ident, $t:ty) => (
    static mut $var_name: Option<$t> = None;
    #[inline]
    fn $fn_name() -> &'static $t {
        unsafe {
            match $var_name {
                Some(ref n) => n,
                None => std::process::exit(1),
            }
       }
    })
}

make_static_var_and_getter!(get_id_size, ID_SIZE, usize);
make_static_var_and_getter!(get_bucket_size, BUCKET_SIZE, usize);
make_static_var_and_getter!(get_nr_shards, NR_SHARDS, usize);
make_static_var_and_getter!(get_shard_size, SHARD_SIZE, usize);

fn read_bucket(mut file: &File, addr: u64, len: u64) -> Vec<(u32, u8, u8, u8)> {
    let id_size = get_id_size();
    let bk_size = get_bucket_size();
    file.seek(SeekFrom::Start(addr)).unwrap();
    let mut handle = file.take((bk_size * id_size) as u64);
    let mut buf = vec![0u8; bk_size * id_size];

    let vlen = len as usize;
    let mut vector = Vec::<(u32, u8, u8, u8)>::with_capacity(vlen);

    // failure to read returns 0
    let n = handle.read(&mut buf).unwrap_or(0);

    if n > 0 {
        for i in 0..vlen {
            let j = i * id_size;
            vector.push((
                LittleEndian::read_u32(&buf[j..j + 4]),
                buf[j + 4],
                buf[j + 5],
                buf[j + 6],
            ));
        }
    }

    vector
}

// reading part
#[inline]
fn get_addr_and_len(ngram: &str, map: &fst::Map) -> Option<(u64, u64)> {
    match map.get(ngram) {
        Some(val) => return Some(util::elegant_pair_inv(val)),
        None => return None,
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Sid {
    pub id: u64,
    pub sc: f32,
}

struct ShardIds {
    ids: Vec<Sid>,
    norm: f32,
}

impl PartialOrd for Sid {
    fn partial_cmp(&self, other: &Sid) -> Option<Ordering> {
        if self.eq(&other) {
            self.id.partial_cmp(&other.id)
        } else {
            self.sc.partial_cmp(&other.sc)
        }
    }
}

impl PartialEq for Sid {
    fn eq(&self, other: &Sid) -> bool {
        self.sc == other.sc
    }
}

fn get_query_ids(
    ngrams: &Vec<(String, f32)>,
    map: &fst::Map,
    ifd: &File,
    count: usize,
) -> Result<ShardIds, Error> {
    let mut _ids = HashMap::new();
    let mut _norm: f32 = 0.0;
    let id_size = *get_id_size();
    let n = *get_shard_size() as f32;
    for &(ref ngram, ntr) in ngrams {
        // IDF score for the ngram
        let mut _idf: f32 = 0.0;
        match get_addr_and_len(ngram, &map) {
            // returns physical memory address and length of the vector (not a number of bytes)
            Some((addr, len)) => {
                for pqid_rem_tr_f in read_bucket(&ifd, addr * id_size as u64, len).iter() {
                    let pqid = pqid_rem_tr_f.0;
                    let reminder = pqid_rem_tr_f.1;
                    let qid = util::pqid2qid(pqid as u64, reminder, *get_nr_shards());
                    // TODO cosine similarity, normalize ngrams relevance at indexing time
                    let f = pqid_rem_tr_f.3;
                    let tr = pqid_rem_tr_f.2;
                    let weight = util::min((tr as f32) / 100.0, ntr) * (1.0 + f as f32 / 1000.0);
                    *_ids.entry(qid).or_insert(0.0) += weight * (n / len as f32).log(2.0);
                }
                // IDF for existing ngram
                _idf = (n / len as f32).log(2.0);
            }
            None => {
                // IDF for non existing ngram, occurs for the 1st time
                _idf = n.log(2.0);
            }
        }
        // compute the normalization score
        _norm += ntr * _idf;
    }

    let mut v: Vec<Sid> = _ids.iter()
        .map(|(id, sc)| Sid { id: *id, sc: *sc })
        .collect::<Vec<_>>();
    v.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Less).reverse());
    v.truncate(count);

    Ok(ShardIds {
        ids: v,
        norm: _norm,
    })
}

use std::cell::RefCell;

pub struct Qpick {
    path: String,
    config: config::Config,
    stopwords: HashSet<String>,
    terms_relevance: fst::Map,
    shards: Arc<Vec<Shard>>,
    thread_pool: RefCell<Pool>,
    shard_range: Range<u32>,
}

pub struct Shard {
    id: u32,
    map: fst::Map,
    shard: File,
}

#[derive(Debug)]
pub struct QpickResults {
    pub items_iter: std::vec::IntoIter<Sid>,
}

impl QpickResults {
    pub fn new(items_iter: std::vec::IntoIter<Sid>) -> QpickResults {
        QpickResults {
            items_iter: items_iter,
        }
    }

    pub fn next(&mut self) -> Option<Sid> {
        <std::vec::IntoIter<Sid> as std::iter::Iterator>::next(&mut self.items_iter)
    }
}

impl Qpick {
    fn new(path: String, shard_range_opt: Option<Range<u32>>) -> Qpick {
        let c = config::Config::init(path.clone());

        unsafe {
            // TODO set up globals, later should be available via self.config
            NR_SHARDS = Some(c.nr_shards);
            ID_SIZE = Some(c.id_size);
            BUCKET_SIZE = Some(c.bucket_size);
            SHARD_SIZE = Some(c.shard_size);
        }

        let shard_range = shard_range_opt.unwrap_or((0..c.nr_shards as u32));

        let stopwords = match stopwords::load(&c.stopwords_path) {
            Ok(stopwords) => stopwords,
            Err(_) => panic!("Failed to load stop-words!"),
        };

        let terms_relevance = match Map::from_path(&c.terms_relevance_path) {
            Ok(terms_relevance) => terms_relevance,
            Err(_) => panic!(
                "Failed to load terms rel. map: {}!",
                &c.terms_relevance_path
            ),
        };

        let mut shards = vec![];
        for i in shard_range.start..shard_range.end {
            let map_name = format!("{}/map.{}", path, i);
            let map = match Map::from_path(&map_name) {
                Ok(map) => map,
                Err(_) => panic!("Failed to load index map: {}!", &map_name),
            };

            let shard = OpenOptions::new()
                .read(true)
                .open(format!("{}/shard.{}", path, i))
                .unwrap();
            shards.push(Shard {
                id: i as u32,
                shard: shard,
                map: map,
            });
        }

        let thread_pool = Pool::new(shard_range.len() as u32);

        Qpick {
            config: c,
            path: path,
            stopwords: stopwords,
            terms_relevance: terms_relevance,
            shards: Arc::new(shards),
            thread_pool: RefCell::new(thread_pool),
            shard_range: shard_range,
        }
    }

    pub fn from_path(path: String) -> Self {
        Qpick::new(path, None)
    }

    pub fn from_path_with_shard_range(path: String, shard_range: Range<u32>) -> Self {
        Qpick::new(path, Some(shard_range))
    }

    fn get_ids(
        &self,
        ngrams: &HashMap<String, f32>,
        count: Option<usize>,
    ) -> Result<Vec<Sid>, Error> {
        let shard_count = match count {
            Some(1...50) => 100,
            _ => count.unwrap(),
        };


        let (sender, receiver): (Sender<f32>, Receiver<f32>) = mpsc::channel();

        let ref mut shards_ngrams: HashMap<usize, HashMap<String, f32>> = HashMap::new();

        for (ngram, sc) in ngrams {
            let shard_id = util::jump_consistent_hash_str(ngram, self.config.nr_shards as u32);

            if shard_id >= self.shard_range.end || shard_id < self.shard_range.start {
                continue;
            }

            let sh_ngrams = shards_ngrams
                .entry(shard_id as usize)
                .or_insert(HashMap::new());
            sh_ngrams.insert(ngram.to_string(), *sc);
        }

        let nr_threads = shards_ngrams.len();
        let nr_shards = (self.shard_range.end - self.shard_range.start) as usize;
        let mut _ids: Arc<Mutex<Vec<Vec<Sid>>>> = Arc::new(Mutex::new(vec![vec![]; nr_shards]));

        self.thread_pool.borrow_mut().scoped(|scoped| {
            for sh_id_sh_ngram in shards_ngrams {
                let j = *sh_id_sh_ngram.0 - self.shard_range.start as usize;
                let sender = sender.clone();
                let _ids = _ids.clone();
                let shards = self.shards.clone();

                let mut sh_ngrams: Vec<(String, f32)> = sh_id_sh_ngram
                    .1
                    .clone()
                    .into_iter()
                    .map(|(n, sc)| (n, sc))
                    .collect();
                sh_ngrams.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Less));

                scoped.execute(move || {
                    let sh_ids = match get_query_ids(
                        &sh_ngrams,
                        &shards[j].map,
                        &shards[j].shard,
                        shard_count,
                    ) {
                        Ok(ids) => ids,
                        Err(_) => {
                            println!("Failed to retrive ids from shard: {}", &shards[j].id);
                            ShardIds {
                                ids: vec![],
                                norm: 0.0,
                            }
                        }
                    };

                    // obtaining lock might fail, handle it
                    let mut _ids = match _ids.lock() {
                        Ok(_ids) => _ids,
                        Err(poisoned) => poisoned.into_inner(),
                    };

                    _ids[j] = sh_ids.ids;
                    sender.send((sh_ids.norm)).unwrap();
                });
            }
        });

        let mut norm: f32 = 0.0;
        for _ in 0..nr_threads {
            norm += receiver.recv().unwrap();
        }

        // deref MutexGuard returned from _ids.lock().unwrap()
        let data = (*_ids.lock().unwrap()).clone();
        let mut hdata: HashMap<u64, f32> = HashMap::new();
        for sh_vec in data {
            for s in sh_vec {
                *hdata.entry(s.id).or_insert(0.0) += s.sc;
            }
        }

        // turn for into vec and sort, that is expected from python extension
        // TODO avoid sorting per shard and turning into vector and sorting again,
        //      use a different data structure
        let mut vdata: Vec<Sid> = hdata
            .into_iter()
            .map(|(id, sc)| {
                Sid {
                    id: id,
                    sc: sc / norm,
                }
            })
            .collect();
        vdata.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Less).reverse());
        vdata.truncate(count.unwrap_or(100)); //TODO put into config

        Ok(vdata)
    }

    pub fn get_str(&self, query: &str, count: u32) -> String {
        let mut res: Vec<(u64, f32)> = self.get(query, 30*count)
            .into_iter()
            .map(|s| (s.id, s.sc))
            .collect();
        res.truncate(count as usize);

        serde_json::to_string(&res).unwrap()
    }

    pub fn nget_str(&self, queries: &str, count: u32) -> String {
        let qvec: Vec<String> = serde_json::from_str(queries).unwrap();
        let mut res: Vec<(u64, f32)> = self.nget(&qvec, 30*count)
            .into_iter()
            .map(|s| (s.id, s.sc))
            .collect();
        res.truncate(count as usize);

        serde_json::to_string(&res).unwrap()
    }

    pub fn get_results(&self, query: &str, count: u32) -> QpickResults {
        QpickResults::new(self.get(query, count).into_iter())
    }

    pub fn nget_results(&self, qvec: &Vec<String>, count: u32) -> QpickResults {
        QpickResults::new(self.nget(qvec, count).into_iter())
    }

    pub fn get(&self, query: &str, count: u32) -> Vec<Sid> {
        if query == "" || count == 0 {
            return vec![];
        }

        let ref ngrams: HashMap<String, f32> =
            ngrams::parse(&query, &self.stopwords, &self.terms_relevance, QueryType::Q);

        self.get_ids(ngrams, Some(count as usize)).unwrap()
    }

    pub fn nget(&self, qvec: &Vec<String>, count: u32) -> Vec<Sid> {
        if qvec.len() == 0 || count == 0 {
            return vec![];
        }

        let ref mut ngrams: HashMap<String, f32> = HashMap::new();
        for query in qvec.iter() {
            for (ngram, sc) in
                ngrams::parse(&query, &self.stopwords, &self.terms_relevance, QueryType::Q)
            {
                ngrams.insert(ngram, sc);
            }
        }

        self.get_ids(ngrams, Some(count as usize)).unwrap()
    }

    pub fn merge(&self) -> Result<(), Error> {
        println!("Merging index maps from: {:?}", &self.path);
        merge::merge(&self.path, self.config.nr_shards as usize)
    }

    pub fn shard(
        file_path: String,
        nr_shards: usize,
        output_dir: String,
        concurrency: usize,
    ) -> Result<(), std::io::Error> {
        println!(
            "Creating {:?} shards from {:?} to {:?}",
            nr_shards,
            file_path,
            output_dir
        );
        shard::shard(&file_path, nr_shards, &output_dir, concurrency)
    }

    pub fn index(
        input_dir: String,
        first_shard: usize,
        last_shard: usize,
        output_dir: String,
    ) -> Result<(), Error> {
        println!(
            "Compiling {:?} shards from {:?} to {:?}",
            last_shard - first_shard,
            input_dir,
            output_dir
        );

        builder::index(&input_dir, first_shard, last_shard, &output_dir)
    }
}

#[allow(dead_code)]
fn main() {}
