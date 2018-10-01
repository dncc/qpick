#[macro_use]
extern crate lazy_static;
extern crate byteorder;
extern crate fst;
extern crate libc;
#[macro_use]
extern crate serde_derive;
extern crate memmap;
extern crate serde_json;

use std::io;
use std::sync::Arc;
use std::fs::OpenOptions;
use std::ops::Range;
use std::cmp::{Ordering, PartialOrd};
use std::collections::{HashMap, HashSet};

use byteorder::{ByteOrder, LittleEndian};
use fst::Map;
use fst::raw::{Fst, MmapReadOnly};
use memmap::Mmap;

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

use util::{BRED, BYELL, ECOL};

macro_rules! make_static_var_and_getter {
    ($fn_name: ident, $var_name: ident, $t: ty) => {
        static mut $var_name: Option<$t> = None;
        #[inline]
        fn $fn_name() -> &'static $t {
            unsafe {
                match $var_name {
                    Some(ref n) => n,
                    None => std::process::exit(1),
                }
            }
        }
    };
}

extern crate rayon;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use rayon::ThreadPoolBuilder;

make_static_var_and_getter!(get_bucket_size, BUCKET_SIZE, usize);
make_static_var_and_getter!(get_nr_shards, NR_SHARDS, usize);
make_static_var_and_getter!(get_shard_size, SHARD_SIZE, usize);
make_static_var_and_getter!(get_thread_pool_size, THREAD_POOL_SIZE, usize);

#[inline]
fn read_bucket(
    mmap: &memmap::Mmap,
    addr: usize,
    len: usize,
    id_size: usize,
) -> Vec<(u32, u8, u8, u8)> {
    let buf = &mmap[addr..addr + len * id_size];
    (0..len)
        .map(|i| {
            let j = i * id_size;
            (
                LittleEndian::read_u32(&buf[j..j + 4]),
                buf[j + 4],
                buf[j + 5],
                buf[j + 6],
            )
        })
        .collect::<Vec<(u32, u8, u8, u8)>>()
}

// reading part
#[inline]
fn get_addr_and_len(ngram: &str, map: &fst::Map) -> Option<(u64, u64)> {
    match map.get(ngram) {
        Some(val) => return Some(util::elegant_pair_inv(val)),
        None => return None,
    }
}

// Advise the OS on the random access pattern of data.
// Taken from https://docs.rs/crate/madvise/0.1.0
#[cfg(unix)]
fn advise_ram(data: &[u8]) -> io::Result<()> {
    unsafe {
        let result = libc::madvise(
            util::as_ptr(data) as *mut libc::c_void,
            data.len(),
            libc::MADV_RANDOM as libc::c_int,
        );

        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
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

#[inline]
fn get_query_ids(
    ngrams: &HashMap<String, f32>,
    map: &fst::Map,
    ifd: &memmap::Mmap,
    id_size: usize,
) -> Result<ShardIds, Error> {
    let mut _norm: f32 = 0.0;
    let mut _ids: Vec<Sid> = vec![];
    let n = *get_shard_size() as f32;
    let nr_shards = *get_nr_shards();
    let bucket_size = *get_bucket_size();
    for (ngram, ntr) in ngrams {
        // IDF score for the ngram
        let mut _idf: f32 = 0.0;
        match get_addr_and_len(ngram, &map) {
            // returns physical memory address and length of the vector (not a number of bytes)
            Some((addr, len)) => {
                // IDF for existing ngram
                _idf = (n / len as f32).log(2.0);
                let mem_addr = addr as usize * id_size;
                let len = util::min(len as usize, bucket_size);
                for &(pqid, rem, trel, freq) in read_bucket(ifd, mem_addr, len, id_size).iter() {
                    let tr = util::min((trel as f32) / 100.0, *ntr);
                    let tf = tr * (1.0 + freq as f32 / 1000.0);
                    _ids.push(Sid {
                        id: util::pqid2qid(pqid as u64, rem, nr_shards),
                        sc: tf * _idf,
                    });
                }
            }
            None => {
                // IDF ngram that occurs for the 1st time
                _idf = n.log(2.0);
            }
        }
        // normalization score
        _norm += ntr * _idf;
    }

    Ok(ShardIds {
        ids: _ids,
        norm: _norm,
    })
}

pub struct Qpick {
    path: String,
    config: config::Config,
    stopwords: HashSet<String>,
    terms_relevance: fst::Map,
    shards: Arc<Vec<Shard>>,
    shard_range: Range<u32>,
    id_size: usize,
}

pub struct Shard {
    map: fst::Map,
    shard: Mmap,
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
        let id_size = c.id_size;
        unsafe {
            // TODO set up globals, later should be available via self.config
            NR_SHARDS = Some(c.nr_shards);
            BUCKET_SIZE = Some(c.bucket_size);
            SHARD_SIZE = Some(c.shard_size);
            THREAD_POOL_SIZE = Some(c.thread_pool_size);
        }

        let shard_range = shard_range_opt.unwrap_or(0..c.nr_shards as u32);

        let stopwords_path = &format!("{}/{}", path, c.stopwords_file);
        let stopwords = match stopwords::load(stopwords_path) {
            Ok(stopwords) => stopwords,
            Err(_) => panic!(
                [
                    BYELL,
                    "No such file or directory: ",
                    ECOL,
                    BRED,
                    stopwords_path,
                    ECOL
                ].join("")
            ),
        };

        let terms_relevance_path = &format!("{}/{}", path, c.terms_relevance_file);
        let terms_relevance = match Map::from_path(terms_relevance_path) {
            Ok(terms_relevance) => terms_relevance,
            Err(_) => panic!(
                [
                    BYELL,
                    "No such file or directory: ",
                    ECOL,
                    BRED,
                    terms_relevance_path,
                    ECOL
                ].join("")
            ),
        };

        let mut shards = vec![];
        for i in shard_range.start..shard_range.end {
            let map_path = format!("{}/map.{}", path, i);

            // advice OS on random access to the map file and create Fst object from it
            let map_file = MmapReadOnly::open_path(&map_path).unwrap();
            unsafe {
                advise_ram(map_file.as_slice()).expect(&format!("Advisory failed for map {}", i))
            };
            let map = match Fst::from_mmap(map_file) {
                Ok(fst) => Map::from(fst),
                Err(_) => panic!("Failed to load index map: {}!", &map_path),
            };

            let shard = unsafe {
                Mmap::map(&OpenOptions::new()
                    .read(true)
                    .open(format!("{}/shard.{}", path, i))
                    .unwrap())
                    .unwrap()
            };

            advise_ram(&shard[..]).expect(&format!("Advisory failed for shard {}", i));

            shards.push(Shard {
                shard: shard,
                map: map,
            });
        }

        ThreadPoolBuilder::new()
            .num_threads(*get_thread_pool_size())
            .build()
            .unwrap();

        Qpick {
            config: c,
            path: path,
            stopwords: stopwords,
            terms_relevance: terms_relevance,
            shards: Arc::new(shards),
            shard_range: shard_range,
            id_size: id_size,
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

        let shard_ids: Vec<ShardIds> = shards_ngrams
            .par_iter()
            .map(|(shard_id, ngrams)| {
                get_query_ids(
                    ngrams,
                    &self.shards[*shard_id].map,
                    &self.shards[*shard_id].shard,
                    self.id_size,
                ).unwrap()
            })
            .collect();

        let mut norm: f32 = 0.0;
        let mut hdata: HashMap<u64, f32> = HashMap::new();
        for sh_id in shard_ids.iter() {
            for s in sh_id.ids.iter() {
                *hdata.entry(s.id).or_insert(0.0) += s.sc;
            }
            norm += sh_id.norm;
        }

        let mut vdata: Vec<Sid> = hdata
            .par_iter()
            .map(|(id, sc)| Sid {
                id: *id,
                sc: *sc / norm,
            })
            .collect();
        vdata.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Less).reverse());
        vdata.truncate(count.unwrap_or(100)); //TODO put into config

        Ok(vdata)
    }

    pub fn get_str(&self, query: &str, count: u32) -> String {
        let mut res: Vec<(u64, f32)> = self.get(query, 30 * count)
            .into_iter()
            .map(|s| (s.id, s.sc))
            .collect();
        res.truncate(count as usize);

        serde_json::to_string(&res).unwrap()
    }

    pub fn nget_str(&self, queries: &str, count: u32) -> String {
        let qvec: Vec<String> = serde_json::from_str(queries).unwrap();
        let mut res: Vec<(u64, f32)> = self.nget(&qvec, 30 * count)
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

        match self.get_ids(ngrams, Some(count as usize)) {
            Ok(ids) => ids,
            Err(err) => panic!("Failed to get ids with: {message}", message = err),
        }
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

        match self.get_ids(ngrams, Some(count as usize)) {
            Ok(ids) => ids,
            Err(err) => panic!("Failed to get ids with: {message}", message = err),
        }
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
            nr_shards, file_path, output_dir
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
