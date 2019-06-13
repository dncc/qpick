#[macro_use]
extern crate lazy_static;
extern crate byteorder;
extern crate fst;
extern crate libc;
#[macro_use]
extern crate serde_derive;
extern crate flate2;
extern crate fnv;
extern crate fs2;
extern crate memmap;
extern crate pbr;
extern crate rand;
extern crate serde_json;

use std::sync::Arc;
use std::fs::OpenOptions;
use std::ops::Range;
use std::path::PathBuf;
use std::cmp::{Ordering, PartialOrd};
use std::collections::HashMap;
use fnv::{FnvHashMap, FnvHashSet};

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
pub mod stringvec;

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

pub const LOW_SIM_THRESH: f32 = 0.099; // take only queries with similarity above

make_static_var_and_getter!(get_nr_shards, NR_SHARDS, usize);
make_static_var_and_getter!(get_shard_size, SHARD_SIZE, usize);
make_static_var_and_getter!(get_thread_pool_size, THREAD_POOL_SIZE, usize);

#[inline]
fn read_bucket(mmap: &memmap::Mmap, addr: usize, len: usize, id_size: usize) -> Vec<(u32, u8, u8)> {
    let buf = &mmap[addr..addr + len * id_size];
    (0..len)
        .map(|i| {
            let j = i * id_size;
            (
                LittleEndian::read_u32(&buf[j..j + 4]),
                buf[j + 4],
                buf[j + 5],
            )
        })
        .collect::<Vec<(u32, u8, u8)>>()
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
pub struct DistanceResult {
    pub query: String,
    pub dist: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchShardResult {
    pub id: u64, // query id unique globally
    pub sc: f32,
    pub query: Option<String>,
    pub sh_id: u8,   // shard id
    pub sh_qid: u32, // query id unique on a shard level
}

struct ShardResults {
    results: Vec<SearchShardResult>,
    norm: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub id: u64, // query id unique globally
    pub sc: f32,
    pub query: Option<String>,
}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &SearchResult) -> Option<Ordering> {
        if self.eq(&other) {
            self.id.partial_cmp(&other.id)
        } else {
            self.sc.partial_cmp(&other.sc)
        }
    }
}

impl PartialEq for SearchResult {
    fn eq(&self, other: &SearchResult) -> bool {
        self.sc == other.sc
    }
}

#[inline]
fn get_idfs(ngrams: &HashMap<String, f32>, map: &fst::Map) -> HashMap<String, (f32, f32)> {
    let mut idfs: HashMap<String, (f32, f32)> = HashMap::new();
    let n = *get_shard_size() as f32;
    for (ngram, ntr) in ngrams {
        // IDF score for the ngram
        let mut idf: f32;
        match get_addr_and_len(ngram, &map) {
            // returns physical memory address and length of the vector (not a number of bytes)
            Some((_addr, len)) => {
                // IDF for existing ngram
                idf = (n / len as f32).log(2.0);
            }
            None => {
                // IDF ngram that occurs for the 1st time
                idf = n.log(2.0);
            }
        }
        idfs.insert(ngram.to_string(), (*ntr, idf));
    }

    return idfs;
}

#[inline]
fn get_query_ids(
    ngrams: &HashMap<String, f32>,
    map: &fst::Map,
    ifd: &memmap::Mmap,
    id_size: usize,
) -> Result<ShardResults, Error> {
    let n = *get_shard_size() as f32;
    let nr_shards = *get_nr_shards();

    let mut norm: f32 = 0.0;
    let mut results: Vec<SearchShardResult> = vec![];

    for (ngram, ngram_tr) in ngrams {
        // IDF for the ngram that hasn't been seen before
        let mut idf: f32 = n.log(2.0);
        // returns physical memory address and length of the vector (not a number of bytes)
        if let Some((addr, len)) = get_addr_and_len(ngram, &map) {
            // IDF for existing ngram
            idf = (n / len as f32).log(2.0);
            let mem_addr = addr as usize * id_size;
            let ids_arr = read_bucket(ifd, mem_addr, len as usize, id_size);
            for &(shard_query_id, shard_id, trel) in ids_arr.iter() {
                let tr = util::min((trel as f32) / 100.0, *ngram_tr);
                results.push(SearchShardResult {
                    id: util::shard_id_2_query_id(shard_query_id as u64, shard_id, nr_shards),
                    sc: tr * idf,
                    query: None,
                    sh_id: shard_id,
                    sh_qid: shard_query_id,
                });
            }
        }
        // normalization score
        norm += ngram_tr * idf;
    }

    Ok(ShardResults {
        results: results,
        norm: norm,
    })
}

pub struct Qpick {
    path: String,
    config: config::Config,
    stopwords: FnvHashSet<String>,
    terms_relevance: fst::Map,
    shards: Arc<Vec<Shard>>,
    shard_range: Range<u32>,
    id_size: usize,
    i2q_loaded: bool,
}

pub struct Shard {
    map: fst::Map,
    shard: Mmap,
    i2q: Option<stringvec::StrVec>,
}

#[derive(Debug)]
pub struct DistResults {
    pub items_iter: std::vec::IntoIter<DistanceResult>,
}

impl DistResults {
    pub fn new(items_iter: std::vec::IntoIter<DistanceResult>) -> DistResults {
        DistResults {
            items_iter: items_iter,
        }
    }

    pub fn next(&mut self) -> Option<DistanceResult> {
        <std::vec::IntoIter<DistanceResult> as std::iter::Iterator>::next(&mut self.items_iter)
    }
}

#[derive(Debug)]
pub struct SearchResults {
    pub items_iter: std::vec::IntoIter<SearchResult>,
}

impl SearchResults {
    pub fn new(items_iter: std::vec::IntoIter<SearchResult>) -> SearchResults {
        SearchResults {
            items_iter: items_iter,
        }
    }

    pub fn next(&mut self) -> Option<SearchResult> {
        <std::vec::IntoIter<SearchResult> as std::iter::Iterator>::next(&mut self.items_iter)
    }
}

impl Qpick {
    fn new(path: String, shard_range_opt: Option<Range<u32>>) -> Qpick {
        let c = config::Config::init(path.clone());
        let id_size = c.id_size;
        unsafe {
            // TODO set up globals, later should be available via self.config
            NR_SHARDS = Some(c.nr_shards);
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

        let shard_indexes: Vec<u32> = (shard_range.start..shard_range.end).collect();
        let shards: Vec<(bool, Shard)> = shard_indexes
            .par_iter()
            .map(|i| {
                let map_path = format!("{}/map.{}", path, i);

                // advice OS on random access to the map file and create Fst object from it
                let map_file = MmapReadOnly::open_path(&map_path).unwrap();
                unsafe {
                    util::advise_ram(map_file.as_slice())
                        .expect(&format!("Advisory failed for map {}", i))
                };
                let map = match Fst::from_mmap(map_file) {
                    Ok(fst) => Map::from(fst),
                    Err(_) => panic!("Failed to load index map: {}!", &map_path),
                };

                let shard_name = format!("{}/shard.{}", path, i);
                let shard_file = OpenOptions::new().read(true).open(shard_name).unwrap();
                let shard = unsafe { Mmap::map(&shard_file).unwrap() };

                util::advise_ram(&shard[..]).expect(&format!("Advisory failed for shard {}", i));

                let i2q_path = PathBuf::from(&path).join(&format!("{}.{}", c.i2q_file, i));
                let i2q = if i2q_path.exists() {
                    Some(stringvec::StrVec::load(&i2q_path))
                } else {
                    None
                };

                (
                    !i2q.is_none(),
                    Shard {
                        shard: shard,
                        map: map,
                        i2q: i2q,
                    },
                )
            })
            .collect();

        let i2q_loaded = shards
            .iter()
            .fold(true, |b, (is_loaded, _)| b && *is_loaded);
        let shards = shards.into_iter().map(|(_, s)| s).collect();

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
            i2q_loaded: i2q_loaded,
        }
    }

    pub fn i2q_is_loaded(&self) -> bool {
        self.i2q_loaded
    }

    pub fn from_path(path: String) -> Self {
        Qpick::new(path, None)
    }

    pub fn from_path_with_shard_range(path: String, shard_range: Range<u32>) -> Self {
        Qpick::new(path, Some(shard_range))
    }

    #[inline]
    fn shard_ngrams(
        &self,
        ngrams: &FnvHashMap<String, f32>,
    ) -> HashMap<usize, HashMap<String, f32>> {
        let mut shards_ngrams: HashMap<usize, HashMap<String, f32>> = HashMap::new();
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

        return shards_ngrams;
    }

    fn get_ids(
        &self,
        ngrams: &FnvHashMap<String, f32>,
        count: Option<usize>,
    ) -> Result<Vec<SearchResult>, Error> {
        let shards_ngrams = self.shard_ngrams(ngrams);
        let shard_results: Vec<ShardResults> = shards_ngrams
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
        // query_id -> (shard_query_id, shard_id)
        let mut ids_map: HashMap<u64, (u32, u8)> = HashMap::new();
        // query_id -> score
        let mut res_data: HashMap<u64, f32> = HashMap::new();

        for sh_res in shard_results.iter() {
            for r in sh_res.results.iter() {
                ids_map.entry(r.id).or_insert((r.sh_qid, r.sh_id));
                *res_data.entry(r.id).or_insert(0.0) += r.sc;
            }
            norm += sh_res.norm;
        }

        let mut res_data: Vec<(u64, f32)> = res_data
            .into_iter()
            .filter(|(_, sc)| *sc / norm > LOW_SIM_THRESH)
            .collect::<Vec<(u64, f32)>>();
        res_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Less).reverse());

        Ok(res_data
            .into_iter()
            .take(count.unwrap_or(100))
            .map(|(id, sc)| {
                let (sh_qid, sh_id) = ids_map.get(&id).unwrap();
                SearchResult {
                    id: id,
                    sc: util::max(1.0 - sc / norm, 0.0),
                    query: if let Some(ref i2q) = &self.shards[*sh_id as usize].i2q {
                        Some(i2q[*sh_qid as usize].to_string())
                    } else {
                        None
                    },
                }
            })
            .collect())
    }

    pub fn get_distances(&self, query: &str, candidates: &Vec<String>) -> Vec<DistanceResult> {
        if query == "" {
            return vec![];
        }

        let mut dist_results: Vec<DistanceResult> = vec![];

        let ref ngrams: FnvHashMap<String, f32> =
            ngrams::parse(&query, &self.stopwords, &self.terms_relevance);

        let mut dist_norm: f32 = 0.0;
        let mut ngram_tr_idfs: HashMap<String, (f32, f32)> = HashMap::new();
        self.shard_ngrams(ngrams)
            .into_iter()
            .map(|(shard_id, ngrams)| {
                for (ngram, (tr, idf)) in get_idfs(&ngrams, &self.shards[shard_id].map) {
                    dist_norm += tr * idf;
                    ngram_tr_idfs.insert(ngram, (tr, idf));
                }
            })
            .for_each(drop);

        for cand_query in candidates.into_iter() {
            let ref cand_ngrams: FnvHashMap<String, f32> =
                ngrams::parse(&cand_query, &self.stopwords, &self.terms_relevance);

            let mut dist_sim: f32 = 0.0;
            for (cngram, ctr) in cand_ngrams {
                if ngram_tr_idfs.contains_key(cngram) {
                    let (tr, idf) = ngram_tr_idfs.get(cngram).unwrap();
                    dist_sim += util::min(ctr, tr) * idf;
                }
            }
            let dist = util::max(1.0 - dist_sim / dist_norm, 0.0);
            dist_results.push(DistanceResult {
                query: cand_query.to_string(),
                dist: dist,
            });
        }

        return dist_results;
    }

    pub fn get(&self, query: &str, count: u32) -> Vec<SearchResult> {
        if query == "" || count == 0 {
            return vec![];
        }

        let ref ngrams: FnvHashMap<String, f32> =
            ngrams::parse(&query, &self.stopwords, &self.terms_relevance);

        match self.get_ids(ngrams, Some(count as usize)) {
            Ok(ids) => ids,
            Err(err) => panic!("Failed to get ids with: {message}", message = err),
        }
    }

    pub fn merge(&self) -> Result<(), Error> {
        println!("Merging index maps from: {:?}", &self.path);
        merge::merge(&self.path, self.config.nr_shards as usize)
    }

    pub fn get_search_results(&self, query: &str, count: u32) -> SearchResults {
        SearchResults::new(self.get(query, count).into_iter())
    }

    pub fn get_dist_results(&self, query: &str, candidates: &Vec<String>) -> DistResults {
        DistResults::new(self.get_distances(query, candidates).into_iter())
    }
}

#[allow(dead_code)]
fn main() {}
