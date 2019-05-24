#[macro_use]
extern crate lazy_static;
extern crate byteorder;
extern crate fst;
extern crate libc;
#[macro_use]
extern crate serde_derive;
extern crate flate2;
extern crate fs2;
extern crate memmap;
extern crate pbr;
extern crate rand;
extern crate serde_json;

use std::io;
use std::sync::Arc;
use std::fs::OpenOptions;
use std::ops::Range;
use std::path::PathBuf;
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
pub mod stringvec;

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

pub const LOW_SIM_THRESH: f32 = 0.099; // take only queries with similarity above

make_static_var_and_getter!(get_bucket_size, BUCKET_SIZE, usize);
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

// --- simd
use std::mem;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[target_feature(enable = "avx")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
#[allow(unsafe_code)]
unsafe fn get_query_ids_with_simd(
    ngrams: &HashMap<String, f32>,
    map: &fst::Map,
    ifd: &memmap::Mmap,
    id_size: usize,
) -> Result<ShardResults, Error> {
    // normalization constant for tf-idf score
    let mut norm: f32 = 0.0;
    let n = *get_shard_size() as f32;
    let nr_shards = *get_nr_shards();
    let bucket_size = *get_bucket_size();

    // max capacity for results vector
    let mut results: Vec<SearchShardResult> = Vec::with_capacity(ngrams.len() * bucket_size);
    results.reserve_exact(ngrams.len() * bucket_size);

    for (ngram, ngram_tr) in ngrams {
        // IDF score for the ngram
        let mut idf: f32 = n.log(2.0);
        if let Some((addr, len)) = get_addr_and_len(ngram, &map) {
            // IDF for existing ngram
            idf = (n / len as f32).log(2.0);

            // -- start simd tf-idf calculation for each id
            let _idf8 = _mm256_set1_ps(idf);

            let mem_addr = addr as usize * id_size;
            let mut ids_arr: &[(u32, u8, u8)] = &read_bucket(ifd, mem_addr, len as usize, id_size);
            // println!("ids_arr: {:?}", ids_arr);

            while ids_arr.len() >= 8 {
                let (
                    (shard_query_id0, shard_id0, trel0),
                    (shard_query_id1, shard_id1, trel1),
                    (shard_query_id2, shard_id2, trel2),
                    (shard_query_id3, shard_id3, trel3),
                    (shard_query_id4, shard_id4, trel4),
                    (shard_query_id5, shard_id5, trel5),
                    (shard_query_id6, shard_id6, trel6),
                    (shard_query_id7, shard_id7, trel7),
                ) = (
                    ids_arr[0],
                    ids_arr[1],
                    ids_arr[2],
                    ids_arr[3],
                    ids_arr[4],
                    ids_arr[5],
                    ids_arr[6],
                    ids_arr[7],
                );

                let _trel = _mm256_set_ps(
                    util::min(trel0 as f32 / 100.0, *ngram_tr) as f32,
                    util::min(trel1 as f32 / 100.0, *ngram_tr) as f32,
                    util::min(trel2 as f32 / 100.0, *ngram_tr) as f32,
                    util::min(trel3 as f32 / 100.0, *ngram_tr) as f32,
                    util::min(trel4 as f32 / 100.0, *ngram_tr) as f32,
                    util::min(trel5 as f32 / 100.0, *ngram_tr) as f32,
                    util::min(trel6 as f32 / 100.0, *ngram_tr) as f32,
                    util::min(trel7 as f32 / 100.0, *ngram_tr) as f32,
                );

                // println!("trel {:?}", _trel);

                let tf_idf: (f32, f32, f32, f32, f32, f32, f32, f32) =
                    mem::transmute(_mm256_mul_ps(_trel, _idf8));

                // println!("tf_idf {:?}", tf_idf);

                let result_slice = &[
                    SearchShardResult {
                        id: util::shard_id_2_query_id(shard_query_id0 as u64, shard_id0, nr_shards),
                        sc: tf_idf.7,
                        query: None,
                        sh_id: shard_id0,
                        sh_qid: shard_query_id0,
                    },
                    SearchShardResult {
                        id: util::shard_id_2_query_id(shard_query_id1 as u64, shard_id1, nr_shards),
                        sc: tf_idf.6,
                        query: None,
                        sh_id: shard_id1,
                        sh_qid: shard_query_id1,
                    },
                    SearchShardResult {
                        id: util::shard_id_2_query_id(shard_query_id2 as u64, shard_id2, nr_shards),
                        sc: tf_idf.5,
                        query: None,
                        sh_id: shard_id2,
                        sh_qid: shard_query_id2,
                    },
                    SearchShardResult {
                        id: util::shard_id_2_query_id(shard_query_id3 as u64, shard_id3, nr_shards),
                        sc: tf_idf.4,
                        query: None,
                        sh_id: shard_id3,
                        sh_qid: shard_query_id3,
                    },
                    SearchShardResult {
                        id: util::shard_id_2_query_id(shard_query_id4 as u64, shard_id4, nr_shards),
                        sc: tf_idf.3,
                        query: None,
                        sh_id: shard_id4,
                        sh_qid: shard_query_id4,
                    },
                    SearchShardResult {
                        id: util::shard_id_2_query_id(shard_query_id5 as u64, shard_id5, nr_shards),
                        sc: tf_idf.2,
                        query: None,
                        sh_id: shard_id5,
                        sh_qid: shard_query_id5,
                    },
                    SearchShardResult {
                        id: util::shard_id_2_query_id(shard_query_id6 as u64, shard_id6, nr_shards),
                        sc: tf_idf.1,
                        query: None,
                        sh_id: shard_id6,
                        sh_qid: shard_query_id6,
                    },
                    SearchShardResult {
                        id: util::shard_id_2_query_id(shard_query_id7 as u64, shard_id7, nr_shards),
                        sc: tf_idf.0,
                        query: None,
                        sh_id: shard_id7,
                        sh_qid: shard_query_id7,
                    },
                ];
                results.extend_from_slice(result_slice);

                ids_arr = &ids_arr[8..];
            }

            // compute tf-idf for remining results if any
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

        norm += *ngram_tr * idf;
    }

    // -- end simd
    Ok(ShardResults {
        results: results,
        norm: norm,
    })
}

// --- simd end

#[inline]
fn get_query_ids_wo_simd(
    ngrams: &HashMap<String, f32>,
    map: &fst::Map,
    ifd: &memmap::Mmap,
    id_size: usize,
) -> Result<ShardResults, Error> {
    let n = *get_shard_size() as f32;
    let nr_shards = *get_nr_shards();
    let bucket_size = *get_bucket_size();

    let mut norm: f32 = 0.0;
    let mut results: Vec<SearchShardResult> = vec![];

    for (ngram, ngram_tr) in ngrams {
        // IDF score for the ngram
        let mut idf: f32;
        match get_addr_and_len(ngram, &map) {
            // returns physical memory address and length of the vector (not a number of bytes)
            Some((addr, len)) => {
                // IDF for existing ngram
                idf = (n / len as f32).log(2.0);
                let mem_addr = addr as usize * id_size;
                let len = util::min(len as usize, bucket_size);
                let ids_arr = read_bucket(ifd, mem_addr, len, id_size);
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
            None => {
                // IDF ngram that occurs for the 1st time
                idf = n.log(2.0);
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

#[inline]
fn get_query_ids(
    ngrams: &HashMap<String, f32>,
    map: &fst::Map,
    ifd: &memmap::Mmap,
    id_size: usize,
) -> Result<ShardResults, Error> {
    if cfg!(target_feature = "avx") {
        unsafe { get_query_ids_with_simd(ngrams, map, ifd, id_size) }
    } else {
        get_query_ids_wo_simd(ngrams, map, ifd, id_size)
    }
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

            let shard_name = format!("{}/shard.{}", path, i);
            let shard_file = OpenOptions::new().read(true).open(shard_name).unwrap();
            let shard = unsafe { Mmap::map(&shard_file).unwrap() };

            advise_ram(&shard[..]).expect(&format!("Advisory failed for shard {}", i));

            let i2q_path = PathBuf::from(&path).join(&format!("{}.{}", c.i2q_file, i));
            let i2q = if i2q_path.exists() {
                Some(stringvec::StrVec::load(&i2q_path))
            } else {
                None
            };

            shards.push(Shard {
                shard: shard,
                map: map,
                i2q: i2q,
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

    #[inline]
    fn shard_ngrams(&self, ngrams: &HashMap<String, f32>) -> HashMap<usize, HashMap<String, f32>> {
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
        ngrams: &HashMap<String, f32>,
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

        let mut res_data: Vec<SearchResult> = res_data
            .par_iter()
            .filter(|(_, sc)| *sc / norm > LOW_SIM_THRESH)
            .map(|(id, sc)| {
                let (sh_qid, sh_id) = ids_map.get(id).unwrap();
                SearchResult {
                    id: *id,
                    sc: util::max(1.0 - *sc / norm, 0.0),
                    query: if let Some(ref i2q) = &self.shards[*sh_id as usize].i2q {
                        Some(i2q[*sh_qid as usize].to_string())
                    } else {
                        None
                    },
                }
            })
            .collect();

        res_data.sort_by(|a, b| a.partial_cmp(&b).unwrap_or(Ordering::Less));
        res_data.truncate(count.unwrap_or(100)); //TODO put into config

        Ok(res_data)
    }

    pub fn get_distances(&self, query: &str, candidates: &Vec<String>) -> Vec<DistanceResult> {
        if query == "" {
            return vec![];
        }

        let mut dist_results: Vec<DistanceResult> = vec![];

        let ref ngrams: HashMap<String, f32> =
            ngrams::parse(&query, &self.stopwords, &self.terms_relevance, QueryType::Q);

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
            let ref cand_ngrams: HashMap<String, f32> = ngrams::parse(
                &cand_query,
                &self.stopwords,
                &self.terms_relevance,
                QueryType::Q,
            );

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

        let ref ngrams: HashMap<String, f32> =
            ngrams::parse(&query, &self.stopwords, &self.terms_relevance, QueryType::Q);

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
// TODO clean nget_str etc... fix python bindings to show search results,  rename get to search
