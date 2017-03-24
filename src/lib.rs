extern crate fst;
extern crate memmap;
extern crate byteorder;

use std::thread;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use std::fs;
use std::io;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::SeekFrom;
use memmap::{Mmap, Protection};
use std::cmp::{Ordering, PartialOrd};
use std::mem;
use std::cmp;
use std::collections::HashMap;
use std::collections::HashSet;

use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};
use fst::{IntoStreamer, Streamer, Map, MapBuilder};

use std::f64::INFINITY;
use std::f64::NEG_INFINITY;

use fst::Error;

pub struct Qi {
    len: usize,
}

// qid % NR_SHARDS = shard_id,  where NR_SHARDS is a power of 2
// sqid - sharded query id,
// sqid = qid >> math.log(NR_SHARDS, 2)
// qid = sqid << math.log(NR_SHARDS, 2) + shard_id
impl Qi {
    fn new(len: usize) -> Self {
        Qi {
            len: len,
        }
    }
}

/*
    Elegant pairing function http://szudzik.com/ElegantPairing.pdf
    TODO implement with bignum, otherwise might overflow!
*/
#[inline]
fn elegant_pair(x: u64, y: u64) -> u64 {
    let z: u64 = match x >= y {
        true => x * x + x + y,
        false => y * y + x,
    };
    if elegant_pair_inv(z) != (x, y) {
        panic!("Numbers {} and {} cannot be paired!", x, y);
    };

    z
}

/*
    Inverse elegant pairing function http://szudzik.com/ElegantPairing.pdf
    TODO implement with bignum or f128, otherwise might overflow!
*/
#[inline]
fn elegant_pair_inv(z: u64) -> (u64, u64) {
    let q = z as f64;
    let w = (q.sqrt()).floor() as u64;
    let t = (w * w) as u64;
    if (z - t) >= w {
        (w, z - t - w)
    } else {
        (z - t, w)
    }
}

pub fn load_stopwords() -> Result<HashSet<String>, std::io::Error> {
    let mut stopwords = HashSet::new();

    let f = try!(File::open("stopwords.txt"));
    let mut file = BufReader::new(&f);

    for line in file.lines() {
        let sw = line.unwrap();
        stopwords.insert(sw);

    }

    Ok(stopwords)
}

fn write_bucket(mut file: &File, addr: u64, data: &HashSet<u32>) {
    file.seek(SeekFrom::Start(addr)).unwrap();
    let mut w = Vec::with_capacity(data.len()*4);
    for n in data.iter() {w.write_u32::<LittleEndian>(*n).unwrap()};
    file.write_all(w.as_slice());
}

fn read_bucket(mut file: &File, addr: u64, len: u64) -> Vec<u32> {
    file.seek(SeekFrom::Start(addr)).unwrap();
    let mut handle = file.take(NR_IDS as u64 * 4);
    let mut buf=[0u8; NR_IDS*4];
    handle.read(&mut buf);

    let vlen = len as usize;
    let mut vector = Vec::<u32>::with_capacity(vlen);
    for i in 0..vlen {
        let j = i*4;
        vector.push(LittleEndian::read_u32(&buf[j..j+4]));
    }
    vector
}

fn parse_ngrams(query: &str, length: usize, stopwords: &HashSet<String>) -> Vec<String> {

    let mut ngrams = vec![];
    let mut wvec = query.split(" ").collect::<Vec<&str>>();

    if wvec.len() == 1 {
        return ngrams;
    }

    wvec.reverse();

    // concatenate terms with stopwords if any
    let mut termv = vec![];
    let mut terms = vec![];
    let mut has_stopword = false;
    while wvec.len() > 0 {
        let w = wvec.pop().unwrap();
        termv.push(w);
        if stopwords.contains(w) || w.parse::<u32>().is_ok() {
            has_stopword = true;

        } else if termv.len() >= length {
            if has_stopword {
                let s: String = termv.into_iter().collect::<Vec<_>>().connect(" ");
                terms.push(s);
            } else {
                for w in termv.into_iter() {
                    terms.push(w.to_string());
                }
            }
            has_stopword = false;
            termv = vec![];
        }
    }

    // combine the new terms into ngrams a b c d -> ab, ac, bc, bd, cd
    if terms.len() > 0 {
        for i in 0..terms.len()-1 {
            ngrams.push(format!("{} {}", terms[i], terms[i+1]));
            if i < terms.len()-2 {
                ngrams.push(format!("{} {}", terms[i], terms[i+2]));
            }
        }
    }

    return ngrams
}


// build inverted query index, ngram_i -> [q1, q2, ... qi]
fn build_inverted_index(iid: u32, input_file: &str) -> Result<(), Error>{

    let ref stopwords = match load_stopwords() {
        Ok(stopwords) => stopwords,
        Err(_) => panic!("Failed to load stop-words!")
    };

    let mut qid = 0;
    let mut invert: HashMap<String, HashSet<u32>> = HashMap::new();

    let f = try!(File::open(input_file));
    let mut reader = BufReader::with_capacity(5 * 1024 * 1024, &f);
    for line in reader.lines() {

        let line = match line {
            Ok(line) => line,
            Err(e) => {
                println!("Read line error: {:?}", e);
                continue
            }
        };

        let query = line.trim();

        for ngram in parse_ngrams(query, 2, stopwords).into_iter() {
            let set = invert.entry(ngram).or_insert(HashSet::new());
            set.insert(qid);
        }

        qid += 1;
        if qid as u32 % 1_000_000 == 0 {
            println!("Reading {:.1}M", qid / 1_000_000);
        }
    }

    // sort inverted query index by keys (ngrams) and store it to fst file
    let mut vinvert: Vec<(String, HashSet<u32>)> = invert.into_iter()
        .map(|(ngram, ids)| (ngram, ids))
        .collect();

    println!("Sorting ngrams...");
    vinvert.sort_by(|a, b| {a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)});

    // create index dir if it doesn't exist
    try!(fs::create_dir_all("./index"));

    let index_map_file_name = "./index/map.".to_string() + &iid.to_string();
    // remove previous index first if exists
    std::fs::remove_file(&index_map_file_name);
    let mut wtr = io::BufWriter::new(try!(File::create(&index_map_file_name)));

    println!("Map {} init...", index_map_file_name);
    // Create a builder that can be used to insert new key-value pairs.
    let mut build = try!(MapBuilder::new(wtr));

    let index_file_name = "./index/shard.".to_string() + &iid.to_string();
    // remove previous index first if exists
    std::fs::remove_file(&index_file_name);
    let index_file = &OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&index_file_name).unwrap();

    let mut cursor: u64 = 0;
    for (ngram, ids) in vinvert.into_iter() {
        let ids_len: u64 = ids.len() as u64;
        if ids_len > NR_IDS as u64 {
            continue
        }
        write_bucket(index_file, cursor*4, &ids);
        build.insert(ngram, elegant_pair(cursor, ids_len));
        cursor += ids_len;
    }

    // Finish construction of the map and flush its contents to disk.
    try!(build.finish());

    println!("file {} created", index_file_name);
    Ok(())
}

// reading part
#[inline]
fn get_addr_and_len(key: String, map: &fst::Map) -> Option<(u64, u64)> {
    match map.get(key) {
        Some(val) => {
            // println!("key {}, val {:?}", key, val);
            return Some(elegant_pair_inv(val))
        },
        None => return None,
    }
}

pub fn get_ann_ids(query: String) -> Result<Vec<Vec<(String, f32)>>, Error> {

    let mut ann_ids: Arc<Mutex<Vec<Vec<(String, f32)>>>> = Arc::new(Mutex::new(vec![vec![]; NR_SHARDS]));
    let (sender, receiver) = mpsc::channel();

    for i in 0..NR_SHARDS {
        let q = query.clone();
        let sender = sender.clone();
        let ann_ids = ann_ids.clone();

        thread::spawn(move || {
            let stopwords = match load_stopwords() {
                Ok(stopwords) => stopwords,
                Err(_) => panic!("Failed to load stop-words!")
            };
            let map_name = "./index/map.".to_string() + &i.to_string();
            let mut map = match Map::from_path(&map_name) {
                Ok(map) => map,
                Err(_) => panic!("Failed to load index map!")
            };
            let shard_name = "./index/shard.".to_string() + &i.to_string();
            let mut shard = OpenOptions::new().read(true).open(&shard_name).unwrap();

            let sh_ann_ids = match get_shard_ann_ids(&q, &stopwords, &map, &shard) {
                Ok(ids) => ids,
                Err(_) => {
                    println!("Failed to retrive ids from shard: {}", i);
                    vec![]
                }
            };

            // obtaining lock might fail! handle it!
            let mut ann_ids = ann_ids.lock().unwrap();
            ann_ids[i] = sh_ann_ids;
            sender.send(()).unwrap();
        });
    }

    for _ in 0..NR_SHARDS {
        receiver.recv().unwrap();
    }

    // deref MutexGuard returned from ann_ids.lock().unwrap()
    let data = (*ann_ids.lock().unwrap()).clone();
    Ok(data)
}

fn get_shard_ann_ids(query: &str,
                   stopwords: &HashSet<String>,
                   map: &fst::Map,
                   mut ifd: &File) -> Result<Vec<(String, f32)>, Error>{

    let mut ann_ids = HashMap::new();

    let mut ngrams: Vec<String> = vec![];
    ngrams.extend(parse_ngrams(query, 2, stopwords));

    for ngram in ngrams.into_iter() {
        match get_addr_and_len(ngram, &map) {
            Some((addr, len)) => {
                for id in read_bucket(&ifd, addr*4, len).iter() {
                    let sc = ann_ids.entry(id.to_string()).or_insert(0.0);
                    *sc += (N/len as f32).log(2.0);
                }
            },
            None => (),
        }
    }

    // Ok(ann_ids)
    // let mut v: Vec<_> = ann_ids.iter().collect();
    let mut v: Vec<(String, f32)> = ann_ids.iter().map(|(id, sc)| (id.to_string(), *sc)).collect::<Vec<_>>();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Less).reverse());
    v.truncate(100);
    Ok(v)
}

const N: f32 = 450_000_000.0; // number of queries per shard
const START_SHARD: usize = 0;
const NR_SHARDS:usize = 4;
const NR_IDS: usize = 30000;  // number of query ids per index bucket

fn main() {
    let (sender, receiver) = mpsc::channel();

    for i in START_SHARD..NR_SHARDS {
        let sender = sender.clone();
        // let input_file_name = "./queries_sorted.".to_string() + &i.to_string() + &".txt".to_string();
        let input_file_name = "./parts/p".to_string() + &i.to_string();

        thread::spawn(move || {
            build_inverted_index(i as u32, &input_file_name);
            sender.send(()).unwrap();
        });
    }

    for _ in START_SHARD..NR_SHARDS {
        receiver.recv().unwrap();
    }

    println!("Finished building {} shards.", NR_SHARDS);
}
