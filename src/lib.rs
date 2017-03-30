extern crate fst;
extern crate byteorder;
extern crate serde_json;

use std::thread;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use std::fs;
use std::io::BufWriter;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::io::BufReader;
use std::io::SeekFrom;
use std::cmp::{Ordering, PartialOrd};
use std::collections::HashMap;
use std::collections::HashSet;

use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};
use fst::{Map, MapBuilder};

use fst::Error;

pub mod config;
pub mod stopwords;

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

#[inline]
fn qid2pid(qid: u64) -> u64 {
    qid % *get_nr_shards() as u64
}

#[inline]
fn qid2pqid(qid: u64) -> u64 {
    qid >> (*get_nr_shards() as f32).log(2.0) as u64
}

#[inline]
fn pqid2qid(pqid: u64, pid: u64) -> u64 {
    (pqid << (*get_nr_shards() as f32).log(2.0) as u64) + pid
}

fn write_bucket(mut file: &File, addr: u64, data: &Vec<(u32, u8)>) {
    file.seek(SeekFrom::Start(addr)).unwrap();
    let id_size = get_id_size();
    let mut w = Vec::with_capacity(data.len()*id_size);
    for n in data.iter() {
        w.write_u32::<LittleEndian>(n.0).unwrap();
        w.write_u8(n.1).unwrap();
    };
    file.write_all(w.as_slice());
}

fn read_bucket(mut file: &File, addr: u64, len: u64) -> Vec<(u32, u8)> {
    let id_size = get_id_size();
    let bk_size = get_bucket_size();
    file.seek(SeekFrom::Start(addr)).unwrap();
    let mut handle = file.take((bk_size * id_size) as u64);
    let mut buf=vec![0u8; bk_size*id_size];
    handle.read(&mut buf);

    let vlen = len as usize;
    let mut vector = Vec::<(u32, u8)>::with_capacity(vlen);
    for i in 0..vlen {
        let j = i*id_size;
        vector.push((LittleEndian::read_u32(&buf[j..j+4]), buf[j+4]));
    }

    vector
}

fn get_terms_relevance(terms: &Vec<&str>, tr_map: &fst::Map) -> HashMap<String, f32> {

    let mut missing: HashSet<String> = HashSet::new();
    let mut terms_rel: HashMap<String, f32> = HashMap::new();

    let tset = terms.clone().into_iter().collect::<HashSet<&str>>();
    for t in &tset {
        match tr_map.get(t) {
            Some(tr) => {
                terms_rel.insert(t.to_string(), tr as f32);
            },
            None => {
                missing.insert(t.to_string());
            },
        };
    }

    // avg and sum
    let mut sum: f32 = terms_rel.values().fold(0.0, |a, b| a + *b);
    let mut avg: f32 = sum/terms_rel.len() as f32;
    // terms may repeat in the query or/and sum might be zero
    if sum > 0.0 {
        sum = terms.iter().fold(0.0, |a, t| a + terms_rel.get(t.clone()).unwrap_or(&avg));
        avg = sum/terms.len() as f32;
    } else {
        avg = 1.0;
        sum = terms.len() as f32;
    }

    // set an average term relevance to the missing terms and normalize
    for t in tset.iter() {
        let rel = terms_rel.entry(t.to_string()).or_insert(avg);
        *rel /= sum;
    }

    terms_rel
}

fn parse_ngrams(query: &str, length: usize, stopwords: &HashSet<String>, tr_map: &fst::Map)
                -> HashMap<String, f32> {

    let mut ngrams: HashMap<String, f32> = HashMap::new();

    let mut wvec = query.split(" ").collect::<Vec<&str>>();
    let terms_rel = get_terms_relevance(&wvec, tr_map);

    wvec.reverse();

    // concatenate terms with stopwords if any
    let mut termv = vec![];
    let mut terms: Vec<(String, f32)> = vec![]; // [('the best', 0.2), ('search', 0.3)]
    let mut has_stopword = false;
    while wvec.len() > 0 {
        let w = wvec.pop().unwrap();
        termv.push(w);
        if stopwords.contains(w) {
            has_stopword = true;

        } else if termv.len() >= length {
            if has_stopword {
                let r = termv.iter().fold(0.0, |a, t| a + terms_rel.get(t.clone()).unwrap());
                let s: String = termv.into_iter().collect::<Vec<_>>().join(" ");
                terms.push((s, r));
            } else {
                for t in termv.into_iter() {
                    terms.push((t.to_string(), *terms_rel.get(t).unwrap()));
                }
            }
            has_stopword = false;
            termv = vec![];
        }
    }

    // combine the new terms into ngrams a b c d -> ab, ac, bc, bd, cd
    if terms.len() > 0 {
        for i in 0..terms.len()-1 {
            ngrams.insert(format!("{}", terms[i].0), terms[i].1);
            ngrams.insert(format!("{} {}", terms[i].0, terms[i+1].0), terms[i].1+terms[i+1].1);
            if i < terms.len()-2 {
                ngrams.insert(format!("{} {}", terms[i].0, terms[i+2].0), terms[i].1+terms[i+2].1);
            }
        }
        ngrams.insert(format!("{}", terms[terms.len()-1].0), terms[terms.len()-1].1);
    }

    return ngrams
}

// build inverted query index, ngram_i -> [q1, q2, ... qi]
fn build_inverted_index(iid: u32, input_file: &str, tr_map: &fst::Map) -> Result<(), Error>{

    let ref stopwords = match stopwords::load() {
        Ok(stopwords) => stopwords,
        Err(_) => panic!("Failed to load stop-words!")
    };

    let mut qcount = 0;
    let mut invert: HashMap<String, HashMap<u32, u8>> = HashMap::new();

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

        let mut split = line.trim().split("\t");

        let qid = match split.next() {
            Some(qid) => match qid.parse::<u64>() {
                Ok(n) => n,
                Err(err) => {
                    println!("Shard {:?} - failed to parse query id {:?}: {:?}", iid, qid, err);
                    continue
                }
            },
            None => {
                println!("Shard {:?} - No query id found", iid);
                continue
            }
        };

        let query = match split.next() {
                Some(q) => q.trim(),
                None => {
                        println!("Shard {:?} - No query found", iid);
                        continue
                }
        };

        for (ngram, sc) in &parse_ngrams(query, 2, stopwords, tr_map) {
            let imap = invert.entry(ngram.to_string()).or_insert(HashMap::new());
            let pqid = qid2pqid(qid);
            imap.insert(pqid as u32, (sc * 100.0).round() as u8);
        }

        qcount += 1;
        if qcount as u32 % 1_000_000 == 0 {
            println!("Reading {:.1}M", qcount / 1_000_000);
        }
    }

    // sort inverted query index by keys (ngrams) and store it to fst file
    let mut vinvert: Vec<(String, HashMap<u32, u8>)> = invert.into_iter()
        .map(|(ngram, ids)| (ngram, ids))
        .collect();

    println!("Sorting ngrams...");
    vinvert.sort_by(|a, b| {a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)});

    // create index dir if it doesn't exist
    try!(fs::create_dir_all("./index"));

    let index_map_file_name = "./index/map.".to_string() + &iid.to_string();
    // remove previous index first if exists
    std::fs::remove_file(&index_map_file_name);
    let mut wtr = BufWriter::new(try!(File::create(&index_map_file_name)));

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
    let id_size = *get_id_size();
    let bk_size = *get_bucket_size();
    for (ngram, ids_tr) in vinvert.into_iter() {
        let ids_len: u64 = ids_tr.len() as u64;
        if ids_len > bk_size as u64 {
            continue
        }
        let ids = ids_tr.iter().map(|(k,v)| (*k, *v)).collect::<Vec<(u32, u8)>>();
        write_bucket(index_file, cursor*id_size as u64, &ids);
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
fn get_addr_and_len(key: &str, map: &fst::Map) -> Option<(u64, u64)> {
    match map.get(key) {
        Some(val) => {
            return Some(elegant_pair_inv(val))
        },
        None => return None,
    }
}

fn get_shard_ids(pid: usize,
                 ngrams: &HashMap<String, f32>,
                 map: &fst::Map,
                 ifd: &File) -> Result<Vec<(u64, f32)>, Error>{

    let mut _ids = HashMap::new();
    let id_size = *get_id_size();
    let n = *get_shard_size() as f32;

    for (ngram, ntr) in ngrams {
        match get_addr_and_len(ngram, &map) {
            Some((addr, len)) => {
                for id_tr in read_bucket(&ifd, addr*id_size as u64, len).iter() {
                    let qid = pqid2qid(id_tr.0 as u64, pid as u64);
                    let sc = _ids.entry(qid).or_insert(0.0);
                    *sc += (id_tr.1 as f32)/100.0 * (n/len as f32).log(2.0);
                }
            },
            None => (),
        }
    }

    // Ok(_ids)
    // let mut v: Vec<_> = _ids.iter().collect();
    let mut v: Vec<(u64, f32)> = _ids.iter().map(|(id, sc)| (*id, *sc)).collect::<Vec<_>>();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Less).reverse());
    v.truncate(100); //TODO put into config
    Ok(v)
}

pub struct Qi {
    path: String,
    config: config::Config,
    stopwords: HashSet<String>,
    terms_relevance: fst::Map,
    shards: Arc<Vec<Shard>>,
}

pub struct Shard {
    id: u32,
    map: fst::Map,
    shard: File,
}

impl Qi {
    fn new(path: String) -> Qi {

        let c = config::Config::init();

        unsafe {
            // TODO set up globals, later should be available if possible via self.config
            NR_SHARDS = Some(c.nr_shards);
            ID_SIZE = Some(c.id_size);
            BUCKET_SIZE = Some(c.bucket_size);
            SHARD_SIZE = Some(c.shard_size);
        }

        let stopwords = match stopwords::load() {
            Ok(stopwords) => stopwords,
            Err(_) => panic!("Failed to load stop-words!")
        };

        // TODO put the name in config
        let terms_relevance = match Map::from_path(&c.terms_relevance_path) {
            Ok(terms_relevance) => terms_relevance,
            Err(_) => panic!("Failed to load terms rel. map: {}!", &c.terms_relevance_path)
        };

        let mut shards = vec![];
        for i in c.first_shard..c.last_shard {
            // TODO initialize a thread/worker pool with maps and shards at Qi init
            let map_name = format!("{}/map.{}", path, i);
            let map = match Map::from_path(&map_name) {
                Ok(map) => map,
                Err(_) => panic!("Failed to load index map: {}!", &map_name)
            };

            let shard = OpenOptions::new().read(true).open(format!("{}/shard.{}", path, i)).unwrap();
            shards.push(Shard{id: i, shard: shard, map: map});
        };

        Qi {
            config: c,
            path: path,
            stopwords: stopwords,
            terms_relevance: terms_relevance,
            shards: Arc::new(shards),
        }
    }

    pub fn from_path(path: String) -> Self {
        Qi::new(path)
    }

    fn get_ids(&self, query: String) -> Result<Vec<Vec<(u64, f32)>>, Error> {

        let n_shards = (self.config.last_shard - self.config.first_shard) as usize;
        let mut _ids: Arc<Mutex<Vec<Vec<(u64, f32)>>>> = Arc::new(Mutex::new(vec![vec![]; n_shards]));
        let (sender, receiver) = mpsc::channel();

        let ref ngrams: HashMap<String, f32> = parse_ngrams(&query, 2, &self.stopwords, &self.terms_relevance);

        for i in self.config.first_shard..self.config.last_shard {
            let j = (i - self.config.first_shard) as usize;
            let ngrams = ngrams.clone();
            let sender = sender.clone();
            let _ids = _ids.clone();
            let shards = self.shards.clone();

            thread::spawn(move || {

                let sh_ids = match get_shard_ids(j as usize, &ngrams, &shards[j].map, &shards[j].shard) {
                    Ok(ids) => ids,
                    Err(_) => {
                        println!("Failed to retrive ids from shard: {}", i);
                        vec![]
                    }
                };

                // obtaining lock might fail! handle it!
                let mut _ids = _ids.lock().unwrap();
                _ids[j] = sh_ids;
                sender.send(()).unwrap();
            });
        }

        for _ in self.config.first_shard..self.config.last_shard {
            receiver.recv().unwrap();
        }

        // deref MutexGuard returned from _ids.lock().unwrap()
        let data = (*_ids.lock().unwrap()).clone();
        Ok(data)
    }

    pub fn search(&self, query: &str) -> String {
        let ids = match self.get_ids(query.to_string()) {
            Ok(ids) => serde_json::to_string(&ids).unwrap(),
            Err(err) => err.to_string(),
        };

        ids
    }
}

fn main() {

    let c = config::Config::init();
    unsafe {
        NR_SHARDS = Some(c.nr_shards);
        ID_SIZE = Some(c.id_size);
        BUCKET_SIZE = Some(c.bucket_size);
        SHARD_SIZE = Some(c.shard_size);
    }

    let tr_map = match Map::from_path(&c.terms_relevance_path) {
        Ok(tr_map) => tr_map,
        Err(_) => panic!("Failed to load terms rel. map!")
    };
    let arc_tr_map = Arc::new(tr_map);

    let (sender, receiver) = mpsc::channel();

    for i in c.first_shard..c.last_shard {
        let sender = sender.clone();
        let tr_map = arc_tr_map.clone();

        let input_file_name = format!("{}/{}.{}.txt", c.dir_path, c.file_name, i);

        thread::spawn(move || {
            build_inverted_index(i, &input_file_name, &tr_map);
            sender.send(()).unwrap();
        });
    }

    for _ in c.first_shard..c.last_shard {
        receiver.recv().unwrap();
    }

    println!("Compiled {} shards.", c.last_shard - c.first_shard);
}
