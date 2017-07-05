extern crate fst;
extern crate byteorder;
extern crate serde_json;

use std::thread;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

use std::fs::File;
use std::fs::OpenOptions;
use std::io::prelude::*;
use std::cmp::{Ordering, PartialOrd};
use std::collections::HashMap;
use std::collections::HashSet;

use byteorder::{ByteOrder, LittleEndian};
use fst::Map;
use std::io::SeekFrom;

use fst::Error;

pub mod util;
pub mod config;
pub mod ngrams;
pub mod merge;
pub mod builder;
pub mod stopwords;

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

// reading part
#[inline]
fn get_addr_and_len(ngram: &str, pid: usize, map: &fst::Map) -> Option<(u64, u64)> {
    let ref key = util::ngram2key(ngram, pid as u32);
    match map.get(key) {
        Some(val) => {
            return Some(util::elegant_pair_inv(val))
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

        match get_addr_and_len(ngram, pid, &map) {
            Some((addr, len)) => {
                for id_tr in read_bucket(&ifd, addr*id_size as u64, len).iter() {
                    let qid = util::pqid2qid(id_tr.0 as u64, pid as u64, *get_nr_shards());
                    let sc = _ids.entry(qid).or_insert(0.0);
                    // println!("{:?} {:?} {:?}", ngram, qid, sc);
                    // TODO cosine similarity, normalize ngrams relevance at indexing time
                    // *sc += weight * ntr;
                    let mut weight = (id_tr.1 as f32)/100.0 ;
                    weight = util::max(0.0, ntr - (ntr - weight).abs() as f32);
                    *sc += weight * (n/len as f32).log(2.0);
                }
            },
            None => (),
        }
    }

    let mut v: Vec<(u64, f32)> = _ids.iter().map(|(id, sc)| (*id, *sc)).collect::<Vec<_>>();
    v.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Less).reverse());
    v.truncate(100); //TODO put into config
    Ok(v)
}

use std::cell::RefCell;
pub struct Qpick {
    path: String,
    config: config::Config,
    stopwords: HashSet<String>,
    terms_relevance: fst::Map,
    shards: Arc<Vec<Shard>>,
    thread_pool: RefCell<Pool>,
}

pub struct Shard {
    id: u32,
    map: fst::Map,
    shard: File,
}

impl Qpick {
    fn new(path: String) -> Qpick {

        let c = config::Config::init();

        unsafe {
            // TODO set up globals, later should be available via self.config
            NR_SHARDS = Some(c.nr_shards);
            ID_SIZE = Some(c.id_size);
            BUCKET_SIZE = Some(c.bucket_size);
            SHARD_SIZE = Some(c.shard_size);
        }

        let stopwords = match stopwords::load(&c.stopwords_path) {
            Ok(stopwords) => stopwords,
            Err(_) => panic!("Failed to load stop-words!")
        };

        let terms_relevance = match Map::from_path(&c.terms_relevance_path) {
            Ok(terms_relevance) => terms_relevance,
            Err(_) => panic!("Failed to load terms rel. map: {}!", &c.terms_relevance_path)
        };

        let mut shards = vec![];
        for i in c.first_shard..c.last_shard {
            let map_name = format!("{}/map.{}", path, i);
            let map = match Map::from_path(&map_name) {
                Ok(map) => map,
                Err(_) => panic!("Failed to load index map: {}!", &map_name)
            };

            let shard = OpenOptions::new().read(true).open(format!("{}/shard.{}", path, i)).unwrap();
            shards.push(Shard{id: i, shard: shard, map: map});
        };

        let mut thread_pool = Pool::new(c.last_shard - c.first_shard);

        Qpick {
            config: c,
            path: path,
            stopwords: stopwords,
            terms_relevance: terms_relevance,
            shards: Arc::new(shards),
            thread_pool: RefCell::new(thread_pool),
        }
    }

    pub fn from_path(path: String) -> Self {
        Qpick::new(path)
    }

    fn get_ids(&self, query: String) -> Result<Vec<Vec<(u64, f32)>>, Error> {

        let mut _ids: Arc<Mutex<Vec<Vec<(u64, f32)>>>> = Arc::new(Mutex::new(vec![vec![]; self.config.nr_shards]));
        let (sender, receiver) = mpsc::channel();

        let ref ngrams: HashMap<String, f32> = ngrams::parse(
            &query, 2, &self.stopwords, &self.terms_relevance, ngrams::ParseMode::Searching);

        self.thread_pool.borrow_mut().scoped(|scoped| {

            for i in self.config.first_shard..self.config.last_shard {
                let j = (i - self.config.first_shard) as usize;
                let ngrams = ngrams.clone();
                let sender = sender.clone();
                let _ids = _ids.clone();
                let shards = self.shards.clone();

                scoped.execute(move || {

                    let sh_ids = match get_shard_ids(j as usize, &ngrams, &shards[j].map, &shards[j].shard) {
                        Ok(ids) => ids,
                        Err(_) => {
                            println!("Failed to retrive ids from shard: {}", i);
                            vec![]
                        }
                    };

                    // obtaining lock might fail! handle it!
                    let mut _ids = _ids.lock().unwrap();
                    _ids[i as usize] = sh_ids;
                    sender.send(()).unwrap();
                });
            }

        });

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

    pub fn merge(&self) -> Result<(), Error> {
        println!("Merging index maps from: {:?}", &self.path);
        merge::merge(&self.path, (self.config.last_shard - self.config.first_shard) as usize)
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

    let ref stopwords = match stopwords::load(&c.stopwords_path) {
        Ok(stopwords) => stopwords,
        Err(_) => panic!("Failed to load stop-words!")
    };

    let tr_map = match Map::from_path(&c.terms_relevance_path) {
        Ok(tr_map) => tr_map,
        Err(_) => panic!("Failed to load terms rel. map!")
    };
    let arc_tr_map = Arc::new(tr_map);

    let (sender, receiver) = mpsc::channel();

    for i in c.first_shard..c.last_shard {
        let sender = sender.clone();
        let tr_map = arc_tr_map.clone();
        let stopwords = stopwords.clone();

        let input_file_name = format!("{}/{}.{}.txt", c.dir_path, c.file_name, i);

        thread::spawn(move || {
            builder::build_shard(i, &input_file_name, &tr_map, &stopwords,
                                 *get_id_size(), *get_bucket_size(), *get_nr_shards());
            sender.send(()).unwrap();
        });
    }

    for _ in c.first_shard..c.last_shard {
        receiver.recv().unwrap();
    }

    println!("Compiled {} shards.", c.last_shard - c.first_shard);
}
