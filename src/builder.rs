extern crate fst;

use std::fs;
use fst::{MapBuilder, Error};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::cmp::{Ordering, PartialOrd};
use std::io::BufWriter;
use std::fs::OpenOptions;
use byteorder::{LittleEndian, WriteBytesExt};
use std::io::SeekFrom;
use std::io::prelude::*;
use std::sync::mpsc;
use std::thread;

use util;
use config;

use std::collections::BinaryHeap;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct Qid {
    id: u32,        // query id: is equal to globally_unique_u64_query_id << log(nr_shards), unique on shard level
    reminder: u8,   // reminder: from globally_unique_u64_query_id % number_of_shards
    sc: u8,         // score: ngram relevance/score for the query
}

// The priority queue depends on `Ord`.
// Explicitly implement the trait to turn the queue into a min-heap (by default it's a max-heap).
impl Ord for Qid {
    fn cmp(&self, other: &Qid) -> Ordering {
        other.sc.cmp(&self.sc) // flipped ordering here!
    }
}

// `PartialOrd` needs to be implemented as well.
impl PartialOrd for Qid {
    fn partial_cmp(&self, other: &Qid) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn write_bucket(mut file: &File, addr: u64, data: &Vec<(u32, u8, u8)>, id_size: usize) {
    file.seek(SeekFrom::Start(addr)).unwrap();
    let mut w = Vec::with_capacity(data.len()*id_size);
    for n in data.iter() {
        w.write_u32::<LittleEndian>(n.0).unwrap();
        w.write_u8(n.1).unwrap();
        w.write_u8(n.2).unwrap();
    };
    file.write_all(w.as_slice()).unwrap();
}

pub fn index(input_dir: &str,
             shard_name: &str,
             first_shard: usize,
             last_shard: usize,
             output_dir: &str) -> Result<(), Error> {

    let c = config::Config::init();

    // create index dir if it doesn't exist
    try!(fs::create_dir_all(output_dir));

    let (sender, receiver) = mpsc::channel();

    for i in first_shard..last_shard {
        let sender = sender.clone();

        let id_size = c.id_size.clone();
        let bucket_size = c.bucket_size.clone();

        let input_file_name = format!("{}/{}.{}", input_dir, shard_name, i);
        let out_shard_name = format!("{}/{}.{}", output_dir, "shard", i);
        let out_map_name = format!("{}/{}.{}", output_dir, "map", i);

        thread::spawn(move || {
            build_shard(i as u32, &input_file_name, id_size, bucket_size,
                        &out_shard_name, &out_map_name).unwrap();

            sender.send(()).unwrap();
        });
    }

    for _ in first_shard..last_shard {
        receiver.recv().unwrap();
    }

    println!("Compiled {} shards.", last_shard - first_shard);

    Ok(())
}

macro_rules! remove_file_if_exists {
    ($fn_name:ident) => (
        match fs::remove_file($fn_name) {
            Err(err) => println!("Failed to delete previous {}, err: {:?}", $fn_name, err),
            Ok(_) => println!("Deleted previous file {}", $fn_name),
        };
    )
}

// build inverted query index, ngram_i -> [q1, q2, ... qi]
pub fn build_shard(

    iid: u32,
    input_file: &str,
    id_size: usize,
    bk_size: usize,
    out_shard_name: &str,
    out_map_name: &str) -> Result<(), Error>{

    let mut qcount: u64 = 0;
    let mut invert: HashMap<String, BinaryHeap<Qid>> = HashMap::new();

    let f = try!(File::open(input_file));
    let reader = BufReader::with_capacity(5 * 1024 * 1024, &f);
    for line in reader.lines() {

        let line = match line {
            Ok(line) => line,
            Err(e) => {
                println!("Read line error: {:?}", e);
                continue
            }
        };

        let mut split = line.trim().split("\t");

        let pqid = match split.next() {
            Some(pqid) => match pqid.parse::<u32>() {
                Ok(n) => n,
                Err(err) => {
                    println!("Shard {:?} - failed to parse query id {:?}: {:?}", iid, pqid, err);
                    continue
                }
            },
            None => {
                println!("Shard {:?} - No query id found", iid);
                continue
            }
        };

        let reminder = match split.next() {
            Some(r) => match r.parse::<u8>() {
                Ok(n) => n,
                Err(err) => {
                    println!("Shard {:?} - failed to parse query id {:?}: {:?}", iid, pqid, err);
                    continue
                }
            },
            None => {
                println!("Shard {:?} - No query id found", iid);
                continue
            }
        };

        let ngram = match split.next() {
                Some(ng) => ng.trim(),
                None => {
                        println!("Shard {:?} - No ngram found", iid);
                        continue
                }
        };

        let nsc = match split.next() {
            Some(nsc) => match nsc.parse::<u8>() {
                Ok(n) => n,
                Err(err) => {
                    println!("Shard {:?} - failed to parse ngram score {:?}: {:?}", iid, nsc, err);
                    continue
                }
            },
            None => {
                println!("Shard {:?} - No query score found", iid);
                continue
            }
        };

        let imap = invert.entry(ngram.to_string()).or_insert(BinaryHeap::new());

        let qid_obj = Qid{ id: pqid, reminder: reminder, sc: nsc};

        if imap.len() >= bk_size {
            let mut mqid = imap.peek_mut().unwrap();
            if qid_obj < *mqid { //in fact qid.sc > mqid.sc, due to the flipped ordering
                *mqid = qid_obj
            }
        } else {
            imap.push(qid_obj);
        }

        qcount += 1;
        if qcount as u64 % 1_000_000 == 0 {
            println!("Reading {:.1}M", qcount / 1_000_000);
        }
    }

    // sort inverted query index by keys (ngrams) and store it to fst file
    let mut vinvert: Vec<(String, BinaryHeap<Qid>)> = invert.into_iter()
        .map(|(key, qids)| (key, qids))
        .collect();

    println!("Sorting {:.1}M keys...", vinvert.len()/1_000_000);
    vinvert.sort_by(|a, b| {a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)});

    // remove previous index first if exists

    remove_file_if_exists!(out_map_name);
    let wtr = BufWriter::new(try!(File::create(out_map_name)));

    println!("Map {} init...", out_map_name);
    // Create a builder that can be used to insert new key-value pairs.
    let mut build = try!(MapBuilder::new(wtr));

    // remove previous index first if exists
    remove_file_if_exists!(out_shard_name);
    let index_file = &OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(out_shard_name).unwrap();

    let mut cursor: u64 = 0;
    for (key, qids) in vinvert.into_iter() {
        let qids_len: u64 = qids.len() as u64;
        if qids_len > bk_size as u64 {
            panic!("Error bucket for {:?} is has more than {:?} elements", key, bk_size);
        }
        let ids = qids.iter().map(|qid| (qid.id, qid.reminder, qid.sc)).collect::<Vec<(u32, u8, u8)>>();
        write_bucket(index_file, cursor*id_size as u64, &ids, id_size);
        let val = util::elegant_pair(cursor, qids_len).unwrap();
        build.insert(key, val).unwrap();

        cursor += qids_len;
    }

    // Finish construction of the map and flush its contents to disk.
    try!(build.finish());

    println!("Shard {} created", out_shard_name);
    Ok(())
}
