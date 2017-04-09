extern crate fst;

use std::fs;
use fst::{Map, MapBuilder, Error};
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader;
use std::cmp::{Ordering, PartialOrd};
use std::io::BufWriter;
use std::fs::OpenOptions;
use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};
use std::io::SeekFrom;
use std::collections::HashSet;
use std::io::prelude::*;

use util;
use ngrams;

use std::collections::BinaryHeap;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct Qid {
    id: u32,
    sc: u8,
}

// The priority queue depends on `Ord`.
// Explicitly implement the trait so the queue becomes a min-heap
// instead of a max-heap.
impl Ord for Qid {
    fn cmp(&self, other: &Qid) -> Ordering {
    // Notice that the we flip the ordering here
    other.sc.cmp(&self.sc)
}
}

// `PartialOrd` needs to be implemented as well.
impl PartialOrd for Qid {
    fn partial_cmp(&self, other: &Qid) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn write_bucket(mut file: &File, addr: u64, data: &Vec<(u32, u8)>, id_size: usize) {
    file.seek(SeekFrom::Start(addr)).unwrap();
    let mut w = Vec::with_capacity(data.len()*id_size);
    for n in data.iter() {
        w.write_u32::<LittleEndian>(n.0).unwrap();
        w.write_u8(n.1).unwrap();
    };
    file.write_all(w.as_slice());
}

// build inverted query index, ngram_i -> [q1, q2, ... qi]
pub fn build_shard(
    iid: u32,
    input_file: &str,
    tr_map: &fst::Map,
    stopwords: &HashSet<String>,
    id_size: usize,
    bk_size: usize,
    nr_shards: usize) -> Result<(), Error>{

    let mut qcount = 0;
    let mut invert: HashMap<String, BinaryHeap<Qid>> = HashMap::new();

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

        for (ngram, sc) in &ngrams::parse(query, 2, stopwords, tr_map, ngrams::ParseMode::Indexing) {
            let imap = invert.entry(ngram.to_string()).or_insert(BinaryHeap::new());

            let pqid = util::qid2pqid(qid, nr_shards) as u32;
            let qsc = (sc * 100.0).round() as u8;
            let qid_obj = Qid{ id: pqid, sc: qsc};

            if imap.len() >= bk_size {
                let mut mqid = imap.peek_mut().unwrap();
                if qid_obj < *mqid { //in fact qid.sc > mqid.sc because the ordering is flipped
                    *mqid = qid_obj
                }
            } else {
                imap.push(qid_obj);
            }
        }

        qcount += 1;
        if qcount as u32 % 1_000_000 == 0 {
            println!("Reading {:.1}M", qcount / 1_000_000);
        }
    }

    // sort inverted query index by keys (ngrams) and store it to fst file
    let mut vinvert: Vec<(String, BinaryHeap<Qid>)> = invert.into_iter()
        .map(|(ngram, qids)| (ngram, qids))
        .collect();

    println!("Sorting ngrams...");
    vinvert.sort_by(|a, b| {a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal)});

    // create index dir if it doesn't exist
    try!(fs::create_dir_all("./index"));

    let index_map_file_name = "./index/map.".to_string() + &iid.to_string();
    // remove previous index first if exists
    fs::remove_file(&index_map_file_name);
    let mut wtr = BufWriter::new(try!(File::create(&index_map_file_name)));

    println!("Map {} init...", index_map_file_name);
    // Create a builder that can be used to insert new key-value pairs.
    let mut build = try!(MapBuilder::new(wtr));

    let index_file_name = "./index/shard.".to_string() + &iid.to_string();
    // remove previous index first if exists
    fs::remove_file(&index_file_name);
    let index_file = &OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&index_file_name).unwrap();

    let mut cursor: u64 = 0;
    for (ngram, qids) in vinvert.into_iter() {
        let qids_len: u64 = qids.len() as u64;
        if qids_len > bk_size as u64 {
            panic!("Error bucket for {:?} is has more than {:?} elements", ngram, bk_size);
        }
        let ids = qids.iter().map(|qid| (qid.id, qid.sc)).collect::<Vec<(u32, u8)>>();
        write_bucket(index_file, cursor*id_size as u64, &ids, id_size);
        build.insert(ngram, util::elegant_pair(cursor, qids_len));
        cursor += qids_len;
    }

    // Finish construction of the map and flush its contents to disk.
    try!(build.finish());

    println!("file {} created", index_file_name);
    Ok(())
}
