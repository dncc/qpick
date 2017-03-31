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
use std::io::prelude::*;

use util;
use ngrams;
use stopwords;

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
    id_size: usize,
    bk_size: usize,
    nr_shards: usize) -> Result<(), Error>{

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

        for (ngram, sc) in &ngrams::parse(query, 2, stopwords, tr_map) {
            let imap = invert.entry(ngram.to_string()).or_insert(HashMap::new());
            let pqid = util::qid2pqid(qid, nr_shards);
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
    for (ngram, ids_tr) in vinvert.into_iter() {
        let ids_len: u64 = ids_tr.len() as u64;
        if ids_len > bk_size as u64 {
            continue
        }
        let ids = ids_tr.iter().map(|(k,v)| (*k, *v)).collect::<Vec<(u32, u8)>>();
        write_bucket(index_file, cursor*id_size as u64, &ids, id_size);
        build.insert(ngram, util::elegant_pair(cursor, ids_len));
        cursor += ids_len;
    }

    // Finish construction of the map and flush its contents to disk.
    try!(build.finish());

    println!("file {} created", index_file_name);
    Ok(())
}
