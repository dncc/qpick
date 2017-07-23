use std::io::Error;
use std::fs::File;
use std::io::{Write, BufWriter, BufReader};
use std::io::prelude::*;
use fst::Map;

use util;
use config;
use stopwords;
use ngrams;

pub fn shard(file_path: &str, nr_shards: usize, output_dir: &str) -> Result<(), Error> {
    println!("Sharding...");

    let f = try!(File::open(file_path));
    let mut reader = BufReader::with_capacity(5 * 1024 * 1024, &f);

    let mut qid: u64 = 0;

    let mut shards = vec![];
    for i in 0..nr_shards {
        let file_path = format!("{}/queries.{}", output_dir, i);
        let f = File::create(file_path).expect("Unable to create file");
        let mut f = BufWriter::new(f);
        shards.push(f);
    }

    let c = config::Config::init();

    let ref stopwords = match stopwords::load(&c.stopwords_path) {
        Ok(stopwords) => stopwords,
        Err(_) => panic!("Failed to load stop-words!")
    };

    let ref tr_map = match Map::from_path(&c.terms_relevance_path) {
        Ok(tr_map) => tr_map,
        Err(_) => panic!("Failed to load terms rel. map!")
    };

    for line in reader.lines() {
        let line = match line {
            Ok(line) => line,
            Err(e) => {
                println!("Read line error: {:?}", e);
                continue
            }
        };

        let mut v: Vec<&str> = line.split(":").map(|t| t.trim()).collect();

        let ref query = match v.len() {
            2 => v[1].to_string(),
            _ => v[1..v.len()-1].join(" "),
        };

        for (ngram, sc) in &ngrams::parse(query, 2, stopwords, tr_map, ngrams::ParseMode::Indexing) {

            let shard_id = util::jump_consistent_hash_str(ngram, nr_shards as u32);

            let (pqid, reminder) = util::qid2pqid(qid, nr_shards); // shard id for the query id
            let qsc = (sc * 100.0).round() as u8;

            // Note: writes shard id for the query (u32), not query id (u64),
            // because a query id that is bigger than 2**32 overflows u64 in pairing function.
            // When reading from the index the shard id is used to get the original query id.
            let line = format!("{}\t{}\t{}\t{}\n", pqid, reminder, ngram, qsc);

            shards[shard_id as usize].write_all(line.as_bytes()).expect("Unable to write data");

        }

        qid += 1;
        if qid as u64 % 1_000_000 == 0 {
            println!("Reading {:.1}M", qid / 1_000_000);
        }
    }

    Ok(())
}
