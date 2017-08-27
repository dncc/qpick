use std::io::Error;
use std::fs::File;
use std::io::{Write, BufWriter, BufReader};
use std::io::prelude::*;
use fst::Map;

use std::fs::OpenOptions;

use util;
use config;
use stopwords;
use ngrams;

pub fn shard(file_path: &str, nr_shards: usize, output_dir: &str) -> Result<(), Error> {
    println!("Sharding...");

    let f = try!(File::open(file_path));
    let reader = BufReader::with_capacity(5 * 1024 * 1024, &f);

    let mut qcount: u64 = 0;

    let mut shards = vec![];
    for i in 0..nr_shards {
        let file_path = format!("{}/queries.{}", output_dir, i);
        let file =
            OpenOptions::new()
            .create(true)
            .append(true)
            .open(file_path)
            .unwrap();

        let f = BufWriter::new(file);
        shards.push(f);
    }

    let c = config::Config::init(output_dir.to_string());

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

        let v: Vec<&str> = line.split(":").map(|t| t.trim()).collect();

        let qid = v[0].parse::<u64>().unwrap();
        let ref query = match v.len() {
            3 => v[2].to_string(),
            _ => v[2..v.len()-1].join(" "),
        };

        for (ngram, sc) in &ngrams::parse(query, stopwords, tr_map) {

            let shard_id = util::jump_consistent_hash_str(ngram, nr_shards as u32);

            let (pqid, reminder) = util::qid2pqid(qid, nr_shards); // shard id for the query id
            let qsc = (sc * 100.0).round() as u8;

            // Note: writes shard id for the query (u32), not query id (u64),
            // because a query id that is bigger than 2**32 overflows u64 in pairing function.
            // When reading from the index the shard id is used to get the original query id.
            let line = format!("{}\t{}\t{}\t{}\n", pqid, reminder, ngram, qsc);

            shards[shard_id as usize].write_all(line.as_bytes()).expect("Unable to write data");

        }

        qcount += 1;
        if qcount as u64 % 1_000_000 == 0 {
            println!("Reading {:.1}M", qcount / 1_000_000);
        }
    }

    Ok(())
}
