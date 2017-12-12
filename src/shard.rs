use std::io::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::io::prelude::*;
use fst::Map;

use std::fs::OpenOptions;

use util;
use config;
use stopwords;
use ngrams;

use std::fs;
use std::thread;
use std::sync::mpsc;
use std::sync::mpsc::{Sender, Receiver};

/**
Reads data from a single input file by multiple threads so that each i-th thread
reads every i-th row in the file. This should have probably been done with multi
threaded single-sender-multiple-receivers message passing, w/o broadcasting, had
it been supported in the rust's std library.
**/

pub fn shard(
    file_path: &str,
    nr_shards: usize,
    output_dir: &str,
    nthreads: usize
) -> Result<(), Error> {
    println!("Sharding...");

    // delete previous shards if they exist
    for i in 0..nr_shards {
        let ref shard_path = format!("{}/queries.{}", output_dir, i);
        remove_file_if_exists!(shard_path);
    }

    let c = config::Config::init(output_dir.to_string());

    let stopwords = match stopwords::load(&c.stopwords_path) {
        Ok(stopwords) => stopwords,
        Err(_) => panic!("Failed to load stop-words!"),
    };

    let (sender, receiver):(Sender<u64>, Receiver<u64>) = mpsc::channel();

    for i in 0..nthreads {

        let sender = sender.clone();
        let stopwords = stopwords.clone();

        let mut shards = vec![];
        for i in 0..nr_shards {
            let file = OpenOptions::new()
                .create(true)
                .append(true) // will be written by multiple threads
                .open(format!("{}/queries.{}", output_dir, i))
                .unwrap();

            let f = BufWriter::new(file);
            shards.push(f);
        }

        let tr_map = match Map::from_path(&c.terms_relevance_path) {
            Ok(tr_map) => tr_map,
            Err(_) => panic!("Failed to load terms rel. map!"),
        };

        let f = try!(File::open(file_path));

        thread::spawn(move || {
            let mut line_count: u64 = 0;
            let reader = BufReader::with_capacity(5 * 1024 * 1024, &f);

            for (lnum, line) in reader.lines().enumerate() {
                if lnum % nthreads != i {
                    continue
                }

                let line = match line {
                    Ok(line) => line,
                    Err(e) => {
                        println!("Read line error: {:?}", e);
                        continue;
                    }
                };

                let v: Vec<&str> = line.split(":").map(|t| t.trim()).collect();

                let qid = v[0].parse::<u64>().unwrap();
                let ref query = match v.len() {
                    3 => v[2].to_string(),
                    _ => v[2..v.len() - 1].join(" "),
                };

                let ngrams = &ngrams::parse(query, &stopwords, &tr_map);
                for (ngram, sc) in ngrams {
                    let shard_id = util::jump_consistent_hash_str(ngram, nr_shards as u32);

                    let (pqid, reminder) = util::qid2pqid(qid, nr_shards); // shard id for the query id
                    let qsc = (sc * 100.0).round() as u8;

                    // Note: writes u32 shard id for the query, not u64 query id, this is because
                    // a query id that is bigger than 2**32 overflows u64 in pairing function.
                    // When reading from the index the shard id is used to get the original query id.
                    let line = format!("{}\t{}\t{}\t{}\n", pqid, reminder, ngram, qsc);
                    shards[shard_id as usize]
                        .write_all(line.as_bytes())
                        .expect("Unable to write data");

                    shards[shard_id as usize]
                        .flush()
                        .expect("Flush failed");
                }

                line_count += 1;
                if line_count as u64 % 1_000_000 == 0 {
                    println!("Processed {:.1}M queries, thread {}", line_count / 1_000_000, i);
                }
            }

            sender.send(line_count).unwrap(); //finished!
        });
    }

    let mut line_count: u64 = 0;
    for _ in 0..nthreads {
        line_count += receiver.recv().unwrap();
    }
    println!("Total count of sharded queries {:.1}", line_count);

    Ok(())
}
