use std::io::Error;
use std::io::prelude::*;
use std::io::{BufReader, BufWriter, Write};
use fst::Map;
use std::fs::{read_dir, File, OpenOptions};
use std::path::Path;

use util;
use util::{BRED, BYELL, ECOL};
use config;
use stopwords;
use ngrams;

use std::fs;
use std::thread;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};

use std::iter::FromIterator;
use std::collections::{HashMap, HashSet};

const WRITE_BUFFER_SIZE: usize = 5 * 1024;

/*

Reads data from a input dir/file by multiple workers, each i-th worker reads every i-th row

*/

pub enum QueryType {
    Q,   // query
    TUW, // url or title words
}

impl From<String> for QueryType {
    fn from(prefix: String) -> Self {
        match prefix.as_ref() {
            "qe" => QueryType::Q,
            "q" => QueryType::Q,
            _ => QueryType::TUW,
        }
    }
}

#[inline]
pub fn parse_query_line(id_count: u64, line: &str) -> Result<(u64, String, String), Error> {
    let query_id = id_count;

    let v: Vec<&str> = line.split("\t").map(|t| t.trim()).collect();
    let v: Vec<&str> = v[0].trim_matches(|s| s == '"')
        .split(":")
        .map(|t| t.trim())
        .collect();
    let (query_id, query_type, query) = match v.len() {
        // prefix/type is missing, assume it's query
        1 => (query_id, "qe".to_string(), v[0].to_string()),
        // q:<query>
        2 => (query_id, v[0].to_string(), v[1].to_string()),
        // pre-appended id 0:q:<query>
        3 => (
            v[0].parse::<u64>().unwrap(),
            v[1].to_string(),
            v[2].to_string(),
        ),
        // something else
        _ => panic!("Unknown format: {:?}!", &line),
    };

    Ok((query_id, query_type, query))
}

pub fn shard(
    queries_path: &str,
    number_of_shards: usize,
    output_dir: &str,
    number_of_workers: usize,
    prefixes: &Vec<String>,
) -> Result<(), Error> {
    println!("Sharding...");

    // delete previous shards if they exist
    for i in 0..number_of_shards {
        let ref shard_path = format!("{}/ngrams.{}", output_dir, i);
        remove_file_if_exists!(shard_path);
    }

    let c = config::Config::init(output_dir.to_string());

    let stopwords_path = &format!("{}/{}", output_dir, c.stopwords_file);
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

    let (sender, receiver): (Sender<u64>, Receiver<u64>) = mpsc::channel();

    let queries_path = &Path::new(&queries_path);
    let queries_parts = if queries_path.is_dir() {
        let mut parts: Vec<_> = read_dir(queries_path)
            .unwrap()
            .map(|p| p.unwrap().path())
            .filter(|p| p.to_str().unwrap().ends_with(".gz"))
            .collect();
        parts.sort();
        parts
    } else {
        vec![queries_path.to_path_buf()]
    };

    let valid_prefixes: HashSet<String> = HashSet::from_iter(prefixes.clone());

    for worker_id in 0..number_of_workers {
        let sender = sender.clone();
        let stopwords = stopwords.clone();
        let queries_parts = queries_parts.clone();
        let valid_prefixes = valid_prefixes.clone();

        let mut shards = vec![];
        for shard_id in 0..number_of_shards {
            let file = OpenOptions::new()
                .create(true)
                .append(true) // will be written by multiple threads
                .open(format!("{}/ngrams.{}", output_dir, shard_id))
                .unwrap();

            let f = BufWriter::new(file);
            shards.push(f);
        }

        let mut shards_ngrams: HashMap<u32, String> = HashMap::new();

        let terms_relevance_path = &format!("{}/{}", output_dir, c.terms_relevance_file);
        let tr_map = match Map::from_path(terms_relevance_path) {
            Ok(tr_map) => tr_map,
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

        let query_files: Vec<(_, _)> = queries_parts
            .iter()
            .map(|query_part| {
                let query_file = File::open(&query_part).unwrap();
                (
                    query_part.clone(),
                    flate2::read::GzDecoder::new(query_file).expect("Not a valid gzip file."),
                )
            })
            .collect();

        thread::spawn(move || {
            let mut id_count: u64 = 0;
            let mut processed_count: u64 = 0;
            for (file_name, query_file) in query_files.into_iter() {
                println!("Worker: {}, Processing: {:?}", worker_id, file_name);
                let reader = BufReader::with_capacity(5 * 1024 * 1024, query_file);
                for line in reader.lines() {
                    id_count += 1;
                    if id_count as usize % number_of_workers != worker_id {
                        continue;
                    }

                    let line = line.unwrap();
                    let (query_id, query_type, query) = match parse_query_line(id_count - 1, &line)
                    {
                        Ok((query_id, query_type, query)) => (query_id, query_type, query),
                        Err(e) => {
                            println!(
                                "Read error: {:?}, line: {:?}, file: {:?}",
                                e,
                                &line.clone(),
                                file_name
                            );
                            continue;
                        }
                    };

                    if !valid_prefixes.is_empty() && !valid_prefixes.contains(&query_type) {
                        continue;
                    }

                    let ngrams =
                        &ngrams::parse(&query, &stopwords, &tr_map, QueryType::from(query_type));
                    for (ngram, sc) in ngrams {
                        let shard_id =
                            util::jump_consistent_hash_str(ngram, number_of_shards as u32);

                        // shard id to the query id
                        let (pqid, reminder) = util::qid2pqid(query_id, number_of_shards);
                        let qsc = (sc * 100.0).round() as u8;

                        // Note: writes u32 shard id for the query, not u64 query id, this is
                        // because a query id that is bigger than 2**32 overflows u64 in pairing
                        // function. When reading the shard id is used to get the original query id.
                        let line = format!("{}\t{}\t{}\t{}\n", pqid, reminder, ngram, qsc);

                        let sh_lines = shards_ngrams.entry(shard_id).or_insert(String::from(""));
                        *sh_lines = format!("{}{}", sh_lines, line);

                        if sh_lines.len() > WRITE_BUFFER_SIZE {
                            shards[shard_id as usize]
                                .write_all(sh_lines.as_bytes())
                                .expect("Unable to write data");

                            shards[shard_id as usize].flush().expect("Flush failed");

                            *sh_lines = String::from("");
                        }
                    }

                    processed_count += 1;
                    if processed_count as u64 % 1_000_000 == 0 {
                        println!(
                            "Processed {:.1}M queries, thread {}",
                            processed_count / 1_000_000,
                            worker_id
                        );
                    }
                }
            }

            // write the reminder
            for (shard_id, lines) in shards_ngrams {
                if lines.len() > 0 {
                    shards[shard_id as usize]
                        .write_all(lines.as_bytes())
                        .expect("Unable to write data");

                    shards[shard_id as usize].flush().expect("Flush failed");
                }
            }

            sender.send(processed_count).unwrap(); //finished!
        });
    }

    let mut total_processed_count: u64 = 0;
    for _ in 0..number_of_workers {
        total_processed_count += receiver.recv().unwrap();
    }
    println!(
        "Total count of sharded queries {:.1}",
        total_processed_count
    );

    Ok(())
}
