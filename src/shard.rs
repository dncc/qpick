use std::io::Error;
use std::fs::File;
use std::io::{Write, BufWriter, BufReader};
use std::io::prelude::*;

use util;

pub fn shard(file_path: &str, nr_shards: usize, output_dir: &str) -> Result<(), Error> {
    println!("Run sharding");

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
        let shard_id = util::jump_consistent_hash_str(query, nr_shards as u32);

        let line = format!("{}\t{}\n", qid, query);
        shards[shard_id as usize].write_all(line.as_bytes()).expect("Unable to write data");

        qid += 1;
        if qid as u64 % 1_000_000 == 0 {
            println!("Reading {:.1}M", qid / 1_000_000);
        }
    }

    Ok(())
}
