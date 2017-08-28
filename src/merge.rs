use std::fs::File;
use std::io::BufWriter;
use fst::{Error, raw, Streamer};

static SEPARATOR: &'static str = "\u{0}\u{0}";

#[inline]
fn npid2key(ngramv: &mut Vec<u8>, pid: usize) -> String {
    let n = String::from_utf8_lossy(&ngramv).into_owned();
    let k = format!("{}{}{}", n, SEPARATOR, pid);

    if key2npid(&k) != (n.clone(), pid) {
        panic!("Failed to turn {} and {} into key {}", n, pid, &k);
    };

    k
}

#[inline]
fn key2npid(key: &str) -> (String, usize) {

    let split = key.split(SEPARATOR);
    let v = split.collect::<Vec<&str>>();

    assert_eq!(v.len(), 2);

    let ngram = v[0].to_string();
    let pid = v[1].parse::<usize>().unwrap();

    (ngram, pid)
}

const PROGRESS: u64 = 1_000_000;

pub fn merge(dir_path: &str, nr_shards: usize) -> Result<(), Error> {
    let mut fsts = vec![];
    for i in 0..nr_shards {
        let fst = try!(raw::Fst::from_path(format!("{}/map.{}", dir_path, i)));
        fsts.push(fst);
    }
    let mut union = fsts.iter().collect::<raw::OpBuilder>().union();

    let wtr =
        BufWriter::new(try!(File::create(format!("{}/union_map.{}.fst", dir_path, nr_shards))));
    let mut builder = try!(raw::Builder::new(wtr));

    let mut count: u64 = 0;
    while let Some((k, vs)) = union.next() {
        // v = [IndexValue{index:0, value: 1}, IndexValue{index:1, value: 0}, ... ]
        let mut v = vs.to_vec();
        // has to be sorted by index, it is sorted by value by default
        v.sort_by(|a, b| a.index.partial_cmp(&b.index).unwrap());

        let ref mut kv = k.to_vec();
        for iv in v.iter() {
            let k = npid2key(kv, iv.index);
            builder.insert(k, iv.value).unwrap();
        }

        count += 1;
        if count % PROGRESS == 0 {
            println!("Merging {:.1}M ...", count / PROGRESS);
        }

    }
    try!(builder.finish());

    Ok(())
}
