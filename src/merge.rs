use std::fs::File;
use std::io::BufWriter;
use fst::{Error, raw, Streamer};

fn merge(dir_path: String, nr_shards: usize) -> Result<(), Error>{
    let mut fsts = vec![];
    for i in 0..nr_shards {
        let fst = try!(raw::Fst::from_path(format!("{}/map.{}", dir_path, i)));
        fsts.push(fst);
    }
    let mut union = fsts.iter().collect::<raw::OpBuilder>().union();

    let wtr = BufWriter::new(try!(File::create(format!("{}/union_map.{}.fst", dir_path, nr_shards))));
    let mut builder = try!(raw::Builder::new(wtr));

    while let Some((k, vs)) = union.next() {
        try!(builder.insert(k, vs[0].value));
    }
    try!(builder.finish());

    Ok(())
}
