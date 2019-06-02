use std::io::Error;
use std::io::BufRead;
use std::io::BufReader;
use std::fs::File;
use fnv::FnvHashSet;

pub fn load(path: &str) -> Result<FnvHashSet<String>, Error> {
    let mut stopwords = FnvHashSet::default();

    let f = try!(File::open(path));
    let file = BufReader::new(&f);

    for line in file.lines() {
        let sw = line.unwrap();
        stopwords.insert(sw);
    }

    if stopwords.len() == 0 {
        panic!("Stopwords are empty!");
    }

    Ok(stopwords)
}
