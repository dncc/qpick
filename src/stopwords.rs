use fnv::FnvHashSet;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Error;

pub fn load(path: &str) -> Result<FnvHashSet<String>, Error> {
    let mut stopwords = FnvHashSet::default();

    let f = File::open(path)?;
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
