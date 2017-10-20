use std::io::Error;
use std::io::BufRead;
use std::io::BufReader;
use std::fs::File;
use std::collections::HashSet;

pub fn load(path: &str) -> Result<HashSet<String>, Error> {
    let mut stopwords = HashSet::new();

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
