use fnv::FnvHashMap;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Error;
use std::path::Path;

#[inline]
pub fn parse_line(line: &str) -> Result<(String, String), Error> {
    let v: Vec<&str> = line.split(" ").map(|t| t.trim()).collect();
    let (word, synonym) = match v.len() {
        2 => (v[0].to_string(), v[1].to_string()),
        _ => panic!("Unknown format: {:?}!", &line),
    };

    Ok((word, synonym))
}

pub fn load(path: &Path) -> Result<FnvHashMap<String, String>, Error> {
    let mut synonyms = FnvHashMap::default();

    let f = try!(File::open(path));
    let file = BufReader::new(&f);

    for line in file.lines() {
        let line = line.unwrap();
        let (word, synonym) = parse_line(&line).unwrap();
        synonyms.insert(word, synonym);
    }

    Ok(synonyms)
}
