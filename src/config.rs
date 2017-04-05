extern crate serde_json;

use std;
use std::io::Error;
use std::io::Read;
use std::io::BufReader;
use std::fs::File;
use serde_json::Value;

pub struct Config {
    pub id_size: usize,     // query id size in bytes (4 for id + 1 for relevance)
    pub bucket_size: usize, // max number of query ids in a ngram bucket
    pub nr_shards: usize,
    pub shard_size: usize,  // number of ids in the shard
    pub first_shard: u32,
    pub last_shard: u32,
    pub dir_path: String,
    pub file_name: String,
    pub terms_relevance_path: String,
    pub stopwords_path: String,
}

impl Config {

    fn load_config_file() -> Result<String, Error> {
        let f = try!(File::open("/home/dnc/workspace/cliqz/qpick/config.json"));
        let mut buf = BufReader::new(&f);
        let mut config = String::new();
        buf.read_to_string(&mut config);

        Ok(config)
    }

    pub fn init() -> Self {
        let config_content = Config::load_config_file().unwrap();
        let config: Value = serde_json::from_str(&config_content).unwrap();

        let first_shard = match config["first_shard"] {
            Value::Number(ref first_shard) => first_shard.as_u64().unwrap(),
            _ => 0,
        };

        let nr_shards = match config["nr_shards"] {
            Value::Number(ref nr_shards) => nr_shards.as_u64().unwrap(),
             _ => 32,
        };

        let last_shard = match config["last_shard"] {
            Value::Number(ref last_shard) => last_shard.as_u64().unwrap(),
            _ => nr_shards,
        };

        let shard_size = match config["shard_size"] {
            Value::Number(ref shard_size) => shard_size.as_u64().unwrap(),
            _ => 250_000_000,
        };

        let bucket_size = match config["bucket_size"] {
            Value::Number(ref bucket_size) => bucket_size.as_u64().unwrap(),
            _ => 10_000,
        };

        let id_size = match config["id_size"] {
            Value::Number(ref id_size) => id_size.as_u64().unwrap(),
            _ => 5,
        };

        let dir_path = match config["dir_path"] {
            Value::String(ref dir_path) => dir_path.as_str(),
            _ => panic!("Failed to load index path from the config file!"),
        };

        let file_name = match config["file_name"] {
            Value::String(ref file_name) => file_name.as_str(),
            _ => panic!("Failed to load index file name from the config file!"),
        };

        let terms_relevance_path = match config["terms_relevance_path"] {
            Value::String(ref terms_relevance_path) => terms_relevance_path.as_str(),
            _ => panic!("Failed to load terms relevance path from the config file!"),
        };

        let stopwords_path = match config["stopwords_path"] {
            Value::String(ref stopwords_path) => stopwords_path.as_str(),
            _ => panic!("Failed to load stopwords path name from the config file!"),
        };

        Config {
            id_size: id_size as usize,
            bucket_size: bucket_size as usize,
            first_shard: first_shard as u32,
            last_shard: last_shard as u32,
            nr_shards: nr_shards as usize,
            shard_size: shard_size as usize,
            dir_path: dir_path.to_string(),
            file_name: file_name.to_string(),
            terms_relevance_path: terms_relevance_path.to_string(),
            stopwords_path: stopwords_path.to_string()
        }
    }
}
