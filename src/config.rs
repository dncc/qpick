extern crate serde_json;
use std::io::Error;
use std::io::Read;
use std::io::BufReader;
use std::fs::File;
use serde_json::Value;

pub struct Config {
    pub id_size: usize,     // query id size in bytes (4 for id + 1 for relevance)
    pub bucket_size: usize, // max number of query ids in a ngram bucket
    pub nr_shards: usize,
    pub shard_size: usize, // number of ids in the shard
    pub terms_relevance_file: String,
    pub thread_pool_size: usize,
    pub stopwords_file: String,
    pub i2q_file: String,
}

impl Config {
    fn load_config_file(path: &str) -> Result<String, Error> {
        let f = try!(File::open(format!("{}/config.json", path)));
        let mut buf = BufReader::new(&f);
        let mut config = String::new();
        buf.read_to_string(&mut config).unwrap();

        Ok(config)
    }

    pub fn init(path: String) -> Self {
        let config_content = match Config::load_config_file(&path) {
            Ok(config) => config,
            Err(err) => panic!("Failed to load {}/config.json, err: {:?}", path, err),
        };

        let config: Value = serde_json::from_str(&config_content).unwrap();

        let nr_shards = match config["nr_shards"] {
            Value::Number(ref nr_shards) => nr_shards.as_u64().unwrap(),
            _ => 32,
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
            _ => 7,
        };

        let terms_relevance_file = match config["terms_relevance_file"] {
            Value::String(ref terms_relevance_file) => terms_relevance_file.as_str(),
            _ => panic!("Failed to load terms relevance file name from the config!"),
        };

        let stopwords_file = match config["stopwords_file"] {
            Value::String(ref stopwords_file) => stopwords_file.as_str(),
            _ => panic!("Failed to load stopwords file name from the config!"),
        };

        let i2q_file = match config["i2q_file"] {
            Value::String(ref i2q_file) => i2q_file.as_str(),
            _ => panic!("Failed to load i2q file name from the config!"),
        };

        let thread_pool_size = match config["thread_pool_size"] {
            Value::Number(ref thread_pool_size) => thread_pool_size.as_u64().unwrap(),
            _ => panic!("Failed to load thread_pool_size from the config file!"),
        };

        Config {
            id_size: id_size as usize,
            bucket_size: bucket_size as usize,
            nr_shards: nr_shards as usize,
            shard_size: shard_size as usize,
            terms_relevance_file: terms_relevance_file.to_string(),
            stopwords_file: stopwords_file.to_string(),
            i2q_file: i2q_file.to_string(),
            thread_pool_size: thread_pool_size as usize,
        }
    }
}
