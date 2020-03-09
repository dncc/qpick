extern crate serde_json;
use serde_json::Value;
use std::fs::File;
use std::io::BufReader;
use std::io::Error;
use std::io::Read;

pub struct Config {
    pub id_size: usize,     // query id size in bytes (4 for id + 1 for relevance)
    pub bucket_size: usize, // max number of query ids in a ngram bucket
    pub nr_shards: usize,
    pub shard_size: usize, // number of ids in the shard
    pub terms_relevance_file: String,
    pub stopwords_file: String,
    pub toponyms_file: String,
    pub synonyms_file: String,
    pub i2q_file: String,
    pub words_file: String,
    pub word_vecs_file: String,
    pub use_word_vectors: bool,
}

impl Config {
    fn load_config_file(path: &str) -> Result<String, Error> {
        let f = File::open(format!("{}/config.json", path))?;
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
            _ => 64,
        };

        let shard_size = match config["shard_size"] {
            Value::Number(ref shard_size) => shard_size.as_u64().unwrap(),
            _ => 250_000_000,
        };

        let bucket_size = match config["bucket_size"] {
            Value::Number(ref bucket_size) => bucket_size.as_u64().unwrap(),
            _ => 2_500,
        };

        let id_size = match config["id_size"] {
            Value::Number(ref id_size) => id_size.as_u64().unwrap(),
            _ => 6,
        };

        let terms_relevance_file = match config["terms_relevance_file"] {
            Value::String(ref terms_relevance_file) => terms_relevance_file.as_str(),
            _ => panic!("Failed to parse file name for word relevances from the config!"),
        };

        let toponyms_file = match config["toponyms_file"] {
            Value::String(ref toponyms_file) => toponyms_file.as_str(),
            _ => {
                println!("File name for toponyms is not provided in the config!");
                ""
            }
        };

        let stopwords_file = match config["stopwords_file"] {
            Value::String(ref stopwords_file) => stopwords_file.as_str(),
            _ => panic!("Failed to parse stopwords file name from the config!"),
        };

        let synonyms_file = match config["synonyms_file"] {
            Value::String(ref synonyms_file) => synonyms_file.as_str(),
            _ => {
                println!("File name for synonyms is not provided in the config!");
                ""
            }
        };

        let i2q_file = match config["i2q_file"] {
            Value::String(ref i2q_file) => i2q_file.as_str(),
            _ => {
                println!("File name for i2q mapping is not provided in the config!");
                ""
            }
        };

        let words_file = match config["words_file"] {
            Value::String(ref words_file) => words_file.as_str(),
            _ => "",
        };

        let word_vecs_file = match config["word_vecs_file"] {
            Value::String(ref word_vecs_file) => word_vecs_file.as_str(),
            _ => "",
        };

        let use_word_vectors = match config["use_word_vectors"] {
            Value::Bool(use_words_vectors) => use_words_vectors,
            _ => panic!("Failed to parse use_words_vectors flag from the config!"),
        };

        Config {
            id_size: id_size as usize,
            bucket_size: bucket_size as usize,
            nr_shards: nr_shards as usize,
            shard_size: shard_size as usize,
            terms_relevance_file: terms_relevance_file.to_string(),
            synonyms_file: synonyms_file.to_string(),
            toponyms_file: toponyms_file.to_string(),
            stopwords_file: stopwords_file.to_string(),
            i2q_file: i2q_file.to_string(),
            words_file: words_file.to_string(),
            word_vecs_file: word_vecs_file.to_string(),
            use_word_vectors: use_word_vectors,
        }
    }
}
