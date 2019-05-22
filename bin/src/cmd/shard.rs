use docopt::Docopt;
use Error;

use qpick::shard;

const USAGE: &'static str = "
Creates ngram shards from an input directory.
Usage:
    qpick shard [options] <path> <nr-shards> <output-dir> <prefixes> <concurrency>
    qpick shard --help
Options:
    -h, --help  path: is a directory path to query files (.gz).
                nr-shards: how many shards to create.
                ouput-dir: where to save shard files
";

#[derive(Debug, Deserialize)]
struct Args {
    arg_path: String,
    arg_nr_shards: usize,
    arg_output_dir: String,
    arg_prefixes: Option<String>,
    arg_concurrency: Option<u32>,
}

pub fn run(argv: Vec<String>) -> Result<(), Error> {
    let args: Args = Docopt::new(USAGE)
        .and_then(|d| d.argv(&argv).deserialize())
        .unwrap_or_else(|e| e.exit());

    let mut concurrency = args.arg_nr_shards;
    if let Some(c) = args.arg_concurrency {
        concurrency = c as usize;
    }

    let prefixes = match args.arg_prefixes {
        Some(prefs) => prefs,
        None => "".to_string(),
    };

    let prefixes: Vec<String> = prefixes
        .split(",")
        .map(|x| x.trim().to_string())
        .filter(|x| x != "")
        .collect::<Vec<String>>();

    println!("{:?}", prefixes);

    let r = shard::shard(
        &args.arg_path,
        args.arg_nr_shards,
        &args.arg_output_dir,
        concurrency,
        &prefixes,
    );
    println!("{:?}", r);

    Ok(())
}
