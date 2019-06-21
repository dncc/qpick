use docopt::Docopt;
use Error;

use qpick::shard;

const USAGE: &'static str = "
Creates ngram shards from an input directory.
Usage:
    qpick shard <path> <nr-shards> <output-dir> <prefixes> [--without-i2q]
    qpick shard --help
Options:
    -h, --help  path: is a directory path to query files (.gz).
                nr-shards: how many shards to create.
                ouput-dir: where to save shard files
                prefixes: csv list of prefixes (e.g. 'q, qe, tuw')
                         determining which queries to shard, if not
                         provided (default) it shards everything
                without-i2q: whether of not to compile i2q index
                        (integer to query mapping), if not provided
                        (default) it will create it
";

#[derive(Debug, Deserialize)]
struct Args {
    arg_path: String,
    arg_nr_shards: usize,
    arg_output_dir: String,
    arg_prefixes: Option<String>,
    flag_without_i2q: bool,
}

pub fn run(argv: Vec<String>) -> Result<(), Error> {
    let args: Args = Docopt::new(USAGE)
        .and_then(|d| d.argv(&argv).deserialize())
        .unwrap_or_else(|e| e.exit());

    let prefixes = match args.arg_prefixes {
        Some(prefs) => prefs,
        None => "".to_string(),
    };

    println!("--without-i2q: {:?}", args.flag_without_i2q);

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
        &prefixes,
        !args.flag_without_i2q,
    );
    println!("{:?}", r);

    Ok(())
}
