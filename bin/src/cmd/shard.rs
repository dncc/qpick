use docopt::Docopt;
use Error;

use qpick;

const USAGE: &'static str = "
Get vector ids and scores for ANN.
Usage:
    qpick shard [options] <path> <nr-shards> <output-dir>
    qpick shard --help
Options:
    -h, --help  path: is a file path to queries input file.
                nr-shards: how many shards to create.
                ouput-dir: where to save shard files
    -c, --concurrency ARG number of concurrent shard processes,
                      default value is nr-shards arg

";

#[derive(Debug, Deserialize)]
struct Args {
    arg_path: String,
    arg_nr_shards: usize,
    arg_output_dir: String,
    flag_concurrency: Option<u32>,
}

pub fn run(argv: Vec<String>) -> Result<(), Error> {
let args: Args = Docopt::new(USAGE)
    .and_then(|d| d.argv(&argv).deserialize())
    .unwrap_or_else(|e| e.exit());

    let mut concurrency = args.arg_nr_shards;
    if let Some(c) = args.flag_concurrency {
        concurrency = c as usize;
    }

    let r = qpick::Qpick::shard(args.arg_path, args.arg_nr_shards, args.arg_output_dir, concurrency);
    println!("{:?}", r);

    Ok(())

}
