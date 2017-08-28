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

";

#[derive(Debug, RustcDecodable)]
struct Args {
    arg_path: String,
    arg_nr_shards: usize,
    arg_output_dir: String,
}

pub fn run(argv: Vec<String>) -> Result<(), Error> {
    let args: Args = Docopt::new(USAGE)
        .and_then(|d| d.argv(&argv).decode())
        .unwrap_or_else(|e| e.exit());

    let r = qpick::Qpick::shard(args.arg_path, args.arg_nr_shards, args.arg_output_dir);
    println!("{:?}", r);

    Ok(())

}
