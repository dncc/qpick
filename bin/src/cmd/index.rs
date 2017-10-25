use docopt::Docopt;
use Error;

use qpick;

const USAGE: &'static str = "
Get vector ids and scores for ANN.
Usage:
    qpick index [options] <input-dir> <shard-name> <first-shard> <last-shard> <output-dir>
    qpick index --help
Options:
    -h, --help  input-dir: a file path to sharded 'queries', bunch of txt files
                shard-name: a file name prefix for a shard file,
                           e.g. 'queries'
                first-shard: starting shard to compile, see bellow last-shard
                last-shard: last shard to compile, together with first-shard creates a
                            non inclusive range, e.g. if first-shard is 0 and last-shard is 2,
                            it will compile 2 shards: queries.0 and queries.1
                output-dir: where to save compiled shards

";

#[derive(Debug, Deserialize)]
struct Args {
    arg_input_dir: String,
    arg_shard_name: String,
    arg_first_shard: usize,
    arg_last_shard: usize,
    arg_output_dir: String,
}

pub fn run(argv: Vec<String>) -> Result<(), Error> {
let args: Args = Docopt::new(USAGE)
    .and_then(|d| d.argv(&argv).deserialize())
    .unwrap_or_else(|e| e.exit());

    let r = qpick::Qpick::index(
        args.arg_input_dir,
        args.arg_shard_name,
        args.arg_first_shard,
        args.arg_last_shard,
        args.arg_output_dir);

    println!("{:?}", r);

    Ok(())

}
