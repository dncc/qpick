use std::io;

use docopt::Docopt;

use qpick;
use fst::Map;

use std::io::Read;
use std::fs::File;
use std::result::Result;

use serde_json;
use std::fs::OpenOptions;

use Error;

const USAGE: &'static str = "
Get vector ids and scores for ANN.
Usage:
    qpick get [options] <query>
    qpick get --help
Options:
    -h, --help  Arg query is a query string.
";

#[derive(Debug, RustcDecodable)]
struct Args {
    arg_query: String,
}

pub fn run(argv: Vec<String>) -> Result<(), Error> {
    let args: Args = Docopt::new(USAGE)
                            .and_then(|d| d.argv(&argv).decode())
                            .unwrap_or_else(|e| e.exit());

    let qpick = qpick::Qpick::from_path("./index".to_string());
    let r = qpick.search(&args.arg_query);
    println!("{:?}", r);

    Ok(())
}
