use std::io;

use docopt::Docopt;
use Error;

use qpick;

const USAGE: &'static str = "
Get vector ids and scores for ANN.
Usage:
    mann merge [options] <path>
    mann merge --help
Options:
    -h, --help  Arg qurey is a query string.
";

#[derive(Debug, RustcDecodable)]
struct Args {
    arg_path: String,
}

pub fn run(argv: Vec<String>) -> Result<(), Error> {
    let args: Args = Docopt::new(USAGE)
        .and_then(|d| d.argv(&argv).decode())
        .unwrap_or_else(|e| e.exit());

    let qpick = qpick::Qpick::from_path(args.arg_path);
    let r = qpick.merge();
    println!("{:?}", r);

    Ok(())

}
