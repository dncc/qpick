use docopt::Docopt;

use qpick;

use std::result::Result;

use Error;

const USAGE: &'static str = "
Get vector ids and scores for ANN.
Usage:
    qpick get [options] <query> <count>
    qpick get --help
Options:
    -h, --help  Arg query is a query string.
";

#[derive(Debug, RustcDecodable)]
struct Args {
    arg_query: String,
    arg_count: u32,
}

pub fn run(argv: Vec<String>) -> Result<(), Error> {
    let args: Args = Docopt::new(USAGE)
                            .and_then(|d| d.argv(&argv).decode())
                            .unwrap_or_else(|e| e.exit());

    let qpick = qpick::Qpick::from_path("./index".to_string());
    let r = qpick.get(&args.arg_query, args.arg_count);
    let v: Vec<(u64, f32)> = r.items_iter.map(|(id, sc)| (id, sc)).collect();
    println!("{:?}", v);

    Ok(())
}
