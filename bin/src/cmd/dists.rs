use docopt::Docopt;

use qpick;

use std::result::Result;

use Error;

const USAGE: &'static str = "
Get distances for a query and query candidates.

Usage:
    qpick dists [options] <query> <candidates>
    qpick dists --help

Options:
    -h, --help         Arg query is a query string.
    -s, --start ARG    Shard to begin with.
    -e, --end ARG      Shard to end with goes together with the --start option.
";

#[derive(Debug, Deserialize)]
struct Args {
    flag_start: Option<u32>,
    flag_end: Option<u32>,
    arg_query: String,
    arg_candidates: String,
}

pub fn run(argv: Vec<String>) -> Result<(), Error> {
    let args: Args = Docopt::new(USAGE)
        .and_then(|d| d.argv(&argv).deserialize())
        .unwrap_or_else(|e| e.exit());

    let qpick: qpick::Qpick;

    println!("{:?}", args);
    if let Some(start_shard) = args.flag_start {
        if let Some(end_shard) = args.flag_end {
            assert!(end_shard > start_shard);
            qpick = qpick::Qpick::from_path_with_shard_range(
                "./index".to_string(),
                start_shard..end_shard,
            );
        } else {
            panic!("Missing the end shard value! Run --help for more info!")
        }
    } else {
        qpick = qpick::Qpick::from_path("./index".to_string());
    }

    let candidates: Vec<String> = args.arg_candidates
        .split(",")
        .map(|x| x.trim().to_string())
        .collect::<Vec<String>>();

    let r = qpick.get_distances(&args.arg_query, &candidates);
    let v: Vec<(String, f32)> = r.into_iter().map(|dr| (dr.query, dr.dist)).collect();
    println!("{:?}", v);

    Ok(())
}
