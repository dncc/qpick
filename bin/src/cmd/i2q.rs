use docopt::Docopt;
use Error;

use qpick::stringvec;

const USAGE: &'static str = "
Creates an index-to-query mapping
Usage:
    qpick i2q [options] <path> <output-dir>
    qpick i2q --help
Options:
    -h, --help  path: is an input directory.
                ouput-dir: where to save i2q index.
";

#[derive(Debug, Deserialize)]
struct Args {
    arg_path: String,
    arg_output_dir: String,
}

pub fn run(argv: Vec<String>) -> Result<(), Error> {
    let args: Args = Docopt::new(USAGE)
        .and_then(|d| d.argv(&argv).deserialize())
        .unwrap_or_else(|e| e.exit());

    let r = stringvec::compile(&args.arg_path, &args.arg_output_dir);
    println!("{:?}", r);

    Ok(())
}
