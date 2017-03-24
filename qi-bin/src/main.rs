#![allow(dead_code)]

extern crate fst;
extern crate docopt;
extern crate qi;
extern crate serde;
extern crate serde_json;
extern crate rustc_serialize;

use std::env;
use std::error;
use std::process;
use std::io::{self, Write};

use docopt::Docopt;

macro_rules! w {
    ($wtr:expr, $($tt:tt)*) => {{
        use std::io::Write;
        let _ = writeln!(&mut $wtr, $($tt)*);
    }}
}

macro_rules! fail {
    ($($tt:tt)*) => { return Err(From::from(format!($($tt)*))); }
}

mod cmd;

pub type Error = Box<error::Error + Send + Sync>;

const USAGE: &'static str = "
Usage:
    mann <command> [<args>...]
    mann --help
    mann --version
Commands:
    index   Create ANN index.
    get     Get ANN matches for the given item.
Options:
    -h, --help     Show this help message.
    -v, --version  Show version.
";

#[derive(Debug, RustcDecodable)]
struct Args {
    arg_command: Option<Command>,
}

#[derive(Debug, RustcDecodable)]

enum Command {
    // Index,
    Get,
}

impl Command {
    fn run(self) -> Result<(), Error> {
        use self::Command::*;

        let argv: Vec<String> = env::args().collect();
        match self {
            // Index => cmd::index::run(argv),
            Get => cmd::get::run(argv),
        }
    }
}

fn main() {
    let args: Args = Docopt::new(USAGE)
                            .and_then(|d| d.options_first(true)
                                           .version(Some(version()))
                                           .decode())
                            .unwrap_or_else(|e| e.exit());
    let cmd = args.arg_command.expect("BUG: expected a command");
    if let Err(err) = cmd.run() {
        writeln!(&mut io::stderr(), "{}", err).unwrap();
        process::exit(1);
    }
}

fn version() -> String {
    let (maj, min, pat) = (
        option_env!("CARGO_PKG_VERSION_MAJOR"),
        option_env!("CARGO_PKG_VERSION_MINOR"),
        option_env!("CARGO_PKG_VERSION_PATCH"),
    );
    match (maj, min, pat) {
        (Some(maj), Some(min), Some(pat)) =>
            format!("{}.{}.{}", maj, min, pat),
        _ => "N/A".to_owned(),
    }
}
