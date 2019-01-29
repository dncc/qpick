#![allow(dead_code)]

extern crate docopt;
extern crate fst;
extern crate qpick;
extern crate serde;
#[macro_use]
extern crate serde_derive;

use std::env;
use std::error;
use std::process;
use std::io::{self, Write};

use docopt::Docopt;

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

#[derive(Debug, Deserialize)]
struct Args {
    arg_command: Option<Command>,
}

#[derive(Debug, Deserialize)]
enum Command {
    Get,
    Shard,
    Index,
    Merge,
    Dists,
}

impl Command {
    fn run(self) -> Result<(), Error> {
        use self::Command::*;

        let argv: Vec<String> = env::args().collect();
        match self {
            Get => cmd::get::run(argv),
            Shard => cmd::shard::run(argv),
            Index => cmd::index::run(argv),
            Merge => cmd::merge::run(argv),
            Dists => cmd::dists::run(argv),
        }
    }
}

fn main() {
    let args: Args = Docopt::new(USAGE)
        .and_then(|d| d.options_first(true).version(Some(version())).deserialize())
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
        (Some(maj), Some(min), Some(pat)) => format!("{}.{}.{}", maj, min, pat),
        _ => "N/A".to_owned(),
    }
}
