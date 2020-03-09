use fst::Set;
use std::path::Path;

use util::{BRED, BYELL, ECOL};

pub fn load(path: &Path) -> Option<fst::Set> {
    if !path.is_file() {
        return None;
    }

    match Set::from_path(path) {
        Ok(toponyms) => Some(toponyms),
        Err(_) => {
            println!(
                "{}",
                [
                    BYELL,
                    "No such file: ",
                    ECOL,
                    BRED,
                    path.to_str().unwrap(),
                    ECOL
                ]
                .join("")
            );

            None
        }
    }
}
