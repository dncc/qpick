/*
  Mmapped string vector (index to string mapping)

    read:
        // loads strings and their offsets from files
        let offsets = unsafe { memmap::Mmap::map(&File::open("./offsets.bin").unwrap()).unwrap() };
        let str_vec = StrVec::load("./queries.bin", &offsets);

    access:
        // string indexes start from 0, access to first query:
        str_vec[0] -> &str
        // access 10th query
        str_vec[9] -> &str
 */
use std::fs::{read_dir, File, OpenOptions};
use std::io::prelude::*;
use std::io::{BufReader, BufWriter, Error, SeekFrom};

use std::path::Path;
use memmap::Mmap;
use byteorder::{ByteOrder, LittleEndian};
use shard::parse_query_line;
use pbr::ProgressBar;

pub static BRED: &str = "\x1B[0;31m";
pub static BYELLOW: &str = "\x1B[0;33m";
pub static END_COL: &str = "\x1B[0m";
pub static NOF_MSG: &str = "No such file or directory: ";

// determines address space size: e.g. for 100 billion strings,
// each string can on average be up to ~2.8KB (2^48/(100*10^9)) in size
pub const BYTES_PER_OFFSET: usize = 6;

#[repr(C, packed)]
#[derive(Debug, Copy, Clone)]
pub struct Offset(pub [u8; BYTES_PER_OFFSET]);

impl From<usize> for Offset {
    fn from(integer: usize) -> Self {
        let mut data: [u8; BYTES_PER_OFFSET] = unsafe { ::std::mem::uninitialized() };
        LittleEndian::write_uint(&mut data, integer as u64, BYTES_PER_OFFSET);
        Offset(data)
    }
}

impl From<Offset> for usize {
    #[inline]
    fn from(offset: Offset) -> Self {
        LittleEndian::read_uint(&offset.0, BYTES_PER_OFFSET) as usize
    }
}

impl From<Offset> for u64 {
    #[inline]
    fn from(offset: Offset) -> Self {
        LittleEndian::read_uint(&offset.0, BYTES_PER_OFFSET) as u64
    }
}

impl Offset {
    #[inline]
    pub fn max_value() -> usize {
        let data = [<u8>::max_value(); BYTES_PER_OFFSET];
        usize::from(Offset(data))
    }

    #[inline]
    pub fn get_size() -> usize {
        BYTES_PER_OFFSET
    }
}

pub struct StrVec<'a> {
    offsets: &'a [Offset],
    strings: Mmap,
}

pub fn load<T>(buffer: &[u8]) -> &[T] {
    let vectors: &[T] = unsafe {
        ::std::slice::from_raw_parts(
            buffer.as_ptr() as *const T,
            buffer.len() / ::std::mem::size_of::<T>(),
        )
    };

    vectors
}

use std::str;
use std::ops::Index;
impl<'a> Index<usize> for StrVec<'a> {
    type Output = str;

    fn index(&self, idx: usize) -> &str {
        let ai: usize = self.offsets[idx].into();
        let aj: usize = self.offsets[idx + 1].into();

        &str::from_utf8(&self.strings[ai..aj]).unwrap()
    }
}

impl<'a> StrVec<'a> {
    pub fn load(path: &str, offset_data: &'a [u8]) -> Self {
        StrVec {
            strings: unsafe {
                Mmap::map(&OpenOptions::new()
                    .read(true)
                    .open(path)
                    .expect(&[BYELLOW, NOF_MSG, END_COL, BRED, &path, END_COL].join("")))
                    .unwrap()
            },
            offsets: load::<Offset>(&offset_data),
        }
    }
}

pub fn compile(queries_path: &str, output_dir: &str) -> Result<(), Error> {
    println!("compiling ...");

    let queries_path = &Path::new(&queries_path);
    let queries_parts = if queries_path.is_dir() {
        let mut parts: Vec<_> = read_dir(queries_path)
            .unwrap()
            .map(|p| p.unwrap().path())
            .filter(|p| p.to_str().unwrap().ends_with(".gz"))
            .collect();
        parts.sort();
        parts
    } else {
        vec![queries_path.to_path_buf()]
    };

    let query_files: Vec<(_, _)> = queries_parts
        .iter()
        .map(|query_part| {
            let query_file = File::open(&query_part).unwrap();
            (
                query_part.clone(),
                flate2::read::GzDecoder::new(query_file).expect("Not a valid gzip file."),
            )
        })
        .collect();

    let mut query_idx: u64 = 0;
    let mut str_vec_writer = StrVecWriter::new(output_dir);
    let mut pb = ProgressBar::new(query_files.len() as u64);
    for (file_name, query_file) in query_files.into_iter() {
        let reader = BufReader::with_capacity(5 * 1024 * 1024, query_file);
        for line in reader.lines() {
            let line = line.unwrap();
            let (_, _, query) = match parse_query_line(query_idx, &line) {
                Ok((query_id, query_type, query)) => (query_id, query_type, query),
                Err(e) => {
                    println!(
                        "Read error: {:?}, line: {:?}, file: {:?}",
                        e,
                        &line.clone(),
                        file_name
                    );
                    continue;
                }
            };
            str_vec_writer.add(query);
            query_idx += 1;
        }
        pb.inc();
    }
    str_vec_writer.complete();
    pb.finish_print("done");

    Ok(())
}

pub struct StrVecWriter {
    off_writer: BufWriter<File>,
    str_writer: BufWriter<File>,
}

impl StrVecWriter {
    pub fn new(output_dir: &str) -> Self {
        let offsets_file = format!("{}/offsets.bin", output_dir);
        let queries_file = format!("{}/queries.bin", output_dir);

        StrVecWriter {
            off_writer: BufWriter::new(
                OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(offsets_file)
                    .unwrap(),
            ),
            str_writer: BufWriter::new(
                OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(queries_file)
                    .unwrap(),
            ),
        }
    }

    #[inline]
    fn add(&mut self, s: String) -> usize {
        // A race condition, if multiple writers access the same file.
        let offset = self.str_writer.seek(SeekFrom::End(0)).unwrap() as usize;
        assert!(
            Offset::max_value() > offset,
            "File exceeds max byte size: {}",
            Offset::max_value()
        );
        &self.off_writer
            .write_all(&Offset::from(offset).0)
            .expect("Unable to query offset");
        &self.off_writer.flush().expect("Offset flush failed!");

        &self.str_writer
            .write_all(&s.as_bytes())
            .expect("Unable to write query text");
        &self.str_writer.flush().expect("String flush failed!");

        s.len()
    }

    #[inline]
    fn complete(&mut self) -> usize {
        // add offset of the last query
        self.add("".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env::temp_dir;
    use fs2::FileExt;
    #[test]
    fn test_queryvec_write_load_read() {
        let string_vec: Vec<String> = vec![
            "aaaaaaaaaa".to_string(),
            "ääääääääää".to_string(),
            "bbbbbbbbbb".to_string(),
            "cccccccccc".to_string(),
            "ääääääääää".to_string(),
        ];

        let output_dir = temp_dir();
        // lock
        let tmp_offsets_file =
            File::open(format!("{}/offsets.bin", output_dir.to_str().unwrap())).unwrap();
        tmp_offsets_file.lock_exclusive().unwrap();

        let tmp_queries_file =
            File::open(format!("{}/queries.bin", output_dir.to_str().unwrap())).unwrap();
        tmp_queries_file.lock_exclusive().unwrap();

        // write
        let mut str_vec_writer = StrVecWriter::new(output_dir.to_str().unwrap());

        for query in string_vec.into_iter() {
            str_vec_writer.add(query);
        }
        // write the offset of the last query
        str_vec_writer.complete();

        //load strings
        let offsets_file = format!("{}/offsets.bin", output_dir.to_str().unwrap());
        let offset_data = unsafe { memmap::Mmap::map(&File::open(offsets_file).unwrap()).unwrap() };
        let queries_file = format!("{}/queries.bin", output_dir.to_str().unwrap());
        let str_vec = StrVec::load(&queries_file, &offset_data);

        // read
        assert_eq!("aaaaaaaaaa", &str_vec[0]);
        assert_eq!("ääääääääää", &str_vec[1]);
        assert_eq!("bbbbbbbbbb", &str_vec[2]);
        assert_eq!("ääääääääää", &str_vec[4]);

        // unlock
        tmp_offsets_file.lock_exclusive().unwrap();
        tmp_queries_file.lock_exclusive().unwrap();
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds_panic() {
        let string_vec: Vec<String> = vec![
            "aaaaaaaaaa".to_string(),
            "ääääääääää".to_string(),
            "bbbbbbbbbb".to_string(),
            "cccccccccc".to_string(),
        ];

        let output_dir = temp_dir();
        // lock
        let tmp_offsets_file =
            File::open(format!("{}/offsets.bin", output_dir.to_str().unwrap())).unwrap();
        tmp_offsets_file.lock_exclusive().unwrap();

        let tmp_queries_file =
            File::open(format!("{}/queries.bin", output_dir.to_str().unwrap())).unwrap();
        tmp_queries_file.lock_exclusive().unwrap();

        // write
        let mut str_vec_writer = StrVecWriter::new(output_dir.to_str().unwrap());
        for query in string_vec.into_iter() {
            str_vec_writer.add(query);
        }
        str_vec_writer.complete();

        let output_dir = temp_dir();
        //load strings
        let offsets_file = format!("{}/offsets.bin", output_dir.to_str().unwrap());
        let offset_data = unsafe { memmap::Mmap::map(&File::open(offsets_file).unwrap()).unwrap() };
        let queries_file = format!("{}/queries.bin", output_dir.to_str().unwrap());
        let str_vec = StrVec::load(&queries_file, &offset_data);

        // read
        assert_eq!("aaaaaaaaaa", &str_vec[4]);

        // unlock
        tmp_offsets_file.lock_exclusive().unwrap();
        tmp_queries_file.lock_exclusive().unwrap();
    }
}
