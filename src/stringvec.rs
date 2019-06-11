/*
  Mmapped string vector (index -> string mapping):

  String vector file layout:

  [8B off_a][[6B off_0] [6B off_1]  ... [6B off_n][[string 1][string 2] ... [string n]]
  ^         ^                                     ^          ^                        ^
  |         |                                     |          |                        |
  0 ------- 8 ------------------------- [8 + off_a] --- [8 + off_a + off_1] --- [8 + off_a + off_n]

  - first 8 bytes (u64) are used to store size of the offsets' array (off_a):

        off_a = [number_of_strings + 1] * offset_size

    where offset_size is 6 bytes and the +1 slot is for the offset zero, set to 0 value

  - offsets start at byte [8] and end at byte [8 + off_a - 1]

  - offset for the first string (*off_1) is stored at the [off_1] address (off_0 stores value 0)

  - offset for the last string (*off_n) is stored at the [off_n] address

  - strings start at [8 + off_a] and end at [8 + off_a + off_n] address, that is also the EOF.


    write:
        // The comment bellow for the file string vec layout
        let mut str_vec_writer = StrVecWriter::init();
        for query in string_vec.into_iter() {
            str_vec_writer.add(query);
        }
        str_vec_writer.write_to_file(&file_path);

    read:
        // loads strings and their offsets from files
        let offsets = unsafe { memmap::Mmap::map(&File::open("./offsets.bin").unwrap()).unwrap() };
        let str_vec = StrVec::load("./queries.bin", &offsets);

    access:
        // string indexes start from 0, access the first string:
        str_vec[0] -> &str
        // access the 10th string
        str_vec[9] -> &str

 */

use std::fs::{read_dir, File, OpenOptions};
use std::io::prelude::*;
use std::io::{BufReader, BufWriter, Error, SeekFrom};

use std::path::Path;
use memmap::Mmap;
use shard::parse_query_line;
use pbr::ProgressBar;
use std::mem::size_of;
use byteorder::{ByteOrder, LittleEndian, WriteBytesExt};

use util;

pub static BRED: &str = "\x1B[0;31m";
pub static BYELLOW: &str = "\x1B[0;33m";
pub static END_COL: &str = "\x1B[0m";
pub static NOF_MSG: &str = "No such file or directory: ";

// determines address space size: e.g. for 100 billion strings,
// each string can on average be up to ~2.8KB (2^48/(100*10^9)) in size
pub const BYTES_PER_OFFSET: usize = 6;
pub const RAND_ALFANUM_LEN: usize = 24;

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

pub struct StrVec {
    strings_addr: usize,
    offsets: Vec<Offset>,
    strings: Mmap,
}

pub fn load<T: Clone>(buffer: &[u8]) -> Vec<T> {
    let vectors: &[T] = unsafe {
        ::std::slice::from_raw_parts(
            buffer.as_ptr() as *const T,
            buffer.len() / ::std::mem::size_of::<T>(),
        )
    };

    vectors.to_vec()
}

use std::str;
use std::ops::Index;
impl Index<usize> for StrVec {
    type Output = str;

    fn index(&self, idx: usize) -> &str {
        let ai: usize = self.strings_addr + usize::from(self.offsets[idx]);
        let aj: usize = self.strings_addr + usize::from(self.offsets[idx + 1]);

        &str::from_utf8(&self.strings[ai..aj]).unwrap()
    }
}

impl StrVec {
    pub fn load(path: &Path) -> Self {
        let mut buf = vec![0u8; size_of::<u64>()];

        // read offsets size
        let ref vec_file = OpenOptions::new().read(true).open(&path).unwrap();
        let mut handle = vec_file.take(size_of::<u64>() as u64);
        let mut bytes_read = handle.read(&mut buf).unwrap_or(0);
        assert!(
            bytes_read == size_of::<u64>(),
            "Failed to read offsets size from the file {:?}",
            path
        );
        let offsets_size = LittleEndian::read_u64(&buf);
        let strings_addr = bytes_read + offsets_size as usize;
        let mut offsets_data = vec![0u8; offsets_size as usize];
        handle = vec_file.take(offsets_size);
        bytes_read = handle.read(&mut offsets_data).unwrap_or(0);
        assert!(
            bytes_read == offsets_size as usize,
            "Failed to read offsets data from the file {:?}, bytes_read: {:?}, expected: {:?}",
            path,
            bytes_read,
            offsets_size
        );
        let offsets = load::<Offset>(&offsets_data[..]);

        let strings = unsafe {
            Mmap::map(&OpenOptions::new().read(true).open(path).expect(&[
                BYELLOW,
                NOF_MSG,
                END_COL,
                BRED,
                &path.to_str().unwrap(),
                END_COL,
            ].join("")))
                .unwrap()
        };
        util::advise_ram(&strings[..]).expect(&format!("Advisory failed for i2q {:?}", &path));

        StrVec {
            strings_addr: strings_addr,
            strings: strings,
            offsets: offsets,
        }
    }
}

pub struct StrVecWriter {
    tmp_off_name: String,
    tmp_str_name: String,
    off_writer: BufWriter<File>,
    str_writer: BufWriter<File>,
}

impl StrVecWriter {
    pub fn init() -> Self {
        let mut str_vec_writer = StrVecWriter::new();
        // add zero offset for an empty file
        str_vec_writer.add_offset();

        str_vec_writer
    }

    fn new() -> Self {
        let offsets_file = util::tmp_file_path(&"tmp_", &".bin", RAND_ALFANUM_LEN);
        let strings_file = util::tmp_file_path(&"tmp_", &".bin", RAND_ALFANUM_LEN);

        StrVecWriter {
            tmp_off_name: offsets_file.to_str().unwrap().to_string(),
            tmp_str_name: strings_file.to_str().unwrap().to_string(),
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
                    .open(strings_file)
                    .unwrap(),
            ),
        }
    }

    fn add_offset(&mut self) -> usize {
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

        offset
    }

    #[inline]
    pub fn add(&mut self, s: String) -> usize {
        // A race condition, if multiple writers access the same file.
        &self.str_writer
            .write_all(&s.as_bytes())
            .expect("Unable to write query text");
        &self.str_writer.flush().expect("String flush failed!");
        let _offset = self.add_offset();

        s.len()
    }

    #[inline]
    pub fn write_to_file(&mut self, out_file_path: &Path) -> u64 {
        let mut out_file = BufWriter::new(
            OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(out_file_path)
                .unwrap(),
        );

        let mut bytes_written: u64 = 0;

        // write offsets size first
        let mut buf = Vec::with_capacity(size_of::<u64>());
        let offsets_size = self.off_writer.seek(SeekFrom::End(0)).unwrap();
        buf.write_u64::<LittleEndian>(offsets_size).unwrap();
        out_file.write_all(buf.as_slice()).unwrap();
        out_file.flush().expect("Failed to flush offset size!");
        bytes_written += size_of::<u64>() as u64;

        // write offsets
        let mut offsets_file = OpenOptions::new()
            .read(true)
            .open(&self.tmp_off_name)
            .unwrap();

        let mut buf = Vec::with_capacity(offsets_size as usize);
        // Note: read_to_end appends data to a buffer
        let bytes_read = offsets_file.read_to_end(&mut buf).unwrap_or(0) as u64;
        assert!(
            bytes_read == offsets_size,
            "Failed to read offsets file, read {:?} out of {:?}",
            bytes_read,
            offsets_size,
        );
        out_file.write_all(&buf).unwrap();
        out_file.flush().expect("Failed to flush offsets!");
        bytes_written += bytes_read;

        // write strings
        let mut strings_file = OpenOptions::new()
            .read(true)
            .open(&self.tmp_str_name)
            .unwrap();

        // TODO assert the last offset == strings file size
        let strings_size = self.str_writer.seek(SeekFrom::End(0)).unwrap();
        let mut buf = Vec::with_capacity(strings_size as usize);
        let bytes_read = strings_file.read_to_end(&mut buf).unwrap_or(0) as u64;
        assert!(
            bytes_read == strings_size,
            "Failed to read offsets file, read {:?} out of {:?}",
            bytes_read,
            strings_size,
        );
        out_file.write_all(&buf).unwrap();
        out_file.flush().expect("Failed to flush strings!");
        bytes_written += bytes_read;

        bytes_written
    }
}

pub fn compile(queries_path: &str, out_file_path: &str) -> Result<(), Error> {
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

    let mut str_vec_writer = StrVecWriter::init();
    let mut pb = ProgressBar::new(query_files.len() as u64);
    for (file_name, query_file) in query_files.into_iter() {
        let reader = BufReader::with_capacity(5 * 1024 * 1024, query_file);
        for line in reader.lines() {
            let line = line.unwrap();
            let (_, query) = match parse_query_line(&line) {
                Ok((query_type, query)) => (query_type, query),
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
        }
        pb.inc();
    }
    pb.finish_print("done");

    let bytes_written = str_vec_writer.write_to_file(&Path::new(&out_file_path));
    println!("total bytes written: {:?}", bytes_written);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use fs2::FileExt;
    use std::env::temp_dir;

    #[test]
    fn test_queryvec_write_load() {
        let string_vec: Vec<String> = vec![
            "aaaaaaaaaa".to_string(),           // len: 10
            "ääääääääää".to_string(), // len: 20
            "bbbbbbbbbb".to_string(),           // len: 10
            "cccccccccc".to_string(),           // len: 10
            "ääääääääää".to_string(), // len: 20
        ];
        // write
        let vec_file_path = temp_dir().join("queries.bin");
        // lock
        let vec_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&vec_file_path)
            .unwrap();
        vec_file.lock_exclusive().unwrap();

        let mut str_vec_writer = StrVecWriter::init();
        for query in string_vec.into_iter() {
            str_vec_writer.add(query);
        }
        str_vec_writer.write_to_file(&vec_file_path);
        let bytes_read;
        let mut buf = vec![0u8; size_of::<u64>()];
        {
            // read offsets size
            let vec_file = OpenOptions::new().read(true).open(&vec_file_path).unwrap();
            let mut handle = vec_file.take(size_of::<u64>() as u64);
            bytes_read = handle.read(&mut buf).unwrap_or(0);
            assert_eq!(bytes_read, 8);
            assert!(
                bytes_read == size_of::<u64>(),
                "Failed to read offsets size from the file {:?}",
                vec_file_path
            );
        }
        let offsets_size = LittleEndian::read_u64(&buf);
        assert_eq!(offsets_size, 36);
        let strings_addr = bytes_read + offsets_size as usize;
        assert_eq!(strings_addr, 44);

        // unlock
        vec_file.unlock().unwrap();
    }

    #[test]
    fn test_queryvec_access() {
        let string_vec: Vec<String> = vec![
            "aaaaaaaaaa".to_string(),           // len: 10
            "ääääääääää".to_string(), // len: 20
            "bbbbbbbbbb".to_string(),           // len: 10
            "cccccccccc".to_string(),           // len: 10
            "ääääääääää".to_string(), // len: 20
        ];

        // write
        let vec_file_path = temp_dir().join("queries.bin");
        // lock
        let vec_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&vec_file_path)
            .unwrap();
        vec_file.lock_exclusive().unwrap();

        let mut str_vec_writer = StrVecWriter::init();
        for query in string_vec.into_iter() {
            str_vec_writer.add(query);
        }
        str_vec_writer.write_to_file(&vec_file_path);

        let str_vec = StrVec::load(&vec_file_path);
        assert_eq!("aaaaaaaaaa", &str_vec[0]);
        assert_eq!("ääääääääää", &str_vec[1]);
        assert_eq!("bbbbbbbbbb", &str_vec[2]);
        assert_eq!("ääääääääää", &str_vec[4]);

        // unlock
        vec_file.unlock().unwrap();
    }

    #[test]
    #[should_panic]
    fn test_index_out_of_bounds_panic() {
        let string_vec: Vec<String> = vec![
            "ääääääääää".to_string(),
            "bbbbbbbbbb".to_string(),
            "cccccccccc".to_string(),
            "ääääääääää".to_string(),
        ];

        // write
        let vec_file_path = temp_dir().join("queries.bin");
        let vec_file = OpenOptions::new().read(true).open(&vec_file_path).unwrap();
        vec_file.lock_exclusive().unwrap();

        let mut str_vec_writer = StrVecWriter::init();
        for query in string_vec.into_iter() {
            str_vec_writer.add(query);
        }
        str_vec_writer.write_to_file(&vec_file_path);

        let str_vec = StrVec::load(&vec_file_path);
        assert_eq!("aaaaaaaaaa", &str_vec[4]);

        // unlock
        vec_file.unlock().unwrap();
    }
}
