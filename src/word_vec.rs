use byteorder::{ByteOrder, LittleEndian};
use fnv::{FnvHashMap, FnvHashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};

use blas;

use ngrams::MISS_WORD_REL;
use std::mem::MaybeUninit;
use util;

pub const DIM: usize = 201; // TODO in the config!
pub const UPPER_COS_BOUND: f32 = 1.0;

#[inline]
pub fn normalize(v: &mut [f32]) {
    unsafe {
        let norm: f32 = blas::snrm2(DIM as i32, v, 1);
        blas::sscal(DIM as i32, 1.0 / norm, v, 1)
    }
}

#[inline]
pub fn dot(v: &[f32], u: &[f32]) -> f32 {
    unsafe { blas::sdot(DIM as i32, v, 1, u, 1) }
}

#[inline]
pub fn cosine_distance(u: &Vec<f32>, v: &Vec<f32>) -> f32 {
    1.0 - util::max(0.0, util::min(UPPER_COS_BOUND, dot(&u, &v)))
}

#[inline]
pub fn subtract(mut v: &mut [f32], u: &[f32]) {
    unsafe { blas::saxpy(DIM as i32, -1f32, u, 1, &mut v, 1) }
}

// --- word dict and word vec
pub struct WordDict {
    word_to_id: FnvHashMap<String, usize>,
}

impl WordDict {
    pub fn load(path: &str) -> Self {
        let word_file = BufReader::new(File::open(&path).unwrap());
        let word_to_id = word_file
            .lines()
            .enumerate()
            .map(|(i, w)| {
                let w = w.unwrap();
                (serde_json::from_str::<String>(&w).unwrap(), i)
            })
            .collect::<FnvHashMap<String, usize>>();

        Self {
            word_to_id: word_to_id,
        }
    }

    #[inline]
    pub fn get_word_rel(
        self: &Self,
        word: &str,
        words_relevances: &fst::Map,
        stopwords: &FnvHashSet<String>,
    ) -> f32 {
        let word_rel: f32 = words_relevances.get(word).unwrap_or(MISS_WORD_REL) as f32;

        if stopwords.contains(word) || word.len() == 1 {
            return 0.25 * word_rel;
        }

        word_rel
    }

    #[inline]
    pub fn get_word_id(self: &Self, word: &str) -> Option<&usize> {
        self.word_to_id.get(word)
    }

    #[inline]
    pub fn get_words_ids(
        self: &Self,
        words: &Vec<String>,
        words_relevances: &fst::Map,
        stopwords: &FnvHashSet<String>,
    ) -> Vec<(usize, f32)> {
        words
            .iter()
            .filter_map(|w| match self.word_to_id.get(w) {
                Some(word_id) => {
                    let word_rel = self.get_word_rel(w, words_relevances, stopwords);

                    Some((*word_id, word_rel))
                }
                None => None,
            })
            .collect()
    }

    #[inline]
    pub fn len(self: &Self) -> usize {
        self.word_to_id.len()
    }
}

const BYTES_PER_WORD_ID: usize = 3;

#[repr(C, packed)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct WordId([u8; BYTES_PER_WORD_ID]);

impl From<usize> for WordId {
    #[inline(always)]
    fn from(integer: usize) -> Self {
        let mut data: [u8; BYTES_PER_WORD_ID] = unsafe { MaybeUninit::uninit().assume_init() };
        LittleEndian::write_uint(&mut data, integer as u64, BYTES_PER_WORD_ID);
        WordId(data)
    }
}

impl Into<usize> for WordId {
    #[inline(always)]
    fn into(self: Self) -> usize {
        LittleEndian::read_uint(&self.0, BYTES_PER_WORD_ID) as usize
    }
}

#[repr(C)]
pub struct WordVec([f32; DIM]);

pub struct WordVecs<'a> {
    word_dict: WordDict,
    word_vecs: &'a [WordVec],
    _data: memmap::Mmap,
}

impl<'a> WordVecs<'a> {
    pub fn load(words_file: &str, word_vecs_file: &str) -> Self {
        let word_dict = WordDict::load(words_file);

        let file = File::open(word_vecs_file).unwrap();
        let data = unsafe { memmap::Mmap::map(&file).unwrap() };
        assert_eq!(0, data.len() % ::std::mem::size_of::<WordVec>());
        let word_vecs: &[WordVec] = unsafe {
            ::std::slice::from_raw_parts(
                data.as_ptr() as *const WordVec,
                data.len() / ::std::mem::size_of::<WordVec>(),
            )
        };

        Self {
            word_dict: word_dict,
            word_vecs: word_vecs,
            _data: data,
        }
    }

    #[inline]
    pub fn get_vec(self: &Self, word: &str) -> Vec<f32> {
        if let Some(word_id) = self.word_dict.get_word_id(word) {
            return self.word_vecs[*word_id].0.to_vec();
        }

        vec![]
    }

    #[inline]
    pub fn get_combined_vec(
        self: &Self,
        words: &Vec<String>,
        words_relevances: &fst::Map,
        stopwords: &FnvHashSet<String>,
    ) -> (Vec<f32>, usize) {
        let word_ids_rels = self
            .word_dict
            .get_words_ids(words, words_relevances, stopwords);

        // let mult: bool = word_ids_rels.len() > 1;
        let not_found = words.len() - word_ids_rels.len();
        let words_vec = word_ids_rels
            .iter()
            .fold([0.0; DIM], |mut data, (i, word_rel)| {
                let WordVec(ref v) = self.word_vecs[*i];
                unsafe {
                    blas::saxpy(
                        DIM as i32, // if mult { *word_rel } else { 1f32 },
                        *word_rel, v, 1, &mut data, 1,
                    )
                }

                data
            })
            .to_vec();

        // if mult {
        // normalize(&mut words_vec);
        // }

        (words_vec, not_found)
    }

    #[inline]
    pub fn len(self: &Self) -> usize {
        self.word_vecs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distributions::Alphanumeric;
    use rand::Rng;
    use std::env::temp_dir;
    use std::fs::File;
    use std::io::Write;
    use std::str;

    fn get_random_words(num: usize, word_len: usize) -> Vec<String> {
        let mut rng = rand::thread_rng();

        (0..num)
            .map(|_| {
                let mut buf = String::with_capacity(word_len + 1);
                buf.push_str("\"");
                unsafe {
                    rng.sample_iter(&Alphanumeric)
                        .take(word_len)
                        .for_each(|b| buf.push_str(str::from_utf8_unchecked(&[b as u8])));
                }
                buf.push_str("\"\n");

                buf
            })
            .collect()
    }

    fn get_random_word_vecs(num: usize) -> Vec<WordVec> {
        let mut rng = rand::thread_rng();

        (0..num)
            .map(|_| {
                let mut vec = [0.0f32; DIM];
                for f in &mut vec[..] {
                    *f = rng.gen();
                }

                WordVec(vec)
            })
            .collect()
    }

    #[test]
    fn test_words_write_load() {
        let mut buffer: Vec<u8> = Vec::new();
        let num_words = 3;
        let words_len = 5;

        let mut rnd_bytes: Vec<u8> = Vec::with_capacity(num_words * words_len);
        for rnd_word in get_random_words(num_words, words_len).into_iter() {
            rnd_bytes.extend(rnd_word.into_bytes());
        }

        let mut tmp_words_file = temp_dir();
        tmp_words_file.push("test_words");
        {
            let mut file = File::create(&tmp_words_file).unwrap();
            file.write_all(&rnd_bytes[..]).unwrap();
        }

        buffer.resize(::std::mem::size_of::<WordVec>() * num_words, 0u8);
        let rnd_word_vecs = get_random_word_vecs(num_words);
        // reinterpret as [u8] in order to write to buffer
        buffer.copy_from_slice(unsafe {
            ::std::slice::from_raw_parts(
                rnd_word_vecs.as_ptr() as *const u8,
                rnd_word_vecs.len() * ::std::mem::size_of::<WordVec>(),
            )
        });

        let mut tmp_word_vecs_file = temp_dir();
        tmp_word_vecs_file.push("test_word_embeddings");
        {
            let mut file = File::create(&tmp_word_vecs_file).unwrap();
            file.write_all(&buffer).unwrap();
        }

        let wv = WordVecs::load(
            tmp_words_file.to_str().unwrap(),
            tmp_word_vecs_file.to_str().unwrap(),
        );

        assert_eq!(num_words, wv.word_vecs.len());
        assert_eq!(num_words, wv.word_dict.len());
    }
}
