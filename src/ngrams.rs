use fnv::{FnvHashMap, FnvHashSet};

use util;

use regex::Regex;
use std::borrow::Cow;

const MISS_WORD_REL: u64 = 6666;
pub const WORDS_PER_QUERY: usize = 15;

const PUNCT_SYMBOLS: &str = "[/@#!,'?:();.+]";

macro_rules! update {
    ($ngrams: ident,
     $relevs: ident,
     $ngrams_ids: ident,
     $ngram: expr,
     $relev: expr,
     $indices: expr) => {
        $relevs.push($relev);
        $ngrams.push($ngram.clone());
        $ngrams_ids.insert($ngram.clone(), $indices);
    };
}

macro_rules! vec_push_str {
    // Base case:
    ($v:ident, $w:expr) => (
        $v.push_str($w);
    );

    ($v:ident, $w1:expr, $($w2:expr),+) => (
        $v.push_str($w1);
        $v.push_str(" ");
        vec_push_str!($v, $($w2),+);
    )
}

macro_rules! bow2 {
    ($v: ident, $w1: expr, $w2: expr) => {{
        if $w1 < $w2 {
            vec_push_str!($v, $w1, $w2);
        } else {
            vec_push_str!($v, $w2, $w1);
        }

        $v
    }};
}

#[inline]
fn bow2(w1: &str, w2: &str) -> String {
    let mut v = String::with_capacity(w1.len() + w2.len() + 1);

    bow2!(v, w1, w2)
}

#[inline]
fn bow3(w1: &str, w2: &str, w3: &str) -> String {
    let mut v = String::with_capacity(w1.len() + w2.len() + w3.len() + 2);
    if w1 < w2 && w1 < w3 {
        v.push_str(w1);
        v.push_str(" ");

        return bow2!(v, w2, w3);
    } else if w2 < w1 && w2 < w3 {
        v.push_str(w2);
        v.push_str(" ");

        return bow2!(v, w1, w3);
    } else {
        v.push_str(w3);
        v.push_str(" ");

        return bow2!(v, w1, w2);
    }
}

fn u8_find_and_replace<'a, S: Into<Cow<'a, str>>>(input: S) -> Cow<'a, str> {
    lazy_static! {
        static ref PUNCT_RE: Regex = Regex::new(PUNCT_SYMBOLS).unwrap();
    }
    let input = input.into();
    if let Some(mat) = PUNCT_RE.find(&input) {
        let start = mat.start();
        let len = input.len();
        let mut output: Vec<u8> = Vec::with_capacity(len + len / 2);
        output.extend_from_slice(input[0..start].as_bytes());
        let rest = input[start..].bytes();
        for c in rest {
            match c {
                b'!' | b',' | b'?' | b':' => (),
                b'#' | b'@' | b'(' | b')' | b';' | b'.' | b'/' | b'\'' | b'+' => {
                    output.extend_from_slice(b" ")
                }
                _ => output.push(c),
            }
        }
        Cow::Owned(unsafe { String::from_utf8_unchecked(output) })
    } else {
        input
    }
}

#[inline]
pub fn normalize(query: &str) -> String {
    u8_find_and_replace(query).trim().to_lowercase()
}

#[inline]
pub fn get_word_relevances(
    words: &Vec<String>,
    tr_map: &fst::Map,
    stopwords: &FnvHashSet<String>,
) -> (Vec<usize>, Vec<usize>, Vec<f32>) {
    let words_len = words.len();
    let mut rels: Vec<f32> = Vec::with_capacity(words_len);
    let mut stop_vec: Vec<usize> = Vec::with_capacity(words_len);
    let mut word_vec: Vec<usize> = Vec::with_capacity(words_len);

    let mut min_rel = std::f32::MAX;
    let mut min_word_idx: usize = 0;
    let word_thresh: f32 = 1.0 / (1 + words_len) as f32;

    for (i, word) in words.iter().enumerate() {
        let mut rel = tr_map.get(word).unwrap_or(MISS_WORD_REL) as f32;
        if stopwords.contains(word) || word.len() == 1 {
            rel = 0.5 * rel;
            stop_vec.push(i);
        } else {
            word_vec.push(i);
        }
        rels.push(rel);

        if rel < min_rel {
            min_rel = rel;
            min_word_idx = i;
        }
    }

    let norm: f32 = rels.iter().fold(0.0, |s, r| s + r);
    let rels: Vec<f32> = rels.into_iter().map(|r| r / norm).collect();

    if stop_vec.is_empty() && words_len > 3 && rels[min_word_idx] < word_thresh {
        stop_vec.push(min_word_idx);
        word_vec.remove(min_word_idx);
    }

    (word_vec, stop_vec, rels)
}

#[inline]
pub fn get_stop_ngrams(
    words: &Vec<String>,
    rels: &Vec<f32>,
    word_idx: &mut Vec<usize>,
    stop_idx: &Vec<usize>,
) -> Vec<(String, f32, Vec<usize>)> {
    let words_len = words.len();
    let mut stop_ngrams: Vec<(String, f32, Vec<usize>)> = Vec::with_capacity(words_len);
    let last_word_idx = words_len - 1;

    word_idx.reverse();

    let stop_idx_set: FnvHashSet<usize> = stop_idx.iter().cloned().collect();
    let mut skip_idx: FnvHashSet<usize> = FnvHashSet::default();
    let mut linked_idx: FnvHashSet<usize> = FnvHashSet::default();

    for i in stop_idx.iter() {
        if skip_idx.contains(i) {
            continue;
        }

        // begins with stopword
        if *i == 0 {
            let j = *i + 1;
            skip_idx.insert(j);
            linked_idx.insert(*i);
            linked_idx.insert(j);
            if !stop_idx_set.contains(&j) || j == last_word_idx {
                stop_ngrams.push((bow2(&words[*i], &words[j]), rels[*i] + rels[j], vec![*i, j]));
            } else {
                linked_idx.insert(j + 1);
                stop_ngrams.push((
                    bow3(&words[*i], &words[j], &words[j + 1]),
                    rels[*i] + rels[j] + rels[j + 1],
                    vec![*i, j, j + 1],
                ));
            }

        // stopword in between
        } else if *i < last_word_idx {
            let j = *i - 1;
            let k = *i + 1;

            // push all single-no-stop words, TODO macro!
            if !word_idx.is_empty() {
                let mut next_i = word_idx.pop().unwrap();
                while next_i < j && !linked_idx.contains(&next_i) {
                    stop_ngrams.push((words[next_i].to_string(), rels[next_i], vec![next_i]));
                    next_i = word_idx.pop().unwrap();
                }
            }

            // only k is a stopword
            if !stop_idx_set.contains(&j) && stop_idx_set.contains(&k) {
                if !linked_idx.contains(&j) {
                    linked_idx.insert(*i);
                    linked_idx.insert(j);
                    stop_ngrams.push((
                        bow2(&words[j], &words[*i]),
                        rels[j] + rels[*i],
                        vec![j, *i],
                    ));
                } else {
                    skip_idx.insert(k);
                    linked_idx.insert(k);
                    linked_idx.insert(*i);
                    stop_ngrams.push((
                        bow2(&words[*i], &words[k]),
                        rels[*i] + rels[k],
                        vec![*i, k],
                    ));
                }

            // only j is a stopword
            } else if stop_idx_set.contains(&j) && !stop_idx_set.contains(&k) {
                linked_idx.insert(k);
                linked_idx.insert(*i);
                stop_ngrams.push((bow2(&words[*i], &words[k]), rels[*i] + rels[k], vec![*i, k]));

            // both j & k are stopwords, since j is linked, take k
            } else if stop_idx_set.contains(&j) && stop_idx_set.contains(&k) {
                skip_idx.insert(k);
                linked_idx.insert(k);
                linked_idx.insert(*i);
                if k == last_word_idx || stop_idx_set.contains(&(k + 1)) {
                    stop_ngrams.push((
                        bow2(&words[*i], &words[k]),
                        rels[*i] + rels[k],
                        vec![*i, k],
                    ));

                // take also k+1 if k is not the last word
                } else {
                    skip_idx.insert(k + 1);
                    linked_idx.insert(k + 1);
                    stop_ngrams.push((
                        bow3(&words[*i], &words[k], &words[k + 1]),
                        rels[*i] + rels[k] + rels[k + 1],
                        vec![*i, k, k + 1],
                    ));
                }

            // neither j, nor k are stopwords
            } else {
                if rels[j] <= rels[k] && !linked_idx.contains(&j) {
                    linked_idx.insert(*i);
                    linked_idx.insert(j);
                    stop_ngrams.push((
                        bow2(&words[j], &words[*i]),
                        rels[j] + rels[*i],
                        vec![j, *i],
                    ));
                } else {
                    linked_idx.insert(k);
                    linked_idx.insert(*i);
                    if !linked_idx.contains(&j) {
                        stop_ngrams.push((words[j].to_string(), rels[j], vec![j]));
                        linked_idx.insert(j);
                    }
                    stop_ngrams.push((
                        bow2(&words[*i], &words[k]),
                        rels[*i] + rels[k],
                        vec![*i, k],
                    ));
                }
            }

        // ends with stopword
        } else {
            let j = *i - 1;
            // push all single-no-stop words, TODO macro
            if !word_idx.is_empty() {
                let mut next_i = word_idx.pop().unwrap();
                while next_i < j && !linked_idx.contains(&next_i) {
                    stop_ngrams.push((words[next_i].to_string(), rels[next_i], vec![next_i]));
                    next_i = word_idx.pop().unwrap();
                }
            }

            if !linked_idx.contains(&j) {
                linked_idx.insert(*i);
                linked_idx.insert(j);
                stop_ngrams.push((bow2(&words[j], &words[*i]), rels[j] + rels[*i], vec![j, *i]));

            // previous word is in ngram, add this word to it and exit
            } else {
                linked_idx.insert(*i);
                let (ngram, mut ngram_rel, mut ngram_idx) = stop_ngrams.pop().unwrap();
                ngram_rel += rels[*i];
                ngram_idx.push(*i);
                stop_ngrams.push((bow2(&ngram, &words[*i]), ngram_rel, ngram_idx));
            }
        }
    }

    while let Some(next_i) = word_idx.pop() {
        if !linked_idx.contains(&next_i) {
            stop_ngrams.push((words[next_i].to_string(), rels[next_i], vec![next_i]));
        }
    }

    stop_ngrams
}

use std::cmp::{Ordering, PartialOrd};

#[inline]
fn get_norm_query_vec(query: &str) -> Vec<String> {
    let mut words = normalize(query)
        .split(" ")
        .map(|w| w.trim().to_string())
        .filter(|w| w.len() > 1 || !w.is_empty() && char::is_numeric(w.chars().next().unwrap()))
        .collect::<Vec<String>>();
    words.truncate(WORDS_PER_QUERY);

    words
}

#[inline]
pub fn parse(
    query: &str,
    stopwords: &FnvHashSet<String>,
    tr_map: &fst::Map,
) -> (
    Vec<String>,
    Vec<f32>,
    FnvHashMap<String, Vec<usize>>,
    Vec<String>,
    Vec<f32>,
) {
    let mut ngrams_relevs: Vec<f32> = Vec::with_capacity(WORDS_PER_QUERY * 3);
    let mut ngrams: Vec<String> = Vec::with_capacity(WORDS_PER_QUERY * 3);
    let mut ngrams_ids: FnvHashMap<String, Vec<usize>> = FnvHashMap::default();

    let words = get_norm_query_vec(query);
    if words.is_empty() {
        return (ngrams, ngrams_relevs, ngrams_ids, words, vec![]);
    }

    let words_len = words.len();
    if words_len == 1 {
        update!(
            ngrams,
            ngrams_relevs,
            ngrams_ids,
            words[0].clone(),
            1.0,
            vec![0]
        );
        return (ngrams, ngrams_relevs, ngrams_ids, words, vec![1.0]);
    }

    let (mut word_idx, stop_idx, words_relevs) = get_word_relevances(&words, tr_map, stopwords);
    let stop_ngrams = get_stop_ngrams(&words, &words_relevs, &mut word_idx, &stop_idx);

    let stop_ngrams_len = stop_ngrams.len();
    let word_thresh = 1.0 / util::max(2.0, words_len as f32 - 1.0);

    let mut words_vec = words
        .iter()
        .enumerate()
        .zip(words_relevs.iter())
        .map(|(i_w, r)| (i_w.0, i_w.1.to_string(), *r))
        .collect::<Vec<(usize, String, f32)>>();
    words_vec.sort_by(|t1, t2| t1.2.partial_cmp(&t2.2).unwrap_or(Ordering::Less).reverse());

    // bigrams of words with n words in between:
    //  a b c d e f-> [ab, ac, ad, ..., bc, bd, ..., ef]
    let ngram_thresh = 1.8 / words_len as f32;
    for i in 0..stop_ngrams_len {
        if stop_ngrams[i].2.len() > 1 && stop_ngrams[i].1 > ngram_thresh {
            update!(
                ngrams,
                ngrams_relevs,
                ngrams_ids,
                stop_ngrams[i].0.clone(),
                stop_ngrams[i].1,
                stop_ngrams[i].2.clone()
            );
        }

        for j in i + 1..stop_ngrams_len {
            let step = j - i - 1;
            let ntr = (1.0 - step as f32 / 100.0) * (stop_ngrams[i].1 + stop_ngrams[j].1);
            if step < 3 || ntr >= ngram_thresh {
                let ngram = bow2(&stop_ngrams[i].0, &stop_ngrams[j].0);
                if ngrams_ids.contains_key(&ngram) {
                    continue;
                }
                let mut ngram_ids_vec = stop_ngrams[i].2.clone();
                ngram_ids_vec.extend(stop_ngrams[j].2.clone());
                update!(ngrams, ngrams_relevs, ngrams_ids, ngram, ntr, ngram_ids_vec);
            }
        }
    }

    if words_len <= 3 || words_vec[0].2 > word_thresh {
        // insert the most relevant word
        update!(
            ngrams,
            ngrams_relevs,
            ngrams_ids,
            words_vec[0].1.clone(),
            words_vec[0].2,
            vec![words_vec[0].0]
        );
    }

    if words_len > 3 {
        // ngram with 3 most relevant words
        update!(
            ngrams,
            ngrams_relevs,
            ngrams_ids,
            bow3(
                &words_vec[0].1.clone(),
                &words_vec[1].1.clone(),
                &words_vec[2].1.clone()
            ),
            words_vec[0].2 + words_vec[1].2 + words_vec[2].2,
            vec![words_vec[0].0, words_vec[1].0, words_vec[2].0]
        );

        // ngram with 2 most relevant words
        update!(
            ngrams,
            ngrams_relevs,
            ngrams_ids,
            bow2(&words_vec[0].1.clone(), &words_vec[1].1.clone()),
            &words_vec[0].2 + &words_vec[1].2,
            vec![words_vec[0].0, words_vec[1].0]
        );

        if let Some(last) = words_vec.pop() {
            // ngram with the most and the least relevant word
            // if any of the top 2 words is bellow the word_thresh
            if words_vec[0].2 <= word_thresh {
                update!(
                    ngrams,
                    ngrams_relevs,
                    ngrams_ids,
                    bow2(&words_vec[0].1.clone(), &last.1.clone()),
                    words_vec[0].2 + last.2,
                    vec![words_vec[0].0, last.0]
                );
            } else {
                update!(
                    ngrams,
                    ngrams_relevs,
                    ngrams_ids,
                    bow2(&words_vec[1].1, &last.1.clone()),
                    words_vec[1].2 + last.2,
                    vec![words_vec[1].0, last.0]
                );
            }
        }
    }

    (ngrams, ngrams_relevs, ngrams_ids, words, words_relevs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use util::*;
    use fst::Map;
    use stopwords;

    #[test]
    fn test_u8_find_and_replace() {
        let stopwords = match stopwords::load("./index/stopwords.txt") {
            Ok(stopwords) => stopwords,
            Err(_) => panic!(
                [
                    BYELL,
                    "No such file or directory: ",
                    ECOL,
                    BRED,
                    "../index/stopwords.txt",
                    ECOL
                ].join("")
            ),
        };

        let tr_map = unsafe {
            match Map::from_path("./index/terms_relevance.fst") {
                Ok(tr_map) => tr_map,
                Err(_) => panic!("Failed to load terms rel. map!"),
            }
        };

        let q = "'Here's@#An ##example!";
        let e = "here s  an   example";
        assert_eq!(normalize(q), e);

        let q = "'Here's@#another one with some question?## and a comma, as well as (parenthesis)!";
        let e = "here s  another one with some question   and a comma as well as  parenthesis";
        assert_eq!(normalize(q), e);
    }

    fn get_ngrams(query: &str, tr_map: &Map, stopwords: &FnvHashSet<String>) -> Vec<String> {
        let words = get_norm_query_vec(query);
        if words.is_empty() {
            return vec![];
        }

        let (mut word_idx, stop_idx, rels) = get_word_relevances(&words, tr_map, stopwords);
        let stop_ngrams = get_stop_ngrams(&words, &rels, &mut word_idx, &stop_idx);

        stop_ngrams
            .iter()
            .map(|tup| tup.0.clone())
            .collect::<Vec<String>>()
    }

    #[test]
    fn test_get_stop_ngrams() {
        let stopwords = match stopwords::load("./index/stopwords.txt") {
            Ok(stopwords) => stopwords,
            Err(_) => panic!(
                [
                    BYELL,
                    "No such file or directory: ",
                    ECOL,
                    BRED,
                    "../index/stopwords.txt",
                    ECOL
                ].join("")
            ),
        };

        let tr_map = unsafe {
            match Map::from_path("./index/terms_relevance.fst") {
                Ok(tr_map) => tr_map,
                Err(_) => panic!("Failed to load terms rel. map!"),
            }
        };

        let q = "laravel has many order by";
        let e = vec!["laravel", "has many", "order by"];
        assert_eq!(get_ngrams(q, &tr_map, &stopwords), e);

        let q = "order by has many laravel";
        let e = vec!["order by", "has many", "laravel"];
        assert_eq!(get_ngrams(q, &tr_map, &stopwords), e);

        let q = "the paws of destiny amazon prime";
        let e = vec!["the paws", "of destiny", "amazon", "prime"];
        assert_eq!(get_ngrams(q, &tr_map, &stopwords), e);

        let q = "the clash influences";
        let e = vec!["the clash", "influences"];
        assert_eq!(get_ngrams(q, &tr_map, &stopwords), e);

        let q = "welche 30 unternehmen sind im dax";
        let e = vec!["welche 30", "unternehmen sind", "im dax"];
        assert_eq!(get_ngrams(q, &tr_map, &stopwords), e);

        let q = "if the word is numeric it has to go into bigram";
        let e = vec![
            "if the word",
            "is numeric",
            "it has",
            "to go",
            "into bigram",
        ];
        assert_eq!(get_ngrams(q, &tr_map, &stopwords), e);

        let q = "if the word is 7 it has to go into bigram";
        let e = vec!["if the word", "is 7", "it has", "to go", "into bigram"];
        assert_eq!(get_ngrams(q, &tr_map, &stopwords), e);

        let q = "remove all of the spaces in JavaScript file";
        let e = vec!["remove all", "of the spaces", "in javascript", "file"];
        assert_eq!(get_ngrams(q, &tr_map, &stopwords), e);

        let q = "allocating memory is not a big deal";
        let e = vec!["allocating", "memory is", "not big", "deal"];
        assert_eq!(get_ngrams(q, &tr_map, &stopwords), e);

        let q = "aber keine";
        let e = vec!["aber keine"];
        assert_eq!(get_ngrams(q, &tr_map, &stopwords), e);

        // w/o stopwords
        let q = "hengstenberg evangelische";
        let e = vec!["hengstenberg", "evangelische"];
        assert_eq!(get_ngrams(q, &tr_map, &stopwords), e);

        let q = "changing mac os menu bar";
        let e = vec!["changing mac", "os", "menu", "bar"];
        assert_eq!(get_ngrams(q, &tr_map, &stopwords), e);

        let q = "emacs bind buffer mode key";
        let e = vec!["emacs", "bind", "buffer", "mode key"];
        assert_eq!(get_ngrams(q, &tr_map, &stopwords), e);

        let q = "sim karte defekt t mobile iphone";
        let e = vec!["sim", "karte defekt", "mobile", "iphone"];
        assert_eq!(get_ngrams(q, &tr_map, &stopwords), e);

        let q = "disneyland paris ticket download";
        let e = vec!["disneyland", "paris", "ticket download"];
        assert_eq!(get_ngrams(q, &tr_map, &stopwords), e);

        let q = "friends s01 e01 stream";
        let e = vec!["friends", "s01", "e01 stream"];
        assert_eq!(get_ngrams(q, &tr_map, &stopwords), e);

        // TODO fix this, should be xselenax
        let q = "@x s e l e n a x";
        let e: Vec<String> = vec![];
        assert_eq!(get_ngrams(q, &tr_map, &stopwords), e);
    }
}
