extern crate fst;
extern crate regex;

use fst::Map;
use fnv::{FnvHashMap, FnvHashSet};

pub const WORDS_PER_QUERY: usize = 15;

macro_rules! bow {
    ($w1: expr, $w2: expr) => {{
        let mut v = String::with_capacity($w1.len() + $w2.len() + 1);
        if $w1 < $w2 {
            v.push_str($w1);
            v.push_str(" ");
            v.push_str($w2);
        } else {
            v.push_str($w2);
            v.push_str(" ");
            v.push_str($w1);
        }
        v
    }};
}

#[inline]
fn bow_ngrams(words: Vec<(String, f32)>, ngrams: &mut FnvHashMap<String, f32>) {
    if words.len() == 1 {
        ngrams.insert(words[0].0.clone(), words[0].1);
    } else if words.len() > 1 {
        for i in 0..words.len() - 1 {
            let (w1, mut w2) = (&words[i].0, &words[i + 1].0);
            let mut tr = words[i].1 + words[i + 1].1;
            ngrams.insert(bow!(w1, w2), tr);

            if i < words.len() - 2 {
                w2 = &words[i + 2].0;
                tr = 0.97 * (words[i].1 + words[i + 2].1);
                ngrams.insert(bow!(w1, w2), tr);
            }
        }

        if words.len() > 3 {
            let tr_threshold = 2.0 / words.len() as f32;
            for i in 0..words.len() - 2 {
                for j in i + 2..words.len() {
                    let mut tr = words[i].1 + words[j].1;
                    if tr < tr_threshold {
                        continue;
                    }
                    let ngram = bow!(&words[i].0, &words[j].0);
                    if !ngrams.contains_key(&ngram) {
                        ngrams.insert(ngram, tr);
                    }
                }
            }
        }
    }
}

const RM_SYMBOLS: &str = "@#!";
lazy_static! {
    static ref RE: regex::Regex = regex::Regex::new(&format!("[{}]", RM_SYMBOLS)).unwrap();
}

#[inline]
fn normalize(query: &str) -> String {
    RE.replace_all(query, "").to_string().to_lowercase()
}

#[inline]
fn fold_to_ngram(terms: Vec<String>, terms_relevance: &FnvHashMap<String, f32>) -> (String, f32) {
    let r = terms
        .iter()
        .fold(0.0, |a, t| a + terms_relevance.get(&t.clone()).unwrap());
    let s: String = terms.into_iter().collect::<Vec<_>>().join(" ");

    (s, r)
}

#[inline]
fn get_terms_relevance(terms: &Vec<String>, tr_map: &fst::Map) -> FnvHashMap<String, f32> {
    let mut missing: FnvHashSet<String> = FnvHashSet::default();
    let mut terms_rel: FnvHashMap<String, f32> = FnvHashMap::default();

    let tset = terms.clone().into_iter().collect::<FnvHashSet<String>>();
    for t in &tset {
        match tr_map.get(t) {
            Some(tr) => {
                terms_rel.insert(t.to_string(), tr as f32);
            }
            None => {
                missing.insert(t.to_string()); // not used!
            }
        };
    }

    // avg and sum
    let mut sum: f32 = terms_rel.values().fold(0.0, |a, b| a + *b);
    let mut avg: f32 = sum / terms_rel.len() as f32;
    // terms may repeat in the query or/and sum might be zero
    if sum > 0.0 {
        sum = terms
            .iter()
            .fold(0.0, |a, t| a + terms_rel.get(&t.clone()).unwrap_or(&avg));
        avg = sum / terms.len() as f32;
    } else {
        avg = 1.0;
        sum = terms.len() as f32;
    }

    // assign the average term relevance to the missing terms and normalize
    for t in tset.iter() {
        let rel = terms_rel.entry(t.to_string()).or_insert(avg);
        *rel /= sum;
    }

    terms_rel
}

#[inline]
pub fn parse(query: &str, stopwords: &FnvHashSet<String>, tr_map: &Map) -> FnvHashMap<String, f32> {
    let mut ngrams: FnvHashMap<String, f32> = FnvHashMap::default();

    let wvec = normalize(query)
        .split(" ")
        .map(|w| w.to_string())
        .collect::<Vec<String>>();
    let terms_rel = get_terms_relevance(&wvec, tr_map);

    let mut tempv = vec![];
    // concatenate terms with stopwords
    let mut w_stop: Vec<(String, f32)> = vec![]; // [('the best', 0.2), ('search', 0.3)]
                                                 // concatenate terms without stopwords
    let mut wo_stop: Vec<(String, f32)> = vec![]; // [('best', 0.15), ('search', 0.3)]
    let mut has_stopword = false;

    for w in wvec {
        // already encountered consecutive 3 stop words, push them to stopwords stack
        if tempv.len() > 2 {
            let (s, r) = fold_to_ngram(tempv, &terms_rel);
            w_stop.push((s, r));
            tempv = vec![];
        }

        if stopwords.contains(&w) || w.len() < 3 {
            has_stopword = true;
            tempv.push(w.clone());
            continue;
        }

        // since we are here the last word is not a stop-word, insert as a unigram
        ngrams.insert(w.clone(), *terms_rel.get(&w).unwrap());

        if has_stopword {
            tempv.push(w.clone());
            let (s, r) = fold_to_ngram(tempv, &terms_rel);
            w_stop.push((s, r));
            wo_stop.push((w.clone(), *terms_rel.get(&w).unwrap()));

            has_stopword = false;
            tempv = vec![];
        } else {
            w_stop.push((w.clone(), *terms_rel.get(&w).unwrap()));
            wo_stop.push((w.clone(), *terms_rel.get(&w).unwrap()));
        }
    }

    // in case query consists of stopwords only
    if tempv.len() > 0 {
        let (s, r) = fold_to_ngram(tempv, &terms_rel);
        w_stop.push((s, r));
    }

    // generate ngrams as bag of words combination of terms c a b d -> ac, bc, ab, ad, bd
    bow_ngrams(w_stop, &mut ngrams);
    bow_ngrams(wo_stop, &mut ngrams);

    return ngrams;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[inline]
    fn normalize_repl() -> String {
        "#Hello@World!#"
            .to_lowercase()
            .replace("@", "")
            .replace("#", "")
            .replace("!", "")
            .to_string()
    }

    #[test]
    fn test_regex_repl() {
        assert_eq!(&normalize("#Hello@World!#"), "helloworld");
    }

    #[test]
    fn test_string_repl() {
        assert_eq!(&normalize_repl(), "helloworld");
    }

    #[test]
    fn test_parse_stopwords_query() {
        let stopwords: FnvHashSet<String> = ["und".to_string()].iter().cloned().collect();
        let terms_relevance = Map::from_iter(vec![("1", 1), ("und", 1)]).unwrap();

        let mut query = "1 und 1";
        let expected: FnvHashMap<String, f32> =
            [("1 und 1".to_string(), 1.0)].iter().cloned().collect();
        assert_eq!(parse(query, &stopwords, &terms_relevance), expected);

        query = "und und";
        let expected: FnvHashMap<String, f32> =
            [("und und".to_string(), 1.0)].iter().cloned().collect();
        assert_eq!(parse(query, &stopwords, &terms_relevance), expected);

        query = "1 und 1 und";
        let expected: FnvHashMap<String, f32> =
            [("1 und 1 und".to_string(), 1.0)].iter().cloned().collect();
        assert_eq!(parse(query, &stopwords, &terms_relevance), expected);

        query = "1 und 1 und 1";
        let expected: FnvHashMap<String, f32> = [("1 und 1 und 1".to_string(), 1.0)]
            .iter()
            .cloned()
            .collect();
        assert_eq!(parse(query, &stopwords, &terms_relevance), expected);
    }

    #[test]
    fn test_parse_query() {
        let query = "1 und 1 account login";
        let stopwords: FnvHashSet<String> = ["und".to_string()].iter().cloned().collect();
        let terms_relevance = Map::from_iter(vec![("1", 1), ("und", 1)]).unwrap();

        let expected: FnvHashMap<String, f32> = [
            ("account".to_string(), 0.2),
            ("login".to_string(), 0.2),
            ("account login".to_string(), 0.4),
            ("1 und 1 login".to_string(), 0.776),
            ("1 und 1 account".to_string(), 0.8),
        ].iter()
            .cloned()
            .collect();

        assert_eq!(parse(query, &stopwords, &terms_relevance), expected);
    }

    #[test]
    fn test_parse_long_query() {
        let query = "what i learned from 200 job interviews at google";

        let stopwords: FnvHashSet<String> =
            ["what".to_string(), "from".to_string(), "at".to_string()]
                .iter()
                .cloned()
                .collect();

        let terms_relevance = Map::from_iter(vec![
            ("200", 1),
            ("google", 1),
            ("i", 1),
            ("interviews", 1),
            ("job", 1),
            ("learned", 1),
        ]).unwrap();

        let expected: FnvHashMap<String, f32> = [
            ("from 200 what i learned".to_string(), 0.5555556),
            ("200 job".to_string(), 0.22222222),
            ("from 200 job".to_string(), 0.33333334),
            ("200 learned".to_string(), 0.22222222),
            ("learned".to_string(), 0.11111111),
            ("interviews".to_string(), 0.11111111),
            ("job what i learned".to_string(), 0.43111113),
            ("google job".to_string(), 0.21555556),
            ("google".to_string(), 0.11111111),
            ("at google interviews".to_string(), 0.33333334),
            ("at google what i learned".to_string(), 0.5555556), // tr > 0.5
            ("at google job".to_string(), 0.32333335),
            ("200".to_string(), 0.11111111),
            ("from 200 interviews".to_string(), 0.32333335),
            ("google interviews".to_string(), 0.22222222),
            ("job".to_string(), 0.11111111),
            ("200 interviews".to_string(), 0.21555556),
            ("interviews job".to_string(), 0.22222222),
            ("job learned".to_string(), 0.21555556),
            ("at google from 200".to_string(), 0.44444445),
            ("interviews what i learned".to_string(), 0.44444445),
        ].iter()
            .cloned()
            .collect();

        assert_eq!(parse(query, &stopwords, &terms_relevance), expected);
    }
}
