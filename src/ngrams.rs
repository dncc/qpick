extern crate fst;
extern crate regex;
extern crate test;

use fst::Map;
use std::collections::HashMap;
use std::collections::HashSet;

use shard::QueryType;

fn get_terms_relevance(terms: &Vec<String>, tr_map: &fst::Map) -> HashMap<String, f32> {
    let mut missing: HashSet<String> = HashSet::new();
    let mut terms_rel: HashMap<String, f32> = HashMap::new();

    let tset = terms.clone().into_iter().collect::<HashSet<String>>();
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

macro_rules! bow_ngrams {
    ($wv:ident, $ngrams: ident, $mult: ident) => (
    if $wv.len() > 0 {
        for i in 0..$wv.len()-1 {
            let (w1, mut w2) = (&$wv[i].0, &$wv[i+1].0);
            let mut tr = $mult * ($wv[i].1 + $wv[i+1].1);
            let mut v = String::with_capacity(w1.len()+w2.len()+1);
            if w1 < w2 {
                v.push_str(w1);
                v.push_str(" ");
                v.push_str(w2);
            } else {
                v.push_str(w2);
                v.push_str(" ");
                v.push_str(w1);
            }
            $ngrams.insert(v, tr);

            if i < $wv.len()-2 {
                w2 = &$wv[i+2].0;
                tr = 0.97 *($wv[i].1 + $wv[i+2].1);
                v = String::with_capacity(w1.len()+w2.len()+1);
                if w1 < w2 {
                    v.push_str(w1);
                    v.push_str(" ");
                    v.push_str(w2);
                } else {
                    v.push_str(w2);
                    v.push_str(" ");
                    v.push_str(w1);
                }
                $ngrams.insert(v, tr);
            }
        }
    })
}

const RM_SYMBOLS: &str = "@#!";
lazy_static! {
    static ref RE: regex::Regex = regex::Regex::new(&format!("[{}]", RM_SYMBOLS)).unwrap();
}

#[inline]
fn normalize(query: &str) -> String {
    RE.replace_all(query, "").to_string().to_lowercase()
}

pub fn parse(
    query: &str,
    stopwords: &HashSet<String>,
    tr_map: &Map,
    query_type: QueryType,
) -> HashMap<String, f32> {
    let mut ngrams: HashMap<String, f32> = HashMap::new();

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
        if stopwords.contains(&w) || w.len() < 3 {
            has_stopword = true;
            tempv.push(w.clone());
            continue;
        }

        // since we are here the last word is not a stop-word, insert as a unigram
        ngrams.insert(w.clone(), *terms_rel.get(&w).unwrap());

        if has_stopword {
            tempv.push(w.clone());
            let r = tempv
                .iter()
                .fold(0.0, |a, t| a + terms_rel.get(&t.clone()).unwrap());
            let s: String = tempv.into_iter().collect::<Vec<_>>().join(" ");
            w_stop.push((s, r));
            wo_stop.push((w.clone(), *terms_rel.get(&w).unwrap()));

            has_stopword = false;
            tempv = vec![];
        } else {
            w_stop.push((w.clone(), *terms_rel.get(&w).unwrap()));
            wo_stop.push((w.clone(), *terms_rel.get(&w).unwrap()));
        }
    }

    let mult = match query_type {
        QueryType::Q => 1.0,
        QueryType::TUW => 0.95,
    };

    // generate ngrams as bag of words combination of terms c a b d -> ac, bc, ab, ad, bd
    bow_ngrams!(w_stop, ngrams, mult);
    bow_ngrams!(wo_stop, ngrams, mult);

    return ngrams;
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::test::Bencher;

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

    #[bench]
    fn bench_regex_repl(b: &mut Bencher) {
        b.iter(|| normalize("#Hello@World!#"));
    }

    #[test]
    fn test_string_repl() {
        assert_eq!(&normalize_repl(), "helloworld");
    }

    #[bench]
    fn bench_string_repl(b: &mut Bencher) {
        b.iter(|| normalize_repl());
    }
}
