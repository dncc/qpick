extern crate fst;
use fst::Map;
use std::collections::HashMap;
use std::collections::HashSet;

fn get_terms_relevance(terms: &Vec<String>, tr_map: &fst::Map) -> HashMap<String, f32> {

    let mut missing: HashSet<String> = HashSet::new();
    let mut terms_rel: HashMap<String, f32> = HashMap::new();

    let tset = terms.clone().into_iter().collect::<HashSet<String>>();
    for t in &tset {
        match tr_map.get(t) {
            Some(tr) => {
                terms_rel.insert(t.to_string(), tr as f32);
            },
            None => {
                missing.insert(t.to_string()); // not used!
            },
        };
    }

    // avg and sum
    let mut sum: f32 = terms_rel.values().fold(0.0, |a, b| a + *b);
    let mut avg: f32 = sum/terms_rel.len() as f32;
    // terms may repeat in the query or/and sum might be zero
    if sum > 0.0 {
        sum = terms.iter().fold(0.0, |a, t| a + terms_rel.get(&t.clone()).unwrap_or(&avg));
        avg = sum/terms.len() as f32;
    } else {
        avg = 1.0;
        sum = terms.len() as f32;
    }

    // set an average term relevance to the missing terms and normalize
    for t in tset.iter() {
        let rel = terms_rel.entry(t.to_string()).or_insert(avg);
        *rel /= sum;
    }

    terms_rel
}

macro_rules! bow_ngrams {
    ($wv:ident, $ngrams: ident) => (
    if $wv.len() > 0 {
        let mut v: Vec<String>;
        for i in 0..$wv.len()-1 {
            v = vec![$wv[i].0.clone(), $wv[i+1].0.clone()];
            v.sort();
            $ngrams.insert(format!("{} {}", v[0], v[1]), $wv[i].1+$wv[i+1].1);

            if i < $wv.len()-2 {
                v = vec![$wv[i].0.clone(), $wv[i+2].0.clone()];
                v.sort();
                $ngrams.insert(format!("{} {}", v[0], v[1]), $wv[i].1+$wv[i+2].1);
            }
        }
    })
}

pub fn parse(query: &str, stopwords: &HashSet<String>, tr_map: &Map)
                -> HashMap<String, f32> {

    let mut ngrams: HashMap<String, f32> = HashMap::new();

    let wvec = query.split(" ").map(|w| w.to_string().to_lowercase()).collect::<Vec<String>>();
    let terms_rel = get_terms_relevance(&wvec, tr_map);

    let mut tempv = vec![];
    // concatenate terms with stopwords
    let mut w_stop: Vec<(String, f32)> = vec![]; // [('the best', 0.2), ('search', 0.3)]
    // concatenate terms without stopwords
    let mut wo_stop: Vec<(String, f32)> = vec![]; // [('best', 0.15), ('search', 0.3)]
    let mut has_stopword = false;

    for i in 0..wvec.len() {
        let w = wvec[i].clone();

        if stopwords.contains(&w) {
            has_stopword = true;
            tempv.push(w.clone());
            continue
        }

        // since we are here the last word is not a stop-word, insert as a unigram
        ngrams.insert(format!("{}", w.clone()), *terms_rel.get(&w).unwrap());

        if has_stopword {
            tempv.push(w.clone());
            let r = tempv.iter().fold(0.0, |a, t| a + terms_rel.get(&t.clone()).unwrap());
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

    // generate ngrams as bag of words combination of terms c a b d -> ac, bc, ab, ad, bd
    bow_ngrams!(w_stop, ngrams);
    bow_ngrams!(wo_stop, ngrams);

    return ngrams
}
