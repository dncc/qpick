extern crate fst;
use fst::Map;
use std::collections::HashMap;
use std::collections::HashSet;

fn get_terms_relevance(terms: &Vec<&str>, tr_map: &fst::Map) -> HashMap<String, f32> {

    let mut missing: HashSet<String> = HashSet::new();
    let mut terms_rel: HashMap<String, f32> = HashMap::new();

    let tset = terms.clone().into_iter().collect::<HashSet<&str>>();
    for t in &tset {
        match tr_map.get(t) {
            Some(tr) => {
                terms_rel.insert(t.to_string(), tr as f32);
            },
            None => {
                missing.insert(t.to_string());
            },
        };
    }

    // avg and sum
    let mut sum: f32 = terms_rel.values().fold(0.0, |a, b| a + *b);
    let mut avg: f32 = sum/terms_rel.len() as f32;
    // terms may repeat in the query or/and sum might be zero
    if sum > 0.0 {
        sum = terms.iter().fold(0.0, |a, t| a + terms_rel.get(t.clone()).unwrap_or(&avg));
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

pub fn parse(query: &str, length: usize, stopwords: &HashSet<String>, tr_map: &Map)
                -> HashMap<String, f32> {

    let mut ngrams: HashMap<String, f32> = HashMap::new();

    let mut wvec = query.split(" ").collect::<Vec<&str>>();
    let terms_rel = get_terms_relevance(&wvec, tr_map);

    wvec.reverse();

    // concatenate terms with stopwords if any
    let mut termv = vec![];
    let mut terms: Vec<(String, f32)> = vec![]; // [('the best', 0.2), ('search', 0.3)]
    let mut has_stopword = false;
    while wvec.len() > 0 {
        let w = wvec.pop().unwrap();
        termv.push(w);
        if stopwords.contains(w) {
            has_stopword = true;

        } else if termv.len() >= length {
            if has_stopword {
                let r = termv.iter().fold(0.0, |a, t| a + terms_rel.get(t.clone()).unwrap());
                let s: String = termv.into_iter().collect::<Vec<_>>().join(" ");
                terms.push((s, r));
            } else {
                for t in termv.into_iter() {
                    terms.push((t.to_string(), *terms_rel.get(t).unwrap()));
                }
            }
            has_stopword = false;
            termv = vec![];
        }
    }

    // combine the new terms into ngrams a b c d -> ab, ac, bc, bd, cd
    if terms.len() > 0 {
        for i in 0..terms.len()-1 {
            ngrams.insert(format!("{}", terms[i].0), terms[i].1);
            ngrams.insert(format!("{} {}", terms[i].0, terms[i+1].0), terms[i].1+terms[i+1].1);
            if i < terms.len()-2 {
                ngrams.insert(format!("{} {}", terms[i].0, terms[i+2].0), terms[i].1+terms[i+2].1);
            }
        }
        ngrams.insert(format!("{}", terms[terms.len()-1].0), terms[terms.len()-1].1);
    }

    return ngrams
}
