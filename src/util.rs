use std::cmp::PartialOrd;

static KEY_SEPARATOR: &'static str = ":";

#[inline]
pub fn qid2pqid(qid: u64, nr_shards: usize) -> (u64, u8) {
    assert!(nr_shards < 256);
    (qid >> (nr_shards as f32).log(2.0) as u64, (qid % nr_shards as u64) as u8)
}

#[inline]
pub fn pqid2qid(pqid: u64, reminder: u8, nr_shards: usize) -> u64 {
    (pqid << (nr_shards as f32).log(2.0) as u64) + reminder as u64
}

#[inline]
pub fn ngram2key(ngram: &str, shard_id: u32) -> String {
    format!("{}:{}", ngram, shard_id)
}

#[inline]
pub fn key2ngram(key: String) -> (String, u32) {
    let i = key.rfind(KEY_SEPARATOR).unwrap();
    let ngram = (&key[..i]).to_string();
    let pid = &key[i+1..].parse::<u32>().unwrap();

    (ngram, *pid)
}

/*
    Elegant pairing function http://szudzik.com/ElegantPairing.pdf
    TODO implement with bignum, otherwise might overflow!
 */
#[inline]
pub fn elegant_pair(x: u64, y: u64) -> u64 {
    let z: u64 = match x >= y {
        true => x * x + x + y,
        false => y * y + x,
    };
    if elegant_pair_inv(z) != (x, y) {
        panic!("Numbers {} and {} cannot be paired!", x, y);
    };

    z
}

/*
    Inverse elegant pairing function http://szudzik.com/ElegantPairing.pdf
    TODO implement with bignum or f128, otherwise might overflow!
*/
#[inline]
pub fn elegant_pair_inv(z: u64) -> (u64, u64) {
    let q = z as f64;
    let w = (q.sqrt()).floor() as u64;
    let t = (w * w) as u64;
    if (z - t) >= w {
        (w, z - t - w)
    } else {
        (z - t, w)
    }
}

#[inline]
pub fn max<T:PartialOrd>(a:T, b:T) -> T {
    if a > b {
        a
    } else {
        b
    }
}

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// A Fast, Minimal Memory, Consistent Hash Algorithm by John Lamping and Eric Veach:
// https://arxiv.org/pdf/1406.2294.pdf
// It outputs a bucket number in the range [0, num_buckets).
pub fn jump_consistent_hash(mut key: u64, num_buckets: u32) -> u32 {

    assert!(num_buckets > 0);

    let mut b: i64 = -1;
    let mut j: i64 = 0;

    while j < num_buckets as i64 {
        b = j;
        key = key.wrapping_mul(2862933555777941757).wrapping_add(1);
        j = ((b.wrapping_add(1) as f64) * ((1i64 << 31) as f64) /
                                          ((key >> 33).wrapping_add(1) as f64)) as i64;
    }

    b as u32
}

#[inline]
pub fn jump_consistent_hash_str(key: &str, num_buckets: u32) -> u32 { //

    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);

    jump_consistent_hash(hasher.finish(), num_buckets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jump_consistent_hash_str_test() {
        assert_eq!(16, jump_consistent_hash_str("how to put on thai fishing pants", 32));
    }

    #[test]
    fn jump_consistent_hash_test() {
        assert_eq!(7, jump_consistent_hash(1000011111111, 32));
        assert_eq!(0, jump_consistent_hash(1000011111111, 1));
    }

    #[test]
    #[should_panic]
    fn jump_consistent_hash_panic_test() {
        assert_eq!(7, jump_consistent_hash(1000011111111, 0));
    }

}
