use std::cmp::PartialOrd;

#[inline]
pub fn qid2pid(qid: u64, nr_shards: usize) -> u64 {
    qid % nr_shards as u64
}

#[inline]
pub fn qid2pqid(qid: u64, nr_shards: usize) -> u64 {
    qid >> (nr_shards as f32).log(2.0) as u64
}

#[inline]
pub fn pqid2qid(pqid: u64, pid: u64, nr_shards: usize) -> u64 {
    (pqid << (nr_shards as f32).log(2.0) as u64) + pid
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
