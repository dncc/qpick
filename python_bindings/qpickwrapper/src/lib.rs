#![crate_type = "dylib"]

extern crate libc;
extern crate qpick;

use std::ffi::{CStr, CString};

/// Get an immutable reference from a raw pointer
macro_rules! ref_from_ptr {
    ($p: ident) => {
        unsafe {
            assert!(!$p.is_null());
            &*$p
        }
    };
}

/// Get the object referenced by the raw pointer
macro_rules! val_from_ptr {
    ($p: ident) => {
        unsafe {
            assert!(!$p.is_null());
            Box::from_raw($p)
        }
    };
}

/// Declare a function that frees a struct's memory
macro_rules! make_free_fn {
    ($name: ident, $t: ty) => {
        #[no_mangle]
        pub extern "C" fn $name(ptr: $t) {
            assert!(!ptr.is_null());
            val_from_ptr!(ptr);
        }
    };
}

// Get a mutable reference from a raw pointer
macro_rules! mutref_from_ptr {
    ($p: ident) => {
        unsafe {
            assert!(!$p.is_null());
            &mut *$p
        }
    };
}

pub fn str_to_cstr(string: &str) -> *mut libc::c_char {
    CString::new(string).unwrap().into_raw()
}

pub fn cstr_to_str<'a>(s: *mut libc::c_char) -> &'a str {
    let cstr = unsafe { CStr::from_ptr(s) };
    cstr.to_str().unwrap()
}

pub fn to_raw_ptr<T>(v: T) -> *mut T {
    Box::into_raw(Box::new(v))
}

// --- string vector
#[no_mangle]
pub extern "C" fn string_vec_init() -> *mut Vec<String> {
    to_raw_ptr(vec![])
}
make_free_fn!(string_vec_free, *mut Vec<String>);

#[no_mangle]
pub extern "C" fn string_vec_push(ptr: *mut Vec<String>, query: *mut libc::c_char) {
    let query = cstr_to_str(query);

    mutref_from_ptr!(ptr).push(query.to_string());
}

// --- string vector end

// --- shard, index and i2q
use qpick::builder;
use qpick::shard;
use qpick::stringvec;
use qpick::Qpick;

// index and shard bindings
#[no_mangle]
pub extern "C" fn qpick_shard(
    file_path: *mut libc::c_char,
    nr_shards: libc::uint32_t,
    output_dir: *mut libc::c_char,
    prefixes: *mut Vec<String>,
    create_i2q: libc::uint8_t,
) {
    let file_path = cstr_to_str(file_path);
    let output_dir = cstr_to_str(output_dir);

    match shard::shard(
        &file_path.to_string(),
        nr_shards as usize,
        &output_dir.to_string(),
        ref_from_ptr!(prefixes),
        create_i2q != 0,
    ) {
        Ok(r) => println!("{:?}", r),
        Err(err) => println!("{:?}", err),
    };
}

#[no_mangle]
pub extern "C" fn qpick_index(
    input_dir: *mut libc::c_char,
    first_shard: libc::uint32_t,
    last_shard: libc::uint32_t,
    output_dir: *mut libc::c_char,
) {
    let input_dir = cstr_to_str(input_dir);
    let output_dir = cstr_to_str(output_dir);

    match builder::index(
        &input_dir.to_string(),
        first_shard as usize,
        last_shard as usize,
        &output_dir.to_string(),
    ) {
        Ok(r) => println!("{:?}", r),
        Err(err) => println!("{:?}", err),
    };
}

#[no_mangle]
pub extern "C" fn qpick_compile_i2q(file_path: *mut libc::c_char, output_dir: *mut libc::c_char) {
    let file_path = cstr_to_str(file_path);
    let output_dir = cstr_to_str(output_dir);

    let r = stringvec::compile(&file_path.to_string(), &output_dir.to_string());
    println!("{:?}", r);
}
// end shard, index and i2q bindings

// `#[no_mangle]` warns for lifetime parameters,
// a known issue: https://github.com/rust-lang/rust/issues/40342
#[no_mangle]
pub extern "C" fn qpick_init(path: *mut libc::c_char) -> *mut Qpick<'static> {
    let path = cstr_to_str(path);
    let qpick = Qpick::from_path(path.to_string());
    to_raw_ptr(qpick)
}

#[no_mangle]
pub extern "C" fn qpick_init_with_shard_range(
    path: *mut libc::c_char,
    start_shard: libc::uint32_t,
    end_shard: libc::uint32_t,
) -> *mut Qpick<'static> {
    let path = cstr_to_str(path);
    let qpick = Qpick::from_path_with_shard_range(path.to_string(), start_shard..end_shard);
    to_raw_ptr(qpick)
}
make_free_fn!(qpick_free, *mut Qpick);

#[no_mangle]
pub extern "C" fn string_free(s: *mut libc::c_char) {
    unsafe { CString::from_raw(s) };
}

// ------ search iterator ---
#[repr(C)]
#[derive(Debug)]
#[allow(dead_code)]
pub struct QpickDistance {
    keyword: libc::c_float,
    cosine: libc::c_float,
}

#[repr(C)]
#[derive(Debug)]
#[allow(dead_code)]
pub struct QpickSearchItem {
    query_id: libc::uint64_t,
    query: *mut libc::c_char,
    dist: *mut QpickDistance,
}

// Declare a function that returns the next item from a qpick vector
#[no_mangle]
pub extern "C" fn qpick_search_iter_next(ptr: *mut qpick::SearchResults) -> *mut QpickSearchItem {
    let res = mutref_from_ptr!(ptr);
    // let mut iter = res.items.iter();
    match res.next() {
        Some(r) => to_raw_ptr(QpickSearchItem {
            query_id: r.query_id,
            dist: to_raw_ptr(QpickDistance {
                keyword: r.dist.keyword,
                cosine: r.dist.cosine.unwrap_or(-1.0),
            }),
            query: if let Some(query) = r.query {
                str_to_cstr(&query)
            } else {
                str_to_cstr("")
            },
        }),
        None => ::std::ptr::null_mut(),
    }
}

make_free_fn!(qpick_search_results_free, *mut qpick::SearchResults);
make_free_fn!(qpick_search_item_free, *mut QpickSearchItem);
make_free_fn!(qpick_distance_free, *mut QpickDistance);

// ------ dist iterator ---

#[repr(C)]
#[derive(Debug)]
#[allow(dead_code)]
pub struct QpickDistItem {
    query: *mut libc::c_char,
    dist: *mut QpickDistance,
}

// Declare a function that returns the next item from a qpick vector
#[no_mangle]
pub extern "C" fn qpick_dist_iter_next(ptr: *mut qpick::DistResults) -> *mut QpickDistItem {
    let res = mutref_from_ptr!(ptr);
    match res.next() {
        Some(r) => to_raw_ptr(QpickDistItem {
            query: str_to_cstr(&r.query),
            dist: to_raw_ptr(QpickDistance {
                keyword: r.dist.keyword,
                cosine: r.dist.cosine.unwrap_or(-1.0),
            }),
        }),
        None => ::std::ptr::null_mut(),
    }
}

make_free_fn!(qpick_dist_results_free, *mut qpick::DistResults);
make_free_fn!(qpick_dist_item_free, *mut QpickDistItem);

// --- end iterators ---

#[no_mangle]
pub extern "C" fn qpick_get(
    ptr: *mut Qpick,
    query: *mut libc::c_char,
    count: libc::uint32_t,
    with_tfidf: libc::uint8_t,
) -> *mut qpick::SearchResults {
    let query = cstr_to_str(query);
    let res = ref_from_ptr!(ptr).get_search_results(query, count, with_tfidf != 0);

    to_raw_ptr(res)
}

#[no_mangle]
pub extern "C" fn qpick_get_distances(
    ptr: *mut Qpick,
    query: *mut libc::c_char,
    queries: *mut Vec<String>,
) -> *mut qpick::DistResults {
    to_raw_ptr(ref_from_ptr!(ptr).get_dist_results(cstr_to_str(query), ref_from_ptr!(queries)))
}
