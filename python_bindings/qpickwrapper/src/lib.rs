#![crate_type = "dylib"]

extern crate libc;
extern crate qpick;

use std::ffi::{CStr, CString};

/// Get an immutable reference from a raw pointer
macro_rules! ref_from_ptr {
    ($p:ident) => (unsafe {
        assert!(!$p.is_null());
        &*$p
    })
}

/// Get the object referenced by the raw pointer
macro_rules! val_from_ptr {
    ($p:ident) => (unsafe {
        assert!(!$p.is_null());
        Box::from_raw($p)
    })
}

/// Declare a function that frees a struct's memory
macro_rules! make_free_fn {
    ($name:ident, $t:ty) => (
    #[no_mangle]
    pub extern fn $name(ptr: $t) {
        assert!(!ptr.is_null());
        val_from_ptr!(ptr);
    })
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

use qpick::Qpick;

// `#[no_mangle]` warns for lifetime parameters,
// a known issue: https://github.com/rust-lang/rust/issues/40342
#[no_mangle]
pub extern "C" fn qpick_init(path: *mut libc::c_char) -> *mut Qpick {
    let path = cstr_to_str(path);
    let qpick = Qpick::from_path(path.to_string());
    to_raw_ptr(qpick)
}

#[no_mangle]
pub extern "C" fn qpick_init_with_shard_range(path: *mut libc::c_char,
                                         start_shard: libc::uint32_t,
                                         end_shard: libc::uint32_t) -> *mut Qpick {
    let path = cstr_to_str(path);
    let qpick = Qpick::from_path_with_shard_range(
        path.to_string(),
        (start_shard..end_shard),
    );
    to_raw_ptr(qpick)
}
make_free_fn!(qpick_free, *mut Qpick);

#[no_mangle]
pub extern "C" fn string_free(s: *mut libc::c_char) {
    unsafe { CString::from_raw(s) };
}

#[no_mangle]
pub extern "C" fn qpick_get_as_string(
    ptr: *mut Qpick,
    query: *mut libc::c_char) -> *const libc::c_char {

    let query = cstr_to_str(query);
    let s = ref_from_ptr!(ptr).search(query);
    CString::new(s).unwrap().into_raw()
}

// ------ iterators ---

#[repr(C)]
#[derive(Debug)]
#[allow(dead_code)]
pub struct QpickItem {
    qid: libc::uint64_t,
    sc: libc::c_float, //f32
}

// Get a mutable reference from a raw pointer
macro_rules! mutref_from_ptr {
    ($p:ident) => (unsafe {
        assert!(!$p.is_null());
        &mut *$p
    })
}

// Declare a function that returns the next item from a qpick vector
#[no_mangle]
pub extern fn qpick_iter_next(ptr: *mut qpick::QpickResults) -> *mut QpickItem {
    let res = mutref_from_ptr!(ptr);
    // let mut iter = res.items.iter();
    match res.next() {
        Some(qid_sc) => to_raw_ptr(QpickItem { qid: qid_sc.0, sc: qid_sc.1 }),
        None         => ::std::ptr::null_mut()
    }
}

make_free_fn!(qpick_results_free, *mut qpick::QpickResults);
make_free_fn!(qpick_item_free, *mut QpickItem);

#[no_mangle]
pub extern "C" fn qpick_get(
    ptr: *mut Qpick,
    query: *mut libc::c_char,
    count: libc::uint32_t) -> *mut qpick::QpickResults {

    let query = cstr_to_str(query);
    to_raw_ptr(ref_from_ptr!(ptr).get(query, count))
}

// --- nget queries api
#[no_mangle]
pub extern "C" fn query_vec_init() -> *mut Vec<String> {
    to_raw_ptr(vec![])
}
make_free_fn!(query_vec_free, *mut Vec<String>);

#[no_mangle]
pub extern "C" fn query_vec_push(ptr: *mut Vec<String>, query: *mut libc::c_char) {
    let query = cstr_to_str(query);

    mutref_from_ptr!(ptr).push(query.to_string());
}

#[no_mangle]
pub extern "C" fn qpick_nget(
    ptr: *mut Qpick,
    queries: *mut Vec<String>,
    count: libc::uint32_t) -> *mut qpick::QpickResults {

    to_raw_ptr(ref_from_ptr!(ptr).nget(ref_from_ptr!(queries), count))
}
