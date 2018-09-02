// simple_checked_arrays.h -- a simple wrapper around pointers used as arrays to allow bounds checking
//
// $Log: /Speech_To_Speech_Translation/dbn/dbn/simple_checked_arrays.h $
// 
// 9     1/05/11 9:21a Fseide
// bug fix in hardcoded_array (n, val)
// 
// 8     7/19/09 15:46 Fseide
// finally had to add default constructors to array_ref and
// const_array_ref so they can be used in std::vector
// 
// 7     7/15/09 11:58 Fseide
// added begin() and end() to array_ref types for easier interfacing with
// STL
// 
// 6     6/22/09 18:54 Fseide
// added array_ref::resize() --seems to make no sense, but some functions
// require it, even if the function does nothing (correct size already)
// 
// 5     3/24/09 21:49 Fseide
// checked_array changed to array_ref, const_checked_array likewise
// 
// 4     3/24/09 21:12 Fseide
// array_ref and const_array_ref can now be instantiated from any
// vector class that has operator[] and size(), i.e. we can pass such
// vectors to functions that take (const_)array_ref
// 
// 3     3/24/09 15:52 Fseide
// new class const_array_ref to allow for const *;
// hardcoded_array now has an initializer;
// hardcoded_array now has a typecast to (const_)array_ref
// 
// 2     3/24/09 9:39 Fseide
// (marked all methods as 'inline')
// 
// 1     3/24/09 9:37 Fseide
// created

#pragma once

#include <stddef.h>     // for size_t
#include <assert.h>

// ---------------------------------------------------------------------------
// array_ref -- wraps a C pointer to an array together with its size.
//
// Called _ref because this is a reference to the array rather than the array
// itself (since it wraps a pointer). No need to pass an array_ref by reference.
//
// operator[] checks index bounds in Debug builds. size() is provided such
// that this class can be substituted for STL vector in many cases.
// ---------------------------------------------------------------------------

template<class _T> class array_ref
{
    _T * data;
    size_t n;
    inline void check_index (size_t i) const { i; assert (i < n); }
    inline void check_ptr() const { n; data; assert (n == 0 || data != NULL); }
public:
    inline array_ref (_T * ptr, size_t size) throw() : data (ptr), n (size) { }
    inline array_ref() throw() : data (NULL), n (0) { }   // in case we have a vector of this
    inline       _T & operator[] (size_t i)       throw() { check_index (i); return data[i]; }
    inline const _T & operator[] (size_t i) const throw() { check_index (i); return data[i]; }
    inline size_t size() const throw() { return n; }
    inline _T * begin() { return data; }
    inline _T * end() { return data + n; }
    inline void resize (size_t sz) { sz; assert (n == sz); }    // allow compatibility with some functions
    // construct from other vector types
    template<class _V> inline array_ref (_V & v) : data (v.size() > 0 ? &v[0] : NULL), n ((size_t) v.size()) { }
};


// ---------------------------------------------------------------------------
// const_array_ref -- same as array_ref for 'const' (read-only) pointers
// ---------------------------------------------------------------------------

template<class _T> class const_array_ref
{
    const _T * data;
    size_t n;
    inline void check_index (size_t i) const { i; assert (i < n); }
    inline void check_ptr() const { n; data; assert (n == 0 || data != NULL); }
public:
    inline const_array_ref (const _T * ptr, size_t size) throw() : data (ptr), n (size) { }
    inline const_array_ref() throw() : data (NULL), n (0) { }   // in case we have a vector of this
    inline const _T & operator[] (size_t i) const throw() { check_index (i); return data[i]; }
    inline size_t size() const throw() { return n; }
    inline const _T * begin() { return data; }
    inline const _T * end() { return data + n; }
    // construct from other vector types
    template<class _V> inline const_array_ref (const _V & v) : data (v.size() > 0 ? &v[0] : NULL), n ((size_t) v.size()) { }
};

// ---------------------------------------------------------------------------
// hardcoded_array -- wraps a fixed-size C array together with its size.
//
// operator[] checks index bounds in Debug builds. size() is provided such
// that this class can be substituted for STL vector in many cases.
// Can be constructed with a size parameter--it will be checked against the
// hard-coded size.
// Can also be constructed with an initialization parameter (typ. 0).
// ---------------------------------------------------------------------------

template<class _T, int _N> class hardcoded_array
{
    _T data[_N];
    inline void check_index (size_t i) const { i; assert (i < _N); }
    inline void check_size  (size_t n) const { n; assert (n == _N); }
public:
    inline hardcoded_array() throw() {}
    inline hardcoded_array (size_t n) throw() { check_size (n); }  // we can instantiate with a size parameter--just checks the size
    inline hardcoded_array (size_t n, const _T & val) throw() { check_size (n); for (size_t i = 0; i < n; i++) data[i] = val; }
    inline       _T & operator[] (size_t i)       throw() { check_index (i); return data[i]; }
    inline const _T & operator[] (size_t i) const throw() { check_index (i); return data[i]; }
    inline size_t size() const throw() { return _N; }
};
