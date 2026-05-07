#ifndef CUSTOM_SORTING_H
#define CUSTOM_SORTING_H

#include <stddef.h>

/**
 * Natively sort a double precision float64 array in-place on the FFI heap.
 */
void native_sort_double(double *array, int size);

/**
 * Natively sort a single precision float32 array in-place.
 */
void native_sort_float(float *array, int size);

/**
 * Natively sort an int64 array in-place.
 */
void native_sort_int64(long long *array, int size);

/**
 * Natively sort an int32 array in-place.
 */
void native_sort_int32(int *array, int size);

/**
 * Natively sort a complex128 array in-place lexicographically.
 */
void native_sort_complex128(double *array, int size);

/**
 * Natively sort a complex64 array in-place lexicographically.
 */
void native_sort_complex64(float *array, int size);

/**
 * Natively compute indirect sort indices for double precision float64 array.
 */
void native_argsort_double(const double *data, int *indices, int size);

/**
 * Natively compute indirect sort indices for single precision float32 array.
 */
void native_argsort_float(const float *data, int *indices, int size);

/**
 * Natively compute indirect sort indices for int64 array.
 */
void native_argsort_int64(const long long *data, int *indices, int size);

/**
 * Natively compute indirect sort indices for int32 array.
 */
void native_argsort_int32(const int *data, int *indices, int size);

/**
 * Zero-allocation optimized direct C memcmp block byte compare.
 */
int custom_memcmp(const void *s1, const void *s2, size_t n);

#endif // CUSTOM_SORTING_H
