#ifndef CUSTOM_SORTING_H
#define CUSTOM_SORTING_H

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

#endif // CUSTOM_SORTING_H
