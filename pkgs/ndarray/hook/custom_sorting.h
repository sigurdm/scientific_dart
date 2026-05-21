#ifndef CUSTOM_SORTING_H
#define CUSTOM_SORTING_H

#include <stddef.h>

// ----------------------------------------------------------------------------
// Public Sorters with Kind Parameter
// kind: 0 = quicksort, 1 = mergesort/stable, 2 = heapsort
// ----------------------------------------------------------------------------
void native_sort_double(double *array, int size, int kind);
void native_sort_float(float *array, int size, int kind);
void native_sort_int64(long long *array, int size, int kind);
void native_sort_int32(int *array, int size, int kind);
void native_sort_complex128(double *array, int size, int kind);
void native_sort_complex64(float *array, int size, int kind);

// ----------------------------------------------------------------------------
// Public Argsort Sorters with Kind Parameter
// ----------------------------------------------------------------------------
void native_argsort_double(const double *data, int *indices, int size, int kind);
void native_argsort_float(const float *data, int *indices, int size, int kind);
void native_argsort_int64(const long long *data, int *indices, int size, int kind);
void native_argsort_int32(const int *data, int *indices, int size, int kind);

// ----------------------------------------------------------------------------
// Public Partition Sorters
// ----------------------------------------------------------------------------
void native_partition_double(double *array, int size, const int *k_list, int k_size);
void native_partition_float(float *array, int size, const int *k_list, int k_size);
void native_partition_int64(long long *array, int size, const int *k_list, int k_size);
void native_partition_int32(int *array, int size, const int *k_list, int k_size);
void native_partition_complex128(double *array, int size, const int *k_list, int k_size);
void native_partition_complex64(float *array, int size, const int *k_list, int k_size);

// ----------------------------------------------------------------------------
// Public Argpartition Sorters
// ----------------------------------------------------------------------------
void native_argpartition_double(const double *data, int *indices, int size, const int *k_list, int k_size);
void native_argpartition_float(const float *data, int *indices, int size, const int *k_list, int k_size);
void native_argpartition_int64(const long long *data, int *indices, int size, const int *k_list, int k_size);
void native_argpartition_int32(const int *data, int *indices, int size, const int *k_list, int k_size);
void native_argpartition_complex128(const double *data, int *indices, int size, const int *k_list, int k_size);
void native_argpartition_complex64(const float *data, int *indices, int size, const int *k_list, int k_size);

// ----------------------------------------------------------------------------
// Public Searchsorted (Binary Search) functions
// ----------------------------------------------------------------------------
void native_searchsorted_double(const double *array, int size, const double *values, int *out_indices, int num_values, int side_left, const int *sorter);
void native_searchsorted_float(const float *array, int size, const float *values, int *out_indices, int num_values, int side_left, const int *sorter);
void native_searchsorted_int64(const long long *array, int size, const long long *values, int *out_indices, int num_values, int side_left, const int *sorter);
void native_searchsorted_int32(const int *array, int size, const int *values, int *out_indices, int num_values, int side_left, const int *sorter);
void native_searchsorted_complex128(const double *array, int size, const double *values, int *out_indices, int num_values, int side_left, const int *sorter);
void native_searchsorted_complex64(const float *array, int size, const float *values, int *out_indices, int num_values, int side_left, const int *sorter);

// ----------------------------------------------------------------------------
// Utility operations
// ----------------------------------------------------------------------------
int custom_memcmp(const void *s1, const void *s2, size_t n);
void native_zero_memory(void *ptr, size_t bytes);
void custom_memcpy(void *dest, const void *src, size_t n);

#endif // CUSTOM_SORTING_H
