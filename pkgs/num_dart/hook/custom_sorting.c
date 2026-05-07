#include "custom_sorting.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

// ----------------------------------------------------------------------------
// Struct definitions for Complex number representations
// ----------------------------------------------------------------------------

typedef struct {
    double real;
    double imag;
} complex128_t;

typedef struct {
    float real;
    float imag;
} complex64_t;

// ----------------------------------------------------------------------------
// Thread-Local globals for Argsort (Indirect Sorting) data tracking
// ----------------------------------------------------------------------------

#ifdef _MSC_VER
#define THREAD_LOCAL __declspec(thread)
#else
#define THREAD_LOCAL __thread
#endif

static THREAD_LOCAL const double *global_double_data = NULL;
static THREAD_LOCAL const float *global_float_data = NULL;
static THREAD_LOCAL const long long *global_int64_data = NULL;
static THREAD_LOCAL const int *global_int32_data = NULL;

// ----------------------------------------------------------------------------
// Inlined Comparators for Direct Sorters
// ----------------------------------------------------------------------------

static inline int compare_double_inline(double a, double b) {
    int nan_a = isnan(a);
    int nan_b = isnan(b);
    if (nan_a && nan_b) return 0;
    if (nan_a) return 1;
    if (nan_b) return -1;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int compare_float_inline(float a, float b) {
    int nan_a = isnan(a);
    int nan_b = isnan(b);
    if (nan_a && nan_b) return 0;
    if (nan_a) return 1;
    if (nan_b) return -1;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int compare_int64_inline(long long a, long long b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int compare_int32_inline(int a, int b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int compare_complex128_inline(complex128_t ca, complex128_t cb) {
    if (ca.real < cb.real) return -1;
    if (ca.real > cb.real) return 1;
    if (ca.imag < cb.imag) return -1;
    if (ca.imag > cb.imag) return 1;
    return 0;
}

static inline int compare_complex64_inline(complex64_t ca, complex64_t cb) {
    if (ca.real < cb.real) return -1;
    if (ca.real > cb.real) return 1;
    if (ca.imag < cb.imag) return -1;
    if (ca.imag > cb.imag) return 1;
    return 0;
}

// ----------------------------------------------------------------------------
// Instantiations of Christopher Swenson's TimSort from third_party/timsort
// ----------------------------------------------------------------------------

#define SORT_NAME tim_double
#define SORT_TYPE double
#define SORT_CMP(x, y) compare_double_inline(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_float
#define SORT_TYPE float
#define SORT_CMP(x, y) compare_float_inline(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_int64
#define SORT_TYPE long long
#define SORT_CMP(x, y) compare_int64_inline(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_int32
#define SORT_TYPE int
#define SORT_CMP(x, y) compare_int32_inline(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_complex128
#define SORT_TYPE complex128_t
#define SORT_CMP(x, y) compare_complex128_inline(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_complex64
#define SORT_TYPE complex64_t
#define SORT_CMP(x, y) compare_complex64_inline(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

// ----------------------------------------------------------------------------
// Comparators for Stable Indirect Sorters (Argsort)
// ----------------------------------------------------------------------------

static inline int compare_indices_double_timsort(int idx_a, int idx_b) {
    double val_a = global_double_data[idx_a];
    double val_b = global_double_data[idx_b];
    int nan_a = isnan(val_a);
    int nan_b = isnan(val_b);
    if (nan_a && nan_b) {
        if (idx_a < idx_b) return -1;
        if (idx_a > idx_b) return 1;
        return 0;
    }
    if (nan_a) return 1;
    if (nan_b) return -1;
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    if (idx_a < idx_b) return -1;
    if (idx_a > idx_b) return 1;
    return 0;
}

static inline int compare_indices_float_timsort(int idx_a, int idx_b) {
    float val_a = global_float_data[idx_a];
    float val_b = global_float_data[idx_b];
    int nan_a = isnan(val_a);
    int nan_b = isnan(val_b);
    if (nan_a && nan_b) {
        if (idx_a < idx_b) return -1;
        if (idx_a > idx_b) return 1;
        return 0;
    }
    if (nan_a) return 1;
    if (nan_b) return -1;
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    if (idx_a < idx_b) return -1;
    if (idx_a > idx_b) return 1;
    return 0;
}

static inline int compare_indices_int64_timsort(int idx_a, int idx_b) {
    long long val_a = global_int64_data[idx_a];
    long long val_b = global_int64_data[idx_b];
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    if (idx_a < idx_b) return -1;
    if (idx_a > idx_b) return 1;
    return 0;
}

static inline int compare_indices_int32_timsort(int idx_a, int idx_b) {
    int val_a = global_int32_data[idx_a];
    int val_b = global_int32_data[idx_b];
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    if (idx_a < idx_b) return -1;
    if (idx_a > idx_b) return 1;
    return 0;
}

// ----------------------------------------------------------------------------
// Instantiations of Christopher Swenson's TimSort for Argsort
// ----------------------------------------------------------------------------

#define SORT_NAME tim_indices_double
#define SORT_TYPE int
#define SORT_CMP(x, y) compare_indices_double_timsort(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_indices_float
#define SORT_TYPE int
#define SORT_CMP(x, y) compare_indices_float_timsort(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_indices_int64
#define SORT_TYPE int
#define SORT_CMP(x, y) compare_indices_int64_timsort(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

#define SORT_NAME tim_indices_int32
#define SORT_TYPE int
#define SORT_CMP(x, y) compare_indices_int32_timsort(x, y)
#include "third_party/timsort/timsort.h"
#undef SORT_NAME
#undef SORT_TYPE
#undef SORT_CMP

// ----------------------------------------------------------------------------
// Public Sorters (Routings to Christopher Swenson's TimSort)
// ----------------------------------------------------------------------------

void native_sort_double(double *array, int size) {
    if (array == NULL || size <= 1) return;
    tim_double_tim_sort(array, size);
}

void native_sort_float(float *array, int size) {
    if (array == NULL || size <= 1) return;
    tim_float_tim_sort(array, size);
}

void native_sort_int64(long long *array, int size) {
    if (array == NULL || size <= 1) return;
    tim_int64_tim_sort(array, size);
}

void native_sort_int32(int *array, int size) {
    if (array == NULL || size <= 1) return;
    tim_int32_tim_sort(array, size);
}

void native_sort_complex128(double *array, int size) {
    if (array == NULL || size <= 1) return;
    tim_complex128_tim_sort((complex128_t *)array, size);
}

void native_sort_complex64(float *array, int size) {
    if (array == NULL || size <= 1) return;
    tim_complex64_tim_sort((complex64_t *)array, size);
}

// ----------------------------------------------------------------------------
// Public Argsort Sorters (Routings to Christopher Swenson's TimSort)
// ----------------------------------------------------------------------------

void native_argsort_double(const double *data, int *indices, int size) {
    if (data == NULL || indices == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        indices[i] = i;
    }
    global_double_data = data;
    tim_indices_double_tim_sort(indices, size);
    global_double_data = NULL;
}

void native_argsort_float(const float *data, int *indices, int size) {
    if (data == NULL || indices == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        indices[i] = i;
    }
    global_float_data = data;
    tim_indices_float_tim_sort(indices, size);
    global_float_data = NULL;
}

void native_argsort_int64(const long long *data, int *indices, int size) {
    if (data == NULL || indices == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        indices[i] = i;
    }
    global_int64_data = data;
    tim_indices_int64_tim_sort(indices, size);
    global_int64_data = NULL;
}

void native_argsort_int32(const int *data, int *indices, int size) {
    if (data == NULL || indices == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        indices[i] = i;
    }
    global_int32_data = data;
    tim_indices_int32_tim_sort(indices, size);
    global_int32_data = NULL;
}

int custom_memcmp(const void *s1, const void *s2, size_t n) {
    if (s1 == NULL || s2 == NULL) return s1 == s2 ? 0 : (s1 == NULL ? -1 : 1);
    return memcmp(s1, s2, n);
}

void native_zero_memory(void *ptr, size_t bytes) {
    if (ptr == NULL || bytes <= 0) return;
    memset(ptr, 0, bytes);
}
