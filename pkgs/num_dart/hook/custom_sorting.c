#include "custom_sorting.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

// ----------------------------------------------------------------------------
// Pure C-to-C Static Callbacks for stdlib qsort
// ----------------------------------------------------------------------------

static int compare_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    int nan_a = isnan(da);
    int nan_b = isnan(db);
    if (nan_a && nan_b) return 0;
    if (nan_a) return 1;
    if (nan_b) return -1;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static int compare_float(const void *a, const void *b) {
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    int nan_a = isnan(fa);
    int nan_b = isnan(fb);
    if (nan_a && nan_b) return 0;
    if (nan_a) return 1;
    if (nan_b) return -1;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

static int compare_int64(const void *a, const void *b) {
    long long ia = *(const long long *)a;
    long long ib = *(const long long *)b;
    if (ia < ib) return -1;
    if (ia > ib) return 1;
    return 0;
}

static int compare_int32(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    if (ia < ib) return -1;
    if (ia > ib) return 1;
    return 0;
}

typedef struct {
    double real;
    double imag;
} complex128_t;

typedef struct {
    float real;
    float imag;
} complex64_t;

static int compare_complex128(const void *a, const void *b) {
    const complex128_t *ca = (const complex128_t *)a;
    const complex128_t *cb = (const complex128_t *)b;
    if (ca->real < cb->real) return -1;
    if (ca->real > cb->real) return 1;
    if (ca->imag < cb->imag) return -1;
    if (ca->imag > cb->imag) return 1;
    return 0;
}

static int compare_complex64(const void *a, const void *b) {
    const complex64_t *ca = (const complex64_t *)a;
    const complex64_t *cb = (const complex64_t *)b;
    if (ca->real < cb->real) return -1;
    if (ca->real > cb->real) return 1;
    if (ca->imag < cb->imag) return -1;
    if (ca->imag > cb->imag) return 1;
    return 0;
}

// ----------------------------------------------------------------------------
// Public Sorter Definitions
// ----------------------------------------------------------------------------

void native_sort_double(double *array, int size) {
    if (array == NULL || size <= 1) return;
    qsort(array, size, sizeof(double), compare_double);
}

void native_sort_float(float *array, int size) {
    if (array == NULL || size <= 1) return;
    qsort(array, size, sizeof(float), compare_float);
}

void native_sort_int64(long long *array, int size) {
    if (array == NULL || size <= 1) return;
    qsort(array, size, sizeof(long long), compare_int64);
}

void native_sort_int32(int *array, int size) {
    if (array == NULL || size <= 1) return;
    qsort(array, size, sizeof(int), compare_int32);
}

void native_sort_complex128(double *array, int size) {
    if (array == NULL || size <= 1) return;
    qsort(array, size, sizeof(double) * 2, compare_complex128);
}

void native_sort_complex64(float *array, int size) {
    if (array == NULL || size <= 1) return;
    qsort(array, size, sizeof(float) * 2, compare_complex64);
}

// ----------------------------------------------------------------------------
// Stable Indirect Sorting / Argsort Definitions
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

static int compare_indices_double(const void *a, const void *b) {
    int idx_a = *(const int *)a;
    int idx_b = *(const int *)b;
    double val_a = global_double_data[idx_a];
    double val_b = global_double_data[idx_b];
    int nan_a = isnan(val_a);
    int nan_b = isnan(val_b);
    if (nan_a && nan_b) return 0;
    if (nan_a) return 1;
    if (nan_b) return -1;
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    if (idx_a < idx_b) return -1;
    if (idx_a > idx_b) return 1;
    return 0;
}

void native_argsort_double(const double *data, int *indices, int size) {
    if (data == NULL || indices == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        indices[i] = i;
    }
    global_double_data = data;
    qsort(indices, size, sizeof(int), compare_indices_double);
    global_double_data = NULL;
}

static int compare_indices_float(const void *a, const void *b) {
    int idx_a = *(const int *)a;
    int idx_b = *(const int *)b;
    float val_a = global_float_data[idx_a];
    float val_b = global_float_data[idx_b];
    int nan_a = isnan(val_a);
    int nan_b = isnan(val_b);
    if (nan_a && nan_b) return 0;
    if (nan_a) return 1;
    if (nan_b) return -1;
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    if (idx_a < idx_b) return -1;
    if (idx_a > idx_b) return 1;
    return 0;
}

void native_argsort_float(const float *data, int *indices, int size) {
    if (data == NULL || indices == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        indices[i] = i;
    }
    global_float_data = data;
    qsort(indices, size, sizeof(int), compare_indices_float);
    global_float_data = NULL;
}

static int compare_indices_int64(const void *a, const void *b) {
    int idx_a = *(const int *)a;
    int idx_b = *(const int *)b;
    long long val_a = global_int64_data[idx_a];
    long long val_b = global_int64_data[idx_b];
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    if (idx_a < idx_b) return -1;
    if (idx_a > idx_b) return 1;
    return 0;
}

void native_argsort_int64(const long long *data, int *indices, int size) {
    if (data == NULL || indices == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        indices[i] = i;
    }
    global_int64_data = data;
    qsort(indices, size, sizeof(int), compare_indices_int64);
    global_int64_data = NULL;
}

static int compare_indices_int32(const void *a, const void *b) {
    int idx_a = *(const int *)a;
    int idx_b = *(const int *)b;
    int val_a = global_int32_data[idx_a];
    int val_b = global_int32_data[idx_b];
    if (val_a < val_b) return -1;
    if (val_a > val_b) return 1;
    if (idx_a < idx_b) return -1;
    if (idx_a > idx_b) return 1;
    return 0;
}

void native_argsort_int32(const int *data, int *indices, int size) {
    if (data == NULL || indices == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        indices[i] = i;
    }
    global_int32_data = data;
    qsort(indices, size, sizeof(int), compare_indices_int32);
    global_int32_data = NULL;
}
