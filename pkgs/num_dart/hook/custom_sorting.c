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
