#include "custom_sorting.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "hwy/contrib/sort/vqsort.h"
#include "hwy/highway.h"

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

static inline int compare_double_fast(double a, double b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int compare_float_fast(float a, float b) {
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

static inline int compare_uint8_inline(uint8_t a, uint8_t b) {
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

#define SORT_NAME tim_fast_double
#define SORT_TYPE double
#define SORT_CMP(x, y) compare_double_fast(x, y)
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

#define SORT_NAME tim_fast_float
#define SORT_TYPE float
#define SORT_CMP(x, y) compare_float_fast(x, y)
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

#define SORT_NAME tim_uint8
#define SORT_TYPE uint8_t
#define SORT_CMP(x, y) compare_uint8_inline(x, y)
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
// Generic Macro Engine for Sorting, Partitioning, and Binary Searching
// ----------------------------------------------------------------------------

#define DEFINE_INSERTION_SORT(NAME, TYPE, CMP) \
static void NAME##_insertion_sort(TYPE *arr, int left, int right) { \
    for (int i = left + 1; i <= right; i++) { \
        TYPE key = arr[i]; \
        int j = i - 1; \
        while (j >= left && CMP(key, arr[j]) < 0) { \
            arr[j + 1] = arr[j]; \
            j--; \
        } \
        arr[j + 1] = key; \
    } \
}

#define DEFINE_QUICKSORT(NAME, TYPE, CMP) \
static void NAME##_quicksort_rec(TYPE *arr, int left, int right) { \
    if (right - left <= 10) { \
        NAME##_insertion_sort(arr, left, right); \
        return; \
    } \
    int mid = left + (right - left) / 2; \
    if (CMP(arr[mid], arr[left]) < 0) { TYPE t = arr[left]; arr[left] = arr[mid]; arr[mid] = t; } \
    if (CMP(arr[right], arr[left]) < 0) { TYPE t = arr[left]; arr[left] = arr[right]; arr[right] = t; } \
    if (CMP(arr[right], arr[mid]) < 0) { TYPE t = arr[mid]; arr[mid] = arr[right]; arr[right] = t; } \
    TYPE pivot = arr[mid]; \
    TYPE t1 = arr[mid]; arr[mid] = arr[right - 1]; arr[right - 1] = t1; \
    int i = left; \
    int j = right - 1; \
    while (1) { \
        while (CMP(arr[++i], pivot) < 0); \
        while (CMP(pivot, arr[--j]) < 0); \
        if (i >= j) break; \
        TYPE t2 = arr[i]; arr[i] = arr[j]; arr[j] = t2; \
    } \
    TYPE t3 = arr[i]; arr[i] = arr[right - 1]; arr[right - 1] = t3; \
    NAME##_quicksort_rec(arr, left, i - 1); \
    NAME##_quicksort_rec(arr, i + 1, right); \
} \
void NAME##_quicksort(TYPE *arr, int size) { \
    if (arr == NULL || size <= 1) return; \
    NAME##_quicksort_rec(arr, 0, size - 1); \
}

#define DEFINE_HEAPSORT(NAME, TYPE, CMP) \
static void NAME##_heapify(TYPE *arr, int n, int i) { \
    int largest = i; \
    int l = 2 * i + 1; \
    int r = 2 * i + 2; \
    if (l < n && CMP(arr[l], arr[largest]) > 0) largest = l; \
    if (r < n && CMP(arr[r], arr[largest]) > 0) largest = r; \
    if (largest != i) { \
        TYPE tmp = arr[i]; \
        arr[i] = arr[largest]; \
        arr[largest] = tmp; \
        NAME##_heapify(arr, n, largest); \
    } \
} \
void NAME##_heapsort(TYPE *arr, int size) { \
    if (arr == NULL || size <= 1) return; \
    for (int i = size / 2 - 1; i >= 0; i--) \
        NAME##_heapify(arr, size, i); \
    for (int i = size - 1; i > 0; i--) { \
        TYPE tmp = arr[0]; \
        arr[0] = arr[i]; \
        arr[i] = tmp; \
        NAME##_heapify(arr, i, 0); \
    } \
}

#define DEFINE_QUICKSELECT(NAME, TYPE, CMP) \
static void NAME##_quickselect(TYPE *arr, int left, int right, int k) { \
    while (left < right) { \
        if (right - left <= 10) { \
            NAME##_insertion_sort(arr, left, right); \
            return; \
        } \
        int pivot_idx = left + (right - left) / 2; \
        TYPE pivot = arr[pivot_idx]; \
        arr[pivot_idx] = arr[right]; \
        arr[right] = pivot; \
        int i = left; \
        for (int j = left; j < right; j++) { \
            if (CMP(arr[j], pivot) < 0) { \
                TYPE tmp = arr[i]; \
                arr[i] = arr[j]; \
                arr[j] = tmp; \
                i++; \
            } \
        } \
        arr[right] = arr[i]; \
        arr[i] = pivot; \
        if (i == k) { \
            return; \
        } else if (i < k) { \
            left = i + 1; \
        } else { \
            right = i - 1; \
        } \
    } \
} \
static void NAME##_quickselect_multi(TYPE *arr, int left, int right, const int *k_list, int k_start, int k_end) { \
    if (k_start > k_end || left >= right) return; \
    int mid_k_idx = k_start + (k_end - k_start) / 2; \
    int k = k_list[mid_k_idx]; \
    NAME##_quickselect(arr, left, right, k); \
    NAME##_quickselect_multi(arr, left, k - 1, k_list, k_start, mid_k_idx - 1); \
    NAME##_quickselect_multi(arr, k + 1, right, k_list, mid_k_idx + 1, k_end); \
} \
void NAME##_partition(TYPE *arr, int size, const int *k_list, int k_size) { \
    if (arr == NULL || size <= 1 || k_list == NULL || k_size <= 0) return; \
    NAME##_quickselect_multi(arr, 0, size - 1, k_list, 0, k_size - 1); \
}

#define DEFINE_ARGQUICKSELECT(NAME, TYPE, CMP) \
static void NAME##_arg_insertion_sort(const TYPE *arr, int *indices, int left, int right) { \
    for (int i = left + 1; i <= right; i++) { \
        int key = indices[i]; \
        int j = i - 1; \
        while (j >= left && CMP(arr[key], arr[indices[j]]) < 0) { \
            indices[j + 1] = indices[j]; \
            j--; \
        } \
        indices[j + 1] = key; \
    } \
} \
static void NAME##_arg_quickselect(const TYPE *arr, int *indices, int left, int right, int k) { \
    while (left < right) { \
        if (right - left <= 10) { \
            NAME##_arg_insertion_sort(arr, indices, left, right); \
            return; \
        } \
        int pivot_idx = left + (right - left) / 2; \
        int pivot_val = indices[pivot_idx]; \
        indices[pivot_idx] = indices[right]; \
        indices[right] = pivot_val; \
        int i = left; \
        for (int j = left; j < right; j++) { \
            if (CMP(arr[indices[j]], arr[pivot_val]) < 0) { \
                int tmp = indices[i]; \
                indices[i] = indices[j]; \
                indices[j] = tmp; \
                i++; \
            } \
        } \
        indices[right] = indices[i]; \
        indices[i] = pivot_val; \
        if (i == k) { \
            return; \
        } else if (i < k) { \
            left = i + 1; \
        } else { \
            right = i - 1; \
        } \
    } \
} \
static void NAME##_arg_quickselect_multi(const TYPE *arr, int *indices, int left, int right, const int *k_list, int k_start, int k_end) { \
    if (k_start > k_end || left >= right) return; \
    int mid_k_idx = k_start + (k_end - k_start) / 2; \
    int k = k_list[mid_k_idx]; \
    NAME##_arg_quickselect(arr, indices, left, right, k); \
    NAME##_arg_quickselect_multi(arr, indices, left, k - 1, k_list, k_start, mid_k_idx - 1); \
    NAME##_arg_quickselect_multi(arr, indices, k + 1, right, k_list, mid_k_idx + 1, k_end); \
} \
void NAME##_argpartition(const TYPE *arr, int *indices, int size, const int *k_list, int k_size) { \
    if (arr == NULL || indices == NULL || size <= 0 || k_list == NULL || k_size <= 0) return; \
    for (int i = 0; i < size; i++) indices[i] = i; \
    NAME##_arg_quickselect_multi(arr, indices, 0, size - 1, k_list, 0, k_size - 1); \
}

#define DEFINE_IND_QUICKSORT(NAME, TYPE, CMP) \
static void NAME##_ind_quicksort_rec(const TYPE *arr, int *indices, int left, int right) { \
    if (right - left <= 10) { \
        NAME##_arg_insertion_sort(arr, indices, left, right); \
        return; \
    } \
    int mid = left + (right - left) / 2; \
    if (CMP(arr[indices[mid]], arr[indices[left]]) < 0) { int t = indices[left]; indices[left] = indices[mid]; indices[mid] = t; } \
    if (CMP(arr[indices[right]], arr[indices[left]]) < 0) { int t = indices[left]; indices[left] = indices[right]; indices[right] = t; } \
    if (CMP(arr[indices[right]], arr[indices[mid]]) < 0) { int t = indices[mid]; indices[mid] = indices[right]; indices[right] = t; } \
    int pivot = indices[mid]; \
    int t1 = indices[mid]; indices[mid] = indices[right - 1]; indices[right - 1] = t1; \
    int i = left; \
    int j = right - 1; \
    while (1) { \
        while (CMP(arr[indices[++i]], arr[pivot]) < 0); \
        while (CMP(arr[pivot], arr[indices[--j]]) < 0); \
        if (i >= j) break; \
        int t2 = indices[i]; indices[i] = indices[j]; indices[j] = t2; \
    } \
    int t3 = indices[i]; indices[i] = indices[right - 1]; indices[right - 1] = t3; \
    NAME##_ind_quicksort_rec(arr, indices, left, i - 1); \
    NAME##_ind_quicksort_rec(arr, indices, i + 1, right); \
} \
void NAME##_ind_quicksort(const TYPE *arr, int *indices, int size) { \
    if (arr == NULL || indices == NULL || size <= 1) return; \
    for (int i = 0; i < size; i++) indices[i] = i; \
    NAME##_ind_quicksort_rec(arr, indices, 0, size - 1); \
}

#define DEFINE_IND_HEAPSORT(NAME, TYPE, CMP) \
static void NAME##_ind_heapify(const TYPE *arr, int *indices, int n, int i) { \
    int largest = i; \
    int l = 2 * i + 1; \
    int r = 2 * i + 2; \
    if (l < n && CMP(arr[indices[l]], arr[indices[largest]]) > 0) largest = l; \
    if (r < n && CMP(arr[indices[r]], arr[indices[largest]]) > 0) largest = r; \
    if (largest != i) { \
        int tmp = indices[i]; \
        indices[i] = indices[largest]; \
        indices[largest] = tmp; \
        NAME##_ind_heapify(arr, indices, n, largest); \
    } \
} \
void NAME##_ind_heapsort(const TYPE *arr, int *indices, int size) { \
    if (arr == NULL || indices == NULL || size <= 1) return; \
    for (int i = 0; i < size; i++) indices[i] = i; \
    for (int i = size / 2 - 1; i >= 0; i--) \
        NAME##_ind_heapify(arr, indices, size, i); \
    for (int i = size - 1; i > 0; i--) { \
        int tmp = indices[0]; \
        indices[0] = indices[i]; \
        indices[i] = tmp; \
        NAME##_ind_heapify(arr, indices, i, 0); \
    } \
}

#define DEFINE_SEARCHSORTED(NAME, TYPE, CMP) \
void NAME##_searchsorted(const TYPE *arr, int size, const TYPE *values, int *out_indices, int num_values, int side_left, const int *sorter) { \
    if (arr == NULL || values == NULL || out_indices == NULL || num_values <= 0) return; \
    for (int v_idx = 0; v_idx < num_values; v_idx++) { \
        TYPE val = values[v_idx]; \
        int low = 0; \
        int high = size; \
        while (low < high) { \
            int mid = low + (high - low) / 2; \
            TYPE mid_val = (sorter != NULL) ? arr[sorter[mid]] : arr[mid]; \
            int comp = CMP(mid_val, val); \
            if (side_left) { \
                if (comp < 0) { \
                    low = mid + 1; \
                } else { \
                    high = mid; \
                } \
            } else { \
                if (comp <= 0) { \
                    low = mid + 1; \
                } else { \
                    high = mid; \
                } \
            } \
        } \
        out_indices[v_idx] = low; \
    } \
}

// ----------------------------------------------------------------------------
// Macro Instantiations
// ----------------------------------------------------------------------------

// double (f64)
DEFINE_INSERTION_SORT(f64, double, compare_double_inline)
DEFINE_QUICKSORT(f64, double, compare_double_inline)
DEFINE_HEAPSORT(f64, double, compare_double_inline)
DEFINE_QUICKSELECT(f64, double, compare_double_inline)
DEFINE_ARGQUICKSELECT(f64, double, compare_double_inline)
DEFINE_IND_QUICKSORT(f64, double, compare_double_inline)
DEFINE_IND_HEAPSORT(f64, double, compare_double_inline)
DEFINE_SEARCHSORTED(f64, double, compare_double_inline)

static void f64_fast_insertion_sort(double *arr, int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        double key = arr[i];
        int j = i - 1;
        while (j >= left && key < arr[j]) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

static void f64_fast_quicksort_rec(double *arr, int left, int right) {
    if (right - left <= 10) {
        f64_fast_insertion_sort(arr, left, right);
        return;
    }
    int mid = left + (right - left) / 2;
    if (arr[mid] < arr[left]) { double t = arr[left]; arr[left] = arr[mid]; arr[mid] = t; }
    if (arr[right] < arr[left]) { double t = arr[left]; arr[left] = arr[right]; arr[right] = t; }
    if (arr[right] < arr[mid]) { double t = arr[mid]; arr[mid] = arr[right]; arr[right] = t; }
    double pivot = arr[mid];
    double t1 = arr[mid]; arr[mid] = arr[right - 1]; arr[right - 1] = t1;
    int i = left;
    int j = right - 1;
    while (1) {
        while (arr[++i] < pivot);
        while (pivot < arr[--j]);
        if (i >= j) break;
        double t2 = arr[i]; arr[i] = arr[j]; arr[j] = t2;
    }
    double t3 = arr[i]; arr[i] = arr[right - 1]; arr[right - 1] = t3;
    f64_fast_quicksort_rec(arr, left, i - 1);
    f64_fast_quicksort_rec(arr, i + 1, right);
}

void f64_fast_quicksort(double *arr, int size) {
    if (arr == NULL || size <= 1) return;
    f64_fast_quicksort_rec(arr, 0, size - 1);
}

static void f64_fast_heapify(double *arr, int n, int i) {
    int largest = i;
    int l = 2 * i + 1;
    int r = 2 * i + 2;
    if (l < n && arr[l] > arr[largest]) largest = l;
    if (r < n && arr[r] > arr[largest]) largest = r;
    if (largest != i) {
        double tmp = arr[i];
        arr[i] = arr[largest];
        arr[largest] = tmp;
        f64_fast_heapify(arr, n, largest);
    }
}

void f64_fast_heapsort(double *arr, int size) {
    if (arr == NULL || size <= 1) return;
    for (int i = size / 2 - 1; i >= 0; i--)
        f64_fast_heapify(arr, size, i);
    for (int i = size - 1; i > 0; i--) {
        double tmp = arr[0];
        arr[0] = arr[i];
        arr[i] = tmp;
        f64_fast_heapify(arr, i, 0);
    }
}


// float (f32)
DEFINE_INSERTION_SORT(f32, float, compare_float_inline)
DEFINE_QUICKSORT(f32, float, compare_float_inline)
DEFINE_HEAPSORT(f32, float, compare_float_inline)
DEFINE_QUICKSELECT(f32, float, compare_float_inline)
DEFINE_ARGQUICKSELECT(f32, float, compare_float_inline)
DEFINE_IND_QUICKSORT(f32, float, compare_float_inline)
DEFINE_IND_HEAPSORT(f32, float, compare_float_inline)
DEFINE_SEARCHSORTED(f32, float, compare_float_inline)

static void f32_fast_insertion_sort(float *arr, int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        float key = arr[i];
        int j = i - 1;
        while (j >= left && key < arr[j]) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

static void f32_fast_quicksort_rec(float *arr, int left, int right) {
    if (right - left <= 10) {
        f32_fast_insertion_sort(arr, left, right);
        return;
    }
    int mid = left + (right - left) / 2;
    if (arr[mid] < arr[left]) { float t = arr[left]; arr[left] = arr[mid]; arr[mid] = t; }
    if (arr[right] < arr[left]) { float t = arr[left]; arr[left] = arr[right]; arr[right] = t; }
    if (arr[right] < arr[mid]) { float t = arr[mid]; arr[mid] = arr[right]; arr[right] = t; }
    float pivot = arr[mid];
    float t1 = arr[mid]; arr[mid] = arr[right - 1]; arr[right - 1] = t1;
    int i = left;
    int j = right - 1;
    while (1) {
        while (arr[++i] < pivot);
        while (pivot < arr[--j]);
        if (i >= j) break;
        float t2 = arr[i]; arr[i] = arr[j]; arr[j] = t2;
    }
    float t3 = arr[i]; arr[i] = arr[right - 1]; arr[right - 1] = t3;
    f32_fast_quicksort_rec(arr, left, i - 1);
    f32_fast_quicksort_rec(arr, i + 1, right);
}

void f32_fast_quicksort(float *arr, int size) {
    if (arr == NULL || size <= 1) return;
    f32_fast_quicksort_rec(arr, 0, size - 1);
}

static void f32_fast_heapify(float *arr, int n, int i) {
    int largest = i;
    int l = 2 * i + 1;
    int r = 2 * i + 2;
    if (l < n && arr[l] > arr[largest]) largest = l;
    if (r < n && arr[r] > arr[largest]) largest = r;
    if (largest != i) {
        float tmp = arr[i];
        arr[i] = arr[largest];
        arr[largest] = tmp;
        f32_fast_heapify(arr, n, largest);
    }
}

void f32_fast_heapsort(float *arr, int size) {
    if (arr == NULL || size <= 1) return;
    for (int i = size / 2 - 1; i >= 0; i--)
        f32_fast_heapify(arr, size, i);
    for (int i = size - 1; i > 0; i--) {
        float tmp = arr[0];
        arr[0] = arr[i];
        arr[i] = tmp;
        f32_fast_heapify(arr, i, 0);
    }
}


// int64 (i64)
DEFINE_INSERTION_SORT(i64, long long, compare_int64_inline)
DEFINE_QUICKSORT(i64, long long, compare_int64_inline)
DEFINE_HEAPSORT(i64, long long, compare_int64_inline)
DEFINE_QUICKSELECT(i64, long long, compare_int64_inline)
DEFINE_ARGQUICKSELECT(i64, long long, compare_int64_inline)
DEFINE_IND_QUICKSORT(i64, long long, compare_int64_inline)
DEFINE_IND_HEAPSORT(i64, long long, compare_int64_inline)
DEFINE_SEARCHSORTED(i64, long long, compare_int64_inline)

// int32 (i32)
DEFINE_INSERTION_SORT(i32, int, compare_int32_inline)
DEFINE_QUICKSORT(i32, int, compare_int32_inline)
DEFINE_HEAPSORT(i32, int, compare_int32_inline)
DEFINE_QUICKSELECT(i32, int, compare_int32_inline)
DEFINE_ARGQUICKSELECT(i32, int, compare_int32_inline)
DEFINE_IND_QUICKSORT(i32, int, compare_int32_inline)
DEFINE_IND_HEAPSORT(i32, int, compare_int32_inline)
DEFINE_SEARCHSORTED(i32, int, compare_int32_inline)

// uint8 (u8)
DEFINE_INSERTION_SORT(u8, uint8_t, compare_uint8_inline)
DEFINE_QUICKSORT(u8, uint8_t, compare_uint8_inline)
DEFINE_HEAPSORT(u8, uint8_t, compare_uint8_inline)
DEFINE_QUICKSELECT(u8, uint8_t, compare_uint8_inline)
DEFINE_ARGQUICKSELECT(u8, uint8_t, compare_uint8_inline)
DEFINE_IND_QUICKSORT(u8, uint8_t, compare_uint8_inline)
DEFINE_IND_HEAPSORT(u8, uint8_t, compare_uint8_inline)
DEFINE_SEARCHSORTED(u8, uint8_t, compare_uint8_inline)

// complex128 (c128)
DEFINE_INSERTION_SORT(c128, complex128_t, compare_complex128_inline)
DEFINE_QUICKSORT(c128, complex128_t, compare_complex128_inline)
DEFINE_HEAPSORT(c128, complex128_t, compare_complex128_inline)
DEFINE_QUICKSELECT(c128, complex128_t, compare_complex128_inline)
DEFINE_ARGQUICKSELECT(c128, complex128_t, compare_complex128_inline)
DEFINE_SEARCHSORTED(c128, complex128_t, compare_complex128_inline)

// complex64 (c64)
DEFINE_INSERTION_SORT(c64, complex64_t, compare_complex64_inline)
DEFINE_QUICKSORT(c64, complex64_t, compare_complex64_inline)
DEFINE_HEAPSORT(c64, complex64_t, compare_complex64_inline)
DEFINE_QUICKSELECT(c64, complex64_t, compare_complex64_inline)
DEFINE_ARGQUICKSELECT(c64, complex64_t, compare_complex64_inline)
DEFINE_SEARCHSORTED(c64, complex64_t, compare_complex64_inline)

// ----------------------------------------------------------------------------
// Public Sorters with Kind Routing
// kind: 0 = quicksort, 1 = mergesort/stable, 2 = heapsort
// ----------------------------------------------------------------------------

void native_sort_double(double *array, int size, int kind) {
    if (array == NULL || size <= 1) return;

    // Segregate NaNs to the end of the array (in-place partitioning)
    int left = 0;
    int right = size - 1;
    while (left <= right) {
        if (isnan(array[left])) {
            double temp = array[left];
            array[left] = array[right];
            array[right] = temp;
            right--;
        } else {
            left++;
        }
    }
    int non_nan_size = right + 1;

    if (non_nan_size <= 1) return;

    if (kind == 0 || kind == 2) {
        hwy::VQSort(array, non_nan_size, hwy::SortAscending());
    } else {
        tim_fast_double_tim_sort(array, non_nan_size);
    }
}

void native_sort_float(float *array, int size, int kind) {
    if (array == NULL || size <= 1) return;

    // Segregate NaNs to the end of the array (in-place partitioning)
    int left = 0;
    int right = size - 1;
    while (left <= right) {
        if (isnan(array[left])) {
            float temp = array[left];
            array[left] = array[right];
            array[right] = temp;
            right--;
        } else {
            left++;
        }
    }
    int non_nan_size = right + 1;

    if (non_nan_size <= 1) return;

    if (kind == 0 || kind == 2) {
        hwy::VQSort(array, non_nan_size, hwy::SortAscending());
    } else {
        tim_fast_float_tim_sort(array, non_nan_size);
    }
}

void native_sort_int64(long long *array, int size, int kind) {
    if (array == NULL || size <= 1) return;
    if (kind == 0) {
        i64_quicksort(array, size);
    } else if (kind == 2) {
        i64_heapsort(array, size);
    } else {
        tim_int64_tim_sort(array, size);
    }
}

void native_sort_int32(int *array, int size, int kind) {
    if (array == NULL || size <= 1) return;
    if (kind == 0) {
        i32_quicksort(array, size);
    } else if (kind == 2) {
        i32_heapsort(array, size);
    } else {
        tim_int32_tim_sort(array, size);
    }
}

void native_sort_uint8(uint8_t *array, int size, int kind) {
    if (array == NULL || size <= 1) return;
    if (kind == 0) {
        u8_quicksort(array, size);
    } else if (kind == 2) {
        u8_heapsort(array, size);
    } else {
        tim_uint8_tim_sort(array, size);
    }
}

void native_sort_complex128(double *array, int size, int kind) {
    if (array == NULL || size <= 1) return;
    if (kind == 0) {
        c128_quicksort((complex128_t *)array, size);
    } else if (kind == 2) {
        c128_heapsort((complex128_t *)array, size);
    } else {
        tim_complex128_tim_sort((complex128_t *)array, size);
    }
}

void native_sort_complex64(float *array, int size, int kind) {
    if (array == NULL || size <= 1) return;
    if (kind == 0) {
        c64_quicksort((complex64_t *)array, size);
    } else if (kind == 2) {
        c64_heapsort((complex64_t *)array, size);
    } else {
        tim_complex64_tim_sort((complex64_t *)array, size);
    }
}

// ----------------------------------------------------------------------------
// Public Argsort Sorters with Kind Routing
// kind: 0 = quicksort, 1 = mergesort/stable, 2 = heapsort
// ----------------------------------------------------------------------------

void native_argsort_double(const double *data, int *indices, int size, int kind) {
    if (data == NULL || indices == NULL || size <= 0) return;
    if (kind == 0) {
        f64_ind_quicksort(data, indices, size);
    } else if (kind == 2) {
        f64_ind_heapsort(data, indices, size);
    } else {
        for (int i = 0; i < size; i++) {
            indices[i] = i;
        }
        global_double_data = data;
        tim_indices_double_tim_sort(indices, size);
        global_double_data = NULL;
    }
}

void native_argsort_float(const float *data, int *indices, int size, int kind) {
    if (data == NULL || indices == NULL || size <= 0) return;
    if (kind == 0) {
        f32_ind_quicksort(data, indices, size);
    } else if (kind == 2) {
        f32_ind_heapsort(data, indices, size);
    } else {
        for (int i = 0; i < size; i++) {
            indices[i] = i;
        }
        global_float_data = data;
        tim_indices_float_tim_sort(indices, size);
        global_float_data = NULL;
    }
}

void native_argsort_int64(const long long *data, int *indices, int size, int kind) {
    if (data == NULL || indices == NULL || size <= 0) return;
    if (kind == 0) {
        i64_ind_quicksort(data, indices, size);
    } else if (kind == 2) {
        i64_ind_heapsort(data, indices, size);
    } else {
        for (int i = 0; i < size; i++) {
            indices[i] = i;
        }
        global_int64_data = data;
        tim_indices_int64_tim_sort(indices, size);
        global_int64_data = NULL;
    }
}

void native_argsort_int32(const int *data, int *indices, int size, int kind) {
    if (data == NULL || indices == NULL || size <= 0) return;
    if (kind == 0) {
        i32_ind_quicksort(data, indices, size);
    } else if (kind == 2) {
        i32_ind_heapsort(data, indices, size);
    } else {
        for (int i = 0; i < size; i++) {
            indices[i] = i;
        }
        global_int32_data = data;
        tim_indices_int32_tim_sort(indices, size);
        global_int32_data = NULL;
    }
}

// ----------------------------------------------------------------------------
// Public Partition Sorters
// ----------------------------------------------------------------------------

void native_partition_double(double *array, int size, const int *k_list, int k_size) {
    f64_partition(array, size, k_list, k_size);
}

void native_partition_float(float *array, int size, const int *k_list, int k_size) {
    f32_partition(array, size, k_list, k_size);
}

void native_partition_int64(long long *array, int size, const int *k_list, int k_size) {
    i64_partition(array, size, k_list, k_size);
}

void native_partition_int32(int *array, int size, const int *k_list, int k_size) {
    i32_partition(array, size, k_list, k_size);
}

void native_partition_complex128(double *array, int size, const int *k_list, int k_size) {
    c128_partition((complex128_t *)array, size, k_list, k_size);
}

void native_partition_complex64(float *array, int size, const int *k_list, int k_size) {
    c64_partition((complex64_t *)array, size, k_list, k_size);
}

// ----------------------------------------------------------------------------
// Public Argpartition Sorters
// ----------------------------------------------------------------------------

void native_argpartition_double(const double *data, int *indices, int size, const int *k_list, int k_size) {
    f64_argpartition(data, indices, size, k_list, k_size);
}

void native_argpartition_float(const float *data, int *indices, int size, const int *k_list, int k_size) {
    f32_argpartition(data, indices, size, k_list, k_size);
}

void native_argpartition_int64(const long long *data, int *indices, int size, const int *k_list, int k_size) {
    i64_argpartition(data, indices, size, k_list, k_size);
}

void native_argpartition_int32(const int *data, int *indices, int size, const int *k_list, int k_size) {
    i32_argpartition(data, indices, size, k_list, k_size);
}

void native_argpartition_complex128(const double *data, int *indices, int size, const int *k_list, int k_size) {
    c128_argpartition((const complex128_t *)data, indices, size, k_list, k_size);
}

void native_argpartition_complex64(const float *data, int *indices, int size, const int *k_list, int k_size) {
    c64_argpartition((const complex64_t *)data, indices, size, k_list, k_size);
}

// ----------------------------------------------------------------------------
// Public Searchsorted Functions
// ----------------------------------------------------------------------------

void native_searchsorted_double(const double *array, int size, const double *values, int *out_indices, int num_values, int side_left, const int *sorter) {
    f64_searchsorted(array, size, values, out_indices, num_values, side_left, sorter);
}

void native_searchsorted_float(const float *array, int size, const float *values, int *out_indices, int num_values, int side_left, const int *sorter) {
    f32_searchsorted(array, size, values, out_indices, num_values, side_left, sorter);
}

void native_searchsorted_int64(const long long *array, int size, const long long *values, int *out_indices, int num_values, int side_left, const int *sorter) {
    i64_searchsorted(array, size, values, out_indices, num_values, side_left, sorter);
}

void native_searchsorted_int32(const int *array, int size, const int *values, int *out_indices, int num_values, int side_left, const int *sorter) {
    i32_searchsorted(array, size, values, out_indices, num_values, side_left, sorter);
}

void native_searchsorted_uint8(const uint8_t *array, int size, const uint8_t *values, int *out_indices, int num_values, int side_left, const int *sorter) {
    u8_searchsorted(array, size, values, out_indices, num_values, side_left, sorter);
}

void native_searchsorted_complex128(const double *array, int size, const double *values, int *out_indices, int num_values, int side_left, const int *sorter) {
    c128_searchsorted((const complex128_t *)array, size, (const complex128_t *)values, out_indices, num_values, side_left, sorter);
}

void native_searchsorted_complex64(const float *array, int size, const float *values, int *out_indices, int num_values, int side_left, const int *sorter) {
    c64_searchsorted((const complex64_t *)array, size, (const complex64_t *)values, out_indices, num_values, side_left, sorter);
}

// ----------------------------------------------------------------------------
// Utility Operations
// ----------------------------------------------------------------------------

int custom_memcmp(const void *s1, const void *s2, size_t n) {
    if (s1 == NULL || s2 == NULL) return s1 == s2 ? 0 : (s1 == NULL ? -1 : 1);
    return memcmp(s1, s2, n);
}

void native_zero_memory(void *ptr, size_t bytes) {
    if (ptr == NULL || bytes <= 0) return;
    memset(ptr, 0, bytes);
}

void custom_memcpy(void *dest, const void *src, size_t n) {
    if (dest == NULL || src == NULL || n <= 0) return;
    memcpy(dest, src, n);
}

void native_collect_nonzero_coords(
    const unsigned char *cond,
    int total_size,
    const int *shape,
    const int *strides,
    int rank,
    int **out_coords
) {
    if (cond == NULL || shape == NULL || strides == NULL || out_coords == NULL || total_size <= 0 || rank <= 0) return;
    int coord[32] = {0};
    int offset = 0;
    int write_idx = 0;

    for (int el = 0; el < total_size; el++) {
        if (cond[offset]) {
            for (int d = 0; d < rank; d++) {
                out_coords[d][write_idx] = coord[d];
            }
            write_idx++;
        }

        // Advance odometer multidimensional walk
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offset += strides[d];
                break;
            }
            coord[d] = 0;
            offset -= (shape[d] - 1) * strides[d];
        }
    }
}

#define DEFINE_TO_BOOL_MASK(NAME, TYPE, COND_EXPR) \
void native_to_bool_mask_##NAME( \
    const void *src_void, \
    int size, \
    const int *shape, \
    const int *strides, \
    int rank, \
    int is_contiguous, \
    unsigned char *dest \
) { \
    if (src_void == NULL || dest == NULL || size <= 0) return; \
    const TYPE *src = (const TYPE *)src_void; \
    if (is_contiguous) { \
        for (int i = 0; i < size; i++) { \
            TYPE val = src[i]; \
            dest[i] = (COND_EXPR) ? 1 : 0; \
        } \
        return; \
    } \
    if (shape == NULL || strides == NULL || rank <= 0) return; \
    int coord[32] = {0}; \
    int offset = 0; \
    for (int i = 0; i < size; i++) { \
        TYPE val = src[offset]; \
        dest[i] = (COND_EXPR) ? 1 : 0; \
        for (int d = rank - 1; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) { \
                offset += strides[d]; \
                break; \
            } \
            coord[d] = 0; \
            offset -= (shape[d] - 1) * strides[d]; \
        } \
    } \
}

DEFINE_TO_BOOL_MASK(double, double, val != 0.0)
DEFINE_TO_BOOL_MASK(float, float, val != 0.0f)
DEFINE_TO_BOOL_MASK(int64, long long, val != 0)
DEFINE_TO_BOOL_MASK(int32, int, val != 0)
DEFINE_TO_BOOL_MASK(complex128, complex128_t, val.real != 0.0 || val.imag != 0.0)
DEFINE_TO_BOOL_MASK(complex64, complex64_t, val.real != 0.0f || val.imag != 0.0f)
DEFINE_TO_BOOL_MASK(uint8, unsigned char, val != 0)
DEFINE_TO_BOOL_MASK(int16, short, val != 0)

#define DEFINE_ARGMINMAX(NAME, TYPE, CMP_OP) \
void native_argminmax_##NAME( \
    const void *src_void, \
    const int *stridesSrc, \
    int *dest, \
    const int *stridesDest, \
    const int *shape, \
    int rank, \
    int axis, \
    int is_max, \
    int is_contiguous \
) { \
    if (src_void == NULL || dest == NULL || shape == NULL || stridesSrc == NULL || stridesDest == NULL || rank <= 0) return; \
    const TYPE *src = (const TYPE *)src_void; \
    if (is_contiguous && axis == -1) { \
        int best_idx = 0; \
        TYPE best_val = src[0]; \
        for (int i = 1; i < shape[0]; i++) { \
            TYPE val = src[i]; \
            if (is_max) { \
                if (val CMP_OP best_val) { \
                    best_val = val; \
                    best_idx = i; \
                } \
            } else { \
                if (best_val CMP_OP val) { \
                    best_val = val; \
                    best_idx = i; \
                } \
            } \
        } \
        dest[0] = best_idx; \
        return; \
    } \
    int dest_size = 1; \
    for (int d = 0; d < rank; d++) { \
        if (d != axis) dest_size *= shape[d]; \
    } \
    int coord_dest[32] = {0}; \
    int strides_dest_clean[32] = {0}; \
    int shape_dest_clean[32] = {0}; \
    int rank_dest = 0; \
    for (int d = 0; d < rank; d++) { \
        if (d != axis) { \
            shape_dest_clean[rank_dest] = shape[d]; \
            strides_dest_clean[rank_dest] = stridesDest[rank_dest]; \
            rank_dest++; \
        } \
    } \
    for (int el = 0; el < dest_size; el++) { \
        int dest_offset = 0; \
        for (int d = 0; d < rank_dest; d++) { \
            dest_offset += coord_dest[d] * strides_dest_clean[d]; \
        } \
        int best_idx = 0; \
        int base_src_offset = 0; \
        int rank_dest_idx = 0; \
        for (int d = 0; d < rank; d++) { \
            if (d != axis) { \
                base_src_offset += coord_dest[rank_dest_idx] * stridesSrc[d]; \
                rank_dest_idx++; \
            } \
        } \
        TYPE best_val = src[base_src_offset]; \
        for (int i = 1; i < shape[axis]; i++) { \
            int src_offset = base_src_offset + i * stridesSrc[axis]; \
            TYPE val = src[src_offset]; \
            if (is_max) { \
                if (val CMP_OP best_val) { \
                    best_val = val; \
                    best_idx = i; \
                } \
            } else { \
                if (best_val CMP_OP val) { \
                    best_val = val; \
                    best_idx = i; \
                } \
            } \
        } \
        dest[dest_offset] = best_idx; \
        if (rank_dest > 0) { \
            for (int d = rank_dest - 1; d >= 0; d--) { \
                coord_dest[d]++; \
                if (coord_dest[d] < shape_dest_clean[d]) break; \
                coord_dest[d] = 0; \
            } \
        } \
    } \
}

DEFINE_ARGMINMAX(double, double, >)
DEFINE_ARGMINMAX(float, float, >)
DEFINE_ARGMINMAX(int64, long long, >)
DEFINE_ARGMINMAX(int32, int, >)
DEFINE_ARGMINMAX(uint8, unsigned char, >)
DEFINE_ARGMINMAX(int16, short, >)

#define DEFINE_COUNT_NONZERO(NAME, TYPE, COND_EXPR) \
void native_count_nonzero_##NAME( \
    const void *src_void, \
    const int *stridesSrc, \
    int *dest, \
    const int *stridesDest, \
    const int *shape, \
    int rank, \
    int axis, \
    int is_contiguous \
) { \
    if (src_void == NULL || dest == NULL || shape == NULL || stridesSrc == NULL || stridesDest == NULL || rank <= 0) return; \
    const TYPE *src = (const TYPE *)src_void; \
    if (is_contiguous && axis == -1) { \
        int count = 0; \
        for (int i = 0; i < shape[0]; i++) { \
            TYPE val = src[i]; \
            if (COND_EXPR) count++; \
        } \
        dest[0] = count; \
        return; \
    } \
    int dest_size = 1; \
    for (int d = 0; d < rank; d++) { \
        if (d != axis) dest_size *= shape[d]; \
    } \
    int coord_dest[32] = {0}; \
    int strides_dest_clean[32] = {0}; \
    int shape_dest_clean[32] = {0}; \
    int rank_dest = 0; \
    for (int d = 0; d < rank; d++) { \
        if (d != axis) { \
            shape_dest_clean[rank_dest] = shape[d]; \
            strides_dest_clean[rank_dest] = stridesDest[rank_dest]; \
            rank_dest++; \
        } \
    } \
    for (int el = 0; el < dest_size; el++) { \
        int dest_offset = 0; \
        for (int d = 0; d < rank_dest; d++) { \
            dest_offset += coord_dest[d] * strides_dest_clean[d]; \
        } \
        int count = 0; \
        int base_src_offset = 0; \
        int rank_dest_idx = 0; \
        for (int d = 0; d < rank; d++) { \
            if (d != axis) { \
                base_src_offset += coord_dest[rank_dest_idx] * stridesSrc[d]; \
                rank_dest_idx++; \
            } \
        } \
        for (int i = 0; i < shape[axis]; i++) { \
            int src_offset = base_src_offset + i * stridesSrc[axis]; \
            TYPE val = src[src_offset]; \
            if (COND_EXPR) count++; \
        } \
        dest[dest_offset] = count; \
        if (rank_dest > 0) { \
            for (int d = rank_dest - 1; d >= 0; d--) { \
                coord_dest[d]++; \
                if (coord_dest[d] < shape_dest_clean[d]) break; \
                coord_dest[d] = 0; \
            } \
        } \
    } \
}

DEFINE_COUNT_NONZERO(double, double, val != 0.0)
DEFINE_COUNT_NONZERO(float, float, val != 0.0f)
DEFINE_COUNT_NONZERO(int64, long long, val != 0)
DEFINE_COUNT_NONZERO(int32, int, val != 0)
DEFINE_COUNT_NONZERO(uint8, unsigned char, val != 0)
DEFINE_COUNT_NONZERO(int16, short, val != 0)

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

int UnpackMaskImpl(const uint8_t *mask_ptr, int size, int *out_indices) {
    const hn::ScalableTag<int32_t> d;
    using Rebind8 = hn::Rebind<uint8_t, decltype(d)>;
    const Rebind8 d8;
    const int L = hn::Lanes(d);

    int count = 0;
    int j = 0;
    int limit = size - L;
    auto v_base = hn::Iota(d, 0);

    for (; j <= limit; j += L) {
        auto mask_bytes = hn::LoadU(d8, mask_ptr + j);
        auto mask_i32 = hn::PromoteTo(d, mask_bytes);
        auto mask = (mask_i32 != hn::Zero(d));
        auto v_index = hn::Add(v_base, hn::Set(d, j));
        count += hn::CompressStore(v_index, mask, d, out_indices + count);
    }
    for (; j < size; j++) {
        if (mask_ptr[j] != 0) {
            out_indices[count++] = j;
        }
    }
    return count;
}

} // namespace HWY_NAMESPACE
} // namespace hwy
HWY_AFTER_NAMESPACE();

extern "C" {
int unpack_mask_c(
    const uint8_t *mask_ptr,
    int size,
    int stride,
    int *out_indices
) {
    if (mask_ptr == NULL || out_indices == NULL || size <= 0) return 0;

    if (stride == 1) {
        return hwy::HWY_NAMESPACE::UnpackMaskImpl(mask_ptr, size, out_indices);
    } else {
        int count = 0;
        for (int j = 0; j < size; j++) {
            if (mask_ptr[j * stride] != 0) {
                out_indices[count++] = j;
            }
        }
        return count;
    }
}
}

extern "C" {
int native_count_mask(const uint8_t *mask, int size) {
    if (mask == NULL || size <= 0) return 0;
    int count = 0;
    for (int i = 0; i < size; i++) {
        if (mask[i] != 0) count++;
    }
    return count;
}

void native_apply_mask(
    int dtype,
    const void *src,
    const uint8_t *mask,
    void *dest,
    int size
) {
    if (src == NULL || mask == NULL || dest == NULL || size <= 0) return;
    
    switch (dtype) {
        case 0: { // float32
            const float *s = (const float *)src;
            float *d = (float *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case 1: { // float64
            const double *s = (const double *)src;
            double *d = (double *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case 2: { // complex64
            struct C64 { float real, imag; };
            const C64 *s = (const C64 *)src;
            C64 *d = (C64 *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case 3: { // complex128
            struct C128 { double real, imag; };
            const C128 *s = (const C128 *)src;
            C128 *d = (C128 *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case 4: { // int32
            const int32_t *s = (const int32_t *)src;
            int32_t *d = (int32_t *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case 5: { // int64
            const int64_t *s = (const int64_t *)src;
            int64_t *d = (int64_t *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case 6: { // uint8
            const uint8_t *s = (const uint8_t *)src;
            uint8_t *d = (uint8_t *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case 7: { // int16
            const int16_t *s = (const int16_t *)src;
            int16_t *d = (int16_t *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
        case 8: { // boolean
            const uint8_t *s = (const uint8_t *)src;
            uint8_t *d = (uint8_t *)dest;
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (mask[i]) d[count++] = s[i];
            }
            break;
        }
    }
}
}


#include <algorithm>
#include <vector>
#include <cmath>

// Lexicographical comparison for complex numbers
template<typename T>
static inline bool comp_complex_impl(T a, T b) {
    bool nan_ar = std::isnan(a.real);
    bool nan_br = std::isnan(b.real);
    if (nan_ar && nan_br) {
        bool nan_ai = std::isnan(a.imag);
        bool nan_bi = std::isnan(b.imag);
        if (nan_ai && nan_bi) return false;
        if (nan_ai) return false;
        if (nan_bi) return true;
        return a.imag < b.imag;
    } else if (nan_ar) {
        return false;
    } else if (nan_br) {
        return true;
    } else if (a.real != b.real) {
        return a.real < b.real;
    }
    
    bool nan_ai = std::isnan(a.imag);
    bool nan_bi = std::isnan(b.imag);
    if (nan_ai && nan_bi) return false;
    if (nan_ai) return false;
    if (nan_bi) return true;
    return a.imag < b.imag;
}

// Equivalence for complex
template<typename T>
static inline bool eq_complex_impl(T a, T b) {
    bool eq_r = (a.real == b.real) || (std::isnan(a.real) && std::isnan(b.real));
    bool eq_i = (a.imag == b.imag) || (std::isnan(a.imag) && std::isnan(b.imag));
    return eq_r && eq_i;
}

static inline bool comp_double_impl(double a, double b) {
    bool nan_a = std::isnan(a);
    bool nan_b = std::isnan(b);
    if (nan_a && nan_b) return false;
    if (nan_a) return false;
    if (nan_b) return true;
    return a < b;
}

static inline bool eq_double_impl(double a, double b) {
    return (a == b) || (std::isnan(a) && std::isnan(b));
}

template<typename T, typename Comp, typename Eq>
int unique_template(const T *src, T *dest, int size,
                    int64_t *out_index, int64_t *out_inverse, int64_t *out_counts,
                    Comp comp, Eq eq) {
    if (size <= 0) return 0;
    
    // If no optional returns, we can optimize by sorting in-place on a copy
    if (out_index == nullptr && out_inverse == nullptr && out_counts == nullptr) {
        memcpy(dest, src, size * sizeof(T));
        std::sort(dest, dest + size, comp);
        // Compact in-place
        int write_idx = 0;
        for (int read_idx = 1; read_idx < size; read_idx++) {
            if (!eq(dest[read_idx], dest[write_idx])) {
                write_idx++;
                dest[write_idx] = dest[read_idx];
            }
        }
        return write_idx + 1;
    }
    
    // We need at least one optional return.
    std::vector<int64_t> idx(size);
    for (int i = 0; i < size; i++) idx[i] = i;
    
    std::stable_sort(idx.begin(), idx.end(), [&](int64_t a, int64_t b) {
        return comp(src[a], src[b]);
    });
    
    int write_idx = 0;
    dest[0] = src[idx[0]];
    if (out_index) out_index[0] = idx[0];
    if (out_inverse) out_inverse[idx[0]] = 0;
    
    int64_t current_count = 1;
    
    for (int read_idx = 1; read_idx < size; read_idx++) {
        if (!eq(src[idx[read_idx]], src[idx[read_idx - 1]])) {
            if (out_counts) out_counts[write_idx] = current_count;
            write_idx++;
            dest[write_idx] = src[idx[read_idx]];
            if (out_index) out_index[write_idx] = idx[read_idx];
            if (out_inverse) out_inverse[idx[read_idx]] = write_idx;
            current_count = 1;
        } else {
            if (out_inverse) out_inverse[idx[read_idx]] = write_idx;
            current_count++;
        }
    }
    if (out_counts) out_counts[write_idx] = current_count;
    
    return write_idx + 1;
}

extern "C" {
int ndarray_unique(const void *src, void *dest, int size, int dtype,
                   int64_t *out_index, int64_t *out_inverse, int64_t *out_counts) {
    if (src == NULL || dest == NULL || size <= 0) return 0;
    
    switch (dtype) {
        case DTYPE_FLOAT64:
            return unique_template<double>(
                (const double *)src, (double *)dest, size,
                out_index, out_inverse, out_counts,
                comp_double_impl, eq_double_impl
            );
        case DTYPE_FLOAT32:
            return unique_template<float>(
                (const float *)src, (float *)dest, size,
                out_index, out_inverse, out_counts,
                [](float a, float b) {
                    bool nan_a = std::isnan(a);
                    bool nan_b = std::isnan(b);
                    if (nan_a && nan_b) return false;
                    if (nan_a) return false;
                    if (nan_b) return true;
                    return a < b;
                },
                [](float a, float b) {
                    return (a == b) || (std::isnan(a) && std::isnan(b));
                }
            );
        case DTYPE_INT32:
            return unique_template<int32_t>(
                (const int32_t *)src, (int32_t *)dest, size,
                out_index, out_inverse, out_counts,
                std::less<int32_t>(), std::equal_to<int32_t>()
            );
        case DTYPE_INT64:
            return unique_template<int64_t>(
                (const int64_t *)src, (int64_t *)dest, size,
                out_index, out_inverse, out_counts,
                std::less<int64_t>(), std::equal_to<int64_t>()
            );
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            return unique_template<uint8_t>(
                (const uint8_t *)src, (uint8_t *)dest, size,
                out_index, out_inverse, out_counts,
                std::less<uint8_t>(), std::equal_to<uint8_t>()
            );
        case DTYPE_COMPLEX128:
            return unique_template<complex128_t>(
                (const complex128_t *)src, (complex128_t *)dest, size,
                out_index, out_inverse, out_counts,
                comp_complex_impl<complex128_t>, eq_complex_impl<complex128_t>
            );
        case DTYPE_COMPLEX64:
            return unique_template<complex64_t>(
                (const complex64_t *)src, (complex64_t *)dest, size,
                out_index, out_inverse, out_counts,
                comp_complex_impl<complex64_t>, eq_complex_impl<complex64_t>
            );
        default:
            return 0;
    }
}
}
