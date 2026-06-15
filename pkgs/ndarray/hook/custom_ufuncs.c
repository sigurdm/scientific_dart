/* ============================================================================
 * ndarray CUSTOM NATIVE EXTENSIONS LIBRARY (custom_ufuncs.c)
 * ============================================================================
 * Contains optimized contiguous vector mathematics, multi-dimensional strided
 * odometer walks, custom complex trigonometric algorithms, pocketfft wrappers,
 * and high-speed sorting utilities.
 *
 * Design Rules:
 * 1. Thread-safe and platform-independent standard C99 arithmetic.
 * 2. Zero-allocation sweeps offloading all layout loops to raw pointer space.
 * 3. Systematically aligned logic groupings mapped directly to FFI bindings.
 * ============================================================================
 */

#include "custom_ufuncs.h"
#include <math.h>
#include <stdlib.h>
#include <complex.h>
#include <stdio.h>
#include "custom_sorting.h"

#if defined(_MSC_VER)
#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#else
#define RESTRICT restrict
#endif

#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__)) && !defined(_WIN32)
#define VECTORIZED_TARGETS __attribute__((target_clones("avx512f", "avx2", "sse4.2", "default")))
#else
#define VECTORIZED_TARGETS
#endif

// Macro to define strided binary operations
#define DEFINE_STRIDED_BINARY_OP(name, typeA, typeB, typeResult, op) \
void name(const typeA *a, const int *stridesA, \
          const typeB *b, const int *stridesB, \
          typeResult *res, const int *stridesRes, \
          const int *shape, int rank) { \
    if (a == NULL || b == NULL || res == NULL || rank <= 0 || rank > 8) return; \
    int total_elements = 1; \
    for (int i = 0; i < rank; i++) total_elements *= shape[i]; \
    int coord[8] = {0}; \
    int offsetA = 0, offsetB = 0, offsetRes = 0; \
    for (int el = 0; el < total_elements; el++) { \
        res[offsetRes] = op(a[offsetA], b[offsetB]); \
        for (int d = rank - 1; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) { \
                offsetA += stridesA[d]; \
                offsetB += stridesB[d]; \
                offsetRes += stridesRes[d]; \
                break; \
            } \
            coord[d] = 0; \
            offsetA -= (shape[d] - 1) * stridesA[d]; \
            offsetB -= (shape[d] - 1) * stridesB[d]; \
            offsetRes -= (shape[d] - 1) * stridesRes[d]; \
        } \
    } \
}

#include <complex.h>

static inline cpx_t cpx_add_cast(cpx_t x, cpx_t y) {
    double complex cx = *((double complex*)&x);
    double complex cy = *((double complex*)&y);
    double complex cres = cx + cy;
    return *((cpx_t*)&cres);
}

static inline cpx_t cpx_sub_cast(cpx_t x, cpx_t y) {
    double complex cx = *((double complex*)&x);
    double complex cy = *((double complex*)&y);
    double complex cres = cx - cy;
    return *((cpx_t*)&cres);
}

static inline cpx_t cpx_mul_cast(cpx_t x, cpx_t y) {
    double complex cx = *((double complex*)&x);
    double complex cy = *((double complex*)&y);
    double complex cres = cx * cy;
    return *((cpx_t*)&cres);
}

static inline cpx_t cpx_div_cast(cpx_t x, cpx_t y) {
    double complex cx = *((double complex*)&x);
    double complex cy = *((double complex*)&y);
    double complex cres = cx / cy;
    return *((cpx_t*)&cres);
}

static inline cpx_f_t cpx_add_f_cast(cpx_f_t x, cpx_f_t y) {
    float complex cx = *((float complex*)&x);
    float complex cy = *((float complex*)&y);
    float complex cres = cx + cy;
    return *((cpx_f_t*)&cres);
}

static inline cpx_f_t cpx_sub_f_cast(cpx_f_t x, cpx_f_t y) {
    float complex cx = *((float complex*)&x);
    float complex cy = *((float complex*)&y);
    float complex cres = cx - cy;
    return *((cpx_f_t*)&cres);
}

static inline cpx_f_t cpx_mul_f_cast(cpx_f_t x, cpx_f_t y) {
    float complex cx = *((float complex*)&x);
    float complex cy = *((float complex*)&y);
    float complex cres = cx * cy;
    return *((cpx_f_t*)&cres);
}

static inline cpx_f_t cpx_div_f_cast(cpx_f_t x, cpx_f_t y) {
    float complex cx = *((float complex*)&x);
    float complex cy = *((float complex*)&y);
    float complex cres = cx / cy;
    return *((cpx_f_t*)&cres);
}

// 1. DOUBLE PRECISION (FLOAT64) FLAT CONTIGUOUS KERNELS
// ============================================================================

#define IMPLEMENT_V_BINARY(name, type, op) \
VECTORIZED_TARGETS \
void v_##name##_##type(const type * RESTRICT a, const type * RESTRICT b, type * RESTRICT res, int size) { \
    if (a == NULL || b == NULL || res == NULL || size <= 0) return; \
    for (int i = 0; i < size; i++) { \
        res[i] = a[i] op b[i]; \
    } \
}

#define IMPLEMENT_V_UNARY(name, type, func) \
VECTORIZED_TARGETS \
void v_##name##_##type(const type * RESTRICT src, type * RESTRICT res, int size) { \
    if (src == NULL || res == NULL || size <= 0) return; \
    for (int i = 0; i < size; i++) { \
        res[i] = func(src[i]); \
    } \
}

#define IMPLEMENT_V_BINARY_FUNC(name, type, func) \
VECTORIZED_TARGETS \
void v_##name##_##type(const type * RESTRICT a, const type * RESTRICT b, type * RESTRICT res, int size) { \
    if (a == NULL || b == NULL || res == NULL || size <= 0) return; \
    for (int i = 0; i < size; i++) { \
        res[i] = func(a[i], b[i]); \
    } \
}

IMPLEMENT_V_BINARY(add, double, +)
IMPLEMENT_V_BINARY(sub, double, -)
IMPLEMENT_V_BINARY(mul, double, *)
IMPLEMENT_V_BINARY(div, double, /)

IMPLEMENT_V_UNARY(sin, double, sin)
IMPLEMENT_V_UNARY(cos, double, cos)
IMPLEMENT_V_UNARY(exp, double, exp)
IMPLEMENT_V_UNARY(log, double, log)
IMPLEMENT_V_UNARY(sinh, double, sinh)
IMPLEMENT_V_UNARY(cosh, double, cosh)
IMPLEMENT_V_UNARY(tanh, double, tanh)
IMPLEMENT_V_UNARY(asinh, double, asinh)
IMPLEMENT_V_UNARY(acosh, double, acosh)
IMPLEMENT_V_UNARY(atanh, double, atanh)
IMPLEMENT_V_UNARY(asin, double, asin)
IMPLEMENT_V_UNARY(acos, double, acos)
IMPLEMENT_V_UNARY(atan, double, atan)

IMPLEMENT_V_BINARY_FUNC(atan2, double, atan2)


double r_sum_double(const double *src, int size) {
    if (src == NULL || size <= 0) return 0.0;
    double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
    double acc4 = 0.0, acc5 = 0.0, acc6 = 0.0, acc7 = 0.0;
    
    int i = 0;
    int limit = size - (size % 8);
    
    for (; i < limit; i += 8) {
        acc0 += src[i];
        acc1 += src[i + 1];
        acc2 += src[i + 2];
        acc3 += src[i + 3];
        acc4 += src[i + 4];
        acc5 += src[i + 5];
        acc6 += src[i + 6];
        acc7 += src[i + 7];
    }
    
    double total = (acc0 + acc1) + (acc2 + acc3) + (acc4 + acc5) + (acc6 + acc7);
    for (; i < size; i++) {
        total += src[i];
    }
    return total;
}

double r_prod_double(const double *src, int size) {
    if (src == NULL || size <= 0) return 1.0;
    double acc = 1.0;
    for (int i = 0; i < size; i++) {
        acc *= src[i];
    }
    return acc;
}

double r_mean_double(const double *src, int size) {
    if (src == NULL || size <= 0) return 0.0;
    return r_sum_double(src, size) / (double)size;
}

void s_sum_double(const double *src, const int *stridesSrc,
                  double *dest, const int *stridesDest,
                  const int *shape, int rank, int axis) {
    if (src == NULL || dest == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    int size_axis = shape[axis];
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    
    for (int o = 0; o < outer_size; o++) {
        int offsetRes = 0;
        int offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesDest[targetD];
                }
            }
        }
        
        double sum = 0.0;
        int stride_axis = stridesSrc[axis];
        for (int i = 0; i < size_axis; i++) {
            sum += src[offsetSrc + i * stride_axis];
        }
        dest[offsetRes] = sum;
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_mean_double(const double *src, const int *stridesSrc,
                   double *dest, const int *stridesDest,
                   const int *shape, int rank, int axis) {
    if (src == NULL || dest == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    int size_axis = shape[axis];
    if (size_axis <= 0) return;
    
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    
    for (int o = 0; o < outer_size; o++) {
        int offsetRes = 0;
        int offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesDest[targetD];
                }
            }
        }
        
        double sum = 0.0;
        int stride_axis = stridesSrc[axis];
        for (int i = 0; i < size_axis; i++) {
            sum += src[offsetSrc + i * stride_axis];
        }
        dest[offsetRes] = sum / (double)size_axis;
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

// ============================================================================
// 2. DOUBLE PRECISION (FLOAT64) GENERIC ND STRIDED BROADCASTING KERNELS
// ============================================================================

// Macros to define the boilerplate iterative odometer index advancements!
#define ADVANCE_ODOMETER_LOOP \
    for (int d = rank - 1; d >= 0; d--) { \
        coord[d]++; \
        if (coord[d] < shape[d]) break; \
        coord[d] = 0; \
    }

void s_add_double(const double *a, const int *stridesA,
                  const double *b, const int *stridesB,
                  double *res, const int *stridesRes,
                  const int *shape, int rank) {
    if (a == NULL || b == NULL || res == NULL || rank <= 0 || rank > 8) return;
    
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    int is_a_contiguous = 1, is_b_contiguous = 1, is_res_contiguous = 1;
    int expected_stride = 1;
    for (int i = rank - 1; i >= 0; i--) {
        if (stridesA[i] != expected_stride) is_a_contiguous = 0;
        if (stridesB[i] != expected_stride) is_b_contiguous = 0;
        if (stridesRes[i] != expected_stride) is_res_contiguous = 0;
        expected_stride *= shape[i];
    }

    int is_a_scalar = 1, is_b_scalar = 1;
    for (int i = 0; i < rank; i++) {
        if (stridesA[i] != 0) is_a_scalar = 0;
        if (stridesB[i] != 0) is_b_scalar = 0;
    }

    if (is_res_contiguous) {
        if (is_a_contiguous && is_b_contiguous) {
            for (int i = 0; i < total_elements; i++) res[i] = a[i] + b[i];
            return;
        }
        if (is_a_contiguous && is_b_scalar) {
            double val = b[0];
            for (int i = 0; i < total_elements; i++) res[i] = a[i] + val;
            return;
        }
        if (is_a_scalar && is_b_contiguous) {
            double val = a[0];
            for (int i = 0; i < total_elements; i++) res[i] = val + b[i];
            return;
        }
    }

    int coord[8] = {0};
    int offsetA = 0, offsetB = 0, offsetRes = 0;
    for (int el = 0; el < total_elements; el++) {
        res[offsetRes] = a[offsetA] + b[offsetB];

        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetA += stridesA[d];
                offsetB += stridesB[d];
                offsetRes += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetA -= (shape[d] - 1) * stridesA[d];
            offsetB -= (shape[d] - 1) * stridesB[d];
            offsetRes -= (shape[d] - 1) * stridesRes[d];
        }
    }
}

void s_sub_double(const double *a, const int *stridesA,
                  const double *b, const int *stridesB,
                  double *res, const int *stridesRes,
                  const int *shape, int rank) {
    if (a == NULL || b == NULL || res == NULL || rank <= 0 || rank > 8) return;
    
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    int is_a_contiguous = 1, is_b_contiguous = 1, is_res_contiguous = 1;
    int expected_stride = 1;
    for (int i = rank - 1; i >= 0; i--) {
        if (stridesA[i] != expected_stride) is_a_contiguous = 0;
        if (stridesB[i] != expected_stride) is_b_contiguous = 0;
        if (stridesRes[i] != expected_stride) is_res_contiguous = 0;
        expected_stride *= shape[i];
    }

    int is_a_scalar = 1, is_b_scalar = 1;
    for (int i = 0; i < rank; i++) {
        if (stridesA[i] != 0) is_a_scalar = 0;
        if (stridesB[i] != 0) is_b_scalar = 0;
    }

    if (is_res_contiguous) {
        if (is_a_contiguous && is_b_contiguous) {
            for (int i = 0; i < total_elements; i++) res[i] = a[i] - b[i];
            return;
        }
        if (is_a_contiguous && is_b_scalar) {
            double val = b[0];
            for (int i = 0; i < total_elements; i++) res[i] = a[i] - val;
            return;
        }
        if (is_a_scalar && is_b_contiguous) {
            double val = a[0];
            for (int i = 0; i < total_elements; i++) res[i] = val - b[i];
            return;
        }
    }

    int coord[8] = {0};
    int offsetA = 0, offsetB = 0, offsetRes = 0;
    for (int el = 0; el < total_elements; el++) {
        res[offsetRes] = a[offsetA] - b[offsetB];

        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetA += stridesA[d];
                offsetB += stridesB[d];
                offsetRes += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetA -= (shape[d] - 1) * stridesA[d];
            offsetB -= (shape[d] - 1) * stridesB[d];
            offsetRes -= (shape[d] - 1) * stridesRes[d];
        }
    }
}

void s_mul_double(const double *a, const int *stridesA,
                  const double *b, const int *stridesB,
                  double *res, const int *stridesRes,
                  const int *shape, int rank) {
    if (a == NULL || b == NULL || res == NULL || rank <= 0 || rank > 8) return;
    
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    int is_a_contiguous = 1, is_b_contiguous = 1, is_res_contiguous = 1;
    int expected_stride = 1;
    for (int i = rank - 1; i >= 0; i--) {
        if (stridesA[i] != expected_stride) is_a_contiguous = 0;
        if (stridesB[i] != expected_stride) is_b_contiguous = 0;
        if (stridesRes[i] != expected_stride) is_res_contiguous = 0;
        expected_stride *= shape[i];
    }

    int is_a_scalar = 1, is_b_scalar = 1;
    for (int i = 0; i < rank; i++) {
        if (stridesA[i] != 0) is_a_scalar = 0;
        if (stridesB[i] != 0) is_b_scalar = 0;
    }

    if (is_res_contiguous) {
        if (is_a_contiguous && is_b_contiguous) {
            for (int i = 0; i < total_elements; i++) res[i] = a[i] * b[i];
            return;
        }
        if (is_a_contiguous && is_b_scalar) {
            double val = b[0];
            for (int i = 0; i < total_elements; i++) res[i] = a[i] * val;
            return;
        }
        if (is_a_scalar && is_b_contiguous) {
            double val = a[0];
            for (int i = 0; i < total_elements; i++) res[i] = val * b[i];
            return;
        }
    }

    int coord[8] = {0};
    int offsetA = 0, offsetB = 0, offsetRes = 0;
    for (int el = 0; el < total_elements; el++) {
        res[offsetRes] = a[offsetA] * b[offsetB];

        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetA += stridesA[d];
                offsetB += stridesB[d];
                offsetRes += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetA -= (shape[d] - 1) * stridesA[d];
            offsetB -= (shape[d] - 1) * stridesB[d];
            offsetRes -= (shape[d] - 1) * stridesRes[d];
        }
    }
}

void s_div_double(const double *a, const int *stridesA,
                  const double *b, const int *stridesB,
                  double *res, const int *stridesRes,
                  const int *shape, int rank) {
    if (a == NULL || b == NULL || res == NULL || rank <= 0 || rank > 8) return;
    
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    int is_a_contiguous = 1, is_b_contiguous = 1, is_res_contiguous = 1;
    int expected_stride = 1;
    for (int i = rank - 1; i >= 0; i--) {
        if (stridesA[i] != expected_stride) is_a_contiguous = 0;
        if (stridesB[i] != expected_stride) is_b_contiguous = 0;
        if (stridesRes[i] != expected_stride) is_res_contiguous = 0;
        expected_stride *= shape[i];
    }

    int is_a_scalar = 1, is_b_scalar = 1;
    for (int i = 0; i < rank; i++) {
        if (stridesA[i] != 0) is_a_scalar = 0;
        if (stridesB[i] != 0) is_b_scalar = 0;
    }

    if (is_res_contiguous) {
        if (is_a_contiguous && is_b_contiguous) {
            for (int i = 0; i < total_elements; i++) res[i] = a[i] / b[i];
            return;
        }
        if (is_a_contiguous && is_b_scalar) {
            double val = b[0];
            for (int i = 0; i < total_elements; i++) res[i] = a[i] / val;
            return;
        }
        if (is_a_scalar && is_b_contiguous) {
            double val = a[0];
            for (int i = 0; i < total_elements; i++) res[i] = val / b[i];
            return;
        }
    }

    int coord[8] = {0};
    int offsetA = 0, offsetB = 0, offsetRes = 0;
    for (int el = 0; el < total_elements; el++) {
        res[offsetRes] = a[offsetA] / b[offsetB];

        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetA += stridesA[d];
                offsetB += stridesB[d];
                offsetRes += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetA -= (shape[d] - 1) * stridesA[d];
            offsetB -= (shape[d] - 1) * stridesB[d];
            offsetRes -= (shape[d] - 1) * stridesRes[d];
        }
    }
}

// ============================================================================
// 3. COMPLEX128 VECTOR KERNELS (CONTIGUOUS & STRIDED)
// ============================================================================

void v_add_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int size) {
    if (a == NULL || b == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i].r = a[i].r + b[i].r;
        res[i].i = a[i].i + b[i].i;
    }
}

void v_sub_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int size) {
    if (a == NULL || b == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i].r = a[i].r - b[i].r;
        res[i].i = a[i].i - b[i].i;
    }
}

void v_mul_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int size) {
    if (a == NULL || b == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        double r1 = a[i].r, i1 = a[i].i;
        double r2 = b[i].r, i2 = b[i].i;
        // Foil complex multiplication: (r1+i1*i)*(r2+i2*i) = (r1*r2 - i1*i2) + i*(r1*i2 + i1*r2)
        res[i].r = r1 * r2 - i1 * i2;
        res[i].i = r1 * i2 + i1 * r2;
    }
}

void v_div_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int size) {
    if (a == NULL || b == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        double r1 = a[i].r, i1 = a[i].i;
        double r2 = b[i].r, i2 = b[i].i;
        // Complex division rationalization:
        double denom = r2 * r2 + i2 * i2;
        if (denom == 0.0) {
            res[i].r = NAN;
            res[i].i = NAN;
        } else {
            res[i].r = (r1 * r2 + i1 * i2) / denom;
            res[i].i = (i1 * r2 - r1 * i2) / denom;
        }
    }
}

void s_add_complex(const cpx_t *a, const int *stridesA,
                  const cpx_t *b, const int *stridesB,
                  cpx_t *res, const int *stridesRes,
                  const int *shape, int rank) {
    if (a == NULL || b == NULL || res == NULL || rank <= 0 || rank > 8) return;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    int coord[8] = {0};
    int offsetA = 0, offsetB = 0, offsetRes = 0;
    for (int el = 0; el < total_elements; el++) {
        res[offsetRes].r = a[offsetA].r + b[offsetB].r;
        res[offsetRes].i = a[offsetA].i + b[offsetB].i;

        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetA += stridesA[d];
                offsetB += stridesB[d];
                offsetRes += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetA -= (shape[d] - 1) * stridesA[d];
            offsetB -= (shape[d] - 1) * stridesB[d];
            offsetRes -= (shape[d] - 1) * stridesRes[d];
        }
    }
}

void s_sub_complex(const cpx_t *a, const int *stridesA,
                  const cpx_t *b, const int *stridesB,
                  cpx_t *res, const int *stridesRes,
                  const int *shape, int rank) {
    if (a == NULL || b == NULL || res == NULL || rank <= 0 || rank > 8) return;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    int coord[8] = {0};
    int offsetA = 0, offsetB = 0, offsetRes = 0;
    for (int el = 0; el < total_elements; el++) {
        res[offsetRes].r = a[offsetA].r - b[offsetB].r;
        res[offsetRes].i = a[offsetA].i - b[offsetB].i;

        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetA += stridesA[d];
                offsetB += stridesB[d];
                offsetRes += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetA -= (shape[d] - 1) * stridesA[d];
            offsetB -= (shape[d] - 1) * stridesB[d];
            offsetRes -= (shape[d] - 1) * stridesRes[d];
        }
    }
}

void s_mul_complex(const cpx_t *a, const int *stridesA,
                  const cpx_t *b, const int *stridesB,
                  cpx_t *res, const int *stridesRes,
                  const int *shape, int rank) {
    if (a == NULL || b == NULL || res == NULL || rank <= 0 || rank > 8) return;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    int coord[8] = {0};
    int offsetA = 0, offsetB = 0, offsetRes = 0;
    for (int el = 0; el < total_elements; el++) {
        double r1 = a[offsetA].r, i1 = a[offsetA].i;
        double r2 = b[offsetB].r, i2 = b[offsetB].i;
        res[offsetRes].r = r1 * r2 - i1 * i2;
        res[offsetRes].i = r1 * i2 + i1 * r2;

        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetA += stridesA[d];
                offsetB += stridesB[d];
                offsetRes += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetA -= (shape[d] - 1) * stridesA[d];
            offsetB -= (shape[d] - 1) * stridesB[d];
            offsetRes -= (shape[d] - 1) * stridesRes[d];
        }
    }
}

void s_div_complex(const cpx_t *a, const int *stridesA,
                  const cpx_t *b, const int *stridesB,
                  cpx_t *res, const int *stridesRes,
                  const int *shape, int rank) {
    if (a == NULL || b == NULL || res == NULL || rank <= 0 || rank > 8) return;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    int coord[8] = {0};
    int offsetA = 0, offsetB = 0, offsetRes = 0;
    for (int el = 0; el < total_elements; el++) {
        double r1 = a[offsetA].r, i1 = a[offsetA].i;
        double r2 = b[offsetB].r, i2 = b[offsetB].i;
        double denom = r2 * r2 + i2 * i2;
        if (denom == 0.0) {
            res[offsetRes].r = NAN;
            res[offsetRes].i = NAN;
        } else {
            res[offsetRes].r = (r1 * r2 + i1 * i2) / denom;
            res[offsetRes].i = (i1 * r2 - r1 * i2) / denom;
        }

        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetA += stridesA[d];
                offsetB += stridesB[d];
                offsetRes += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetA -= (shape[d] - 1) * stridesA[d];
            offsetB -= (shape[d] - 1) * stridesB[d];
            offsetRes -= (shape[d] - 1) * stridesRes[d];
        }
    }
}

// ============================================================================
// 4. SINGLE PRECISION (FLOAT32) VECTOR CONTIGUOUS KERNELS
// ============================================================================

void v_add_float(const float *a, const float *b, float *res, int size) {
    if (a == NULL || b == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = a[i] + b[i];
    }
}

void v_sub_float(const float *a, const float *b, float *res, int size) {
    if (a == NULL || b == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = a[i] - b[i];
    }
}

void v_mul_float(const float *a, const float *b, float *res, int size) {
    if (a == NULL || b == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = a[i] * b[i];
    }
}

void v_div_float(const float *a, const float *b, float *res, int size) {
    if (a == NULL || b == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = a[i] / b[i];
    }
}

void v_sin_float(const float *src, float *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = sinf(src[i]);
    }
}

void v_cos_float(const float *src, float *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = cosf(src[i]);
    }
}

void v_exp_float(const float *src, float *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = expf(src[i]);
    }
}

void v_log_float(const float *src, float *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = logf(src[i]);
    }
}

void v_sinh_float(const float *src, float *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = sinhf(src[i]);
    }
}

void v_cosh_float(const float *src, float *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = coshf(src[i]);
    }
}

void v_tanh_float(const float *src, float *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = tanhf(src[i]);
    }
}

void v_asinh_float(const float *src, float *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = asinhf(src[i]);
    }
}

void v_acosh_float(const float *src, float *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = acoshf(src[i]);
    }
}

void v_atanh_float(const float *src, float *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = atanhf(src[i]);
    }
}

void v_asin_float(const float *src, float *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = asinf(src[i]);
    }
}

void v_acos_float(const float *src, float *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = acosf(src[i]);
    }
}

void v_atan_float(const float *src, float *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = atanf(src[i]);
    }
}

void v_atan2_float(const float *y, const float *x, float *res, int size) {
    if (y == NULL || x == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = atan2f(y[i], x[i]);
    }
}


float r_sum_float(const float *src, int size) {
    if (src == NULL || size <= 0) return 0.0f;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    
    int i = 0;
    int limit = size - (size % 8);
    
    for (; i < limit; i += 8) {
        acc0 += src[i];
        acc1 += src[i + 1];
        acc2 += src[i + 2];
        acc3 += src[i + 3];
        acc4 += src[i + 4];
        acc5 += src[i + 5];
        acc6 += src[i + 6];
        acc7 += src[i + 7];
    }
    
    float total = (acc0 + acc1) + (acc2 + acc3) + (acc4 + acc5) + (acc6 + acc7);
    for (; i < size; i++) {
        total += src[i];
    }
    return total;
}

float r_prod_float(const float *src, int size) {
    if (src == NULL || size <= 0) return 1.0f;
    float acc = 1.0f;
    for (int i = 0; i < size; i++) {
        acc *= src[i];
    }
    return acc;
}

float r_mean_float(const float *src, int size) {
    if (src == NULL || size <= 0) return 0.0f;
    return r_sum_float(src, size) / (float)size;
}

void s_sum_float(const float *src, const int *stridesSrc,
                 float *dest, const int *stridesDest,
                 const int *shape, int rank, int axis) {
    if (src == NULL || dest == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    int size_axis = shape[axis];
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    
    for (int o = 0; o < outer_size; o++) {
        int offsetRes = 0;
        int offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesDest[targetD];
                }
            }
        }
        
        float sum = 0.0f;
        int stride_axis = stridesSrc[axis];
        for (int i = 0; i < size_axis; i++) {
            sum += src[offsetSrc + i * stride_axis];
        }
        dest[offsetRes] = sum;
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_mean_float(const float *src, const int *stridesSrc,
                  float *dest, const int *stridesDest,
                  const int *shape, int rank, int axis) {
    if (src == NULL || dest == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    int size_axis = shape[axis];
    if (size_axis <= 0) return;
    
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    
    for (int o = 0; o < outer_size; o++) {
        int offsetRes = 0;
        int offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesDest[targetD];
                }
            }
        }
        
        float sum = 0.0f;
        int stride_axis = stridesSrc[axis];
        for (int i = 0; i < size_axis; i++) {
            sum += src[offsetSrc + i * stride_axis];
        }
        dest[offsetRes] = sum / (float)size_axis;
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

// ============================================================================
// 5. ADDITIONAL MATH, ROUNDING & CLIPPING KERNELS (SIMD AUTOVECTORIZABLE)
// ============================================================================

void v_sqrt_double(const double *src, double *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = sqrt(src[i]);
    }
}

void v_tan_double(const double *src, double *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = tan(src[i]);
    }
}

void v_abs_double(const double *src, double *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = fabs(src[i]);
    }
}

void v_ceil_double(const double *src, double *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = ceil(src[i]);
    }
}

void v_floor_double(const double *src, double *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = floor(src[i]);
    }
}

void v_round_double(const double *src, double *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = round(src[i]);
    }
}

void v_clip_double(const double *src, double *res, double min_val, double max_val, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        double val = src[i];
        if (val < min_val) val = min_val;
        if (val > max_val) val = max_val;
        res[i] = val;
    }
}

void v_sqrt_float(const float *src, float *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = sqrtf(src[i]);
    }
}

void v_tan_float(const float *src, float *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = tanf(src[i]);
    }
}

void v_abs_float(const float *src, float *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = fabsf(src[i]);
    }
}

void v_ceil_float(const float *src, float *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = ceilf(src[i]);
    }
}

void v_floor_float(const float *src, float *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = floorf(src[i]);
    }
}

void v_round_float(const float *src, float *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = roundf(src[i]);
    }
}

void v_clip_float(const float *src, float *res, float min_val, float max_val, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        float val = src[i];
        if (val < min_val) val = min_val;
        if (val > max_val) val = max_val;
        res[i] = val;
    }
}

// ============================================================================
// 6. GENERIC ND STRIDED BROADCASTING TERNARY "where" KERNELS (RANK <= 8)
// ============================================================================

#define DEFINE_S_WHERE(NAME, TYPE) \
void s_where_##NAME(const unsigned char *cond, const int *stridesCond, \
                    const TYPE *x, const int *stridesX, \
                    const TYPE *y, const int *stridesY, \
                    TYPE *res, const int *stridesRes, \
                    const int *shape, int rank) { \
    if (cond == NULL || x == NULL || y == NULL || res == NULL || rank < 0 || rank > 8) return; \
    if (rank == 0) { \
        res[0] = cond[0] ? x[0] : y[0]; \
        return; \
    } \
    int is_contiguous = 1; \
    int expected_stride = 1; \
    for (int i = rank - 1; i >= 0; i--) { \
        if (stridesCond[i] != expected_stride || \
            stridesX[i] != expected_stride || \
            stridesY[i] != expected_stride || \
            stridesRes[i] != expected_stride) { \
            is_contiguous = 0; \
            break; \
        } \
        expected_stride *= shape[i]; \
    } \
    int total_elements = 1; \
    for (int i = 0; i < rank; i++) total_elements *= shape[i]; \
    if (is_contiguous) { \
        for (int i = 0; i < total_elements; i++) { \
            res[i] = cond[i] ? x[i] : y[i]; \
        } \
        return; \
    } \
    int coord[8] = {0}; \
    int offsetCond = 0, offsetX = 0, offsetY = 0, offsetRes = 0; \
    for (int el = 0; el < total_elements; el++) { \
        res[offsetRes] = cond[offsetCond] ? x[offsetX] : y[offsetY]; \
        for (int d = rank - 1; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) { \
                offsetCond += stridesCond[d]; \
                offsetX    += stridesX[d]; \
                offsetY    += stridesY[d]; \
                offsetRes  += stridesRes[d]; \
                break; \
            } \
            coord[d] = 0; \
            offsetCond -= (shape[d] - 1) * stridesCond[d]; \
            offsetX    -= (shape[d] - 1) * stridesX[d]; \
            offsetY    -= (shape[d] - 1) * stridesY[d]; \
            offsetRes  -= (shape[d] - 1) * stridesRes[d]; \
        } \
    } \
}

DEFINE_S_WHERE(double, double)
DEFINE_S_WHERE(float, float)
DEFINE_S_WHERE(int64, int64_t)
DEFINE_S_WHERE(int32, int32_t)
DEFINE_S_WHERE(uint8, uint8_t)
DEFINE_S_WHERE(int16, int16_t)
DEFINE_S_WHERE(complex128, cpx_t)
DEFINE_S_WHERE(complex64, cpx_f_t)

#undef DEFINE_S_WHERE

static inline uint64_t splitmix64(uint64_t *x) {
    uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline void xoshiro256_seed(uint64_t seed, uint64_t s[4]) {
    uint64_t x = seed;
    s[0] = splitmix64(&x);
    s[1] = splitmix64(&x);
    s[2] = splitmix64(&x);
    s[3] = splitmix64(&x);
}

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoshiro256_next(uint64_t s[4]) {
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = rotl(s[3], 45);

    return result;
}

void v_normal_double(double *res, int size, double loc, double scale, unsigned long long seed) {
    if (res == NULL || size <= 0 || scale <= 0.0) return;

    uint64_t s[4];
    xoshiro256_seed(seed, s);

    int i = 0;
    while (i < size) {
        double u1;
        do {
            u1 = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);
        } while (u1 == 0.0);

        double u2 = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);

        double mag = scale * sqrt(-2.0 * log(u1));
        double angle = 2.0 * M_PI * u2;

        res[i] = loc + mag * cos(angle);
        if (i + 1 < size) {
            res[i + 1] = loc + mag * sin(angle);
        }
        i += 2;
    }
}

void v_normal_float(float *res, int size, float loc, float scale, unsigned long long seed) {
    if (res == NULL || size <= 0 || scale <= 0.0f) return;

    uint64_t s[4];
    xoshiro256_seed(seed, s);

    int i = 0;
    while (i < size) {
        float u1;
        do {
            u1 = (float)((double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0));
        } while (u1 == 0.0f);

        float u2 = (float)((double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0));

        float mag = scale * sqrtf(-2.0f * logf(u1));
        float angle = 2.0f * (float)M_PI * u2;

        res[i] = loc + mag * cosf(angle);
        if (i + 1 < size) {
            res[i + 1] = loc + mag * sinf(angle);
        }
        i += 2;
    }
}

void v_uniform_double(double *res, int size, unsigned long long seed) {
    if (res == NULL || size <= 0) return;

    uint64_t s[4];
    xoshiro256_seed(seed, s);

    for (int i = 0; i < size; i++) {
        res[i] = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);
    }
}

void v_uniform_float(float *res, int size, unsigned long long seed) {
    if (res == NULL || size <= 0) return;

    uint64_t s[4];
    xoshiro256_seed(seed, s);

    for (int i = 0; i < size; i++) {
        res[i] = (float)((double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0));
    }
}

#define IMPLEMENT_V_RANDINT(TYPE_NAME, TYPE, BOUNDS_TYPE, RANGE_TYPE) \
void v_randint_##TYPE_NAME(TYPE *res, int size, BOUNDS_TYPE low, BOUNDS_TYPE high, unsigned long long seed) { \
    if (res == NULL || size <= 0 || low >= high) return; \
    uint64_t s[4]; \
    xoshiro256_seed(seed, s); \
    RANGE_TYPE range = (RANGE_TYPE)high - (RANGE_TYPE)low; \
    for (int i = 0; i < size; i++) { \
        res[i] = (TYPE)(low + (RANGE_TYPE)(xoshiro256_next(s) % (unsigned long long)range)); \
    } \
}

IMPLEMENT_V_RANDINT(int64, int64_t, int64_t, int64_t)
IMPLEMENT_V_RANDINT(int32, int32_t, int32_t, int32_t)
IMPLEMENT_V_RANDINT(int16, int16_t, int, int)
IMPLEMENT_V_RANDINT(uint8, uint8_t, int, int)

#define IMPLEMENT_V_FILL(TYPE_NAME, TYPE) \
VECTORIZED_TARGETS \
void v_fill_##TYPE_NAME(TYPE * RESTRICT res, TYPE value, int size) { \
    if (res == NULL || size <= 0) return; \
    for (int i = 0; i < size; i++) { \
        res[i] = value; \
    } \
}

IMPLEMENT_V_FILL(double, double)
IMPLEMENT_V_FILL(float, float)
IMPLEMENT_V_FILL(int64, int64_t)
IMPLEMENT_V_FILL(int32, int32_t)

void v_linspace_double(double *res, double start, double step, int size) {
    if (res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = start + i * step;
    }
}

void v_linspace_float(float *res, float start, float step, int size) {
    if (res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = start + i * step;
    }
}

void v_linspace_complex128(cpx_t *res, double startR, double startI, double stepR, double stepI, int size) {
    if (res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i].r = startR + i * stepR;
        res[i].i = startI + i * stepI;
    }
}

void v_linspace_complex64(cpx_f_t *res, float startR, float startI, float stepR, float stepI, int size) {
    if (res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i].r = startR + i * stepR;
        res[i].i = startI + i * stepI;
    }
}

void v_linspace_int64(int64_t *res, double start, double step, int size) {
    if (res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = (int64_t)(start + i * step);
    }
}

void v_linspace_int32(int32_t *res, double start, double step, int size) {
    if (res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = (int32_t)(start + i * step);
    }
}

void v_linspace_int16(int16_t *res, double start, double step, int size) {
    if (res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = (int16_t)(start + i * step);
    }
}

void v_linspace_uint8(uint8_t *res, double start, double step, int size) {
    if (res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = (uint8_t)(start + i * step);
    }
}

void v_logspace_double(double *res, double start, double step, double base, int size) {
    if (res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = pow(base, start + i * step);
    }
}

void v_logspace_float(float *res, float start, float step, float base, int size) {
    if (res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = powf(base, start + i * step);
    }
}

void v_logspace_complex128(cpx_t *res, double startR, double startI, double stepR, double stepI, double baseR, double baseI, int size) {
    if (res == NULL || size <= 0) return;
    double complex cbase = baseR + baseI * I;
    for (int i = 0; i < size; i++) {
        double complex cexp = (startR + i * stepR) + (startI + i * stepI) * I;
        double complex cres = cpow(cbase, cexp);
        res[i].r = creal(cres);
        res[i].i = cimag(cres);
    }
}

void v_logspace_complex64(cpx_f_t *res, float startR, float startI, float stepR, float stepI, float baseR, float baseI, int size) {
    if (res == NULL || size <= 0) return;
    float complex cbase = baseR + baseI * I;
    for (int i = 0; i < size; i++) {
        float complex cexp = (startR + i * stepR) + (startI + i * stepI) * I;
        float complex cres = cpowf(cbase, cexp);
        res[i].r = crealf(cres);
        res[i].i = cimagf(cres);
    }
}

void v_geomspace_double(double *res, double logStart, double step, double sign, int size) {
    if (res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = sign * pow(10.0, logStart + i * step);
    }
}

void v_geomspace_float(float *res, float logStart, float step, float sign, int size) {
    if (res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = sign * powf(10.0f, logStart + i * step);
    }
}

void v_geomspace_complex128(cpx_t *res, double logStartR, double logStartI, double stepR, double stepI, int size) {
    if (res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        double complex cexp = (logStartR + i * stepR) + (logStartI + i * stepI) * I;
        double complex cres = cpow(10.0, cexp);
        res[i].r = creal(cres);
        res[i].i = cimag(cres);
    }
}

void v_geomspace_complex64(cpx_f_t *res, float logStartR, float logStartI, float stepR, float stepI, int size) {
    if (res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        float complex cexp = (logStartR + i * stepR) + (logStartI + i * stepI) * I;
        float complex cres = cpowf(10.0f, cexp);
        res[i].r = crealf(cres);
        res[i].i = cimagf(cres);
    }
}

// Helper functions to perform rolling FNV-1a hash updates
static inline void hash_double(uint32_t *hash, double val) {
    if (val == 0.0) {
        val = 0.0;
    }
    if (val != val) { // isNaN
        const uint64_t nan_bits = 0x7ff8000000000000ULL;
        const uint8_t *bytes = (const uint8_t*)&nan_bits;
        for (int i = 0; i < 8; i++) {
            *hash ^= bytes[i];
            *hash *= 16777619U;
        }
        return;
    }
    const uint8_t *bytes = (const uint8_t*)&val;
    for (int i = 0; i < 8; i++) {
        *hash ^= bytes[i];
        *hash *= 16777619U;
    }
}

static inline void hash_float(uint32_t *hash, float val) {
    if (val == 0.0f) {
        val = 0.0f;
    }
    if (val != val) { // isNaN
        const uint32_t nan_bits = 0x7fc00000U;
        const uint8_t *bytes = (const uint8_t*)&nan_bits;
        for (int i = 0; i < 4; i++) {
            *hash ^= bytes[i];
            *hash *= 16777619U;
        }
        return;
    }
    const uint8_t *bytes = (const uint8_t*)&val;
    for (int i = 0; i < 4; i++) {
        *hash ^= bytes[i];
        *hash *= 16777619U;
    }
}

static inline void hash_int64(uint32_t *hash, int64_t val) {
    const uint8_t *bytes = (const uint8_t*)&val;
    for (int i = 0; i < 8; i++) {
        *hash ^= bytes[i];
        *hash *= 16777619U;
    }
}

static inline void hash_int32(uint32_t *hash, int32_t val) {
    const uint8_t *bytes = (const uint8_t*)&val;
    for (int i = 0; i < 4; i++) {
        *hash ^= bytes[i];
        *hash *= 16777619U;
    }
}

static inline void hash_boolean(uint32_t *hash, uint8_t val) {
    uint8_t b = val ? 1 : 0;
    *hash ^= b;
    *hash *= 16777619U;
}

// ============================================================================
// 7. NATIVE C HIGH-SPEED STRIDED FLATTENING/COPYING KERNELS
// ============================================================================

void s_flatten_double(const double *src, const int *stridesSrc, double *dest, const int *shape, int rank) {
    if (src == NULL || dest == NULL || rank < 0 || rank > 8) return;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    if (rank == 0) {
        dest[0] = src[0];
        return;
    }
    int coord[8] = {0};
    int offsetSrc = 0;
    for (int el = 0; el < total_elements; el++) {
        dest[el] = src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
        }
    }
}

void s_flatten_float(const float *src, const int *stridesSrc, float *dest, const int *shape, int rank) {
    if (src == NULL || dest == NULL || rank < 0 || rank > 8) return;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    if (rank == 0) {
        dest[0] = src[0];
        return;
    }
    int coord[8] = {0};
    int offsetSrc = 0;
    for (int el = 0; el < total_elements; el++) {
        dest[el] = src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
        }
    }
}

void s_flatten_int64(const int64_t *src, const int *stridesSrc, int64_t *dest, const int *shape, int rank) {
    if (src == NULL || dest == NULL || rank < 0 || rank > 8) return;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    if (rank == 0) {
        dest[0] = src[0];
        return;
    }
    int coord[8] = {0};
    int offsetSrc = 0;
    for (int el = 0; el < total_elements; el++) {
        dest[el] = src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
        }
    }
}

void s_flatten_int32(const int32_t *src, const int *stridesSrc, int32_t *dest, const int *shape, int rank) {
    if (src == NULL || dest == NULL || rank < 0 || rank > 8) return;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    if (rank == 0) {
        dest[0] = src[0];
        return;
    }
    int coord[8] = {0};
    int offsetSrc = 0;
    for (int el = 0; el < total_elements; el++) {
        dest[el] = src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
        }
    }
}

void s_flatten_complex128(const double *src, const int *stridesSrc, double *dest, const int *shape, int rank) {
    if (src == NULL || dest == NULL || rank < 0 || rank > 8) return;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    if (rank == 0) {
        dest[0] = src[0];
        dest[1] = src[1];
        return;
    }
    const cpx_t *c_src = (const cpx_t*)src;
    cpx_t *c_dest = (cpx_t*)dest;
    int coord[8] = {0};
    int offsetSrc = 0;
    for (int el = 0; el < total_elements; el++) {
        c_dest[el] = c_src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
        }
    }
}

void s_flatten_complex64(const float *src, const int *stridesSrc, float *dest, const int *shape, int rank) {
    if (src == NULL || dest == NULL || rank < 0 || rank > 8) return;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    if (rank == 0) {
        dest[0] = src[0];
        dest[1] = src[1];
        return;
    }
    const cpx_f_t *c_src = (const cpx_f_t*)src;
    cpx_f_t *c_dest = (cpx_f_t*)dest;
    int coord[8] = {0};
    int offsetSrc = 0;
    for (int el = 0; el < total_elements; el++) {
        c_dest[el] = c_src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
        }
    }
}

void s_flatten_uint8(const uint8_t *src, const int *stridesSrc, uint8_t *dest, const int *shape, int rank) {
    if (src == NULL || dest == NULL || rank < 0 || rank > 8) return;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    if (rank == 0) {
        dest[0] = src[0];
        return;
    }
    int coord[8] = {0};
    int offsetSrc = 0;
    for (int el = 0; el < total_elements; el++) {
        dest[el] = src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
        }
    }
}

void s_flatten_int16(const int16_t *src, const int *stridesSrc, int16_t *dest, const int *shape, int rank) {
    if (src == NULL || dest == NULL || rank < 0 || rank > 8) return;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    if (rank == 0) {
        dest[0] = src[0];
        return;
    }
    int coord[8] = {0};
    int offsetSrc = 0;
    for (int el = 0; el < total_elements; el++) {
        dest[el] = src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
        }
    }
}


// ============================================================================
// 8. NATIVE C HIGH-SPEED ELEMENTS HASHING KERNELS
// ============================================================================

uint32_t s_hash_double(const double *a, const int *strides, const int *shape, int rank, int is_contiguous) {
    if (a == NULL) return 0;
    uint32_t hash = 2166136261U;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    if (is_contiguous) {
        for (int i = 0; i < total_elements; i++) {
            hash_double(&hash, a[i]);
        }
        return hash;
    }
    if (rank <= 0 || rank > 8) {
        if (rank == 0) {
            hash_double(&hash, a[0]);
        }
        return hash;
    }
    int coord[8] = {0};
    int offset = 0;
    for (int el = 0; el < total_elements; el++) {
        hash_double(&hash, a[offset]);
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
    return hash;
}

uint32_t s_hash_float(const float *a, const int *strides, const int *shape, int rank, int is_contiguous) {
    if (a == NULL) return 0;
    uint32_t hash = 2166136261U;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    if (is_contiguous) {
        for (int i = 0; i < total_elements; i++) {
            hash_float(&hash, a[i]);
        }
        return hash;
    }
    if (rank <= 0 || rank > 8) {
        if (rank == 0) {
            hash_float(&hash, a[0]);
        }
        return hash;
    }
    int coord[8] = {0};
    int offset = 0;
    for (int el = 0; el < total_elements; el++) {
        hash_float(&hash, a[offset]);
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
    return hash;
}

uint32_t s_hash_int64(const int64_t *a, const int *strides, const int *shape, int rank, int is_contiguous) {
    if (a == NULL) return 0;
    uint32_t hash = 2166136261U;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    if (is_contiguous) {
        for (int i = 0; i < total_elements; i++) {
            hash_int64(&hash, a[i]);
        }
        return hash;
    }
    if (rank <= 0 || rank > 8) {
        if (rank == 0) {
            hash_int64(&hash, a[0]);
        }
        return hash;
    }
    int coord[8] = {0};
    int offset = 0;
    for (int el = 0; el < total_elements; el++) {
        hash_int64(&hash, a[offset]);
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
    return hash;
}

uint32_t s_hash_int32(const int32_t *a, const int *strides, const int *shape, int rank, int is_contiguous) {
    if (a == NULL) return 0;
    uint32_t hash = 2166136261U;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    if (is_contiguous) {
        for (int i = 0; i < total_elements; i++) {
            hash_int32(&hash, a[i]);
        }
        return hash;
    }
    if (rank <= 0 || rank > 8) {
        if (rank == 0) {
            hash_int32(&hash, a[0]);
        }
        return hash;
    }
    int coord[8] = {0};
    int offset = 0;
    for (int el = 0; el < total_elements; el++) {
        hash_int32(&hash, a[offset]);
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
    return hash;
}

uint32_t s_hash_complex128(const double *a, const int *strides, const int *shape, int rank, int is_contiguous) {
    if (a == NULL) return 0;
    uint32_t hash = 2166136261U;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    const cpx_t *c_a = (const cpx_t*)a;
    if (is_contiguous) {
        for (int i = 0; i < total_elements; i++) {
            hash_double(&hash, c_a[i].r);
            hash_double(&hash, c_a[i].i);
        }
        return hash;
    }
    if (rank <= 0 || rank > 8) {
        if (rank == 0) {
            hash_double(&hash, c_a[0].r);
            hash_double(&hash, c_a[0].i);
        }
        return hash;
    }
    int coord[8] = {0};
    int offset = 0;
    for (int el = 0; el < total_elements; el++) {
        hash_double(&hash, c_a[offset].r);
        hash_double(&hash, c_a[offset].i);
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
    return hash;
}

uint32_t s_hash_complex64(const float *a, const int *strides, const int *shape, int rank, int is_contiguous) {
    if (a == NULL) return 0;
    uint32_t hash = 2166136261U;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    const cpx_f_t *c_a = (const cpx_f_t*)a;
    if (is_contiguous) {
        for (int i = 0; i < total_elements; i++) {
            hash_float(&hash, c_a[i].r);
            hash_float(&hash, c_a[i].i);
        }
        return hash;
    }
    if (rank <= 0 || rank > 8) {
        if (rank == 0) {
            hash_float(&hash, c_a[0].r);
            hash_float(&hash, c_a[0].i);
        }
        return hash;
    }
    int coord[8] = {0};
    int offset = 0;
    for (int el = 0; el < total_elements; el++) {
        hash_float(&hash, c_a[offset].r);
        hash_float(&hash, c_a[offset].i);
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
    return hash;
}

uint32_t s_hash_boolean(const uint8_t *a, const int *strides, const int *shape, int rank, int is_contiguous) {
    if (a == NULL) return 0;
    uint32_t hash = 2166136261U;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    if (is_contiguous) {
        for (int i = 0; i < total_elements; i++) {
            hash_boolean(&hash, a[i]);
        }
        return hash;
    }
    if (rank <= 0 || rank > 8) {
        if (rank == 0) {
            hash_boolean(&hash, a[0]);
        }
        return hash;
    }
    int coord[8] = {0};
    int offset = 0;
    for (int el = 0; el < total_elements; el++) {
        hash_boolean(&hash, a[offset]);
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
    return hash;
}

// ============================================================================
// 9. NATIVE C HIGH-SPEED RANDOM DISTRIBUTION GENERATORS
// ============================================================================

void v_poisson_int64(int64_t *res, int size, double lam, unsigned long long seed) {
    if (res == NULL || size <= 0 || lam <= 0.0) return;

    uint64_t s[4];
    xoshiro256_seed(seed, s);

    if (lam < 30.0) {
        double limit = exp(-lam);
        for (int i = 0; i < size; i++) {
            int k = 0;
            double p = 1.0;
            do {
                k++;
                double u = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);
                p *= u;
            } while (p > limit);
            res[i] = k - 1;
        }
    } else {
        double stddev = sqrt(lam);
        int i = 0;
        while (i < size) {
            double u1;
            do {
                u1 = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);
            } while (u1 == 0.0);

            double u2 = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);

            double mag = sqrt(-2.0 * log(u1));
            double angle = 2.0 * M_PI * u2;

            double z0 = mag * cos(angle);
            double z1 = mag * sin(angle);

            double val0 = lam + stddev * z0;
            res[i] = val0 < 0 ? 0 : (int64_t)round(val0);

            if (i + 1 < size) {
                double val1 = lam + stddev * z1;
                res[i + 1] = val1 < 0 ? 0 : (int64_t)round(val1);
            }
            i += 2;
        }
    }
}

void v_poisson_int32(int32_t *res, int size, double lam, unsigned long long seed) {
    if (res == NULL || size <= 0 || lam <= 0.0) return;

    uint64_t s[4];
    xoshiro256_seed(seed, s);

    if (lam < 30.0) {
        double limit = exp(-lam);
        for (int i = 0; i < size; i++) {
            int k = 0;
            double p = 1.0;
            do {
                k++;
                double u = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);
                p *= u;
            } while (p > limit);
            res[i] = k - 1;
        }
    } else {
        double stddev = sqrt(lam);
        int i = 0;
        while (i < size) {
            double u1;
            do {
                u1 = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);
            } while (u1 == 0.0);

            double u2 = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);

            double mag = sqrt(-2.0 * log(u1));
            double angle = 2.0 * M_PI * u2;

            double z0 = mag * cos(angle);
            double z1 = mag * sin(angle);

            double val0 = lam + stddev * z0;
            res[i] = val0 < 0 ? 0 : (int32_t)round(val0);

            if (i + 1 < size) {
                double val1 = lam + stddev * z1;
                res[i + 1] = val1 < 0 ? 0 : (int32_t)round(val1);
            }
            i += 2;
        }
    }
}

void v_binomial_int64(int64_t *res, int size, int n, double p, unsigned long long seed) {
    if (res == NULL || size <= 0 || n < 0 || p < 0.0 || p > 1.0) return;

    uint64_t s[4];
    xoshiro256_seed(seed, s);

    if (n == 0) {
        for (int i = 0; i < size; i++) res[i] = 0;
        return;
    }

    if (n < 50) {
        for (int i = 0; i < size; i++) {
            int successes = 0;
            for (int t = 0; t < n; t++) {
                double u = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);
                if (u < p) {
                    successes++;
                }
            }
            res[i] = successes;
        }
    } else {
        double mean = n * p;
        double stddev = sqrt(n * p * (1.0 - p));

        if (stddev == 0.0) {
            for (int i = 0; i < size; i++) {
                res[i] = (int64_t)round(mean);
            }
        } else {
            int i = 0;
            while (i < size) {
                double u1;
                do {
                    u1 = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);
                } while (u1 == 0.0);

                double u2 = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);

                double mag = sqrt(-2.0 * log(u1));
                double angle = 2.0 * M_PI * u2;

                double z0 = mag * cos(angle);
                double z1 = mag * sin(angle);

                double val0 = round(mean + stddev * z0);
                if (val0 < 0) val0 = 0;
                if (val0 > n) val0 = n;
                res[i] = (int64_t)val0;

                if (i + 1 < size) {
                    double val1 = round(mean + stddev * z1);
                    if (val1 < 0) val1 = 0;
                    if (val1 > n) val1 = n;
                    res[i + 1] = (int64_t)val1;
                }
                i += 2;
            }
        }
    }
}

void v_binomial_int32(int32_t *res, int size, int n, double p, unsigned long long seed) {
    if (res == NULL || size <= 0 || n < 0 || p < 0.0 || p > 1.0) return;

    uint64_t s[4];
    xoshiro256_seed(seed, s);

    if (n == 0) {
        for (int i = 0; i < size; i++) res[i] = 0;
        return;
    }

    if (n < 50) {
        for (int i = 0; i < size; i++) {
            int successes = 0;
            for (int t = 0; t < n; t++) {
                double u = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);
                if (u < p) {
                    successes++;
                }
            }
            res[i] = successes;
        }
    } else {
        double mean = n * p;
        double stddev = sqrt(n * p * (1.0 - p));

        if (stddev == 0.0) {
            for (int i = 0; i < size; i++) {
                res[i] = (int32_t)round(mean);
            }
        } else {
            int i = 0;
            while (i < size) {
                double u1;
                do {
                    u1 = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);
                } while (u1 == 0.0);

                double u2 = (double)(xoshiro256_next(s) >> 11) * (1.0 / 9007199254740992.0);

                double mag = sqrt(-2.0 * log(u1));
                double angle = 2.0 * M_PI * u2;

                double z0 = mag * cos(angle);
                double z1 = mag * sin(angle);

                double val0 = round(mean + stddev * z0);
                if (val0 < 0) val0 = 0;
                if (val0 > n) val0 = n;
                res[i] = (int32_t)val0;

                if (i + 1 < size) {
                    double val1 = round(mean + stddev * z1);
                    if (val1 < 0) val1 = 0;
                    if (val1 > n) val1 = n;
                    res[i + 1] = (int32_t)val1;
                }
                i += 2;
            }
        }
    }
}

#ifdef _WIN32
#include <windows.h>

typedef LONG (WINAPI *BCryptGenRandomFunc)(
    void* hAlgorithm,
    unsigned char* pbBuffer,
    unsigned long cbBuffer,
    unsigned long dwFlags
);

static void fill_secure_bytes_win(void *dest, size_t size) {
    HMODULE hBcrypt = LoadLibraryA("bcrypt.dll");
    if (hBcrypt != NULL) {
        BCryptGenRandomFunc pBCryptGenRandom = (BCryptGenRandomFunc)GetProcAddress(hBcrypt, "BCryptGenRandom");
        if (pBCryptGenRandom != NULL) {
            pBCryptGenRandom(NULL, (unsigned char*)dest, (unsigned long)size, 0x00000002);
        }
        FreeLibrary(hBcrypt);
    }
}
#else
#include <fcntl.h>
#include <unistd.h>
#endif

static void fill_secure_bytes(void *dest, size_t size) {
#ifdef _WIN32
    fill_secure_bytes_win(dest, size);
#else
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd >= 0) {
        size_t bytes_read = 0;
        while (bytes_read < size) {
            ssize_t res = read(fd, (char *)dest + bytes_read, size - bytes_read);
            if (res < 0) break;
            bytes_read += res;
        }
        close(fd);
    }
#endif
}

void v_secure_uniform_double(double *res, int size) {
    if (res == NULL || size <= 0) return;
    fill_secure_bytes(res, size * sizeof(double));
    unsigned long long *temp = (unsigned long long *)res;
    for (int i = 0; i < size; i++) {
        res[i] = (double)(temp[i] >> 11) * (1.0 / 9007199254740992.0);
    }
}

void v_secure_uniform_float(float *res, int size) {
    if (res == NULL || size <= 0) return;
    fill_secure_bytes(res, size * sizeof(float));
    unsigned int *temp = (unsigned int *)res;
    for (int i = 0; i < size; i++) {
        res[i] = (float)((double)(temp[i] >> 5) * (1.0 / 134217728.0));
    }
}

#define IMPLEMENT_V_SECURE_RANDINT(TYPE_NAME, TYPE, BOUNDS_TYPE, RANGE_TYPE) \
void v_secure_randint_##TYPE_NAME(TYPE *res, int size, BOUNDS_TYPE low, BOUNDS_TYPE high) { \
    if (res == NULL || size <= 0 || low >= high) return; \
    fill_secure_bytes(res, size * sizeof(TYPE)); \
    RANGE_TYPE range = (RANGE_TYPE)high - (RANGE_TYPE)low; \
    for (int i = 0; i < size; i++) { \
        res[i] = (TYPE)(low + (RANGE_TYPE)((unsigned long long)res[i] % (unsigned long long)range)); \
    } \
}

IMPLEMENT_V_SECURE_RANDINT(int64, int64_t, int64_t, int64_t)
IMPLEMENT_V_SECURE_RANDINT(int32, int32_t, int32_t, int32_t)
IMPLEMENT_V_SECURE_RANDINT(int16, int16_t, int, int)
IMPLEMENT_V_SECURE_RANDINT(uint8, uint8_t, int, int)

void v_secure_normal_double(double *res, int size, double loc, double scale) {
    if (res == NULL || size <= 0 || scale <= 0.0) return;
    v_secure_uniform_double(res, size);
    int i = 0;
    while (i < size) {
        double u1 = res[i];
        if (u1 == 0.0) u1 = 1e-15;
        double u2 = (i + 1 < size) ? res[i + 1] : 0.5;
        double mag = scale * sqrt(-2.0 * log(u1));
        double angle = 2.0 * M_PI * u2;
        res[i] = loc + mag * cos(angle);
        if (i + 1 < size) {
            res[i + 1] = loc + mag * sin(angle);
        }
        i += 2;
    }
}

void v_secure_normal_float(float *res, int size, float loc, float scale) {
    if (res == NULL || size <= 0 || scale <= 0.0f) return;
    v_secure_uniform_float(res, size);
    int i = 0;
    while (i < size) {
        float u1 = res[i];
        if (u1 == 0.0f) u1 = 1e-15f;
        float u2 = (i + 1 < size) ? res[i + 1] : 0.5f;
        float mag = scale * sqrtf(-2.0f * logf(u1));
        float angle = 2.0f * (float)M_PI * u2;
        res[i] = loc + mag * cosf(angle);
        if (i + 1 < size) {
            res[i + 1] = loc + mag * sinf(angle);
        }
        i += 2;
    }
}

void v_tril_double(const double *src, double *res, int batch_count, int rows, int cols, int k) {
    if (src == NULL || res == NULL || batch_count <= 0 || rows <= 0 || cols <= 0) return;
    int matrix_size = rows * cols;
    for (int b = 0; b < batch_count; b++) {
        const double *s_mat = src + b * matrix_size;
        double *r_mat = res + b * matrix_size;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int idx = r * cols + c;
                r_mat[idx] = (c <= r + k) ? s_mat[idx] : 0.0;
            }
        }
    }
}

void v_tril_float(const float *src, float *res, int batch_count, int rows, int cols, int k) {
    if (src == NULL || res == NULL || batch_count <= 0 || rows <= 0 || cols <= 0) return;
    int matrix_size = rows * cols;
    for (int b = 0; b < batch_count; b++) {
        const float *s_mat = src + b * matrix_size;
        float *r_mat = res + b * matrix_size;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int idx = r * cols + c;
                r_mat[idx] = (c <= r + k) ? s_mat[idx] : 0.0f;
            }
        }
    }
}

void v_triu_double(const double *src, double *res, int batch_count, int rows, int cols, int k) {
    if (src == NULL || res == NULL || batch_count <= 0 || rows <= 0 || cols <= 0) return;
    int matrix_size = rows * cols;
    for (int b = 0; b < batch_count; b++) {
        const double *s_mat = src + b * matrix_size;
        double *r_mat = res + b * matrix_size;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int idx = r * cols + c;
                r_mat[idx] = (c >= r + k) ? s_mat[idx] : 0.0;
            }
        }
    }
}

void v_triu_float(const float *src, float *res, int batch_count, int rows, int cols, int k) {
    if (src == NULL || res == NULL || batch_count <= 0 || rows <= 0 || cols <= 0) return;
    int matrix_size = rows * cols;
    for (int b = 0; b < batch_count; b++) {
        const float *s_mat = src + b * matrix_size;
        float *r_mat = res + b * matrix_size;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int idx = r * cols + c;
                r_mat[idx] = (c >= r + k) ? s_mat[idx] : 0.0f;
            }
        }
    }
}

#define DEFINE_STRIDED_CUM_OP(FUNCNAME, T, OP) \
void FUNCNAME(const T *src, const int *stridesSrc, \
              T *res, const int *stridesRes, \
              const int *shape, int rank, int axis) { \
    if (src == NULL || res == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return; \
    int coord[8] = {0}; \
    int outer_size = 1; \
    for (int d = 0; d < rank; d++) { \
        if (d != axis) outer_size *= shape[d]; \
    } \
    for (int o = 0; o < outer_size; o++) { \
        int offsetSrc = 0; \
        int offsetRes = 0; \
        for (int d = 0; d < rank; d++) { \
            if (d != axis) { \
                offsetSrc += coord[d] * stridesSrc[d]; \
                offsetRes += coord[d] * stridesRes[d]; \
            } \
        } \
        T acc; \
        for (int i = 0; i < shape[axis]; i++) { \
            int idxSrc = offsetSrc + i * stridesSrc[axis]; \
            int idxRes = offsetRes + i * stridesRes[axis]; \
            acc = (i == 0) ? src[idxSrc] : OP(acc, src[idxSrc]); \
            res[idxRes] = acc; \
        } \
        for (int d = rank - 1; d >= 0; d--) { \
            if (d == axis) continue; \
            coord[d]++; \
            if (coord[d] < shape[d]) break; \
            coord[d] = 0; \
        } \
    } \
}

// standard real ops
#define OP_ADD(x, y) ((x) + (y))
#define OP_MUL(x, y) ((x) * (y))
#define OP_MIN(x, y) (((x) < (y)) ? (x) : (y))
#define OP_MAX(x, y) (((x) > (y)) ? (x) : (y))

DEFINE_STRIDED_CUM_OP(s_cumsum_double, double, OP_ADD)
DEFINE_STRIDED_CUM_OP(s_cumsum_float, float, OP_ADD)
DEFINE_STRIDED_CUM_OP(s_cumsum_int64, int64_t, OP_ADD)
DEFINE_STRIDED_CUM_OP(s_cumsum_int32, int32_t, OP_ADD)

DEFINE_STRIDED_CUM_OP(s_cumprod_double, double, OP_MUL)
DEFINE_STRIDED_CUM_OP(s_cumprod_float, float, OP_MUL)
DEFINE_STRIDED_CUM_OP(s_cumprod_int64, int64_t, OP_MUL)
DEFINE_STRIDED_CUM_OP(s_cumprod_int32, int32_t, OP_MUL)

DEFINE_STRIDED_CUM_OP(s_cummin_double, double, OP_MIN)
DEFINE_STRIDED_CUM_OP(s_cummin_float, float, OP_MIN)
DEFINE_STRIDED_CUM_OP(s_cummin_int64, int64_t, OP_MIN)
DEFINE_STRIDED_CUM_OP(s_cummin_int32, int32_t, OP_MIN)

DEFINE_STRIDED_CUM_OP(s_cummax_double, double, OP_MAX)
DEFINE_STRIDED_CUM_OP(s_cummax_float, float, OP_MAX)
DEFINE_STRIDED_CUM_OP(s_cummax_int64, int64_t, OP_MAX)
DEFINE_STRIDED_CUM_OP(s_cummax_int32, int32_t, OP_MAX)

// custom complex ops
static inline cpx_t cpx_add(cpx_t a, cpx_t b) {
    return (cpx_t){a.r + b.r, a.i + b.i};
}
static inline cpx_t cpx_mul(cpx_t a, cpx_t b) {
    return (cpx_t){a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}
static inline cpx_f_t cpx_add_f(cpx_f_t a, cpx_f_t b) {
    return (cpx_f_t){a.r + b.r, a.i + b.i};
}
static inline cpx_f_t cpx_mul_f(cpx_f_t a, cpx_f_t b) {
    return (cpx_f_t){a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r};
}

DEFINE_STRIDED_CUM_OP(s_cumsum_complex128, cpx_t, cpx_add)
DEFINE_STRIDED_CUM_OP(s_cumsum_complex64, cpx_f_t, cpx_add_f)
DEFINE_STRIDED_CUM_OP(s_cumprod_complex128, cpx_t, cpx_mul)
DEFINE_STRIDED_CUM_OP(s_cumprod_complex64, cpx_f_t, cpx_mul_f)

// complex subtraction helpers
static inline cpx_t cpx_sub(cpx_t a, cpx_t b) {
    return (cpx_t){a.r - b.r, a.i - b.i};
}
static inline cpx_f_t cpx_sub_f(cpx_f_t a, cpx_f_t b) {
    return (cpx_f_t){a.r - b.r, a.i - b.i};
}

#define DEFINE_STRIDED_DIFF_OP(FUNCNAME, T, SUB_OP) \
void FUNCNAME(const T *src, const int *stridesSrc, \
              T *res, const int *stridesRes, \
              const int *shape, int rank, int axis) { \
    if (src == NULL || res == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return; \
    int coord[8] = {0}; \
    int outer_size = 1; \
    for (int d = 0; d < rank; d++) { \
        if (d != axis) outer_size *= shape[d]; \
    } \
    for (int o = 0; o < outer_size; o++) { \
        int offsetSrc = 0; \
        int offsetRes = 0; \
        for (int d = 0; d < rank; d++) { \
            if (d != axis) { \
                offsetSrc += coord[d] * stridesSrc[d]; \
                offsetRes += coord[d] * stridesRes[d]; \
            } \
        } \
        for (int i = 0; i < shape[axis] - 1; i++) { \
            int idxSrcCurr = offsetSrc + i * stridesSrc[axis]; \
            int idxSrcNext = offsetSrc + (i + 1) * stridesSrc[axis]; \
            int idxRes = offsetRes + i * stridesRes[axis]; \
            res[idxRes] = SUB_OP(src[idxSrcNext], src[idxSrcCurr]); \
        } \
        for (int d = rank - 1; d >= 0; d--) { \
            if (d == axis) continue; \
            coord[d]++; \
            if (coord[d] < shape[d]) break; \
            coord[d] = 0; \
        } \
    } \
}

#define SUB_REAL(x, y) ((x) - (y))

DEFINE_STRIDED_DIFF_OP(s_diff_double, double, SUB_REAL)
DEFINE_STRIDED_DIFF_OP(s_diff_float, float, SUB_REAL)
DEFINE_STRIDED_DIFF_OP(s_diff_int64, int64_t, SUB_REAL)
DEFINE_STRIDED_DIFF_OP(s_diff_int32, int32_t, SUB_REAL)
DEFINE_STRIDED_DIFF_OP(s_diff_complex128, cpx_t, cpx_sub)
DEFINE_STRIDED_DIFF_OP(s_diff_complex64, cpx_f_t, cpx_sub_f)

#define DEFINE_STRIDED_UNARY_OP(FUNCNAME, T, OP) \
void FUNCNAME(const T *src, const int *stridesSrc, \
              T *res, const int *stridesRes, \
              const int *shape, int rank) { \
    if (src == NULL || res == NULL || shape == NULL || rank <= 0) return; \
    int coord[8] = {0}; \
    int total_size = 1; \
    for (int d = 0; d < rank; d++) total_size *= shape[d]; \
    for (int i = 0; i < total_size; i++) { \
        int offsetSrc = 0; \
        int offsetRes = 0; \
        for (int d = 0; d < rank; d++) { \
            offsetSrc += coord[d] * stridesSrc[d]; \
            offsetRes += coord[d] * stridesRes[d]; \
        } \
        res[offsetRes] = OP(src[offsetSrc]); \
        for (int d = rank - 1; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) break; \
            coord[d] = 0; \
        } \
    } \
}

#define OP_SIN_D(x) sin(x)
#define OP_SIN_F(x) sinf(x)
#define OP_COS_D(x) cos(x)
#define OP_COS_F(x) cosf(x)

DEFINE_STRIDED_UNARY_OP(s_sin_double, double, OP_SIN_D)
DEFINE_STRIDED_UNARY_OP(s_sin_float, float, OP_SIN_F)
DEFINE_STRIDED_UNARY_OP(s_cos_double, double, OP_COS_D)
DEFINE_STRIDED_UNARY_OP(s_cos_float, float, OP_COS_F)

// complex trig helper definitions
static inline cpx_t cpx_sin(cpx_t z) {
    return (cpx_t){sin(z.r) * cosh(z.i), cos(z.r) * sinh(z.i)};
}
static inline cpx_f_t cpx_sin_f(cpx_f_t z) {
    return (cpx_f_t){sinf(z.r) * coshf(z.i), cosf(z.r) * sinhf(z.i)};
}

static inline cpx_t cpx_cos(cpx_t z) {
    return (cpx_t){cos(z.r) * cosh(z.i), -sin(z.r) * sinh(z.i)};
}
static inline cpx_f_t cpx_cos_f(cpx_f_t z) {
    return (cpx_f_t){cosf(z.r) * coshf(z.i), -sinf(z.r) * sinhf(z.i)};
}

static inline cpx_t cpx_tan(cpx_t z) {
    double denom = cos(2.0 * z.r) + cosh(2.0 * z.i);
    if (denom == 0.0) return (cpx_t){0.0, 0.0};
    return (cpx_t){sin(2.0 * z.r) / denom, sinh(2.0 * z.i) / denom};
}
static inline cpx_f_t cpx_tan_f(cpx_f_t z) {
    float denom = cosf(2.0f * z.r) + coshf(2.0f * z.i);
    if (denom == 0.0f) return (cpx_f_t){0.0f, 0.0f};
    return (cpx_f_t){sinf(2.0f * z.r) / denom, sinhf(2.0f * z.i) / denom};
}

static inline cpx_t cpx_asin(cpx_t z) {
    double A = 0.5 * sqrt((z.r + 1.0)*(z.r + 1.0) + z.i*z.i);
    double B = 0.5 * sqrt((z.r - 1.0)*(z.r - 1.0) + z.i*z.i);
    double u = A + B;
    double v = A - B;
    if (v < -1.0) v = -1.0;
    if (v > 1.0) v = 1.0;
    double r = asin(v);
    double s = (z.i >= 0 ? 1.0 : -1.0) * log(u + sqrt(u*u - 1.0));
    return (cpx_t){r, s};
}

static inline cpx_f_t cpx_asin_f(cpx_f_t z) {
    float A = 0.5f * sqrtf((z.r + 1.0f)*(z.r + 1.0f) + z.i*z.i);
    float B = 0.5f * sqrtf((z.r - 1.0f)*(z.r - 1.0f) + z.i*z.i);
    float u = A + B;
    float v = A - B;
    if (v < -1.0f) v = -1.0f;
    if (v > 1.0f) v = 1.0f;
    float r = asinf(v);
    float s = (z.i >= 0.0f ? 1.0f : -1.0f) * logf(u + sqrtf(u*u - 1.0f));
    return (cpx_f_t){r, s};
}

static inline cpx_t cpx_acos(cpx_t z) {
    cpx_t s = cpx_asin(z);
    return (cpx_t){3.14159265358979323846 / 2.0 - s.r, -s.i};
}

static inline cpx_f_t cpx_acos_f(cpx_f_t z) {
    cpx_f_t s = cpx_asin_f(z);
    return (cpx_f_t){3.14159265358979323846f / 2.0f - s.r, -s.i};
}

static inline cpx_t cpx_atan(cpx_t z) {
    double r = 0.5 * atan2(2.0 * z.r, 1.0 - z.r*z.r - z.i*z.i);
    double s = 0.25 * log((z.r*z.r + (z.i + 1.0)*(z.i + 1.0)) / (z.r*z.r + (z.i - 1.0)*(z.i - 1.0)));
    return (cpx_t){r, s};
}

static inline cpx_f_t cpx_atan_f(cpx_f_t z) {
    float r = 0.5f * atan2f(2.0f * z.r, 1.0f - z.r*z.r - z.i*z.i);
    float s = 0.25f * logf((z.r*z.r + (z.i + 1.0f)*(z.i + 1.0f)) / (z.r*z.r + (z.i - 1.0f)*(z.i - 1.0f)));
    return (cpx_f_t){r, s};
}

#define DEFINE_COMPLEX_UNARY_VEC(FUNCNAME, T, OP) \
void FUNCNAME(const T *src, T *res, int size) { \
    if (src == NULL || res == NULL || size <= 0) return; \
    for (int i = 0; i < size; i++) { \
        res[i] = OP(src[i]); \
    } \
}

DEFINE_COMPLEX_UNARY_VEC(v_sin_complex128, cpx_t, cpx_sin)
DEFINE_COMPLEX_UNARY_VEC(v_sin_complex64, cpx_f_t, cpx_sin_f)
DEFINE_COMPLEX_UNARY_VEC(v_cos_complex128, cpx_t, cpx_cos)
DEFINE_COMPLEX_UNARY_VEC(v_cos_complex64, cpx_f_t, cpx_cos_f)
DEFINE_COMPLEX_UNARY_VEC(v_tan_complex128, cpx_t, cpx_tan)
DEFINE_COMPLEX_UNARY_VEC(v_tan_complex64, cpx_f_t, cpx_tan_f)

DEFINE_COMPLEX_UNARY_VEC(v_asin_complex128, cpx_t, cpx_asin)
DEFINE_COMPLEX_UNARY_VEC(v_asin_complex64, cpx_f_t, cpx_asin_f)
DEFINE_COMPLEX_UNARY_VEC(v_acos_complex128, cpx_t, cpx_acos)
DEFINE_COMPLEX_UNARY_VEC(v_acos_complex64, cpx_f_t, cpx_acos_f)
DEFINE_COMPLEX_UNARY_VEC(v_atan_complex128, cpx_t, cpx_atan)
DEFINE_COMPLEX_UNARY_VEC(v_atan_complex64, cpx_f_t, cpx_atan_f)

DEFINE_STRIDED_UNARY_OP(s_sin_complex128, cpx_t, cpx_sin)
DEFINE_STRIDED_UNARY_OP(s_sin_complex64, cpx_f_t, cpx_sin_f)
DEFINE_STRIDED_UNARY_OP(s_cos_complex128, cpx_t, cpx_cos)
DEFINE_STRIDED_UNARY_OP(s_cos_complex64, cpx_f_t, cpx_cos_f)
DEFINE_STRIDED_UNARY_OP(s_tan_complex128, cpx_t, cpx_tan)
DEFINE_STRIDED_UNARY_OP(s_tan_complex64, cpx_f_t, cpx_tan_f)

DEFINE_STRIDED_UNARY_OP(s_asin_complex128, cpx_t, cpx_asin)
DEFINE_STRIDED_UNARY_OP(s_asin_complex64, cpx_f_t, cpx_asin_f)
DEFINE_STRIDED_UNARY_OP(s_acos_complex128, cpx_t, cpx_acos)
DEFINE_STRIDED_UNARY_OP(s_acos_complex64, cpx_f_t, cpx_acos_f)
DEFINE_STRIDED_UNARY_OP(s_atan_complex128, cpx_t, cpx_atan)
DEFINE_STRIDED_UNARY_OP(s_atan_complex64, cpx_f_t, cpx_atan_f)

#define OP_ASIN_D(x) asin(x)
#define OP_ASIN_F(x) asinf(x)
#define OP_ACOS_D(x) acos(x)
#define OP_ACOS_F(x) acosf(x)
#define OP_ATAN_D(x) atan(x)
#define OP_ATAN_F(x) atanf(x)

DEFINE_STRIDED_UNARY_OP(s_asin_double, double, OP_ASIN_D)
DEFINE_STRIDED_UNARY_OP(s_asin_float, float, OP_ASIN_F)
DEFINE_STRIDED_UNARY_OP(s_acos_double, double, OP_ACOS_D)
DEFINE_STRIDED_UNARY_OP(s_acos_float, float, OP_ACOS_F)
DEFINE_STRIDED_UNARY_OP(s_atan_double, double, OP_ATAN_D)
DEFINE_STRIDED_UNARY_OP(s_atan_float, float, OP_ATAN_F)

void s_atan2_double(const double *y, const int *stridesY,
                   const double *x, const int *stridesX,
                   double *res, const int *stridesRes,
                   const int *shape, int rank) {
    if (y == NULL || x == NULL || res == NULL || rank <= 0 || rank > 8) return;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    int coord[8] = {0};
    int offsetY = 0, offsetX = 0, offsetRes = 0;
    for (int el = 0; el < total_elements; el++) {
        res[offsetRes] = atan2(y[offsetY], x[offsetX]);

        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetY += stridesY[d];
                offsetX += stridesX[d];
                offsetRes += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetY -= (shape[d] - 1) * stridesY[d];
            offsetX -= (shape[d] - 1) * stridesX[d];
            offsetRes -= (shape[d] - 1) * stridesRes[d];
        }
    }
}

void s_atan2_float(const float *y, const int *stridesY,
                  const float *x, const int *stridesX,
                  float *res, const int *stridesRes,
                  const int *shape, int rank) {
    if (y == NULL || x == NULL || res == NULL || rank <= 0 || rank > 8) return;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    int coord[8] = {0};
    int offsetY = 0, offsetX = 0, offsetRes = 0;
    for (int el = 0; el < total_elements; el++) {
        res[offsetRes] = atan2f(y[offsetY], x[offsetX]);

        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetY += stridesY[d];
                offsetX += stridesX[d];
                offsetRes += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetY -= (shape[d] - 1) * stridesY[d];
            offsetX -= (shape[d] - 1) * stridesX[d];
            offsetRes -= (shape[d] - 1) * stridesRes[d];
        }
    }
}

#define OP_TAN_D(x) tan(x)
#define OP_TAN_F(x) tanf(x)
#define OP_EXP_D(x) exp(x)
#define OP_EXP_F(x) expf(x)
#define OP_LOG_D(x) log(x)
#define OP_LOG_F(x) logf(x)

DEFINE_STRIDED_UNARY_OP(s_tan_double, double, OP_TAN_D)
DEFINE_STRIDED_UNARY_OP(s_tan_float, float, OP_TAN_F)
DEFINE_STRIDED_UNARY_OP(s_exp_double, double, OP_EXP_D)
DEFINE_STRIDED_UNARY_OP(s_exp_float, float, OP_EXP_F)
DEFINE_STRIDED_UNARY_OP(s_log_double, double, OP_LOG_D)
DEFINE_STRIDED_UNARY_OP(s_log_float, float, OP_LOG_F)

#define OP_SINH_D(x) sinh(x)
#define OP_SINH_F(x) sinhf(x)
#define OP_COSH_D(x) cosh(x)
#define OP_COSH_F(x) coshf(x)
#define OP_TANH_D(x) tanh(x)
#define OP_TANH_F(x) tanhf(x)

DEFINE_STRIDED_UNARY_OP(s_sinh_double, double, OP_SINH_D)
DEFINE_STRIDED_UNARY_OP(s_sinh_float, float, OP_SINH_F)
DEFINE_STRIDED_UNARY_OP(s_cosh_double, double, OP_COSH_D)
DEFINE_STRIDED_UNARY_OP(s_cosh_float, float, OP_COSH_F)
DEFINE_STRIDED_UNARY_OP(s_tanh_double, double, OP_TANH_D)
DEFINE_STRIDED_UNARY_OP(s_tanh_float, float, OP_TANH_F)

#define OP_ASINH_D(x) asinh(x)
#define OP_ASINH_F(x) asinhf(x)
#define OP_ACOSH_D(x) acosh(x)
#define OP_ACOSH_F(x) acoshf(x)
#define OP_ATANH_D(x) atanh(x)
#define OP_ATANH_F(x) atanhf(x)

DEFINE_STRIDED_UNARY_OP(s_asinh_double, double, OP_ASINH_D)
DEFINE_STRIDED_UNARY_OP(s_asinh_float, float, OP_ASINH_F)
DEFINE_STRIDED_UNARY_OP(s_acosh_double, double, OP_ACOSH_D)
DEFINE_STRIDED_UNARY_OP(s_acosh_float, float, OP_ACOSH_F)
DEFINE_STRIDED_UNARY_OP(s_atanh_double, double, OP_ATANH_D)
DEFINE_STRIDED_UNARY_OP(s_atanh_float, float, OP_ATANH_F)

static inline cpx_t cpx_atanh(cpx_t z) {
    double r = 0.25 * log(((1.0 + z.r)*(1.0 + z.r) + z.i*z.i) / ((1.0 - z.r)*(1.0 - z.r) + z.i*z.i));
    double s = 0.5 * atan2(2.0 * z.i, 1.0 - z.r*z.r - z.i*z.i);
    return (cpx_t){r, s};
}

static inline cpx_f_t cpx_atanh_f(cpx_f_t z) {
    float r = 0.25f * logf(((1.0f + z.r)*(1.0f + z.r) + z.i*z.i) / ((1.0f - z.r)*(1.0f - z.r) + z.i*z.i));
    float s = 0.5f * atan2f(2.0f * z.i, 1.0f - z.r*z.r - z.i*z.i);
    return (cpx_f_t){r, s};
}

DEFINE_COMPLEX_UNARY_VEC(v_atanh_complex128, cpx_t, cpx_atanh)
DEFINE_COMPLEX_UNARY_VEC(v_atanh_complex64, cpx_f_t, cpx_atanh_f)
DEFINE_STRIDED_UNARY_OP(s_atanh_complex128, cpx_t, cpx_atanh)
DEFINE_STRIDED_UNARY_OP(s_atanh_complex64, cpx_f_t, cpx_atanh_f)

void v_hypot_complex128(const cpx_t *x1, const cpx_t *x2, double *res, int size) {
    if (x1 == NULL || x2 == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = sqrt(x1[i].r*x1[i].r + x1[i].i*x1[i].i + x2[i].r*x2[i].r + x2[i].i*x2[i].i);
    }
}

void v_hypot_complex64(const cpx_f_t *x1, const cpx_f_t *x2, float *res, int size) {
    if (x1 == NULL || x2 == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = sqrtf(x1[i].r*x1[i].r + x1[i].i*x1[i].i + x2[i].r*x2[i].r + x2[i].i*x2[i].i);
    }
}

void s_hypot_complex128(const cpx_t *x1, const int *stridesX1, const cpx_t *x2, const int *stridesX2, double *res, const int *stridesRes, const int *shape, int rank) {
    if (x1 == NULL || x2 == NULL || res == NULL || shape == NULL || rank <= 0) return;
    int coord[8] = {0};
    int total_size = 1;
    for (int d = 0; d < rank; d++) total_size *= shape[d];
    for (int i = 0; i < total_size; i++) {
        int offsetX1 = 0, offsetX2 = 0, offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetX1 += coord[d] * stridesX1[d];
            offsetX2 += coord[d] * stridesX2[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        res[offsetRes] = sqrt(x1[offsetX1].r*x1[offsetX1].r + x1[offsetX1].i*x1[offsetX1].i + x2[offsetX2].r*x2[offsetX2].r + x2[offsetX2].i*x2[offsetX2].i);
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_hypot_complex64(const cpx_f_t *x1, const int *stridesX1, const cpx_f_t *x2, const int *stridesX2, float *res, const int *stridesRes, const int *shape, int rank) {
    if (x1 == NULL || x2 == NULL || res == NULL || shape == NULL || rank <= 0) return;
    int coord[8] = {0};
    int total_size = 1;
    for (int d = 0; d < rank; d++) total_size *= shape[d];
    for (int i = 0; i < total_size; i++) {
        int offsetX1 = 0, offsetX2 = 0, offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetX1 += coord[d] * stridesX1[d];
            offsetX2 += coord[d] * stridesX2[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        res[offsetRes] = sqrtf(x1[offsetX1].r*x1[offsetX1].r + x1[offsetX1].i*x1[offsetX1].i + x2[offsetX2].r*x2[offsetX2].r + x2[offsetX2].i*x2[offsetX2].i);
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

static inline cpx_t cpx_pow(cpx_t z1, cpx_t z2) {
    double mag = sqrt(z1.r*z1.r + z1.i*z1.i);
    if (mag == 0.0) return (cpx_t){0.0, 0.0};
    double L = log(mag);
    double theta = atan2(z1.i, z1.r);
    double R = z2.r * L - z2.i * theta;
    double imag_val = z2.i * L + z2.r * theta;
    double eR = exp(R);
    return (cpx_t){eR * cos(imag_val), eR * sin(imag_val)};
}

static inline cpx_f_t cpx_pow_f(cpx_f_t z1, cpx_f_t z2) {
    float mag = sqrtf(z1.r*z1.r + z1.i*z1.i);
    if (mag == 0.0f) return (cpx_f_t){0.0f, 0.0f};
    float L = logf(mag);
    float theta = atan2f(z1.i, z1.r);
    float R = z2.r * L - z2.i * theta;
    float imag_val = z2.i * L + z2.r * theta;
    float eR = expf(R);
    return (cpx_f_t){eR * cosf(imag_val), eR * sinf(imag_val)};
}

void v_pow_complex128(const cpx_t *x1, const cpx_t *x2, cpx_t *res, int size) {
    if (x1 == NULL || x2 == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = cpx_pow(x1[i], x2[i]);
    }
}

void v_pow_complex64(const cpx_f_t *x1, const cpx_f_t *x2, cpx_f_t *res, int size) {
    if (x1 == NULL || x2 == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = cpx_pow_f(x1[i], x2[i]);
    }
}

void s_pow_complex128(const cpx_t *x1, const int *stridesX1, const cpx_t *x2, const int *stridesX2, cpx_t *res, const int *stridesRes, const int *shape, int rank) {
    if (x1 == NULL || x2 == NULL || res == NULL || shape == NULL || rank <= 0) return;
    int coord[8] = {0};
    int total_size = 1;
    for (int d = 0; d < rank; d++) total_size *= shape[d];
    for (int i = 0; i < total_size; i++) {
        int offsetX1 = 0, offsetX2 = 0, offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetX1 += coord[d] * stridesX1[d];
            offsetX2 += coord[d] * stridesX2[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        res[offsetRes] = cpx_pow(x1[offsetX1], x2[offsetX2]);
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_pow_complex64(const cpx_f_t *x1, const int *stridesX1, const cpx_f_t *x2, const int *stridesX2, cpx_f_t *res, const int *stridesRes, const int *shape, int rank) {
    if (x1 == NULL || x2 == NULL || res == NULL || shape == NULL || rank <= 0) return;
    int coord[8] = {0};
    int total_size = 1;
    for (int d = 0; d < rank; d++) total_size *= shape[d];
    for (int i = 0; i < total_size; i++) {
        int offsetX1 = 0, offsetX2 = 0, offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetX1 += coord[d] * stridesX1[d];
            offsetX2 += coord[d] * stridesX2[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        res[offsetRes] = cpx_pow_f(x1[offsetX1], x2[offsetX2]);
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void v_conj_complex128(const cpx_t *src, cpx_t *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i].r = src[i].r;
        res[i].i = -src[i].i;
    }
}

void v_conj_complex64(const cpx_f_t *src, cpx_f_t *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i].r = src[i].r;
        res[i].i = -src[i].i;
    }
}

void s_conj_complex128(const cpx_t *src, const int *stridesSrc, cpx_t *res, const int *stridesRes, const int *shape, int rank) {
    if (src == NULL || res == NULL || shape == NULL || rank <= 0) return;
    int coord[8] = {0};
    int total_size = 1;
    for (int d = 0; d < rank; d++) total_size *= shape[d];
    for (int i = 0; i < total_size; i++) {
        int offsetSrc = 0, offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetSrc += coord[d] * stridesSrc[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        res[offsetRes].r = src[offsetSrc].r;
        res[offsetRes].i = -src[offsetSrc].i;
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_conj_complex64(const cpx_f_t *src, const int *stridesSrc, cpx_f_t *res, const int *stridesRes, const int *shape, int rank) {
    if (src == NULL || res == NULL || shape == NULL || rank <= 0) return;
    int coord[8] = {0};
    int total_size = 1;
    for (int d = 0; d < rank; d++) total_size *= shape[d];
    for (int i = 0; i < total_size; i++) {
        int offsetSrc = 0, offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetSrc += coord[d] * stridesSrc[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        res[offsetRes].r = src[offsetSrc].r;
        res[offsetRes].i = -src[offsetSrc].i;
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

/* ============================================================================
 * CROSS-TYPE BINARY MATH FFI HELPERS IMPLEMENTATION
 * ============================================================================
 */

static inline cpx_t cpx_from_double(double v) { return (cpx_t){v, 0.0}; }
static inline cpx_t cpx_from_float(float v) { return (cpx_t){(double)v, 0.0}; }
static inline cpx_t cpx_from_int64(int64_t v) { return (cpx_t){(double)v, 0.0}; }
static inline cpx_t cpx_from_int32(int32_t v) { return (cpx_t){(double)v, 0.0}; }
static inline cpx_t cpx_from_uint8(uint8_t v) { return (cpx_t){(double)v, 0.0}; }
static inline cpx_t cpx_from_int16(int16_t v) { return (cpx_t){(double)v, 0.0}; }
static inline cpx_t cpx_from_cpx(cpx_t v) { return v; }

static inline cpx_t cpx_from_cpx64(cpx_f_t v) { return (cpx_t){(double)v.r, (double)v.i}; }

static inline cpx_f_t cpx_f_from_cpx(cpx_t v) { return (cpx_f_t){(float)v.r, (float)v.i}; }

static inline cpx_t cpx_div(cpx_t x, cpx_t y) {
    double denom = y.r * y.r + y.i * y.i;
    if (denom == 0.0) return (cpx_t){NAN, NAN};
    return (cpx_t){(x.r * y.r + x.i * y.i) / denom, (x.i * y.r - x.r * y.i) / denom};
}

static inline cpx_f_t cpx_div_f(cpx_f_t x, cpx_f_t y) {
    float denom = y.r * y.r + y.i * y.i;
    if (denom == 0.0f) return (cpx_f_t){NAN, NAN};
    return (cpx_f_t){(x.r * y.r + x.i * y.i) / denom, (x.i * y.r - x.r * y.i) / denom};
}

#define EXPR_double(OP, Ta, Tb, x, y, OP_SYM) ((double)x OP_SYM (double)y)
#define EXPR_float(OP, Ta, Tb, x, y, OP_SYM) ((float)x OP_SYM (float)y)
#define EXPR_int64(OP, Ta, Tb, x, y, OP_SYM) (x OP_SYM y)
#define EXPR_int32(OP, Ta, Tb, x, y, OP_SYM) (x OP_SYM y)
#define EXPR_int16(OP, Ta, Tb, x, y, OP_SYM) (x OP_SYM y)
#define EXPR_uint8(OP, Ta, Tb, x, y, OP_SYM) (x OP_SYM y)
#define EXPR_cpx(OP, Ta, Tb, x, y, OP_SYM) cpx_##OP(cpx_from_##Ta(x), cpx_from_##Tb(y))
#define EXPR_cpx64(OP, Ta, Tb, x, y, OP_SYM) cpx_f_from_cpx(cpx_##OP(cpx_from_##Ta(x), cpx_from_##Tb(y)))

#define DEFINE_V_UFUNC(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr, OP_SYM) \
void v_##OP##_##Ta_tok##_##Tb_tok##_##Tr_tok(const Ta *a, const Tb *b, Tr *res, int size) { \
    if (a == NULL || b == NULL || res == NULL || size <= 0) return; \
    for (int i = 0; i < size; i++) { \
        Ta x = a[i]; \
        Tb y = b[i]; \
        res[i] = EXPR_##Tr_tok(OP, Ta_tok, Tb_tok, x, y, OP_SYM); \
        (void)x; (void)y; \
    } \
}

#define DEFINE_S_UFUNC(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr, OP_SYM) \
void s_##OP##_##Ta_tok##_##Tb_tok##_##Tr_tok(const Ta *a, const int *stridesA, \
                                             const Tb *b, const int *stridesB, \
                                             Tr *res, const int *stridesRes, \
                                             const int *shape, int rank) { \
    if (a == NULL || b == NULL || res == NULL || rank <= 0 || rank > 8) return; \
    int total_elements = 1; \
    for (int i = 0; i < rank; i++) total_elements *= shape[i]; \
    int coord[8] = {0}; \
    int offsetA = 0, offsetB = 0, offsetRes = 0; \
    for (int el = 0; el < total_elements; el++) { \
        Ta x = a[offsetA]; \
        Tb y = b[offsetB]; \
        res[offsetRes] = EXPR_##Tr_tok(OP, Ta_tok, Tb_tok, x, y, OP_SYM); \
        (void)x; (void)y; \
        for (int d = rank - 1; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) { \
                offsetA += stridesA[d]; \
                offsetB += stridesB[d]; \
                offsetRes += stridesRes[d]; \
                break; \
            } \
            coord[d] = 0; \
            offsetA -= (shape[d] - 1) * stridesA[d]; \
            offsetB -= (shape[d] - 1) * stridesB[d]; \
            offsetRes -= (shape[d] - 1) * stridesRes[d]; \
        } \
    } \
}

#define DEFINE_V_S_UFUNC(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr, OP_SYM) \
  DEFINE_V_UFUNC(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr, OP_SYM) \
  DEFINE_S_UFUNC(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr, OP_SYM)

#define BUILD_ADD_COMBINATIONS(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr) \
  DEFINE_V_S_UFUNC(add, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr, +)

#define BUILD_SUB_COMBINATIONS(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr) \
  DEFINE_V_S_UFUNC(sub, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr, -)

#define BUILD_MUL_COMBINATIONS(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr) \
  DEFINE_V_S_UFUNC(mul, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr, *)

#define BUILD_DIV_COMBINATIONS(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr) \
  DEFINE_V_S_UFUNC(div, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr, /)

GENERATE_COMMUTATIVE_COMBINATIONS(add, BUILD_ADD_COMBINATIONS)
GENERATE_OP_COMBINATIONS(sub, BUILD_SUB_COMBINATIONS)
GENERATE_COMMUTATIVE_COMBINATIONS(mul, BUILD_MUL_COMBINATIONS)
GENERATE_DIV_COMBINATIONS(div, BUILD_DIV_COMBINATIONS)

void cast_uint8_to_double(const uint8_t *src, double *dst, int size) {
    if (src == NULL || dst == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        dst[i] = (double)src[i];
    }
}

void cast_int16_to_double(const int16_t *src, double *dst, int size) {
    if (src == NULL || dst == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        dst[i] = (double)src[i];
    }
}

void cast_double_to_uint8(const double *src, uint8_t *dst, int size) {
    if (src == NULL || dst == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        dst[i] = (uint8_t)src[i];
    }
}

void cast_double_to_int16(const double *src, int16_t *dst, int size) {
    if (src == NULL || dst == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        dst[i] = (int16_t)src[i];
    }
}

void s_cast_uint8_to_double(const uint8_t *src, const int *stridesSrc, double *dst, const int *stridesDst, const int *shape, int rank) {
    if (src == NULL || dst == NULL || rank <= 0 || rank > 8) return;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    int coord[8] = {0};
    int offsetSrc = 0, offsetDst = 0;
    for (int el = 0; el < total_elements; el++) {
        dst[offsetDst] = (double)src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                offsetDst += stridesDst[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            offsetDst -= (shape[d] - 1) * stridesDst[d];
        }
    }
}

void s_cast_int16_to_double(const int16_t *src, const int *stridesSrc, double *dst, const int *stridesDst, const int *shape, int rank) {
    if (src == NULL || dst == NULL || rank <= 0 || rank > 8) return;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    int coord[8] = {0};
    int offsetSrc = 0, offsetDst = 0;
    for (int el = 0; el < total_elements; el++) {
        dst[offsetDst] = (double)src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                offsetDst += stridesDst[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            offsetDst -= (shape[d] - 1) * stridesDst[d];
        }
    }
}

void s_cast_double_to_uint8(const double *src, const int *stridesSrc, uint8_t *dst, const int *stridesDst, const int *shape, int rank) {
    if (src == NULL || dst == NULL || rank <= 0 || rank > 8) return;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    int coord[8] = {0};
    int offsetSrc = 0, offsetDst = 0;
    for (int el = 0; el < total_elements; el++) {
        dst[offsetDst] = (uint8_t)src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                offsetDst += stridesDst[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            offsetDst -= (shape[d] - 1) * stridesDst[d];
        }
    }
}

void s_cast_double_to_int16(const double *src, const int *stridesSrc, int16_t *dst, const int *stridesDst, const int *shape, int rank) {
    if (src == NULL || dst == NULL || rank <= 0 || rank > 8) return;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    int coord[8] = {0};
    int offsetSrc = 0, offsetDst = 0;
    for (int el = 0; el < total_elements; el++) {
        dst[offsetDst] = (int16_t)src[offsetSrc];
        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetSrc += stridesSrc[d];
                offsetDst += stridesDst[d];
                break;
            }
            coord[d] = 0;
            offsetSrc -= (shape[d] - 1) * stridesSrc[d];
            offsetDst -= (shape[d] - 1) * stridesDst[d];
        }
    }
}

/* ============================================================================
 * copy_and_cast_strided
 * ============================================================================
 * A fused native C kernel that flattens any strided multi-dimensional view
 * (ranks 0 to 8) and casts its elements to the target data type in a single
 * contiguous pass.
 *
 * RATIONALE & DESIGN TRADE-OFFS (LOOP FUSION OPTIMIZATION):
 * - While it is modular to split this into a type-independent blind copy
 *   (memcpy elements along strides) followed by a contiguous linear cast sweep,
 *   that modularity requires allocating temporary contiguous arrays on the unmanaged
 *   heap, writing to them, reading them back to cast, and disposing them.
 * - This fused implementation performs both layout flattening and type upcasting
 *   simultaneously in a single native pass directly into the destination FFI buffer,
 *   bypassing intermediate allocations and halving memory bus traffic.
 * - Type dispatching is done via nested switch blocks. Since type tags remain
 *   constant throughout the entire loop, CPU branch prediction completely
 *   eliminates the branching overhead inside the hot loop.
 */
void copy_and_cast_strided(
    int src_type, const void *src_ptr, const int *strides_src,
    int dest_type, void *dest_ptr, const int *shape, int rank) {
    
    if (src_ptr == NULL || dest_ptr == NULL || rank < 0 || rank > 8) return;

    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    int coord[8] = {0};
    int offsetSrc = 0;

    for (int el = 0; el < total_elements; el++) {
        double r_val = 0.0;
        double i_val = 0.0;

        if (rank == 0) {
            offsetSrc = 0;
        }

        switch (src_type) {
            case 0: // float32
                r_val = (double)((const float*)src_ptr)[offsetSrc];
                break;
            case 1: // float64
                r_val = ((const double*)src_ptr)[offsetSrc];
                break;
            case 2: // int32
                r_val = (double)((const int32_t*)src_ptr)[offsetSrc];
                break;
            case 3: // int64
                r_val = (double)((const int64_t*)src_ptr)[offsetSrc];
                break;
            case 4: // uint8
                r_val = (double)((const uint8_t*)src_ptr)[offsetSrc];
                break;
            case 5: // int16
                r_val = (double)((const int16_t*)src_ptr)[offsetSrc];
                break;
            case 6: // complex64
                r_val = (double)((const cpx_f_t*)src_ptr)[offsetSrc].r;
                i_val = (double)((const cpx_f_t*)src_ptr)[offsetSrc].i;
                break;
            case 7: // complex128
                r_val = ((const cpx_t*)src_ptr)[offsetSrc].r;
                i_val = ((const cpx_t*)src_ptr)[offsetSrc].i;
                break;
            case 8: // boolean
                r_val = ((const uint8_t*)src_ptr)[offsetSrc] ? 1.0 : 0.0;
                break;
        }

        switch (dest_type) {
            case 0: // float32
                ((float*)dest_ptr)[el] = (float)r_val;
                break;
            case 1: // float64
                ((double*)dest_ptr)[el] = r_val;
                break;
            case 2: // int32
                ((int32_t*)dest_ptr)[el] = (int32_t)r_val;
                break;
            case 3: // int64
                ((int64_t*)dest_ptr)[el] = (int64_t)r_val;
                break;
            case 4: // uint8
                ((uint8_t*)dest_ptr)[el] = (uint8_t)r_val;
                break;
            case 5: // int16
                ((int16_t*)dest_ptr)[el] = (int16_t)r_val;
                break;
            case 6: // complex64
                ((cpx_f_t*)dest_ptr)[el].r = (float)r_val;
                ((cpx_f_t*)dest_ptr)[el].i = (float)i_val;
                break;
            case 7: // complex128
                ((cpx_t*)dest_ptr)[el].r = r_val;
                ((cpx_t*)dest_ptr)[el].i = i_val;
                break;
            case 8: // boolean
                ((uint8_t*)dest_ptr)[el] = (r_val != 0.0 || i_val != 0.0) ? 1 : 0;
                break;
        }

        if (rank > 0) {
            for (int d = rank - 1; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetSrc += strides_src[d];
                    break;
                }
                coord[d] = 0;
                offsetSrc -= (shape[d] - 1) * strides_src[d];
            }
        }
    }
}

#include <string.h>

static void copy_advanced_recursive_c(
    const char *src_ptr,
    char **dest_ptr_ref,
    const int *src_strides,
    const int *src_shape,
    int rank,
    int byte_width,
    const int *types,
    const int *index_vals,
    const int *slice_starts,
    const int *slice_stops,
    const int *slice_steps,
    int **indices_ptrs,
    const int *indices_lens,
    int src_dim,
    int current_offset
) {
    if (src_dim == rank) {
        memcpy(*dest_ptr_ref, src_ptr + current_offset, byte_width);
        *dest_ptr_ref += byte_width;
        return;
    }

    int type = types[src_dim];
    int stride = src_strides[src_dim];

    if (type == 0) { // Index
        int idx = index_vals[src_dim];
        copy_advanced_recursive_c(
            src_ptr, dest_ptr_ref, src_strides, src_shape, rank, byte_width,
            types, index_vals, slice_starts, slice_stops, slice_steps,
            indices_ptrs, indices_lens, src_dim + 1,
            current_offset + idx * stride * byte_width
        );
    } else if (type == 1) { // Slice
        int start = slice_starts[src_dim];
        int stop = slice_stops[src_dim];
        int step = slice_steps[src_dim];
        
        if (step > 0) {
            for (int idx = start; idx < stop; idx += step) {
                copy_advanced_recursive_c(
                    src_ptr, dest_ptr_ref, src_strides, src_shape, rank, byte_width,
                    types, index_vals, slice_starts, slice_stops, slice_steps,
                    indices_ptrs, indices_lens, src_dim + 1,
                    current_offset + idx * stride * byte_width
                );
            }
        } else {
            for (int idx = start; idx > stop; idx += step) {
                copy_advanced_recursive_c(
                    src_ptr, dest_ptr_ref, src_strides, src_shape, rank, byte_width,
                    types, index_vals, slice_starts, slice_stops, slice_steps,
                    indices_ptrs, indices_lens, src_dim + 1,
                    current_offset + idx * stride * byte_width
                );
            }
        }
    } else if (type == 2) { // Indices
        int *indices = indices_ptrs[src_dim];
        int len = indices_lens[src_dim];
        for (int i = 0; i < len; i++) {
            int idx = indices[i];
            copy_advanced_recursive_c(
                src_ptr, dest_ptr_ref, src_strides, src_shape, rank, byte_width,
                types, index_vals, slice_starts, slice_stops, slice_steps,
                indices_ptrs, indices_lens, src_dim + 1,
                current_offset + idx * stride * byte_width
            );
        }
    }
}

void copy_advanced_c(
    const void *src_ptr,
    void *dest_ptr,
    const int *src_strides,
    const int *src_shape,
    int rank,
    int byte_width,
    const int *types,
    const int *index_vals,
    const int *slice_starts,
    const int *slice_stops,
    const int *slice_steps,
    int **indices_ptrs,
    const int *indices_lens
) {
    if (src_ptr == NULL || dest_ptr == NULL || rank <= 0) return;
    char *dest_ptr_cursor = (char *)dest_ptr;
    copy_advanced_recursive_c(
        (const char *)src_ptr,
        &dest_ptr_cursor,
        src_strides,
        src_shape,
        rank,
        byte_width,
        types,
        index_vals,
        slice_starts,
        slice_stops,
        slice_steps,
        indices_ptrs,
        indices_lens,
        0,
        0
    );
}

/* ============================================================================
 * SECTION 10: NumPy-Compatible Universal Functions (ufuncs)
 * ============================================================================
 */

static int division_error_flag = 0;

int get_and_reset_division_error(void) {
    int err = division_error_flag;
    division_error_flag = 0;
    return err;
}

static inline double double_floordiv(double x, double y) {
    if (y == 0.0) return NAN;
    return floor(x / y);
}
static inline float float_floordiv(float x, float y) {
    if (y == 0.0f) return NAN;
    return floorf(x / y);
}
static inline int64_t int64_floordiv(int64_t x, int64_t y) {
    if (y == 0) {
        division_error_flag = 1;
        return 0;
    }
    if (x == INT64_MIN && y == -1) {
        division_error_flag = 2;
        return INT64_MIN;
    }
    int64_t res = x / y;
    int64_t rem = x % y;
    if (rem != 0 && ((x < 0) ^ (y < 0))) {
        res--;
    }
    return res;
}
static inline int32_t int32_floordiv(int32_t x, int32_t y) {
    if (y == 0) {
        division_error_flag = 1;
        return 0;
    }
    if (x == INT32_MIN && y == -1) {
        division_error_flag = 2;
        return INT32_MIN;
    }
    int32_t res = x / y;
    int32_t rem = x % y;
    if (rem != 0 && ((x < 0) ^ (y < 0))) {
        res--;
    }
    return res;
}

static inline double double_remainder(double x, double y) {
    if (y == 0.0) return NAN;
    double rem = fmod(x, y);
    if (rem != 0.0 && ((rem < 0.0) != (y < 0.0))) {
        rem += y;
    }
    return rem;
}
static inline float float_remainder(float x, float y) {
    if (y == 0.0f) return NAN;
    float rem = fmodf(x, y);
    if (rem != 0.0f && ((rem < 0.0f) != (y < 0.0f))) {
        rem += y;
    }
    return rem;
}
static inline int64_t int64_remainder(int64_t x, int64_t y) {
    if (y == 0) {
        division_error_flag = 1;
        return 0;
    }
    if (x == INT64_MIN && y == -1) {
        division_error_flag = 2;
        return 0;
    }
    int64_t rem = x % y;
    if (rem != 0 && ((rem < 0) != (y < 0))) {
        rem += y;
    }
    return rem;
}
static inline int32_t int32_remainder(int32_t x, int32_t y) {
    if (y == 0) {
        division_error_flag = 1;
        return 0;
    }
    if (x == INT32_MIN && y == -1) {
        division_error_flag = 2;
        return 0;
    }
    int32_t rem = x % y;
    if (rem != 0 && ((rem < 0) != (y < 0))) {
        rem += y;
    }
    return rem;
}

static inline cpx_t cpx_square(cpx_t z) {
    return (cpx_t){z.r * z.r - z.i * z.i, 2.0 * z.r * z.i};
}
static inline cpx_f_t cpx_square_f(cpx_f_t z) {
    return (cpx_f_t){z.r * z.r - z.i * z.i, 2.0f * z.r * z.i};
}

#define DEFINE_CONTIGUOUS_UNARY_IMPL(name, typeSrc, typeRes, expr) \
void name(const typeSrc *src, typeRes *res, int size) { \
    if (src == NULL || res == NULL || size <= 0) return; \
    for (int i = 0; i < size; i++) { \
        typeSrc x = src[i]; \
        res[i] = (expr); \
    } \
}

#define DEFINE_CONTIGUOUS_BINARY_IMPL(name, typeA, typeB, typeRes, expr) \
void name(const typeA *a, const typeB *b, typeRes *res, int size) { \
    if (a == NULL || b == NULL || res == NULL || size <= 0) return; \
    for (int i = 0; i < size; i++) { \
        typeA x = a[i]; \
        typeB y = b[i]; \
        res[i] = (expr); \
    } \
}

/* 1. Contiguous (Vector) Implementations */

DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_double, double, double, x * x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_float, float, float, x * x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_int64, int64_t, int64_t, x * x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_int32, int32_t, int32_t, x * x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_complex128, cpx_t, cpx_t, cpx_square(x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_square_complex64, cpx_f_t, cpx_f_t, cpx_square_f(x))

DEFINE_CONTIGUOUS_BINARY_IMPL(v_pow_double, double, double, double, pow(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_pow_float, float, float, float, powf(x, y))

DEFINE_CONTIGUOUS_BINARY_IMPL(v_floordiv_double, double, double, double, double_floordiv(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_floordiv_float, float, float, float, float_floordiv(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_floordiv_int64, int64_t, int64_t, int64_t, int64_floordiv(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_floordiv_int32, int32_t, int32_t, int32_t, int32_floordiv(x, y))

DEFINE_CONTIGUOUS_BINARY_IMPL(v_remainder_double, double, double, double, double_remainder(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_remainder_float, float, float, float, float_remainder(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_remainder_int64, int64_t, int64_t, int64_t, int64_remainder(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_remainder_int32, int32_t, int32_t, int32_t, int32_remainder(x, y))

DEFINE_CONTIGUOUS_UNARY_IMPL(v_isnan_double, double, uint8_t, isnan(x) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_isnan_float, float, uint8_t, isnan(x) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_isnan_complex128, cpx_t, uint8_t, (isnan(x.r) || isnan(x.i)) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_isnan_complex64, cpx_f_t, uint8_t, (isnan(x.r) || isnan(x.i)) ? 1 : 0)

DEFINE_CONTIGUOUS_UNARY_IMPL(v_isinf_double, double, uint8_t, isinf(x) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_isinf_float, float, uint8_t, isinf(x) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_isinf_complex128, cpx_t, uint8_t, (isinf(x.r) || isinf(x.i)) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_isinf_complex64, cpx_f_t, uint8_t, (isinf(x.r) || isinf(x.i)) ? 1 : 0)

DEFINE_CONTIGUOUS_UNARY_IMPL(v_isfinite_double, double, uint8_t, isfinite(x) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_isfinite_float, float, uint8_t, isfinite(x) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_isfinite_complex128, cpx_t, uint8_t, (isfinite(x.r) && isfinite(x.i)) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_isfinite_complex64, cpx_f_t, uint8_t, (isfinite(x.r) && isfinite(x.i)) ? 1 : 0)

DEFINE_CONTIGUOUS_BINARY_IMPL(v_copysign_double, double, double, double, copysign(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_copysign_float, float, float, float, copysignf(x, y))

/* 2. Strided Multidimensional Implementations */

#define DEFINE_STRIDED_UNARY_IMPL(name, typeSrc, typeRes, expr) \
void name(const typeSrc *src, const int *stridesSrc, \
          typeRes *res, const int *stridesRes, \
          const int *shape, int rank) { \
    if (src == NULL || res == NULL || shape == NULL || rank <= 0 || rank > 8) return; \
    int total_elements = 1; \
    for (int i = 0; i < rank; i++) total_elements *= shape[i]; \
    int coord[8] = {0}; \
    int offsetSrc = 0, offsetRes = 0; \
    for (int el = 0; el < total_elements; el++) { \
        typeSrc x = src[offsetSrc]; \
        res[offsetRes] = (expr); \
        for (int d = rank - 1; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) { \
                offsetSrc += stridesSrc[d]; \
                offsetRes += stridesRes[d]; \
                break; \
            } \
            coord[d] = 0; \
            offsetSrc -= (shape[d] - 1) * stridesSrc[d]; \
            offsetRes -= (shape[d] - 1) * stridesRes[d]; \
        } \
    } \
}

DEFINE_STRIDED_UNARY_IMPL(s_square_double, double, double, x * x)
DEFINE_STRIDED_UNARY_IMPL(s_square_float, float, float, x * x)
DEFINE_STRIDED_UNARY_IMPL(s_square_int64, int64_t, int64_t, x * x)
DEFINE_STRIDED_UNARY_IMPL(s_square_int32, int32_t, int32_t, x * x)
DEFINE_STRIDED_UNARY_IMPL(s_square_complex128, cpx_t, cpx_t, cpx_square(x))
DEFINE_STRIDED_UNARY_IMPL(s_square_complex64, cpx_f_t, cpx_f_t, cpx_square_f(x))

#define DEFINE_STRIDED_BINARY_IMPL(name, typeA, typeB, typeRes, expr) \
void name(const typeA *a, const int *stridesA, \
          const typeB *b, const int *stridesB, \
          typeRes *res, const int *stridesRes, \
          const int *shape, int rank) { \
    if (a == NULL || b == NULL || res == NULL || shape == NULL || rank <= 0 || rank > 8) return; \
    int total_elements = 1; \
    for (int i = 0; i < rank; i++) total_elements *= shape[i]; \
    int coord[8] = {0}; \
    int offsetA = 0, offsetB = 0, offsetRes = 0; \
    for (int el = 0; el < total_elements; el++) { \
        typeA x = a[offsetA]; \
        typeB y = b[offsetB]; \
        res[offsetRes] = (expr); \
        for (int d = rank - 1; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) { \
                offsetA += stridesA[d]; \
                offsetB += stridesB[d]; \
                offsetRes += stridesRes[d]; \
                break; \
            } \
            coord[d] = 0; \
            offsetA -= (shape[d] - 1) * stridesA[d]; \
            offsetB -= (shape[d] - 1) * stridesB[d]; \
            offsetRes -= (shape[d] - 1) * stridesRes[d]; \
        } \
    } \
}

DEFINE_STRIDED_BINARY_IMPL(s_pow_double, double, double, double, pow(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_pow_float, float, float, float, powf(x, y))

DEFINE_STRIDED_BINARY_IMPL(s_floordiv_double, double, double, double, double_floordiv(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_floordiv_float, float, float, float, float_floordiv(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_floordiv_int64, int64_t, int64_t, int64_t, int64_floordiv(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_floordiv_int32, int32_t, int32_t, int32_t, int32_floordiv(x, y))

DEFINE_STRIDED_BINARY_IMPL(s_remainder_double, double, double, double, double_remainder(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_remainder_float, float, float, float, float_remainder(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_remainder_int64, int64_t, int64_t, int64_t, int64_remainder(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_remainder_int32, int32_t, int32_t, int32_t, int32_remainder(x, y))

DEFINE_STRIDED_BINARY_IMPL(s_copysign_double, double, double, double, copysign(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_copysign_float, float, float, float, copysignf(x, y))

DEFINE_STRIDED_UNARY_IMPL(s_isnan_double, double, uint8_t, isnan(x) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_isnan_float, float, uint8_t, isnan(x) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_isnan_complex128, cpx_t, uint8_t, (isnan(x.r) || isnan(x.i)) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_isnan_complex64, cpx_f_t, uint8_t, (isnan(x.r) || isnan(x.i)) ? 1 : 0)

DEFINE_STRIDED_UNARY_IMPL(s_isinf_double, double, uint8_t, isinf(x) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_isinf_float, float, uint8_t, isinf(x) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_isinf_complex128, cpx_t, uint8_t, (isinf(x.r) || isinf(x.i)) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_isinf_complex64, cpx_f_t, uint8_t, (isinf(x.r) || isinf(x.i)) ? 1 : 0)

DEFINE_STRIDED_UNARY_IMPL(s_isfinite_double, double, uint8_t, isfinite(x) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_isfinite_float, float, uint8_t, isfinite(x) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_isfinite_complex128, cpx_t, uint8_t, (isfinite(x.r) && isfinite(x.i)) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_isfinite_complex64, cpx_f_t, uint8_t, (isfinite(x.r) && isfinite(x.i)) ? 1 : 0)

/* Logical and Casting-to-Boolean Implementations */
DEFINE_CONTIGUOUS_UNARY_IMPL(v_to_bool_double, double, uint8_t, (x != 0.0) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_to_bool_float, float, uint8_t, (x != 0.0f) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_to_bool_int64, int64_t, uint8_t, (x != 0) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_to_bool_int32, int32_t, uint8_t, (x != 0) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_to_bool_uint8, uint8_t, uint8_t, (x != 0) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_to_bool_int16, int16_t, uint8_t, (x != 0) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_to_bool_complex128, cpx_t, uint8_t, (x.r != 0.0 || x.i != 0.0) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_to_bool_complex64, cpx_f_t, uint8_t, (x.r != 0.0f || x.i != 0.0f) ? 1 : 0)

DEFINE_STRIDED_UNARY_IMPL(s_to_bool_double, double, uint8_t, (x != 0.0) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_to_bool_float, float, uint8_t, (x != 0.0f) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_to_bool_int64, int64_t, uint8_t, (x != 0) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_to_bool_int32, int32_t, uint8_t, (x != 0) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_to_bool_uint8, uint8_t, uint8_t, (x != 0) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_to_bool_int16, int16_t, uint8_t, (x != 0) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_to_bool_complex128, cpx_t, uint8_t, (x.r != 0.0 || x.i != 0.0) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_to_bool_complex64, cpx_f_t, uint8_t, (x.r != 0.0f || x.i != 0.0f) ? 1 : 0)

DEFINE_CONTIGUOUS_BINARY_IMPL(v_logical_and, uint8_t, uint8_t, uint8_t, (x && y) ? 1 : 0)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_logical_or, uint8_t, uint8_t, uint8_t, (x || y) ? 1 : 0)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_logical_xor, uint8_t, uint8_t, uint8_t, ((x != 0) != (y != 0)) ? 1 : 0)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_logical_not, uint8_t, uint8_t, (!x) ? 1 : 0)

DEFINE_STRIDED_BINARY_IMPL(s_logical_and, uint8_t, uint8_t, uint8_t, (x && y) ? 1 : 0)
DEFINE_STRIDED_BINARY_IMPL(s_logical_or, uint8_t, uint8_t, uint8_t, (x || y) ? 1 : 0)
DEFINE_STRIDED_BINARY_IMPL(s_logical_xor, uint8_t, uint8_t, uint8_t, ((x != 0) != (y != 0)) ? 1 : 0)
DEFINE_STRIDED_UNARY_IMPL(s_logical_not, uint8_t, uint8_t, (!x) ? 1 : 0)

/* Safe shift helpers to prevent undefined C behavior */
static inline int32_t safe_left_shift_int32(int32_t x, int32_t y) {
    if (y < 0 || y >= 32) return 0;
    return x << y;
}
static inline int32_t safe_right_shift_int32(int32_t x, int32_t y) {
    if (y < 0 || y >= 32) return 0;
    return x >> y;
}

static inline int64_t safe_left_shift_int64(int64_t x, int64_t y) {
    if (y < 0 || y >= 64) return 0;
    return x << y;
}
static inline int64_t safe_right_shift_int64(int64_t x, int64_t y) {
    if (y < 0 || y >= 64) return 0;
    return x >> y;
}

static inline uint8_t safe_left_shift_uint8(uint8_t x, uint8_t y) {
    if (y >= 8) return 0;
    return (uint8_t)(x << y);
}
static inline uint8_t safe_right_shift_uint8(uint8_t x, uint8_t y) {
    if (y >= 8) return 0;
    return (uint8_t)(x >> y);
}

static inline int16_t safe_left_shift_int16(int16_t x, int16_t y) {
    if (y < 0 || y >= 16) return 0;
    return (int16_t)(x << y);
}
static inline int16_t safe_right_shift_int16(int16_t x, int16_t y) {
    if (y < 0 || y >= 16) return 0;
    return (int16_t)(x >> y);
}

/* Bitwise AND implementations */
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_and_int32, int32_t, int32_t, int32_t, x & y)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_and_int64, int64_t, int64_t, int64_t, x & y)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_and_uint8, uint8_t, uint8_t, uint8_t, x & y)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_and_int16, int16_t, int16_t, int16_t, x & y)

DEFINE_STRIDED_BINARY_IMPL(s_bitwise_and_int32, int32_t, int32_t, int32_t, x & y)
DEFINE_STRIDED_BINARY_IMPL(s_bitwise_and_int64, int64_t, int64_t, int64_t, x & y)
DEFINE_STRIDED_BINARY_IMPL(s_bitwise_and_uint8, uint8_t, uint8_t, uint8_t, x & y)
DEFINE_STRIDED_BINARY_IMPL(s_bitwise_and_int16, int16_t, int16_t, int16_t, x & y)

/* Bitwise OR implementations */
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_or_int32, int32_t, int32_t, int32_t, x | y)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_or_int64, int64_t, int64_t, int64_t, x | y)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_or_uint8, uint8_t, uint8_t, uint8_t, x | y)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_or_int16, int16_t, int16_t, int16_t, x | y)

DEFINE_STRIDED_BINARY_IMPL(s_bitwise_or_int32, int32_t, int32_t, int32_t, x | y)
DEFINE_STRIDED_BINARY_IMPL(s_bitwise_or_int64, int64_t, int64_t, int64_t, x | y)
DEFINE_STRIDED_BINARY_IMPL(s_bitwise_or_uint8, uint8_t, uint8_t, uint8_t, x | y)
DEFINE_STRIDED_BINARY_IMPL(s_bitwise_or_int16, int16_t, int16_t, int16_t, x | y)

/* Bitwise XOR implementations */
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_xor_int32, int32_t, int32_t, int32_t, x ^ y)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_xor_int64, int64_t, int64_t, int64_t, x ^ y)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_xor_uint8, uint8_t, uint8_t, uint8_t, x ^ y)
DEFINE_CONTIGUOUS_BINARY_IMPL(v_bitwise_xor_int16, int16_t, int16_t, int16_t, x ^ y)

DEFINE_STRIDED_BINARY_IMPL(s_bitwise_xor_int32, int32_t, int32_t, int32_t, x ^ y)
DEFINE_STRIDED_BINARY_IMPL(s_bitwise_xor_int64, int64_t, int64_t, int64_t, x ^ y)
DEFINE_STRIDED_BINARY_IMPL(s_bitwise_xor_uint8, uint8_t, uint8_t, uint8_t, x ^ y)
DEFINE_STRIDED_BINARY_IMPL(s_bitwise_xor_int16, int16_t, int16_t, int16_t, x ^ y)

/* Left Shift implementations */
DEFINE_CONTIGUOUS_BINARY_IMPL(v_left_shift_int32, int32_t, int32_t, int32_t, safe_left_shift_int32(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_left_shift_int64, int64_t, int64_t, int64_t, safe_left_shift_int64(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_left_shift_uint8, uint8_t, uint8_t, uint8_t, safe_left_shift_uint8(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_left_shift_int16, int16_t, int16_t, int16_t, safe_left_shift_int16(x, y))

DEFINE_STRIDED_BINARY_IMPL(s_left_shift_int32, int32_t, int32_t, int32_t, safe_left_shift_int32(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_left_shift_int64, int64_t, int64_t, int64_t, safe_left_shift_int64(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_left_shift_uint8, uint8_t, uint8_t, uint8_t, safe_left_shift_uint8(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_left_shift_int16, int16_t, int16_t, int16_t, safe_left_shift_int16(x, y))

/* Right Shift implementations */
DEFINE_CONTIGUOUS_BINARY_IMPL(v_right_shift_int32, int32_t, int32_t, int32_t, safe_right_shift_int32(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_right_shift_int64, int64_t, int64_t, int64_t, safe_right_shift_int64(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_right_shift_uint8, uint8_t, uint8_t, uint8_t, safe_right_shift_uint8(x, y))
DEFINE_CONTIGUOUS_BINARY_IMPL(v_right_shift_int16, int16_t, int16_t, int16_t, safe_right_shift_int16(x, y))

DEFINE_STRIDED_BINARY_IMPL(s_right_shift_int32, int32_t, int32_t, int32_t, safe_right_shift_int32(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_right_shift_int64, int64_t, int64_t, int64_t, safe_right_shift_int64(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_right_shift_uint8, uint8_t, uint8_t, uint8_t, safe_right_shift_uint8(x, y))
DEFINE_STRIDED_BINARY_IMPL(s_right_shift_int16, int16_t, int16_t, int16_t, safe_right_shift_int16(x, y))

/* Invert (Bitwise Negation) implementations */
DEFINE_CONTIGUOUS_UNARY_IMPL(v_invert_int32, int32_t, int32_t, ~x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_invert_int64, int64_t, int64_t, ~x)
DEFINE_CONTIGUOUS_UNARY_IMPL(v_invert_uint8, uint8_t, uint8_t, (uint8_t)(~x))
DEFINE_CONTIGUOUS_UNARY_IMPL(v_invert_int16, int16_t, int16_t, (int16_t)(~x))

DEFINE_STRIDED_UNARY_IMPL(s_invert_int32, int32_t, int32_t, ~x)
DEFINE_STRIDED_UNARY_IMPL(s_invert_int64, int64_t, int64_t, ~x)
DEFINE_STRIDED_UNARY_IMPL(s_invert_uint8, uint8_t, uint8_t, (uint8_t)(~x))
DEFINE_STRIDED_UNARY_IMPL(s_invert_int16, int16_t, int16_t, (int16_t)(~x))



/* ============================================================================
 * SECTION: Kronecker Product, Vector Outer Product, Cross Product & Norms
 * ============================================================================
 */

#define DEFINE_KRON_IMPL(name, type, op) \
void name(const type *a, const int *stridesA, const int *shapeA, \
          const type *b, const int *stridesB, const int *shapeB, \
          type *res, const int *stridesRes, const int *shapeRes, int rank) { \
    if (a == NULL || b == NULL || res == NULL || rank <= 0 || rank > 8) return; \
    int total_elements = 1; \
    for (int i = 0; i < rank; i++) total_elements *= shapeRes[i]; \
    int coord[8] = {0}; \
    for (int el = 0; el < total_elements; el++) { \
        int offsetA = 0; \
        int offsetB = 0; \
        int offsetRes = 0; \
        for (int i = 0; i < rank; i++) { \
            int ca = coord[i] / shapeB[i]; \
            int cb = coord[i] % shapeB[i]; \
            offsetA += ca * stridesA[i]; \
            offsetB += cb * stridesB[i]; \
            offsetRes += coord[i] * stridesRes[i]; \
        } \
        res[offsetRes] = op(a[offsetA], b[offsetB]); \
        for (int d = rank - 1; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shapeRes[d]) break; \
            coord[d] = 0; \
        } \
    } \
}

#define NUM_MUL_OP(x, y) ((x) * (y))
#define BOOL_AND_OP(x, y) ((x) && (y))

DEFINE_KRON_IMPL(s_kron_double, double, NUM_MUL_OP)
DEFINE_KRON_IMPL(s_kron_float, float, NUM_MUL_OP)
DEFINE_KRON_IMPL(s_kron_int64, int64_t, NUM_MUL_OP)
DEFINE_KRON_IMPL(s_kron_int32, int32_t, NUM_MUL_OP)
DEFINE_KRON_IMPL(s_kron_uint8, uint8_t, NUM_MUL_OP)
DEFINE_KRON_IMPL(s_kron_int16, int16_t, NUM_MUL_OP)
DEFINE_KRON_IMPL(s_kron_boolean, uint8_t, BOOL_AND_OP)

static inline cpx_t cpx_kron_mul(cpx_t c1, cpx_t c2) {
    cpx_t res;
    res.r = c1.r * c2.r - c1.i * c2.i;
    res.i = c1.r * c2.i + c1.i * c2.r;
    return res;
}

static inline cpx_f_t cpx_f_kron_mul(cpx_f_t c1, cpx_f_t c2) {
    cpx_f_t res;
    res.r = c1.r * c2.r - c1.i * c2.i;
    res.i = c1.r * c2.i + c1.i * c2.r;
    return res;
}

DEFINE_KRON_IMPL(s_kron_complex128, cpx_t, cpx_kron_mul)
DEFINE_KRON_IMPL(s_kron_complex64, cpx_f_t, cpx_f_kron_mul)

/* Vector Outer Product */
#define DEFINE_OUTER_IMPL(name, type, op) \
void name(const type *a, int strideA, int sizeA, \
          const type *b, int strideB, int sizeB, \
          type *res, int strideRowRes, int strideColRes) { \
    if (a == NULL || b == NULL || res == NULL || sizeA <= 0 || sizeB <= 0) return; \
    for (int i = 0; i < sizeA; i++) { \
        type valA = a[i * strideA]; \
        for (int j = 0; j < sizeB; j++) { \
            res[i * strideRowRes + j * strideColRes] = op(valA, b[j * strideB]); \
        } \
    } \
}

DEFINE_OUTER_IMPL(s_outer_double, double, NUM_MUL_OP)
DEFINE_OUTER_IMPL(s_outer_float, float, NUM_MUL_OP)
DEFINE_OUTER_IMPL(s_outer_int64, int64_t, NUM_MUL_OP)
DEFINE_OUTER_IMPL(s_outer_int32, int32_t, NUM_MUL_OP)
DEFINE_OUTER_IMPL(s_outer_uint8, uint8_t, NUM_MUL_OP)
DEFINE_OUTER_IMPL(s_outer_int16, int16_t, NUM_MUL_OP)
DEFINE_OUTER_IMPL(s_outer_boolean, uint8_t, BOOL_AND_OP)
DEFINE_OUTER_IMPL(s_outer_complex128, cpx_t, cpx_kron_mul)
DEFINE_OUTER_IMPL(s_outer_complex64, cpx_f_t, cpx_f_kron_mul)

/* Vector Cross Product */
#define DEFINE_CROSS_3D_IMPL(name, type, op_mul, op_sub) \
void name(const type *a, int strideA, const type *b, int strideB, type *res, int strideRes) { \
    type a0 = a[0], a1 = a[strideA], a2 = a[2 * strideA]; \
    type b0 = b[0], b1 = b[strideB], b2 = b[2 * strideB]; \
    res[0] = op_sub(op_mul(a1, b2), op_mul(a2, b1)); \
    res[strideRes] = op_sub(op_mul(a2, b0), op_mul(a0, b2)); \
    res[2 * strideRes] = op_sub(op_mul(a0, b1), op_mul(a1, b0)); \
}

#define DEFINE_CROSS_2D_IMPL(name, type, op_mul, op_sub) \
void name(const type *a, int strideA, const type *b, int strideB, type *res) { \
    type a0 = a[0], a1 = a[strideA]; \
    type b0 = b[0], b1 = b[strideB]; \
    res[0] = op_sub(op_mul(a0, b1), op_mul(a1, b0)); \
}

#define NUM_SUB_OP(x, y) ((x) - (y))
#define BOOL_SUB_OP(x, y) ((x) ^ (y))

DEFINE_CROSS_3D_IMPL(s_cross_3d_double, double, NUM_MUL_OP, NUM_SUB_OP)
DEFINE_CROSS_2D_IMPL(s_cross_2d_double, double, NUM_MUL_OP, NUM_SUB_OP)

DEFINE_CROSS_3D_IMPL(s_cross_3d_float, float, NUM_MUL_OP, NUM_SUB_OP)
DEFINE_CROSS_2D_IMPL(s_cross_2d_float, float, NUM_MUL_OP, NUM_SUB_OP)

DEFINE_CROSS_3D_IMPL(s_cross_3d_int64, int64_t, NUM_MUL_OP, NUM_SUB_OP)
DEFINE_CROSS_2D_IMPL(s_cross_2d_int64, int64_t, NUM_MUL_OP, NUM_SUB_OP)

DEFINE_CROSS_3D_IMPL(s_cross_3d_int32, int32_t, NUM_MUL_OP, NUM_SUB_OP)
DEFINE_CROSS_2D_IMPL(s_cross_2d_int32, int32_t, NUM_MUL_OP, NUM_SUB_OP)

DEFINE_CROSS_3D_IMPL(s_cross_3d_uint8, uint8_t, NUM_MUL_OP, NUM_SUB_OP)
DEFINE_CROSS_2D_IMPL(s_cross_2d_uint8, uint8_t, NUM_MUL_OP, NUM_SUB_OP)

DEFINE_CROSS_3D_IMPL(s_cross_3d_int16, int16_t, NUM_MUL_OP, NUM_SUB_OP)
DEFINE_CROSS_2D_IMPL(s_cross_2d_int16, int16_t, NUM_MUL_OP, NUM_SUB_OP)

DEFINE_CROSS_3D_IMPL(s_cross_3d_boolean, uint8_t, BOOL_AND_OP, BOOL_SUB_OP)
DEFINE_CROSS_2D_IMPL(s_cross_2d_boolean, uint8_t, BOOL_AND_OP, BOOL_SUB_OP)

DEFINE_CROSS_3D_IMPL(s_cross_3d_complex128, cpx_t, cpx_kron_mul, cpx_sub)
DEFINE_CROSS_2D_IMPL(s_cross_2d_complex128, cpx_t, cpx_kron_mul, cpx_sub)

DEFINE_CROSS_3D_IMPL(s_cross_3d_complex64, cpx_f_t, cpx_f_kron_mul, cpx_sub_f)
DEFINE_CROSS_2D_IMPL(s_cross_2d_complex64, cpx_f_t, cpx_f_kron_mul, cpx_sub_f)

/* Vector Norm Reductions */
#define DEFINE_NORM_REDUCTIONS(suffix, type) \
double r_norm_l1_##suffix(const type *src, int stride, int size) { \
    if (src == NULL || size <= 0) return 0.0; \
    double sum = 0.0; \
    for (int i = 0; i < size; i++) { \
        double val = (double)src[i * stride]; \
        sum += val >= 0 ? val : -val; \
    } \
    return sum; \
} \
double r_norm_l2_##suffix(const type *src, int stride, int size) { \
    if (src == NULL || size <= 0) return 0.0; \
    double sum = 0.0; \
    for (int i = 0; i < size; i++) { \
        double val = (double)src[i * stride]; \
        sum += val * val; \
    } \
    return sum; \
} \
double r_norm_lp_##suffix(const type *src, int stride, int size, double p) { \
    if (src == NULL || size <= 0) return 0.0; \
    double sum = 0.0; \
    for (int i = 0; i < size; i++) { \
        double val = (double)src[i * stride]; \
        double abs_val = val >= 0 ? val : -val; \
        sum += pow(abs_val, p); \
    } \
    return sum; \
} \
double r_norm_inf_##suffix(const type *src, int stride, int size) { \
    if (src == NULL || size <= 0) return 0.0; \
    double max_val = -1.0; \
    for (int i = 0; i < size; i++) { \
        double val = (double)src[i * stride]; \
        double abs_val = val >= 0 ? val : -val; \
        if (abs_val > max_val || max_val < 0) max_val = abs_val; \
    } \
    return max_val; \
} \
double r_norm_neg_inf_##suffix(const type *src, int stride, int size) { \
    if (src == NULL || size <= 0) return 0.0; \
    double min_val = -1.0; \
    for (int i = 0; i < size; i++) { \
        double val = (double)src[i * stride]; \
        double abs_val = val >= 0 ? val : -val; \
        if (abs_val < min_val || min_val < 0) min_val = abs_val; \
    } \
    return min_val; \
}

DEFINE_NORM_REDUCTIONS(double, double)

#define DEFINE_NORM_REDUCTIONS_FLOAT(suffix, type) \
float r_norm_l1_##suffix(const type *src, int stride, int size) { \
    if (src == NULL || size <= 0) return 0.0f; \
    float sum = 0.0f; \
    for (int i = 0; i < size; i++) { \
        float val = (float)src[i * stride]; \
        sum += val >= 0 ? val : -val; \
    } \
    return sum; \
} \
float r_norm_l2_##suffix(const type *src, int stride, int size) { \
    if (src == NULL || size <= 0) return 0.0f; \
    float sum = 0.0f; \
    for (int i = 0; i < size; i++) { \
        float val = (float)src[i * stride]; \
        sum += val * val; \
    } \
    return sum; \
} \
float r_norm_lp_##suffix(const type *src, int stride, int size, float p) { \
    if (src == NULL || size <= 0) return 0.0f; \
    float sum = 0.0f; \
    for (int i = 0; i < size; i++) { \
        float val = (float)src[i * stride]; \
        float abs_val = val >= 0 ? val : -val; \
        sum += powf(abs_val, p); \
    } \
    return sum; \
} \
float r_norm_inf_##suffix(const type *src, int stride, int size) { \
    if (src == NULL || size <= 0) return 0.0f; \
    float max_val = -1.0f; \
    for (int i = 0; i < size; i++) { \
        float val = (float)src[i * stride]; \
        float abs_val = val >= 0 ? val : -val; \
        if (abs_val > max_val || max_val < 0) max_val = abs_val; \
    } \
    return max_val; \
} \
float r_norm_neg_inf_##suffix(const type *src, int stride, int size) { \
    if (src == NULL || size <= 0) return 0.0f; \
    float min_val = -1.0f; \
    for (int i = 0; i < size; i++) { \
        float val = (float)src[i * stride]; \
        float abs_val = val >= 0 ? val : -val; \
        if (abs_val < min_val || min_val < 0) min_val = abs_val; \
    } \
    return min_val; \
}

DEFINE_NORM_REDUCTIONS_FLOAT(float, float)

/* Complex Norms */
#define DEFINE_COMPLEX_NORM_REDUCTIONS(suffix, type, float_type, sqrt_fn, pow_fn) \
float_type r_norm_l1_##suffix(const type *src, int stride, int size) { \
    if (src == NULL || size <= 0) return (float_type)0.0; \
    float_type sum = (float_type)0.0; \
    for (int i = 0; i < size; i++) { \
        float_type r = (float_type)src[i * stride].r; \
        float_type imag = (float_type)src[i * stride].i; \
        sum += sqrt_fn(r * r + imag * imag); \
    } \
    return sum; \
} \
float_type r_norm_l2_##suffix(const type *src, int stride, int size) { \
    if (src == NULL || size <= 0) return (float_type)0.0; \
    float_type sum = (float_type)0.0; \
    for (int i = 0; i < size; i++) { \
        float_type r = (float_type)src[i * stride].r; \
        float_type imag = (float_type)src[i * stride].i; \
        sum += r * r + imag * imag; \
    } \
    return sum; \
} \
float_type r_norm_lp_##suffix(const type *src, int stride, int size, float_type p) { \
    if (src == NULL || size <= 0) return (float_type)0.0; \
    float_type sum = (float_type)0.0; \
    for (int i = 0; i < size; i++) { \
        float_type r = (float_type)src[i * stride].r; \
        float_type imag = (float_type)src[i * stride].i; \
        sum += pow_fn(sqrt_fn(r * r + imag * imag), p); \
    } \
    return sum; \
} \
float_type r_norm_inf_##suffix(const type *src, int stride, int size) { \
    if (src == NULL || size <= 0) return (float_type)0.0; \
    float_type max_val = (float_type)-1.0; \
    for (int i = 0; i < size; i++) { \
        float_type r = (float_type)src[i * stride].r; \
        float_type imag = (float_type)src[i * stride].i; \
        float_type val = sqrt_fn(r * r + imag * imag); \
        if (val > max_val || max_val < (float_type)0.0) max_val = val; \
    } \
    return max_val; \
} \
float_type r_norm_neg_inf_##suffix(const type *src, int stride, int size) { \
    if (src == NULL || size <= 0) return (float_type)0.0; \
    float_type min_val = (float_type)-1.0; \
    for (int i = 0; i < size; i++) { \
        float_type r = (float_type)src[i * stride].r; \
        float_type imag = (float_type)src[i * stride].i; \
        float_type val = sqrt_fn(r * r + imag * imag); \
        if (val < min_val || min_val < (float_type)0.0) min_val = val; \
    } \
    return min_val; \
}

DEFINE_COMPLEX_NORM_REDUCTIONS(complex128, cpx_t, double, sqrt, pow)
DEFINE_COMPLEX_NORM_REDUCTIONS(complex64, cpx_f_t, float, sqrtf, powf)

/* Window Functions */
void v_hanning_double(double *res, int M) {
    if (res == NULL || M <= 0) return;
    if (M == 1) {
        res[0] = 1.0;
        return;
    }
    double pi2 = 2.0 * M_PI;
    for (int n = 0; n < M; n++) {
        res[n] = 0.5 - 0.5 * cos(pi2 * n / (M - 1));
    }
}

void v_hanning_float(float *res, int M) {
    if (res == NULL || M <= 0) return;
    if (M == 1) {
        res[0] = 1.0f;
        return;
    }
    float pi2 = 2.0f * (float)M_PI;
    for (int n = 0; n < M; n++) {
        res[n] = 0.5f - 0.5f * cosf(pi2 * n / (M - 1));
    }
}

void v_hamming_double(double *res, int M) {
    if (res == NULL || M <= 0) return;
    if (M == 1) {
        res[0] = 1.0;
        return;
    }
    double pi2 = 2.0 * M_PI;
    for (int n = 0; n < M; n++) {
        res[n] = 0.54 - 0.46 * cos(pi2 * n / (M - 1));
    }
}

void v_hamming_float(float *res, int M) {
    if (res == NULL || M <= 0) return;
    if (M == 1) {
        res[0] = 1.0f;
        return;
    }
    float pi2 = 2.0f * (float)M_PI;
    for (int n = 0; n < M; n++) {
        res[n] = 0.54f - 0.46f * cosf(pi2 * n / (M - 1));
    }
}

// ============================================================================
// 14. STRIDED TERNARY CLIP FUNCTIONS FOR ALL NUMERIC TYPES
// ============================================================================

#define IMPLEMENT_S_CLIP(TYPE_NAME, TYPE) \
void s_clip_##TYPE_NAME(const TYPE *a, const int *stridesA, \
                        const TYPE *min_val, const int *stridesMin, \
                        const TYPE *max_val, const int *stridesMax, \
                        TYPE *res, const int *stridesRes, \
                        const int *shape, int rank) { \
    if (a == NULL || min_val == NULL || max_val == NULL || res == NULL || rank < 0 || rank > 8) return; \
    int total_elements = 1; \
    for (int i = 0; i < rank; i++) total_elements *= shape[i]; \
    if (rank == 0) { \
        TYPE val = a[0]; \
        if (val < min_val[0]) val = min_val[0]; \
        if (val > max_val[0]) val = max_val[0]; \
        res[0] = val; \
        return; \
    } \
    int is_contiguous = 1; \
    int expected_stride = 1; \
    for (int i = rank - 1; i >= 0; i--) { \
        if (stridesA[i] != expected_stride || \
            stridesMin[i] != expected_stride || \
            stridesMax[i] != expected_stride || \
            stridesRes[i] != expected_stride) { \
            is_contiguous = 0; \
            break; \
        } \
        expected_stride *= shape[i]; \
    } \
    if (is_contiguous) { \
        for (int i = 0; i < total_elements; i++) { \
            TYPE val = a[i]; \
            if (val < min_val[i]) val = min_val[i]; \
            if (val > max_val[i]) val = max_val[i]; \
            res[i] = val; \
        } \
        return; \
    } \
    int coord[8] = {0}; \
    int offsetA = 0, offsetMin = 0, offsetMax = 0, offsetRes = 0; \
    for (int el = 0; el < total_elements; el++) { \
        TYPE val = a[offsetA]; \
        if (val < min_val[offsetMin]) val = min_val[offsetMin]; \
        if (val > max_val[offsetMax]) val = max_val[offsetMax]; \
        res[offsetRes] = val; \
        for (int d = rank - 1; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) { \
                offsetA   += stridesA[d]; \
                offsetMin += stridesMin[d]; \
                offsetMax += stridesMax[d]; \
                offsetRes += stridesRes[d]; \
                break; \
            } \
            coord[d] = 0; \
            offsetA   -= (shape[d] - 1) * stridesA[d]; \
            offsetMin -= (shape[d] - 1) * stridesMin[d]; \
            offsetMax -= (shape[d] - 1) * stridesMax[d]; \
            offsetRes -= (shape[d] - 1) * stridesRes[d]; \
        } \
    } \
}

IMPLEMENT_S_CLIP(double, double)
IMPLEMENT_S_CLIP(float, float)
IMPLEMENT_S_CLIP(int64, int64_t)
IMPLEMENT_S_CLIP(int32, int32_t)
IMPLEMENT_S_CLIP(uint8, uint8_t)
IMPLEMENT_S_CLIP(int16, int16_t)

// ============================================================================
// 15. CALCULUS SOLVERS: TRAPEZOIDAL INTEGRATION & N-DIMENSIONAL GRADIENT
// ============================================================================

void s_trapz_double(const double *y, const int *stridesY,
                    const double *x, int strideX, double dx,
                    double *res, const int *stridesRes,
                    const int *shape, int rank, int axis) {
    if (y == NULL || res == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    
    for (int o = 0; o < outer_size; o++) {
        int offsetRes = 0;
        int offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesY[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesRes[targetD];
                }
            }
        }
        
        double sum = 0.0;
        for (int i = 0; i < shape[axis] - 1; i++) {
            int idxSrc = offsetSrc + i * stridesY[axis];
            int idxSrcNext = offsetSrc + (i + 1) * stridesY[axis];
            double y_curr = y[idxSrc];
            double y_next = y[idxSrcNext];
            
            double h = dx;
            if (x != NULL) {
                h = x[(i + 1) * strideX] - x[i * strideX];
            }
            sum += 0.5 * (y_curr + y_next) * h;
        }
        
        res[offsetRes] = sum;
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_trapz_float(const float *y, const int *stridesY,
                   const float *x, int strideX, float dx,
                   float *res, const int *stridesRes,
                   const int *shape, int rank, int axis) {
    if (y == NULL || res == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    
    for (int o = 0; o < outer_size; o++) {
        int offsetRes = 0;
        int offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesY[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesRes[targetD];
                }
            }
        }
        
        float sum = 0.0f;
        for (int i = 0; i < shape[axis] - 1; i++) {
            int idxSrc = offsetSrc + i * stridesY[axis];
            int idxSrcNext = offsetSrc + (i + 1) * stridesY[axis];
            float y_curr = y[idxSrc];
            float y_next = y[idxSrcNext];
            
            float h = dx;
            if (x != NULL) {
                h = x[(i + 1) * strideX] - x[i * strideX];
            }
            sum += 0.5f * (y_curr + y_next) * h;
        }
        
        res[offsetRes] = sum;
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_trapz_complex128(const cpx_t *y, const int *stridesY,
                        const double *x, int strideX, double dx,
                        cpx_t *res, const int *stridesRes,
                        const int *shape, int rank, int axis) {
    if (y == NULL || res == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    
    for (int o = 0; o < outer_size; o++) {
        int offsetRes = 0;
        int offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesY[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesRes[targetD];
                }
            }
        }
        
        cpx_t sum = {0.0, 0.0};
        for (int i = 0; i < shape[axis] - 1; i++) {
            int idxSrc = offsetSrc + i * stridesY[axis];
            int idxSrcNext = offsetSrc + (i + 1) * stridesY[axis];
            cpx_t y_curr = y[idxSrc];
            cpx_t y_next = y[idxSrcNext];
            
            double h = dx;
            if (x != NULL) {
                h = x[(i + 1) * strideX] - x[i * strideX];
            }
            sum.r += 0.5 * (y_curr.r + y_next.r) * h;
            sum.i += 0.5 * (y_curr.i + y_next.i) * h;
        }
        
        res[offsetRes] = sum;
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_trapz_complex64(const cpx_f_t *y, const int *stridesY,
                   const float *x, int strideX, float dx,
                   cpx_f_t *res, const int *stridesRes,
                   const int *shape, int rank, int axis) {
    if (y == NULL || res == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }

    for (int o = 0; o < outer_size; o++) {
        int offsetRes = 0;
        int offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesY[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesRes[targetD];
                }
            }
        }

        cpx_f_t sum = {0.0f, 0.0f};
        for (int i = 0; i < shape[axis] - 1; i++) {
            int idxSrc = offsetSrc + i * stridesY[axis];
            int idxSrcNext = offsetSrc + (i + 1) * stridesY[axis];
            cpx_f_t y_curr = y[idxSrc];
            cpx_f_t y_next = y[idxSrcNext];

            float h = dx;
            if (x != NULL) {
                h = x[(i + 1) * strideX] - x[i * strideX];
            }
            sum.r += 0.5f * (y_curr.r + y_next.r) * h;
            sum.i += 0.5f * (y_curr.i + y_next.i) * h;
        }

        res[offsetRes] = sum;

        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_trapz_complex128_all(const cpx_t *y, const int *stridesY,
                            const cpx_t *x, int strideX, cpx_t dx,
                            cpx_t *res, const int *stridesRes,
                            const int *shape, int rank, int axis) {
    if (y == NULL || res == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }

    for (int o = 0; o < outer_size; o++) {
        int offsetRes = 0;
        int offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesY[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesRes[targetD];
                }
            }
        }

        cpx_t sum = {0.0, 0.0};
        for (int i = 0; i < shape[axis] - 1; i++) {
            int idxSrc = offsetSrc + i * stridesY[axis];
            int idxSrcNext = offsetSrc + (i + 1) * stridesY[axis];
            cpx_t y_curr = y[idxSrc];
            cpx_t y_next = y[idxSrcNext];

            cpx_t h = dx;
            if (x != NULL) {
                h.r = x[(i + 1) * strideX].r - x[i * strideX].r;
                h.i = x[(i + 1) * strideX].i - x[i * strideX].i;
            }

            double yr = (y_curr.r + y_next.r) * 0.5;
            double yi = (y_curr.i + y_next.i) * 0.5;

            // Complex multiplication: (yr + i*yi) * (h.r + i*h.i)
            sum.r += yr * h.r - yi * h.i;
            sum.i += yr * h.i + yi * h.r;
        }

        res[offsetRes] = sum;

        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}
void s_gradient_double(const double *src, const int *stridesSrc,
                       const double *x, int strideX, double dx,
                       double *res, const int *stridesRes,
                       const int *shape, int rank, int axis, int edge_order) {
    if (src == NULL || res == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    
    int N = shape[axis];
    
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    
    for (int o = 0; o < outer_size; o++) {
        int offsetSrc = 0;
        int offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetSrc += coord[d] * stridesSrc[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        
        if (N == 1) {
            res[offsetRes] = 0.0;
        } else if (N == 2) {
            double h = dx;
            if (x != NULL) {
                h = x[strideX] - x[0];
            }
            double diff = (src[offsetSrc + stridesSrc[axis]] - src[offsetSrc]) / h;
            res[offsetRes] = diff;
            res[offsetRes + stridesRes[axis]] = diff;
        } else {
            // Left boundary (i = 0)
            if (edge_order == 1) {
                double h = dx;
                if (x != NULL) {
                    h = x[strideX] - x[0];
                }
                double diff = (src[offsetSrc + stridesSrc[axis]] - src[offsetSrc]) / h;
                res[offsetRes] = diff;
            } else {
                double h0 = dx;
                double h1 = dx;
                if (x != NULL) {
                    h0 = x[strideX] - x[0];
                    h1 = x[2 * strideX] - x[strideX];
                }
                double f0 = src[offsetSrc];
                double f1 = src[offsetSrc + stridesSrc[axis]];
                double f2 = src[offsetSrc + 2 * stridesSrc[axis]];
                
                double a = -(2.0 * h0 + h1) / (h0 * (h0 + h1));
                double b = (h0 + h1) / (h0 * h1);
                double c = -h0 / (h1 * (h0 + h1));
                
                res[offsetRes] = a * f0 + b * f1 + c * f2;
            }
            
            // Interior points
            for (int i = 1; i < N - 1; i++) {
                int idxSrcCurr = offsetSrc + i * stridesSrc[axis];
                int idxSrcPrev = offsetSrc + (i - 1) * stridesSrc[axis];
                int idxSrcNext = offsetSrc + (i + 1) * stridesSrc[axis];
                int idxRes = offsetRes + i * stridesRes[axis];
                
                double f_curr = src[idxSrcCurr];
                double f_prev = src[idxSrcPrev];
                double f_next = src[idxSrcNext];
                
                double h_s = dx;
                double h_d = dx;
                if (x != NULL) {
                    h_s = x[i * strideX] - x[(i - 1) * strideX];
                    h_d = x[(i + 1) * strideX] - x[i * strideX];
                }
                
                res[idxRes] = (h_s * h_s * f_next + (h_d * h_d - h_s * h_s) * f_curr - h_d * h_d * f_prev) / (h_s * h_d * (h_s + h_d));
            }
            
            // Right boundary
            int idxResEnd = offsetRes + (N - 1) * stridesRes[axis];
            int idxSrcEnd = offsetSrc + (N - 1) * stridesSrc[axis];
            if (edge_order == 1) {
                double h = dx;
                if (x != NULL) {
                    h = x[(N - 1) * strideX] - x[(N - 2) * strideX];
                }
                res[idxResEnd] = (src[idxSrcEnd] - src[idxSrcEnd - stridesSrc[axis]]) / h;
            } else {
                double h0 = dx;
                double h1 = dx;
                if (x != NULL) {
                    h0 = x[(N - 2) * strideX] - x[(N - 3) * strideX];
                    h1 = x[(N - 1) * strideX] - x[(N - 2) * strideX];
                }
                double f0 = src[idxSrcEnd - 2 * stridesSrc[axis]];
                double f1 = src[idxSrcEnd - stridesSrc[axis]];
                double f2 = src[idxSrcEnd];
                
                double a = h1 / (h0 * (h0 + h1));
                double b = -(h0 + h1) / (h0 * h1);
                double c = (2.0 * h1 + h0) / (h1 * (h0 + h1));
                
                res[idxResEnd] = a * f0 + b * f1 + c * f2;
            }
        }
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_gradient_float(const float *src, const int *stridesSrc,
                     const float *x, int strideX, float dx,
                     float *res, const int *stridesRes,
                     const int *shape, int rank, int axis, int edge_order) {
    if (src == NULL || res == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    
    int N = shape[axis];
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    
    for (int o = 0; o < outer_size; o++) {
        int offsetSrc = 0;
        int offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetSrc += coord[d] * stridesSrc[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        
        if (N == 1) {
            res[offsetRes] = 0.0f;
        } else if (N == 2) {
            float h = dx;
            if (x != NULL) {
                h = x[strideX] - x[0];
            }
            float diff = (src[offsetSrc + stridesSrc[axis]] - src[offsetSrc]) / h;
            res[offsetRes] = diff;
            res[offsetRes + stridesRes[axis]] = diff;
        } else {
            // Left boundary (i = 0)
            if (edge_order == 1) {
                float h = dx;
                if (x != NULL) {
                    h = x[strideX] - x[0];
                }
                res[offsetRes] = (src[offsetSrc + stridesSrc[axis]] - src[offsetSrc]) / h;
            } else {
                float h0 = dx;
                float h1 = dx;
                if (x != NULL) {
                    h0 = x[strideX] - x[0];
                    h1 = x[2 * strideX] - x[strideX];
                }
                float f0 = src[offsetSrc];
                float f1 = src[offsetSrc + stridesSrc[axis]];
                float f2 = src[offsetSrc + 2 * stridesSrc[axis]];
                
                float a = -(2.0f * h0 + h1) / (h0 * (h0 + h1));
                float b = (h0 + h1) / (h0 * h1);
                float c = -h0 / (h1 * (h0 + h1));
                
                res[offsetRes] = a * f0 + b * f1 + c * f2;
            }
            
            // Interior points
            for (int i = 1; i < N - 1; i++) {
                int idxSrcCurr = offsetSrc + i * stridesSrc[axis];
                int idxSrcPrev = offsetSrc + (i - 1) * stridesSrc[axis];
                int idxSrcNext = offsetSrc + (i + 1) * stridesSrc[axis];
                int idxRes = offsetRes + i * stridesRes[axis];
                
                float f_curr = src[idxSrcCurr];
                float f_prev = src[idxSrcPrev];
                float f_next = src[idxSrcNext];
                
                float h_s = dx;
                float h_d = dx;
                if (x != NULL) {
                    h_s = x[i * strideX] - x[(i - 1) * strideX];
                    h_d = x[(i + 1) * strideX] - x[i * strideX];
                }
                
                res[idxRes] = (h_s * h_s * f_next + (h_d * h_d - h_s * h_s) * f_curr - h_d * h_d * f_prev) / (h_s * h_d * (h_s + h_d));
            }
            
            // Right boundary
            int idxResEnd = offsetRes + (N - 1) * stridesRes[axis];
            int idxSrcEnd = offsetSrc + (N - 1) * stridesSrc[axis];
            if (edge_order == 1) {
                float h = dx;
                if (x != NULL) {
                    h = x[(N - 1) * strideX] - x[(N - 2) * strideX];
                }
                res[idxResEnd] = (src[idxSrcEnd] - src[idxSrcEnd - stridesSrc[axis]]) / h;
            } else {
                float h0 = dx;
                float h1 = dx;
                if (x != NULL) {
                    h0 = x[(N - 2) * strideX] - x[(N - 3) * strideX];
                    h1 = x[(N - 1) * strideX] - x[(N - 2) * strideX];
                }
                float f0 = src[idxSrcEnd - 2 * stridesSrc[axis]];
                float f1 = src[idxSrcEnd - stridesSrc[axis]];
                float f2 = src[idxSrcEnd];
                
                float a = h1 / (h0 * (h0 + h1));
                float b = -(h0 + h1) / (h0 * h1);
                float c = (2.0f * h1 + h0) / (h1 * (h0 + h1));
                
                res[idxResEnd] = a * f0 + b * f1 + c * f2;
            }
        }
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_gradient_complex128(const cpx_t *src, const int *stridesSrc,
                           const double *x, int strideX, double dx,
                           cpx_t *res, const int *stridesRes,
                           const int *shape, int rank, int axis, int edge_order) {
    if (src == NULL || res == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    
    int N = shape[axis];
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    
    for (int o = 0; o < outer_size; o++) {
        int offsetSrc = 0;
        int offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetSrc += coord[d] * stridesSrc[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        
        if (N == 1) {
            res[offsetRes] = (cpx_t){0.0, 0.0};
        } else if (N == 2) {
            double h = dx;
            if (x != NULL) {
                h = x[strideX] - x[0];
            }
            cpx_t f0 = src[offsetSrc];
            cpx_t f1 = src[offsetSrc + stridesSrc[axis]];
            cpx_t diff = (cpx_t){(f1.r - f0.r) / h, (f1.i - f0.i) / h};
            res[offsetRes] = diff;
            res[offsetRes + stridesRes[axis]] = diff;
        } else {
            // Left boundary (i = 0)
            if (edge_order == 1) {
                double h = dx;
                if (x != NULL) {
                    h = x[strideX] - x[0];
                }
                cpx_t f0 = src[offsetSrc];
                cpx_t f1 = src[offsetSrc + stridesSrc[axis]];
                res[offsetRes] = (cpx_t){(f1.r - f0.r) / h, (f1.i - f0.i) / h};
            } else {
                double h0 = dx;
                double h1 = dx;
                if (x != NULL) {
                    h0 = x[strideX] - x[0];
                    h1 = x[2 * strideX] - x[strideX];
                }
                cpx_t f0 = src[offsetSrc];
                cpx_t f1 = src[offsetSrc + stridesSrc[axis]];
                cpx_t f2 = src[offsetSrc + 2 * stridesSrc[axis]];
                
                double a = -(2.0 * h0 + h1) / (h0 * (h0 + h1));
                double b = (h0 + h1) / (h0 * h1);
                double c = -h0 / (h1 * (h0 + h1));
                
                res[offsetRes] = (cpx_t){
                    a * f0.r + b * f1.r + c * f2.r,
                    a * f0.i + b * f1.i + c * f2.i
                };
            }
            
            // Interior points
            for (int i = 1; i < N - 1; i++) {
                int idxSrcCurr = offsetSrc + i * stridesSrc[axis];
                int idxSrcPrev = offsetSrc + (i - 1) * stridesSrc[axis];
                int idxSrcNext = offsetSrc + (i + 1) * stridesSrc[axis];
                int idxRes = offsetRes + i * stridesRes[axis];
                
                cpx_t f_curr = src[idxSrcCurr];
                cpx_t f_prev = src[idxSrcPrev];
                cpx_t f_next = src[idxSrcNext];
                
                double h_s = dx;
                double h_d = dx;
                if (x != NULL) {
                    h_s = x[i * strideX] - x[(i - 1) * strideX];
                    h_d = x[(i + 1) * strideX] - x[i * strideX];
                }
                
                double denom = h_s * h_d * (h_s + h_d);
                res[idxRes] = (cpx_t){
                    (h_s * h_s * f_next.r + (h_d * h_d - h_s * h_s) * f_curr.r - h_d * h_d * f_prev.r) / denom,
                    (h_s * h_s * f_next.i + (h_d * h_d - h_s * h_s) * f_curr.i - h_d * h_d * f_prev.i) / denom
                };
            }
            
            // Right boundary
            int idxResEnd = offsetRes + (N - 1) * stridesRes[axis];
            int idxSrcEnd = offsetSrc + (N - 1) * stridesSrc[axis];
            if (edge_order == 1) {
                double h = dx;
                if (x != NULL) {
                    h = x[(N - 1) * strideX] - x[(N - 2) * strideX];
                }
                cpx_t f0 = src[idxSrcEnd - stridesSrc[axis]];
                cpx_t f1 = src[idxSrcEnd];
                res[idxResEnd] = (cpx_t){(f1.r - f0.r) / h, (f1.i - f0.i) / h};
            } else {
                double h0 = dx;
                double h1 = dx;
                if (x != NULL) {
                    h0 = x[(N - 2) * strideX] - x[(N - 3) * strideX];
                    h1 = x[(N - 1) * strideX] - x[(N - 2) * strideX];
                }
                cpx_t f0 = src[idxSrcEnd - 2 * stridesSrc[axis]];
                cpx_t f1 = src[idxSrcEnd - stridesSrc[axis]];
                cpx_t f2 = src[idxSrcEnd];
                
                double a = h1 / (h0 * (h0 + h1));
                double b = -(h0 + h1) / (h0 * h1);
                double c = (2.0 * h1 + h0) / (h1 * (h0 + h1));
                
                res[idxResEnd] = (cpx_t){
                    a * f0.r + b * f1.r + c * f2.r,
                    a * f0.i + b * f1.i + c * f2.i
                };
            }
        }
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_gradient_complex64(const cpx_f_t *src, const int *stridesSrc,
                          const float *x, int strideX, float dx,
                          cpx_f_t *res, const int *stridesRes,
                          const int *shape, int rank, int axis, int edge_order) {
    if (src == NULL || res == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    
    int N = shape[axis];
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    
    for (int o = 0; o < outer_size; o++) {
        int offsetSrc = 0;
        int offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetSrc += coord[d] * stridesSrc[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        
        if (N == 1) {
            res[offsetRes] = (cpx_f_t){0.0f, 0.0f};
        } else if (N == 2) {
            float h = dx;
            if (x != NULL) {
                h = x[strideX] - x[0];
            }
            cpx_f_t f0 = src[offsetSrc];
            cpx_f_t f1 = src[offsetSrc + stridesSrc[axis]];
            cpx_f_t diff = (cpx_f_t){(f1.r - f0.r) / h, (f1.i - f0.i) / h};
            res[offsetRes] = diff;
            res[offsetRes + stridesRes[axis]] = diff;
        } else {
            // Left boundary (i = 0)
            if (edge_order == 1) {
                float h = dx;
                if (x != NULL) {
                    h = x[strideX] - x[0];
                }
                cpx_f_t f0 = src[offsetSrc];
                cpx_f_t f1 = src[offsetSrc + stridesSrc[axis]];
                res[offsetRes] = (cpx_f_t){(f1.r - f0.r) / h, (f1.i - f0.i) / h};
            } else {
                float h0 = dx;
                float h1 = dx;
                if (x != NULL) {
                    h0 = x[strideX] - x[0];
                    h1 = x[2 * strideX] - x[strideX];
                }
                cpx_f_t f0 = src[offsetSrc];
                cpx_f_t f1 = src[offsetSrc + stridesSrc[axis]];
                cpx_f_t f2 = src[offsetSrc + 2 * stridesSrc[axis]];
                
                float a = -(2.0f * h0 + h1) / (h0 * (h0 + h1));
                float b = (h0 + h1) / (h0 * h1);
                float c = -h0 / (h1 * (h0 + h1));
                
                res[offsetRes] = (cpx_f_t){
                    a * f0.r + b * f1.r + c * f2.r,
                    a * f0.i + b * f1.i + c * f2.i
                };
            }
            
            // Interior points
            for (int i = 1; i < N - 1; i++) {
                int idxSrcCurr = offsetSrc + i * stridesSrc[axis];
                int idxSrcPrev = offsetSrc + (i - 1) * stridesSrc[axis];
                int idxSrcNext = offsetSrc + (i + 1) * stridesSrc[axis];
                int idxRes = offsetRes + i * stridesRes[axis];
                
                cpx_f_t f_curr = src[idxSrcCurr];
                cpx_f_t f_prev = src[idxSrcPrev];
                cpx_f_t f_next = src[idxSrcNext];
                
                float h_s = dx;
                float h_d = dx;
                if (x != NULL) {
                    h_s = x[i * strideX] - x[(i - 1) * strideX];
                    h_d = x[(i + 1) * strideX] - x[i * strideX];
                }
                
                float denom = h_s * h_d * (h_s + h_d);
                res[idxRes] = (cpx_f_t){
                    (h_s * h_s * f_next.r + (h_d * h_d - h_s * h_s) * f_curr.r - h_d * h_d * f_prev.r) / denom,
                    (h_s * h_s * f_next.i + (h_d * h_d - h_s * h_s) * f_curr.i - h_d * h_d * f_prev.i) / denom
                };
            }
            
            // Right boundary
            int idxResEnd = offsetRes + (N - 1) * stridesRes[axis];
            int idxSrcEnd = offsetSrc + (N - 1) * stridesSrc[axis];
            if (edge_order == 1) {
                float h = dx;
                if (x != NULL) {
                    h = x[(N - 1) * strideX] - x[(N - 2) * strideX];
                }
                cpx_f_t f0 = src[idxSrcEnd - stridesSrc[axis]];
                cpx_f_t f1 = src[idxSrcEnd];
                res[idxResEnd] = (cpx_f_t){(f1.r - f0.r) / h, (f1.i - f0.i) / h};
            } else {
                float h0 = dx;
                float h1 = dx;
                if (x != NULL) {
                    h0 = x[(N - 2) * strideX] - x[(N - 3) * strideX];
                    h1 = x[(N - 1) * strideX] - x[(N - 2) * strideX];
                }
                cpx_f_t f0 = src[idxSrcEnd - 2 * stridesSrc[axis]];
                cpx_f_t f1 = src[idxSrcEnd - stridesSrc[axis]];
                cpx_f_t f2 = src[idxSrcEnd];
                
                float a = h1 / (h0 * (h0 + h1));
                float b = -(h0 + h1) / (h0 * h1);
                float c = (2.0f * h1 + h0) / (h1 * (h0 + h1));
                
                res[idxResEnd] = (cpx_f_t){
                    a * f0.r + b * f1.r + c * f2.r,
                    a * f0.i + b * f1.i + c * f2.i
                };
            }
        }
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}



static cpx_t c_mul(cpx_t a, cpx_t b) {
    cpx_t res;
    res.r = a.r * b.r - a.i * b.i;
    res.i = a.r * b.i + a.i * b.r;
    return res;
}

static cpx_t c_div(cpx_t n, cpx_t d) {
    cpx_t res;
    double denom = d.r * d.r + d.i * d.i;
    res.r = (n.r * d.r + n.i * d.i) / denom;
    res.i = (n.i * d.r - n.r * d.i) / denom;
    return res;
}

void s_gradient_complex128_all(const cpx_t *src, const int *stridesSrc,
                               const cpx_t *x, int strideX, cpx_t dx,
                               cpx_t *res, const int *stridesRes,
                               const int *shape, int rank, int axis, int edge_order) {
    if (src == NULL || res == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    
    int N = shape[axis];
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    
    for (int o = 0; o < outer_size; o++) {
        int offsetSrc = 0;
        int offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetSrc += coord[d] * stridesSrc[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        
        if (N == 1) {
            res[offsetRes].r = 0.0;
            res[offsetRes].i = 0.0;
        } else if (N == 2) {
            cpx_t h = dx;
            if (x != NULL) {
                h.r = x[strideX].r - x[0].r;
                h.i = x[strideX].i - x[0].i;
            }
            cpx_t diff;
            diff.r = src[offsetSrc + stridesSrc[axis]].r - src[offsetSrc].r;
            diff.i = src[offsetSrc + stridesSrc[axis]].i - src[offsetSrc].i;
            res[offsetRes] = c_div(diff, h);
            res[offsetRes + stridesRes[axis]] = res[offsetRes];
        } else {
            // Left boundary (i = 0)
            if (edge_order == 1) {
                cpx_t h = dx;
                if (x != NULL) {
                    h.r = x[strideX].r - x[0].r;
                    h.i = x[strideX].i - x[0].i;
                }
                cpx_t num = {src[offsetSrc + stridesSrc[axis]].r - src[offsetSrc].r,
                             src[offsetSrc + stridesSrc[axis]].i - src[offsetSrc].i};
                res[offsetRes] = c_div(num, h);
            } else {
                cpx_t h0 = dx;
                cpx_t h1 = dx;
                if (x != NULL) {
                    h0.r = x[strideX].r - x[0].r;
                    h0.i = x[strideX].i - x[0].i;
                    h1.r = x[2 * strideX].r - x[strideX].r;
                    h1.i = x[2 * strideX].i - x[strideX].i;
                }
                cpx_t f0 = src[offsetSrc];
                cpx_t f1 = src[offsetSrc + stridesSrc[axis]];
                cpx_t f2 = src[offsetSrc + 2 * stridesSrc[axis]];
                
                cpx_t h0h1 = {h0.r + h1.r, h0.i + h1.i};
                cpx_t denom_a = c_mul(h0, h0h1);
                cpx_t num_a = {-(2.0 * h0.r + h1.r), -(2.0 * h0.i + h1.i)};
                cpx_t a = c_div(num_a, denom_a);
                
                cpx_t denom_b = c_mul(h0, h1);
                cpx_t b = c_div(h0h1, denom_b);
                
                cpx_t denom_c = c_mul(h1, h0h1);
                cpx_t num_c = {-h0.r, -h0.i};
                cpx_t c = c_div(num_c, denom_c);
                
                cpx_t term1 = c_mul(a, f0);
                cpx_t term2 = c_mul(b, f1);
                cpx_t term3 = c_mul(c, f2);
                
                res[offsetRes].r = term1.r + term2.r + term3.r;
                res[offsetRes].i = term1.i + term2.i + term3.i;
            }
            
            // Interior points
            for (int i = 1; i < N - 1; i++) {
                int idxSrcCurr = offsetSrc + i * stridesSrc[axis];
                int idxSrcPrev = offsetSrc + (i - 1) * stridesSrc[axis];
                int idxSrcNext = offsetSrc + (i + 1) * stridesSrc[axis];
                int idxRes = offsetRes + i * stridesRes[axis];
                
                cpx_t f_curr = src[idxSrcCurr];
                cpx_t f_prev = src[idxSrcPrev];
                cpx_t f_next = src[idxSrcNext];
                
                cpx_t h_s = dx;
                cpx_t h_d = dx;
                if (x != NULL) {
                    h_s.r = x[i * strideX].r - x[(i - 1) * strideX].r;
                    h_s.i = x[i * strideX].i - x[(i - 1) * strideX].i;
                    h_d.r = x[(i + 1) * strideX].r - x[i * strideX].r;
                    h_d.i = x[(i + 1) * strideX].i - x[i * strideX].i;
                }
                
                cpx_t hs2 = c_mul(h_s, h_s);
                cpx_t hd2 = c_mul(h_d, h_d);
                cpx_t hd2_hs2 = {hd2.r - hs2.r, hd2.i - hs2.i};
                
                cpx_t term1 = c_mul(hs2, f_next);
                cpx_t term2 = c_mul(hd2_hs2, f_curr);
                cpx_t term3 = c_mul(hd2, f_prev);
                
                cpx_t num = {term1.r + term2.r - term3.r, term1.i + term2.i - term3.i};
                
                cpx_t hshd = c_mul(h_s, h_d);
                cpx_t hshdsum = {h_s.r + h_d.r, h_s.i + h_d.i};
                cpx_t denom = c_mul(hshd, hshdsum);
                
                res[idxRes] = c_div(num, denom);
            }
            
            // Right boundary
            int idxResEnd = offsetRes + (N - 1) * stridesRes[axis];
            int idxSrcEnd = offsetSrc + (N - 1) * stridesSrc[axis];
            if (edge_order == 1) {
                cpx_t h = dx;
                if (x != NULL) {
                    h.r = x[(N - 1) * strideX].r - x[(N - 2) * strideX].r;
                    h.i = x[(N - 1) * strideX].i - x[(N - 2) * strideX].i;
                }
                cpx_t num = {src[idxSrcEnd].r - src[idxSrcEnd - stridesSrc[axis]].r,
                             src[idxSrcEnd].i - src[idxSrcEnd - stridesSrc[axis]].i};
                res[idxResEnd] = c_div(num, h);
            } else {
                cpx_t h0 = dx;
                cpx_t h1 = dx;
                if (x != NULL) {
                    h0.r = x[(N - 2) * strideX].r - x[(N - 3) * strideX].r;
                    h0.i = x[(N - 2) * strideX].i - x[(N - 3) * strideX].i;
                    h1.r = x[(N - 1) * strideX].r - x[(N - 2) * strideX].r;
                    h1.i = x[(N - 1) * strideX].i - x[(N - 2) * strideX].i;
                }
                cpx_t f0 = src[idxSrcEnd - 2 * stridesSrc[axis]];
                cpx_t f1 = src[idxSrcEnd - stridesSrc[axis]];
                cpx_t f2 = src[idxSrcEnd];
                
                cpx_t h0h1 = {h0.r + h1.r, h0.i + h1.i};
                cpx_t denom_a = c_mul(h0, h0h1);
                cpx_t a = c_div(h1, denom_a);
                cpx_t denom_b = c_mul(h0, h1);
                cpx_t num_b = {-h0h1.r, -h0h1.i};
                cpx_t b = c_div(num_b, denom_b);
                cpx_t denom_c = c_mul(h1, h0h1);
                cpx_t num_c = {2.0 * h1.r + h0.r, 2.0 * h1.i + h0.i};
                cpx_t c = c_div(num_c, denom_c);
                
                cpx_t term1 = c_mul(a, f0);
                cpx_t term2 = c_mul(b, f1);
                cpx_t term3 = c_mul(c, f2);
                
                res[idxResEnd].r = term1.r + term2.r + term3.r;
                res[idxResEnd].i = term1.i + term2.i + term3.i;
            }
        }
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

void s_trapz_complex64_all(const cpx_f_t *y, const int *stridesY,
                           const cpx_f_t *x, int strideX, cpx_f_t dx,
                           cpx_f_t *res, const int *stridesRes,
                           const int *shape, int rank, int axis) {
    if (y == NULL || res == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    
    for (int o = 0; o < outer_size; o++) {
        int offsetRes = 0;
        int offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesY[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesRes[targetD];
                }
            }
        }
        
        cpx_f_t sum = {0.0f, 0.0f};
        for (int i = 0; i < shape[axis] - 1; i++) {
            int idxSrc = offsetSrc + i * stridesY[axis];
            int idxSrcNext = offsetSrc + (i + 1) * stridesY[axis];
            cpx_f_t y_curr = y[idxSrc];
            cpx_f_t y_next = y[idxSrcNext];
            
            cpx_f_t h = dx;
            if (x != NULL) {
                h.r = x[(i + 1) * strideX].r - x[i * strideX].r;
                h.i = x[(i + 1) * strideX].i - x[i * strideX].i;
            }
            
            float yr = (y_curr.r + y_next.r) * 0.5f;
            float yi = (y_curr.i + y_next.i) * 0.5f;
            
            sum.r += yr * h.r - yi * h.i;
            sum.i += yr * h.i + yi * h.r;
        }
        
        res[offsetRes] = sum;
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

static cpx_f_t cf_mul(cpx_f_t a, cpx_f_t b) {
    cpx_f_t res;
    res.r = a.r * b.r - a.i * b.i;
    res.i = a.r * b.i + a.i * b.r;
    return res;
}

static cpx_f_t cf_div(cpx_f_t n, cpx_f_t d) {
    cpx_f_t res;
    float denom = d.r * d.r + d.i * d.i;
    res.r = (n.r * d.r + n.i * d.i) / denom;
    res.i = (n.i * d.r - n.r * d.i) / denom;
    return res;
}

void s_gradient_complex64_all(const cpx_f_t *src, const int *stridesSrc,
                              const cpx_f_t *x, int strideX, cpx_f_t dx,
                              cpx_f_t *res, const int *stridesRes,
                              const int *shape, int rank, int axis, int edge_order) {
    if (src == NULL || res == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    
    int N = shape[axis];
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    
    for (int o = 0; o < outer_size; o++) {
        int offsetSrc = 0;
        int offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetSrc += coord[d] * stridesSrc[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        
        if (N == 1) {
            res[offsetRes].r = 0.0f;
            res[offsetRes].i = 0.0f;
        } else if (N == 2) {
            cpx_f_t h = dx;
            if (x != NULL) {
                h.r = x[strideX].r - x[0].r;
                h.i = x[strideX].i - x[0].i;
            }
            cpx_f_t diff;
            diff.r = src[offsetSrc + stridesSrc[axis]].r - src[offsetSrc].r;
            diff.i = src[offsetSrc + stridesSrc[axis]].i - src[offsetSrc].i;
            res[offsetRes] = cf_div(diff, h);
            res[offsetRes + stridesRes[axis]] = res[offsetRes];
        } else {
            if (edge_order == 1) {
                cpx_f_t h = dx;
                if (x != NULL) {
                    h.r = x[strideX].r - x[0].r;
                    h.i = x[strideX].i - x[0].i;
                }
                cpx_f_t num = {src[offsetSrc + stridesSrc[axis]].r - src[offsetSrc].r,
                             src[offsetSrc + stridesSrc[axis]].i - src[offsetSrc].i};
                res[offsetRes] = cf_div(num, h);
            } else {
                cpx_f_t h0 = dx;
                cpx_f_t h1 = dx;
                if (x != NULL) {
                    h0.r = x[strideX].r - x[0].r;
                    h0.i = x[strideX].i - x[0].i;
                    h1.r = x[2 * strideX].r - x[strideX].r;
                    h1.i = x[2 * strideX].i - x[strideX].i;
                }
                cpx_f_t f0 = src[offsetSrc];
                cpx_f_t f1 = src[offsetSrc + stridesSrc[axis]];
                cpx_f_t f2 = src[offsetSrc + 2 * stridesSrc[axis]];
                
                cpx_f_t h0h1 = {h0.r + h1.r, h0.i + h1.i};
                cpx_f_t denom_a = cf_mul(h0, h0h1);
                cpx_f_t num_a = {-(2.0f * h0.r + h1.r), -(2.0f * h0.i + h1.i)};
                cpx_f_t a = cf_div(num_a, denom_a);
                cpx_f_t denom_b = cf_mul(h0, h1);
                cpx_f_t b = cf_div(h0h1, denom_b);
                cpx_f_t denom_c = cf_mul(h1, h0h1);
                cpx_f_t num_c = {-h0.r, -h0.i};
                cpx_f_t c = cf_div(num_c, denom_c);
                
                cpx_f_t term1 = cf_mul(a, f0);
                cpx_f_t term2 = cf_mul(b, f1);
                cpx_f_t term3 = cf_mul(c, f2);
                res[offsetRes].r = term1.r + term2.r + term3.r;
                res[offsetRes].i = term1.i + term2.i + term3.i;
            }
            
            for (int i = 1; i < N - 1; i++) {
                int idxSrcCurr = offsetSrc + i * stridesSrc[axis];
                int idxSrcPrev = offsetSrc + (i - 1) * stridesSrc[axis];
                int idxSrcNext = offsetSrc + (i + 1) * stridesSrc[axis];
                int idxRes = offsetRes + i * stridesRes[axis];
                
                cpx_f_t f_curr = src[idxSrcCurr];
                cpx_f_t f_prev = src[idxSrcPrev];
                cpx_f_t f_next = src[idxSrcNext];
                
                cpx_f_t h_s = dx;
                cpx_f_t h_d = dx;
                if (x != NULL) {
                    h_s.r = x[i * strideX].r - x[(i - 1) * strideX].r;
                    h_s.i = x[i * strideX].i - x[(i - 1) * strideX].i;
                    h_d.r = x[(i + 1) * strideX].r - x[i * strideX].r;
                    h_d.i = x[(i + 1) * strideX].i - x[i * strideX].i;
                }
                
                cpx_f_t hs2 = cf_mul(h_s, h_s);
                cpx_f_t hd2 = cf_mul(h_d, h_d);
                cpx_f_t hd2_hs2 = {hd2.r - hs2.r, hd2.i - hs2.i};
                cpx_f_t term1 = cf_mul(hs2, f_next);
                cpx_f_t term2 = cf_mul(hd2_hs2, f_curr);
                cpx_f_t term3 = cf_mul(hd2, f_prev);
                cpx_f_t num = {term1.r + term2.r - term3.r, term1.i + term2.i - term3.i};
                cpx_f_t hshd = cf_mul(h_s, h_d);
                cpx_f_t hshdsum = {h_s.r + h_d.r, h_s.i + h_d.i};
                cpx_f_t denom = cf_mul(hshd, hshdsum);
                res[idxRes] = cf_div(num, denom);
            }
            
            int idxResEnd = offsetRes + (N - 1) * stridesRes[axis];
            int idxSrcEnd = offsetSrc + (N - 1) * stridesSrc[axis];
            if (edge_order == 1) {
                cpx_f_t h = dx;
                if (x != NULL) {
                    h.r = x[(N - 1) * strideX].r - x[(N - 2) * strideX].r;
                    h.i = x[(N - 1) * strideX].i - x[(N - 2) * strideX].i;
                }
                cpx_f_t num = {src[idxSrcEnd].r - src[idxSrcEnd - stridesSrc[axis]].r,
                             src[idxSrcEnd].i - src[idxSrcEnd - stridesSrc[axis]].i};
                res[idxResEnd] = cf_div(num, h);
            } else {
                cpx_f_t h0 = dx;
                cpx_f_t h1 = dx;
                if (x != NULL) {
                    h0.r = x[(N - 2) * strideX].r - x[(N - 3) * strideX].r;
                    h0.i = x[(N - 2) * strideX].i - x[(N - 3) * strideX].i;
                    h1.r = x[(N - 1) * strideX].r - x[(N - 2) * strideX].r;
                    h1.i = x[(N - 1) * strideX].i - x[(N - 2) * strideX].i;
                }
                cpx_f_t f0 = src[idxSrcEnd - 2 * stridesSrc[axis]];
                cpx_f_t f1 = src[idxSrcEnd - stridesSrc[axis]];
                cpx_f_t f2 = src[idxSrcEnd];
                cpx_f_t h0h1 = {h0.r + h1.r, h0.i + h1.i};
                cpx_f_t denom_a = cf_mul(h0, h0h1);
                cpx_f_t a = cf_div(h1, denom_a);
                cpx_f_t denom_b = cf_mul(h0, h1);
                cpx_f_t num_b = {-h0h1.r, -h0h1.i};
                cpx_f_t b = cf_div(num_b, denom_b);
                cpx_f_t denom_c = cf_mul(h1, h0h1);
                cpx_f_t num_c = {2.0f * h1.r + h0.r, 2.0f * h1.i + h0.i};
                cpx_f_t c = cf_div(num_c, denom_c);
                cpx_f_t term1 = cf_mul(a, f0);
                cpx_f_t term2 = cf_mul(b, f1);
                cpx_f_t term3 = cf_mul(c, f2);
                res[idxResEnd].r = term1.r + term2.r + term3.r;
                res[idxResEnd].i = term1.i + term2.i + term3.i;
            }
        }
        
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
}

/* ============================================================================
 * SECTION 10: linspace GRID INTRINSIC KERNELS
 * ============================================================================
 */

#define DEFINE_LINSPACE_GRID(name, type, val_expr, step_expr) \
void name(const type *start, const int *stridesStart, \
          const type *stop, const int *stridesStop, \
          type *res, const int *stridesRes, \
          type *step, const int *stridesStep, \
          const int *shape, int rank, int axis, int numSamples, int endpoint) { \
    if (start == NULL || stop == NULL || res == NULL || rank <= 0 || rank > 8) return; \
    int total_elements = 1; \
    for (int i = 0; i < rank; i++) total_elements *= shape[i]; \
    double div = endpoint ? (double)(numSamples - 1) : (double)numSamples; \
    int coord[8] = {0}; \
    int offsetStart = 0, offsetStop = 0, offsetRes = 0, offsetStep = 0; \
    for (int el = 0; el < total_elements; el++) { \
        int i = coord[axis]; \
        double t = (div <= 0.0) ? 0.0 : (double)i / div; \
        val_expr; \
        if (step != NULL && coord[axis] == 0) { \
            step_expr; \
        } \
        for (int d = rank - 1; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) { \
                offsetStart += stridesStart[d]; \
                offsetStop += stridesStop[d]; \
                offsetRes += stridesRes[d]; \
                if (step != NULL) offsetStep += stridesStep[d]; \
                break; \
            } \
            coord[d] = 0; \
            offsetStart -= (shape[d] - 1) * stridesStart[d]; \
            offsetStop -= (shape[d] - 1) * stridesStop[d]; \
            offsetRes -= (shape[d] - 1) * stridesRes[d]; \
            if (step != NULL) offsetStep -= (shape[d] - 1) * stridesStep[d]; \
        } \
    } \
}

DEFINE_LINSPACE_GRID(
    s_linspace_grid_double, double,
    { res[offsetRes] = start[offsetStart] + (stop[offsetStop] - start[offsetStart]) * t; },
    { step[offsetStep] = (div <= 0.0) ? 0.0 : (stop[offsetStop] - start[offsetStart]) / div; }
)

DEFINE_LINSPACE_GRID(
    s_linspace_grid_float, float,
    { res[offsetRes] = start[offsetStart] + (stop[offsetStop] - start[offsetStart]) * (float)t; },
    { step[offsetStep] = (div <= 0.0) ? 0.0f : (stop[offsetStop] - start[offsetStart]) / (float)div; }
)

DEFINE_LINSPACE_GRID(
    s_linspace_grid_complex128, cpx_t,
    {
        res[offsetRes].r = start[offsetStart].r + (stop[offsetStop].r - start[offsetStart].r) * t;
        res[offsetRes].i = start[offsetStart].i + (stop[offsetStop].i - start[offsetStart].i) * t;
    },
    {
        step[offsetStep].r = (div <= 0.0) ? 0.0 : (stop[offsetStop].r - start[offsetStart].r) / div;
        step[offsetStep].i = (div <= 0.0) ? 0.0 : (stop[offsetStop].i - start[offsetStart].i) / div;
    }
)

DEFINE_LINSPACE_GRID(
    s_linspace_grid_complex64, cpx_f_t,
    {
        res[offsetRes].r = start[offsetStart].r + (stop[offsetStop].r - start[offsetStart].r) * (float)t;
        res[offsetRes].i = start[offsetStart].i + (stop[offsetStop].i - start[offsetStart].i) * (float)t;
    },
    {
        step[offsetStep].r = (div <= 0.0) ? 0.0f : (stop[offsetStop].r - start[offsetStart].r) / (float)div;
        step[offsetStep].i = (div <= 0.0) ? 0.0f : (stop[offsetStop].i - start[offsetStart].i) / (float)div;
    }
)

DEFINE_LINSPACE_GRID(
    s_linspace_grid_int64, int64_t,
    { res[offsetRes] = (int64_t)((double)start[offsetStart] + ((double)stop[offsetStop] - (double)start[offsetStart]) * t); },
    { step[offsetStep] = (div <= 0.0) ? 0 : (int64_t)(((double)stop[offsetStop] - (double)start[offsetStart]) / div); }
)

DEFINE_LINSPACE_GRID(
    s_linspace_grid_int32, int32_t,
    { res[offsetRes] = (int32_t)((double)start[offsetStart] + ((double)stop[offsetStop] - (double)start[offsetStart]) * t); },
    { step[offsetStep] = (div <= 0.0) ? 0 : (int32_t)(((double)stop[offsetStop] - (double)start[offsetStart]) / div); }
)

DEFINE_LINSPACE_GRID(
    s_linspace_grid_int16, int16_t,
    { res[offsetRes] = (int16_t)((double)start[offsetStart] + ((double)stop[offsetStop] - (double)start[offsetStart]) * t); },
    { step[offsetStep] = (div <= 0.0) ? 0 : (int16_t)(((double)stop[offsetStop] - (double)start[offsetStart]) / div); }
)

DEFINE_LINSPACE_GRID(
    s_linspace_grid_uint8, uint8_t,
    { res[offsetRes] = (uint8_t)((double)start[offsetStart] + ((double)stop[offsetStop] - (double)start[offsetStart]) * t); },
    { step[offsetStep] = (div <= 0.0) ? 0 : (uint8_t)(((double)stop[offsetStop] - (double)start[offsetStart]) / div); }
)

/* ============================================================================
 * SECTION 12: DYNAMICALLY DISPATCHED MATRIX DETERMINANT INTRA-OPERATIONS
 * ============================================================================
 * Strided determinants offloading LU-factorization call dynamically to a
 * LAPACK function pointer passed from Dart FFI space. This makes the routine
 * 100% decoupled from compile-time OpenBLAS/LAPACK dependencies.
 */

#define DEFINE_DET_REAL(name, type) \
void name(const type *a, const int *stridesA, \
          type *res, const int *stridesRes, \
          const int *shape, int rank, \
          type *aCopy, int *ipiv, \
          int (*lapack_getrf)(int, int, int, void *, int, int *)) { \
    if (a == NULL || res == NULL || aCopy == NULL || ipiv == NULL || lapack_getrf == NULL || rank < 2 || rank > 8) return; \
    int n = shape[rank - 1]; \
    int stack_elements = 1; \
    for (int i = 0; i < rank - 2; i++) stack_elements *= shape[i]; \
    int coord[8] = {0}; \
    int offsetA = 0, offsetRes = 0; \
    for (int el = 0; el < stack_elements; el++) { \
        for (int i = 0; i < n; i++) { \
            for (int j = 0; j < n; j++) { \
                aCopy[i * n + j] = a[offsetA + i * stridesA[rank - 2] + j * stridesA[rank - 1]]; \
            } \
        } \
        int info = lapack_getrf(101, n, n, aCopy, n, ipiv); \
        type detValue = 1.0; \
        if (info > 0) { \
            detValue = 0.0; \
        } else if (info < 0) { \
            detValue = NAN; \
        } else { \
            for (int i = 0; i < n; i++) { \
                detValue *= aCopy[i * n + i]; \
            } \
            int swaps = 0; \
            for (int i = 0; i < n; i++) { \
                if (ipiv[i] != i + 1) swaps++; \
            } \
            if (swaps % 2 != 0) detValue = -detValue; \
        } \
        res[offsetRes] = detValue; \
        for (int d = rank - 3; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) { \
                offsetA += stridesA[d]; \
                offsetRes += stridesRes[d]; \
                break; \
            } \
            coord[d] = 0; \
            offsetA -= (shape[d] - 1) * stridesA[d]; \
            offsetRes -= (shape[d] - 1) * stridesRes[d]; \
        } \
    } \
}

#define DEFINE_DET_COMPLEX(name, type) \
void name(const type *a, const int *stridesA, \
          type *res, const int *stridesRes, \
          const int *shape, int rank, \
          type *aCopy, int *ipiv, \
          int (*lapack_getrf)(int, int, int, void *, int, int *)) { \
    if (a == NULL || res == NULL || aCopy == NULL || ipiv == NULL || lapack_getrf == NULL || rank < 2 || rank > 8) return; \
    int n = shape[rank - 1]; \
    int stack_elements = 1; \
    for (int i = 0; i < rank - 2; i++) stack_elements *= shape[i]; \
    int coord[8] = {0}; \
    int offsetA = 0, offsetRes = 0; \
    for (int el = 0; el < stack_elements; el++) { \
        for (int i = 0; i < n; i++) { \
            for (int j = 0; j < n; j++) { \
                aCopy[i * n + j] = a[offsetA + i * stridesA[rank - 2] + j * stridesA[rank - 1]]; \
            } \
        } \
        int info = lapack_getrf(101, n, n, aCopy, n, ipiv); \
        type detValue = {1.0, 0.0}; \
        if (info > 0) { \
            detValue.r = 0.0; \
            detValue.i = 0.0; \
        } else if (info < 0) { \
            detValue.r = NAN; \
            detValue.i = NAN; \
        } else { \
            for (int i = 0; i < n; i++) { \
                double r1 = detValue.r, i1 = detValue.i; \
                double r2 = aCopy[i * n + i].r, i2 = aCopy[i * n + i].i; \
                detValue.r = r1 * r2 - i1 * i2; \
                detValue.i = r1 * i2 + i1 * r2; \
            } \
            int swaps = 0; \
            for (int i = 0; i < n; i++) { \
                if (ipiv[i] != i + 1) swaps++; \
            } \
            if (swaps % 2 != 0) { \
                detValue.r = -detValue.r; \
                detValue.i = -detValue.i; \
            } \
        } \
        res[offsetRes] = detValue; \
        for (int d = rank - 3; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) { \
                offsetA += stridesA[d]; \
                offsetRes += stridesRes[d]; \
                break; \
            } \
            coord[d] = 0; \
            offsetA -= (shape[d] - 1) * stridesA[d]; \
            offsetRes -= (shape[d] - 1) * stridesRes[d]; \
        } \
    } \
}

DEFINE_DET_REAL(s_det_double, double)
DEFINE_DET_REAL(s_det_float, float)
DEFINE_DET_COMPLEX(s_det_complex_double, cpx_t)
DEFINE_DET_COMPLEX(s_det_complex_float, cpx_f_t)

void assemble_eigenvectors_double(
    cpx_t *w,
    int strideWLast,
    cpx_t *vr,
    int strideVR1,
    int strideVR2,
    const double *wr,
    const double *wi,
    const double *vrReal,
    int n
) {
    if (w == NULL || vr == NULL || wr == NULL || wi == NULL || vrReal == NULL || n <= 0) {
        return;
    }

    for (int j = 0; j < n; j++) {
        w[j * strideWLast].r = wr[j];
        w[j * strideWLast].i = wi[j];
    }

    int j = 0;
    while (j < n) {
        if (wi[j] == 0.0) {
            for (int r = 0; r < n; r++) {
                vr[r * strideVR1 + j * strideVR2].r = vrReal[r * n + j];
                vr[r * strideVR1 + j * strideVR2].i = 0.0;
            }
            j++;
        } else {
            for (int r = 0; r < n; r++) {
                double realPart = vrReal[r * n + j];
                double imagPart = vrReal[r * n + j + 1];
                vr[r * strideVR1 + j * strideVR2].r = realPart;
                vr[r * strideVR1 + j * strideVR2].i = imagPart;
                vr[r * strideVR1 + (j + 1) * strideVR2].r = realPart;
                vr[r * strideVR1 + (j + 1) * strideVR2].i = -imagPart;
            }
            j += 2;
        }
    }
}

void assemble_eigenvectors_float(
    cpx_f_t *w,
    int strideWLast,
    cpx_f_t *vr,
    int strideVR1,
    int strideVR2,
    const float *wr,
    const float *wi,
    const float *vrReal,
    int n
) {
    if (w == NULL || vr == NULL || wr == NULL || wi == NULL || vrReal == NULL || n <= 0) {
        return;
    }

    for (int j = 0; j < n; j++) {
        w[j * strideWLast].r = wr[j];
        w[j * strideWLast].i = wi[j];
    }

    int j = 0;
    while (j < n) {
        if (wi[j] == 0.0f) {
            for (int r = 0; r < n; r++) {
                vr[r * strideVR1 + j * strideVR2].r = vrReal[r * n + j];
                vr[r * strideVR1 + j * strideVR2].i = 0.0f;
            }
            j++;
        } else {
            for (int r = 0; r < n; r++) {
                float realPart = vrReal[r * n + j];
                float imagPart = vrReal[r * n + j + 1];
                vr[r * strideVR1 + j * strideVR2].r = realPart;
                vr[r * strideVR1 + j * strideVR2].i = imagPart;
                vr[r * strideVR1 + (j + 1) * strideVR2].r = realPart;
                vr[r * strideVR1 + (j + 1) * strideVR2].i = -imagPart;
            }
            j += 2;
        }
    }
}

#define DEFINE_MATMUL_INT(name, type) \
void name( \
    type *res, \
    int strideResRow, \
    int strideResCol, \
    const type *a, \
    int strideARow, \
    int strideACol, \
    const type *b, \
    int strideBRow, \
    int strideBCol, \
    int m, \
    int n, \
    int k \
) { \
    if (res == NULL || a == NULL || b == NULL || m <= 0 || n <= 0 || k <= 0) { \
        return; \
    } \
    for (int r = 0; r < m; r++) { \
        for (int c = 0; c < n; c++) { \
            type sum = 0; \
            for (int i = 0; i < k; i++) { \
                sum += a[r * strideARow + i * strideACol] * b[i * strideBRow + c * strideBCol]; \
            } \
            res[r * strideResRow + c * strideResCol] = sum; \
        } \
    } \
}

DEFINE_MATMUL_INT(matmul_int64, int64_t)
DEFINE_MATMUL_INT(matmul_int32, int32_t)
DEFINE_MATMUL_INT(matmul_int16, int16_t)
DEFINE_MATMUL_INT(matmul_uint8, uint8_t)

uint8_t v_any_less_than_zero_int32(const int32_t *arr, int size) {
    if (arr == NULL || size <= 0) return 0;
    for (int i = 0; i < size; i++) {
        if (arr[i] < 0) return 1;
    }
    return 0;
}

uint8_t v_any_less_than_zero_int64(const int64_t *arr, int size) {
    if (arr == NULL || size <= 0) return 0;
    for (int i = 0; i < size; i++) {
        if (arr[i] < 0) return 1;
    }
    return 0;
}

uint8_t v_any_equal_to_zero_int32(const int32_t *arr, int size) {
    if (arr == NULL || size <= 0) return 0;
    for (int i = 0; i < size; i++) {
        if (arr[i] == 0) return 1;
    }
    return 0;
}

uint8_t v_any_equal_to_zero_int64(const int64_t *arr, int size) {
    if (arr == NULL || size <= 0) return 0;
    for (int i = 0; i < size; i++) {
        if (arr[i] == 0) return 1;
    }
    return 0;
}



#define TO_DOUBLE(x) ((double)(x))
#define TO_COMPLEX(x) ((cpx_t){(double)(x), 0.0})
#define COMPLEX_TO_DOUBLE(x) ((x).r)
#define COMPLEXF_TO_DOUBLE(x) ((double)(x).r)
#define COMPLEXF_TO_COMPLEX(x) ((cpx_t){(double)(x).r, (double)(x).i})
#define BOOL_TO_DOUBLE(x) ((double)(x))
#define BOOL_TO_COMPLEX(x) ((cpx_t){(double)(x), 0.0})

#define DEFINE_CAST_LOOP(src_type, dest_type, expr) \
    { \
        const src_type *src = (const src_type *)src_ptr; \
        dest_type *dest = (dest_type *)dest_ptr; \
        if (rank == 0) { \
            dest[0] = expr(src[0]); \
            return; \
        } \
        int coord[8] = {0}; \
        int offsetSrc = 0; \
        for (int el = 0; el < total_elements; el++) { \
            dest[el] = expr(src[offsetSrc]); \
            for (int d = rank - 1; d >= 0; d--) { \
                coord[d]++; \
                if (coord[d] < shape[d]) { \
                    offsetSrc += stridesSrc[d]; \
                    break; \
                } \
                coord[d] = 0; \
                offsetSrc -= (shape[d] - 1) * stridesSrc[d]; \
            } \
        } \
    }

void s_cast_generic(
    const void *src_ptr, const int *stridesSrc, int dtypeSrc,
    void *dest_ptr, int dtypeDst,
    const int *shape, int rank
) {
    if (src_ptr == NULL || dest_ptr == NULL || rank < 0 || rank > 8) return;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    if (dtypeDst == DTYPE_FLOAT64) { // dest is double
        switch (dtypeSrc) {
            case DTYPE_FLOAT64: DEFINE_CAST_LOOP(double, double, TO_DOUBLE) break;
            case DTYPE_FLOAT32: DEFINE_CAST_LOOP(float, double, TO_DOUBLE) break;
            case DTYPE_INT32: DEFINE_CAST_LOOP(int32_t, double, TO_DOUBLE) break;
            case DTYPE_INT64: DEFINE_CAST_LOOP(int64_t, double, TO_DOUBLE) break;
            case DTYPE_UINT8: DEFINE_CAST_LOOP(uint8_t, double, TO_DOUBLE) break;
            case DTYPE_INT16: DEFINE_CAST_LOOP(int16_t, double, TO_DOUBLE) break;
            case DTYPE_COMPLEX128: DEFINE_CAST_LOOP(cpx_t, double, COMPLEX_TO_DOUBLE) break;
            case DTYPE_COMPLEX64: DEFINE_CAST_LOOP(cpx_f_t, double, COMPLEXF_TO_DOUBLE) break;
            case DTYPE_BOOLEAN: DEFINE_CAST_LOOP(uint8_t, double, BOOL_TO_DOUBLE) break;
        }
    } else if (dtypeDst == DTYPE_COMPLEX128) { // dest is complex128 (cpx_t)
        switch (dtypeSrc) {
            case DTYPE_FLOAT64: DEFINE_CAST_LOOP(double, cpx_t, TO_COMPLEX) break;
            case DTYPE_FLOAT32: DEFINE_CAST_LOOP(float, cpx_t, TO_COMPLEX) break;
            case DTYPE_INT32: DEFINE_CAST_LOOP(int32_t, cpx_t, TO_COMPLEX) break;
            case DTYPE_INT64: DEFINE_CAST_LOOP(int64_t, cpx_t, TO_COMPLEX) break;
            case DTYPE_UINT8: DEFINE_CAST_LOOP(uint8_t, cpx_t, TO_COMPLEX) break;
            case DTYPE_INT16: DEFINE_CAST_LOOP(int16_t, cpx_t, TO_COMPLEX) break;
            case DTYPE_COMPLEX128: DEFINE_CAST_LOOP(cpx_t, cpx_t, (cpx_t)) break;
            case DTYPE_COMPLEX64: DEFINE_CAST_LOOP(cpx_f_t, cpx_t, COMPLEXF_TO_COMPLEX) break;
            case DTYPE_BOOLEAN: DEFINE_CAST_LOOP(uint8_t, cpx_t, BOOL_TO_COMPLEX) break;
        }
    }
}

void v_extract_upper_triangular(
    const void *src_ptr,
    void *dest_ptr,
    int k,
    int n,
    int dtype
) {
    if (src_ptr == NULL || dest_ptr == NULL || k <= 0 || n <= 0) return;
    
    switch (dtype) {
        case DTYPE_FLOAT64: {
            const double *src = (const double *)src_ptr;
            double *dest = (double *)dest_ptr;
            for (int i = 0; i < k; i++) {
                for (int j = i; j < n; j++) {
                    dest[i * n + j] = src[i * n + j];
                }
            }
            break;
        }
        case DTYPE_FLOAT32: {
            const float *src = (const float *)src_ptr;
            float *dest = (float *)dest_ptr;
            for (int i = 0; i < k; i++) {
                for (int j = i; j < n; j++) {
                    dest[i * n + j] = src[i * n + j];
                }
            }
            break;
        }
        case DTYPE_COMPLEX128: {
            const cpx_t *src = (const cpx_t *)src_ptr;
            cpx_t *dest = (cpx_t *)dest_ptr;
            for (int i = 0; i < k; i++) {
                for (int j = i; j < n; j++) {
                    dest[i * n + j] = src[i * n + j];
                }
            }
            break;
        }
        case DTYPE_COMPLEX64: {
            const cpx_f_t *src = (const cpx_f_t *)src_ptr;
            cpx_f_t *dest = (cpx_f_t *)dest_ptr;
            for (int i = 0; i < k; i++) {
                for (int j = i; j < n; j++) {
                    dest[i * n + j] = src[i * n + j];
                }
            }
            break;
        }
    }
}

void v_zero_upper_triangular(
    void *ptr,
    int n,
    int dtype
) {
    if (ptr == NULL || n <= 0) return;
    
    switch (dtype) {
        case DTYPE_FLOAT64: {
            double *data = (double *)ptr;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    data[i * n + j] = 0.0;
                }
            }
            break;
        }
        case DTYPE_FLOAT32: {
            float *data = (float *)ptr;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    data[i * n + j] = 0.0f;
                }
            }
            break;
        }
        case DTYPE_COMPLEX128: {
            cpx_t *data = (cpx_t *)ptr;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    data[i * n + j].r = 0.0;
                    data[i * n + j].i = 0.0;
                }
            }
            break;
        }
        case DTYPE_COMPLEX64: {
            cpx_f_t *data = (cpx_f_t *)ptr;
            for (int i = 0; i < n; i++) {
                for (int j = i + 1; j < n; j++) {
                    data[i * n + j].r = 0.0f;
                    data[i * n + j].i = 0.0f;
                }
            }
            break;
        }
    }
}

/* ============================================================================
 * SECTION 9: PADDING KERNELS IMPLEMENTATION
 * ============================================================================
 */

static inline int reflect_map(int i, int N) {
    if (N <= 1) return 0;
    int P = 2 * N - 2;
    int i_mod = abs(i) % P;
    return i_mod < N ? i_mod : P - i_mod;
}

static inline int symmetric_map(int i, int N) {
    if (N <= 0) return 0;
    int P = 2 * N;
    int i_mod;
    if (i < 0) {
        i_mod = (-i - 1) % P;
    } else {
        i_mod = i % P;
    }
    return i_mod < N ? i_mod : P - 1 - i_mod;
}

static inline void insertion_sort_uint8(uint8_t *arr, int n, int kind) {
    (void)kind;
    for (int i = 1; i < n; i++) {
        uint8_t key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j = j - 1;
        }
        arr[j + 1] = key;
    }
}

static inline int cmp_double(const void *a, const void *b) {
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}
static inline int cmp_float(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}
static inline int cmp_int64(const void *a, const void *b) {
    int64_t ia = *(const int64_t*)a;
    int64_t ib = *(const int64_t*)b;
    return (ia > ib) - (ia < ib);
}
static inline int cmp_int32(const void *a, const void *b) {
    int32_t ia = *(const int32_t*)a;
    int32_t ib = *(const int32_t*)b;
    return (ia > ib) - (ia < ib);
}
static inline int cmp_uint8(const void *a, const void *b) {
    uint8_t ia = *(const uint8_t*)a;
    uint8_t ib = *(const uint8_t*)b;
    return (ia > ib) - (ia < ib);
}

static inline int cmp_cpx_lex_d(cpx_t a, cpx_t b) {
    if (a.r < b.r) return -1;
    if (a.r > b.r) return 1;
    if (a.i < b.i) return -1;
    if (a.i > b.i) return 1;
    return 0;
}

static inline int cmp_cpx_lex_f(cpx_f_t a, cpx_f_t b) {
    if (a.r < b.r) return -1;
    if (a.r > b.r) return 1;
    if (a.i < b.i) return -1;
    if (a.i > b.i) return 1;
    return 0;
}

static inline double interpolate_double(double start, double end, int step, int total_steps) {
    if (total_steps <= 0) return end;
    return start + (end - start) * (double)step / total_steps;
}
static inline float interpolate_float(float start, float end, int step, int total_steps) {
    if (total_steps <= 0) return end;
    return start + (end - start) * (float)step / total_steps;
}
static inline int64_t interpolate_int64(int64_t start, int64_t end, int step, int total_steps) {
    if (total_steps <= 0) return end;
    double val = (double)start + ((double)end - (double)start) * (double)step / total_steps;
    return (int64_t)round(val);
}
static inline int32_t interpolate_int32(int32_t start, int32_t end, int step, int total_steps) {
    if (total_steps <= 0) return end;
    double val = (double)start + ((double)end - (double)start) * (double)step / total_steps;
    return (int32_t)round(val);
}
static inline uint8_t interpolate_uint8(uint8_t start, uint8_t end, int step, int total_steps) {
    if (total_steps <= 0) return end;
    double val = (double)start + ((double)end - (double)start) * (double)step / total_steps;
    return (uint8_t)round(val);
}
static inline cpx_t interpolate_complex128(cpx_t start, cpx_t end, int step, int total_steps) {
    if (total_steps <= 0) return end;
    double r = start.r + (end.r - start.r) * (double)step / total_steps;
    double i = start.i + (end.i - start.i) * (double)step / total_steps;
    return (cpx_t){r, i};
}
static inline cpx_f_t interpolate_complex64(cpx_f_t start, cpx_f_t end, int step, int total_steps) {
    if (total_steps <= 0) return end;
    float r = start.r + (end.r - start.r) * (float)step / total_steps;
    float i = start.i + (end.i - start.i) * (float)step / total_steps;
    return (cpx_f_t){r, i};
}

// Helper macros for defining numeric statistics (double, float, int64, int32, uint8)
#define DEFINE_NUMERIC_STATS(TYPE, NAME_SUFFIX, SORTER, CAST_TYPE) \
static inline TYPE stats_min_##NAME_SUFFIX(const TYPE *base, int stride, int len) { \
    TYPE m = *base; \
    for (int i = 1; i < len; i++) { \
        TYPE v = *(base + i * stride); \
        if (v < m) m = v; \
    } \
    return m; \
} \
static inline TYPE stats_max_##NAME_SUFFIX(const TYPE *base, int stride, int len) { \
    TYPE m = *base; \
    for (int i = 1; i < len; i++) { \
        TYPE v = *(base + i * stride); \
        if (v > m) m = v; \
    } \
    return m; \
} \
static inline TYPE stats_mean_##NAME_SUFFIX(const TYPE *base, int stride, int len) { \
    double sum = 0; \
    for (int i = 0; i < len; i++) { \
        sum += (double)*(base + i * stride); \
    } \
    return (TYPE)(sum / len); \
} \
static inline TYPE stats_median_##NAME_SUFFIX(const TYPE *base, int _stride, int len) { \
    /* Median calculation requires temporary buffer. We use malloc here. */ \
    /* In actual usage, len is capped by dimension size. */ \
    TYPE *buf = (TYPE*)malloc(len * sizeof(TYPE)); \
    if (buf == NULL) return (TYPE)0; \
    for (int i = 0; i < len; i++) { \
        buf[i] = *(base + i * _stride); \
    } \
    SORTER((CAST_TYPE)buf, len, 0); /* 0 = quicksort */ \
    TYPE res; \
    if (len % 2 == 1) { \
        res = buf[len / 2]; \
    } else { \
        res = (TYPE)(((double)buf[len / 2 - 1] + (double)buf[len / 2]) / 2.0); \
    } \
    free(buf); \
    return res; \
}

DEFINE_NUMERIC_STATS(double, double, native_sort_double, double*)
DEFINE_NUMERIC_STATS(float, float, native_sort_float, float*)
DEFINE_NUMERIC_STATS(int64_t, int64, native_sort_int64, long long*)
DEFINE_NUMERIC_STATS(int32_t, int32, native_sort_int32, int*)
DEFINE_NUMERIC_STATS(uint8_t, uint8, insertion_sort_uint8, uint8_t*)


// Complex statistics helpers
static inline cpx_t stats_min_complex128(const cpx_t *base, int stride, int len) {
    cpx_t m = *base;
    for (int i = 1; i < len; i++) {
        cpx_t v = *(base + i * stride);
        if (cmp_cpx_lex_d(v, m) < 0) m = v;
    }
    return m;
}
static inline cpx_t stats_max_complex128(const cpx_t *base, int stride, int len) {
    cpx_t m = *base;
    for (int i = 1; i < len; i++) {
        cpx_t v = *(base + i * stride);
        if (cmp_cpx_lex_d(v, m) > 0) m = v;
    }
    return m;
}
static inline cpx_t stats_mean_complex128(const cpx_t *base, int stride, int len) {
    double sum_r = 0;
    double sum_i = 0;
    for (int i = 0; i < len; i++) {
        cpx_t v = *(base + i * stride);
        sum_r += v.r;
        sum_i += v.i;
    }
    return (cpx_t){sum_r / len, sum_i / len};
}
static inline cpx_t stats_median_complex128(const cpx_t *base, int stride, int len) {
    double *buf_r = (double*)malloc(len * sizeof(double));
    double *buf_i = (double*)malloc(len * sizeof(double));
    if (buf_r == NULL || buf_i == NULL) {
        if (buf_r) free(buf_r);
        if (buf_i) free(buf_i);
        return (cpx_t){0, 0};
    }
    for (int i = 0; i < len; i++) {
        cpx_t v = *(base + i * stride);
        buf_r[i] = v.r;
        buf_i[i] = v.i;
    }
    native_sort_double(buf_r, len, 0);
    native_sort_double(buf_i, len, 0);
    double res_r, res_i;
    if (len % 2 == 1) {
        res_r = buf_r[len / 2];
        res_i = buf_i[len / 2];
    } else {
        res_r = (buf_r[len / 2 - 1] + buf_r[len / 2]) / 2.0;
        res_i = (buf_i[len / 2 - 1] + buf_i[len / 2]) / 2.0;
    }
    free(buf_r);
    free(buf_i);
    return (cpx_t){res_r, res_i};
}

static inline cpx_f_t stats_min_complex64(const cpx_f_t *base, int stride, int len) {
    cpx_f_t m = *base;
    for (int i = 1; i < len; i++) {
        cpx_f_t v = *(base + i * stride);
        if (cmp_cpx_lex_f(v, m) < 0) m = v;
    }
    return m;
}
static inline cpx_f_t stats_max_complex64(const cpx_f_t *base, int stride, int len) {
    cpx_f_t m = *base;
    for (int i = 1; i < len; i++) {
        cpx_f_t v = *(base + i * stride);
        if (cmp_cpx_lex_f(v, m) > 0) m = v;
    }
    return m;
}
static inline cpx_f_t stats_mean_complex64(const cpx_f_t *base, int stride, int len) {
    float sum_r = 0;
    float sum_i = 0;
    for (int i = 0; i < len; i++) {
        cpx_f_t v = *(base + i * stride);
        sum_r += v.r;
        sum_i += v.i;
    }
    return (cpx_f_t){sum_r / len, sum_i / len};
}
static inline cpx_f_t stats_median_complex64(const cpx_f_t *base, int stride, int len) {
    float *buf_r = (float*)malloc(len * sizeof(float));
    float *buf_i = (float*)malloc(len * sizeof(float));
    if (buf_r == NULL || buf_i == NULL) {
        if (buf_r) free(buf_r);
        if (buf_i) free(buf_i);
        return (cpx_f_t){0, 0};
    }
    for (int i = 0; i < len; i++) {
        cpx_f_t v = *(base + i * stride);
        buf_r[i] = v.r;
        buf_i[i] = v.i;
    }
    native_sort_float(buf_r, len, 0);
    native_sort_float(buf_i, len, 0);
    float res_r, res_i;
    if (len % 2 == 1) {
        res_r = buf_r[len / 2];
        res_i = buf_i[len / 2];
    } else {
        res_r = (buf_r[len / 2 - 1] + buf_r[len / 2]) / 2.0f;
        res_i = (buf_i[len / 2 - 1] + buf_i[len / 2]) / 2.0f;
    }
    free(buf_r);
    free(buf_i);
    return (cpx_f_t){res_r, res_i};
}

#define DEFINE_PAD_AXIS(TYPE_NAME, T, STATS_MIN, STATS_MAX, STATS_MEAN, STATS_MEDIAN, INTERPOLATE) \
void pad_axis_##TYPE_NAME( \
    const T *src, const int *shapeSrc, const int *stridesSrc, \
    T *dest, const int *shapeDest, \
    int rank, int axis, \
    int padBefore, int padAfter, \
    int mode, \
    T constantBefore, T constantAfter, \
    T endBefore, T endAfter, \
    int statLengthBefore, int statLengthAfter \
) { \
    if (src == NULL || dest == NULL || shapeSrc == NULL || shapeDest == NULL || stridesSrc == NULL || rank <= 0 || rank > 8) return; \
    int total_elements_dest = 1; \
    for (int i = 0; i < rank; i++) total_elements_dest *= shapeDest[i]; \
    int coordDest[8] = {0}; \
    int offsetSrcSlice = 0; \
    T cached_stat_before = (T)0; \
    T cached_stat_after = (T)0; \
    int stats_cached = 0; \
    int new_slice = 1; \
    for (int el = 0; el < total_elements_dest; el++) { \
        if (new_slice) { \
            stats_cached = 0; \
            new_slice = 0; \
        } \
        const T *src_vector_base = src + offsetSrcSlice; \
        int src_stride = stridesSrc[axis]; \
        int N = shapeSrc[axis]; \
        int c = coordDest[axis]; \
        T val; \
        if (c < padBefore) { \
            switch(mode) { \
                case 0: /* constant */ \
                    val = constantBefore; \
                    break; \
                case 1: /* edge */ \
                    val = *src_vector_base; \
                    break; \
                case 2: /* reflect */ { \
                    int src_idx = reflect_map(c - padBefore, N); \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 3: /* symmetric */ { \
                    int src_idx = symmetric_map(c - padBefore, N); \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 4: /* wrap */ { \
                    int src_idx = ((c - padBefore) % N + N) % N; \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 5: /* linear_ramp */ \
                    val = INTERPOLATE(endBefore, *src_vector_base, c, padBefore); \
                    break; \
                case 6: /* maximum */ \
                case 7: /* mean */ \
                case 8: /* median */ \
                case 9: /* minimum */ \
                    if (!stats_cached) { \
                        switch(mode) { \
                            case 6: \
                                cached_stat_before = STATS_MAX(src_vector_base, src_stride, statLengthBefore); \
                                cached_stat_after = STATS_MAX(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                break; \
                            case 7: \
                                cached_stat_before = STATS_MEAN(src_vector_base, src_stride, statLengthBefore); \
                                cached_stat_after = STATS_MEAN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                break; \
                            case 8: \
                                cached_stat_before = STATS_MEDIAN(src_vector_base, src_stride, statLengthBefore); \
                                cached_stat_after = STATS_MEDIAN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                break; \
                            case 9: \
                                cached_stat_before = STATS_MIN(src_vector_base, src_stride, statLengthBefore); \
                                cached_stat_after = STATS_MIN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                break; \
                        } \
                        stats_cached = 1; \
                    } \
                    val = cached_stat_before; \
                    break; \
                default: \
                    val = constantBefore; \
            } \
        } else if (c >= padBefore + N) { \
            switch(mode) { \
                case 0: /* constant */ \
                    val = constantAfter; \
                    break; \
                case 1: /* edge */ \
                    val = *(src_vector_base + (N - 1) * src_stride); \
                    break; \
                case 2: /* reflect */ { \
                    int src_idx = reflect_map(c - padBefore, N); \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 3: /* symmetric */ { \
                    int src_idx = symmetric_map(c - padBefore, N); \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 4: /* wrap */ { \
                    int src_idx = (c - padBefore) % N; \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 5: /* linear_ramp */ { \
                    T edge_val = *(src_vector_base + (N - 1) * src_stride); \
                    int diff = c - (padBefore + N - 1); \
                    val = INTERPOLATE(edge_val, endAfter, diff, padAfter); \
                    break; \
                } \
                case 6: /* maximum */ \
                case 7: /* mean */ \
                case 8: /* median */ \
                case 9: /* minimum */ \
                    if (!stats_cached) { \
                        switch(mode) { \
                            case 6: \
                                cached_stat_before = STATS_MAX(src_vector_base, src_stride, statLengthBefore); \
                                cached_stat_after = STATS_MAX(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                break; \
                            case 7: \
                                cached_stat_before = STATS_MEAN(src_vector_base, src_stride, statLengthBefore); \
                                cached_stat_after = STATS_MEAN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                break; \
                            case 8: \
                                cached_stat_before = STATS_MEDIAN(src_vector_base, src_stride, statLengthBefore); \
                                cached_stat_after = STATS_MEDIAN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                break; \
                            case 9: \
                                cached_stat_before = STATS_MIN(src_vector_base, src_stride, statLengthBefore); \
                                cached_stat_after = STATS_MIN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                break; \
                        } \
                        stats_cached = 1; \
                    } \
                    val = cached_stat_after; \
                    break; \
                default: \
                    val = constantAfter; \
            } \
        } else { \
            val = *(src_vector_base + (c - padBefore) * src_stride); \
        } \
        dest[el] = val; \
        for (int d = rank - 1; d >= 0; d--) { \
            coordDest[d]++; \
            if (coordDest[d] < shapeDest[d]) { \
                if (d != axis) { \
                    offsetSrcSlice += stridesSrc[d]; \
                    new_slice = 1; \
                } \
                break; \
            } \
            coordDest[d] = 0; \
            if (d != axis) { \
                offsetSrcSlice -= (shapeDest[d] - 1) * stridesSrc[d]; \
                new_slice = 1; \
            } \
        } \
    } \
}

// Instantiate DEFINE_PAD_AXIS for double, float, int64, int32, uint8
DEFINE_PAD_AXIS(double, double, stats_min_double, stats_max_double, stats_mean_double, stats_median_double, interpolate_double)
DEFINE_PAD_AXIS(float, float, stats_min_float, stats_max_float, stats_mean_float, stats_median_float, interpolate_float)
DEFINE_PAD_AXIS(int64, int64_t, stats_min_int64, stats_max_int64, stats_mean_int64, stats_median_int64, interpolate_int64)
DEFINE_PAD_AXIS(int32, int32_t, stats_min_int32, stats_max_int32, stats_mean_int32, stats_median_int32, interpolate_int32)
DEFINE_PAD_AXIS(uint8, uint8_t, stats_min_uint8, stats_max_uint8, stats_mean_uint8, stats_median_uint8, interpolate_uint8)

// Complex pad axis implementation (needs separate macro because cpx_t has zeroInit {0,0} and different stats/interpolation signatures)
#define DEFINE_PAD_AXIS_COMPLEX(TYPE_NAME, T, STATS_MIN, STATS_MAX, STATS_MEAN, STATS_MEDIAN, INTERPOLATE) \
void pad_axis_##TYPE_NAME( \
    const T *src, const int *shapeSrc, const int *stridesSrc, \
    T *dest, const int *shapeDest, \
    int rank, int axis, \
    int padBefore, int padAfter, \
    int mode, \
    T constantBefore, T constantAfter, \
    T endBefore, T endAfter, \
    int statLengthBefore, int statLengthAfter \
) { \
    if (src == NULL || dest == NULL || shapeSrc == NULL || shapeDest == NULL || stridesSrc == NULL || rank <= 0 || rank > 8) return; \
    int total_elements_dest = 1; \
    for (int i = 0; i < rank; i++) total_elements_dest *= shapeDest[i]; \
    int coordDest[8] = {0}; \
    int offsetSrcSlice = 0; \
    T cached_stat_before = {0, 0}; \
    T cached_stat_after = {0, 0}; \
    int stats_cached = 0; \
    int new_slice = 1; \
    for (int el = 0; el < total_elements_dest; el++) { \
        if (new_slice) { \
            stats_cached = 0; \
            new_slice = 0; \
        } \
        const T *src_vector_base = src + offsetSrcSlice; \
        int src_stride = stridesSrc[axis]; \
        int N = shapeSrc[axis]; \
        int c = coordDest[axis]; \
        T val; \
        if (c < padBefore) { \
            switch(mode) { \
                case 0: /* constant */ \
                    val = constantBefore; \
                    break; \
                case 1: /* edge */ \
                    val = *src_vector_base; \
                    break; \
                case 2: /* reflect */ { \
                    int src_idx = reflect_map(c - padBefore, N); \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 3: /* symmetric */ { \
                    int src_idx = symmetric_map(c - padBefore, N); \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 4: /* wrap */ { \
                    int src_idx = ((c - padBefore) % N + N) % N; \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 5: /* linear_ramp */ \
                    val = INTERPOLATE(endBefore, *src_vector_base, c, padBefore); \
                    break; \
                case 6: /* maximum */ \
                case 7: /* mean */ \
                case 8: /* median */ \
                case 9: /* minimum */ \
                    if (!stats_cached) { \
                        switch(mode) { \
                            case 6: \
                                cached_stat_before = STATS_MAX(src_vector_base, src_stride, statLengthBefore); \
                                cached_stat_after = STATS_MAX(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                break; \
                            case 7: \
                                cached_stat_before = STATS_MEAN(src_vector_base, src_stride, statLengthBefore); \
                                cached_stat_after = STATS_MEAN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                break; \
                            case 8: \
                                cached_stat_before = STATS_MEDIAN(src_vector_base, src_stride, statLengthBefore); \
                                cached_stat_after = STATS_MEDIAN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                break; \
                            case 9: \
                                cached_stat_before = STATS_MIN(src_vector_base, src_stride, statLengthBefore); \
                                cached_stat_after = STATS_MIN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                break; \
                        } \
                        stats_cached = 1; \
                    } \
                    val = cached_stat_before; \
                    break; \
                default: \
                    val = constantBefore; \
            } \
        } else if (c >= padBefore + N) { \
            switch(mode) { \
                case 0: /* constant */ \
                    val = constantAfter; \
                    break; \
                case 1: /* edge */ \
                    val = *(src_vector_base + (N - 1) * src_stride); \
                    break; \
                case 2: /* reflect */ { \
                    int src_idx = reflect_map(c - padBefore, N); \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 3: /* symmetric */ { \
                    int src_idx = symmetric_map(c - padBefore, N); \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 4: /* wrap */ { \
                    int src_idx = (c - padBefore) % N; \
                    val = *(src_vector_base + src_idx * src_stride); \
                    break; \
                } \
                case 5: /* linear_ramp */ { \
                    T edge_val = *(src_vector_base + (N - 1) * src_stride); \
                    int diff = c - (padBefore + N - 1); \
                    val = INTERPOLATE(edge_val, endAfter, diff, padAfter); \
                    break; \
                } \
                case 6: /* maximum */ \
                case 7: /* mean */ \
                case 8: /* median */ \
                case 9: /* minimum */ \
                    if (!stats_cached) { \
                        switch(mode) { \
                            case 6: \
                                cached_stat_before = STATS_MAX(src_vector_base, src_stride, statLengthBefore); \
                                cached_stat_after = STATS_MAX(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                break; \
                            case 7: \
                                cached_stat_before = STATS_MEAN(src_vector_base, src_stride, statLengthBefore); \
                                cached_stat_after = STATS_MEAN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                break; \
                            case 8: \
                                cached_stat_before = STATS_MEDIAN(src_vector_base, src_stride, statLengthBefore); \
                                cached_stat_after = STATS_MEDIAN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                break; \
                            case 9: \
                                cached_stat_before = STATS_MIN(src_vector_base, src_stride, statLengthBefore); \
                                cached_stat_after = STATS_MIN(src_vector_base + (N - statLengthAfter) * src_stride, src_stride, statLengthAfter); \
                                break; \
                        } \
                        stats_cached = 1; \
                    } \
                    val = cached_stat_after; \
                    break; \
                default: \
                    val = constantAfter; \
            } \
        } else { \
            val = *(src_vector_base + (c - padBefore) * src_stride); \
        } \
        dest[el] = val; \
        for (int d = rank - 1; d >= 0; d--) { \
            coordDest[d]++; \
            if (coordDest[d] < shapeDest[d]) { \
                if (d != axis) { \
                    offsetSrcSlice += stridesSrc[d]; \
                    new_slice = 1; \
                } \
                break; \
            } \
            coordDest[d] = 0; \
            if (d != axis) { \
                offsetSrcSlice -= (shapeDest[d] - 1) * stridesSrc[d]; \
                new_slice = 1; \
            } \
        } \
    } \
}

DEFINE_PAD_AXIS_COMPLEX(complex128, cpx_t, stats_min_complex128, stats_max_complex128, stats_mean_complex128, stats_median_complex128, interpolate_complex128)
DEFINE_PAD_AXIS_COMPLEX(complex64, cpx_f_t, stats_min_complex64, stats_max_complex64, stats_mean_complex64, stats_median_complex64, interpolate_complex64)

/* ============================================================================
 * SECTION 10: SET OPERATIONS KERNELS
 * ============================================================================
 */

#define DEFINE_COMPACT_SORTED(TYPE_NAME, T, EQ_OP) \
int compact_sorted_##TYPE_NAME(T *arr, int size) { \
    if (size <= 1) return size; \
    int write_idx = 1; \
    for (int read_idx = 1; read_idx < size; read_idx++) { \
        if (!(EQ_OP(arr[read_idx], arr[write_idx - 1]))) { \
            arr[write_idx] = arr[read_idx]; \
            write_idx++; \
        } \
    } \
    return write_idx; \
}

#define EQ_FLOAT64(a, b) ((a) == (b) || (isnan(a) && isnan(b)))
#define EQ_FLOAT32(a, b) ((a) == (b) || (isnan(a) && isnan(b)))
#define EQ_INT32(a, b) ((a) == (b))
#define EQ_INT64(a, b) ((a) == (b))
#define EQ_UINT8(a, b) ((a) == (b))
#define EQ_COMPLEX128(a, b) (((a).r == (b).r || (isnan((a).r) && isnan((b).r))) && ((a).i == (b).i || (isnan((a).i) && isnan((b).i))))
#define EQ_COMPLEX64(a, b) (((a).r == (b).r || (isnan((a).r) && isnan((b).r))) && ((a).i == (b).i || (isnan((a).i) && isnan((b).i))))

DEFINE_COMPACT_SORTED(float64, double, EQ_FLOAT64)
DEFINE_COMPACT_SORTED(float32, float, EQ_FLOAT32)
DEFINE_COMPACT_SORTED(int32, int32_t, EQ_INT32)
DEFINE_COMPACT_SORTED(int64, int64_t, EQ_INT64)
DEFINE_COMPACT_SORTED(uint8, uint8_t, EQ_UINT8)
DEFINE_COMPACT_SORTED(complex128, cpx_t, EQ_COMPLEX128)
DEFINE_COMPACT_SORTED(complex64, cpx_f_t, EQ_COMPLEX64)

#define DEFINE_UNIQUE_OP(TYPE_NAME, T, SORT_FN, COMPACT_FN) \
int unique_##TYPE_NAME(const T *src, T *dest, int size) { \
    if (size <= 0) return 0; \
    custom_memcpy(dest, src, size * sizeof(T)); \
    SORT_FN(dest, size, 0); \
    return COMPACT_FN(dest, size); \
}

static void sort_float64(double *arr, int size, int kind) { native_sort_double(arr, size, kind); }
static void sort_float32(float *arr, int size, int kind) { native_sort_float(arr, size, kind); }
static void sort_int32(int32_t *arr, int size, int kind) { native_sort_int32((int *)arr, size, kind); }
static void sort_int64(int64_t *arr, int size, int kind) { native_sort_int64((long long *)arr, size, kind); }
static void sort_uint8(uint8_t *arr, int size, int kind) { native_sort_uint8(arr, size, kind); }
static void sort_complex128(cpx_t *arr, int size, int kind) { native_sort_complex128((double *)arr, size, kind); }
static void sort_complex64(cpx_f_t *arr, int size, int kind) { native_sort_complex64((float *)arr, size, kind); }

DEFINE_UNIQUE_OP(float64, double, sort_float64, compact_sorted_float64)
DEFINE_UNIQUE_OP(float32, float, sort_float32, compact_sorted_float32)
DEFINE_UNIQUE_OP(int32, int32_t, sort_int32, compact_sorted_int32)
DEFINE_UNIQUE_OP(int64, int64_t, sort_int64, compact_sorted_int64)
DEFINE_UNIQUE_OP(uint8, uint8_t, sort_uint8, compact_sorted_uint8)
DEFINE_UNIQUE_OP(complex128, cpx_t, sort_complex128, compact_sorted_complex128)
DEFINE_UNIQUE_OP(complex64, cpx_f_t, sort_complex64, compact_sorted_complex64)

/* ndarray_unique moved to custom_sorting.cpp */

static inline int set_cmp_double(double a, double b) {
    int nan_a = isnan(a);
    int nan_b = isnan(b);
    if (nan_a && nan_b) return 0;
    if (nan_a) return 1;
    if (nan_b) return -1;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int set_cmp_float(float a, float b) {
    int nan_a = isnan(a);
    int nan_b = isnan(b);
    if (nan_a && nan_b) return 0;
    if (nan_a) return 1;
    if (nan_b) return -1;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int set_cmp_int32(int32_t a, int32_t b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int set_cmp_int64(int64_t a, int64_t b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int set_cmp_uint8(uint8_t a, uint8_t b) {
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

static inline int set_cmp_complex128(cpx_t a, cpx_t b) {
    int cmp_r = set_cmp_double(a.r, b.r);
    if (cmp_r != 0) return cmp_r;
    return set_cmp_double(a.i, b.i);
}

static inline int set_cmp_complex64(cpx_f_t a, cpx_f_t b) {
    int cmp_r = set_cmp_float(a.r, b.r);
    if (cmp_r != 0) return cmp_r;
    return set_cmp_float(a.i, b.i);
}

#define DEFINE_INTERSECT1D_OP(TYPE_NAME, T, CMP_FN) \
int intersect1d_##TYPE_NAME(const T *ar1, int size1, const T *ar2, int size2, T *dest) { \
    int i = 0, j = 0, k = 0; \
    while (i < size1 && j < size2) { \
        int cmp = CMP_FN(ar1[i], ar2[j]); \
        if (cmp < 0) { \
            i++; \
        } else if (cmp > 0) { \
            j++; \
        } else { \
            dest[k++] = ar1[i]; \
            i++; \
            j++; \
        } \
    } \
    return k; \
}

DEFINE_INTERSECT1D_OP(float64, double, set_cmp_double)
DEFINE_INTERSECT1D_OP(float32, float, set_cmp_float)
DEFINE_INTERSECT1D_OP(int32, int32_t, set_cmp_int32)
DEFINE_INTERSECT1D_OP(int64, int64_t, set_cmp_int64)
DEFINE_INTERSECT1D_OP(uint8, uint8_t, set_cmp_uint8)
DEFINE_INTERSECT1D_OP(complex128, cpx_t, set_cmp_complex128)
DEFINE_INTERSECT1D_OP(complex64, cpx_f_t, set_cmp_complex64)

#define DEFINE_SETDIFF1D_OP(TYPE_NAME, T, CMP_FN) \
int setdiff1d_##TYPE_NAME(const T *ar1, int size1, const T *ar2, int size2, T *dest) { \
    int i = 0, j = 0, k = 0; \
    while (i < size1) { \
        if (j >= size2) { \
            dest[k++] = ar1[i++]; \
            continue; \
        } \
        int cmp = CMP_FN(ar1[i], ar2[j]); \
        if (cmp < 0) { \
            dest[k++] = ar1[i++]; \
        } else if (cmp > 0) { \
            j++; \
        } else { \
            i++; \
        } \
    } \
    return k; \
}

DEFINE_SETDIFF1D_OP(float64, double, set_cmp_double)
DEFINE_SETDIFF1D_OP(float32, float, set_cmp_float)
DEFINE_SETDIFF1D_OP(int32, int32_t, set_cmp_int32)
DEFINE_SETDIFF1D_OP(int64, int64_t, set_cmp_int64)
DEFINE_SETDIFF1D_OP(uint8, uint8_t, set_cmp_uint8)
DEFINE_SETDIFF1D_OP(complex128, cpx_t, set_cmp_complex128)
DEFINE_SETDIFF1D_OP(complex64, cpx_f_t, set_cmp_complex64)

#define DEFINE_SETXOR1D_OP(TYPE_NAME, T, CMP_FN) \
int setxor1d_##TYPE_NAME(const T *ar1, int size1, const T *ar2, int size2, T *dest) { \
    int i = 0, j = 0, k = 0; \
    while (i < size1 || j < size2) { \
        if (i >= size1) { \
            dest[k++] = ar2[j++]; \
            continue; \
        } \
        if (j >= size2) { \
            dest[k++] = ar1[i++]; \
            continue; \
        } \
        int cmp = CMP_FN(ar1[i], ar2[j]); \
        if (cmp < 0) { \
            dest[k++] = ar1[i++]; \
        } else if (cmp > 0) { \
            dest[k++] = ar2[j++]; \
        } else { \
            i++; \
            j++; \
        } \
    } \
    return k; \
}

DEFINE_SETXOR1D_OP(float64, double, set_cmp_double)
DEFINE_SETXOR1D_OP(float32, float, set_cmp_float)
DEFINE_SETXOR1D_OP(int32, int32_t, set_cmp_int32)
DEFINE_SETXOR1D_OP(int64, int64_t, set_cmp_int64)
DEFINE_SETXOR1D_OP(uint8, uint8_t, set_cmp_uint8)
DEFINE_SETXOR1D_OP(complex128, cpx_t, set_cmp_complex128)
DEFINE_SETXOR1D_OP(complex64, cpx_f_t, set_cmp_complex64)

#define DEFINE_UNION1D_OP(TYPE_NAME, T, CMP_FN) \
int union1d_##TYPE_NAME(const T *ar1, int size1, const T *ar2, int size2, T *dest) { \
    int i = 0, j = 0, k = 0; \
    while (i < size1 || j < size2) { \
        if (i >= size1) { \
            dest[k++] = ar2[j++]; \
            continue; \
        } \
        if (j >= size2) { \
            dest[k++] = ar1[i++]; \
            continue; \
        } \
        int cmp = CMP_FN(ar1[i], ar2[j]); \
        if (cmp < 0) { \
            dest[k++] = ar1[i++]; \
        } else if (cmp > 0) { \
            dest[k++] = ar2[j++]; \
        } else { \
            dest[k++] = ar1[i]; \
            i++; \
            j++; \
        } \
    } \
    return k; \
}

DEFINE_UNION1D_OP(float64, double, set_cmp_double)
DEFINE_UNION1D_OP(float32, float, set_cmp_float)
DEFINE_UNION1D_OP(int32, int32_t, set_cmp_int32)
DEFINE_UNION1D_OP(int64, int64_t, set_cmp_int64)
DEFINE_UNION1D_OP(uint8, uint8_t, set_cmp_uint8)
DEFINE_UNION1D_OP(complex128, cpx_t, set_cmp_complex128)
DEFINE_UNION1D_OP(complex64, cpx_f_t, set_cmp_complex64)

#define DISPATCH_SET_OP(OP_NAME) \
int ndarray_##OP_NAME(const void *ar1, int size1, const void *ar2, int size2, void *dest, int dtype) { \
    if (ar1 == NULL || ar2 == NULL || dest == NULL) return 0; \
    switch (dtype) { \
        case DTYPE_FLOAT64: return OP_NAME##_float64((const double *)ar1, size1, (const double *)ar2, size2, (double *)dest); \
        case DTYPE_FLOAT32: return OP_NAME##_float32((const float *)ar1, size1, (const float *)ar2, size2, (float *)dest); \
        case DTYPE_INT32: return OP_NAME##_int32((const int32_t *)ar1, size1, (const int32_t *)ar2, size2, (int32_t *)dest); \
        case DTYPE_INT64: return OP_NAME##_int64((const int64_t *)ar1, size1, (const int64_t *)ar2, size2, (int64_t *)dest); \
        case DTYPE_UINT8: \
        case DTYPE_BOOLEAN: return OP_NAME##_uint8((const uint8_t *)ar1, size1, (const uint8_t *)ar2, size2, (uint8_t *)dest); \
        case DTYPE_COMPLEX128: return OP_NAME##_complex128((const cpx_t *)ar1, size1, (const cpx_t *)ar2, size2, (cpx_t *)dest); \
        case DTYPE_COMPLEX64: return OP_NAME##_complex64((const cpx_f_t *)ar1, size1, (const cpx_f_t *)ar2, size2, (cpx_f_t *)dest); \
        default: return 0; \
    } \
}

DISPATCH_SET_OP(intersect1d)
DISPATCH_SET_OP(setdiff1d)
DISPATCH_SET_OP(setxor1d)
DISPATCH_SET_OP(union1d)

#define DEFINE_ISIN_OP(TYPE_NAME, T, CMP_FN) \
void isin_##TYPE_NAME(const T *ar1, int size1, const T *ar2, int size2, uint8_t *dest, int invert) { \
    for (int i = 0; i < size1; i++) { \
        T val = ar1[i]; \
        int found = 0; \
        int low = 0; \
        int high = size2 - 1; \
        while (low <= high) { \
            int mid = low + (high - low) / 2; \
            int cmp = CMP_FN(ar2[mid], val); \
            if (cmp < 0) { \
                low = mid + 1; \
            } else if (cmp > 0) { \
                high = mid - 1; \
            } else { \
                found = 1; \
                break; \
            } \
        } \
        dest[i] = invert ? !found : found; \
    } \
}

DEFINE_ISIN_OP(float64, double, set_cmp_double)
DEFINE_ISIN_OP(float32, float, set_cmp_float)
DEFINE_ISIN_OP(int32, int32_t, set_cmp_int32)
DEFINE_ISIN_OP(int64, int64_t, set_cmp_int64)
DEFINE_ISIN_OP(uint8, uint8_t, set_cmp_uint8)
DEFINE_ISIN_OP(complex128, cpx_t, set_cmp_complex128)
DEFINE_ISIN_OP(complex64, cpx_f_t, set_cmp_complex64)

void ndarray_isin(const void *ar1, int size1, const void *ar2, int size2, uint8_t *dest, int dtype, int invert) { \
    if (ar1 == NULL || ar2 == NULL || dest == NULL) return; \
    switch (dtype) { \
        case DTYPE_FLOAT64: isin_float64((const double *)ar1, size1, (const double *)ar2, size2, dest, invert); break; \
        case DTYPE_FLOAT32: isin_float32((const float *)ar1, size1, (const float *)ar2, size2, dest, invert); break; \
        case DTYPE_INT32: isin_int32((const int32_t *)ar1, size1, (const int32_t *)ar2, size2, dest, invert); break; \
        case DTYPE_INT64: isin_int64((const int64_t *)ar1, size1, (const int64_t *)ar2, size2, dest, invert); break; \
        case DTYPE_UINT8: \
        case DTYPE_BOOLEAN: isin_uint8((const uint8_t *)ar1, size1, (const uint8_t *)ar2, size2, dest, invert); break; \
        case DTYPE_COMPLEX128: isin_complex128((const cpx_t *)ar1, size1, (const cpx_t *)ar2, size2, dest, invert); break; \
        case DTYPE_COMPLEX64: isin_complex64((const cpx_f_t *)ar1, size1, (const cpx_f_t *)ar2, size2, dest, invert); break; \
    } \
}
// --- MEDIAN & QUANTILE IMPLEMENTATIONS ---

// Helper macro for median reduction to avoid code duplication.
// Allocates tmp_buf once per reduction call.
#define IMPLEMENT_MEDIAN_REDUCTION(NAME, TYPE, SORT_FUNC, CAST_TYPE) \
void NAME(const TYPE *src, const int *stridesSrc, \
          TYPE *dest, const int *stridesDest, \
          const int *shape, int rank, int axis) { \
    if (src == NULL || dest == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return; \
    int size_axis = shape[axis]; \
    if (size_axis <= 0) return; \
    TYPE *tmp_buf = (TYPE *)malloc(size_axis * sizeof(TYPE)); \
    if (tmp_buf == NULL) return; \
    int coord[8] = {0}; \
    int outer_size = 1; \
    for (int d = 0; d < rank; d++) { \
        if (d != axis) outer_size *= shape[d]; \
    } \
    for (int o = 0; o < outer_size; o++) { \
        int offsetRes = 0; \
        int offsetSrc = 0; \
        for (int d = 0; d < rank; d++) { \
            if (d != axis) { \
                offsetSrc += coord[d] * stridesSrc[d]; \
                if (rank > 1) { \
                    int targetD = (d < axis) ? d : (d - 1); \
                    offsetRes += coord[d] * stridesDest[targetD]; \
                } \
            } \
        } \
        int stride_axis = stridesSrc[axis]; \
        for (int i = 0; i < size_axis; i++) { \
            tmp_buf[i] = src[offsetSrc + i * stride_axis]; \
        } \
        SORT_FUNC((CAST_TYPE)tmp_buf, size_axis, 0); \
        TYPE median_val; \
        if (size_axis % 2 == 1) { \
            median_val = tmp_buf[size_axis / 2]; \
        } else { \
            median_val = (TYPE)(((double)tmp_buf[size_axis / 2 - 1] + (double)tmp_buf[size_axis / 2]) / 2.0); \
        } \
        dest[offsetRes] = median_val; \
        for (int d = rank - 1; d >= 0; d--) { \
            if (d == axis) continue; \
            coord[d]++; \
            if (coord[d] < shape[d]) break; \
            coord[d] = 0; \
        } \
    } \
    free(tmp_buf); \
}

// Special case for double and float where division by 2 should be floating point division
void s_median_double(const double *src, const int *stridesSrc,
                     double *dest, const int *stridesDest,
                     const int *shape, int rank, int axis) {
    if (src == NULL || dest == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    int size_axis = shape[axis];
    if (size_axis <= 0) return;
    double *tmp_buf = (double *)malloc(size_axis * sizeof(double));
    if (tmp_buf == NULL) return;
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    for (int o = 0; o < outer_size; o++) {
        int offsetRes = 0;
        int offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesDest[targetD];
                }
            }
        }
        int stride_axis = stridesSrc[axis];
        for (int i = 0; i < size_axis; i++) {
            tmp_buf[i] = src[offsetSrc + i * stride_axis];
        }
        native_sort_double(tmp_buf, size_axis, 0);
        double median_val;
        if (size_axis % 2 == 1) {
            median_val = tmp_buf[size_axis / 2];
        } else {
            median_val = (tmp_buf[size_axis / 2 - 1] + tmp_buf[size_axis / 2]) / 2.0;
        }
        dest[offsetRes] = median_val;
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
    free(tmp_buf);
}

void s_median_float(const float *src, const int *stridesSrc,
                    float *dest, const int *stridesDest,
                    const int *shape, int rank, int axis) {
    if (src == NULL || dest == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    int size_axis = shape[axis];
    if (size_axis <= 0) return;
    float *tmp_buf = (float *)malloc(size_axis * sizeof(float));
    if (tmp_buf == NULL) return;
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    for (int o = 0; o < outer_size; o++) {
        int offsetRes = 0;
        int offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesDest[targetD];
                }
            }
        }
        int stride_axis = stridesSrc[axis];
        for (int i = 0; i < size_axis; i++) {
            tmp_buf[i] = src[offsetSrc + i * stride_axis];
        }
        native_sort_float(tmp_buf, size_axis, 0);
        float median_val;
        if (size_axis % 2 == 1) {
            median_val = tmp_buf[size_axis / 2];
        } else {
            median_val = (tmp_buf[size_axis / 2 - 1] + tmp_buf[size_axis / 2]) / 2.0f;
        }
        dest[offsetRes] = median_val;
        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
    free(tmp_buf);
}

IMPLEMENT_MEDIAN_REDUCTION(s_median_int64, int64_t, native_sort_int64, long long *)
IMPLEMENT_MEDIAN_REDUCTION(s_median_int32, int32_t, native_sort_int32, int *)
IMPLEMENT_MEDIAN_REDUCTION(s_median_uint8, uint8_t, insertion_sort_uint8, uint8_t *)

void s_median_complex128(const cpx_t *src, const int *stridesSrc,
                         cpx_t *dest, const int *stridesDest,
                         const int *shape, int rank, int axis) {
    if (src == NULL || dest == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    int size_axis = shape[axis];
    if (size_axis <= 0) return;
    double *tmp_buf = (double *)malloc(size_axis * sizeof(double));
    if (tmp_buf == NULL) return;
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    for (int o = 0; o < outer_size; o++) {
        int offsetRes = 0;
        int offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesDest[targetD];
                }
            }
        }
        int stride_axis = stridesSrc[axis];
        
        // Real part
        for (int i = 0; i < size_axis; i++) {
            tmp_buf[i] = src[offsetSrc + i * stride_axis].r;
        }
        native_sort_double(tmp_buf, size_axis, 0);
        double median_r;
        if (size_axis % 2 == 1) {
            median_r = tmp_buf[size_axis / 2];
        } else {
            median_r = (tmp_buf[size_axis / 2 - 1] + tmp_buf[size_axis / 2]) / 2.0;
        }

        // Imaginary part
        for (int i = 0; i < size_axis; i++) {
            tmp_buf[i] = src[offsetSrc + i * stride_axis].i;
        }
        native_sort_double(tmp_buf, size_axis, 0);
        double median_i;
        if (size_axis % 2 == 1) {
            median_i = tmp_buf[size_axis / 2];
        } else {
            median_i = (tmp_buf[size_axis / 2 - 1] + tmp_buf[size_axis / 2]) / 2.0;
        }

        dest[offsetRes].r = median_r;
        dest[offsetRes].i = median_i;

        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
    free(tmp_buf);
}

void s_median_complex64(const cpx_f_t *src, const int *stridesSrc,
                        cpx_f_t *dest, const int *stridesDest,
                        const int *shape, int rank, int axis) {
    if (src == NULL || dest == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return;
    int size_axis = shape[axis];
    if (size_axis <= 0) return;
    float *tmp_buf = (float *)malloc(size_axis * sizeof(float));
    if (tmp_buf == NULL) return;
    int coord[8] = {0};
    int outer_size = 1;
    for (int d = 0; d < rank; d++) {
        if (d != axis) outer_size *= shape[d];
    }
    for (int o = 0; o < outer_size; o++) {
        int offsetRes = 0;
        int offsetSrc = 0;
        for (int d = 0; d < rank; d++) {
            if (d != axis) {
                offsetSrc += coord[d] * stridesSrc[d];
                if (rank > 1) {
                    int targetD = (d < axis) ? d : (d - 1);
                    offsetRes += coord[d] * stridesDest[targetD];
                }
            }
        }
        int stride_axis = stridesSrc[axis];
        
        // Real part
        for (int i = 0; i < size_axis; i++) {
            tmp_buf[i] = src[offsetSrc + i * stride_axis].r;
        }
        native_sort_float(tmp_buf, size_axis, 0);
        float median_r;
        if (size_axis % 2 == 1) {
            median_r = tmp_buf[size_axis / 2];
        } else {
            median_r = (tmp_buf[size_axis / 2 - 1] + tmp_buf[size_axis / 2]) / 2.0f;
        }

        // Imaginary part
        for (int i = 0; i < size_axis; i++) {
            tmp_buf[i] = src[offsetSrc + i * stride_axis].i;
        }
        native_sort_float(tmp_buf, size_axis, 0);
        float median_i;
        if (size_axis % 2 == 1) {
            median_i = tmp_buf[size_axis / 2];
        } else {
            median_i = (tmp_buf[size_axis / 2 - 1] + tmp_buf[size_axis / 2]) / 2.0f;
        }

        dest[offsetRes].r = median_r;
        dest[offsetRes].i = median_i;

        for (int d = rank - 1; d >= 0; d--) {
            if (d == axis) continue;
            coord[d]++;
            if (coord[d] < shape[d]) break;
            coord[d] = 0;
        }
    }
    free(tmp_buf);
}

// Global reductions for median (contiguous)
double r_median_double(const double *src, int size) { return stats_median_double(src, 1, size); }
float r_median_float(const float *src, int size) { return stats_median_float(src, 1, size); }
int64_t r_median_int64(const int64_t *src, int size) { return stats_median_int64(src, 1, size); }
int32_t r_median_int32(const int32_t *src, int size) { return stats_median_int32(src, 1, size); }
uint8_t r_median_uint8(const uint8_t *src, int size) { return stats_median_uint8(src, 1, size); }
cpx_t r_median_complex128(const cpx_t *src, int size) { return stats_median_complex128(src, 1, size); }
cpx_f_t r_median_complex64(const cpx_f_t *src, int size) { return stats_median_complex64(src, 1, size); }

// Quantile helper definitions

typedef struct {
    int idx_low;
    int idx_high;
    double weight;
} QuantileInterpolationSpecs;

static QuantileInterpolationSpecs get_quantile_specs(int N, double p, int method) {
    QuantileInterpolationSpecs specs;
    specs.idx_low = 0;
    specs.idx_high = 0;
    specs.weight = 0.0;

    if (N <= 0) return specs;

    if (method < 0 || method > 12) {
        method = QUANTILE_LINEAR;
    }

    // Continuous methods (4-9)
    if (method >= QUANTILE_INTERPOLATED_INVERTED_CDF && method <= QUANTILE_NORMAL_UNBIASED) {
        double alpha = 0.0, beta = 0.0;
        double idx;
        if (method == QUANTILE_LINEAR) {
            idx = (double)(N - 1) * p;
        } else {
            switch (method) {
                case QUANTILE_INTERPOLATED_INVERTED_CDF: alpha = 0.0; beta = 1.0; break;
                case QUANTILE_HAZEN:                     alpha = 0.5; beta = 0.5; break;
                case QUANTILE_WEIBULL:                   alpha = 0.0; beta = 0.0; break;
                case QUANTILE_MEDIAN_UNBIASED:           alpha = 1.0/3.0; beta = 1.0/3.0; break;
                case QUANTILE_NORMAL_UNBIASED:           alpha = 3.0/8.0; beta = 3.0/8.0; break;
                default: break;
            }
            idx = (double)N * p + (alpha + p * (1.0 - alpha - beta)) - 1.0;
        }
        int j = (int)floor(idx);
        specs.idx_low = j;
        specs.idx_high = j + 1;
        specs.weight = idx - (double)j;
    }
    // Discontinuous methods (1-3)
    else if (method == QUANTILE_INVERTED_CDF) {
        double idx = p * N - 1.0;
        double prev = floor(idx);
        double gamma = idx - prev;
        int res_idx = (gamma == 0.0) ? (int)prev : (int)prev + 1;
        specs.idx_low = res_idx;
        specs.idx_high = res_idx;
        specs.weight = 0.0;
    }
    else if (method == QUANTILE_AVERAGED_INVERTED_CDF) {
        double idx = p * N - 1.0;
        int j = (int)floor(idx);
        double gamma = idx - (double)j;
        specs.idx_low = j;
        specs.idx_high = j + 1;
        specs.weight = (gamma == 0.0) ? 0.5 : 1.0;
    }
    else if (method == QUANTILE_CLOSEST_OBSERVATION) {
        double idx = p * N - 1.5;
        double prev = floor(idx);
        double gamma = idx - prev;
        int prev_int = (int)prev;
        int is_odd = (prev_int % 2 != 0);
        int cond = (gamma == 0.0) && is_odd;
        int res_idx = cond ? prev_int : prev_int + 1;
        specs.idx_low = res_idx;
        specs.idx_high = res_idx;
        specs.weight = 0.0;
    }
    // Backward compatibility methods (10-13)
    else if (method == QUANTILE_LOWER) {
        double idx = p * (N - 1);
        int res_idx = (int)floor(idx);
        specs.idx_low = res_idx;
        specs.idx_high = res_idx;
        specs.weight = 0.0;
    }
    else if (method == QUANTILE_HIGHER) {
        double idx = p * (N - 1);
        int res_idx = (int)ceil(idx);
        specs.idx_low = res_idx;
        specs.idx_high = res_idx;
        specs.weight = 0.0;
    }
    else if (method == QUANTILE_MIDPOINT) {
        double idx = p * (N - 1);
        specs.idx_low = (int)floor(idx);
        specs.idx_high = (int)ceil(idx);
        specs.weight = 0.5;
    }
    else if (method == QUANTILE_NEAREST) {
        double idx = p * (N - 1);
        int res_idx = (int)rint(idx);
        specs.idx_low = res_idx;
        specs.idx_high = res_idx;
        specs.weight = 0.0;
    }

    // Clip indices
    if (specs.idx_low < 0) specs.idx_low = 0;
    if (specs.idx_low >= N) specs.idx_low = N - 1;
    if (specs.idx_high < 0) specs.idx_high = 0;
    if (specs.idx_high >= N) specs.idx_high = N - 1;

    return specs;
}

#define DEFINE_NUMERIC_QUANTILE(TYPE, NAME_SUFFIX, SORTER, CAST_TYPE) \
double stats_quantile_##NAME_SUFFIX(const TYPE *base, int _stride, int len, double q, int method) { \
    if (len <= 0) return 0.0; \
    if (q < 0.0) q = 0.0; \
    if (q > 1.0) q = 1.0; \
    TYPE *buf = (TYPE*)malloc(len * sizeof(TYPE)); \
    if (buf == NULL) return 0.0; \
    for (int i = 0; i < len; i++) { \
        buf[i] = *(base + i * _stride); \
    } \
    SORTER((CAST_TYPE)buf, len, 0); \
    QuantileInterpolationSpecs specs = get_quantile_specs(len, q, method); \
    double res = (1.0 - specs.weight) * (double)buf[specs.idx_low] + specs.weight * (double)buf[specs.idx_high]; \
    free(buf); \
    return res; \
}

DEFINE_NUMERIC_QUANTILE(double, double, native_sort_double, double*)
DEFINE_NUMERIC_QUANTILE(float, float, native_sort_float, float*)
DEFINE_NUMERIC_QUANTILE(int64_t, int64, native_sort_int64, long long*)
DEFINE_NUMERIC_QUANTILE(int32_t, int32, native_sort_int32, int*)
DEFINE_NUMERIC_QUANTILE(uint8_t, uint8, insertion_sort_uint8, uint8_t*)

// Global reductions for quantile (contiguous)
double r_quantile_double(const double *src, int size, double q, int method) { return stats_quantile_double(src, 1, size, q, method); }
double r_quantile_float(const float *src, int size, double q, int method) { return stats_quantile_float(src, 1, size, q, method); }
double r_quantile_int64(const int64_t *src, int size, double q, int method) { return stats_quantile_int64(src, 1, size, q, method); }
double r_quantile_int32(const int32_t *src, int size, double q, int method) { return stats_quantile_int32(src, 1, size, q, method); }
double r_quantile_uint8(const uint8_t *src, int size, double q, int method) { return stats_quantile_uint8(src, 1, size, q, method); }

// Helper macro for quantile reduction.
// Allocates tmp_buf once per reduction call.
#define IMPLEMENT_QUANTILE_REDUCTION(NAME, TYPE, SORT_FUNC, CAST_TYPE) \
void NAME(const TYPE *src, const int *stridesSrc, \
          double *dest, const int *stridesDest, \
          const int *shape, int rank, int axis, double q, int method) { \
    if (src == NULL || dest == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return; \
    int size_axis = shape[axis]; \
    if (size_axis <= 0) return; \
    if (q < 0.0 || q > 1.0) return; \
    TYPE *tmp_buf = (TYPE *)malloc(size_axis * sizeof(TYPE)); \
    if (tmp_buf == NULL) return; \
    int coord[8] = {0}; \
    int outer_size = 1; \
    for (int d = 0; d < rank; d++) { \
        if (d != axis) outer_size *= shape[d]; \
    } \
    QuantileInterpolationSpecs specs = get_quantile_specs(size_axis, q, method); \
    for (int o = 0; o < outer_size; o++) { \
        int offsetRes = 0; \
        int offsetSrc = 0; \
        for (int d = 0; d < rank; d++) { \
            if (d != axis) { \
                offsetSrc += coord[d] * stridesSrc[d]; \
                if (rank > 1) { \
                    int targetD = (d < axis) ? d : (d - 1); \
                    offsetRes += coord[d] * stridesDest[targetD]; \
                } \
            } \
        } \
        int stride_axis = stridesSrc[axis]; \
        for (int i = 0; i < size_axis; i++) { \
            tmp_buf[i] = src[offsetSrc + i * stride_axis]; \
        } \
        SORT_FUNC((CAST_TYPE)tmp_buf, size_axis, 0); \
        double val = (1.0 - specs.weight) * (double)tmp_buf[specs.idx_low] + specs.weight * (double)tmp_buf[specs.idx_high]; \
        dest[offsetRes] = val; \
        for (int d = rank - 1; d >= 0; d--) { \
            if (d == axis) continue; \
            coord[d]++; \
            if (coord[d] < shape[d]) break; \
            coord[d] = 0; \
        } \
    } \
    free(tmp_buf); \
}

IMPLEMENT_QUANTILE_REDUCTION(s_quantile_double, double, native_sort_double, double *)
IMPLEMENT_QUANTILE_REDUCTION(s_quantile_float, float, native_sort_float, float *)
IMPLEMENT_QUANTILE_REDUCTION(s_quantile_int64, int64_t, native_sort_int64, long long *)
IMPLEMENT_QUANTILE_REDUCTION(s_quantile_int32, int32_t, native_sort_int32, int *)
IMPLEMENT_QUANTILE_REDUCTION(s_quantile_uint8, uint8_t, insertion_sort_uint8, uint8_t *)
/* ============================================================================
 * SECTION 11: INTERPOLATION KERNELS
 * ============================================================================
 */

int is_strictly_increasing_double(const double *arr, int size, int stride) {
    if (size <= 1) return 1;
    for (int i = 1; i < size; i++) {
        if (arr[i * stride] <= arr[(i - 1) * stride]) {
            return 0;
        }
    }
    return 1;
}

static inline int find_interval_double(const double *xp, int xp_size, int stride, double x_val) {
    if (x_val == xp[(xp_size - 1) * stride]) {
        return xp_size - 2;
    }
    int low = 0;
    int high = xp_size - 1;
    while (low < high - 1) {
        int mid = low + (high - low) / 2;
        if (xp[mid * stride] <= x_val) {
            low = mid;
        } else {
            high = mid;
        }
    }
    return low;
}

void v_interp_double(const double *x, int x_size,
                     const double *xp, int xp_size,
                     const double *fp, double *res,
                     const double *left, const double *right) {
    if (x == NULL || xp == NULL || fp == NULL || res == NULL || x_size <= 0 || xp_size <= 0) {
        return;
    }
    if (xp_size == 1) {
        double val = fp[0];
        for (int i = 0; i < x_size; i++) {
            if (x[i] < xp[0]) {
                res[i] = (left != NULL) ? *left : val;
            } else if (x[i] > xp[0]) {
                res[i] = (right != NULL) ? *right : val;
            } else {
                res[i] = val;
            }
        }
        return;
    }

    double xp_min = xp[0];
    double xp_max = xp[xp_size - 1];

    for (int i = 0; i < x_size; i++) {
        double xv = x[i];
        if (xv < xp_min) {
            res[i] = (left != NULL) ? *left : fp[0];
        } else if (xv > xp_max) {
            res[i] = (right != NULL) ? *right : fp[xp_size - 1];
        } else {
            int j = find_interval_double(xp, xp_size, 1, xv);
            double x0 = xp[j];
            double x1 = xp[j + 1];
            double y0 = fp[j];
            double y1 = fp[j + 1];
            if (x0 == x1) {
                res[i] = y0;
            } else {
                res[i] = y0 + (y1 - y0) * (xv - x0) / (x1 - x0);
            }
        }
    }
}

void s_interp_double(const double *x, const int *stridesX,
                     const double *xp, int strideXP, int xp_size,
                     const double *fp, int strideFP,
                     double *res, const int *stridesRes,
                     const int *shape, int rank,
                     const double *left, const double *right) {
    if (x == NULL || xp == NULL || fp == NULL || res == NULL || rank <= 0 || rank > 8 || xp_size <= 0) {
        return;
    }

    if (xp_size == 1) {
        double val = fp[0];
        double xp0 = xp[0];
        int total_elements = 1;
        for (int i = 0; i < rank; i++) total_elements *= shape[i];
        int coord[8] = {0};
        int offsetX = 0, offsetRes = 0;
        for (int el = 0; el < total_elements; el++) {
            double xv = x[offsetX];
            if (xv < xp0) {
                res[offsetRes] = (left != NULL) ? *left : val;
            } else if (xv > xp0) {
                res[offsetRes] = (right != NULL) ? *right : val;
            } else {
                res[offsetRes] = val;
            }
            for (int d = rank - 1; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetX += stridesX[d];
                    offsetRes += stridesRes[d];
                    break;
                }
                coord[d] = 0;
                offsetX -= (shape[d] - 1) * stridesX[d];
                offsetRes -= (shape[d] - 1) * stridesRes[d];
            }
        }
        return;
    }

    double xp_min = xp[0];
    double xp_max = xp[(xp_size - 1) * strideXP];

    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    int coord[8] = {0};
    int offsetX = 0, offsetRes = 0;

    for (int el = 0; el < total_elements; el++) {
        double xv = x[offsetX];
        if (xv < xp_min) {
            res[offsetRes] = (left != NULL) ? *left : fp[0];
        } else if (xv > xp_max) {
            res[offsetRes] = (right != NULL) ? *right : fp[(xp_size - 1) * strideFP];
        } else {
            int j = find_interval_double(xp, xp_size, strideXP, xv);
            double x0 = xp[j * strideXP];
            double x1 = xp[(j + 1) * strideXP];
            double y0 = fp[j * strideFP];
            double y1 = fp[(j + 1) * strideFP];
            if (x0 == x1) {
                res[offsetRes] = y0;
            } else {
                res[offsetRes] = y0 + (y1 - y0) * (xv - x0) / (x1 - x0);
            }
        }

        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetX += stridesX[d];
                offsetRes += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetX -= (shape[d] - 1) * stridesX[d];
            offsetRes -= (shape[d] - 1) * stridesRes[d];
        }
    }
}

static inline int find_interval_float(const float *xp, int xp_size, int stride, float x_val) {
    if (x_val == xp[(xp_size - 1) * stride]) {
        return xp_size - 2;
    }
    int low = 0;
    int high = xp_size - 1;
    while (low < high - 1) {
        int mid = low + (high - low) / 2;
        if (xp[mid * stride] <= x_val) {
            low = mid;
        } else {
            high = mid;
        }
    }
    return low;
}

void v_interp_float(const float *x, int x_size,
                    const float *xp, int xp_size,
                    const float *fp, float *res,
                    const float *left, const float *right) {
    if (x == NULL || xp == NULL || fp == NULL || res == NULL || x_size <= 0 || xp_size <= 0) {
        return;
    }
    if (xp_size == 1) {
        float val = fp[0];
        for (int i = 0; i < x_size; i++) {
            if (x[i] < xp[0]) {
                res[i] = (left != NULL) ? *left : val;
            } else if (x[i] > xp[0]) {
                res[i] = (right != NULL) ? *right : val;
            } else {
                res[i] = val;
            }
        }
        return;
    }

    float xp_min = xp[0];
    float xp_max = xp[xp_size - 1];

    for (int i = 0; i < x_size; i++) {
        float xv = x[i];
        if (xv < xp_min) {
            res[i] = (left != NULL) ? *left : fp[0];
        } else if (xv > xp_max) {
            res[i] = (right != NULL) ? *right : fp[xp_size - 1];
        } else {
            int j = find_interval_float(xp, xp_size, 1, xv);
            float x0 = xp[j];
            float x1 = xp[j + 1];
            float y0 = fp[j];
            float y1 = fp[j + 1];
            if (x0 == x1) {
                res[i] = y0;
            } else {
                res[i] = y0 + (y1 - y0) * (xv - x0) / (x1 - x0);
            }
        }
    }
}

void s_interp_float(const float *x, const int *stridesX,
                    const float *xp, int strideXP, int xp_size,
                    const float *fp, int strideFP,
                    float *res, const int *stridesRes,
                    const int *shape, int rank,
                    const float *left, const float *right) {
    if (x == NULL || xp == NULL || fp == NULL || res == NULL || rank <= 0 || rank > 8 || xp_size <= 0) {
        return;
    }

    if (xp_size == 1) {
        float val = fp[0];
        float xp0 = xp[0];
        int total_elements = 1;
        for (int i = 0; i < rank; i++) total_elements *= shape[i];
        int coord[8] = {0};
        int offsetX = 0, offsetRes = 0;
        for (int el = 0; el < total_elements; el++) {
            float xv = x[offsetX];
            if (xv < xp0) {
                res[offsetRes] = (left != NULL) ? *left : val;
            } else if (xv > xp0) {
                res[offsetRes] = (right != NULL) ? *right : val;
            } else {
                res[offsetRes] = val;
            }
            for (int d = rank - 1; d >= 0; d--) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetX += stridesX[d];
                    offsetRes += stridesRes[d];
                    break;
                }
                coord[d] = 0;
                offsetX -= (shape[d] - 1) * stridesX[d];
                offsetRes -= (shape[d] - 1) * stridesRes[d];
            }
        }
        return;
    }

    float xp_min = xp[0];
    float xp_max = xp[(xp_size - 1) * strideXP];

    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    int coord[8] = {0};
    int offsetX = 0, offsetRes = 0;

    for (int el = 0; el < total_elements; el++) {
        float xv = x[offsetX];
        if (xv < xp_min) {
            res[offsetRes] = (left != NULL) ? *left : fp[0];
        } else if (xv > xp_max) {
            res[offsetRes] = (right != NULL) ? *right : fp[(xp_size - 1) * strideFP];
        } else {
            int j = find_interval_float(xp, xp_size, strideXP, xv);
            float x0 = xp[j * strideXP];
            float x1 = xp[(j + 1) * strideXP];
            float y0 = fp[j * strideFP];
            float y1 = fp[(j + 1) * strideFP];
            if (x0 == x1) {
                res[offsetRes] = y0;
            } else {
                res[offsetRes] = y0 + (y1 - y0) * (xv - x0) / (x1 - x0);
            }
        }

        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetX += stridesX[d];
                offsetRes += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetX -= (shape[d] - 1) * stridesX[d];
            offsetRes -= (shape[d] - 1) * stridesRes[d];
        }
    }
}

// ============================================================================
// SECTION 12: COMPARISON & EQUALITY KERNELS
// ============================================================================

#define EXPR_EQ(x, y) ((x) == (y))
#define EXPR_NE(x, y) ((x) != (y))
#define EXPR_LT(x, y) ((x) < (y))
#define EXPR_LE(x, y) ((x) <= (y))
#define EXPR_GT(x, y) ((x) > (y))
#define EXPR_GE(x, y) ((x) >= (y))

#define EXPR_EQ_C128_C128(x, y) ((x).r == (y).r && (x).i == (y).i)
#define EXPR_NE_C128_C128(x, y) ((x).r != (y).r || (x).i != (y).i)
#define EXPR_EQ_C64_C64(x, y) ((x).r == (y).r && (x).i == (y).i)
#define EXPR_NE_C64_C64(x, y) ((x).r != (y).r || (x).i != (y).i)

#define EXPR_EQ_C128_REAL(x, y) ((x).r == (y) && (x).i == 0)
#define EXPR_NE_C128_REAL(x, y) ((x).r != (y) || (x).i != 0)
#define EXPR_EQ_REAL_C128(x, y) ((x) == (y).r && (y).i == 0)
#define EXPR_NE_REAL_C128(x, y) ((x) != (y).r || (y).i != 0)

#define EXPR_EQ_C64_REAL(x, y) ((x).r == (y) && (x).i == 0)
#define EXPR_NE_C64_REAL(x, y) ((x).r != (y) || (x).i != 0)
#define EXPR_EQ_REAL_C64(x, y) ((x) == (y).r && (y).i == 0)
#define EXPR_NE_REAL_C64(x, y) ((x) != (y).r || (y).i != 0)

#define EXPR_EQ_C128_C64(x, y) ((x).r == (y).r && (x).i == (y).i)
#define EXPR_NE_C128_C64(x, y) ((x).r != (y).r || (x).i != (y).i)
#define EXPR_EQ_C64_C128(x, y) ((x).r == (y).r && (x).i == (y).i)
#define EXPR_NE_C64_C128(x, y) ((x).r != (y).r || (x).i != (y).i)

#define DEFINE_COMPARE_FUNC(NAME, T1, T2, EXPR) \
void NAME(const T1 *a, const int *stridesA, \
          const T2 *b, const int *stridesB, \
          uint8_t *res, const int *stridesRes, \
          const int *shape, int rank) { \
    if (a == NULL || b == NULL || res == NULL || rank < 0 || rank > 8) return; \
    int total_elements = 1; \
    for (int i = 0; i < rank; i++) total_elements *= shape[i]; \
    int is_a_contig = 1, is_b_contig = 1, is_res_contig = 1; \
    int expected_stride = 1; \
    for (int i = rank - 1; i >= 0; i--) { \
        if (stridesA[i] != expected_stride) is_a_contig = 0; \
        if (stridesB[i] != expected_stride) is_b_contig = 0; \
        if (stridesRes[i] != expected_stride) is_res_contig = 0; \
        expected_stride *= shape[i]; \
    } \
    int is_a_scal = 1, is_b_scal = 1; \
    for (int i = 0; i < rank; i++) { \
        if (stridesA[i] != 0) is_a_scal = 0; \
        if (stridesB[i] != 0) is_b_scal = 0; \
    } \
    if (is_res_contig) { \
        if (is_a_contig && is_b_contig) { \
            for (int i = 0; i < total_elements; i++) { \
                res[i] = (EXPR(a[i], b[i])) ? 1 : 0; \
            } \
            return; \
        } \
        if (is_a_contig && is_b_scal) { \
            T2 val = b[0]; \
            for (int i = 0; i < total_elements; i++) { \
                res[i] = (EXPR(a[i], val)) ? 1 : 0; \
            } \
            return; \
        } \
        if (is_a_scal && is_b_contig) { \
            T1 val = a[0]; \
            for (int i = 0; i < total_elements; i++) { \
                res[i] = (EXPR(val, b[i])) ? 1 : 0; \
            } \
            return; \
        } \
    } \
    int coord[8] = {0}; \
    int offsetA = 0, offsetB = 0, offsetRes = 0; \
    for (int el = 0; el < total_elements; el++) { \
        res[offsetRes] = (EXPR(a[offsetA], b[offsetB])) ? 1 : 0; \
        for (int d = rank - 1; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) { \
                offsetA += stridesA[d]; \
                offsetB += stridesB[d]; \
                offsetRes += stridesRes[d]; \
                break; \
            } \
            coord[d] = 0; \
            offsetA -= (shape[d] - 1) * stridesA[d]; \
            offsetB -= (shape[d] - 1) * stridesB[d]; \
            offsetRes -= (shape[d] - 1) * stridesRes[d]; \
        } \
    } \
}

// Instantiate real-real comparisons
#define DEFINE_REAL_COMPARE_T2(OP_NAME, T1, T1_NAME, T2, T2_NAME, EXPR) \
    DEFINE_COMPARE_FUNC(s_compare_##OP_NAME##_##T1_NAME##_##T2_NAME, T1, T2, EXPR)

#define DEFINE_REAL_COMPARE_T1(OP_NAME, T1, T1_NAME, EXPR) \
    DEFINE_REAL_COMPARE_T2(OP_NAME, T1, T1_NAME, double, double, EXPR) \
    DEFINE_REAL_COMPARE_T2(OP_NAME, T1, T1_NAME, float, float, EXPR) \
    DEFINE_REAL_COMPARE_T2(OP_NAME, T1, T1_NAME, int32_t, int32, EXPR) \
    DEFINE_REAL_COMPARE_T2(OP_NAME, T1, T1_NAME, int64_t, int64, EXPR) \
    DEFINE_REAL_COMPARE_T2(OP_NAME, T1, T1_NAME, uint8_t, uint8, EXPR) \
    DEFINE_REAL_COMPARE_T2(OP_NAME, T1, T1_NAME, int16_t, int16, EXPR)

#define DEFINE_REAL_COMPARE_ALL(OP_NAME, EXPR) \
    DEFINE_REAL_COMPARE_T1(OP_NAME, double, double, EXPR) \
    DEFINE_REAL_COMPARE_T1(OP_NAME, float, float, EXPR) \
    DEFINE_REAL_COMPARE_T1(OP_NAME, int32_t, int32, EXPR) \
    DEFINE_REAL_COMPARE_T1(OP_NAME, int64_t, int64, EXPR) \
    DEFINE_REAL_COMPARE_T1(OP_NAME, uint8_t, uint8, EXPR) \
    DEFINE_REAL_COMPARE_T1(OP_NAME, int16_t, int16, EXPR)

DEFINE_REAL_COMPARE_ALL(eq, EXPR_EQ)
DEFINE_REAL_COMPARE_ALL(ne, EXPR_NE)
DEFINE_REAL_COMPARE_ALL(lt, EXPR_LT)
DEFINE_REAL_COMPARE_ALL(le, EXPR_LE)
DEFINE_REAL_COMPARE_ALL(gt, EXPR_GT)
DEFINE_REAL_COMPARE_ALL(ge, EXPR_GE)

#undef DEFINE_REAL_COMPARE_ALL
#undef DEFINE_REAL_COMPARE_T1
#undef DEFINE_REAL_COMPARE_T2

// Complex vs Complex
#define DEFINE_COMPLEX_COMP_CC(OP_NAME, T1, T1_NAME, T2, T2_NAME, EXPR) \
    DEFINE_COMPARE_FUNC(s_compare_##OP_NAME##_##T1_NAME##_##T2_NAME, T1, T2, EXPR)

// Complex vs Real
#define DEFINE_COMPLEX_COMP_CR(OP_NAME, CPX_T, CPX_NAME, R_T, R_NAME, EXPR) \
    DEFINE_COMPARE_FUNC(s_compare_##OP_NAME##_##CPX_NAME##_##R_NAME, CPX_T, R_T, EXPR)

// Real vs Complex
#define DEFINE_COMPLEX_COMP_RC(OP_NAME, R_T, R_NAME, CPX_T, CPX_NAME, EXPR) \
    DEFINE_COMPARE_FUNC(s_compare_##OP_NAME##_##R_NAME##_##CPX_NAME, R_T, CPX_T, EXPR)

#define INSTANTIATE_CR_RC_C128(OP_NAME, R_T, R_NAME, EXPR_CR, EXPR_RC) \
    DEFINE_COMPLEX_COMP_CR(OP_NAME, cpx_t, complex128, R_T, R_NAME, EXPR_CR) \
    DEFINE_COMPLEX_COMP_RC(OP_NAME, R_T, R_NAME, cpx_t, complex128, EXPR_RC)

#define INSTANTIATE_CR_RC_C64(OP_NAME, R_T, R_NAME, EXPR_CR, EXPR_RC) \
    DEFINE_COMPLEX_COMP_CR(OP_NAME, cpx_f_t, complex64, R_T, R_NAME, EXPR_CR) \
    DEFINE_COMPLEX_COMP_RC(OP_NAME, R_T, R_NAME, cpx_f_t, complex64, EXPR_RC)

// EQ
DEFINE_COMPLEX_COMP_CC(eq, cpx_t, complex128, cpx_t, complex128, EXPR_EQ_C128_C128)
DEFINE_COMPLEX_COMP_CC(eq, cpx_f_t, complex64, cpx_f_t, complex64, EXPR_EQ_C64_C64)
DEFINE_COMPLEX_COMP_CC(eq, cpx_t, complex128, cpx_f_t, complex64, EXPR_EQ_C128_C64)
DEFINE_COMPLEX_COMP_CC(eq, cpx_f_t, complex64, cpx_t, complex128, EXPR_EQ_C64_C128)

INSTANTIATE_CR_RC_C128(eq, double, double, EXPR_EQ_C128_REAL, EXPR_EQ_REAL_C128)
INSTANTIATE_CR_RC_C128(eq, float, float, EXPR_EQ_C128_REAL, EXPR_EQ_REAL_C128)
INSTANTIATE_CR_RC_C128(eq, int32_t, int32, EXPR_EQ_C128_REAL, EXPR_EQ_REAL_C128)
INSTANTIATE_CR_RC_C128(eq, int64_t, int64, EXPR_EQ_C128_REAL, EXPR_EQ_REAL_C128)
INSTANTIATE_CR_RC_C128(eq, uint8_t, uint8, EXPR_EQ_C128_REAL, EXPR_EQ_REAL_C128)
INSTANTIATE_CR_RC_C128(eq, int16_t, int16, EXPR_EQ_C128_REAL, EXPR_EQ_REAL_C128)

INSTANTIATE_CR_RC_C64(eq, double, double, EXPR_EQ_C64_REAL, EXPR_EQ_REAL_C64)
INSTANTIATE_CR_RC_C64(eq, float, float, EXPR_EQ_C64_REAL, EXPR_EQ_REAL_C64)
INSTANTIATE_CR_RC_C64(eq, int32_t, int32, EXPR_EQ_C64_REAL, EXPR_EQ_REAL_C64)
INSTANTIATE_CR_RC_C64(eq, int64_t, int64, EXPR_EQ_C64_REAL, EXPR_EQ_REAL_C64)
INSTANTIATE_CR_RC_C64(eq, uint8_t, uint8, EXPR_EQ_C64_REAL, EXPR_EQ_REAL_C64)
INSTANTIATE_CR_RC_C64(eq, int16_t, int16, EXPR_EQ_C64_REAL, EXPR_EQ_REAL_C64)

// NE
DEFINE_COMPLEX_COMP_CC(ne, cpx_t, complex128, cpx_t, complex128, EXPR_NE_C128_C128)
DEFINE_COMPLEX_COMP_CC(ne, cpx_f_t, complex64, cpx_f_t, complex64, EXPR_NE_C64_C64)
DEFINE_COMPLEX_COMP_CC(ne, cpx_t, complex128, cpx_f_t, complex64, EXPR_NE_C128_C64)
DEFINE_COMPLEX_COMP_CC(ne, cpx_f_t, complex64, cpx_t, complex128, EXPR_NE_C64_C128)

INSTANTIATE_CR_RC_C128(ne, double, double, EXPR_NE_C128_REAL, EXPR_NE_REAL_C128)
INSTANTIATE_CR_RC_C128(ne, float, float, EXPR_NE_C128_REAL, EXPR_NE_REAL_C128)
INSTANTIATE_CR_RC_C128(ne, int32_t, int32, EXPR_NE_C128_REAL, EXPR_NE_REAL_C128)
INSTANTIATE_CR_RC_C128(ne, int64_t, int64, EXPR_NE_C128_REAL, EXPR_NE_REAL_C128)
INSTANTIATE_CR_RC_C128(ne, uint8_t, uint8, EXPR_NE_C128_REAL, EXPR_NE_REAL_C128)
INSTANTIATE_CR_RC_C128(ne, int16_t, int16, EXPR_NE_C128_REAL, EXPR_NE_REAL_C128)

INSTANTIATE_CR_RC_C64(ne, double, double, EXPR_NE_C64_REAL, EXPR_NE_REAL_C64)
INSTANTIATE_CR_RC_C64(ne, float, float, EXPR_NE_C64_REAL, EXPR_NE_REAL_C64)
INSTANTIATE_CR_RC_C64(ne, int32_t, int32, EXPR_NE_C64_REAL, EXPR_NE_REAL_C64)
INSTANTIATE_CR_RC_C64(ne, int64_t, int64, EXPR_NE_C64_REAL, EXPR_NE_REAL_C64)
INSTANTIATE_CR_RC_C64(ne, uint8_t, uint8, EXPR_NE_C64_REAL, EXPR_NE_REAL_C64)
INSTANTIATE_CR_RC_C64(ne, int16_t, int16, EXPR_NE_C64_REAL, EXPR_NE_REAL_C64)

#undef INSTANTIATE_CR_RC_C64
#undef INSTANTIATE_CR_RC_C128
#undef DEFINE_COMPLEX_COMP_RC
#undef DEFINE_COMPLEX_COMP_CR
#undef DEFINE_COMPLEX_COMP_CC

#define DISPATCH_COMPARE_B_REAL(OP_NAME, T1_NAME) \
    switch (dtypeB) { \
        case DTYPE_FLOAT64: s_compare_##OP_NAME##_##T1_NAME##_double(a, stridesA, b, stridesB, res, stridesRes, shape, rank); break; \
        case DTYPE_FLOAT32: s_compare_##OP_NAME##_##T1_NAME##_float(a, stridesA, b, stridesB, res, stridesRes, shape, rank); break; \
        case DTYPE_INT32: s_compare_##OP_NAME##_##T1_NAME##_int32(a, stridesA, b, stridesB, res, stridesRes, shape, rank); break; \
        case DTYPE_INT64: s_compare_##OP_NAME##_##T1_NAME##_int64(a, stridesA, b, stridesB, res, stridesRes, shape, rank); break; \
        case DTYPE_UINT8: s_compare_##OP_NAME##_##T1_NAME##_uint8(a, stridesA, b, stridesB, res, stridesRes, shape, rank); break; \
        case DTYPE_INT16: s_compare_##OP_NAME##_##T1_NAME##_int16(a, stridesA, b, stridesB, res, stridesRes, shape, rank); break; \
        case DTYPE_BOOLEAN: s_compare_##OP_NAME##_##T1_NAME##_uint8(a, stridesA, b, stridesB, res, stridesRes, shape, rank); break; \
    }

#define DISPATCH_COMPARE_B_ALL(OP_NAME, T1_NAME) \
    switch (dtypeB) { \
        case DTYPE_FLOAT64: s_compare_##OP_NAME##_##T1_NAME##_double(a, stridesA, b, stridesB, res, stridesRes, shape, rank); break; \
        case DTYPE_FLOAT32: s_compare_##OP_NAME##_##T1_NAME##_float(a, stridesA, b, stridesB, res, stridesRes, shape, rank); break; \
        case DTYPE_INT32: s_compare_##OP_NAME##_##T1_NAME##_int32(a, stridesA, b, stridesB, res, stridesRes, shape, rank); break; \
        case DTYPE_INT64: s_compare_##OP_NAME##_##T1_NAME##_int64(a, stridesA, b, stridesB, res, stridesRes, shape, rank); break; \
        case DTYPE_UINT8: s_compare_##OP_NAME##_##T1_NAME##_uint8(a, stridesA, b, stridesB, res, stridesRes, shape, rank); break; \
        case DTYPE_INT16: s_compare_##OP_NAME##_##T1_NAME##_int16(a, stridesA, b, stridesB, res, stridesRes, shape, rank); break; \
        case DTYPE_BOOLEAN: s_compare_##OP_NAME##_##T1_NAME##_uint8(a, stridesA, b, stridesB, res, stridesRes, shape, rank); break; \
        case DTYPE_COMPLEX128: s_compare_##OP_NAME##_##T1_NAME##_complex128(a, stridesA, b, stridesB, res, stridesRes, shape, rank); break; \
        case DTYPE_COMPLEX64: s_compare_##OP_NAME##_##T1_NAME##_complex64(a, stridesA, b, stridesB, res, stridesRes, shape, rank); break; \
    }

#define DISPATCH_COMPARE_A_REAL(OP_NAME) \
    switch (dtypeA) { \
        case DTYPE_FLOAT64: DISPATCH_COMPARE_B_REAL(OP_NAME, double); break; \
        case DTYPE_FLOAT32: DISPATCH_COMPARE_B_REAL(OP_NAME, float); break; \
        case DTYPE_INT32: DISPATCH_COMPARE_B_REAL(OP_NAME, int32); break; \
        case DTYPE_INT64: DISPATCH_COMPARE_B_REAL(OP_NAME, int64); break; \
        case DTYPE_UINT8: DISPATCH_COMPARE_B_REAL(OP_NAME, uint8); break; \
        case DTYPE_INT16: DISPATCH_COMPARE_B_REAL(OP_NAME, int16); break; \
        case DTYPE_BOOLEAN: DISPATCH_COMPARE_B_REAL(OP_NAME, uint8); break; \
    }

#define DISPATCH_COMPARE_A_ALL(OP_NAME) \
    switch (dtypeA) { \
        case DTYPE_FLOAT64: DISPATCH_COMPARE_B_ALL(OP_NAME, double); break; \
        case DTYPE_FLOAT32: DISPATCH_COMPARE_B_ALL(OP_NAME, float); break; \
        case DTYPE_INT32: DISPATCH_COMPARE_B_ALL(OP_NAME, int32); break; \
        case DTYPE_INT64: DISPATCH_COMPARE_B_ALL(OP_NAME, int64); break; \
        case DTYPE_UINT8: DISPATCH_COMPARE_B_ALL(OP_NAME, uint8); break; \
        case DTYPE_INT16: DISPATCH_COMPARE_B_ALL(OP_NAME, int16); break; \
        case DTYPE_BOOLEAN: DISPATCH_COMPARE_B_ALL(OP_NAME, uint8); break; \
        case DTYPE_COMPLEX128: DISPATCH_COMPARE_B_ALL(OP_NAME, complex128); break; \
        case DTYPE_COMPLEX64: DISPATCH_COMPARE_B_ALL(OP_NAME, complex64); break; \
    }

void ndarray_compare(
    int op, int dtypeA, int dtypeB,
    const void *a, const int *stridesA,
    const void *b, const int *stridesB,
    uint8_t *res, const int *stridesRes,
    const int *shape, int rank
) {
    switch (op) {
        case CMP_OP_EQ: DISPATCH_COMPARE_A_ALL(eq); break;
        case CMP_OP_NE: DISPATCH_COMPARE_A_ALL(ne); break;
        case CMP_OP_LT: DISPATCH_COMPARE_A_REAL(lt); break;
        case CMP_OP_LE: DISPATCH_COMPARE_A_REAL(le); break;
        case CMP_OP_GT: DISPATCH_COMPARE_A_REAL(gt); break;
        case CMP_OP_GE: DISPATCH_COMPARE_A_REAL(ge); break;
    }
}

#undef DISPATCH_COMPARE_A_ALL
#undef DISPATCH_COMPARE_A_REAL
#undef DISPATCH_COMPARE_B_ALL
#undef DISPATCH_COMPARE_B_REAL

// Structural Equality
#define DEFINE_EQUALS_FUNC(NAME, TYPE, EXPR) \
int NAME(const TYPE *a, const int *stridesA, \
         const TYPE *b, const int *stridesB, \
         const int *shape, int rank) { \
    if (a == NULL || b == NULL || rank < 0 || rank > 8) return 0; \
    if (rank == 0) return EXPR(a[0], b[0]) ? 1 : 0; \
    int total_elements = 1; \
    for (int i = 0; i < rank; i++) total_elements *= shape[i]; \
    int coord[8] = {0}; \
    int offsetA = 0, offsetB = 0; \
    for (int el = 0; el < total_elements; el++) { \
        if (!(EXPR(a[offsetA], b[offsetB]))) return 0; \
        for (int d = rank - 1; d >= 0; d--) { \
            coord[d]++; \
            if (coord[d] < shape[d]) { \
                offsetA += stridesA[d]; \
                offsetB += stridesB[d]; \
                break; \
            } \
            coord[d] = 0; \
            offsetA -= (shape[d] - 1) * stridesA[d]; \
            offsetB -= (shape[d] - 1) * stridesB[d]; \
        } \
    } \
    return 1; \
}

DEFINE_EQUALS_FUNC(s_equals_double, double, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_float, float, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_int32, int32_t, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_int64, int64_t, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_uint8, uint8_t, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_int16, int16_t, EXPR_EQ)
DEFINE_EQUALS_FUNC(s_equals_complex128, cpx_t, EXPR_EQ_C128_C128)
DEFINE_EQUALS_FUNC(s_equals_complex64, cpx_f_t, EXPR_EQ_C64_C64)

#undef DEFINE_EQUALS_FUNC

int ndarray_equals(
    int dtype,
    const void *a, const int *stridesA,
    const void *b, const int *stridesB,
    const int *shape, int rank
) {
    switch (dtype) {
        case DTYPE_FLOAT64: return s_equals_double((const double*)a, stridesA, (const double*)b, stridesB, shape, rank);
        case DTYPE_FLOAT32: return s_equals_float((const float*)a, stridesA, (const float*)b, stridesB, shape, rank);
        case DTYPE_INT32: return s_equals_int32((const int32_t*)a, stridesA, (const int32_t*)b, stridesB, shape, rank);
        case DTYPE_INT64: return s_equals_int64((const int64_t*)a, stridesA, (const int64_t*)b, stridesB, shape, rank);
        case DTYPE_UINT8: return s_equals_uint8((const uint8_t*)a, stridesA, (const uint8_t*)b, stridesB, shape, rank);
        case DTYPE_INT16: return s_equals_int16((const int16_t*)a, stridesA, (const int16_t*)b, stridesB, shape, rank);
        case DTYPE_COMPLEX128: return s_equals_complex128((const cpx_t*)a, stridesA, (const cpx_t*)b, stridesB, shape, rank);
        case DTYPE_COMPLEX64: return s_equals_complex64((const cpx_f_t*)a, stridesA, (const cpx_f_t*)b, stridesB, shape, rank);
        case DTYPE_BOOLEAN: return s_equals_uint8((const uint8_t*)a, stridesA, (const uint8_t*)b, stridesB, shape, rank);
        default: return 0;
    }
}

// Reduction Min/Max

#define DEFINE_R_MINMAX(NAME, TYPE, OP) \
TYPE r_##NAME##_##TYPE(const TYPE *src, int size) { \
    if (src == NULL || size <= 0) return (TYPE)0; \
    TYPE acc = src[0]; \
    for (int i = 1; i < size; i++) { \
        if (src[i] OP acc) acc = src[i]; \
    } \
    return acc; \
}

DEFINE_R_MINMAX(min, double, <)
DEFINE_R_MINMAX(min, float, <)
DEFINE_R_MINMAX(min, int64_t, <)
DEFINE_R_MINMAX(min, int32_t, <)
DEFINE_R_MINMAX(min, uint8_t, <)
DEFINE_R_MINMAX(min, int16_t, <)

DEFINE_R_MINMAX(max, double, >)
DEFINE_R_MINMAX(max, float, >)
DEFINE_R_MINMAX(max, int64_t, >)
DEFINE_R_MINMAX(max, int32_t, >)
DEFINE_R_MINMAX(max, uint8_t, >)
DEFINE_R_MINMAX(max, int16_t, >)

#undef DEFINE_R_MINMAX

// Strided Reduction Min/Max

#define DEFINE_S_MINMAX(NAME, TYPE, OP) \
void s_##NAME##_##TYPE(const TYPE *src, const int *stridesSrc, \
                       TYPE *dest, const int *stridesDest, \
                       const int *shape, int rank, int axis) { \
    if (src == NULL || dest == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return; \
    int size_axis = shape[axis]; \
    if (size_axis <= 0) return; \
    int coord[8] = {0}; \
    int outer_size = 1; \
    for (int d = 0; d < rank; d++) { \
        if (d != axis) outer_size *= shape[d]; \
    } \
    for (int o = 0; o < outer_size; o++) { \
        int offsetRes = 0; \
        int offsetSrc = 0; \
        for (int d = 0; d < rank; d++) { \
            if (d != axis) { \
                offsetSrc += coord[d] * stridesSrc[d]; \
                if (rank > 1) { \
                    int targetD = (d < axis) ? d : (d - 1); \
                    offsetRes += coord[d] * stridesDest[targetD]; \
                } \
            } \
        } \
        TYPE val_acc = src[offsetSrc]; \
        int stride_axis = stridesSrc[axis]; \
        for (int i = 1; i < size_axis; i++) { \
            TYPE val = src[offsetSrc + i * stride_axis]; \
            if (val OP val_acc) val_acc = val; \
        } \
        dest[offsetRes] = val_acc; \
        for (int d = rank - 1; d >= 0; d--) { \
            if (d == axis) continue; \
            coord[d]++; \
            if (coord[d] < shape[d]) break; \
            coord[d] = 0; \
        } \
    } \
}

DEFINE_S_MINMAX(min, double, <)
DEFINE_S_MINMAX(min, float, <)
DEFINE_S_MINMAX(min, int64_t, <)
DEFINE_S_MINMAX(min, int32_t, <)
DEFINE_S_MINMAX(min, uint8_t, <)
DEFINE_S_MINMAX(min, int16_t, <)

DEFINE_S_MINMAX(max, double, >)
DEFINE_S_MINMAX(max, float, >)
DEFINE_S_MINMAX(max, int64_t, >)
DEFINE_S_MINMAX(max, int32_t, >)
DEFINE_S_MINMAX(max, uint8_t, >)
DEFINE_S_MINMAX(max, int16_t, >)

#undef DEFINE_S_MINMAX

// Nanmin/Nanmax Reduction (float/double only)

#define DEFINE_R_NANMINMAX(NAME, TYPE, OP) \
TYPE r_##NAME##_##TYPE(const TYPE *src, int size) { \
    if (src == NULL || size <= 0) return (TYPE)NAN; \
    TYPE acc = (TYPE)NAN; \
    int i = 0; \
    for (; i < size; i++) { \
        if (!isnan(src[i])) { \
            acc = src[i]; \
            break; \
        } \
    } \
    if (isnan(acc)) return (TYPE)NAN; \
    for (; i < size; i++) { \
        if (!isnan(src[i]) && src[i] OP acc) { \
            acc = src[i]; \
        } \
    } \
    return acc; \
}

DEFINE_R_NANMINMAX(nanmin, double, <)
DEFINE_R_NANMINMAX(nanmin, float, <)
DEFINE_R_NANMINMAX(nanmax, double, >)
DEFINE_R_NANMINMAX(nanmax, float, >)

#undef DEFINE_R_NANMINMAX

// Strided Nanmin/Nanmax Reduction (float/double only)

#define DEFINE_S_NANMINMAX(NAME, TYPE, OP) \
void s_##NAME##_##TYPE(const TYPE *src, const int *stridesSrc, \
                       TYPE *dest, const int *stridesDest, \
                       const int *shape, int rank, int axis) { \
    if (src == NULL || dest == NULL || shape == NULL || rank <= 0 || axis < 0 || axis >= rank) return; \
    int size_axis = shape[axis]; \
    if (size_axis <= 0) return; \
    int coord[8] = {0}; \
    int outer_size = 1; \
    for (int d = 0; d < rank; d++) { \
        if (d != axis) outer_size *= shape[d]; \
    } \
    for (int o = 0; o < outer_size; o++) { \
        int offsetRes = 0; \
        int offsetSrc = 0; \
        for (int d = 0; d < rank; d++) { \
            if (d != axis) { \
                offsetSrc += coord[d] * stridesSrc[d]; \
                if (rank > 1) { \
                    int targetD = (d < axis) ? d : (d - 1); \
                    offsetRes += coord[d] * stridesDest[targetD]; \
                } \
            } \
        } \
        TYPE val_acc = (TYPE)NAN; \
        int stride_axis = stridesSrc[axis]; \
        int i = 0; \
        for (; i < size_axis; i++) { \
            TYPE val = src[offsetSrc + i * stride_axis]; \
            if (!isnan(val)) { \
                val_acc = val; \
                break; \
            } \
        } \
        if (!isnan(val_acc)) { \
            for (; i < size_axis; i++) { \
                TYPE val = src[offsetSrc + i * stride_axis]; \
                if (!isnan(val) && val OP val_acc) { \
                    val_acc = val; \
                } \
            } \
        } \
        dest[offsetRes] = val_acc; \
        for (int d = rank - 1; d >= 0; d--) { \
            if (d == axis) continue; \
            coord[d]++; \
            if (coord[d] < shape[d]) break; \
            coord[d] = 0; \
        } \
    } \
}

DEFINE_S_NANMINMAX(nanmin, double, <)
DEFINE_S_NANMINMAX(nanmin, float, <)
DEFINE_S_NANMINMAX(nanmax, double, >)
DEFINE_S_NANMINMAX(nanmax, float, >)

#undef DEFINE_S_NANMINMAX

// Find Index
#define DEFINE_FIND_INDEX_FUNC(NAME, TYPE) \
int s_find_index_##NAME(const TYPE *a, const int *stridesA, \
                        const int *shape, int rank, \
                        int op, TYPE target, \
                        const int *startCoords, const int *directions, \
                        int *matchCoords) { \
    if (a == NULL || rank < 0 || rank > 8) return 0; \
    int total_elements = 1; \
    for (int i = 0; i < rank; i++) total_elements *= shape[i]; \
    if (total_elements <= 0) return 0; \
    if (rank == 0) { \
        TYPE val = a[0]; \
        int match = 0; \
        switch(op) { \
            case CMP_OP_EQ: match = (val == target); break; \
            case CMP_OP_NE: match = (val != target); break; \
            case CMP_OP_LT: match = (val < target); break; \
            case CMP_OP_LE: match = (val <= target); break; \
            case CMP_OP_GT: match = (val > target); break; \
            case CMP_OP_GE: match = (val >= target); break; \
        } \
        return match ? 1 : 0; \
    } \
    int coord[8]; \
    for (int i = 0; i < rank; i++) { \
        if (startCoords[i] < 0 || startCoords[i] >= shape[i]) return 0; \
        coord[i] = startCoords[i]; \
    } \
    int offsetA = 0; \
    for (int i = 0; i < rank; i++) { \
        offsetA += coord[i] * stridesA[i]; \
    } \
    while (1) { \
        TYPE val = a[offsetA]; \
        int match = 0; \
        switch(op) { \
            case CMP_OP_EQ: match = (val == target); break; \
            case CMP_OP_NE: match = (val != target); break; \
            case CMP_OP_LT: match = (val < target); break; \
            case CMP_OP_LE: match = (val <= target); break; \
            case CMP_OP_GT: match = (val > target); break; \
            case CMP_OP_GE: match = (val >= target); break; \
        } \
        if (match) { \
            if (matchCoords != NULL) { \
                for (int i = 0; i < rank; i++) { \
                    matchCoords[i] = coord[i]; \
                } \
            } \
            return 1; \
        } \
        int finished = 0; \
        for (int d = rank - 1; d >= 0; d--) { \
            if (directions[d] >= 0) { \
                coord[d]++; \
                if (coord[d] < shape[d]) { \
                    offsetA += stridesA[d]; \
                    break; \
                } \
                if (d == 0) { \
                    finished = 1; \
                    break; \
                } \
                coord[d] = 0; \
                offsetA -= (shape[d] - 1) * stridesA[d]; \
            } else { \
                coord[d]--; \
                if (coord[d] >= 0) { \
                    offsetA -= stridesA[d]; \
                    break; \
                } \
                if (d == 0) { \
                    finished = 1; \
                    break; \
                } \
                coord[d] = shape[d] - 1; \
                offsetA += (shape[d] - 1) * stridesA[d]; \
            } \
        } \
        if (finished) break; \
    } \
    return 0; \
}

DEFINE_FIND_INDEX_FUNC(double, double)
DEFINE_FIND_INDEX_FUNC(float, float)
DEFINE_FIND_INDEX_FUNC(int32, int32_t)
DEFINE_FIND_INDEX_FUNC(int64, int64_t)
DEFINE_FIND_INDEX_FUNC(uint8, uint8_t)
DEFINE_FIND_INDEX_FUNC(int16, int16_t)

#undef DEFINE_FIND_INDEX_FUNC

int s_find_index_complex128(const cpx_t *a, const int *stridesA,
                             const int *shape, int rank,
                             int op, cpx_t target,
                             const int *startCoords, const int *directions,
                             int *matchCoords) {
    if (a == NULL || rank < 0 || rank > 8) return 0;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    if (total_elements <= 0) return 0;
    if (rank == 0) {
        cpx_t val = a[0];
        int match = 0;
        if (op == CMP_OP_EQ) {
            match = (val.r == target.r && val.i == target.i);
        } else if (op == CMP_OP_NE) {
            match = (val.r != target.r || val.i != target.i);
        }
        return match ? 1 : 0;
    }
    int coord[8];
    for (int i = 0; i < rank; i++) {
        if (startCoords[i] < 0 || startCoords[i] >= shape[i]) return 0;
        coord[i] = startCoords[i];
    }
    int offsetA = 0;
    for (int i = 0; i < rank; i++) {
        offsetA += coord[i] * stridesA[i];
    }
    while (1) {
        cpx_t val = a[offsetA];
        int match = 0;
        if (op == CMP_OP_EQ) {
            match = (val.r == target.r && val.i == target.i);
        } else if (op == CMP_OP_NE) {
            match = (val.r != target.r || val.i != target.i);
        }
        if (match) {
            if (matchCoords != NULL) {
                for (int i = 0; i < rank; i++) {
                    matchCoords[i] = coord[i];
                }
            }
            return 1;
        }
        int finished = 0;
        for (int d = rank - 1; d >= 0; d--) {
            if (directions[d] >= 0) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetA += stridesA[d];
                    break;
                }
                if (d == 0) {
                    finished = 1;
                    break;
                }
                coord[d] = 0;
                offsetA -= (shape[d] - 1) * stridesA[d];
            } else {
                coord[d]--;
                if (coord[d] >= 0) {
                    offsetA -= stridesA[d];
                    break;
                }
                if (d == 0) {
                    finished = 1;
                    break;
                }
                coord[d] = shape[d] - 1;
                offsetA += (shape[d] - 1) * stridesA[d];
            }
        }
        if (finished) break;
    }
    return 0;
}

int s_find_index_complex64(const cpx_f_t *a, const int *stridesA,
                            const int *shape, int rank,
                            int op, cpx_f_t target,
                            const int *startCoords, const int *directions,
                            int *matchCoords) {
    if (a == NULL || rank < 0 || rank > 8) return 0;
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];
    if (total_elements <= 0) return 0;
    if (rank == 0) {
        cpx_f_t val = a[0];
        int match = 0;
        if (op == CMP_OP_EQ) {
            match = (val.r == target.r && val.i == target.i);
        } else if (op == CMP_OP_NE) {
            match = (val.r != target.r || val.i != target.i);
        }
        return match ? 1 : 0;
    }
    int coord[8];
    for (int i = 0; i < rank; i++) {
        if (startCoords[i] < 0 || startCoords[i] >= shape[i]) return 0;
        coord[i] = startCoords[i];
    }
    int offsetA = 0;
    for (int i = 0; i < rank; i++) {
        offsetA += coord[i] * stridesA[i];
    }
    while (1) {
        cpx_f_t val = a[offsetA];
        int match = 0;
        if (op == CMP_OP_EQ) {
            match = (val.r == target.r && val.i == target.i);
        } else if (op == CMP_OP_NE) {
            match = (val.r != target.r || val.i != target.i);
        }
        if (match) {
            if (matchCoords != NULL) {
                for (int i = 0; i < rank; i++) {
                    matchCoords[i] = coord[i];
                }
            }
            return 1;
        }
        int finished = 0;
        for (int d = rank - 1; d >= 0; d--) {
            if (directions[d] >= 0) {
                coord[d]++;
                if (coord[d] < shape[d]) {
                    offsetA += stridesA[d];
                    break;
                }
                if (d == 0) {
                    finished = 1;
                    break;
                }
                coord[d] = 0;
                offsetA -= (shape[d] - 1) * stridesA[d];
            } else {
                coord[d]--;
                if (coord[d] >= 0) {
                    offsetA -= stridesA[d];
                    break;
                }
                if (d == 0) {
                    finished = 1;
                    break;
                }
                coord[d] = shape[d] - 1;
                offsetA += (shape[d] - 1) * stridesA[d];
            }
        }
        if (finished) break;
    }
    return 0;
}

int ndarray_find_index(
    int op, int dtype,
    const void *a, const int *stridesA,
    const int *shape, int rank,
    const void *target,
    const int *startCoords,
    const int *directions,
    int *matchCoords
) {
    if (a == NULL || target == NULL) return 0;
    switch (dtype) {
        case DTYPE_FLOAT64:
            return s_find_index_double((const double*)a, stridesA, shape, rank, op, *(const double*)target, startCoords, directions, matchCoords);
        case DTYPE_FLOAT32:
            return s_find_index_float((const float*)a, stridesA, shape, rank, op, *(const float*)target, startCoords, directions, matchCoords);
        case DTYPE_INT32:
            return s_find_index_int32((const int32_t*)a, stridesA, shape, rank, op, *(const int32_t*)target, startCoords, directions, matchCoords);
        case DTYPE_INT64:
            return s_find_index_int64((const int64_t*)a, stridesA, shape, rank, op, *(const int64_t*)target, startCoords, directions, matchCoords);
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            return s_find_index_uint8((const uint8_t*)a, stridesA, shape, rank, op, *(const uint8_t*)target, startCoords, directions, matchCoords);
        case DTYPE_INT16:
            return s_find_index_int16((const int16_t*)a, stridesA, shape, rank, op, *(const int16_t*)target, startCoords, directions, matchCoords);
        case DTYPE_COMPLEX128:
            return s_find_index_complex128((const cpx_t*)a, stridesA, shape, rank, op, *(const cpx_t*)target, startCoords, directions, matchCoords);
        case DTYPE_COMPLEX64:
            return s_find_index_complex64((const cpx_f_t*)a, stridesA, shape, rank, op, *(const cpx_f_t*)target, startCoords, directions, matchCoords);
        default:
            return 0;
    }
}

#define COMPUTE_EUCLIDEAN(type, u, strideU, v, strideV, N, result) do { \
    double sum = 0.0; \
    for (int i = 0; i < N; i++) { \
        double diff = (double)((const type*)u)[i * strideU] - (double)((const type*)v)[i * strideV]; \
        sum += diff * diff; \
    } \
    result = sqrt(sum); \
} while(0)

#define COMPUTE_COSINE(type, u, strideU, v, strideV, N, result) do { \
    double dot = 0.0; \
    double norm_u = 0.0; \
    double norm_v = 0.0; \
    for (int i = 0; i < N; i++) { \
        double val_u = (double)((const type*)u)[i * strideU]; \
        double val_v = (double)((const type*)v)[i * strideV]; \
        dot += val_u * val_v; \
        norm_u += val_u * val_u; \
        norm_v += val_v * val_v; \
    } \
    if (norm_u == 0.0 || norm_v == 0.0) { \
        result = NAN; \
    } else { \
        result = 1.0 - (dot / (sqrt(norm_u) * sqrt(norm_v))); \
    } \
} while(0)

#define COMPUTE_HAMMING(type, u, strideU, v, strideV, N, result) do { \
    if (N == 0) { \
        result = NAN; \
    } else { \
        double diff_count = 0.0; \
        for (int i = 0; i < N; i++) { \
            if (((const type*)u)[i * strideU] != ((const type*)v)[i * strideV]) { \
                diff_count += 1.0; \
            } \
        } \
        result = diff_count / (double)N; \
    } \
} while(0)

#define COMPUTE_DIST(type, u, strideU, v, strideV, N, metric, result) do { \
    switch(metric) { \
        case METRIC_EUCLIDEAN: COMPUTE_EUCLIDEAN(type, u, strideU, v, strideV, N, result); break; \
        case METRIC_COSINE: COMPUTE_COSINE(type, u, strideU, v, strideV, N, result); break; \
        case METRIC_HAMMING: COMPUTE_HAMMING(type, u, strideU, v, strideV, N, result); break; \
        default: result = NAN; break; \
    } \
} while(0)

#define DEFINE_PDIST(type) \
static void pdist_##type(const type *x, int M, int N, int strideRowX, int strideColX, int metric, double *out, int strideOut) { \
    int idx = 0; \
    for (int i = 0; i < M; i++) { \
        const type *u = x + i * strideRowX; \
        for (int j = i + 1; j < M; j++) { \
            const type *v = x + j * strideRowX; \
            double res; \
            COMPUTE_DIST(type, u, strideColX, v, strideColX, N, metric, res); \
            out[idx * strideOut] = res; \
            idx++; \
        } \
    } \
}

DEFINE_PDIST(double)
DEFINE_PDIST(float)
DEFINE_PDIST(int64_t)
DEFINE_PDIST(int32_t)
DEFINE_PDIST(int16_t)
DEFINE_PDIST(uint8_t)

#undef DEFINE_PDIST

void ndarray_pdist(
    int dtype,
    const void *x,
    int M, int N,
    int strideRowX, int strideColX,
    int metric,
    double *out,
    int strideOut
) {
    if (x == NULL || out == NULL || M < 2 || N < 0) return;
    switch(dtype) {
        case DTYPE_FLOAT64:
            pdist_double((const double*)x, M, N, strideRowX, strideColX, metric, out, strideOut);
            break;
        case DTYPE_FLOAT32:
            pdist_float((const float*)x, M, N, strideRowX, strideColX, metric, out, strideOut);
            break;
        case DTYPE_INT32:
            pdist_int32_t((const int32_t*)x, M, N, strideRowX, strideColX, metric, out, strideOut);
            break;
        case DTYPE_INT64:
            pdist_int64_t((const int64_t*)x, M, N, strideRowX, strideColX, metric, out, strideOut);
            break;
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            pdist_uint8_t((const uint8_t*)x, M, N, strideRowX, strideColX, metric, out, strideOut);
            break;
        case DTYPE_INT16:
            pdist_int16_t((const int16_t*)x, M, N, strideRowX, strideColX, metric, out, strideOut);
            break;
        default:
            break;
    }
}

#define DEFINE_CDIST(type) \
static void cdist_##type(const type *xa, const type *xb, int M, int K, int N, \
                  int strideRowXA, int strideColXA, \
                  int strideRowXB, int strideColXB, \
                  int metric, \
                  double *out, int strideRowOut, int strideColOut) { \
    for (int i = 0; i < M; i++) { \
        const type *u = xa + i * strideRowXA; \
        for (int j = 0; j < K; j++) { \
            const type *v = xb + j * strideRowXB; \
            double res; \
            COMPUTE_DIST(type, u, strideColXA, v, strideColXB, N, metric, res); \
            out[i * strideRowOut + j * strideColOut] = res; \
        } \
    } \
}

DEFINE_CDIST(double)
DEFINE_CDIST(float)
DEFINE_CDIST(int64_t)
DEFINE_CDIST(int32_t)
DEFINE_CDIST(int16_t)
DEFINE_CDIST(uint8_t)

#undef DEFINE_CDIST

void ndarray_cdist(
    int dtype,
    const void *xa, const void *xb,
    int M, int K, int N,
    int strideRowXA, int strideColXA,
    int strideRowXB, int strideColXB,
    int metric,
    double *out,
    int strideRowOut, int strideColOut
) {
    if (xa == NULL || xb == NULL || out == NULL || M <= 0 || K <= 0 || N < 0) return;
    switch(dtype) {
        case DTYPE_FLOAT64:
            cdist_double((const double*)xa, (const double*)xb, M, K, N, strideRowXA, strideColXA, strideRowXB, strideColXB, metric, out, strideRowOut, strideColOut);
            break;
        case DTYPE_FLOAT32:
            cdist_float((const float*)xa, (const float*)xb, M, K, N, strideRowXA, strideColXA, strideRowXB, strideColXB, metric, out, strideRowOut, strideColOut);
            break;
        case DTYPE_INT32:
            cdist_int32_t((const int32_t*)xa, (const int32_t*)xb, M, K, N, strideRowXA, strideColXA, strideRowXB, strideColXB, metric, out, strideRowOut, strideColOut);
            break;
        case DTYPE_INT64:
            cdist_int64_t((const int64_t*)xa, (const int64_t*)xb, M, K, N, strideRowXA, strideColXA, strideRowXB, strideColXB, metric, out, strideRowOut, strideColOut);
            break;
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            cdist_uint8_t((const uint8_t*)xa, (const uint8_t*)xb, M, K, N, strideRowXA, strideColXA, strideRowXB, strideColXB, metric, out, strideRowOut, strideColOut);
            break;
        case DTYPE_INT16:
            cdist_int16_t((const int16_t*)xa, (const int16_t*)xb, M, K, N, strideRowXA, strideColXA, strideRowXB, strideColXB, metric, out, strideRowOut, strideColOut);
            break;
        default:
            break;
    }
}
