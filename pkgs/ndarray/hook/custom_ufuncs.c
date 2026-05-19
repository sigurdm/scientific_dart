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

void v_add_double(const double *a, const double *b, double *res, int size) {
    if (a == NULL || b == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = a[i] + b[i];
    }
}

void v_sub_double(const double *a, const double *b, double *res, int size) {
    if (a == NULL || b == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = a[i] - b[i];
    }
}

void v_mul_double(const double *a, const double *b, double *res, int size) {
    if (a == NULL || b == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = a[i] * b[i];
    }
}

void v_div_double(const double *a, const double *b, double *res, int size) {
    if (a == NULL || b == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = a[i] / b[i];
    }
}

void v_sin_double(const double *src, double *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = sin(src[i]);
    }
}

void v_cos_double(const double *src, double *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = cos(src[i]);
    }
}

void v_exp_double(const double *src, double *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = exp(src[i]);
    }
}

void v_log_double(const double *src, double *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = log(src[i]);
    }
}

void v_sinh_double(const double *src, double *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = sinh(src[i]);
    }
}

void v_cosh_double(const double *src, double *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = cosh(src[i]);
    }
}

void v_tanh_double(const double *src, double *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = tanh(src[i]);
    }
}

void v_asinh_double(const double *src, double *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = asinh(src[i]);
    }
}

void v_acosh_double(const double *src, double *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = acosh(src[i]);
    }
}

void v_atanh_double(const double *src, double *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = atanh(src[i]);
    }
}

void v_asin_double(const double *src, double *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = asin(src[i]);
    }
}

void v_acos_double(const double *src, double *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = acos(src[i]);
    }
}

void v_atan_double(const double *src, double *res, int size) {
    if (src == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = atan(src[i]);
    }
}

void v_atan2_double(const double *y, const double *x, double *res, int size) {
    if (y == NULL || x == NULL || res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = atan2(y[i], x[i]);
    }
}


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

void s_where_double(const unsigned char *cond, const int *stridesCond,
                    const double *x, const int *stridesX,
                    const double *y, const int *stridesY,
                    double *res, const int *stridesRes,
                    const int *shape, int rank) {
    if (cond == NULL || x == NULL || y == NULL || res == NULL || rank <= 0 || rank > 8) return;

    int is_contiguous = 1;
    int expected_stride = 1;
    for (int i = rank - 1; i >= 0; i--) {
        if (stridesCond[i] != expected_stride || 
            stridesX[i] != expected_stride || 
            stridesY[i] != expected_stride || 
            stridesRes[i] != expected_stride) {
            is_contiguous = 0;
            break;
        }
        expected_stride *= shape[i];
    }

    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    if (is_contiguous) {
        for (int i = 0; i < total_elements; i++) {
            res[i] = cond[i] ? x[i] : y[i];
        }
        return;
    }

    int coord[8] = {0};
    int offsetCond = 0, offsetX = 0, offsetY = 0, offsetRes = 0;

    for (int el = 0; el < total_elements; el++) {
        res[offsetRes] = cond[offsetCond] ? x[offsetX] : y[offsetY];

        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetCond += stridesCond[d];
                offsetX    += stridesX[d];
                offsetY    += stridesY[d];
                offsetRes  += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetCond -= (shape[d] - 1) * stridesCond[d];
            offsetX    -= (shape[d] - 1) * stridesX[d];
            offsetY    -= (shape[d] - 1) * stridesY[d];
            offsetRes  -= (shape[d] - 1) * stridesRes[d];
        }
    }
}

void s_where_float(const unsigned char *cond, const int *stridesCond,
                   const float *x, const int *stridesX,
                   const float *y, const int *stridesY,
                   float *res, const int *stridesRes,
                   const int *shape, int rank) {
    if (cond == NULL || x == NULL || y == NULL || res == NULL || rank <= 0 || rank > 8) return;

    int is_contiguous = 1;
    int expected_stride = 1;
    for (int i = rank - 1; i >= 0; i--) {
        if (stridesCond[i] != expected_stride || 
            stridesX[i] != expected_stride || 
            stridesY[i] != expected_stride || 
            stridesRes[i] != expected_stride) {
            is_contiguous = 0;
            break;
        }
        expected_stride *= shape[i];
    }

    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    if (is_contiguous) {
        for (int i = 0; i < total_elements; i++) {
            res[i] = cond[i] ? x[i] : y[i];
        }
        return;
    }

    int coord[8] = {0};
    int offsetCond = 0, offsetX = 0, offsetY = 0, offsetRes = 0;

    for (int el = 0; el < total_elements; el++) {
        res[offsetRes] = cond[offsetCond] ? x[offsetX] : y[offsetY];

        for (int d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < shape[d]) {
                offsetCond += stridesCond[d];
                offsetX    += stridesX[d];
                offsetY    += stridesY[d];
                offsetRes  += stridesRes[d];
                break;
            }
            coord[d] = 0;
            offsetCond -= (shape[d] - 1) * stridesCond[d];
            offsetX    -= (shape[d] - 1) * stridesX[d];
            offsetY    -= (shape[d] - 1) * stridesY[d];
        }
    }
}

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

void v_randint_int64(int64_t *res, int size, int64_t low, int64_t high, unsigned long long seed) {
    if (res == NULL || size <= 0 || low >= high) return;

    uint64_t s[4];
    xoshiro256_seed(seed, s);
    int64_t range = high - low;

    for (int i = 0; i < size; i++) {
        res[i] = low + (int64_t)(xoshiro256_next(s) % (unsigned long long)range);
    }
}

void v_randint_int32(int32_t *res, int size, int32_t low, int32_t high, unsigned long long seed) {
    if (res == NULL || size <= 0 || low >= high) return;

    uint64_t s[4];
    xoshiro256_seed(seed, s);
    int32_t range = high - low;

    for (int i = 0; i < size; i++) {
        res[i] = low + (int32_t)(xoshiro256_next(s) % (unsigned long long)range);
    }
}

void v_fill_double(double *res, double value, int size) {
    if (res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = value;
    }
}

void v_fill_float(float *res, float value, int size) {
    if (res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = value;
    }
}

void v_fill_int64(int64_t *res, int64_t value, int size) {
    if (res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = value;
    }
}

void v_fill_int32(int32_t *res, int32_t value, int size) {
    if (res == NULL || size <= 0) return;
    for (int i = 0; i < size; i++) {
        res[i] = value;
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

void s_flatten_boolean(const uint8_t *src, const int *stridesSrc, uint8_t *dest, const int *shape, int rank) {
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

void v_secure_randint_int64(int64_t *res, int size, int64_t low, int64_t high) {
    if (res == NULL || size <= 0 || low >= high) return;
    fill_secure_bytes(res, size * sizeof(int64_t));
    int64_t range = high - low;
    for (int i = 0; i < size; i++) {
        res[i] = low + (int64_t)((unsigned long long)res[i] % (unsigned long long)range);
    }
}

void v_secure_randint_int32(int32_t *res, int size, int32_t low, int32_t high) {
    if (res == NULL || size <= 0 || low >= high) return;
    fill_secure_bytes(res, size * sizeof(int32_t));
    int32_t range = high - low;
    for (int i = 0; i < size; i++) {
        res[i] = low + (int32_t)((unsigned int)res[i] % (unsigned int)range);
    }
}

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



