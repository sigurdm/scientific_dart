#include "custom_ufuncs.h"
#include <math.h>
#include <stdlib.h>

// ============================================================================
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
            offsetRes  -= (shape[d] - 1) * stridesRes[d];
        }
    }
}

void v_normal_double(double *res, int size, double loc, double scale, unsigned long long seed) {
    if (res == NULL || size <= 0 || scale <= 0.0) return;

    unsigned long long state = seed ^ 0x5555555555555555ULL;

    int i = 0;
    while (i < size) {
        double u1;
        do {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            u1 = (double)(state >> 11) * (1.0 / 9007199254740992.0);
        } while (u1 == 0.0);

        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        double u2 = (double)(state >> 11) * (1.0 / 9007199254740992.0);

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

    unsigned long long state = seed ^ 0x5555555555555555ULL;

    int i = 0;
    while (i < size) {
        float u1;
        do {
            state = state * 6364136223846793005ULL + 1442695040888963407ULL;
            u1 = (float)((double)(state >> 11) * (1.0 / 9007199254740992.0));
        } while (u1 == 0.0f);

        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        float u2 = (float)((double)(state >> 11) * (1.0 / 9007199254740992.0));

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

    unsigned long long state = seed ^ 0x5555555555555555ULL;

    for (int i = 0; i < size; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        res[i] = (double)(state >> 11) * (1.0 / 9007199254740992.0);
    }
}

void v_uniform_float(float *res, int size, unsigned long long seed) {
    if (res == NULL || size <= 0) return;

    unsigned long long state = seed ^ 0x5555555555555555ULL;

    for (int i = 0; i < size; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        res[i] = (float)((double)(state >> 11) * (1.0 / 9007199254740992.0));
    }
}

void v_randint_int64(int64_t *res, int size, int64_t low, int64_t high, unsigned long long seed) {
    if (res == NULL || size <= 0 || low >= high) return;

    unsigned long long state = seed ^ 0x5555555555555555ULL;
    int64_t range = high - low;

    for (int i = 0; i < size; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        res[i] = low + (int64_t)(state % (unsigned long long)range);
    }
}

void v_randint_int32(int32_t *res, int size, int32_t low, int32_t high, unsigned long long seed) {
    if (res == NULL || size <= 0 || low >= high) return;

    unsigned long long state = seed ^ 0x5555555555555555ULL;
    int32_t range = high - low;

    for (int i = 0; i < size; i++) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        res[i] = low + (int32_t)(state % (unsigned long long)range);
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
