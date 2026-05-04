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
    double acc = 0.0;
    for (int i = 0; i < size; i++) {
        acc += src[i];
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
    
    int coord[8] = {0}; // Support up to rank 8 tensors
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    for (int el = 0; el < total_elements; el++) {
        // Calculate FFI unmanaged cell heap byte pointer offsets from strides
        int offsetA = 0, offsetB = 0, offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetA += coord[d] * stridesA[d];
            offsetB += coord[d] * stridesB[d];
            offsetRes += coord[d] * stridesRes[d];
        }

        res[offsetRes] = a[offsetA] + b[offsetB];

        // Advance multidimensional coordinate walk
        ADVANCE_ODOMETER_LOOP
    }
}

void s_sub_double(const double *a, const int *stridesA,
                  const double *b, const int *stridesB,
                  double *res, const int *stridesRes,
                  const int *shape, int rank) {
    if (a == NULL || b == NULL || res == NULL || rank <= 0 || rank > 8) return;
    int coord[8] = {0};
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    for (int el = 0; el < total_elements; el++) {
        int offsetA = 0, offsetB = 0, offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetA += coord[d] * stridesA[d];
            offsetB += coord[d] * stridesB[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        res[offsetRes] = a[offsetA] - b[offsetB];
        ADVANCE_ODOMETER_LOOP
    }
}

void s_mul_double(const double *a, const int *stridesA,
                  const double *b, const int *stridesB,
                  double *res, const int *stridesRes,
                  const int *shape, int rank) {
    if (a == NULL || b == NULL || res == NULL || rank <= 0 || rank > 8) return;
    int coord[8] = {0};
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    for (int el = 0; el < total_elements; el++) {
        int offsetA = 0, offsetB = 0, offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetA += coord[d] * stridesA[d];
            offsetB += coord[d] * stridesB[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        res[offsetRes] = a[offsetA] * b[offsetB];
        ADVANCE_ODOMETER_LOOP
    }
}

void s_div_double(const double *a, const int *stridesA,
                  const double *b, const int *stridesB,
                  double *res, const int *stridesRes,
                  const int *shape, int rank) {
    if (a == NULL || b == NULL || res == NULL || rank <= 0 || rank > 8) return;
    int coord[8] = {0};
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    for (int el = 0; el < total_elements; el++) {
        int offsetA = 0, offsetB = 0, offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetA += coord[d] * stridesA[d];
            offsetB += coord[d] * stridesB[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        res[offsetRes] = a[offsetA] / b[offsetB];
        ADVANCE_ODOMETER_LOOP
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
    int coord[8] = {0};
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    for (int el = 0; el < total_elements; el++) {
        int offsetA = 0, offsetB = 0, offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetA += coord[d] * stridesA[d];
            offsetB += coord[d] * stridesB[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        res[offsetRes].r = a[offsetA].r + b[offsetB].r;
        res[offsetRes].i = a[offsetA].i + b[offsetB].i;
        ADVANCE_ODOMETER_LOOP
    }
}

void s_sub_complex(const cpx_t *a, const int *stridesA,
                  const cpx_t *b, const int *stridesB,
                  cpx_t *res, const int *stridesRes,
                  const int *shape, int rank) {
    if (a == NULL || b == NULL || res == NULL || rank <= 0 || rank > 8) return;
    int coord[8] = {0};
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    for (int el = 0; el < total_elements; el++) {
        int offsetA = 0, offsetB = 0, offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetA += coord[d] * stridesA[d];
            offsetB += coord[d] * stridesB[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        res[offsetRes].r = a[offsetA].r - b[offsetB].r;
        res[offsetRes].i = a[offsetA].i - b[offsetB].i;
        ADVANCE_ODOMETER_LOOP
    }
}

void s_mul_complex(const cpx_t *a, const int *stridesA,
                  const cpx_t *b, const int *stridesB,
                  cpx_t *res, const int *stridesRes,
                  const int *shape, int rank) {
    if (a == NULL || b == NULL || res == NULL || rank <= 0 || rank > 8) return;
    int coord[8] = {0};
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    for (int el = 0; el < total_elements; el++) {
        int offsetA = 0, offsetB = 0, offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetA += coord[d] * stridesA[d];
            offsetB += coord[d] * stridesB[d];
            offsetRes += coord[d] * stridesRes[d];
        }
        double r1 = a[offsetA].r, i1 = a[offsetA].i;
        double r2 = b[offsetB].r, i2 = b[offsetB].i;
        res[offsetRes].r = r1 * r2 - i1 * i2;
        res[offsetRes].i = r1 * i2 + i1 * r2;
        ADVANCE_ODOMETER_LOOP
    }
}

void s_div_complex(const cpx_t *a, const int *stridesA,
                  const cpx_t *b, const int *stridesB,
                  cpx_t *res, const int *stridesRes,
                  const int *shape, int rank) {
    if (a == NULL || b == NULL || res == NULL || rank <= 0 || rank > 8) return;
    int coord[8] = {0};
    int total_elements = 1;
    for (int i = 0; i < rank; i++) total_elements *= shape[i];

    for (int el = 0; el < total_elements; el++) {
        int offsetA = 0, offsetB = 0, offsetRes = 0;
        for (int d = 0; d < rank; d++) {
            offsetA += coord[d] * stridesA[d];
            offsetB += coord[d] * stridesB[d];
            offsetRes += coord[d] * stridesRes[d];
        }
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
        ADVANCE_ODOMETER_LOOP
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
    float acc = 0.0f;
    for (int i = 0; i < size; i++) {
        acc += src[i];
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
