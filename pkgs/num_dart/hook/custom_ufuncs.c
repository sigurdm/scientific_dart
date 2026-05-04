#include "custom_ufuncs.h"
#include <math.h>
#include <stdlib.h>

// ----------------------------------------------------------------------------
// Double Precision (Float64) Implementations
// ----------------------------------------------------------------------------

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

// ----------------------------------------------------------------------------
// Single Precision (Float32) Implementations
// ----------------------------------------------------------------------------

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
