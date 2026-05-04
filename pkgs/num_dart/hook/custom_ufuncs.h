#ifndef CUSTOM_UFUNCS_H
#define CUSTOM_UFUNCS_H

// ----------------------------------------------------------------------------
// Double Precision (Float64) Vector Kernels
// ----------------------------------------------------------------------------

void v_add_double(const double *a, const double *b, double *res, int size);
void v_sub_double(const double *a, const double *b, double *res, int size);
void v_mul_double(const double *a, const double *b, double *res, int size);
void v_div_double(const double *a, const double *b, double *res, int size);

void v_sin_double(const double *src, double *res, int size);
void v_cos_double(const double *src, double *res, int size);
void v_exp_double(const double *src, double *res, int size);
void v_log_double(const double *src, double *res, int size);

double r_sum_double(const double *src, int size);

// ----------------------------------------------------------------------------
// Single Precision (Float32) Vector Kernels
// ----------------------------------------------------------------------------

void v_add_float(const float *a, const float *b, float *res, int size);
void v_sub_float(const float *a, const float *b, float *res, int size);
void v_mul_float(const float *a, const float *b, float *res, int size);
void v_div_float(const float *a, const float *b, float *res, int size);

void v_sin_float(const float *src, float *res, int size);
void v_cos_float(const float *src, float *res, int size);
void v_exp_float(const float *src, float *res, int size);
void v_log_float(const float *src, float *res, int size);

float r_sum_float(const float *src, int size);

#endif // CUSTOM_UFUNCS_H
