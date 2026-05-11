#ifndef CUSTOM_UFUNCS_H
#define CUSTOM_UFUNCS_H

#include <stdint.h>

// ----------------------------------------------------------------------------
// Binary-Compatible Complex Number Type (Matches kiss_fft_cpx and ndarray.Complex)
// ----------------------------------------------------------------------------
typedef struct {
    double r; // real part
    double i; // imaginary part
} cpx_t;

typedef struct {
    float r; // real part
    float i; // imaginary part
} cpx_f_t;

// ----------------------------------------------------------------------------
// Double Precision (Float64) Flat Contiguous Kernels
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
double r_prod_double(const double *src, int size);

// ----------------------------------------------------------------------------
// Double Precision (Float64) Generic ND Strided Broadcasting Kernels
// ----------------------------------------------------------------------------
void s_add_double(const double *a, const int *stridesA,
                  const double *b, const int *stridesB,
                  double *res, const int *stridesRes,
                  const int *shape, int rank);
void s_sub_double(const double *a, const int *stridesA,
                  const double *b, const int *stridesB,
                  double *res, const int *stridesRes,
                  const int *shape, int rank);
void s_mul_double(const double *a, const int *stridesA,
                  const double *b, const int *stridesB,
                  double *res, const int *stridesRes,
                  const int *shape, int rank);
void s_div_double(const double *a, const int *stridesA,
                  const double *b, const int *stridesB,
                  double *res, const int *stridesRes,
                  const int *shape, int rank);

// ----------------------------------------------------------------------------
// Complex128 (Dual Float64) Vector Kernels (Contiguous and Strided)
// ----------------------------------------------------------------------------
void v_add_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int size);
void v_sub_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int size);
void v_mul_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int size);
void v_div_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int size);

void s_add_complex(const cpx_t *a, const int *stridesA,
                  const cpx_t *b, const int *stridesB,
                  cpx_t *res, const int *stridesRes,
                  const int *shape, int rank);
void s_sub_complex(const cpx_t *a, const int *stridesA,
                  const cpx_t *b, const int *stridesB,
                  cpx_t *res, const int *stridesRes,
                  const int *shape, int rank);
void s_mul_complex(const cpx_t *a, const int *stridesA,
                  const cpx_t *b, const int *stridesB,
                  cpx_t *res, const int *stridesRes,
                  const int *shape, int rank);
void s_div_complex(const cpx_t *a, const int *stridesA,
                  const cpx_t *b, const int *stridesB,
                  cpx_t *res, const int *stridesRes,
                  const int *shape, int rank);

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
float r_prod_float(const float *src, int size);

// ----------------------------------------------------------------------------
// Additional Math, Rounding, and Clipping Ufuncs (Float64 & Float32 Tiers)
// ----------------------------------------------------------------------------

void v_sqrt_double(const double *src, double *res, int size);
void v_tan_double(const double *src, double *res, int size);
void v_abs_double(const double *src, double *res, int size);
void v_ceil_double(const double *src, double *res, int size);
void v_floor_double(const double *src, double *res, int size);
void v_round_double(const double *src, double *res, int size);
void v_clip_double(const double *src, double *res, double min_val, double max_val, int size);

void v_sqrt_float(const float *src, float *res, int size);
void v_tan_float(const float *src, float *res, int size);
void v_abs_float(const float *src, float *res, int size);
void v_ceil_float(const float *src, float *res, int size);
void v_floor_float(const float *src, float *res, int size);
void v_round_float(const float *src, float *res, int size);
void v_clip_float(const float *src, float *res, float min_val, float max_val, int size);

// ----------------------------------------------------------------------------
// Generic ND Strided Broadcasting Ternary "where" Sorters (Rank <= 8)
// ----------------------------------------------------------------------------

void s_where_double(const unsigned char *cond, const int *stridesCond,
                    const double *x, const int *stridesX,
                    const double *y, const int *stridesY,
                    double *res, const int *stridesRes,
                    const int *shape, int rank);

void s_where_float(const unsigned char *cond, const int *stridesCond,
                   const float *x, const int *stridesX,
                   const float *y, const int *stridesY,
                   float *res, const int *stridesRes,
                   const int *shape, int rank);

void v_normal_double(double *res, int size, double loc, double scale, unsigned long long seed);
void v_normal_float(float *res, int size, float loc, float scale, unsigned long long seed);

void v_uniform_double(double *res, int size, unsigned long long seed);
void v_uniform_float(float *res, int size, unsigned long long seed);
void v_randint_int64(int64_t *res, int size, int64_t low, int64_t high, unsigned long long seed);
void v_randint_int32(int32_t *res, int size, int32_t low, int32_t high, unsigned long long seed);
void v_fill_double(double *res, double value, int size);
void v_fill_float(float *res, float value, int size);
void v_fill_int64(int64_t *res, int64_t value, int size);
void v_fill_int32(int32_t *res, int32_t value, int size);
void v_secure_uniform_double(double *res, int size);
void v_secure_uniform_float(float *res, int size);
void v_secure_randint_int64(int64_t *res, int size, int64_t low, int64_t high);
void v_secure_randint_int32(int32_t *res, int size, int32_t low, int32_t high);
void v_secure_normal_double(double *res, int size, double loc, double scale);
void v_secure_normal_float(float *res, int size, float loc, float scale);
void v_tril_double(const double *src, double *res, int batch_count, int rows, int cols, int k);
void v_tril_float(const float *src, float *res, int batch_count, int rows, int cols, int k);
void v_triu_double(const double *src, double *res, int batch_count, int rows, int cols, int k);
void v_triu_float(const float *src, float *res, int batch_count, int rows, int cols, int k);

void v_asin_double(const double *src, double *res, int size);
void v_asin_float(const float *src, float *res, int size);
void v_acos_double(const double *src, double *res, int size);
void v_acos_float(const float *src, float *res, int size);
void v_atan_double(const double *src, double *res, int size);
void v_atan_float(const float *src, float *res, int size);
void v_atan2_double(const double *y, const double *x, double *res, int size);
void v_atan2_float(const float *y, const float *x, float *res, int size);

void v_sinh_double(const double *src, double *res, int size);
void v_sinh_float(const float *src, float *res, int size);
void v_cosh_double(const double *src, double *res, int size);
void v_cosh_float(const float *src, float *res, int size);
void v_tanh_double(const double *src, double *res, int size);
void v_tanh_float(const float *src, float *res, int size);
void v_asinh_double(const double *src, double *res, int size);
void v_asinh_float(const float *src, float *res, int size);
void v_acosh_double(const double *src, double *res, int size);
void v_acosh_float(const float *src, float *res, int size);
void v_atanh_double(const double *src, double *res, int size);
void v_sin_complex128(const cpx_t *src, cpx_t *res, int size);
void v_sin_complex64(const cpx_f_t *src, cpx_f_t *res, int size);
void v_cos_complex128(const cpx_t *src, cpx_t *res, int size);
void v_cos_complex64(const cpx_f_t *src, cpx_f_t *res, int size);
void v_tan_complex128(const cpx_t *src, cpx_t *res, int size);
void v_tan_complex64(const cpx_f_t *src, cpx_f_t *res, int size);

void s_sin_complex128(const cpx_t *src, const int *stridesSrc, cpx_t *res, const int *stridesRes, const int *shape, int rank);
void s_sin_complex64(const cpx_f_t *src, const int *stridesSrc, cpx_f_t *res, const int *stridesRes, const int *shape, int rank);
void s_cos_complex128(const cpx_t *src, const int *stridesSrc, cpx_t *res, const int *stridesRes, const int *shape, int rank);
void s_cos_complex64(const cpx_f_t *src, const int *stridesSrc, cpx_f_t *res, const int *stridesRes, const int *shape, int rank);
void s_tan_complex128(const cpx_t *src, const int *stridesSrc, cpx_t *res, const int *stridesRes, const int *shape, int rank);
void s_tan_complex64(const cpx_f_t *src, const int *stridesSrc, cpx_f_t *res, const int *stridesRes, const int *shape, int rank);



void s_cumsum_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_cumsum_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_cumsum_int64(const int64_t *src, const int *stridesSrc, int64_t *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_cumsum_int32(const int32_t *src, const int *stridesSrc, int32_t *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_cumsum_complex128(const cpx_t *src, const int *stridesSrc, cpx_t *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_cumsum_complex64(const cpx_f_t *src, const int *stridesSrc, cpx_f_t *res, const int *stridesRes, const int *shape, int rank, int axis);

void s_cumprod_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_cumprod_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_cumprod_int64(const int64_t *src, const int *stridesSrc, int64_t *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_cumprod_int32(const int32_t *src, const int *stridesSrc, int32_t *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_cumprod_complex128(const cpx_t *src, const int *stridesSrc, cpx_t *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_cumprod_complex64(const cpx_f_t *src, const int *stridesSrc, cpx_f_t *res, const int *stridesRes, const int *shape, int rank, int axis);

void s_cummin_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_cummin_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_cummin_int64(const int64_t *src, const int *stridesSrc, int64_t *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_cummin_int32(const int32_t *src, const int *stridesSrc, int32_t *res, const int *stridesRes, const int *shape, int rank, int axis);

void s_cummax_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_cummax_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_cummax_int64(const int64_t *src, const int *stridesSrc, int64_t *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_cummax_int32(const int32_t *src, const int *stridesSrc, int32_t *res, const int *stridesRes, const int *shape, int rank, int axis);

void s_diff_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_diff_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_diff_int64(const int64_t *src, const int *stridesSrc, int64_t *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_diff_int32(const int32_t *src, const int *stridesSrc, int32_t *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_diff_complex128(const cpx_t *src, const int *stridesSrc, cpx_t *res, const int *stridesRes, const int *shape, int rank, int axis);
void s_diff_complex64(const cpx_f_t *src, const int *stridesSrc, cpx_f_t *res, const int *stridesRes, const int *shape, int rank, int axis);

void s_sin_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank);
void s_sin_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank);
void s_cos_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank);
void s_cos_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank);

// ------------------------------------------------------------
// Native C High-Speed Strided Flattening/Copying Kernels
// ----------------------------------------------------------------------------
void s_flatten_double(const double *src, const int *stridesSrc, double *dest, const int *shape, int rank);
void s_flatten_float(const float *src, const int *stridesSrc, float *dest, const int *shape, int rank);
void s_flatten_int64(const int64_t *src, const int *stridesSrc, int64_t *dest, const int *shape, int rank);
void s_flatten_int32(const int32_t *src, const int *stridesSrc, int32_t *dest, const int *shape, int rank);
void s_flatten_complex128(const double *src, const int *stridesSrc, double *dest, const int *shape, int rank);
void s_flatten_complex64(const float *src, const int *stridesSrc, float *dest, const int *shape, int rank);
void s_flatten_boolean(const uint8_t *src, const int *stridesSrc, uint8_t *dest, const int *shape, int rank);

// ----------------------------------------------------------------------------
// Hashing Kernels
// ----------------------------------------------------------------------------
uint32_t s_hash_double(const double *a, const int *strides, const int *shape, int rank, int is_contiguous);
uint32_t s_hash_float(const float *a, const int *strides, const int *shape, int rank, int is_contiguous);
uint32_t s_hash_int64(const int64_t *a, const int *strides, const int *shape, int rank, int is_contiguous);
uint32_t s_hash_int32(const int32_t *a, const int *strides, const int *shape, int rank, int is_contiguous);
uint32_t s_hash_complex128(const double *a, const int *strides, const int *shape, int rank, int is_contiguous);
uint32_t s_hash_complex64(const float *a, const int *strides, const int *shape, int rank, int is_contiguous);
uint32_t s_hash_boolean(const uint8_t *a, const int *strides, const int *shape, int rank, int is_contiguous);

// ----------------------------------------------------------------------------
// Random Distribution Generators
// ----------------------------------------------------------------------------
void v_poisson_int64(int64_t *res, int size, double lam, unsigned long long seed);
void v_poisson_int32(int32_t *res, int size, double lam, unsigned long long seed);
void v_binomial_int64(int64_t *res, int size, int n, double p, unsigned long long seed);
void v_binomial_int32(int32_t *res, int size, int n, double p, unsigned long long seed);

#endif // CUSTOM_UFUNCS_H
