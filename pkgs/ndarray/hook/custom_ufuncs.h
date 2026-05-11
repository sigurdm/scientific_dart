#ifndef CUSTOM_UFUNCS_H
#define CUSTOM_UFUNCS_H

#include <stdint.h>

/* ============================================================================
 * SECTION 1: CORE COMPLEX TYPE DEFINITIONS
 * ============================================================================
 * Binary-compatible complex number structures matching kiss_fft_cpx
 * and mapped directly to ndarray's custom Complex structures.
 */

typedef struct {
    double r; /** Real component */
    double i; /** Imaginary component */
} cpx_t;

typedef struct {
    float r;  /** Real component */
    float i;  /** Imaginary component */
} cpx_f_t;

/* ============================================================================
 * SECTION 2: FLAT CONTIGUOUS VECTOR MATHEMATICS (ufuncs)
 * ============================================================================
 * Optimized flat vector sweeps operating on sequential same-shape memory arrays.
 * Bypasses coordinate translation and stride checking entirely for peak performance.
 */

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

void v_add_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int size);
void v_sub_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int size);
void v_mul_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int size);
void v_div_complex(const cpx_t *a, const cpx_t *b, cpx_t *res, int size);

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

void v_asin_complex128(const cpx_t *src, cpx_t *res, int size);
void v_asin_complex64(const cpx_f_t *src, cpx_f_t *res, int size);
void v_acos_complex128(const cpx_t *src, cpx_t *res, int size);
void v_acos_complex64(const cpx_f_t *src, cpx_f_t *res, int size);
void v_atan_complex128(const cpx_t *src, cpx_t *res, int size);
void v_atan_complex64(const cpx_f_t *src, cpx_f_t *res, int size);

void v_atanh_complex128(const cpx_t *src, cpx_t *res, int size);
void v_atanh_complex64(const cpx_f_t *src, cpx_f_t *res, int size);

void v_hypot_complex128(const cpx_t *x1, const cpx_t *x2, double *res, int size);
void v_hypot_complex64(const cpx_f_t *x1, const cpx_f_t *x2, float *res, int size);

void v_pow_complex128(const cpx_t *x1, const cpx_t *x2, cpx_t *res, int size);
void v_pow_complex64(const cpx_f_t *x1, const cpx_f_t *x2, cpx_f_t *res, int size);

void v_conj_complex128(const cpx_t *src, cpx_t *res, int size);
void v_conj_complex64(const cpx_f_t *src, cpx_f_t *res, int size);

/* ============================================================================
 * SECTION 3: STRIDED MULTI-DIMENSIONAL ODOMETER WALKS (ufuncs)
 * ============================================================================
 * High-speed ND strided broadcasting and view loops executing on unmanaged C.
 * Resolves sliced views, transposes, and dimension broadcasting recursively.
 */

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

void s_sin_complex128(const cpx_t *src, const int *stridesSrc, cpx_t *res, const int *stridesRes, const int *shape, int rank);
void s_sin_complex64(const cpx_f_t *src, const int *stridesSrc, cpx_f_t *res, const int *stridesRes, const int *shape, int rank);
void s_cos_complex128(const cpx_t *src, const int *stridesSrc, cpx_t *res, const int *stridesRes, const int *shape, int rank);
void s_cos_complex64(const cpx_f_t *src, const int *stridesSrc, cpx_f_t *res, const int *stridesRes, const int *shape, int rank);
void s_tan_complex128(const cpx_t *src, const int *stridesSrc, cpx_t *res, const int *stridesRes, const int *shape, int rank);
void s_tan_complex64(const cpx_f_t *src, const int *stridesSrc, cpx_f_t *res, const int *stridesRes, const int *shape, int rank);

void s_asin_complex128(const cpx_t *src, const int *stridesSrc, cpx_t *res, const int *stridesRes, const int *shape, int rank);
void s_asin_complex64(const cpx_f_t *src, const int *stridesSrc, cpx_f_t *res, const int *stridesRes, const int *shape, int rank);
void s_acos_complex128(const cpx_t *src, const int *stridesSrc, cpx_t *res, const int *stridesRes, const int *shape, int rank);
void s_acos_complex64(const cpx_f_t *src, const int *stridesSrc, cpx_f_t *res, const int *stridesRes, const int *shape, int rank);
void s_atan_complex128(const cpx_t *src, const int *stridesSrc, cpx_t *res, const int *stridesRes, const int *shape, int rank);
void s_atan_complex64(const cpx_f_t *src, const int *stridesSrc, cpx_f_t *res, const int *stridesRes, const int *shape, int rank);

void s_atanh_complex128(const cpx_t *src, const int *stridesSrc, cpx_t *res, const int *stridesRes, const int *shape, int rank);
void s_atanh_complex64(const cpx_f_t *src, const int *stridesSrc, cpx_f_t *res, const int *stridesRes, const int *shape, int rank);

void s_hypot_complex128(const cpx_t *x1, const int *stridesX1, const cpx_t *x2, const int *stridesX2, double *res, const int *stridesRes, const int *shape, int rank);
void s_hypot_complex64(const cpx_f_t *x1, const int *stridesX1, const cpx_f_t *x2, const int *stridesX2, float *res, const int *stridesRes, const int *shape, int rank);

void s_pow_complex128(const cpx_t *x1, const int *stridesX1, const cpx_t *x2, const int *stridesX2, cpx_t *res, const int *stridesRes, const int *shape, int rank);
void s_pow_complex64(const cpx_f_t *x1, const int *stridesX1, const cpx_f_t *x2, const int *stridesX2, cpx_f_t *res, const int *stridesRes, const int *shape, int rank);

void s_conj_complex128(const cpx_t *src, const int *stridesSrc, cpx_t *res, const int *stridesRes, const int *shape, int rank);
void s_conj_complex64(const cpx_f_t *src, const int *stridesSrc, cpx_f_t *res, const int *stridesRes, const int *shape, int rank);

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

void s_asin_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank);
void s_asin_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank);
void s_acos_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank);
void s_acos_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank);
void s_atan_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank);
void s_atan_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank);

void s_atan2_double(const double *y, const int *stridesY, const double *x, const int *stridesX, double *res, const int *stridesRes, const int *shape, int rank);
void s_atan2_float(const float *y, const int *stridesY, const float *x, const int *stridesX, float *res, const int *stridesRes, const int *shape, int rank);

void s_tan_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank);
void s_tan_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank);
void s_exp_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank);
void s_exp_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank);
void s_log_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank);
void s_log_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank);

void s_sinh_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank);
void s_sinh_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank);
void s_cosh_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank);
void s_cosh_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank);
void s_tanh_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank);
void s_tanh_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank);

void s_asinh_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank);
void s_asinh_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank);
void s_acosh_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank);
void s_acosh_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank);
void s_atanh_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank);
void s_atanh_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank);

/* ============================================================================
 * SECTION 4: FLAT FLATTENING & STRIDED COPYING KERNELS
 * ============================================================================
 * Used inside ravel() and flatten() for fast layout conversions.
 */

void s_flatten_double(const double *src, const int *stridesSrc, double *dest, const int *shape, int rank);
void s_flatten_float(const float *src, const int *stridesSrc, float *dest, const int *shape, int rank);
void s_flatten_int64(const int64_t *src, const int *stridesSrc, int64_t *dest, const int *shape, int rank);
void s_flatten_int32(const int32_t *src, const int *stridesSrc, int32_t *dest, const int *shape, int rank);
void s_flatten_complex128(const double *src, const int *stridesSrc, double *dest, const int *shape, int rank);
void s_flatten_complex64(const float *src, const int *stridesSrc, float *dest, const int *shape, int rank);
void s_flatten_boolean(const uint8_t *src, const int *stridesSrc, uint8_t *dest, const int *shape, int rank);

/* ============================================================================
 * SECTION 5: HASHING KERNELS
 * ============================================================================
 * High-speed hashing algorithms for quick comparison validation.
 */

uint32_t s_hash_double(const double *a, const int *strides, const int *shape, int rank, int is_contiguous);
uint32_t s_hash_float(const float *a, const int *strides, const int *shape, int rank, int is_contiguous);
uint32_t s_hash_int64(const int64_t *a, const int *strides, const int *shape, int rank, int is_contiguous);
uint32_t s_hash_int32(const int32_t *a, const int *strides, const int *shape, int rank, int is_contiguous);
uint32_t s_hash_complex128(const double *a, const int *strides, const int *shape, int rank, int is_contiguous);
uint32_t s_hash_complex64(const float *a, const int *strides, const int *shape, int rank, int is_contiguous);
uint32_t s_hash_boolean(const uint8_t *a, const int *strides, const int *shape, int rank, int is_contiguous);

/* ============================================================================
 * SECTION 6: RANDOM GENERATION KERNELS
 * ============================================================================
 * High-performance random statistical simulators.
 */

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
void v_poisson_int64(int64_t *res, int size, double lam, unsigned long long seed);
void v_poisson_int32(int32_t *res, int size, double lam, unsigned long long seed);
void v_binomial_int64(int64_t *res, int size, int n, double p, unsigned long long seed);
void v_binomial_int32(int32_t *res, int size, int n, double p, unsigned long long seed);

/* ============================================================================
 * SECTION 7: TRIANGULAR MATRIX EXTRACTORS
 * ============================================================================
 */

void v_tril_double(const double *src, double *res, int batch_count, int rows, int cols, int k);
void v_tril_float(const float *src, float *res, int batch_count, int rows, int cols, int k);
void v_triu_double(const double *src, double *res, int batch_count, int rows, int cols, int k);
void v_triu_float(const float *src, float *res, int batch_count, int rows, int cols, int k);

/* ============================================================================
 * SECTION 8: AUTO-GENERATED CROSS-TYPE BINARY MATH FFI HELPERS
 * ============================================================================
 */

#define GENERATE_OP_COMBINATIONS(OP, MACRO) \
  MACRO(OP, double, double, double, double, double, double) \
  MACRO(OP, double, float, double, double, float, double) \
  MACRO(OP, double, int64, double, double, int64_t, double) \
  MACRO(OP, double, int32, double, double, int32_t, double) \
  MACRO(OP, double, cpx, cpx, double, cpx_t, cpx_t) \
  MACRO(OP, float, double, double, float, double, double) \
  MACRO(OP, float, float, float, float, float, float) \
  MACRO(OP, float, int64, float, float, int64_t, float) \
  MACRO(OP, float, int32, float, float, int32_t, float) \
  MACRO(OP, float, cpx, cpx, float, cpx_t, cpx_t) \
  MACRO(OP, int64, double, double, int64_t, double, double) \
  MACRO(OP, int64, float, float, int64_t, float, float) \
  MACRO(OP, int64, int64, int64, int64_t, int64_t, int64_t) \
  MACRO(OP, int64, int32, int64, int64_t, int32_t, int64_t) \
  MACRO(OP, int64, cpx, cpx, int64_t, cpx_t, cpx_t) \
  MACRO(OP, int32, double, double, int32_t, double, double) \
  MACRO(OP, int32, float, float, int32_t, float, float) \
  MACRO(OP, int32, int64, int64, int32_t, int64_t, int64_t) \
  MACRO(OP, int32, int32, int32, int32_t, int32_t, int32_t) \
  MACRO(OP, int32, cpx, cpx, int32_t, cpx_t, cpx_t) \
  MACRO(OP, cpx, double, cpx, cpx_t, double, cpx_t) \
  MACRO(OP, cpx, float, cpx, cpx_t, float, cpx_t) \
  MACRO(OP, cpx, int64, cpx, cpx_t, int64_t, cpx_t) \
  MACRO(OP, cpx, int32, cpx, cpx_t, int32_t, cpx_t) \
  MACRO(OP, cpx, cpx, cpx, cpx_t, cpx_t, cpx_t)

#define GENERATE_DIV_COMBINATIONS(OP, MACRO) \
  MACRO(OP, double, double, double, double, double, double) \
  MACRO(OP, double, float, double, double, float, double) \
  MACRO(OP, double, int64, double, double, int64_t, double) \
  MACRO(OP, double, int32, double, double, int32_t, double) \
  MACRO(OP, double, cpx, cpx, double, cpx_t, cpx_t) \
  MACRO(OP, float, double, double, float, double, double) \
  MACRO(OP, float, float, float, float, float, float) \
  MACRO(OP, float, int64, float, float, int64_t, float) \
  MACRO(OP, float, int32, float, float, int32_t, float) \
  MACRO(OP, float, cpx, cpx, float, cpx_t, cpx_t) \
  MACRO(OP, int64, double, double, int64_t, double, double) \
  MACRO(OP, int64, float, float, int64_t, float, float) \
  MACRO(OP, int64, int64, double, int64_t, int64_t, double) \
  MACRO(OP, int64, int32, double, int64_t, int32_t, double) \
  MACRO(OP, int64, cpx, cpx, int64_t, cpx_t, cpx_t) \
  MACRO(OP, int32, double, double, int32_t, double, double) \
  MACRO(OP, int32, float, float, int32_t, float, float) \
  MACRO(OP, int32, int64, double, int32_t, int64_t, double) \
  MACRO(OP, int32, int32, double, int32_t, int32_t, double) \
  MACRO(OP, int32, cpx, cpx, int32_t, cpx_t, cpx_t) \
  MACRO(OP, cpx, double, cpx, cpx_t, double, cpx_t) \
  MACRO(OP, cpx, float, cpx, cpx_t, float, cpx_t) \
  MACRO(OP, cpx, int64, cpx, cpx_t, int64_t, cpx_t) \
  MACRO(OP, cpx, int32, cpx, cpx_t, int32_t, cpx_t) \
  MACRO(OP, cpx, cpx, cpx, cpx_t, cpx_t, cpx_t)

#define DECLARE_FFI_HELPER(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr) \
  void v_##OP##_##Ta_tok##_##Tb_tok##_##Tr_tok(const Ta *a, const Tb *b, Tr *res, int size); \
  void s_##OP##_##Ta_tok##_##Tb_tok##_##Tr_tok(const Ta *a, const int *stridesA, \
                                               const Tb *b, const int *stridesB, \
                                               Tr *res, const int *stridesRes, \
                                               const int *shape, int rank);

GENERATE_OP_COMBINATIONS(add, DECLARE_FFI_HELPER)
GENERATE_OP_COMBINATIONS(sub, DECLARE_FFI_HELPER)
GENERATE_OP_COMBINATIONS(mul, DECLARE_FFI_HELPER)
GENERATE_DIV_COMBINATIONS(div, DECLARE_FFI_HELPER)

#endif /* CUSTOM_UFUNCS_H */
