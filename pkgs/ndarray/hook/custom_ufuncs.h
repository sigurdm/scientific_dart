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
void v_atanh_float(const float *src, float *res, int size);

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

void v_square_double(const double *src, double *res, int size);
void v_square_float(const float *src, float *res, int size);
void v_square_int64(const int64_t *src, int64_t *res, int size);
void v_square_int32(const int32_t *src, int32_t *res, int size);
void v_square_complex128(const cpx_t *src, cpx_t *res, int size);
void v_square_complex64(const cpx_f_t *src, cpx_f_t *res, int size);

void v_pow_double(const double *x1, const double *x2, double *res, int size);
void v_pow_float(const float *x1, const float *x2, float *res, int size);

void v_floordiv_double(const double *x1, const double *x2, double *res, int size);
void v_floordiv_float(const float *x1, const float *x2, float *res, int size);
void v_floordiv_int64(const int64_t *x1, const int64_t *x2, int64_t *res, int size);
void v_floordiv_int32(const int32_t *x1, const int32_t *x2, int32_t *res, int size);

void v_remainder_double(const double *x1, const double *x2, double *res, int size);
void v_remainder_float(const float *x1, const float *x2, float *res, int size);
void v_remainder_int64(const int64_t *x1, const int64_t *x2, int64_t *res, int size);
void v_remainder_int32(const int32_t *x1, const int32_t *x2, int32_t *res, int size);

void v_isnan_double(const double *src, uint8_t *res, int size);
void v_isnan_float(const float *src, uint8_t *res, int size);
void v_isnan_complex128(const cpx_t *src, uint8_t *res, int size);
void v_isnan_complex64(const cpx_f_t *src, uint8_t *res, int size);

void v_isinf_double(const double *src, uint8_t *res, int size);
void v_isinf_float(const float *src, uint8_t *res, int size);
void v_isinf_complex128(const cpx_t *src, uint8_t *res, int size);
void v_isinf_complex64(const cpx_f_t *src, uint8_t *res, int size);

void v_isfinite_double(const double *src, uint8_t *res, int size);
void v_isfinite_float(const float *src, uint8_t *res, int size);
void v_isfinite_complex128(const cpx_t *src, uint8_t *res, int size);
void v_isfinite_complex64(const cpx_f_t *src, uint8_t *res, int size);

void v_copysign_double(const double *x1, const double *x2, double *res, int size);
void v_copysign_float(const float *x1, const float *x2, float *res, int size);

/* Vectorized Bitwise Universal Functions */
void v_bitwise_and_int32(const int32_t *a, const int32_t *b, int32_t *res, int size);
void v_bitwise_and_int64(const int64_t *a, const int64_t *b, int64_t *res, int size);
void v_bitwise_and_uint8(const uint8_t *a, const uint8_t *b, uint8_t *res, int size);
void v_bitwise_and_int16(const int16_t *a, const int16_t *b, int16_t *res, int size);

void v_bitwise_or_int32(const int32_t *a, const int32_t *b, int32_t *res, int size);
void v_bitwise_or_int64(const int64_t *a, const int64_t *b, int64_t *res, int size);
void v_bitwise_or_uint8(const uint8_t *a, const uint8_t *b, uint8_t *res, int size);
void v_bitwise_or_int16(const int16_t *a, const int16_t *b, int16_t *res, int size);

void v_bitwise_xor_int32(const int32_t *a, const int32_t *b, int32_t *res, int size);
void v_bitwise_xor_int64(const int64_t *a, const int64_t *b, int64_t *res, int size);
void v_bitwise_xor_uint8(const uint8_t *a, const uint8_t *b, uint8_t *res, int size);
void v_bitwise_xor_int16(const int16_t *a, const int16_t *b, int16_t *res, int size);

void v_left_shift_int32(const int32_t *a, const int32_t *b, int32_t *res, int size);
void v_left_shift_int64(const int64_t *a, const int64_t *b, int64_t *res, int size);
void v_left_shift_uint8(const uint8_t *a, const uint8_t *b, uint8_t *res, int size);
void v_left_shift_int16(const int16_t *a, const int16_t *b, int16_t *res, int size);

void v_right_shift_int32(const int32_t *a, const int32_t *b, int32_t *res, int size);
void v_right_shift_int64(const int64_t *a, const int64_t *b, int64_t *res, int size);
void v_right_shift_uint8(const uint8_t *a, const uint8_t *b, uint8_t *res, int size);
void v_right_shift_int16(const int16_t *a, const int16_t *b, int16_t *res, int size);

void v_invert_int32(const int32_t *src, int32_t *res, int size);
void v_invert_int64(const int64_t *src, int64_t *res, int size);
void v_invert_uint8(const uint8_t *src, uint8_t *res, int size);
void v_invert_int16(const int16_t *src, int16_t *res, int size);


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
void s_square_double(const double *src, const int *stridesSrc, double *res, const int *stridesRes, const int *shape, int rank);
void s_square_float(const float *src, const int *stridesSrc, float *res, const int *stridesRes, const int *shape, int rank);
void s_square_int64(const int64_t *src, const int *stridesSrc, int64_t *res, const int *stridesRes, const int *shape, int rank);
void s_square_int32(const int32_t *src, const int *stridesSrc, int32_t *res, const int *stridesRes, const int *shape, int rank);
void s_square_complex128(const cpx_t *src, const int *stridesSrc, cpx_t *res, const int *stridesRes, const int *shape, int rank);
void s_square_complex64(const cpx_f_t *src, const int *stridesSrc, cpx_f_t *res, const int *stridesRes, const int *shape, int rank);

void s_pow_double(const double *x1, const int *stridesX1, const double *x2, const int *stridesX2, double *res, const int *stridesRes, const int *shape, int rank);
void s_pow_float(const float *x1, const int *stridesX1, const float *x2, const int *stridesX2, float *res, const int *stridesRes, const int *shape, int rank);

void s_floordiv_double(const double *x1, const int *stridesX1, const double *x2, const int *stridesX2, double *res, const int *stridesRes, const int *shape, int rank);
void s_floordiv_float(const float *x1, const int *stridesX1, const float *x2, const int *stridesX2, float *res, const int *stridesRes, const int *shape, int rank);
void s_floordiv_int64(const int64_t *x1, const int *stridesX1, const int64_t *x2, const int *stridesX2, int64_t *res, const int *stridesRes, const int *shape, int rank);
void s_floordiv_int32(const int32_t *x1, const int *stridesX1, const int32_t *x2, const int *stridesX2, int32_t *res, const int *stridesRes, const int *shape, int rank);

void s_remainder_double(const double *x1, const int *stridesX1, const double *x2, const int *stridesX2, double *res, const int *stridesRes, const int *shape, int rank);
void s_remainder_float(const float *x1, const int *stridesX1, const float *x2, const int *stridesX2, float *res, const int *stridesRes, const int *shape, int rank);
void s_remainder_int64(const int64_t *x1, const int *stridesX1, const int64_t *x2, const int *stridesX2, int64_t *res, const int *stridesRes, const int *shape, int rank);
void s_remainder_int32(const int32_t *x1, const int *stridesX1, const int32_t *x2, const int *stridesX2, int32_t *res, const int *stridesRes, const int *shape, int rank);

void s_isnan_double(const double *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_isnan_float(const float *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_isnan_complex128(const cpx_t *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_isnan_complex64(const cpx_f_t *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);

void s_isinf_double(const double *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_isinf_float(const float *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_isinf_complex128(const cpx_t *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_isinf_complex64(const cpx_f_t *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);

void s_isfinite_double(const double *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_isfinite_float(const float *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_isfinite_complex128(const cpx_t *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_isfinite_complex64(const cpx_f_t *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);

void s_copysign_double(const double *x1, const int *stridesX1, const double *x2, const int *stridesX2, double *res, const int *stridesRes, const int *shape, int rank);
void s_copysign_float(const float *x1, const int *stridesX1, const float *x2, const int *stridesX2, float *res, const int *stridesRes, const int *shape, int rank);

/* Strided Vectorized Bitwise Universal Functions */
void s_bitwise_and_int32(const int32_t *a, const int *stridesA, const int32_t *b, const int *stridesB, int32_t *res, const int *stridesRes, const int *shape, int rank);
void s_bitwise_and_int64(const int64_t *a, const int *stridesA, const int64_t *b, const int *stridesB, int64_t *res, const int *stridesRes, const int *shape, int rank);
void s_bitwise_and_uint8(const uint8_t *a, const int *stridesA, const uint8_t *b, const int *stridesB, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_bitwise_and_int16(const int16_t *a, const int *stridesA, const int16_t *b, const int *stridesB, int16_t *res, const int *stridesRes, const int *shape, int rank);

void s_bitwise_or_int32(const int32_t *a, const int *stridesA, const int32_t *b, const int *stridesB, int32_t *res, const int *stridesRes, const int *shape, int rank);
void s_bitwise_or_int64(const int64_t *a, const int *stridesA, const int64_t *b, const int *stridesB, int64_t *res, const int *stridesRes, const int *shape, int rank);
void s_bitwise_or_uint8(const uint8_t *a, const int *stridesA, const uint8_t *b, const int *stridesB, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_bitwise_or_int16(const int16_t *a, const int *stridesA, const int16_t *b, const int *stridesB, int16_t *res, const int *stridesRes, const int *shape, int rank);

void s_bitwise_xor_int32(const int32_t *a, const int *stridesA, const int32_t *b, const int *stridesB, int32_t *res, const int *stridesRes, const int *shape, int rank);
void s_bitwise_xor_int64(const int64_t *a, const int *stridesA, const int64_t *b, const int *stridesB, int64_t *res, const int *stridesRes, const int *shape, int rank);
void s_bitwise_xor_uint8(const uint8_t *a, const int *stridesA, const uint8_t *b, const int *stridesB, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_bitwise_xor_int16(const int16_t *a, const int *stridesA, const int16_t *b, const int *stridesB, int16_t *res, const int *stridesRes, const int *shape, int rank);

void s_left_shift_int32(const int32_t *a, const int *stridesA, const int32_t *b, const int *stridesB, int32_t *res, const int *stridesRes, const int *shape, int rank);
void s_left_shift_int64(const int64_t *a, const int *stridesA, const int64_t *b, const int *stridesB, int64_t *res, const int *stridesRes, const int *shape, int rank);
void s_left_shift_uint8(const uint8_t *a, const int *stridesA, const uint8_t *b, const int *stridesB, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_left_shift_int16(const int16_t *a, const int *stridesA, const int16_t *b, const int *stridesB, int16_t *res, const int *stridesRes, const int *shape, int rank);

void s_right_shift_int32(const int32_t *a, const int *stridesA, const int32_t *b, const int *stridesB, int32_t *res, const int *stridesRes, const int *shape, int rank);
void s_right_shift_int64(const int64_t *a, const int *stridesA, const int64_t *b, const int *stridesB, int64_t *res, const int *stridesRes, const int *shape, int rank);
void s_right_shift_uint8(const uint8_t *a, const int *stridesA, const uint8_t *b, const int *stridesB, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_right_shift_int16(const int16_t *a, const int *stridesA, const int16_t *b, const int *stridesB, int16_t *res, const int *stridesRes, const int *shape, int rank);

void s_invert_int32(const int32_t *src, const int *stridesSrc, int32_t *res, const int *stridesRes, const int *shape, int rank);
void s_invert_int64(const int64_t *src, const int *stridesSrc, int64_t *res, const int *stridesRes, const int *shape, int rank);
void s_invert_uint8(const uint8_t *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_invert_int16(const int16_t *src, const int *stridesSrc, int16_t *res, const int *stridesRes, const int *shape, int rank);


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

#define GENERATE_COMMUTATIVE_COMBINATIONS(OP, MACRO) \
  MACRO(OP, double, double, double, double, double, double) \
  MACRO(OP, double, float, double, double, float, double) \
  MACRO(OP, double, int64, double, double, int64_t, double) \
  MACRO(OP, double, int32, double, double, int32_t, double) \
  MACRO(OP, double, uint8, double, double, uint8_t, double) \
  MACRO(OP, double, int16, double, double, int16_t, double) \
  MACRO(OP, double, cpx, cpx, double, cpx_t, cpx_t) \
  MACRO(OP, double, cpx64, cpx, double, cpx_f_t, cpx_t) \
  MACRO(OP, float, float, float, float, float, float) \
  MACRO(OP, float, int64, float, float, int64_t, float) \
  MACRO(OP, float, int32, float, float, int32_t, float) \
  MACRO(OP, float, uint8, float, float, uint8_t, float) \
  MACRO(OP, float, int16, float, float, int16_t, float) \
  MACRO(OP, float, cpx, cpx, float, cpx_t, cpx_t) \
  MACRO(OP, float, cpx64, cpx64, float, cpx_f_t, cpx_f_t) \
  MACRO(OP, int64, int64, int64, int64_t, int64_t, int64_t) \
  MACRO(OP, int64, int32, int64, int64_t, int32_t, int64_t) \
  MACRO(OP, int64, uint8, int64, int64_t, uint8_t, int64_t) \
  MACRO(OP, int64, int16, int64, int64_t, int16_t, int64_t) \
  MACRO(OP, int64, cpx, cpx, int64_t, cpx_t, cpx_t) \
  MACRO(OP, int64, cpx64, cpx64, int64_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, int32, int32, int32, int32_t, int32_t, int32_t) \
  MACRO(OP, int32, uint8, int32, int32_t, uint8_t, int32_t) \
  MACRO(OP, int32, int16, int32, int32_t, int16_t, int32_t) \
  MACRO(OP, int32, cpx, cpx, int32_t, cpx_t, cpx_t) \
  MACRO(OP, int32, cpx64, cpx64, int32_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, uint8, uint8, uint8, uint8_t, uint8_t, uint8_t) \
  MACRO(OP, uint8, int16, int16, uint8_t, int16_t, int16_t) \
  MACRO(OP, uint8, cpx, cpx, uint8_t, cpx_t, cpx_t) \
  MACRO(OP, uint8, cpx64, cpx64, uint8_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, int16, int16, int16, int16_t, int16_t, int16_t) \
  MACRO(OP, int16, cpx, cpx, int16_t, cpx_t, cpx_t) \
  MACRO(OP, int16, cpx64, cpx64, int16_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, cpx, cpx, cpx, cpx_t, cpx_t, cpx_t) \
  MACRO(OP, cpx, cpx64, cpx, cpx_t, cpx_f_t, cpx_t) \
  MACRO(OP, cpx64, cpx64, cpx64, cpx_f_t, cpx_f_t, cpx_f_t)

#define GENERATE_OP_COMBINATIONS(OP, MACRO) \
  MACRO(OP, double, double, double, double, double, double) \
  MACRO(OP, double, float, double, double, float, double) \
  MACRO(OP, double, int64, double, double, int64_t, double) \
  MACRO(OP, double, int32, double, double, int32_t, double) \
  MACRO(OP, double, uint8, double, double, uint8_t, double) \
  MACRO(OP, double, int16, double, double, int16_t, double) \
  MACRO(OP, double, cpx, cpx, double, cpx_t, cpx_t) \
  MACRO(OP, float, double, double, float, double, double) \
  MACRO(OP, float, float, float, float, float, float) \
  MACRO(OP, float, int64, float, float, int64_t, float) \
  MACRO(OP, float, int32, float, float, int32_t, float) \
  MACRO(OP, float, uint8, float, float, uint8_t, float) \
  MACRO(OP, float, int16, float, float, int16_t, float) \
  MACRO(OP, float, cpx, cpx, float, cpx_t, cpx_t) \
  MACRO(OP, int64, double, double, int64_t, double, double) \
  MACRO(OP, int64, float, float, int64_t, float, float) \
  MACRO(OP, int64, int64, int64, int64_t, int64_t, int64_t) \
  MACRO(OP, int64, int32, int64, int64_t, int32_t, int64_t) \
  MACRO(OP, int64, uint8, int64, int64_t, uint8_t, int64_t) \
  MACRO(OP, int64, int16, int64, int64_t, int16_t, int64_t) \
  MACRO(OP, int64, cpx, cpx, int64_t, cpx_t, cpx_t) \
  MACRO(OP, int32, double, double, int32_t, double, double) \
  MACRO(OP, int32, float, float, int32_t, float, float) \
  MACRO(OP, int32, int64, int64, int32_t, int64_t, int64_t) \
  MACRO(OP, int32, int32, int32, int32_t, int32_t, int32_t) \
  MACRO(OP, int32, uint8, int32, int32_t, uint8_t, int32_t) \
  MACRO(OP, int32, int16, int32, int32_t, int16_t, int32_t) \
  MACRO(OP, int32, cpx, cpx, int32_t, cpx_t, cpx_t) \
  MACRO(OP, uint8, double, double, uint8_t, double, double) \
  MACRO(OP, uint8, float, float, uint8_t, float, float) \
  MACRO(OP, uint8, int64, int64, uint8_t, int64_t, int64_t) \
  MACRO(OP, uint8, int32, int32, uint8_t, int32_t, int32_t) \
  MACRO(OP, uint8, uint8, uint8, uint8_t, uint8_t, uint8_t) \
  MACRO(OP, uint8, int16, int16, uint8_t, int16_t, int16_t) \
  MACRO(OP, uint8, cpx, cpx, uint8_t, cpx_t, cpx_t) \
  MACRO(OP, int16, double, double, int16_t, double, double) \
  MACRO(OP, int16, float, float, int16_t, float, float) \
  MACRO(OP, int16, int64, int64, int16_t, int64_t, int64_t) \
  MACRO(OP, int16, int32, int32, int16_t, int32_t, int32_t) \
  MACRO(OP, int16, uint8, int16, int16_t, uint8_t, int16_t) \
  MACRO(OP, int16, int16, int16, int16_t, int16_t, int16_t) \
  MACRO(OP, int16, cpx, cpx, int16_t, cpx_t, cpx_t) \
  MACRO(OP, cpx, double, cpx, cpx_t, double, cpx_t) \
  MACRO(OP, cpx, float, cpx, cpx_t, float, cpx_t) \
  MACRO(OP, cpx, int64, cpx, cpx_t, int64_t, cpx_t) \
  MACRO(OP, cpx, int32, cpx, cpx_t, int32_t, cpx_t) \
  MACRO(OP, cpx, uint8, cpx, cpx_t, uint8_t, cpx_t) \
  MACRO(OP, cpx, int16, cpx, cpx_t, int16_t, cpx_t) \
  MACRO(OP, cpx, cpx, cpx, cpx_t, cpx_t, cpx_t) \
  MACRO(OP, cpx64, double, cpx, cpx_f_t, double, cpx_t) \
  MACRO(OP, cpx64, float, cpx64, cpx_f_t, float, cpx_f_t) \
  MACRO(OP, cpx64, int64, cpx64, cpx_f_t, int64_t, cpx_f_t) \
  MACRO(OP, cpx64, int32, cpx64, cpx_f_t, int32_t, cpx_f_t) \
  MACRO(OP, cpx64, uint8, cpx64, cpx_f_t, uint8_t, cpx_f_t) \
  MACRO(OP, cpx64, int16, cpx64, cpx_f_t, int16_t, cpx_f_t) \
  MACRO(OP, cpx64, cpx, cpx, cpx_f_t, cpx_t, cpx_t) \
  MACRO(OP, double, cpx64, cpx, double, cpx_f_t, cpx_t) \
  MACRO(OP, float, cpx64, cpx64, float, cpx_f_t, cpx_f_t) \
  MACRO(OP, int64, cpx64, cpx64, int64_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, int32, cpx64, cpx64, int32_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, uint8, cpx64, cpx64, uint8_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, int16, cpx64, cpx64, int16_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, cpx, cpx64, cpx, cpx_t, cpx_f_t, cpx_t) \
  MACRO(OP, cpx64, cpx64, cpx64, cpx_f_t, cpx_f_t, cpx_f_t)

#define GENERATE_DIV_COMBINATIONS(OP, MACRO) \
  MACRO(OP, double, double, double, double, double, double) \
  MACRO(OP, double, float, double, double, float, double) \
  MACRO(OP, double, int64, double, double, int64_t, double) \
  MACRO(OP, double, int32, double, double, int32_t, double) \
  MACRO(OP, double, uint8, double, double, uint8_t, double) \
  MACRO(OP, double, int16, double, double, int16_t, double) \
  MACRO(OP, double, cpx, cpx, double, cpx_t, cpx_t) \
  MACRO(OP, float, double, double, float, double, double) \
  MACRO(OP, float, float, float, float, float, float) \
  MACRO(OP, float, int64, float, float, int64_t, float) \
  MACRO(OP, float, int32, float, float, int32_t, float) \
  MACRO(OP, float, uint8, float, float, uint8_t, float) \
  MACRO(OP, float, int16, float, float, int16_t, float) \
  MACRO(OP, float, cpx, cpx, float, cpx_t, cpx_t) \
  MACRO(OP, int64, double, double, int64_t, double, double) \
  MACRO(OP, int64, float, float, int64_t, float, float) \
  MACRO(OP, int64, int64, double, int64_t, int64_t, double) \
  MACRO(OP, int64, int32, double, int64_t, int32_t, double) \
  MACRO(OP, int64, uint8, double, int64_t, uint8_t, double) \
  MACRO(OP, int64, int16, double, int64_t, int16_t, double) \
  MACRO(OP, int64, cpx, cpx, int64_t, cpx_t, cpx_t) \
  MACRO(OP, int32, double, double, int32_t, double, double) \
  MACRO(OP, int32, float, float, int32_t, float, float) \
  MACRO(OP, int32, int64, double, int32_t, int64_t, double) \
  MACRO(OP, int32, int32, double, int32_t, int32_t, double) \
  MACRO(OP, int32, uint8, double, int32_t, uint8_t, double) \
  MACRO(OP, int32, int16, double, int32_t, int16_t, double) \
  MACRO(OP, int32, cpx, cpx, int32_t, cpx_t, cpx_t) \
  MACRO(OP, uint8, double, double, uint8_t, double, double) \
  MACRO(OP, uint8, float, float, uint8_t, float, float) \
  MACRO(OP, uint8, int64, double, uint8_t, int64_t, double) \
  MACRO(OP, uint8, int32, double, uint8_t, int32_t, double) \
  MACRO(OP, uint8, uint8, double, uint8_t, uint8_t, double) \
  MACRO(OP, uint8, int16, double, uint8_t, int16_t, double) \
  MACRO(OP, uint8, cpx, cpx, uint8_t, cpx_t, cpx_t) \
  MACRO(OP, int16, double, double, int16_t, double, double) \
  MACRO(OP, int16, float, float, int16_t, float, float) \
  MACRO(OP, int16, int64, double, int16_t, int64_t, double) \
  MACRO(OP, int16, int32, double, int16_t, int32_t, double) \
  MACRO(OP, int16, uint8, double, int16_t, uint8_t, double) \
  MACRO(OP, int16, int16, double, int16_t, int16_t, double) \
  MACRO(OP, int16, cpx, cpx, int16_t, cpx_t, cpx_t) \
  MACRO(OP, cpx, double, cpx, cpx_t, double, cpx_t) \
  MACRO(OP, cpx, float, cpx, cpx_t, float, cpx_t) \
  MACRO(OP, cpx, int64, cpx, cpx_t, int64_t, cpx_t) \
  MACRO(OP, cpx, int32, cpx, cpx_t, int32_t, cpx_t) \
  MACRO(OP, cpx, uint8, cpx, cpx_t, uint8_t, cpx_t) \
  MACRO(OP, cpx, int16, cpx, cpx_t, int16_t, cpx_t) \
  MACRO(OP, cpx, cpx, cpx, cpx_t, cpx_t, cpx_t) \
  MACRO(OP, cpx64, double, cpx, cpx_f_t, double, cpx_t) \
  MACRO(OP, cpx64, float, cpx64, cpx_f_t, float, cpx_f_t) \
  MACRO(OP, cpx64, int64, cpx64, cpx_f_t, int64_t, cpx_f_t) \
  MACRO(OP, cpx64, int32, cpx64, cpx_f_t, int32_t, cpx_f_t) \
  MACRO(OP, cpx64, uint8, cpx64, cpx_f_t, uint8_t, cpx_f_t) \
  MACRO(OP, cpx64, int16, cpx64, cpx_f_t, int16_t, cpx_f_t) \
  MACRO(OP, cpx64, cpx, cpx, cpx_f_t, cpx_t, cpx_t) \
  MACRO(OP, double, cpx64, cpx, double, cpx_f_t, cpx_t) \
  MACRO(OP, float, cpx64, cpx64, float, cpx_f_t, cpx_f_t) \
  MACRO(OP, int64, cpx64, cpx64, int64_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, int32, cpx64, cpx64, int32_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, uint8, cpx64, cpx64, uint8_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, int16, cpx64, cpx64, int16_t, cpx_f_t, cpx_f_t) \
  MACRO(OP, cpx, cpx64, cpx, cpx_t, cpx_f_t, cpx_t) \
  MACRO(OP, cpx64, cpx64, cpx64, cpx_f_t, cpx_f_t, cpx_f_t)

#define DECLARE_FFI_HELPER(OP, Ta_tok, Tb_tok, Tr_tok, Ta, Tb, Tr) \
  void v_##OP##_##Ta_tok##_##Tb_tok##_##Tr_tok(const Ta *a, const Tb *b, Tr *res, int size); \
  void s_##OP##_##Ta_tok##_##Tb_tok##_##Tr_tok(const Ta *a, const int *stridesA, \
                                               const Tb *b, const int *stridesB, \
                                               Tr *res, const int *stridesRes, \
                                               const int *shape, int rank);

GENERATE_COMMUTATIVE_COMBINATIONS(add, DECLARE_FFI_HELPER)
GENERATE_OP_COMBINATIONS(sub, DECLARE_FFI_HELPER)
GENERATE_COMMUTATIVE_COMBINATIONS(mul, DECLARE_FFI_HELPER)
GENERATE_DIV_COMBINATIONS(div, DECLARE_FFI_HELPER)

/* ============================================================================
 * SECTION 9: BUFFERED CONVERTER KERNELS (CASTING)
 * ============================================================================
 */
void cast_uint8_to_double(const uint8_t *src, double *dst, int size);
void cast_int16_to_double(const int16_t *src, double *dst, int size);
void cast_double_to_uint8(const double *src, uint8_t *dst, int size);
void cast_double_to_int16(const double *src, int16_t *dst, int size);

void s_cast_uint8_to_double(const uint8_t *src, const int *stridesSrc, double *dst, const int *stridesDst, const int *shape, int rank);
void s_cast_int16_to_double(const int16_t *src, const int *stridesSrc, double *dst, const int *stridesDst, const int *shape, int rank);
void s_cast_double_to_uint8(const double *src, const int *stridesSrc, uint8_t *dst, const int *stridesDst, const int *shape, int rank);
void s_cast_double_to_int16(const double *src, const int *stridesSrc, int16_t *dst, const int *stridesDst, const int *shape, int rank);

/* Logical & Casting-to-Boolean operations */
void v_to_bool_double(const double *src, uint8_t *res, int size);
void v_to_bool_float(const float *src, uint8_t *res, int size);
void v_to_bool_int64(const int64_t *src, uint8_t *res, int size);
void v_to_bool_int32(const int32_t *src, uint8_t *res, int size);
void v_to_bool_uint8(const uint8_t *src, uint8_t *res, int size);
void v_to_bool_int16(const int16_t *src, uint8_t *res, int size);
void v_to_bool_complex128(const cpx_t *src, uint8_t *res, int size);
void v_to_bool_complex64(const cpx_f_t *src, uint8_t *res, int size);

void s_to_bool_double(const double *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_to_bool_float(const float *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_to_bool_int64(const int64_t *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_to_bool_int32(const int32_t *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_to_bool_uint8(const uint8_t *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_to_bool_int16(const int16_t *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_to_bool_complex128(const cpx_t *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_to_bool_complex64(const cpx_f_t *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);

void v_logical_and(const uint8_t *a, const uint8_t *b, uint8_t *res, int size);
void v_logical_or(const uint8_t *a, const uint8_t *b, uint8_t *res, int size);
void v_logical_xor(const uint8_t *a, const uint8_t *b, uint8_t *res, int size);
void v_logical_not(const uint8_t *src, uint8_t *res, int size);

void s_logical_and(const uint8_t *a, const int *stridesA, const uint8_t *b, const int *stridesB, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_logical_or(const uint8_t *a, const int *stridesA, const uint8_t *b, const int *stridesB, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_logical_xor(const uint8_t *a, const int *stridesA, const uint8_t *b, const int *stridesB, uint8_t *res, const int *stridesRes, const int *shape, int rank);
void s_logical_not(const uint8_t *src, const int *stridesSrc, uint8_t *res, const int *stridesRes, const int *shape, int rank);

/* Optimized native advanced indexing recursive copy kernel */
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
);

/* Optimized native boolean mask unpacking kernel */
int unpack_mask_c(
    const uint8_t *mask_ptr,
    int size,
    int stride,
    int *out_indices
);

/* Kronecker Product */
void s_kron_double(const double *a, const int *stridesA, const int *shapeA,
                   const double *b, const int *stridesB, const int *shapeB,
                   double *res, const int *stridesRes, const int *shapeRes, int rank);
void s_kron_float(const float *a, const int *stridesA, const int *shapeA,
                  const float *b, const int *stridesB, const int *shapeB,
                  float *res, const int *stridesRes, const int *shapeRes, int rank);
void s_kron_int64(const int64_t *a, const int *stridesA, const int *shapeA,
                  const int64_t *b, const int *stridesB, const int *shapeB,
                  int64_t *res, const int *stridesRes, const int *shapeRes, int rank);
void s_kron_int32(const int32_t *a, const int *stridesA, const int *shapeA,
                  const int32_t *b, const int *stridesB, const int *shapeB,
                  int32_t *res, const int *stridesRes, const int *shapeRes, int rank);
void s_kron_uint8(const uint8_t *a, const int *stridesA, const int *shapeA,
                  const uint8_t *b, const int *stridesB, const int *shapeB,
                  uint8_t *res, const int *stridesRes, const int *shapeRes, int rank);
void s_kron_int16(const int16_t *a, const int *stridesA, const int *shapeA,
                  const int16_t *b, const int *stridesB, const int *shapeB,
                  int16_t *res, const int *stridesRes, const int *shapeRes, int rank);
void s_kron_complex128(const cpx_t *a, const int *stridesA, const int *shapeA,
                       const cpx_t *b, const int *stridesB, const int *shapeB,
                       cpx_t *res, const int *stridesRes, const int *shapeRes, int rank);
void s_kron_complex64(const cpx_f_t *a, const int *stridesA, const int *shapeA,
                      const cpx_f_t *b, const int *stridesB, const int *shapeB,
                      cpx_f_t *res, const int *stridesRes, const int *shapeRes, int rank);
void s_kron_boolean(const uint8_t *a, const int *stridesA, const int *shapeA,
                    const uint8_t *b, const int *stridesB, const int *shapeB,
                    uint8_t *res, const int *stridesRes, const int *shapeRes, int rank);

/* Vector Outer Product */
void s_outer_double(const double *a, int strideA, int sizeA,
                    const double *b, int strideB, int sizeB,
                    double *res, int strideRowRes, int strideColRes);
void s_outer_float(const float *a, int strideA, int sizeA,
                   const float *b, int strideB, int sizeB,
                   float *res, int strideRowRes, int strideColRes);
void s_outer_int64(const int64_t *a, int strideA, int sizeA,
                   const int64_t *b, int strideB, int sizeB,
                   int64_t *res, int strideRowRes, int strideColRes);
void s_outer_int32(const int32_t *a, int strideA, int sizeA,
                   const int32_t *b, int strideB, int sizeB,
                   int32_t *res, int strideRowRes, int strideColRes);
void s_outer_uint8(const uint8_t *a, int strideA, int sizeA,
                   const uint8_t *b, int strideB, int sizeB,
                   uint8_t *res, int strideRowRes, int strideColRes);
void s_outer_int16(const int16_t *a, int strideA, int sizeA,
                   const int16_t *b, int strideB, int sizeB,
                   int16_t *res, int strideRowRes, int strideColRes);
void s_outer_complex128(const cpx_t *a, int strideA, int sizeA,
                        const cpx_t *b, int strideB, int sizeB,
                        cpx_t *res, int strideRowRes, int strideColRes);
void s_outer_complex64(const cpx_f_t *a, int strideA, int sizeA,
                       const cpx_f_t *b, int strideB, int sizeB,
                       cpx_f_t *res, int strideRowRes, int strideColRes);
void s_outer_boolean(const uint8_t *a, int strideA, int sizeA,
                     const uint8_t *b, int strideB, int sizeB,
                     uint8_t *res, int strideRowRes, int strideColRes);

/* Vector Cross Product */
void s_cross_3d_double(const double *a, int strideA, const double *b, int strideB, double *res, int strideRes);
void s_cross_2d_double(const double *a, int strideA, const double *b, int strideB, double *res);
void s_cross_3d_float(const float *a, int strideA, const float *b, int strideB, float *res, int strideRes);
void s_cross_2d_float(const float *a, int strideA, const float *b, int strideB, float *res);
void s_cross_3d_int64(const int64_t *a, int strideA, const int64_t *b, int strideB, int64_t *res, int strideRes);
void s_cross_2d_int64(const int64_t *a, int strideA, const int64_t *b, int strideB, int64_t *res);
void s_cross_3d_int32(const int32_t *a, int strideA, const int32_t *b, int strideB, int32_t *res, int strideRes);
void s_cross_2d_int32(const int32_t *a, int strideA, const int32_t *b, int strideB, int32_t *res);
void s_cross_3d_uint8(const uint8_t *a, int strideA, const uint8_t *b, int strideB, uint8_t *res, int strideRes);
void s_cross_2d_uint8(const uint8_t *a, int strideA, const uint8_t *b, int strideB, uint8_t *res);
void s_cross_3d_int16(const int16_t *a, int strideA, const int16_t *b, int strideB, int16_t *res, int strideRes);
void s_cross_2d_int16(const int16_t *a, int strideA, const int16_t *b, int strideB, int16_t *res);
void s_cross_3d_complex128(const cpx_t *a, int strideA, const cpx_t *b, int strideB, cpx_t *res, int strideRes);
void s_cross_2d_complex128(const cpx_t *a, int strideA, const cpx_t *b, int strideB, cpx_t *res);
void s_cross_3d_complex64(const cpx_f_t *a, int strideA, const cpx_f_t *b, int strideB, cpx_f_t *res, int strideRes);
void s_cross_2d_complex64(const cpx_f_t *a, int strideA, const cpx_f_t *b, int strideB, cpx_f_t *res);
void s_cross_3d_boolean(const uint8_t *a, int strideA, const uint8_t *b, int strideB, uint8_t *res, int strideRes);
void s_cross_2d_boolean(const uint8_t *a, int strideA, const uint8_t *b, int strideB, uint8_t *res);

/* Vector Norm Reductions */
double r_norm_l1_double(const double *src, int stride, int size);
double r_norm_l2_double(const double *src, int stride, int size);
double r_norm_lp_double(const double *src, int stride, int size, double p);
double r_norm_inf_double(const double *src, int stride, int size);
double r_norm_neg_inf_double(const double *src, int stride, int size);

float r_norm_l1_float(const float *src, int stride, int size);
float r_norm_l2_float(const float *src, int stride, int size);
float r_norm_lp_float(const float *src, int stride, int size, float p);
float r_norm_inf_float(const float *src, int stride, int size);
float r_norm_neg_inf_float(const float *src, int stride, int size);

double r_norm_l1_complex128(const cpx_t *src, int stride, int size);
double r_norm_l2_complex128(const cpx_t *src, int stride, int size);
double r_norm_lp_complex128(const cpx_t *src, int stride, int size, double p);
double r_norm_inf_complex128(const cpx_t *src, int stride, int size);
double r_norm_neg_inf_complex128(const cpx_t *src, int stride, int size);

float r_norm_l1_complex64(const cpx_f_t *src, int stride, int size);
float r_norm_l2_complex64(const cpx_f_t *src, int stride, int size);
float r_norm_lp_complex64(const cpx_f_t *src, int stride, int size, float p);
float r_norm_inf_complex64(const cpx_f_t *src, int stride, int size);
float r_norm_neg_inf_complex64(const cpx_f_t *src, int stride, int size);

/* Window Functions */
void v_hanning_double(double *res, int M);
void v_hanning_float(float *res, int M);
void v_hamming_double(double *res, int M);
void v_hamming_float(float *res, int M);

#endif /* CUSTOM_UFUNCS_H */
