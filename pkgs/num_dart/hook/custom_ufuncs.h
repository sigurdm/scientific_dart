#ifndef CUSTOM_UFUNCS_H
#define CUSTOM_UFUNCS_H

// ----------------------------------------------------------------------------
// Binary-Compatible Complex Number Type (Matches kiss_fft_cpx and num_dart.Complex)
// ----------------------------------------------------------------------------
typedef struct {
    double r; // real part
    double i; // imaginary part
} cpx_t;

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

#endif // CUSTOM_UFUNCS_H
