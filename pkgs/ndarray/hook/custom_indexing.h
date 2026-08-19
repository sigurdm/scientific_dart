#pragma once

#include <stddef.h>
#include <stdint.h>

#define DTYPE_FLOAT64 0
#define DTYPE_FLOAT32 1
#define DTYPE_INT32 2
#define DTYPE_INT64 3
#define DTYPE_UINT8 4
#define DTYPE_INT16 5
#define DTYPE_COMPLEX128 6
#define DTYPE_COMPLEX64 7
#define DTYPE_BOOLEAN 8

#ifdef __cplusplus
extern "C" {
#endif

int native_take_along_axis(
    int dtype,
    int index_dtype,
    const void *src,
    const int64_t *arr_shape,
    const int64_t *arr_strides,
    const void *indices,
    const int64_t *idx_shape,
    const int64_t *idx_strides,
    void *dest,
    const int64_t *out_shape,
    const int64_t *out_strides,
    int64_t rank,
    int64_t axis,
    int64_t *out_error_idx
);

int native_put_along_axis(
    int dtype,
    int index_dtype,
    void *target,
    const int64_t *target_shape,
    const int64_t *target_strides,
    const void *indices,
    const int64_t *idx_shape,
    const int64_t *idx_strides,
    const void *values,
    const int64_t *val_shape,
    const int64_t *val_strides,
    int64_t rank,
    int64_t axis,
    int64_t *out_error_idx
);

int native_tile_contiguous(
    int dtype,
    const void *src,
    const int64_t *src_shape,
    const int64_t *reps,
    void *dest,
    const int64_t *out_shape,
    int64_t rank
);

int native_tile_strided(
    int dtype,
    const void *src,
    const int64_t *src_shape,
    const int64_t *src_strides,
    const int64_t *reps,
    void *dest,
    const int64_t *out_shape,
    const int64_t *out_strides,
    int64_t rank
);

#ifdef __cplusplus
}
#endif
