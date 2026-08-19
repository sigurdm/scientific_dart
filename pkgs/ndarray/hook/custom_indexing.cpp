/* ============================================================================
 * ndarray CUSTOM INDEXING KERNELS (custom_indexing.cpp)
 * ============================================================================
 * High-performance C++ implementation of take_along_axis and put_along_axis
 * for arbitrary rank arrays, strided and contiguous layouts, with fast paths
 * for 1D/2D arrays.
 * ============================================================================
 */

#include "custom_indexing.h"
#include <cstdint>
#include <vector>
#include <cstring>

#if defined(_MSC_VER)
#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#else
#define RESTRICT restrict
#endif

typedef struct {
    double real;
    double imag;
} complex128_t;

typedef struct {
    float real;
    float imag;
} complex64_t;

// ============================================================================
// 1. TAKE_ALONG_AXIS KERNEL TEMPLATE
// ============================================================================

template <typename T, typename IndexT>
static int take_along_axis_impl(
    const T *RESTRICT src,
    const int64_t *RESTRICT arr_shape,
    const int64_t *RESTRICT arr_strides,
    const IndexT *RESTRICT idx_ptr,
    const int64_t *RESTRICT idx_shape,
    const int64_t *RESTRICT idx_strides,
    T *RESTRICT dest,
    const int64_t *RESTRICT out_shape,
    const int64_t *RESTRICT out_strides,
    int64_t rank,
    int64_t axis,
    int64_t *RESTRICT out_error_idx
) {
    if (rank <= 0) return 0;
    const int64_t axis_size = arr_shape[axis];
    if (axis_size <= 0) return 0;

    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) {
        total_elements *= out_shape[i];
    }
    if (total_elements == 0) return 0;

    // --- Fast Path: 1D ---
    if (rank == 1) {
        const int64_t arr_s = arr_strides[0];
        const int64_t idx_s = idx_strides[0];
        const int64_t out_s = out_strides[0];

        if (arr_s == 1 && idx_s == 1 && out_s == 1) {
            for (int64_t i = 0; i < total_elements; i++) {
                IndexT raw_idx = idx_ptr[i];
                int64_t idx = (int64_t)raw_idx;
                if (idx < 0) idx += axis_size;
                if ((uint64_t)idx >= (uint64_t)axis_size) {
                    *out_error_idx = (int64_t)raw_idx;
                    return -1;
                }
                dest[i] = src[idx];
            }
            return 0;
        } else {
            for (int64_t i = 0; i < total_elements; i++) {
                IndexT raw_idx = idx_ptr[i * idx_s];
                int64_t idx = (int64_t)raw_idx;
                if (idx < 0) idx += axis_size;
                if ((uint64_t)idx >= (uint64_t)axis_size) {
                    *out_error_idx = (int64_t)raw_idx;
                    return -1;
                }
                dest[i * out_s] = src[idx * arr_s];
            }
            return 0;
        }
    }

    // --- Fast Path: 2D Contiguous (Row-Major) ---
    if (rank == 2) {
        const int64_t M_out = out_shape[0];
        const int64_t N_out = out_shape[1];

        bool out_c = (out_strides[0] == N_out && out_strides[1] == 1);
        bool arr_c = (arr_strides[0] == arr_shape[1] && arr_strides[1] == 1);

        // Case 2A: axis == 0 (gather across rows)
        if (axis == 0 && out_c && arr_c) {
            if (arr_shape[1] == N_out && (idx_shape[1] == N_out || idx_shape[1] == 1)) {
                const int64_t arr_row_stride = arr_strides[0];
                const int64_t idx_s0 = idx_strides[0];
                const int64_t idx_s1 = (idx_shape[1] == 1 ? 0 : idx_strides[1]);

                for (int64_t i = 0; i < M_out; i++) {
                    const IndexT *row_idx = idx_ptr + i * idx_s0;
                    T *row_dest = dest + i * N_out;
                    for (int64_t j = 0; j < N_out; j++) {
                        IndexT raw_idx = row_idx[j * idx_s1];
                        int64_t idx = (int64_t)raw_idx;
                        if (idx < 0) idx += axis_size;
                        if ((uint64_t)idx >= (uint64_t)axis_size) {
                            *out_error_idx = (int64_t)raw_idx;
                            return -1;
                        }
                        row_dest[j] = src[idx * arr_row_stride + j];
                    }
                }
                return 0;
            }
        }

        // Case 2B: axis == 1 (gather across cols)
        if (axis == 1 && out_c && arr_c) {
            if (arr_shape[0] == M_out && (idx_shape[0] == M_out || idx_shape[0] == 1)) {
                const int64_t arr_row_stride = arr_strides[0];
                const int64_t idx_s0 = (idx_shape[0] == 1 ? 0 : idx_strides[0]);
                const int64_t idx_s1 = idx_strides[1];

                for (int64_t i = 0; i < M_out; i++) {
                    const T *row_src = src + i * arr_row_stride;
                    const IndexT *row_idx = idx_ptr + i * idx_s0;
                    T *row_dest = dest + i * N_out;
                    for (int64_t j = 0; j < N_out; j++) {
                        IndexT raw_idx = row_idx[j * idx_s1];
                        int64_t idx = (int64_t)raw_idx;
                        if (idx < 0) idx += axis_size;
                        if ((uint64_t)idx >= (uint64_t)axis_size) {
                            *out_error_idx = (int64_t)raw_idx;
                            return -1;
                        }
                        row_dest[j] = row_src[idx];
                    }
                }
                return 0;
            }
        }
    }

    // --- General Multi-Dimensional Strided Path ---
    int64_t eff_idx_strides[32];
    int64_t eff_arr_base_strides[32];
    std::vector<int64_t> eff_idx_vec, eff_arr_vec;
    int64_t *p_eff_idx = eff_idx_strides;
    int64_t *p_eff_arr = eff_arr_base_strides;
    if (rank > 32) {
        eff_idx_vec.resize(rank);
        eff_arr_vec.resize(rank);
        p_eff_idx = eff_idx_vec.data();
        p_eff_arr = eff_arr_vec.data();
    }

    for (int64_t d = 0; d < rank; d++) {
        p_eff_idx[d] = (idx_shape[d] == 1 ? 0 : idx_strides[d]);
        if (d == axis) {
            p_eff_arr[d] = 0;
        } else {
            p_eff_arr[d] = (arr_shape[d] == 1 ? 0 : arr_strides[d]);
        }
    }

    const int64_t axis_stride = arr_strides[axis];

    int64_t coord_stack[32] = {0};
    std::vector<int64_t> coord_vec;
    int64_t *coord = coord_stack;
    if (rank > 32) {
        coord_vec.assign(rank, 0);
        coord = coord_vec.data();
    }

    int64_t offsetOut = 0;
    int64_t offsetIdx = 0;
    int64_t baseOffsetArr = 0;

    for (int64_t el = 0; el < total_elements; el++) {
        IndexT raw_idx = idx_ptr[offsetIdx];
        int64_t idx = (int64_t)raw_idx;
        if (idx < 0) idx += axis_size;
        if ((uint64_t)idx >= (uint64_t)axis_size) {
            *out_error_idx = (int64_t)raw_idx;
            return -1;
        }
        int64_t offsetArr = baseOffsetArr + idx * axis_stride;
        dest[offsetOut] = src[offsetArr];

        for (int64_t d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < out_shape[d]) {
                offsetOut += out_strides[d];
                offsetIdx += p_eff_idx[d];
                baseOffsetArr += p_eff_arr[d];
                break;
            }
            coord[d] = 0;
            offsetOut -= (out_shape[d] - 1) * out_strides[d];
            offsetIdx -= (out_shape[d] - 1) * p_eff_idx[d];
            baseOffsetArr -= (out_shape[d] - 1) * p_eff_arr[d];
        }
    }

    return 0;
}

template <typename IndexT>
static int dispatch_take_along_axis_by_dtype(
    int dtype,
    const void *src,
    const int64_t *arr_shape,
    const int64_t *arr_strides,
    const IndexT *idx_ptr,
    const int64_t *idx_shape,
    const int64_t *idx_strides,
    void *dest,
    const int64_t *out_shape,
    const int64_t *out_strides,
    int64_t rank,
    int64_t axis,
    int64_t *out_error_idx
) {
    switch (dtype) {
        case DTYPE_FLOAT64:
            return take_along_axis_impl<double, IndexT>(
                (const double *)src, arr_shape, arr_strides,
                idx_ptr, idx_shape, idx_strides,
                (double *)dest, out_shape, out_strides,
                rank, axis, out_error_idx);
        case DTYPE_FLOAT32:
            return take_along_axis_impl<float, IndexT>(
                (const float *)src, arr_shape, arr_strides,
                idx_ptr, idx_shape, idx_strides,
                (float *)dest, out_shape, out_strides,
                rank, axis, out_error_idx);
        case DTYPE_INT64:
            return take_along_axis_impl<int64_t, IndexT>(
                (const int64_t *)src, arr_shape, arr_strides,
                idx_ptr, idx_shape, idx_strides,
                (int64_t *)dest, out_shape, out_strides,
                rank, axis, out_error_idx);
        case DTYPE_INT32:
            return take_along_axis_impl<int32_t, IndexT>(
                (const int32_t *)src, arr_shape, arr_strides,
                idx_ptr, idx_shape, idx_strides,
                (int32_t *)dest, out_shape, out_strides,
                rank, axis, out_error_idx);
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            return take_along_axis_impl<uint8_t, IndexT>(
                (const uint8_t *)src, arr_shape, arr_strides,
                idx_ptr, idx_shape, idx_strides,
                (uint8_t *)dest, out_shape, out_strides,
                rank, axis, out_error_idx);
        case DTYPE_INT16:
            return take_along_axis_impl<int16_t, IndexT>(
                (const int16_t *)src, arr_shape, arr_strides,
                idx_ptr, idx_shape, idx_strides,
                (int16_t *)dest, out_shape, out_strides,
                rank, axis, out_error_idx);
        case DTYPE_COMPLEX128:
            return take_along_axis_impl<complex128_t, IndexT>(
                (const complex128_t *)src, arr_shape, arr_strides,
                idx_ptr, idx_shape, idx_strides,
                (complex128_t *)dest, out_shape, out_strides,
                rank, axis, out_error_idx);
        case DTYPE_COMPLEX64:
            return take_along_axis_impl<complex64_t, IndexT>(
                (const complex64_t *)src, arr_shape, arr_strides,
                idx_ptr, idx_shape, idx_strides,
                (complex64_t *)dest, out_shape, out_strides,
                rank, axis, out_error_idx);
        default:
            return -2;
    }
}

extern "C" int native_take_along_axis(
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
) {
    if (src == nullptr || indices == nullptr || dest == nullptr ||
        arr_shape == nullptr || arr_strides == nullptr ||
        idx_shape == nullptr || idx_strides == nullptr ||
        out_shape == nullptr || out_strides == nullptr ||
        out_error_idx == nullptr) {
        return -3;
    }

    if (index_dtype == DTYPE_INT64) {
        return dispatch_take_along_axis_by_dtype<int64_t>(
            dtype, src, arr_shape, arr_strides,
            (const int64_t *)indices, idx_shape, idx_strides,
            dest, out_shape, out_strides, rank, axis, out_error_idx);
    } else if (index_dtype == DTYPE_INT32) {
        return dispatch_take_along_axis_by_dtype<int32_t>(
            dtype, src, arr_shape, arr_strides,
            (const int32_t *)indices, idx_shape, idx_strides,
            dest, out_shape, out_strides, rank, axis, out_error_idx);
    } else if (index_dtype == DTYPE_INT16) {
        return dispatch_take_along_axis_by_dtype<int16_t>(
            dtype, src, arr_shape, arr_strides,
            (const int16_t *)indices, idx_shape, idx_strides,
            dest, out_shape, out_strides, rank, axis, out_error_idx);
    } else if (index_dtype == DTYPE_UINT8) {
        return dispatch_take_along_axis_by_dtype<uint8_t>(
            dtype, src, arr_shape, arr_strides,
            (const uint8_t *)indices, idx_shape, idx_strides,
            dest, out_shape, out_strides, rank, axis, out_error_idx);
    } else {
        return -2;
    }
}

// ============================================================================
// 2. PUT_ALONG_AXIS KERNEL TEMPLATE
// ============================================================================

template <typename T, typename IndexT>
static int put_along_axis_impl(
    T *RESTRICT target,
    const int64_t *RESTRICT target_shape,
    const int64_t *RESTRICT target_strides,
    const IndexT *RESTRICT idx_ptr,
    const int64_t *RESTRICT idx_shape,
    const int64_t *RESTRICT idx_strides,
    const T *RESTRICT val_ptr,
    const int64_t *RESTRICT val_shape,
    const int64_t *RESTRICT val_strides,
    int64_t rank,
    int64_t axis,
    int64_t *RESTRICT out_error_idx
) {
    if (rank <= 0) return 0;
    const int64_t axis_size = target_shape[axis];
    if (axis_size <= 0) return 0;

    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) {
        total_elements *= idx_shape[i];
    }
    if (total_elements == 0) return 0;

    // --- Fast Path: 1D ---
    if (rank == 1) {
        const int64_t tgt_s = target_strides[0];
        const int64_t idx_s = idx_strides[0];
        const int64_t val_s = (val_shape[0] == 1 ? 0 : val_strides[0]);

        for (int64_t i = 0; i < total_elements; i++) {
            IndexT raw_idx = idx_ptr[i * idx_s];
            int64_t idx = (int64_t)raw_idx;
            if (idx < 0) idx += axis_size;
            if ((uint64_t)idx >= (uint64_t)axis_size) {
                *out_error_idx = (int64_t)raw_idx;
                return -1;
            }
            target[idx * tgt_s] = val_ptr[i * val_s];
        }
        return 0;
    }

    // --- Fast Path: 2D Contiguous (Row-Major) ---
    if (rank == 2) {
        const int64_t M_idx = idx_shape[0];
        const int64_t N_idx = idx_shape[1];

        bool tgt_c = (target_strides[0] == target_shape[1] && target_strides[1] == 1);

        if (axis == 0 && tgt_c) {
            if (target_shape[1] == N_idx) {
                const int64_t tgt_row_stride = target_strides[0];
                const int64_t idx_s0 = idx_strides[0];
                const int64_t idx_s1 = idx_strides[1];
                const int64_t val_s0 = (val_shape[0] == 1 ? 0 : val_strides[0]);
                const int64_t val_s1 = (val_shape[1] == 1 ? 0 : val_strides[1]);

                for (int64_t i = 0; i < M_idx; i++) {
                    const IndexT *row_idx = idx_ptr + i * idx_s0;
                    const T *row_val = val_ptr + i * val_s0;
                    for (int64_t j = 0; j < N_idx; j++) {
                        IndexT raw_idx = row_idx[j * idx_s1];
                        int64_t idx = (int64_t)raw_idx;
                        if (idx < 0) idx += axis_size;
                        if ((uint64_t)idx >= (uint64_t)axis_size) {
                            *out_error_idx = (int64_t)raw_idx;
                            return -1;
                        }
                        target[idx * tgt_row_stride + j] = row_val[j * val_s1];
                    }
                }
                return 0;
            }
        }

        if (axis == 1 && tgt_c) {
            if (target_shape[0] == M_idx) {
                const int64_t tgt_row_stride = target_strides[0];
                const int64_t idx_s0 = idx_strides[0];
                const int64_t idx_s1 = idx_strides[1];
                const int64_t val_s0 = (val_shape[0] == 1 ? 0 : val_strides[0]);
                const int64_t val_s1 = (val_shape[1] == 1 ? 0 : val_strides[1]);

                for (int64_t i = 0; i < M_idx; i++) {
                    T *row_tgt = target + i * tgt_row_stride;
                    const IndexT *row_idx = idx_ptr + i * idx_s0;
                    const T *row_val = val_ptr + i * val_s0;
                    for (int64_t j = 0; j < N_idx; j++) {
                        IndexT raw_idx = row_idx[j * idx_s1];
                        int64_t idx = (int64_t)raw_idx;
                        if (idx < 0) idx += axis_size;
                        if ((uint64_t)idx >= (uint64_t)axis_size) {
                            *out_error_idx = (int64_t)raw_idx;
                            return -1;
                        }
                        row_tgt[idx] = row_val[j * val_s1];
                    }
                }
                return 0;
            }
        }
    }

    // --- General Multi-Dimensional Strided Path ---
    int64_t eff_val_strides[32];
    int64_t eff_tgt_base_strides[32];
    std::vector<int64_t> eff_val_vec, eff_tgt_vec;
    int64_t *p_eff_val = eff_val_strides;
    int64_t *p_eff_tgt = eff_tgt_base_strides;
    if (rank > 32) {
        eff_val_vec.resize(rank);
        eff_tgt_vec.resize(rank);
        p_eff_val = eff_val_vec.data();
        p_eff_tgt = eff_tgt_vec.data();
    }

    for (int64_t d = 0; d < rank; d++) {
        p_eff_val[d] = (val_shape[d] == 1 ? 0 : val_strides[d]);
        if (d == axis) {
            p_eff_tgt[d] = 0;
        } else {
            p_eff_tgt[d] = (target_shape[d] == 1 ? 0 : target_strides[d]);
        }
    }

    const int64_t axis_stride = target_strides[axis];

    int64_t coord_stack[32] = {0};
    std::vector<int64_t> coord_vec;
    int64_t *coord = coord_stack;
    if (rank > 32) {
        coord_vec.assign(rank, 0);
        coord = coord_vec.data();
    }

    int64_t offsetIdx = 0;
    int64_t offsetVal = 0;
    int64_t baseOffsetTarget = 0;

    for (int64_t el = 0; el < total_elements; el++) {
        IndexT raw_idx = idx_ptr[offsetIdx];
        int64_t idx = (int64_t)raw_idx;
        if (idx < 0) idx += axis_size;
        if ((uint64_t)idx >= (uint64_t)axis_size) {
            *out_error_idx = (int64_t)raw_idx;
            return -1;
        }
        int64_t offsetTarget = baseOffsetTarget + idx * axis_stride;
        target[offsetTarget] = val_ptr[offsetVal];

        for (int64_t d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < idx_shape[d]) {
                offsetIdx += idx_strides[d];
                offsetVal += p_eff_val[d];
                baseOffsetTarget += p_eff_tgt[d];
                break;
            }
            coord[d] = 0;
            offsetIdx -= (idx_shape[d] - 1) * idx_strides[d];
            offsetVal -= (idx_shape[d] - 1) * p_eff_val[d];
            baseOffsetTarget -= (idx_shape[d] - 1) * p_eff_tgt[d];
        }
    }

    return 0;
}

template <typename IndexT>
static int dispatch_put_along_axis_by_dtype(
    int dtype,
    void *target,
    const int64_t *target_shape,
    const int64_t *target_strides,
    const IndexT *idx_ptr,
    const int64_t *idx_shape,
    const int64_t *idx_strides,
    const void *values,
    const int64_t *val_shape,
    const int64_t *val_strides,
    int64_t rank,
    int64_t axis,
    int64_t *out_error_idx
) {
    switch (dtype) {
        case DTYPE_FLOAT64:
            return put_along_axis_impl<double, IndexT>(
                (double *)target, target_shape, target_strides,
                idx_ptr, idx_shape, idx_strides,
                (const double *)values, val_shape, val_strides,
                rank, axis, out_error_idx);
        case DTYPE_FLOAT32:
            return put_along_axis_impl<float, IndexT>(
                (float *)target, target_shape, target_strides,
                idx_ptr, idx_shape, idx_strides,
                (const float *)values, val_shape, val_strides,
                rank, axis, out_error_idx);
        case DTYPE_INT64:
            return put_along_axis_impl<int64_t, IndexT>(
                (int64_t *)target, target_shape, target_strides,
                idx_ptr, idx_shape, idx_strides,
                (const int64_t *)values, val_shape, val_strides,
                rank, axis, out_error_idx);
        case DTYPE_INT32:
            return put_along_axis_impl<int32_t, IndexT>(
                (int32_t *)target, target_shape, target_strides,
                idx_ptr, idx_shape, idx_strides,
                (const int32_t *)values, val_shape, val_strides,
                rank, axis, out_error_idx);
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            return put_along_axis_impl<uint8_t, IndexT>(
                (uint8_t *)target, target_shape, target_strides,
                idx_ptr, idx_shape, idx_strides,
                (const uint8_t *)values, val_shape, val_strides,
                rank, axis, out_error_idx);
        case DTYPE_INT16:
            return put_along_axis_impl<int16_t, IndexT>(
                (int16_t *)target, target_shape, target_strides,
                idx_ptr, idx_shape, idx_strides,
                (const int16_t *)values, val_shape, val_strides,
                rank, axis, out_error_idx);
        case DTYPE_COMPLEX128:
            return put_along_axis_impl<complex128_t, IndexT>(
                (complex128_t *)target, target_shape, target_strides,
                idx_ptr, idx_shape, idx_strides,
                (const complex128_t *)values, val_shape, val_strides,
                rank, axis, out_error_idx);
        case DTYPE_COMPLEX64:
            return put_along_axis_impl<complex64_t, IndexT>(
                (complex64_t *)target, target_shape, target_strides,
                idx_ptr, idx_shape, idx_strides,
                (const complex64_t *)values, val_shape, val_strides,
                rank, axis, out_error_idx);
        default:
            return -2;
    }
}

extern "C" int native_put_along_axis(
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
) {
    if (target == nullptr || indices == nullptr || values == nullptr ||
        target_shape == nullptr || target_strides == nullptr ||
        idx_shape == nullptr || idx_strides == nullptr ||
        val_shape == nullptr || val_strides == nullptr ||
        out_error_idx == nullptr) {
        return -3;
    }

    if (index_dtype == DTYPE_INT64) {
        return dispatch_put_along_axis_by_dtype<int64_t>(
            dtype, target, target_shape, target_strides,
            (const int64_t *)indices, idx_shape, idx_strides,
            values, val_shape, val_strides, rank, axis, out_error_idx);
    } else if (index_dtype == DTYPE_INT32) {
        return dispatch_put_along_axis_by_dtype<int32_t>(
            dtype, target, target_shape, target_strides,
            (const int32_t *)indices, idx_shape, idx_strides,
            values, val_shape, val_strides, rank, axis, out_error_idx);
    } else if (index_dtype == DTYPE_INT16) {
        return dispatch_put_along_axis_by_dtype<int16_t>(
            dtype, target, target_shape, target_strides,
            (const int16_t *)indices, idx_shape, idx_strides,
            values, val_shape, val_strides, rank, axis, out_error_idx);
    } else if (index_dtype == DTYPE_UINT8) {
        return dispatch_put_along_axis_by_dtype<uint8_t>(
            dtype, target, target_shape, target_strides,
            (const uint8_t *)indices, idx_shape, idx_strides,
            values, val_shape, val_strides, rank, axis, out_error_idx);
    } else {
        return -2;
    }
}

// ============================================================================
// 3. TILE / BLOCK REPLICATION KERNELS
// ============================================================================

static void fill_tile_contiguous_nd(
    uint8_t *RESTRICT dest,
    const uint8_t *RESTRICT src,
    const int64_t *RESTRICT src_shape,
    const int64_t *RESTRICT reps,
    const size_t *RESTRICT src_block_bytes,
    const size_t *RESTRICT dest_block_bytes,
    int64_t dim,
    int64_t rank,
    size_t itemsize
) {
    if (dim == rank - 1) {
        size_t s_dim = (size_t)src_shape[dim];
        size_t r_dim = (size_t)reps[dim];
        size_t chunk_bytes = s_dim * itemsize;
        size_t total_bytes = r_dim * chunk_bytes;

        memcpy(dest, src, chunk_bytes);
        size_t copied = chunk_bytes;
        while (copied < total_bytes) {
            size_t to_copy = (copied <= total_bytes - copied) ? copied : (total_bytes - copied);
            memcpy(dest + copied, dest, to_copy);
            copied += to_copy;
        }
        return;
    }

    int64_t count = src_shape[dim];
    size_t child_dest_stride = dest_block_bytes[dim + 1];
    size_t child_src_stride = src_block_bytes[dim + 1];

    for (int64_t i = 0; i < count; i++) {
        fill_tile_contiguous_nd(
            dest + i * child_dest_stride,
            src + i * child_src_stride,
            src_shape,
            reps,
            src_block_bytes,
            dest_block_bytes,
            dim + 1,
            rank,
            itemsize
        );
    }

    size_t block_bytes = (size_t)count * child_dest_stride;
    size_t total_dim_bytes = (size_t)reps[dim] * block_bytes;
    size_t copied_block = block_bytes;
    while (copied_block < total_dim_bytes) {
        size_t to_copy = (copied_block <= total_dim_bytes - copied_block) ? copied_block : (total_dim_bytes - copied_block);
        memcpy(dest + copied_block, dest, to_copy);
        copied_block += to_copy;
    }
}

extern "C" int native_tile_contiguous(
    int dtype,
    const void *src,
    const int64_t *src_shape,
    const int64_t *reps,
    void *dest,
    const int64_t *out_shape,
    int64_t rank
) {
    if (src == nullptr || dest == nullptr || src_shape == nullptr || reps == nullptr || out_shape == nullptr) {
        return -3;
    }

    size_t itemsize = 0;
    switch (dtype) {
        case DTYPE_FLOAT64:
        case DTYPE_INT64:
        case DTYPE_COMPLEX64:
            itemsize = 8;
            break;
        case DTYPE_FLOAT32:
        case DTYPE_INT32:
            itemsize = 4;
            break;
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            itemsize = 1;
            break;
        case DTYPE_INT16:
            itemsize = 2;
            break;
        case DTYPE_COMPLEX128:
            itemsize = 16;
            break;
        default:
            return -2;
    }

    if (rank <= 0) {
        memcpy(dest, src, itemsize);
        return 0;
    }

    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) {
        if (src_shape[i] <= 0 || reps[i] <= 0 || out_shape[i] <= 0) return 0;
        total_elements *= out_shape[i];
    }
    if (total_elements == 0) return 0;

    // --- Fast Path: 1D ---
    if (rank == 1) {
        size_t chunk_bytes = (size_t)src_shape[0] * itemsize;
        size_t total_bytes = (size_t)reps[0] * chunk_bytes;
        uint8_t *d = (uint8_t *)dest;
        memcpy(d, src, chunk_bytes);
        size_t copied = chunk_bytes;
        while (copied < total_bytes) {
            size_t to_copy = (copied <= total_bytes - copied) ? copied : (total_bytes - copied);
            memcpy(d + copied, d, to_copy);
            copied += to_copy;
        }
        return 0;
    }

    // --- Fast Path: 2D ---
    if (rank == 2) {
        int64_t M = src_shape[0];
        int64_t N = src_shape[1];
        int64_t R_rows = reps[0];
        int64_t R_cols = reps[1];
        size_t row_src_bytes = (size_t)N * itemsize;
        size_t row_dest_bytes = (size_t)N * R_cols * itemsize;
        const uint8_t *s = (const uint8_t *)src;
        uint8_t *d = (uint8_t *)dest;

        for (int64_t i = 0; i < M; i++) {
            uint8_t *dest_row = d + i * row_dest_bytes;
            const uint8_t *src_row = s + i * row_src_bytes;
            memcpy(dest_row, src_row, row_src_bytes);
            size_t copied = row_src_bytes;
            while (copied < row_dest_bytes) {
                size_t to_copy = (copied <= row_dest_bytes - copied) ? copied : (row_dest_bytes - copied);
                memcpy(dest_row + copied, dest_row, to_copy);
                copied += to_copy;
            }
        }

        size_t block_bytes = (size_t)M * row_dest_bytes;
        size_t total_dest_bytes = (size_t)R_rows * block_bytes;
        size_t copied_block = block_bytes;
        while (copied_block < total_dest_bytes) {
            size_t to_copy = (copied_block <= total_dest_bytes - copied_block) ? copied_block : (total_dest_bytes - copied_block);
            memcpy(d + copied_block, d, to_copy);
            copied_block += to_copy;
        }
        return 0;
    }

    // --- N-D General Path ---
    size_t src_block_bytes[32];
    size_t dest_block_bytes[32];
    std::vector<size_t> src_block_vec, dest_block_vec;
    size_t *p_src_block = src_block_bytes;
    size_t *p_dest_block = dest_block_bytes;
    if (rank > 32) {
        src_block_vec.resize(rank);
        dest_block_vec.resize(rank);
        p_src_block = src_block_vec.data();
        p_dest_block = dest_block_vec.data();
    }

    p_src_block[rank - 1] = (size_t)src_shape[rank - 1] * itemsize;
    p_dest_block[rank - 1] = (size_t)out_shape[rank - 1] * itemsize;
    for (int64_t d = rank - 2; d >= 0; d--) {
        p_src_block[d] = p_src_block[d + 1] * (size_t)src_shape[d];
        p_dest_block[d] = p_dest_block[d + 1] * (size_t)out_shape[d];
    }

    fill_tile_contiguous_nd(
        (uint8_t *)dest,
        (const uint8_t *)src,
        src_shape,
        reps,
        p_src_block,
        p_dest_block,
        0,
        rank,
        itemsize
    );

    return 0;
}

template <typename T>
static int tile_strided_impl(
    const T *RESTRICT src,
    const int64_t *RESTRICT src_shape,
    const int64_t *RESTRICT src_strides,
    const int64_t *RESTRICT reps,
    T *RESTRICT dest,
    const int64_t *RESTRICT out_shape,
    const int64_t *RESTRICT out_strides,
    int64_t rank
) {
    if (rank <= 0) {
        dest[0] = src[0];
        return 0;
    }
    int64_t total_elements = 1;
    for (int64_t i = 0; i < rank; i++) {
        if (src_shape[i] <= 0 || reps[i] <= 0 || out_shape[i] <= 0) return 0;
        total_elements *= out_shape[i];
    }
    if (total_elements == 0) return 0;

    int64_t coord_stack[32] = {0};
    std::vector<int64_t> coord_vec;
    int64_t *coord = coord_stack;
    if (rank > 32) {
        coord_vec.assign(rank, 0);
        coord = coord_vec.data();
    }

    int64_t offsetOut = 0;
    for (int64_t el = 0; el < total_elements; el++) {
        int64_t offsetSrc = 0;
        for (int64_t d = 0; d < rank; d++) {
            int64_t src_coord = coord[d] % src_shape[d];
            offsetSrc += src_coord * src_strides[d];
        }
        dest[offsetOut] = src[offsetSrc];

        for (int64_t d = rank - 1; d >= 0; d--) {
            coord[d]++;
            if (coord[d] < out_shape[d]) {
                offsetOut += out_strides[d];
                break;
            }
            coord[d] = 0;
            offsetOut -= (out_shape[d] - 1) * out_strides[d];
        }
    }
    return 0;
}

extern "C" int native_tile_strided(
    int dtype,
    const void *src,
    const int64_t *src_shape,
    const int64_t *src_strides,
    const int64_t *reps,
    void *dest,
    const int64_t *out_shape,
    const int64_t *out_strides,
    int64_t rank
) {
    if (src == nullptr || dest == nullptr || src_shape == nullptr ||
        src_strides == nullptr || reps == nullptr || out_shape == nullptr ||
        out_strides == nullptr) {
        return -3;
    }
    switch (dtype) {
        case DTYPE_FLOAT64:
            return tile_strided_impl<double>(
                (const double *)src, src_shape, src_strides, reps,
                (double *)dest, out_shape, out_strides, rank);
        case DTYPE_FLOAT32:
            return tile_strided_impl<float>(
                (const float *)src, src_shape, src_strides, reps,
                (float *)dest, out_shape, out_strides, rank);
        case DTYPE_INT64:
            return tile_strided_impl<int64_t>(
                (const int64_t *)src, src_shape, src_strides, reps,
                (int64_t *)dest, out_shape, out_strides, rank);
        case DTYPE_INT32:
            return tile_strided_impl<int32_t>(
                (const int32_t *)src, src_shape, src_strides, reps,
                (int32_t *)dest, out_shape, out_strides, rank);
        case DTYPE_UINT8:
        case DTYPE_BOOLEAN:
            return tile_strided_impl<uint8_t>(
                (const uint8_t *)src, src_shape, src_strides, reps,
                (uint8_t *)dest, out_shape, out_strides, rank);
        case DTYPE_INT16:
            return tile_strided_impl<int16_t>(
                (const int16_t *)src, src_shape, src_strides, reps,
                (int16_t *)dest, out_shape, out_strides, rank);
        case DTYPE_COMPLEX128:
            return tile_strided_impl<complex128_t>(
                (const complex128_t *)src, src_shape, src_strides, reps,
                (complex128_t *)dest, out_shape, out_strides, rank);
        case DTYPE_COMPLEX64:
            return tile_strided_impl<complex64_t>(
                (const complex64_t *)src, src_shape, src_strides, reps,
                (complex64_t *)dest, out_shape, out_strides, rank);
        default:
            return -2;
    }
}

