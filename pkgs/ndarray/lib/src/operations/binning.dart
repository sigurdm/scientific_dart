// ignore_for_file: non_constant_identifier_names
import 'dart:math' as math;
import 'dart:ffi' as ffi;
import '../ndarray.dart';
import '../ndarray_bindings.dart';
import '../scratch_arena.dart';
import 'helpers.dart';
import 'stats.dart'; // For min, max, sum
import 'math.dart'; // For diff, multiply, divide, equal
import 'manipulation.dart'; // For flip, ravel, where
import 'sorting.dart'; // For searchsorted, count_nonzero
import 'spacers.dart'; // For linspace

// Helper to check list equality
bool _listEquals<T>(List<T>? a, List<T>? b) {
  if (a == null) return b == null;
  if (b == null) return false;
  if (identical(a, b)) return true;
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

// Fast copy and cast between NDArrays using FFI s_cast_generic
void _fastCopyAndCast(NDArray src, NDArray dest) {
  assert(src.size == dest.size);
  final ndim = src.shape.length;
  final marker = ScratchArena.marker;
  try {
    final cBuffer = ScratchArena.getStridedBuffer(ndim * 3);
    final cShape = cBuffer;
    final cStridesSrc = cBuffer + ndim;
    final cStridesDest = cBuffer + ndim * 2;

    for (var i = 0; i < ndim; i++) {
      cShape[i] = src.shape[i];
      cStridesSrc[i] = src.strides[i];
      cStridesDest[i] = dest.strides[i];
    }

    s_cast_generic(
      src.pointer.cast(),
      cStridesSrc,
      encodeDType(src.dtype),
      dest.pointer.cast(),
      encodeDType(dest.dtype),
      cShape,
      ndim,
    );
  } finally {
    ScratchArena.reset(marker);
  }
}

/// Computes the frequency of each value in an array of non-negative ints.
NDArray<T> bincount<T extends num>(
  NDArray<int> x, {
  NDArray<T>? weights,
  int? minlength,
  NDArray<T>? out,
}) {
  if (x.isDisposed) {
    throw StateError('Cannot compute bincount of a disposed array.');
  }
  if (x.shape.length != 1) {
    throw ArgumentError('Input array x must be 1D.');
  }
  if (minlength != null && minlength < 0) {
    throw ArgumentError('minlength must be non-negative.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Output array is disposed.');
  }

  return NDArray.scope(() {
    if (x.size == 0) {
      final outSize = minlength ?? 0;
      final result =
          out ??
          NDArray<T>.zeros([
            outSize,
          ], (weights?.dtype ?? DType.int64) as DType<T>);
      if (out != null) {
        result.fill(normalizeScalar(0, result.dtype) as T);
      }
      return result.detachToParentScope();
    }

    // Validate non-negative
    final minVal = min(x).scalar;
    if (minVal < 0) {
      throw ArgumentError('Input array x must be non-negative.');
    }

    final maxVal = max(x).scalar;
    final outSize = math.max(maxVal + 1, minlength ?? 0);

    if (weights != null) {
      if (weights.isDisposed) {
        throw StateError('Weights array is disposed.');
      }
      if (!_listEquals(weights.shape, x.shape)) {
        throw ArgumentError('Weights must have the same shape as x.');
      }
    }

    // Determine target DType for the result
    final DType<T> targetDType =
        (out?.dtype ?? weights?.dtype ?? DType.int64) as DType<T>;

    if (out != null) {
      if (out.shape.length != 1 || out.shape[0] < outSize) {
        throw ArgumentError(
          'Output array must be 1D and have size at least $outSize.',
        );
      }
    }

    final size = x.size;
    final resSize = outSize;

    // Cast x to int32 or int64 if it is int16 or uint8
    NDArray<int> xCast = x;
    if (x.dtype != DType.int32 && x.dtype != DType.int64) {
      xCast = castNDArray<Int64>(x, DType.int64);
    }

    if (weights == null) {
      // Unweighted bincount. C++ kernels write to int64.
      final bool useTempResult = targetDType != DType.int64;
      final NDArray<Int64> res64 = useTempResult
          ? NDArray<Int64>.zeros([outSize], DType.int64)
          : (out as NDArray<Int64>? ??
                NDArray<Int64>.zeros([outSize], DType.int64));

      if (out != null && !useTempResult) {
        res64.fill(Int64(0));
      }

      if (xCast.dtype == DType.int64) {
        if (xCast.isContiguous && res64.isContiguous) {
          v_bincount_int64(
            xCast.pointer.cast(),
            res64.pointer.cast(),
            size,
            resSize,
          );
        } else {
          s_bincount_int64(
            xCast.pointer.cast(),
            xCast.strides[0],
            res64.pointer.cast(),
            res64.strides[0],
            size,
            resSize,
          );
        }
      } else {
        if (xCast.isContiguous && res64.isContiguous) {
          v_bincount_int32(
            xCast.pointer.cast(),
            res64.pointer.cast(),
            size,
            resSize,
          );
        } else {
          s_bincount_int32(
            xCast.pointer.cast(),
            xCast.strides[0],
            res64.pointer.cast(),
            res64.strides[0],
            size,
            resSize,
          );
        }
      }

      if (useTempResult) {
        final result = out ?? NDArray<T>.zeros([outSize], targetDType);
        _fastCopyAndCast(res64, result);
        return result.detachToParentScope();
      } else {
        return res64.detachToParentScope() as NDArray<T>;
      }
    } else {
      // Weighted bincount. Target DType must be float32 or float64.
      final DType<num> wDType = targetDType.isFloating
          ? targetDType
          : DType.float64;
      NDArray<num> wCast = weights;
      if (weights.dtype != wDType) {
        wCast = castNDArray(weights, wDType);
      }

      final bool useTempResult = out == null || out.dtype != wDType;
      final NDArray<num> resFloat = useTempResult
          ? NDArray<num>.zeros([outSize], wDType)
          : out;

      if (out != null && !useTempResult) {
        resFloat.fill(normalizeScalar(0, resFloat.dtype) as num);
      }

      if (xCast.dtype == DType.int64) {
        if (wCast.dtype == DType.float64) {
          if (xCast.isContiguous &&
              wCast.isContiguous &&
              resFloat.isContiguous) {
            v_bincount_weights_int64_double(
              xCast.pointer.cast(),
              wCast.pointer.cast(),
              resFloat.pointer.cast(),
              size,
              resSize,
            );
          } else {
            s_bincount_weights_int64_double(
              xCast.pointer.cast(),
              xCast.strides[0],
              wCast.pointer.cast(),
              wCast.strides[0],
              resFloat.pointer.cast(),
              resFloat.strides[0],
              size,
              resSize,
            );
          }
        } else {
          // float32
          if (xCast.isContiguous &&
              wCast.isContiguous &&
              resFloat.isContiguous) {
            v_bincount_weights_int64_float(
              xCast.pointer.cast(),
              wCast.pointer.cast(),
              resFloat.pointer.cast(),
              size,
              resSize,
            );
          } else {
            s_bincount_weights_int64_float(
              xCast.pointer.cast(),
              xCast.strides[0],
              wCast.pointer.cast(),
              wCast.strides[0],
              resFloat.pointer.cast(),
              resFloat.strides[0],
              size,
              resSize,
            );
          }
        }
      } else {
        // int32
        if (wCast.dtype == DType.float64) {
          if (xCast.isContiguous &&
              wCast.isContiguous &&
              resFloat.isContiguous) {
            v_bincount_weights_int32_double(
              xCast.pointer.cast(),
              wCast.pointer.cast(),
              resFloat.pointer.cast(),
              size,
              resSize,
            );
          } else {
            s_bincount_weights_int32_double(
              xCast.pointer.cast(),
              xCast.strides[0],
              wCast.pointer.cast(),
              wCast.strides[0],
              resFloat.pointer.cast(),
              resFloat.strides[0],
              size,
              resSize,
            );
          }
        } else {
          // float32
          if (xCast.isContiguous &&
              wCast.isContiguous &&
              resFloat.isContiguous) {
            v_bincount_weights_int32_float(
              xCast.pointer.cast(),
              wCast.pointer.cast(),
              resFloat.pointer.cast(),
              size,
              resSize,
            );
          } else {
            s_bincount_weights_int32_float(
              xCast.pointer.cast(),
              xCast.strides[0],
              wCast.pointer.cast(),
              wCast.strides[0],
              resFloat.pointer.cast(),
              resFloat.strides[0],
              size,
              resSize,
            );
          }
        }
      }

      if (useTempResult) {
        final result = out ?? NDArray<T>.zeros([outSize], targetDType);
        _fastCopyAndCast(resFloat, result);
        return result.detachToParentScope();
      } else {
        return resFloat.detachToParentScope() as NDArray<T>;
      }
    }
  });
}

/// Return the indices of the bins to which each value in input array belongs.
NDArray<int> digitize(
  NDArray<num> x,
  NDArray<num> bins, {
  bool right = false,
  NDArray<int>? out,
}) {
  if (x.isDisposed || bins.isDisposed) {
    throw StateError('Cannot execute digitize() on disposed array(s).');
  }
  if (bins.shape.length != 1) {
    throw ArgumentError('bins must be a 1-D array.');
  }
  if (bins.size == 0) {
    throw ArgumentError('bins must not be empty.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write digitize result to a disposed output array.',
    );
  }
  if (x.dtype.isComplex || bins.dtype.isComplex) {
    throw ArgumentError('Complex arrays are not supported in digitize.');
  }

  return NDArray.scope(() {
    // Check monotonicity
    final binsList = bins.toList();
    bool increasing = true;
    bool decreasing = true;
    for (var i = 1; i < binsList.length; i++) {
      final d = binsList[i].toDouble() - binsList[i - 1].toDouble();
      if (d < 0) increasing = false;
      if (d > 0) decreasing = false;
    }
    if (!increasing && !decreasing) {
      throw ArgumentError('bins must be monotonic.');
    }

    final side = right ? SearchSide.left : SearchSide.right;
    NDArray<int> res;

    if (increasing) {
      res = searchsorted(bins, x, side: side);
    } else {
      final flippedBins = flip(bins);
      final j = searchsorted(flippedBins, x, side: side);
      final nArr = NDArray<int>.scalar(bins.size, dtype: DType.int32);
      res = subtract<int, int, int>(nArr, j);
    }

    if (out != null) {
      if (!listEquals(out.shape, res.shape) || out.dtype != res.dtype) {
        throw ArgumentError('Incompatible out buffer shape or dtype.');
      }
      _fastCopyAndCast(res, out);
      return out;
    }

    return res.detachToParentScope();
  });
}

/// Computes the histogram of a set of data.
(NDArray<num> hist, NDArray<Float64> binEdges) histogram(
  NDArray<num> x, {
  dynamic bins = 10,
  (double, double)? range,
  bool density = false,
  NDArray<num>? weights,
}) {
  if (x.isDisposed) {
    throw StateError('Cannot compute histogram of a disposed array.');
  }
  if (weights != null && weights.isDisposed) {
    throw StateError('Weights array is disposed.');
  }

  return NDArray.scope(() {
    final flatX = x.rank == 1 ? x : x.ravel();
    if (weights != null && !listEquals(weights.shape, x.shape)) {
      throw ArgumentError('Weights must have the same shape as x.');
    }
    final flatWeights = weights?.rank == 1 ? weights : weights?.ravel();

    NDArray<Float64> resolvedBinEdges;

    if (bins is int) {
      if (bins <= 0) {
        throw ArgumentError('bins must be positive.');
      }
      double minX;
      double maxX;
      if (range != null) {
        minX = range.$1;
        maxX = range.$2;
      } else {
        if (flatX.size == 0) {
          minX = 0.0;
          maxX = 1.0;
        } else {
          final xDouble = promoteToDouble(flatX);
          minX = min(xDouble).scalar;
          maxX = max(xDouble).scalar;
        }
      }

      var start = minX;
      var stop = maxX;
      if (start == stop) {
        start -= 0.5;
        stop += 0.5;
      }

      resolvedBinEdges = linspace<Float64>(
        Float64(start),
        Float64(stop),
        bins + 1,
        dtype: DType.float64,
      );
    } else if (bins is NDArray) {
      if (bins.shape.length != 1) {
        throw ArgumentError('bins must be a 1-D array.');
      }
      // Validate monotonicity (must be increasing for histogram bin edges)
      final binsList = bins.toList();
      for (var i = 1; i < binsList.length; i++) {
        if ((binsList[i] as num) < (binsList[i - 1] as num)) {
          throw ArgumentError('bins array must be monotonically increasing.');
        }
      }

      resolvedBinEdges = bins.dtype == DType.float64
          ? bins as NDArray<Float64>
          : castNDArray<Float64>(bins, DType.float64);
    } else {
      throw ArgumentError('bins must be an int or an NDArray.');
    }

    final M = resolvedBinEdges.size;
    if (M < 2) {
      throw ArgumentError('bins must have at least 2 edges (1 bin).');
    }

    // Vectorized boundary handling and bincount
    final binIndices = digitize(flatX, resolvedBinEdges, right: false);
    final counts = bincount(binIndices, weights: flatWeights, minlength: M + 1);

    final lastEdgeVal = resolvedBinEdges.getCell([M - 1]);
    final lastEdgeArr = NDArray<Float64>.scalar(
      lastEdgeVal,
      dtype: DType.float64,
    );
    final equalLastEdge = equal(flatX, lastEdgeArr);

    num equalLastEdgeWeightSum = 0;
    if (flatWeights == null) {
      equalLastEdgeWeightSum = count_nonzero(equalLastEdge).scalar;
    } else {
      final zeroScalar = NDArray<num>.scalar(0.0, dtype: flatWeights.dtype);
      final lastEdgeWeights =
          where(equalLastEdge, flatWeights, zeroScalar) as NDArray<num>;
      equalLastEdgeWeightSum = sum<num>(lastEdgeWeights).scalar;
    }

    final currentLastBinVal = counts.getCell([M - 1]);
    counts.setCell([M - 1], currentLastBinVal + equalLastEdgeWeightSum);

    final histView = counts.slice([Slice(start: 1, stop: M)]);
    final hist = histView.copy();

    NDArray<num> finalHist = hist;
    if (density) {
      final totalSum = sum<num>(hist).scalar;
      final widths = subtract<Float64, Float64, Float64>(
        resolvedBinEdges.slice([Slice(start: 1)]),
        resolvedBinEdges.slice([Slice(stop: M - 1)]),
      );
      final totalSumArr = NDArray<Float64>.scalar(
        Float64(totalSum.toDouble()),
        dtype: DType.float64,
      );
      final divisor = multiply<Float64, Float64, Float64>(widths, totalSumArr);
      finalHist = divide<num, Float64, Float64>(hist, divisor);
    }

    return (
      finalHist.detachToParentScope(),
      resolvedBinEdges.detachToParentScope(),
    );
  });
}
