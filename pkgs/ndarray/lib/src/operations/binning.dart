// ignore_for_file: non_constant_identifier_names
import 'dart:math' as math;
import 'dart:ffi' as ffi;
import '../ndarray.dart';
import '../ndarray_bindings.dart';
import 'helpers.dart';
import 'stats.dart'; // For min, max, sum
import 'math.dart'; // For diff, multiply, divide, equal
import 'manipulation.dart'; // For flip, ravel, where
import 'sorting.dart'; // For searchsorted, count_nonzero
import 'spacers.dart'; // For linspace

// Helper to check list equality
bool _listEquals<T extends AnyDType>(List<T>? a, List<T>? b) {
  if (a == null) return b == null;
  if (b == null) return false;
  if (identical(a, b)) return true;
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

// Fast copy and cast between NDArrays
void _fastCopyAndCast(NDArray src, NDArray dest) {
  assert(src.size == dest.size);
  if (src.dtype == dest.dtype) {
    src.copy(out: dest);
    return;
  }
  NDArray.scope(() {
    final casted = castNDArray(src, dest.dtype);
    casted.copy(out: dest);
  });
}

/// Computes the frequency of each value in an array of non-negative ints.
///
/// **Preconditions:**
/// - [x] must not be disposed.
/// - [x] must be a 1-D array containing non-negative integers.
/// - If [weights] is provided, it must not be disposed and must have the same shape as [x].
/// - [minlength] must be non-negative ($\ge 0$).
///
/// **Throws:**
/// - It is an error if [x] or [weights] is disposed.
/// - It is an error if [x] is not 1-D or contains negative values.
/// - It is an error if [weights] shape does not match [x] shape.
/// - It is an error if [minlength] is negative.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// **Example:**
/// ```dart
/// final a = NDArray<AnyInt>.fromList([0, 1, 1, 3, 2, 1, 7], [7], DType.int32);
/// final counts = bincount(a);
/// ```
///
/// Refer to the [NumPy bincount reference](https://numpy.org/doc/stable/reference/generated/numpy.bincount.html)
/// for details.
NDArray<T> bincount<T extends AnyReal>(
  NDArray<AnyInt> x, {
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
      if (out != null) {
        if (out.shape.length != 1 || out.shape[0] < outSize) {
          throw ArgumentError(
            'Output array must be 1D and have size at least $outSize.',
          );
        }
        out.fill(normalizeScalar(0, out.dtype) as T);
        return out;
      }
      final result = NDArray<T>.zeros([
        outSize,
      ], (weights?.dtype ?? DType.int64) as DType<T>);
      return result.detachToParentScope();
    }

    // Validate non-negative
    final minVal = min(x).scalar;
    if (minVal < 0) {
      throw ArgumentError('Input array x must be non-negative.');
    }

    final maxVal = max(x).scalar;
    final minRequiredSize = math.max(maxVal + 1, minlength ?? 0);

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
      if (out.shape.length != 1 || out.shape[0] < minRequiredSize) {
        throw ArgumentError(
          'Output array must be 1D and have size at least $minRequiredSize.',
        );
      }
    }

    final aliasesInput =
        out != null &&
        (!out.isContiguous ||
            sharesMemory(x, out) ||
            (weights != null && sharesMemory(weights, out)));
    if (aliasesInput) {
      final temp = bincount<num>(x, weights: weights, minlength: out.shape[0]);
      _fastCopyAndCast(temp, out);
      return out;
    }

    final outSize = out != null ? out.shape[0] : minRequiredSize;

    final size = x.size;
    final resSize = outSize;

    // Cast x to int32 or int64 if it is int16 or uint8
    NDArray<AnyInt> xCast;
    switch (x.dtype) {
      case DType.int32:
      case DType.int64:
        xCast = x;
      default:
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
        res64.fill(0);
      }

      switch (xCast.dtype) {
        case DType.int64:
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
        default:
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
        return out ?? result.detachToParentScope();
      } else {
        return out ?? (res64.detachToParentScope() as NDArray<T>);
      }
    } else {
      // Weighted bincount. Target DType must be float32 or float64.
      final DType<num> wDType = targetDType.isFloating
          ? targetDType
          : DType.float64;
      NDArray<AnyReal> wCast = weights;
      if (weights.dtype != wDType) {
        wCast = castNDArray(weights, wDType);
      }

      final bool useTempResult = out == null || out.dtype != wDType;
      final NDArray<AnyReal> resFloat = useTempResult
          ? NDArray<AnyReal>.zeros([outSize], wDType)
          : out;

      if (out != null && !useTempResult) {
        resFloat.fill(normalizeScalar(0, resFloat.dtype) as num);
      }

      switch ((xCast.dtype, wCast.dtype)) {
        case (DType.int64, DType.float64):
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
        case (DType.int64, _):
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
        case (_, DType.float64):
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
        case _:
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

      if (useTempResult) {
        final result = out ?? NDArray<T>.zeros([outSize], targetDType);
        _fastCopyAndCast(resFloat, result);
        return out ?? result.detachToParentScope();
      } else {
        return out;
      }
    }
  });
}

/// Return the indices of the bins to which each value in input array belongs.
///
/// **Preconditions:**
/// - [x] and [bins] must not be disposed.
/// - [bins] must be a 1-D monotonic array.
/// - [bins] must not be empty.
/// - [x] and [bins] must not have complex data types.
///
/// **Throws:**
/// - It is an error if [x] or [bins] is disposed.
/// - It is an error if [bins] is not 1-D or is empty.
/// - It is an error if [bins] is not monotonic.
/// - It is an error if [x] or [bins] contains complex numbers.
/// - It is an error if [out] has incompatible shape or dtype.
///
/// **Example:**
/// ```dart
/// final x = NDArray<Float64>.fromList([0.2, 6.4, 3.0, 1.6], [4], DType.float64);
/// final bins = NDArray<Float64>.fromList([0.0, 1.0, 2.5, 4.0, 10.0], [5], DType.float64);
/// final inds = digitize(x, bins);
/// ```
///
/// Refer to the [NumPy digitize reference](https://numpy.org/doc/stable/reference/generated/numpy.digitize.html)
/// for details.
NDArray<Int32> digitize(
  NDArray<AnyReal> x,
  NDArray<AnyReal> bins, {
  bool right = false,
  NDArray<AnyInt>? out,
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
    bool increasing = true;
    bool decreasing = true;
    final len = bins.size;
    if (bins.dtype == DType.uint64) {
      var prev = bins.getCell([0]) as int;
      for (var i = 1; i < len; i++) {
        final curr = bins.getCell([i]) as int;
        final cmp = uint64Compare(curr, prev);
        if (cmp < 0) increasing = false;
        if (cmp > 0) decreasing = false;
        prev = curr;
      }
    } else if (bins.dtype.isInteger) {
      var prev = bins.getCell([0]).toInt();
      for (var i = 1; i < len; i++) {
        final curr = bins.getCell([i]).toInt();
        if (curr < prev) increasing = false;
        if (curr > prev) decreasing = false;
        prev = curr;
      }
    } else {
      double toDoubleVal(dynamic v) =>
          v is num ? v.toDouble() : (v as dynamic).value as double;
      var prev = toDoubleVal(bins.getCell([0]));
      if (prev.isNaN) {
        throw ArgumentError('bins must be monotonic and must not contain NaN.');
      }
      for (var i = 1; i < len; i++) {
        final curr = toDoubleVal(bins.getCell([i]));
        if (curr.isNaN) {
          throw ArgumentError(
            'bins must be monotonic and must not contain NaN.',
          );
        }
        if (curr < prev) increasing = false;
        if (curr > prev) decreasing = false;
        prev = curr;
      }
    }
    if (!increasing && !decreasing) {
      throw ArgumentError('bins must be monotonic.');
    }

    final commonDType = resolveDType(bins.dtype, x.dtype) as DType<Object>;
    final commonBins = bins.dtype == commonDType
        ? bins as NDArray<AnyDType>
        : castNDArray<AnyDType>(bins, commonDType);
    final commonX = x.dtype == commonDType
        ? x as NDArray<AnyDType>
        : castNDArray<AnyDType>(x, commonDType);

    final side = right ? SearchSide.left : SearchSide.right;
    NDArray<Int32> res;

    if (increasing) {
      res = searchsorted(commonBins, commonX, side: side);
    } else {
      final flippedBins = flip(commonBins);
      final j = searchsorted(flippedBins, commonX, side: side);
      final nArr = NDArray<Int32>.scalar(bins.size, dtype: DType.int32);
      res = subtract<Int32, Int32, Int32>(nArr, j);
    }

    if (out != null) {
      if (!listEquals(out.shape, res.shape) || !out.dtype.isInteger) {
        throw ArgumentError('Incompatible out buffer shape or dtype.');
      }
      _fastCopyAndCast(res, out);
      return out as NDArray<Int32>;
    }

    return res.detachToParentScope();
  });
}

/// Computes the histogram of a set of data.
///
/// **Preconditions:**
/// - [x] must not be disposed.
/// - If [weights] is provided, it must not be disposed and must match the shape of [x].
/// - If [bins] is an integer, it must be strictly positive ($\ge 1$).
/// - If [bins] is an array, it must be 1-D and monotonically increasing with at least 2 edges.
/// - [x] and [weights] must not contain complex numbers.
///
/// **Throws:**
/// - It is an error if [x] or [weights] is disposed.
/// - It is an error if [weights] shape does not match [x] shape.
/// - It is an error if [bins] is non-positive or not a 1-D monotonically increasing array.
/// - It is an error if [bins] has fewer than 2 edges.
/// - It is an error if [x] or [weights] contains complex numbers.
///
/// **Performance considerations:**
/// - For uniform bins (integer [bins]), uses a fused native C kernel with $O(N)$ single-pass binning.
/// - For non-uniform bins, uses a native C binary search kernel with $O(N \log M)$ time complexity.
///
/// **Example:**
/// ```dart
/// final a = NDArray<Float64>.fromList([1, 2, 1], [3], DType.float64);
/// final (:hist, :binEdges) = histogram(a, bins: 2, range: (0.0, 2.0));
/// ```
///
/// Refer to the [NumPy histogram reference](https://numpy.org/doc/stable/reference/generated/numpy.histogram.html)
/// for details.
({NDArray<AnyReal> hist, NDArray<Float64> binEdges}) histogram(
  NDArray<AnyReal> x, {
  dynamic bins = 10,
  (double, double)? range,
  bool density = false,
  NDArray<AnyReal>? weights,
}) {
  if (x.isDisposed) {
    throw StateError('Cannot compute histogram of a disposed array.');
  }
  if (weights != null && weights.isDisposed) {
    throw StateError('Weights array is disposed.');
  }
  if (x.dtype.isComplex || (weights != null && weights.dtype.isComplex)) {
    throw ArgumentError('Complex arrays are not supported in histogram.');
  }

  return NDArray.scope(() {
    final rawFlatX = (x.rank == 1 && x.isContiguous)
        ? x
        : (x.rank == 1 ? x : x.ravel());
    if (weights != null && !listEquals(weights.shape, x.shape)) {
      throw ArgumentError('Weights must have the same shape as x.');
    }
    final rawFlatWeights = weights == null
        ? null
        : ((weights.rank == 1 && weights.isContiguous)
              ? weights
              : (weights.rank == 1 ? weights : weights.ravel()));

    final NDArray<AnyReal> flatX =
        (rawFlatX.dtype == DType.float16 || rawFlatX.dtype == DType.bfloat16)
        ? castNDArray<Float64>(rawFlatX, DType.float64)
        : rawFlatX;
    final NDArray<AnyReal>? flatWeights = rawFlatWeights == null
        ? null
        : ((rawFlatWeights.dtype == DType.float16 ||
                  rawFlatWeights.dtype == DType.bfloat16)
              ? castNDArray<Float64>(rawFlatWeights, DType.float64)
              : rawFlatWeights);

    NDArray<Float64> resolvedBinEdges;
    final bool isUniform = bins is int;
    final int nbins;
    double minX = 0.0;
    double maxX = 1.0;
    double norm = 1.0;

    if (isUniform) {
      nbins = bins;
      if (nbins <= 0) {
        throw ArgumentError('bins must be positive.');
      }
      if (range != null) {
        minX = range.$1;
        maxX = range.$2;
        if (!minX.isFinite || !maxX.isFinite || minX > maxX) {
          throw ArgumentError('range must be finite and min <= max.');
        }
        if (minX == maxX) {
          minX -= 0.5;
          maxX += 0.5;
        }
      } else {
        if (flatX.size == 0) {
          minX = 0.0;
          maxX = 1.0;
        } else {
          final minRes = min<num>(flatX).scalar;
          final maxRes = max<num>(flatX).scalar;
          if (flatX.dtype == DType.uint64) {
            minX = BigInt.from(minRes as int).toUnsigned(64).toDouble();
            maxX = BigInt.from(maxRes as int).toUnsigned(64).toDouble();
          } else {
            minX = minRes.toDouble();
            maxX = maxRes.toDouble();
          }
          if (!minX.isFinite || !maxX.isFinite || minX > maxX) {
            throw ArgumentError('range must be finite and min <= max.');
          }
          if (minX == maxX) {
            minX -= 0.5;
            maxX += 0.5;
          }
        }
      }
      resolvedBinEdges = linspace<Float64>(
        minX,
        maxX,
        nbins + 1,
        dtype: DType.float64,
      );
      norm = nbins / (maxX - minX);
    } else if (bins is NDArray) {
      if (bins.isDisposed) {
        throw StateError('bins array is disposed.');
      }
      if (bins.shape.length != 1) {
        throw ArgumentError('bins must be a 1-D array.');
      }
      resolvedBinEdges = bins.dtype == DType.float64
          ? (bins as NDArray<Float64>).copy()
          : castNDArray<Float64>(bins, DType.float64);
      final M = resolvedBinEdges.size;
      if (M < 2) {
        throw ArgumentError('bins must have at least 2 edges (1 bin).');
      }
      // Check monotonicity
      final cEdges = resolvedBinEdges.pointer.cast<ffi.Double>();
      final strideEdges = resolvedBinEdges.strides[0];
      if (M > 0 && cEdges[0].isNaN) {
        throw ArgumentError('bins must increase monotonically.');
      }
      for (var i = 1; i < M; i++) {
        if (!(cEdges[i * strideEdges] > cEdges[(i - 1) * strideEdges])) {
          throw ArgumentError('bins must increase monotonically.');
        }
      }
      nbins = M - 1;
    } else {
      throw ArgumentError('bins must be an int or an NDArray.');
    }

    final DType<num> targetHistDType = switch (rawFlatWeights?.dtype) {
      null => DType.int64 as DType<num>,
      DType.float64 ||
      DType.float32 ||
      DType.float16 ||
      DType.bfloat16 => rawFlatWeights!.dtype,
      _ => DType.float64 as DType<num>,
    };
    final DType<num> computeHistDType = switch (rawFlatWeights?.dtype) {
      null => DType.int64 as DType<num>,
      DType.float32 => DType.float32 as DType<num>,
      _ => DType.float64 as DType<num>,
    };

    final NDArray<AnyReal> hist = NDArray<AnyReal>.zeros([nbins], computeHistDType);

    final pSrc = flatX.pointer.cast<ffi.Void>();
    final pWeights = flatWeights != null
        ? flatWeights.pointer.cast<ffi.Void>()
        : ffi.Pointer<ffi.Void>.fromAddress(0);
    final pHist = hist.pointer.cast<ffi.Void>();
    final dtypeSrc = encodeDType(flatX.dtype);
    final dtypeWeights = flatWeights != null
        ? encodeDType(flatWeights.dtype)
        : -1;
    final dtypeHist = encodeDType(hist.dtype);

    if (isUniform) {
      if (flatX.isContiguous &&
          (flatWeights == null || flatWeights.isContiguous)) {
        v_histogram_uniform(
          pSrc,
          dtypeSrc,
          pWeights,
          dtypeWeights,
          pHist,
          dtypeHist,
          flatX.size,
          nbins,
          minX,
          maxX,
          norm,
        );
      } else {
        s_histogram_uniform(
          pSrc,
          flatX.strides[0],
          dtypeSrc,
          pWeights,
          flatWeights != null ? flatWeights.strides[0] : 0,
          dtypeWeights,
          pHist,
          hist.strides[0],
          dtypeHist,
          flatX.size,
          nbins,
          minX,
          maxX,
          norm,
        );
      }
    } else {
      final NDArray<Float64> contigEdges = resolvedBinEdges.isContiguous
          ? resolvedBinEdges
          : resolvedBinEdges.copy();
      final pEdges = contigEdges.pointer.cast<ffi.Double>();

      if (flatX.isContiguous &&
          (flatWeights == null || flatWeights.isContiguous)) {
        v_histogram_binsearch(
          pSrc,
          dtypeSrc,
          pWeights,
          dtypeWeights,
          pHist,
          dtypeHist,
          pEdges,
          resolvedBinEdges.size,
          flatX.size,
        );
      } else {
        s_histogram_binsearch(
          pSrc,
          flatX.strides[0],
          dtypeSrc,
          pWeights,
          flatWeights != null ? flatWeights.strides[0] : 0,
          dtypeWeights,
          pHist,
          hist.strides[0],
          dtypeHist,
          pEdges,
          resolvedBinEdges.size,
          flatX.size,
        );
      }
    }

    NDArray<AnyReal> finalHist = hist;
    if (density) {
      final totalSum = sum<num>(hist).scalar;
      final widths = subtract<Float64, Float64, Float64>(
        resolvedBinEdges.slice([Slice(start: 1)]),
        resolvedBinEdges.slice([Slice(stop: resolvedBinEdges.size - 1)]),
      );
      final totalSumArr = NDArray<Float64>.scalar(
        totalSum.toDouble(),
        dtype: DType.float64,
      );
      final divisor = multiply<Float64, Float64, Float64>(widths, totalSumArr);
      finalHist = divide<num, Float64, Float64>(hist, divisor);
    } else if (targetHistDType != computeHistDType) {
      finalHist = castNDArray<AnyReal>(hist, targetHistDType);
    }

    return (
      hist: finalHist.detachToParentScope(),
      binEdges: resolvedBinEdges.detachToParentScope(),
    );
  });
}
