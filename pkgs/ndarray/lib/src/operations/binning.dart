// ignore_for_file: non_constant_identifier_names
import 'dart:math' as math;
import '../ndarray.dart';
import '../ndarray_bindings.dart';
import 'math.dart';
import 'sorting.dart';
import 'stats.dart';
import 'spacers.dart';

enum Monotonicity { increasing, decreasing, none }

Monotonicity _checkMonotonicity(NDArray bins) {
  if (bins.size <= 1) return Monotonicity.increasing;
  final list = bins.toList();

  bool increasing = true;
  bool decreasing = true;

  for (var i = 1; i < list.length; i++) {
    final prev = list[i - 1] as num;
    final curr = list[i] as num;
    if (curr > prev) {
      decreasing = false;
    } else if (curr < prev) {
      increasing = false;
    }
  }

  if (increasing) return Monotonicity.increasing;
  if (decreasing) return Monotonicity.decreasing;
  return Monotonicity.none;
}

NDArray _reverse1D(NDArray a) {
  return NDArray.fromList(a.toList().reversed.toList(), a.shape, a.dtype);
}

/// Count number of occurrences of each value in array of non-negative ints.
NDArray<num> bincount(
  NDArray<int> x, {
  NDArray<num>? weights,
  int minlength = 0,
  NDArray<num>? out,
}) {
  if (x.isDisposed) {
    throw StateError('Cannot execute bincount on a disposed array.');
  }
  if (x.shape.length != 1) {
    throw ArgumentError('Input array must be 1-dimensional.');
  }
  if (minlength < 0) {
    throw ArgumentError('minlength must be non-negative.');
  }
  if (weights != null && weights.isDisposed) {
    throw StateError('Cannot use a disposed weights array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write bincount to a disposed output array.');
  }

  if (x.dtype != DType.int32 && x.dtype != DType.int64) {
    throw ArgumentError(
      'Input array must be of integer type (int32 or int64).',
    );
  }

  if (weights != null &&
      weights.dtype != DType.float64 &&
      weights.dtype != DType.float32) {
    throw ArgumentError('Weights must be of float32 or float64 type.');
  }

  return NDArray.scope(() {
    final minVal = min(x).scalar;
    if (minVal < 0) {
      throw ArgumentError('Input array must contain only non-negative values.');
    }

    final maxVal = max(x).scalar;
    final binCount = math.max(maxVal + 1, minlength);

    final DType<num> outDType = weights?.dtype ?? DType.int64;

    if (out != null) {
      if (!listEquals(out.shape, [binCount]) || out.dtype != outDType) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
      out.fill(0);
    }

    final result =
        out ?? NDArray<num>.create([binCount], outDType, zeroInit: true);

    final isContiguous =
        x.isContiguous &&
        (weights == null || weights.isContiguous) &&
        result.isContiguous;

    final size = x.size;
    final resSize = binCount;

    if (isContiguous) {
      if (weights == null) {
        if (x.dtype == DType.int32) {
          v_bincount_int32(
            x.pointer.cast(),
            result.pointer.cast(),
            size,
            resSize,
          );
        } else if (x.dtype == DType.int64) {
          v_bincount_int64(
            x.pointer.cast(),
            result.pointer.cast(),
            size,
            resSize,
          );
        }
      } else {
        if (x.dtype == DType.int32) {
          if (weights.dtype == DType.float64) {
            v_bincount_weights_int32_double(
              x.pointer.cast(),
              weights.pointer.cast(),
              result.pointer.cast(),
              size,
              resSize,
            );
          } else if (weights.dtype == DType.float32) {
            v_bincount_weights_int32_float(
              x.pointer.cast(),
              weights.pointer.cast(),
              result.pointer.cast(),
              size,
              resSize,
            );
          }
        } else if (x.dtype == DType.int64) {
          if (weights.dtype == DType.float64) {
            v_bincount_weights_int64_double(
              x.pointer.cast(),
              weights.pointer.cast(),
              result.pointer.cast(),
              size,
              resSize,
            );
          } else if (weights.dtype == DType.float32) {
            v_bincount_weights_int64_float(
              x.pointer.cast(),
              weights.pointer.cast(),
              result.pointer.cast(),
              size,
              resSize,
            );
          }
        }
      }
    } else {
      final strideX = x.strides[0];
      final strideRes = result.strides[0];
      if (weights == null) {
        if (x.dtype == DType.int32) {
          s_bincount_int32(
            x.pointer.cast(),
            strideX,
            result.pointer.cast(),
            strideRes,
            size,
            resSize,
          );
        } else if (x.dtype == DType.int64) {
          s_bincount_int64(
            x.pointer.cast(),
            strideX,
            result.pointer.cast(),
            strideRes,
            size,
            resSize,
          );
        }
      } else {
        final strideW = weights.strides[0];
        if (x.dtype == DType.int32) {
          if (weights.dtype == DType.float64) {
            s_bincount_weights_int32_double(
              x.pointer.cast(),
              strideX,
              weights.pointer.cast(),
              strideW,
              result.pointer.cast(),
              strideRes,
              size,
              resSize,
            );
          } else if (weights.dtype == DType.float32) {
            s_bincount_weights_int32_float(
              x.pointer.cast(),
              strideX,
              weights.pointer.cast(),
              strideW,
              result.pointer.cast(),
              strideRes,
              size,
              resSize,
            );
          }
        } else if (x.dtype == DType.int64) {
          if (weights.dtype == DType.float64) {
            s_bincount_weights_int64_double(
              x.pointer.cast(),
              strideX,
              weights.pointer.cast(),
              strideW,
              result.pointer.cast(),
              strideRes,
              size,
              resSize,
            );
          } else if (weights.dtype == DType.float32) {
            s_bincount_weights_int64_float(
              x.pointer.cast(),
              strideX,
              weights.pointer.cast(),
              strideW,
              result.pointer.cast(),
              strideRes,
              size,
              resSize,
            );
          }
        }
      }
    }

    if (out == null) {
      result.detachToParentScope();
    }
    return result;
  });
}

/// Return the indices of the bins to which each value in input array belongs.
NDArray<int> digitize(
  NDArray x,
  NDArray bins, {
  bool right = false,
  NDArray<int>? out,
}) {
  if (x.isDisposed || bins.isDisposed) {
    throw StateError('Cannot execute digitize on disposed arrays.');
  }
  if (bins.shape.length != 1) {
    throw ArgumentError('bins must be 1-dimensional.');
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
    final monotonicity = _checkMonotonicity(bins);
    if (monotonicity == Monotonicity.none) {
      throw ArgumentError('bins must be monotonic.');
    }

    final isIncreasing = monotonicity == Monotonicity.increasing;

    if (isIncreasing) {
      final side = right ? SearchSide.left : SearchSide.right;
      final res = searchsorted(bins, x, side: side);
      if (out != null) {
        if (!out.isContiguous) {
          throw ArgumentError('out buffer must be contiguous.');
        }
        if (!listEquals(out.shape, res.shape) || out.dtype != res.dtype) {
          throw ArgumentError('Incompatible out buffer shape or dtype.');
        }
        custom_memcpy(out.pointer, res.pointer, res.size * 4);
        return out;
      }
      res.detachToParentScope();
      return res;
    } else {
      final binsRev = _reverse1D(bins);
      final side = right ? SearchSide.right : SearchSide.left;
      final resRev = searchsorted(binsRev, x, side: side);
      final nArr = NDArray<int>.fromList([bins.size], [], DType.int32);
      final res = subtract<int, int, int>(nArr, resRev);
      if (out != null) {
        if (!out.isContiguous) {
          throw ArgumentError('out buffer must be contiguous.');
        }
        if (!listEquals(out.shape, res.shape) || out.dtype != res.dtype) {
          throw ArgumentError('Incompatible out buffer shape or dtype.');
        }
        custom_memcpy(out.pointer, res.pointer, res.size * 4);
        return out;
      }
      res.detachToParentScope();
      return res;
    }
  });
}

/// Compute the histogram of a set of data.
({NDArray<num> hist, NDArray<Float64> binEdges}) histogram(
  NDArray a, {
  dynamic bins = 10,
  (double, double)? range,
  bool density = false,
  NDArray<num>? weights,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute histogram on a disposed array.');
  }
  if (weights != null && weights.isDisposed) {
    throw StateError('Cannot use a disposed weights array.');
  }

  return NDArray.scope(() {
    final x = a.ravel() as NDArray<num>;

    NDArray<Float64> binEdges;

    if (bins is int) {
      if (bins <= 0) {
        throw ArgumentError('Number of bins must be positive.');
      }
      final double minVal = (range?.$1 ?? min(x).scalar).toDouble();
      final double maxVal = (range?.$2 ?? max(x).scalar).toDouble();
      if (minVal == maxVal) {
        binEdges = linspace<Float64>(
          (minVal - 0.5) as Float64,
          (maxVal + 0.5) as Float64,
          bins + 1,
          dtype: DType.float64,
        );
      } else {
        binEdges = linspace<Float64>(
          minVal as Float64,
          maxVal as Float64,
          bins + 1,
          dtype: DType.float64,
        );
      }
    } else if (bins is NDArray) {
      if (bins.shape.length != 1) {
        throw ArgumentError('bins array must be 1-dimensional.');
      }
      final monotonicity = _checkMonotonicity(bins);
      if (monotonicity != Monotonicity.increasing) {
        throw ArgumentError('bins array must be monotonically increasing.');
      }
      if (bins.dtype != DType.float64) {
        binEdges = NDArray<Float64>.fromList(
          bins.toList().map((e) => (e as num).toDouble()).toList(),
          bins.shape,
          DType.float64,
        );
      } else {
        binEdges = bins as NDArray<Float64>;
      }
    } else {
      throw ArgumentError('bins must be an integer or an NDArray.');
    }

    final M = binEdges.size;
    if (M < 2) {
      throw ArgumentError('bins must have at least 2 edges (1 bin).');
    }

    if (weights != null && !listEquals(weights.shape, x.shape)) {
      throw ArgumentError('weights must have the same shape as a.');
    }

    final binIndices = digitize(x, binEdges, right: false);
    final counts = bincount(binIndices, weights: weights, minlength: M + 1);

    final lastEdgeVal = binEdges.getCell([M - 1]);
    final lastEdgeArr = NDArray<Float64>.fromList(
      [lastEdgeVal],
      [],
      DType.float64,
    );
    final equalLastEdge = equal(x, lastEdgeArr);

    num equalLastEdgeWeightSum = 0;
    if (weights == null) {
      equalLastEdgeWeightSum = count_nonzero(equalLastEdge).scalar;
    } else {
      final zeroScalar = NDArray<num>.fromList([0.0], [], weights.dtype);
      final lastEdgeWeights =
          where(equalLastEdge, weights, zeroScalar) as NDArray<num>;
      equalLastEdgeWeightSum = sum<num>(lastEdgeWeights).scalar;
    }

    final currentLastBinVal = counts.getCell([M - 1]);
    counts.setCell([M - 1], currentLastBinVal + equalLastEdgeWeightSum);

    final histView = counts.slice([Slice(start: 1, stop: M)]);
    final hist = histView.copy();

    final binEdgesOut = bins is NDArray && bins.dtype == DType.float64
        ? (bins as NDArray<Float64>).copy()
        : binEdges;

    NDArray<num> finalHist = hist;
    if (density) {
      final totalSum = sum<num>(hist).scalar;
      final widths = subtract<Float64, Float64, Float64>(
        binEdges.slice([Slice(start: 1)]),
        binEdges.slice([Slice(stop: M - 1)]),
      );
      final totalSumArr = NDArray<Float64>.fromList(
        [totalSum.toDouble()],
        [],
        DType.float64,
      );
      final divisor = multiply<Float64, Float64, Float64>(widths, totalSumArr);
      final densityHist = divide<num, Float64, Float64>(hist, divisor);
      hist.dispose();
      finalHist = densityHist;
    }

    finalHist.detachToParentScope();
    binEdgesOut.detachToParentScope();
    return (hist: finalHist, binEdges: binEdgesOut);
  });
}
