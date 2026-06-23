// ignore_for_file: non_constant_identifier_names
import 'dart:math' as math;
import '../ndarray.dart';
import 'dart:ffi' as ffi;
import '../ndarray_bindings.dart';
import '../scratch_arena.dart';

// Standalone operational relative cross-imports
import 'math.dart';
import 'helpers.dart';
import 'broadcasting.dart';
import 'manipulation.dart';
import 'linalg.dart';

/// Methods for estimating quantiles/percentiles.
///
/// The descriptions below refer to the taxonomy established by
/// Hyndman and Fan (1996), "Sample Quantiles in Statistical Packages".
///
/// Most methods interpolate between two adjacent order statistics
/// \(x_{(j)}\) and \(x_{(j+1)}\) using:
/// \[Q(p) = (1 - g) \cdot x_{(j)} + g \cdot x_{(j+1)}\]
/// where \(j\) is the floor of the virtual index, and \(g\) is the fractional part.
enum QuantileMethod {
  /// **Type 1**: Inverse of empirical cumulative distribution function.
  /// Discontinuous.
  ///
  /// \(g = 0\) if the virtual index is integer, otherwise \(1\).
  invertedCdf,

  /// **Type 2**: Similar to [invertedCdf] but with averaging at discontinuities.
  /// Discontinuous.
  ///
  /// \(g = 0.5\) if the virtual index is integer, otherwise \(1\).
  averagedInvertedCdf,

  /// **Type 3**: Nearest observation.
  /// Discontinuous.
  ///
  /// Rounds the virtual index to the nearest integer. If the fractional part
  /// is exactly 0.5, rounds to the nearest even index (1-based).
  closestObservation,

  /// **Type 4**: Linear interpolation of the empirical CDF.
  /// Continuous.
  ///
  /// \(p_k = k / N\). Virtual index is \(p \cdot N - 1\) (0-based).
  interpolatedInvertedCdf,

  /// **Type 5**: Hazen's piecewise linear function.
  /// Continuous.
  ///
  /// \(p_k = (k - 0.5) / N\). Virtual index is \(p \cdot N - 0.5\) (0-based).
  hazen,

  /// **Type 6**: Weibull-style interpolation.
  /// Continuous.
  ///
  /// \(p_k = k / (N + 1)\). Used by Minitab and SPSS.
  /// Virtual index is \(p \cdot (N + 1) - 1\) (0-based).
  weibull,

  /// **Type 7**: Linear interpolation (default).
  /// Continuous.
  ///
  /// \(p_k = (k - 1) / (N - 1)\). Used by S and Excel.
  /// Virtual index is \(p \cdot (N - 1)\) (0-based).
  linear,

  /// **Type 8**: Median-unbiased.
  /// Continuous.
  ///
  /// \(p_k = (k - 1/3) / (N + 1/3)\). Approximately median-unbiased
  /// regardless of the distribution. Recommended by Hyndman and Fan.
  medianUnbiased,

  /// **Type 9**: Normal-unbiased.
  /// Continuous.
  ///
  /// \(p_k = (k - 3/8) / (N + 1/4)\). Approximately unbiased if the
  /// underlying distribution is normal.
  normalUnbiased,

  /// **NumPy Compatibility**: Lower.
  /// Discontinuous.
  ///
  /// Always uses the lower of the two nearest observations (\(g = 0\)).
  lower,

  /// **NumPy Compatibility**: Higher.
  /// Discontinuous.
  ///
  /// Always uses the higher of the two nearest observations (\(g = 1\)).
  higher,

  /// **NumPy Compatibility**: Midpoint.
  /// Discontinuous.
  ///
  /// Always uses the average of the two nearest observations (\(g = 0.5\)).
  midpoint,

  /// **NumPy Compatibility**: Nearest.
  /// Discontinuous.
  ///
  /// Uses the nearest observation. Rounds half-integers to the nearest even integer.
  nearest,
}

/// Computes the sum of elements in the array.
///
/// If [axis] is provided, sums along that axis and returns a new array.
/// Otherwise, sums all elements and returns a 0-D array containing the sum.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final s0 = sum(a, axis: 0); // Sum along rows
/// print(s0.data); // [4.0, 6.0]
/// ```
NDArray<T> sum<T extends Object>(NDArray<T> a, {int? axis, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute sum of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write sum to a disposed output array.');
  }
  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    T? acc;
    if (a.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          acc = r_sum_double(a.pointer.cast(), size) as T;
        case DType.float32:
          acc = r_sum_float(a.pointer.cast(), size) as T;
        default:
          break;
      }
    }
    if (acc == null) {
      final List<T> elements = size == a.data.length ? a.data : a.toList();
      var current = elements.first;
      for (var i = 1; i < elements.length; i++) {
        current = ((current as dynamic) + elements[i]) as T;
      }
      acc = current;
    }
    final result = out ?? NDArray<T>.create([], a.dtype);
    result.data[0] = acc;
    return result;
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final result = out ?? NDArray<T>.zeros(newShape, a.dtype);
  if (out != null) {
    result.fill(normalizeScalar(0, a.dtype) as T);
  }

  switch (a.dtype) {
    case DType.float64:
      final rank = a.shape.length;
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesRes = cBuffer + (rank * 2);
      for (var i = 0; i < rank; i++) {
        cShape[i] = a.shape[i];
        cStridesA[i] = a.strides[i];
      }
      for (var i = 0; i < result.shape.length; i++) {
        cStridesRes[i] = result.strides[i];
      }

      s_sum_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        axis,
      );
      return result;
    case DType.float32:
      final rank = a.shape.length;
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesRes = cBuffer + (rank * 2);
      for (var i = 0; i < rank; i++) {
        cShape[i] = a.shape[i];
        cStridesA[i] = a.strides[i];
      }
      for (var i = 0; i < result.shape.length; i++) {
        cStridesRes[i] = result.strides[i];
      }

      s_sum_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        axis,
      );
      return result;
    default:
      break;
  }

  reduceRecursive<T, T>(
    a,
    result,
    List<int>.filled(a.shape.length, 0),
    List<int>.filled(newShape.length, 0),
    axis,
    0,
    (current, val) => ((current as dynamic) + val) as T,
  );
  return result;
}

/// Computes the product of elements in the array.
///
/// If [axis] is provided, multiplies along that axis and returns a new array.
/// Otherwise, multiplies all elements and returns a 0-D array containing the product.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final p0 = prod(a, axis: 0); // Product along rows
/// print(p0.data); // [3.0, 8.0]
/// ```
NDArray<T> prod<T extends Object>(NDArray<T> a, {int? axis, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute product of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write product to a disposed output array.');
  }
  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    T? acc;
    if (a.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          acc = r_prod_double(a.pointer.cast(), size) as T;
        case DType.float32:
          acc = r_prod_float(a.pointer.cast(), size) as T;
        default:
          break;
      }
    }
    if (acc == null) {
      final List<T> elements = size == a.data.length ? a.data : a.toList();
      var current = elements.first;
      for (var i = 1; i < elements.length; i++) {
        current = ((current as dynamic) * elements[i]) as T;
      }
      acc = current;
    }
    final result = out ?? NDArray<T>.create([], a.dtype);
    result.data[0] = acc;
    return result;
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final result = out ?? NDArray<T>.ones(newShape, a.dtype);
  if (out != null) {
    result.fill(normalizeScalar(1, a.dtype) as T);
  }

  reduceRecursive<T, T>(
    a,
    result,
    List<int>.filled(a.shape.length, 0),
    List<int>.filled(newShape.length, 0),
    axis,
    0,
    (current, val) => ((current as dynamic) * val) as T,
  );
  return result;
}

/// Returns true if all elements along a given [axis] evaluate to True.
///
/// If [axis] is omitted/null, performs a global reduction and returns a single Dart [bool].
///
/// **Preconditions:**
/// - The array [a] must not be disposed.
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([true, true, false], [3], DType.boolean);
/// final res = all(a); // false
/// ```
NDArray<bool> all<T extends Object>(
  NDArray<T> a, {
  int? axis,
  NDArray<bool>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute all() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write all() result to a disposed output array.');
  }

  var targetAxis = axis;
  if (targetAxis != null && targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }

  final targetShape = targetAxis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(targetAxis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.boolean) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (targetAxis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final List<T> elements = size == a.data.length ? a.data : a.toList();
    var allTrue = true;
    for (var i = 0; i < elements.length; i++) {
      if (!isTrueHelper(elements[i])) {
        allTrue = false;
        break;
      }
    }
    final result = out ?? NDArray<bool>.create([], DType.boolean);
    result.data[0] = allTrue;
    return result;
  }

  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(targetAxis);
  final result = out ?? NDArray<bool>.create(newShape, DType.boolean);
  result.fill(true); // Initialize to true everywhere

  reduceRecursive<T, bool>(
    a,
    result,
    List<int>.filled(a.shape.length, 0),
    List<int>.filled(newShape.length, 0),
    targetAxis,
    0,
    (current, val) => current && isTrueHelper(val),
  );

  return result;
}

/// Returns true if any element along a given [axis] evaluates to True.
///
/// If [axis] is omitted/null, performs a global reduction and returns a single Dart [bool].
///
/// **Preconditions:**
/// - The array [a] must not be disposed.
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([true, false, false], [3], DType.boolean);
/// final res = any(a); // true
/// ```
NDArray<bool> any<T extends Object>(
  NDArray<T> a, {
  int? axis,
  NDArray<bool>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute any() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write any() result to a disposed output array.');
  }

  var targetAxis = axis;
  if (targetAxis != null && targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }

  final targetShape = targetAxis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(targetAxis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.boolean) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (targetAxis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final List<T> elements = size == a.data.length ? a.data : a.toList();
    var anyTrue = false;
    for (var i = 0; i < elements.length; i++) {
      if (isTrueHelper(elements[i])) {
        anyTrue = true;
        break;
      }
    }
    final result = out ?? NDArray<bool>.create([], DType.boolean);
    result.data[0] = anyTrue;
    return result;
  }

  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(targetAxis);
  final result =
      out ??
      NDArray<bool>.zeros(newShape, DType.boolean); // Pre-initialized to false
  if (out != null) {
    result.fill(false);
  }

  reduceRecursive<T, bool>(
    a,
    result,
    List<int>.filled(a.shape.length, 0),
    List<int>.filled(newShape.length, 0),
    targetAxis,
    0,
    (current, val) => current || isTrueHelper(val),
  );

  return result;
}

/// Computes the arithmetic mean of array elements along a specified axis.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num` or Complex).
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [RangeError] if [axis] is out of range.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final m = mean(a); // returns 0-D array containing 2.5
/// final m0 = mean(a, axis: 0); // returns NDArray [2.0, 3.0]
/// ```
///
/// Reference: [Arithmetic Mean](https://en.wikipedia.org/wiki/Arithmetic_mean)
NDArray<R> mean<R, T>(NDArray<T> a, {int? axis, NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute mean of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write mean to a disposed output array.');
  }
  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  final expectedDType = a.dtype.isComplex ? DType.complex128 : DType.float64;
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != expectedDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }
  final DType<R> targetDType = expectedDType as DType<R>;

  if (axis == null) {
    if (a.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          final res = out ?? NDArray<R>.create([], DType.float64 as DType<R>);
          res.data[0] = r_mean_double(a.pointer.cast(), a.size) as R;
          return res;
        case DType.float32:
          final res = out ?? NDArray<R>.create([], DType.float64 as DType<R>);
          res.data[0] = r_mean_float(a.pointer.cast(), a.size) as R;
          return res;
        default:
          break;
      }
    }

    NDArray promotedA;
    if (a.dtype.isComplex || a.dtype.isFloating) {
      promotedA = a;
    } else {
      promotedA = promoteToDouble(a);
    }

    final s = sum<Object>(promotedA as NDArray<Object>, axis: axis);
    if (promotedA != a) {
      promotedA.dispose();
    }
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final meanVal = (s.data[0] as dynamic) / size;
    final NDArray<R> result;
    if (out != null) {
      result = out;
    } else {
      if (targetDType.isComplex) {
        result = NDArray<Complex>.create([], DType.complex128) as NDArray<R>;
      } else {
        result = NDArray<double>.create([], DType.float64) as NDArray<R>;
      }
    }
    result.data[0] = meanVal as R;
    s.dispose();
    return result;
  } else {
    final NDArray<R> result;
    if (out != null) {
      result = out;
    } else {
      if (targetDType.isComplex) {
        result =
            NDArray<Complex>.create(targetShape, DType.complex128)
                as NDArray<R>;
      } else {
        result =
            NDArray<double>.create(targetShape, DType.float64) as NDArray<R>;
      }
    }

    // Optimized axis-wise mean
    switch ((a.dtype, targetDType)) {
      case (DType.float64, DType.float64):
        final rank = a.shape.length;
        final cBuffer = ScratchArena.getStridedBuffer(rank);
        final cShape = cBuffer;
        final cStridesA = cBuffer + rank;
        final cStridesRes = cBuffer + (rank * 2);
        for (var i = 0; i < rank; i++) {
          cShape[i] = a.shape[i];
          cStridesA[i] = a.strides[i];
        }
        for (var i = 0; i < result.shape.length; i++) {
          cStridesRes[i] = result.strides[i];
        }

        s_mean_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
        return result;
      case (DType.float32, DType.float64):
        final rank = a.shape.length;
        final cBuffer = ScratchArena.getStridedBuffer(rank);
        final cShape = cBuffer;
        final cStridesA = cBuffer + rank;
        final cStridesRes = cBuffer + (rank * 2);
        for (var i = 0; i < rank; i++) {
          cShape[i] = a.shape[i];
          cStridesA[i] = a.strides[i];
        }
        for (var i = 0; i < result.shape.length; i++) {
          cStridesRes[i] = result.strides[i];
        }

        s_mean_float(
          a.pointer.cast(),
          cStridesA,
          ((result as dynamic) as NDArray<double>).pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
        return result;
      default:
        break;
    }

    if (targetDType.isComplex) {
      final promotedA =
          (a.dtype.isComplex ? a : promoteToComplex(a)) as NDArray<Complex>;
      sum<Complex>(
        promotedA,
        axis: axis,
        out: (result as dynamic) as NDArray<Complex>,
      );
      if (promotedA != a) promotedA.dispose();
    } else {
      final promotedA =
          (a.dtype.isFloating ? a : promoteToDouble(a)) as NDArray<double>;
      sum<double>(
        promotedA,
        axis: axis,
        out: (result as dynamic) as NDArray<double>,
      );
      if (promotedA != a) promotedA.dispose();
    }

    final sizeAxis = a.shape[axis];
    for (var i = 0; i < result.data.length; i++) {
      result.data[i] = ((result.data[i] as dynamic) / sizeAxis) as R;
    }
    return result;
  }
}

/// Computes the standard deviation of array elements along a specified axis.
///
/// Standard deviation is a measure of the spread of a distribution. The standard deviation
/// is computed for the flattened array by default, otherwise over the specified axis.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [RangeError] if [axis] is out of range.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
/// final s = std(a); // returns 0-D array containing sqrt(1.25)
/// ```
///
/// Reference: [Standard Deviation](https://en.wikipedia.org/wiki/Standard_deviation)
NDArray<double> std<T extends num>(
  NDArray<T> a, {
  int? axis,
  NDArray<double>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute std of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write std to a disposed output array.');
  }
  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  final v = variance(a, axis: axis);
  if (axis == null) {
    final stdVal = math.sqrt(v.data[0]);
    final result = out ?? NDArray<double>.create([], DType.float64);
    result.data[0] = stdVal;
    v.dispose();
    return result;
  } else {
    final res = sqrt(v, out: out);
    if (out != null) {
      v.dispose();
      return out;
    }
    final resultVal = NDArray<double>.view(
      res,
      shape: res.shape,
      strides: res.strides,
    );
    v.dispose();
    return resultVal;
  }
}

/// Computes the variance along the specified axis, ignoring NaNs.
///
/// **Preconditions:**
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total elements count.
///
/// **Example:**
/// ```dart
/// final a = NDArray<double>.fromList([1.0, double.nan, 2.0, 3.0], [2, 2], DType.float64);
/// final v = nanvar(a); // returns 0-D array containing 0.6666666666666666
/// ```
NDArray<double> nanvar<T extends num>(
  NDArray<T> a, {
  int? axis,
  NDArray<double>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute nanvar of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write nanvar to a disposed output array.');
  }
  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  final m = nanmean(a, axis: axis);

  if (axis == null) {
    var sumSqDiff = 0.0;
    final meanVal = m.data[0] as num;
    m.dispose();
    if (meanVal.toDouble().isNaN) {
      final result = out ?? NDArray<double>.create([], DType.float64);
      result.data[0] = double.nan;
      return result;
    }

    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final List<num> elements = size == a.data.length
        ? a.data as List<num>
        : a.toList() as List<num>;

    var count = 0;
    for (var i = 0; i < elements.length; i++) {
      final val = elements[i].toDouble();
      if (val.isNaN) continue;
      final diff = val - meanVal.toDouble();
      sumSqDiff += diff * diff;
      count++;
    }
    final result = out ?? NDArray<double>.create([], DType.float64);
    if (count == 0) {
      result.data[0] = double.nan;
    } else {
      result.data[0] = sumSqDiff / count;
    }
    return result;
  } else {
    // Reshape m to keep dimensions for broadcasting
    final targetShape = List<int>.from(a.shape);
    targetShape[axis] = 1;
    final reshapedM = m.reshape(targetShape);

    final diff = subtract(a, reshapedM);
    final sqDiff = multiply(diff, diff);

    // Convert to `NDArray<double>` to avoid truncation in nanmean
    final sqDiffDouble = NDArray<double>.create(sqDiff.shape, DType.float64);
    for (var i = 0; i < sqDiff.data.length; i++) {
      sqDiffDouble.data[i] = sqDiff.data[i].toDouble();
    }

    m.dispose();
    reshapedM.dispose();
    diff.dispose();
    sqDiff.dispose();

    final res = nanmean(sqDiffDouble, axis: axis, out: out);
    if (out != null) {
      sqDiffDouble.dispose();
      return out;
    }
    final resultVal = NDArray<double>.view(
      res,
      shape: res.shape,
      strides: res.strides,
    );
    sqDiffDouble.dispose();
    return resultVal;
  }
}

/// Computes the standard deviation along the specified axis, ignoring NaNs.
///
/// **Preconditions:**
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total elements count.
///
/// **Example:**
/// ```dart
/// final a = NDArray<double>.fromList([1.0, double.nan, 2.0, 3.0], [2, 2], DType.float64);
/// final s = nanstd(a); // returns 0-D array containing sqrt(0.6666666666666666)
/// ```
NDArray<double> nanstd<T extends num>(
  NDArray<T> a, {
  int? axis,
  NDArray<double>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute nanstd of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write nanstd to a disposed output array.');
  }
  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  final v = nanvar(a, axis: axis);
  if (axis == null) {
    final stdVal = math.sqrt(v.data[0]);
    final result = out ?? NDArray<double>.create([], DType.float64);
    result.data[0] = stdVal;
    v.dispose();
    return result;
  } else {
    final res = sqrt(v, out: out);
    if (out != null) {
      v.dispose();
      return out;
    }
    final resultVal = NDArray<double>.view(
      res,
      shape: res.shape,
      strides: res.strides,
    );
    v.dispose();
    return resultVal;
  }
}

/// Computes the minimum of elements in the array.
///
/// **Edge cases:**
/// - Returns a 0-dimensional [NDArray] if [axis] is null, or a new [NDArray] if [axis] is provided.
/// - Preserves the original data type (DType) of the input array along the reduction axis.
NDArray<T> min<T extends num>(NDArray<T> a, {int? axis, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute min of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write min to a disposed output array.');
  }
  if (axis == null && a.size == 0) {
    throw ArgumentError('Cannot compute min of an empty array.');
  }
  if (axis != null) {
    if (axis < 0 || axis >= a.shape.length) {
      throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
    }
    if (a.shape[axis] == 0) {
      throw ArgumentError('Cannot compute min along axis $axis of size 0.');
    }
  }

  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    if (!out.isContiguous) {
      throw ArgumentError('Provided out buffer must be contiguous.');
    }
  }

  if (axis == null) {
    final temp = a.isContiguous ? a : a.copy();
    final size = temp.size;
    final ptr = temp.pointer;
    dynamic minVal;
    switch (temp.dtype) {
      case DType.float64:
        minVal = r_min_double(ptr.cast(), size);
      case DType.float32:
        minVal = r_min_float(ptr.cast(), size);
      case DType.int64:
        minVal = r_min_int64_t(ptr.cast(), size);
      case DType.int32:
        minVal = r_min_int32_t(ptr.cast(), size);
      case DType.uint8:
        minVal = r_min_uint8_t(ptr.cast(), size);
      case DType.int16:
        minVal = r_min_int16_t(ptr.cast(), size);
    }
    if (!identical(temp, a)) {
      temp.dispose();
    }
    final result = out ?? NDArray<T>.create([], a.dtype);
    result.setCell([], minVal as T);
    return result;
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final result = out ?? NDArray<T>.create(newShape, a.dtype);

  final rank = a.shape.length;
  final marker = ScratchArena.marker;
  final cBuffer = ScratchArena.getStridedBuffer(rank);
  final cShape = cBuffer;
  final cStridesA = cBuffer + rank;
  final cStridesRes = cBuffer + (rank * 2);
  for (var i = 0; i < rank; i++) {
    cShape[i] = a.shape[i];
    cStridesA[i] = a.strides[i];
  }
  for (var i = 0; i < result.shape.length; i++) {
    cStridesRes[i] = result.strides[i];
  }

  try {
    switch (a.dtype) {
      case DType.float64:
        s_min_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.float32:
        s_min_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.int64:
        s_min_int64_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.int32:
        s_min_int32_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.uint8:
        s_min_uint8_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.int16:
        s_min_int16_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
    }
  } finally {
    ScratchArena.reset(marker);
  }

  return result;
}

/// Computes the minimum of elements along a specified axis, ignoring NaNs.
///
/// This corresponds to NumPy's `nanmin` function.
///
/// Returns a 0-dimensional [NDArray] if [axis] is null, or a new [NDArray] if [axis] is provided.
///
/// **Preconditions:**
/// - [axis], if provided, must be a valid axis index within `[0, rank - 1]`.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds.
/// - [UnsupportedError] if the array contains Complex numbers.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.nan, 3.0], [3], DType.float64);
/// print(nanmin(a).data); // [1.0] (0-D array)
/// ```
NDArray<T> nanmin<T extends Object>(
  NDArray<T> a, {
  int? axis,
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute nanmin of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write nanmin to a disposed output array.');
  }
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for nanmin');
  }
  if (axis == null && a.size == 0) {
    throw ArgumentError('Cannot compute nanmin of an empty array.');
  }
  if (axis != null) {
    if (axis < 0 || axis >= a.shape.length) {
      throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
    }
    if (a.shape[axis] == 0) {
      throw ArgumentError('Cannot compute nanmin along axis $axis of size 0.');
    }
  }

  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    if (!out.isContiguous) {
      throw ArgumentError('Provided out buffer must be contiguous.');
    }
  }

  if (axis == null) {
    final temp = a.isContiguous ? a : a.copy();
    final size = temp.size;
    final ptr = temp.pointer;
    dynamic minVal;
    switch (temp.dtype) {
      case DType.float64:
        minVal = r_nanmin_double(ptr.cast(), size);
      case DType.float32:
        minVal = r_nanmin_float(ptr.cast(), size);
      case DType.int64:
        minVal = r_min_int64_t(ptr.cast(), size);
      case DType.int32:
        minVal = r_min_int32_t(ptr.cast(), size);
      case DType.uint8:
        minVal = r_min_uint8_t(ptr.cast(), size);
      case DType.int16:
        minVal = r_min_int16_t(ptr.cast(), size);
      case DType.boolean:
        minVal = r_min_uint8_t(ptr.cast(), size) != 0;
      default:
        throw UnsupportedError('Unsupported dtype for nanmin: ${temp.dtype}');
    }
    if (!identical(temp, a)) {
      temp.dispose();
    }
    final result = out ?? NDArray<T>.create([], a.dtype);
    result.setCell([], minVal as T);
    return result;
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final result = out ?? NDArray<T>.create(newShape, a.dtype);

  final rank = a.shape.length;
  final marker = ScratchArena.marker;
  final cBuffer = ScratchArena.getStridedBuffer(rank);
  final cShape = cBuffer;
  final cStridesA = cBuffer + rank;
  final cStridesRes = cBuffer + (rank * 2);
  for (var i = 0; i < rank; i++) {
    cShape[i] = a.shape[i];
    cStridesA[i] = a.strides[i];
  }
  for (var i = 0; i < result.shape.length; i++) {
    cStridesRes[i] = result.strides[i];
  }

  try {
    switch (a.dtype) {
      case DType.float64:
        s_nanmin_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.float32:
        s_nanmin_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.int64:
        s_min_int64_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.int32:
        s_min_int32_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.uint8:
        s_min_uint8_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.int16:
        s_min_int16_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.boolean:
        s_min_uint8_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      default:
        throw UnsupportedError('Unsupported dtype for nanmin: ${a.dtype}');
    }
  } finally {
    ScratchArena.reset(marker);
  }

  return result;
}

/// Computes the maximum of elements in the array.
///
/// **Edge cases:**
/// - Returns a 0-dimensional [NDArray] if [axis] is null, or a new [NDArray] if [axis] is provided.
/// - Preserves the original data type (DType) of the input array along the reduction axis.
NDArray<T> max<T extends num>(NDArray<T> a, {int? axis, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute max of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write max to a disposed output array.');
  }
  if (axis == null && a.size == 0) {
    throw ArgumentError('Cannot compute max of an empty array.');
  }
  if (axis != null) {
    if (axis < 0 || axis >= a.shape.length) {
      throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
    }
    if (a.shape[axis] == 0) {
      throw ArgumentError('Cannot compute max along axis $axis of size 0.');
    }
  }

  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    if (!out.isContiguous) {
      throw ArgumentError('Provided out buffer must be contiguous.');
    }
  }

  if (axis == null) {
    final temp = a.isContiguous ? a : a.copy();
    final size = temp.size;
    final ptr = temp.pointer;
    dynamic maxVal;
    switch (temp.dtype) {
      case DType.float64:
        maxVal = r_max_double(ptr.cast(), size);
      case DType.float32:
        maxVal = r_max_float(ptr.cast(), size);
      case DType.int64:
        maxVal = r_max_int64_t(ptr.cast(), size);
      case DType.int32:
        maxVal = r_max_int32_t(ptr.cast(), size);
      case DType.uint8:
        maxVal = r_max_uint8_t(ptr.cast(), size);
      case DType.int16:
        maxVal = r_max_int16_t(ptr.cast(), size);
    }
    if (!identical(temp, a)) {
      temp.dispose();
    }
    final result = out ?? NDArray<T>.create([], a.dtype);
    result.setCell([], maxVal as T);
    return result;
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final result = out ?? NDArray<T>.create(newShape, a.dtype);

  final rank = a.shape.length;
  final marker = ScratchArena.marker;
  final cBuffer = ScratchArena.getStridedBuffer(rank);
  final cShape = cBuffer;
  final cStridesA = cBuffer + rank;
  final cStridesRes = cBuffer + (rank * 2);
  for (var i = 0; i < rank; i++) {
    cShape[i] = a.shape[i];
    cStridesA[i] = a.strides[i];
  }
  for (var i = 0; i < result.shape.length; i++) {
    cStridesRes[i] = result.strides[i];
  }

  try {
    switch (a.dtype) {
      case DType.float64:
        s_max_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.float32:
        s_max_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.int64:
        s_max_int64_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.int32:
        s_max_int32_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.uint8:
        s_max_uint8_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.int16:
        s_max_int16_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
    }
  } finally {
    ScratchArena.reset(marker);
  }

  return result;
}

/// Computes the maximum of elements along a specified axis, ignoring NaNs.
///
/// This corresponds to NumPy's `nanmax` function.
///
/// Returns a 0-dimensional [NDArray] if [axis] is null, or a new [NDArray] if [axis] is provided.
///
/// **Preconditions:**
/// - [axis], if provided, must be a valid axis index within `[0, rank - 1]`.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds.
/// - [UnsupportedError] if the array contains Complex numbers.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.nan, 3.0], [3], DType.float64);
/// print(nanmax(a).data); // [3.0] (0-D array)
/// ```
NDArray<T> nanmax<T extends Object>(
  NDArray<T> a, {
  int? axis,
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute nanmax of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write nanmax to a disposed output array.');
  }
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for nanmax');
  }
  if (axis == null && a.size == 0) {
    throw ArgumentError('Cannot compute nanmax of an empty array.');
  }
  if (axis != null) {
    if (axis < 0 || axis >= a.shape.length) {
      throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
    }
    if (a.shape[axis] == 0) {
      throw ArgumentError('Cannot compute nanmax along axis $axis of size 0.');
    }
  }

  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    if (!out.isContiguous) {
      throw ArgumentError('Provided out buffer must be contiguous.');
    }
  }

  if (axis == null) {
    final temp = a.isContiguous ? a : a.copy();
    final size = temp.size;
    final ptr = temp.pointer;
    dynamic maxVal;
    switch (temp.dtype) {
      case DType.float64:
        maxVal = r_nanmax_double(ptr.cast(), size);
      case DType.float32:
        maxVal = r_nanmax_float(ptr.cast(), size);
      case DType.int64:
        maxVal = r_max_int64_t(ptr.cast(), size);
      case DType.int32:
        maxVal = r_max_int32_t(ptr.cast(), size);
      case DType.uint8:
        maxVal = r_max_uint8_t(ptr.cast(), size);
      case DType.int16:
        maxVal = r_max_int16_t(ptr.cast(), size);
      case DType.boolean:
        maxVal = r_max_uint8_t(ptr.cast(), size) != 0;
      default:
        throw UnsupportedError('Unsupported dtype for nanmax: ${temp.dtype}');
    }
    if (!identical(temp, a)) {
      temp.dispose();
    }
    final result = out ?? NDArray<T>.create([], a.dtype);
    result.setCell([], maxVal as T);
    return result;
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final result = out ?? NDArray<T>.create(newShape, a.dtype);

  final rank = a.shape.length;
  final marker = ScratchArena.marker;
  final cBuffer = ScratchArena.getStridedBuffer(rank);
  final cShape = cBuffer;
  final cStridesA = cBuffer + rank;
  final cStridesRes = cBuffer + (rank * 2);
  for (var i = 0; i < rank; i++) {
    cShape[i] = a.shape[i];
    cStridesA[i] = a.strides[i];
  }
  for (var i = 0; i < result.shape.length; i++) {
    cStridesRes[i] = result.strides[i];
  }

  try {
    switch (a.dtype) {
      case DType.float64:
        s_nanmax_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.float32:
        s_nanmax_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.int64:
        s_max_int64_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.int32:
        s_max_int32_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.uint8:
        s_max_uint8_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.int16:
        s_max_int16_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      case DType.boolean:
        s_max_uint8_t(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          axis,
        );
      default:
        throw UnsupportedError('Unsupported dtype for nanmax: ${a.dtype}');
    }
  } finally {
    ScratchArena.reset(marker);
  }

  return result;
}

/// Computes the cumulative sum of array elements along a specified axis.
///
/// **Preconditions:**
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
/// - If provided, the [out] recycler must have compatible shape and dtype.
///
/// **Throws:**
/// - [StateError] if the array is disposed.
/// - [ArgumentError] if [axis] is out of bounds.
/// - [ArgumentError] if [out] recycler shape or dtype is incompatible.
///
/// **Example:**
/// {@example /example/cumulative_example.dart lang=dart}
NDArray<R> cumsum<T, R>(NDArray<T> a, {int? axis, NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cumsum() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write cumsum result to a disposed output array.');
  }

  final DType<dynamic> targetDType = a.dtype == DType.boolean
      ? DType.int32
      : a.dtype;
  final NDArray<R> result;
  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    result = out ?? NDArray<R>.create([size], targetDType as DType<R>);
    if (out != null) {
      if (!listEquals(out.shape, [size]) || out.dtype != targetDType) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
    }

    final List elements = size == a.data.length ? a.data : a.toList();
    dynamic acc;
    for (var i = 0; i < elements.length; i++) {
      final val = elements[i];
      final numVal = (val is bool) ? (val ? 1 : 0) : val;
      acc = (i == 0) ? numVal : ((acc as dynamic) + numVal) as dynamic;
      result.data[i] = acc as R;
    }
    return result;
  }

  var targetAxis = axis;
  if (targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }
  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  result = out ?? NDArray<R>.create(a.shape, targetDType as DType<R>);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  return cumOpFFI(a, targetAxis, result, CumOpType.sum);
}

/// Computes the cumulative product of array elements along a specified axis.
///
/// **Preconditions:**
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
/// - If provided, the [out] recycler must have compatible shape and dtype.
///
/// **Throws:**
/// - [StateError] if the array is disposed.
/// - [ArgumentError] if [axis] is out of bounds.
/// - [ArgumentError] if [out] recycler shape or dtype is incompatible.
///
/// **Example:**
/// {@example /example/cumulative_example.dart lang=dart}
NDArray<R> cumprod<T, R>(NDArray<T> a, {int? axis, NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cumprod() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write cumprod result to a disposed output array.');
  }

  final DType<dynamic> targetDType = a.dtype == DType.boolean
      ? DType.int32
      : a.dtype;
  final NDArray<R> result;
  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    result = out ?? NDArray<R>.create([size], targetDType as DType<R>);
    if (out != null) {
      if (!listEquals(out.shape, [size]) || out.dtype != a.dtype) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
    }

    final List elements = size == a.data.length ? a.data : a.toList();
    dynamic acc;
    for (var i = 0; i < elements.length; i++) {
      final val = elements[i];
      final numVal = (val is bool) ? (val ? 1 : 0) : val;
      acc = (i == 0) ? numVal : ((acc as dynamic) * numVal) as dynamic;
      result.data[i] = acc as R;
    }
    return result;
  }

  var targetAxis = axis;
  if (targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }
  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  result = out ?? NDArray<R>.create(a.shape, targetDType as DType<R>);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  return cumOpFFI(a, targetAxis, result, CumOpType.prod);
}

/// Computes the cumulative minimum of array elements along a specified axis.
///
/// **Preconditions:**
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
/// - If provided, the [out] recycler must have compatible shape and dtype.
///
/// **Throws:**
/// - [StateError] if the array is disposed.
/// - [ArgumentError] if [axis] is out of bounds.
/// - [ArgumentError] if [out] recycler shape or dtype is incompatible.
///
/// **Example:**
/// {@example /example/cumulative_example.dart lang=dart}
NDArray<T> cummin<T>(NDArray<T> a, {int? axis, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cummin() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write cummin result to a disposed output array.');
  }

  final NDArray<T> result;
  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    result = out ?? NDArray<T>.create([size], a.dtype);
    if (out != null) {
      if (!listEquals(out.shape, [size]) || out.dtype != a.dtype) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
    }

    final flatA = a.reshape([size]);
    cumOpFFI(flatA, 0, result, CumOpType.min);
    flatA.dispose();
    return result;
  }

  var targetAxis = axis;
  if (targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }
  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  result = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  return cumOpFFI(a, targetAxis, result, CumOpType.min);
}

/// Computes the cumulative maximum of array elements along a specified axis.
///
/// **Preconditions:**
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
/// - If provided, the [out] recycler must have compatible shape and dtype.
///
/// **Throws:**
/// - [StateError] if the array is disposed.
/// - [ArgumentError] if [axis] is out of bounds.
/// - [ArgumentError] if [out] recycler shape or dtype is incompatible.
///
/// **Example:**
/// {@example /example/cumulative_example.dart lang=dart}
NDArray<T> cummax<T>(NDArray<T> a, {int? axis, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cummax() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write cummax result to a disposed output array.');
  }

  final NDArray<T> result;
  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    result = out ?? NDArray<T>.create([size], a.dtype);
    if (out != null) {
      if (!listEquals(out.shape, [size]) || out.dtype != a.dtype) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
    }

    final flatA = a.reshape([size]);
    cumOpFFI(flatA, 0, result, CumOpType.max);
    flatA.dispose();
    return result;
  }

  var targetAxis = axis;
  if (targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }
  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  result = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  return cumOpFFI(a, targetAxis, result, CumOpType.max);
}

/// Computes the variance of array elements along a specified axis.
///
/// Variance is a measure of the spread of a distribution. The variance is computed for
/// the flattened array by default, otherwise over the specified axis.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [RangeError] if [axis] is out of range.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
/// final v = variance(a); // returns 0-D array containing 1.25
/// ```
///
/// Reference: [Variance](https://en.wikipedia.org/wiki/Variance)
NDArray<double> variance<T extends num>(
  NDArray<T> a, {
  int? axis,
  NDArray<double>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute variance of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write variance to a disposed output array.');
  }
  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  final m = mean(a, axis: axis);

  if (axis == null) {
    var sumSqDiff = 0.0;
    final meanVal = m.data[0] as num;
    m.dispose();

    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final List<num> elements = size == a.data.length
        ? a.data as List<num>
        : a.toList() as List<num>;

    for (var i = 0; i < elements.length; i++) {
      final val = elements[i];
      if (val is double && val.isNaN) continue;
      final diff = val.toDouble() - meanVal.toDouble();
      sumSqDiff += diff * diff;
    }
    final result = out ?? NDArray<double>.create([], DType.float64);
    result.data[0] = sumSqDiff / elements.length;
    return result;
  } else {
    // Reshape m to keep dimensions for broadcasting
    final targetShape = List<int>.from(a.shape);
    targetShape[axis] = 1;
    final reshapedM = m.reshape(targetShape);

    final diff = subtract(a, reshapedM);
    final sqDiff = multiply(diff, diff);

    final sqDiffDouble = NDArray<double>.create(sqDiff.shape, DType.float64);
    for (var i = 0; i < sqDiff.data.length; i++) {
      sqDiffDouble.data[i] = sqDiff.data[i].toDouble();
    }

    m.dispose();
    reshapedM.dispose();
    diff.dispose();
    sqDiff.dispose();

    final res = mean(sqDiffDouble, axis: axis, out: out);
    if (out != null) {
      sqDiffDouble.dispose();
      return out;
    }
    final resultVal = NDArray<double>.view(
      res,
      shape: res.shape,
      strides: res.strides,
    );
    sqDiffDouble.dispose();
    return resultVal;
  }
}

/// Computes the arithmetic mean along a specified axis, ignoring NaNs.
///
/// **Preconditions:**
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total elements count, walking
///   coordinate strides and tracking counts dynamically.
///
/// **Example:**
/// ```dart
/// final a = NDArray<double>.fromList([1.0, double.nan, 3.0, 4.0], [2, 2], DType.float64);
/// final m = nanmean(a); // returns 0-D array containing 2.6666666666666665
/// ```
NDArray<R> nanmean<R extends Object>(NDArray a, {int? axis, NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute nanmean of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write nanmean to a disposed output array.');
  }
  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  final expectedDType = a.dtype.isComplex ? DType.complex128 : DType.float64;
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != expectedDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }
  final DType<R> targetDType = expectedDType as DType<R>;

  if (axis == null) {
    NDArray promotedA;
    if (a.dtype.isComplex || a.dtype.isFloating) {
      promotedA = a;
    } else {
      promotedA = promoteToDouble(a);
    }

    final size = promotedA.shape.isEmpty
        ? 1
        : promotedA.shape.reduce((x, y) => x * y);
    final List elements =
        (size == promotedA.data.length && promotedA.isContiguous)
        ? promotedA.data
        : promotedA.toList();
    var sumVal = (targetDType.isComplex ? Complex(0, 0) : 0.0) as dynamic;
    var count = 0;
    for (var i = 0; i < elements.length; i++) {
      final val = elements[i];
      if (val is double && val.isNaN) continue;
      if (val is Complex && (val.real.isNaN || val.imag.isNaN)) continue;
      sumVal += val;
      count++;
    }
    if (promotedA != a) {
      promotedA.dispose();
    }
    final NDArray<R> result;
    if (out != null) {
      result = out;
    } else {
      if (targetDType.isComplex) {
        result = NDArray<Complex>.create([], DType.complex128) as NDArray<R>;
      } else {
        result = NDArray<double>.create([], DType.float64) as NDArray<R>;
      }
    }

    if (count == 0) {
      result.data[0] =
          (targetDType.isComplex ? Complex(double.nan, double.nan) : double.nan)
              as R;
    } else {
      result.data[0] = (sumVal / count) as R;
    }
    return result;
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final NDArray<R> result;
  if (out != null) {
    result = out;
    result.fill(normalizeScalar(0, targetDType) as R);
  } else {
    if (targetDType.isComplex) {
      result = NDArray<Complex>.zeros(newShape, DType.complex128) as NDArray<R>;
    } else {
      result = NDArray<double>.zeros(newShape, DType.float64) as NDArray<R>;
    }
  }
  final counts = NDArray<int>.zeros(newShape, DType.int32);

  if (targetDType.isComplex) {
    final promotedA =
        (a.dtype.isComplex ? a : promoteToComplex(a)) as NDArray<Complex>;
    nanReduceRecursive<Complex>(
      promotedA,
      (result as dynamic) as NDArray<Complex>,
      counts,
      List<int>.filled(promotedA.shape.length, 0),
      List<int>.filled(newShape.length, 0),
      axis,
      0,
    );
    if (promotedA != a) promotedA.dispose();
  } else {
    final promotedA =
        (a.dtype.isFloating ? a : promoteToDouble(a)) as NDArray<double>;
    nanReduceRecursive<double>(
      promotedA,
      (result as dynamic) as NDArray<double>,
      counts,
      List<int>.filled(promotedA.shape.length, 0),
      List<int>.filled(newShape.length, 0),
      axis,
      0,
    );
    if (promotedA != a) promotedA.dispose();
  }

  for (var i = 0; i < result.data.length; i++) {
    final c = counts.data[i];
    if (c == 0) {
      result.data[i] =
          (targetDType.isComplex ? Complex(double.nan, double.nan) : double.nan)
              as R;
    } else {
      result.data[i] = ((result.data[i] as dynamic) / c) as R;
    }
  }
  counts.dispose();
  return result;
}

/// Computes the q-th quantile along the specified axis.
///
/// The quantile is a value between 0 and 1.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - [q] must be within `[0.0, 1.0]`.
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [StateError] if the array is disposed.
/// - [ArgumentError] if [q] is out of bounds or [axis] is out of bounds.
NDArray<double> quantile<T extends Object>(
  NDArray<T> a,
  double q, {
  int? axis,
  QuantileMethod method = QuantileMethod.linear,
  NDArray<double>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute quantile of a disposed array.');
  }
  if (a.size == 0) {
    throw ArgumentError('Cannot compute quantile of an empty array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write quantile to a disposed output array.');
  }
  if (q < 0.0 || q > 1.0) {
    throw ArgumentError('Quantile q must be between 0.0 and 1.0. Got $q');
  }

  var targetAxis = axis;
  if (targetAxis != null && targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }

  final targetShape = targetAxis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(targetAxis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (targetAxis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final result = out ?? NDArray<double>.create([], DType.float64);
    if (a.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          result.data[0] = r_quantile_double(
            a.pointer.cast(),
            size,
            q,
            method.index,
          );
          return result;
        case DType.float32:
          result.data[0] = r_quantile_float(
            a.pointer.cast(),
            size,
            q,
            method.index,
          );
          return result;
        case DType.int64:
          result.data[0] = r_quantile_int64(
            a.pointer.cast(),
            size,
            q,
            method.index,
          );
          return result;
        case DType.int32:
          result.data[0] = r_quantile_int32(
            a.pointer.cast(),
            size,
            q,
            method.index,
          );
          return result;
        case DType.uint8:
          result.data[0] = r_quantile_uint8(
            a.pointer.cast(),
            size,
            q,
            method.index,
          );
          return result;
        default:
          throw ArgumentError('Unsupported dtype for quantile: ${a.dtype}');
      }
    } else {
      final flat = a.flatten();
      final resVal = r_quantile_helper(flat, flat.size, q, method.index);
      flat.dispose();
      result.data[0] = resVal;
      return result;
    }
  }

  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final result = out ?? NDArray<double>.zeros(targetShape, DType.float64);

  final rank = a.shape.length;
  final cBuffer = ScratchArena.getStridedBuffer(rank);
  final cShape = cBuffer;
  final cStridesA = cBuffer + rank;
  final cStridesRes = cBuffer + (rank * 2);
  for (var i = 0; i < rank; i++) {
    cShape[i] = a.shape[i];
    cStridesA[i] = a.strides[i];
  }
  for (var i = 0; i < result.shape.length; i++) {
    cStridesRes[i] = result.strides[i];
  }

  switch (a.dtype) {
    case DType.float64:
      s_quantile_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        targetAxis,
        q,
        method.index,
      );
    case DType.float32:
      s_quantile_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        targetAxis,
        q,
        method.index,
      );
    case DType.int64:
      s_quantile_int64(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        targetAxis,
        q,
        method.index,
      );
    case DType.int32:
      s_quantile_int32(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        targetAxis,
        q,
        method.index,
      );
    case DType.uint8:
      s_quantile_uint8(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        targetAxis,
        q,
        method.index,
      );
    default:
      result.dispose();
      throw ArgumentError('Unsupported dtype for quantile: ${a.dtype}');
  }

  return result;
}

double r_quantile_helper(NDArray a, int size, double q, int method) {
  switch (a.dtype) {
    case DType.float64:
      return r_quantile_double(a.pointer.cast(), size, q, method);
    case DType.float32:
      return r_quantile_float(a.pointer.cast(), size, q, method);
    case DType.int64:
      return r_quantile_int64(a.pointer.cast(), size, q, method);
    case DType.int32:
      return r_quantile_int32(a.pointer.cast(), size, q, method);
    case DType.uint8:
      return r_quantile_uint8(a.pointer.cast(), size, q, method);
    default:
      throw ArgumentError('Unsupported dtype for quantile: ${a.dtype}');
  }
}

/// Computes the q-th percentile of the data along the specified axis.
///
/// The percentile is a value between 0 and 100.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - [q] must be within `[0.0, 100.0]`.
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [StateError] if the array is disposed.
/// - [ArgumentError] if [q] is out of bounds or [axis] is out of bounds.
NDArray<double> percentile<T extends Object>(
  NDArray<T> a,
  double q, {
  int? axis,
  QuantileMethod method = QuantileMethod.linear,
  NDArray<double>? out,
}) {
  if (q < 0.0 || q > 100.0) {
    throw ArgumentError('Percentile q must be between 0.0 and 100.0. Got $q');
  }
  return quantile(a, q / 100.0, axis: axis, method: method, out: out);
}

/// Computes the median along the specified axis.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num` or Complex).
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [StateError] if the array is disposed.
/// - [ArgumentError] if [axis] is out of bounds.
NDArray<T> median<T extends Object>(
  NDArray<T> a, {
  int? axis,
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute median of a disposed array.');
  }
  if (a.size == 0) {
    throw ArgumentError('Cannot compute median of an empty array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write median to a disposed output array.');
  }

  var targetAxis = axis;
  if (targetAxis != null && targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }

  final targetShape = targetAxis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(targetAxis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (targetAxis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final result = out ?? NDArray<T>.create([], a.dtype);
    if (a.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          result.data[0] = r_median_double(a.pointer.cast(), size) as T;
          return result;
        case DType.float32:
          result.data[0] = r_median_float(a.pointer.cast(), size) as T;
          return result;
        case DType.int64:
          result.data[0] = r_median_int64(a.pointer.cast(), size) as T;
          return result;
        case DType.int32:
          result.data[0] = r_median_int32(a.pointer.cast(), size) as T;
          return result;
        case DType.uint8:
          result.data[0] = r_median_uint8(a.pointer.cast(), size) as T;
          return result;
        case DType.complex128:
          final res = r_median_complex128(a.pointer.cast(), size);
          result.data[0] = Complex(res.r, res.i) as T;
          return result;
        case DType.complex64:
          final res = r_median_complex64(a.pointer.cast(), size);
          result.data[0] = Complex(res.r, res.i) as T;
          return result;
        default:
          throw ArgumentError('Unsupported dtype for median: ${a.dtype}');
      }
    } else {
      final flat = a.flatten();
      final resVal = r_median_helper(flat, flat.size);
      flat.dispose();
      result.data[0] = resVal as T;
      return result;
    }
  }

  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final result = out ?? NDArray<T>.zeros(targetShape, a.dtype);

  final rank = a.shape.length;
  final cBuffer = ScratchArena.getStridedBuffer(rank);
  final cShape = cBuffer;
  final cStridesA = cBuffer + rank;
  final cStridesRes = cBuffer + (rank * 2);
  for (var i = 0; i < rank; i++) {
    cShape[i] = a.shape[i];
    cStridesA[i] = a.strides[i];
  }
  for (var i = 0; i < result.shape.length; i++) {
    cStridesRes[i] = result.strides[i];
  }

  switch (a.dtype) {
    case DType.float64:
      s_median_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        targetAxis,
      );
    case DType.float32:
      s_median_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        targetAxis,
      );
    case DType.int64:
      s_median_int64(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        targetAxis,
      );
    case DType.int32:
      s_median_int32(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        targetAxis,
      );
    case DType.uint8:
      s_median_uint8(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        targetAxis,
      );
    case DType.complex128:
      s_median_complex128(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        targetAxis,
      );
    case DType.complex64:
      s_median_complex64(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        targetAxis,
      );
    default:
      result.dispose();
      throw ArgumentError('Unsupported dtype for median: ${a.dtype}');
  }

  return result;
}

Object r_median_helper(NDArray a, int size) {
  switch (a.dtype) {
    case DType.float64:
      return r_median_double(a.pointer.cast(), size);
    case DType.float32:
      return r_median_float(a.pointer.cast(), size);
    case DType.int64:
      return r_median_int64(a.pointer.cast(), size);
    case DType.int32:
      return r_median_int32(a.pointer.cast(), size);
    case DType.uint8:
      return r_median_uint8(a.pointer.cast(), size);
    case DType.complex128:
      final res = r_median_complex128(a.pointer.cast(), size);
      return Complex(res.r, res.i);
    case DType.complex64:
      final res = r_median_complex64(a.pointer.cast(), size);
      return Complex(res.r, res.i);
    default:
      throw ArgumentError('Unsupported dtype for median: ${a.dtype}');
  }
}

/// Computes the range of values (maximum - minimum) along the specified axis.
///
/// If [axis] is null, it computes the range over the entire array and returns a 0-D array.
///
/// **Preconditions:**
/// - The array [a] must not be disposed.
/// - If [out] is provided, it must not be disposed, and it must have the correct shape and dtype.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 6.0, 4.0], [2, 2], DType.float64);
/// final p = ptp(a); // returns 0-D array containing 5.0
/// final p0 = ptp(a, axis: 0); // returns NDArray [5.0, 2.0]
/// ```
NDArray<T> ptp<T extends num>(NDArray<T> a, {int? axis, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute ptp of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write ptp to a disposed output array.');
  }

  final resolvedAxis = axis != null && axis < 0 ? a.rank + axis : axis;
  if (resolvedAxis != null && (resolvedAxis < 0 || resolvedAxis >= a.rank)) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final targetShape = resolvedAxis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(resolvedAxis));

  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  return NDArray.scope(() {
    final mx = max(a, axis: resolvedAxis);
    final mn = min(a, axis: resolvedAxis);
    final res = subtract<T, T, T>(mx, mn, out: out);
    if (out == null) {
      res.detachToParentScope();
    }
    return res;
  });
}

/// Helper to cast an NDArray to a target DType using s_cast_generic.
NDArray<R> _castTo<R>(NDArray a, DType<R> targetDType) {
  if (a.isDisposed) {
    throw StateError('Cannot execute _castTo on a disposed array.');
  }
  if (a.dtype == targetDType) {
    return a.copy() as NDArray<R>;
  }

  final res = NDArray<R>.create(a.shape, targetDType);
  final ndim = a.shape.length;
  final marker = ScratchArena.marker;

  try {
    final cBuffer = ScratchArena.getStridedBuffer(ndim);
    final cShape = cBuffer;
    final cStridesSrc = cBuffer + ndim;

    for (var i = 0; i < ndim; i++) {
      cShape[i] = a.shape[i];
      cStridesSrc[i] = a.strides[i];
    }

    s_cast_generic(
      a.pointer.cast(),
      cStridesSrc,
      encodeDType(a.dtype),
      res.pointer.cast(),
      encodeDType(targetDType),
      cShape,
      ndim,
    );
  } finally {
    ScratchArena.reset(marker);
  }
  return res;
}

/// Computes the weighted average along the specified axis.
///
/// If [weights] is null, it is equivalent to [mean].
///
/// **Preconditions:**
/// - The array [a] must not be disposed.
/// - If [weights] is provided, it must not be disposed.
/// - If [weights] is 1-D, its length must match the shape of [a] along [axis].
///   - If [axis] is null, [weights] can only be 1-D if [a] is also 1-D.
/// - If [weights] is not 1-D, it must have the same shape as [a].
/// - If [out] is provided, it must not be disposed and must have correct shape and dtype.
///
/// **Returns:**
/// A record containing:
/// - `average`: The computed weighted average.
/// - `sumOfWeights`: The sum of weights along the axis, promoted to the result type [R],
///   if [returned] is true. Otherwise null.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
/// final w = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
/// final res = average(a, weights: w, returned: true);
/// print(res.average.scalar); // 3.0
/// print(res.sumOfWeights?.scalar); // 10.0
/// ```
({NDArray<R> average, NDArray<R>? sumOfWeights})
average<T extends num, W extends num, R extends num>(
  NDArray<T> a, {
  int? axis,
  NDArray<W>? weights,
  bool returned = false,
  NDArray<R>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute average of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write average to a disposed output array.');
  }

  final resolvedAxis = axis != null && axis < 0 ? a.rank + axis : axis;
  if (resolvedAxis != null && (resolvedAxis < 0 || resolvedAxis >= a.rank)) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final targetShape = resolvedAxis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(resolvedAxis));

  if (out != null) {
    final DType expectedDType;
    if (weights == null) {
      expectedDType = a.dtype.isComplex ? DType.complex128 : DType.float64;
    } else {
      var resolved = resolveDType(a.dtype, weights.dtype);
      if (resolved.isInteger) {
        resolved = DType.float64;
      }
      expectedDType = resolved;
    }
    if (!listEquals(out.shape, targetShape) || out.dtype != expectedDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (weights == null) {
    final avg = mean<R, T>(a, axis: resolvedAxis, out: out);
    return (average: avg, sumOfWeights: null);
  }

  if (weights.isDisposed) {
    throw StateError('Cannot compute average with disposed weights.');
  }

  // Validate shapes
  if (weights.shape.length == 1) {
    if (resolvedAxis == null) {
      if (a.shape.length != 1) {
        throw ArgumentError(
          'If axis is null and weights is 1-D, input array must also be 1-D.',
        );
      }
      if (weights.size != a.size) {
        throw ArgumentError(
          'weights length (${weights.size}) must match a length (${a.size}).',
        );
      }
    } else {
      if (weights.shape[0] != a.shape[resolvedAxis]) {
        throw ArgumentError(
          'Length of 1-D weights (${weights.shape[0]}) must match shape of input along axis $resolvedAxis (${a.shape[resolvedAxis]}).',
        );
      }
    }
  } else {
    if (!listEquals(weights.shape, a.shape)) {
      throw ArgumentError(
        'Shape of weights ${weights.shape} must match shape of input ${a.shape} if weights is not 1-D.',
      );
    }
  }

  return NDArray.scope(() {
    NDArray<W> broadcastedWeights = weights;

    if (weights.shape.length == 1 && a.shape.length > 1) {
      final targetAxis = resolvedAxis!;
      final reshapedShape = List<int>.filled(a.shape.length, 1);
      reshapedShape[targetAxis] = weights.shape[0];
      broadcastedWeights = weights.reshape(reshapedShape);
    }

    final weighted_a = multiply<T, W, num>(a, broadcastedWeights);
    final weighted_sum = sum<num>(weighted_a, axis: resolvedAxis);
    final sum_of_weights = sum<num>(broadcastedWeights, axis: resolvedAxis);
    final avg = divide<num, num, R>(weighted_sum, sum_of_weights, out: out);

    NDArray<R>? sumOfWeightsResult;
    if (returned) {
      final promoted = _castTo<R>(sum_of_weights, avg.dtype);
      sumOfWeightsResult = broadcastTo<R>(promoted, avg.shape);
    }

    if (out == null) {
      avg.detachToParentScope();
    }
    sumOfWeightsResult?.detachToParentScope();

    return (average: avg, sumOfWeights: sumOfWeightsResult);
  });
}

NDArray<Float64> _diagonal(NDArray<Float64> a) {
  final M = a.shape[0];
  final s0 = a.strides[0];
  final s1 = a.strides[1];
  return NDArray<Float64>.view(a, shape: [M], strides: [s0 + s1]);
}

/// Estimate a covariance matrix, given data and weights.
NDArray<Float64> cov(
  NDArray m, {
  NDArray? y,
  bool rowvar = true,
  int? ddof,
  NDArray<int>? fweights,
  NDArray<num>? aweights,
}) {
  if (m.isDisposed) {
    throw StateError('Cannot execute cov on a disposed array.');
  }
  if (y != null && y.isDisposed) {
    throw StateError('Cannot execute cov with a disposed y array.');
  }
  if (fweights != null && fweights.isDisposed) {
    throw StateError('Cannot execute cov with disposed fweights.');
  }
  if (aweights != null && aweights.isDisposed) {
    throw StateError('Cannot execute cov with disposed aweights.');
  }

  if (m.dtype.isComplex || (y != null && y.dtype.isComplex)) {
    throw ArgumentError('Complex arrays are not supported in cov.');
  }

  return NDArray.scope(() {
    NDArray m2D;
    if (m.shape.length == 1) {
      m2D = m.reshape([1, m.shape[0]]);
    } else if (m.shape.length == 2) {
      m2D = rowvar ? m : m.transpose();
    } else {
      throw ArgumentError('m must be 1D or 2D.');
    }

    final N = m2D.shape[1];

    NDArray? y2D;
    if (y != null) {
      if (y.shape.length == 1) {
        y2D = y.reshape([1, y.shape[0]]);
      } else if (y.shape.length == 2) {
        y2D = rowvar ? y : y.transpose();
      } else {
        throw ArgumentError('y must be 1D or 2D.');
      }
      if (y2D.shape[1] != N) {
        throw ArgumentError(
          'm and y must have the same number of observations.',
        );
      }
    }

    if (fweights != null) {
      if (!listEquals(fweights.shape, [N])) {
        throw ArgumentError('fweights must be 1D of size $N.');
      }
      if (min(fweights).scalar < 0) {
        throw ArgumentError('fweights must be non-negative.');
      }
    }
    if (aweights != null) {
      if (!listEquals(aweights.shape, [N])) {
        throw ArgumentError('aweights must be 1D of size $N.');
      }
      if (min(aweights).scalar < 0) {
        throw ArgumentError('aweights must be non-negative.');
      }
    }

    final mDouble = _castTo(m2D, DType.float64);
    final yDouble = y2D != null ? _castTo(y2D, DType.float64) : null;

    NDArray<Float64> X_double;
    if (yDouble != null) {
      X_double = concatenate<Float64>([mDouble, yDouble], axis: 0);
    } else {
      X_double = mDouble;
    }

    final w_f = fweights != null ? _castTo(fweights, DType.float64) : null;
    final w_a = aweights != null ? _castTo(aweights, DType.float64) : null;

    final w = (w_f != null && w_a != null)
        ? multiply<Float64, Float64, Float64>(w_f, w_a)
        : (w_f ?? w_a);

    NDArray<Float64> mu;
    if (w == null) {
      mu = mean<Float64, Float64>(X_double, axis: 1);
    } else {
      final X_w = multiply<Float64, Float64, Float64>(X_double, w);
      final sum_X_w = sum<Float64>(X_w, axis: 1);
      final sum_w = sum<Float64>(w).scalar;
      final sum_w_arr = NDArray<Float64>.fromList([sum_w], [], DType.float64);
      mu = divide<Float64, Float64, Float64>(sum_X_w, sum_w_arr);
    }

    final mu_reshaped = mu.reshape([X_double.shape[0], 1]);
    final X_centered = subtract<Float64, Float64, Float64>(
      X_double,
      mu_reshaped,
    );

    NDArray<Float64> fact;
    if (w == null) {
      fact = matmul<Float64, Float64, Float64>(
        X_centered,
        X_centered.transpose(),
      );
    } else {
      final X_centered_w = multiply<Float64, Float64, Float64>(X_centered, w);
      fact = matmul<Float64, Float64, Float64>(
        X_centered_w,
        X_centered.transpose(),
      );
    }

    final ddof_val = ddof ?? 1;
    double denom;

    if (w == null) {
      denom = (N - ddof_val).toDouble();
    } else {
      final sum_w = sum<Float64>(w).scalar;
      if (ddof_val == 0) {
        denom = sum_w;
      } else {
        if (w_a == null) {
          denom = sum_w - ddof_val;
        } else {
          final w_times_a = multiply<Float64, Float64, Float64>(w, w_a);
          final sum_w_times_a = sum<Float64>(w_times_a).scalar;
          denom = sum_w - ddof_val * sum_w_times_a / sum_w;
        }
      }
    }

    final denom_arr = NDArray<Float64>.fromList([denom], [], DType.float64);
    final result = divide<Float64, Float64, Float64>(fact, denom_arr);

    var finalResult = result;
    if (m.shape.length == 1 && y == null) {
      final view = result.reshape([]);
      finalResult = view.copy();
    }

    finalResult.detachToParentScope();
    return finalResult;
  });
}

/// Pearson product-moment correlation coefficients.
NDArray<Float64> corrcoef(NDArray x, {NDArray? y, bool rowvar = true}) {
  if (x.isDisposed) {
    throw StateError('Cannot execute corrcoef on a disposed array.');
  }
  if (y != null && y.isDisposed) {
    throw StateError('Cannot execute corrcoef with a disposed y array.');
  }

  return NDArray.scope(() {
    final C = cov(x, y: y, rowvar: rowvar);

    NDArray<Float64> C2D;
    if (C.shape.isEmpty) {
      C2D = C.reshape([1, 1]);
    } else {
      C2D = C;
    }

    final M = C2D.shape[0];
    final d = _diagonal(C2D);
    final stddev = sqrt<Float64, Float64>(d);
    final stddev_reshaped_col = stddev.reshape([M, 1]);
    final stddev_reshaped_row = stddev.reshape([1, M]);
    final divisor = multiply<Float64, Float64, Float64>(
      stddev_reshaped_col,
      stddev_reshaped_row,
    );
    final R = divide<Float64, Float64, Float64>(C2D, divisor);

    R.detachToParentScope();
    return R;
  });
}
