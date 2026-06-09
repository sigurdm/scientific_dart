// ignore_for_file: non_constant_identifier_names
import 'dart:math' as math;
import '../ndarray.dart';
import 'dart:ffi' as ffi;
import '../ndarray_bindings.dart';
import '../scratch_arena.dart';

// Standalone operational relative cross-imports
import 'math.dart';
import 'helpers.dart';

/// Compute the sum of elements in the array.
///
/// If [axis] is provided, sums along that axis and returns a new array.
/// Otherwise, sums all elements and returns a scalar value of type [T].
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final s0 = sum(a, axis: 0); // Sum along rows
/// print(s0.data); // [4.0, 6.0]
/// ```
NDArray<T> sum<T extends Object>(NDArray<T> a, {int? axis, NDArray<T>? out}) {
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

/// Compute the product of elements in the array.
///
/// If [axis] is provided, multiplies along that axis and returns a new array.
/// Otherwise, multiplies all elements and returns a scalar value of type [T].
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final p0 = prod(a, axis: 0); // Product along rows
/// print(p0.data); // [3.0, 8.0]
/// ```
NDArray<T> prod<T extends Object>(NDArray<T> a, {int? axis, NDArray<T>? out}) {
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

/// Compute the arithmetic mean of array elements along a specified axis.
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
/// final m = mean(a); // returns 2.5 scalar
/// final m0 = mean(a, axis: 0); // returns NDArray [2.0, 3.0]
/// ```
///
/// Reference: [Arithmetic Mean](https://en.wikipedia.org/wiki/Arithmetic_mean)
NDArray<R> mean<R, T>(NDArray<T> a, {int? axis, NDArray<R>? out}) {
  final DType<R> targetDType =
      (a.dtype.isComplex ? DType.complex128 : DType.float64) as DType<R>;

  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != targetDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

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

/// Compute the standard deviation of array elements along a specified axis.
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
/// final s = std(a); // returns sqrt(1.25) scalar
/// ```
///
/// Reference: [Standard Deviation](https://en.wikipedia.org/wiki/Standard_deviation)
NDArray<double> std<T extends num>(
  NDArray<T> a, {
  int? axis,
  NDArray<double>? out,
}) {
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

/// Compute the variance along the specified axis, ignoring NaNs.
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
/// final a = `NDArray<double>`.fromList([1.0, double.nan, 2.0, 3.0], [2, 2], DType.float64);
/// final v = nanvar(a); // returns 0.6666666666666666
/// ```
NDArray<double> nanvar<T extends num>(
  NDArray<T> a, {
  int? axis,
  NDArray<double>? out,
}) {
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

/// Compute the standard deviation along the specified axis, ignoring NaNs.
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
/// final a = `NDArray<double>`.fromList([1.0, double.nan, 2.0, 3.0], [2, 2], DType.float64);
/// final s = nanstd(a); // returns sqrt(0.6666666666666666)
/// ```
NDArray<double> nanstd<T extends num>(
  NDArray<T> a, {
  int? axis,
  NDArray<double>? out,
}) {
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

/// Compute the minimum of elements in the array.
///
/// **Gotchas:**
/// - Returns a 0-dimensional [NDArray] if [axis] is null, or a new [NDArray] if [axis] is provided.
/// - Preserves the original data type (DType) of the input array along the reduction axis.
NDArray<T> min<T extends num>(NDArray<T> a, {int? axis, NDArray<T>? out}) {
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
    final offset = temp.offsetElements;
    var minVal = temp.data[offset];
    for (var i = 1; i < temp.size; i++) {
      minVal = math.min(minVal, temp.data[offset + i]);
    }
    if (!identical(temp, a)) {
      temp.dispose();
    }
    final result = out ?? NDArray<T>.create([], a.dtype);
    result.data[0] = minVal;
    return result;
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final selectors = List<Selector>.generate(a.shape.length, (j) {
    if (j == axis) return Index(0);
    return Slice();
  });
  final firstSlice = a.slice(selectors);

  final result = out ?? NDArray<T>.create(newShape, a.dtype);
  firstSlice.copy(out: result);

  for (var i = 1; i < a.shape[axis]; i++) {
    final currentSelectors = List<Selector>.generate(a.shape.length, (j) {
      if (j == axis) return Index(i);
      return Slice();
    });
    final currentSlice = a.slice(currentSelectors);
    elementWiseMin(result, currentSlice);
  }

  return result;
}

/// Compute the minimum of elements along a specified axis, ignoring NaNs.
///
/// This corresponds to NumPy's `nanmin` function.
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
/// print(nanmin(a)); // 1.0
/// ```
NDArray<T> nanmin<T extends Object>(
  NDArray<T> a, {
  int? axis,
  NDArray<T>? out,
}) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for nanmin');
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
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final List<dynamic> elements = size == a.data.length ? a.data : a.toList();

    var minVal = double.infinity;
    var hasValid = false;
    var hasNan = false;

    for (var i = 0; i < elements.length; i++) {
      final val = elements[i];
      if (val is double) {
        if (val.isNaN) {
          hasNan = true;
          continue;
        }
        if (val < minVal) {
          minVal = val;
          hasValid = true;
        }
      } else if (val is num) {
        final dVal = val.toDouble();
        if (dVal < minVal) {
          minVal = dVal;
          hasValid = true;
        }
      }
    }

    final result = out ?? NDArray<T>.create([], a.dtype);
    if (!hasValid) {
      result.data[0] = (hasNan ? double.nan : double.infinity) as dynamic;
    } else {
      result.data[0] =
          ((a.dtype == DType.float64 || a.dtype == DType.float32)
                  ? minVal
                  : minVal.toInt())
              as dynamic;
    }
    return result;
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final selectors = List<Selector>.generate(a.shape.length, (j) {
    if (j == axis) return Index(0);
    return Slice();
  });
  final firstSlice = a.slice(selectors);

  final result = out ?? NDArray<T>.create(newShape, a.dtype);
  firstSlice.copy(out: result);

  for (var i = 1; i < a.shape[axis]; i++) {
    final currentSelectors = List<Selector>.generate(a.shape.length, (j) {
      if (j == axis) return Index(i);
      return Slice();
    });
    final currentSlice = a.slice(currentSelectors);
    elementWiseNanMin(result, currentSlice);
  }

  return result;
}

/// Compute the maximum of elements in the array.
///
/// **Gotchas:**
/// - Returns a 0-dimensional [NDArray] if [axis] is null, or a new [NDArray] if [axis] is provided.
/// - Preserves the original data type (DType) of the input array along the reduction axis.
NDArray<T> max<T extends num>(NDArray<T> a, {int? axis, NDArray<T>? out}) {
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
    final offset = temp.offsetElements;
    var maxVal = temp.data[offset];
    for (var i = 1; i < temp.size; i++) {
      maxVal = math.max(maxVal, temp.data[offset + i]);
    }
    if (!identical(temp, a)) {
      temp.dispose();
    }
    final result = out ?? NDArray<T>.create([], a.dtype);
    result.data[0] = maxVal;
    return result;
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final selectors = List<Selector>.generate(a.shape.length, (j) {
    if (j == axis) return Index(0);
    return Slice();
  });
  final firstSlice = a.slice(selectors);

  final result = out ?? NDArray<T>.create(newShape, a.dtype);
  firstSlice.copy(out: result);

  for (var i = 1; i < a.shape[axis]; i++) {
    final currentSelectors = List<Selector>.generate(a.shape.length, (j) {
      if (j == axis) return Index(i);
      return Slice();
    });
    final currentSlice = a.slice(currentSelectors);
    elementWiseMax(result, currentSlice);
  }

  return result;
}

/// Compute the maximum of elements along a specified axis, ignoring NaNs.
///
/// This corresponds to NumPy's `nanmax` function.
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
/// print(nanmax(a)); // 3.0
/// ```
NDArray<T> nanmax<T extends Object>(
  NDArray<T> a, {
  int? axis,
  NDArray<T>? out,
}) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for nanmax');
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
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final List<dynamic> elements = size == a.data.length ? a.data : a.toList();

    var maxVal = -double.infinity;
    var hasValid = false;
    var hasNan = false;

    for (var i = 0; i < elements.length; i++) {
      final val = elements[i];
      if (val is double) {
        if (val.isNaN) {
          hasNan = true;
          continue;
        }
        if (val > maxVal) {
          maxVal = val;
          hasValid = true;
        }
      } else if (val is num) {
        final dVal = val.toDouble();
        if (dVal > maxVal) {
          maxVal = dVal;
          hasValid = true;
        }
      }
    }

    final result = out ?? NDArray<T>.create([], a.dtype);
    if (!hasValid) {
      result.data[0] = (hasNan ? double.nan : -double.infinity) as dynamic;
    } else {
      result.data[0] =
          ((a.dtype == DType.float64 || a.dtype == DType.float32)
                  ? maxVal
                  : maxVal.toInt())
              as dynamic;
    }
    return result;
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final selectors = List<Selector>.generate(a.shape.length, (j) {
    if (j == axis) return Index(0);
    return Slice();
  });
  final firstSlice = a.slice(selectors);

  final result = out ?? NDArray<T>.create(newShape, a.dtype);
  firstSlice.copy(out: result);

  for (var i = 1; i < a.shape[axis]; i++) {
    final currentSelectors = List<Selector>.generate(a.shape.length, (j) {
      if (j == axis) return Index(i);
      return Slice();
    });
    final currentSlice = a.slice(currentSelectors);
    elementWiseNanMax(result, currentSlice);
  }

  return result;
}

NDArray<R> cumsum<T, R>(NDArray<T> a, {int? axis, NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cumsum() on a disposed array.');
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

/// Compute the cumulative product of array elements along a specified axis.
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

/// Compute the cumulative minimum of array elements along a specified axis.
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

    final List elements = size == a.data.length ? a.data : a.toList();
    dynamic acc;
    for (var i = 0; i < elements.length; i++) {
      acc = (i == 0)
          ? elements[i]
          : (((acc as Comparable).compareTo(elements[i]) < 0)
                ? acc
                : elements[i]);
      result.data[i] = acc as T;
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

/// Compute the cumulative maximum of array elements along a specified axis.
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

    final List elements = size == a.data.length ? a.data : a.toList();
    dynamic acc;
    for (var i = 0; i < elements.length; i++) {
      acc = (i == 0)
          ? elements[i]
          : (((acc as Comparable).compareTo(elements[i]) > 0)
                ? acc
                : elements[i]);
      result.data[i] = acc as T;
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

/// Compute the variance of array elements along a specified axis.
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
/// final v = variance(a); // returns 1.25 scalar
/// ```
///
/// Reference: [Variance](https://en.wikipedia.org/wiki/Variance)
NDArray<double> variance<T extends num>(
  NDArray<T> a, {
  int? axis,
  NDArray<double>? out,
}) {
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

/// Compute the arithmetic mean along a specified axis, ignoring NaNs.
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
/// final a = `NDArray<double>`.fromList([1.0, double.nan, 3.0, 4.0], [2, 2], DType.float64);
/// final m = nanmean(a); // returns 2.6666666666666665
/// ```
NDArray<R> nanmean<R extends Object>(NDArray a, {int? axis, NDArray<R>? out}) {
  final DType<R> targetDType =
      (a.dtype.isComplex ? DType.complex128 : DType.float64) as DType<R>;

  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != targetDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (axis == null) {
    NDArray promotedA;
    if (a.dtype.isComplex || a.dtype.isFloating) {
      promotedA = a;
    } else {
      promotedA = promoteToDouble(a);
    }

    final List elements = promotedA.data;
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
      result =
          NDArray<Complex>.create(newShape, DType.complex128) as NDArray<R>;
    } else {
      result = NDArray<double>.create(newShape, DType.float64) as NDArray<R>;
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
