// ignore_for_file: non_constant_identifier_names
import 'dart:typed_data';
import 'dart:math' as math;
import '../ndarray.dart';
import 'package:openblas/openblas.dart';
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import '../ndarray_bindings.dart';
import '../scratch_arena.dart';

// Standalone operational relative cross-imports
import 'stats.dart';
import 'manipulation.dart';
import 'broadcasting.dart';
import 'helpers.dart';

/// Configure the number of parallel execution threads used by OpenBLAS at runtime.
///
/// **Preconditions:**
/// - [numThreads] must be greater than or equal to 1.
///
/// **Throws:**
/// - [ArgumentError] if [numThreads] is less than 1.
///
/// **Example:**
/// ```dart
/// setNumThreads(1); // Disable multi-threading to bypass overhead on small matrices
/// ```
void setNumThreads(int numThreads) {
  if (numThreads < 1) {
    throw ArgumentError(
      'Number of threads must be at least 1 (was $numThreads)',
    );
  }
  openblas_set_num_threads(numThreads);
}

/// Compute the element-wise square root of the array.
///
/// Returns a new array with the results.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 4.0, 9.0], [3], DType.float64);
/// final b = sqrt(a);
/// print(b.data); // [1.0, 2.0, 3.0]
/// ```
///
/// **Gotchas:**
/// - Negative values will result in [double.nan].
NDArray<R> sqrt<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for sqrt.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_sqrt_double(a.pointer.cast(), result.pointer.cast(), a.size);
      return result;
    } else if (a.dtype == DType.float32) {
      v_sqrt_float(a.pointer.cast(), result.pointer.cast(), a.size);
      return result;
    }
  }

  final temp = a.isContiguous ? a : a.copy();
  final tempNum = temp as NDArray<num>;
  final rData = result.data as List<double>;
  final offset = temp.offsetElements;
  final resOffset = result.offsetElements;
  for (var i = 0; i < temp.size; i++) {
    rData[resOffset + i] = math.sqrt(tempNum.data[offset + i].toDouble());
  }
  if (!identical(temp, a)) {
    temp.dispose();
  }
  return result;
}

/// Compute the element-wise square of the input array.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [ArgumentError] if the provided [out] buffer shape or dtype is incompatible.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([2.0, 3.0], [2], DType.float64);
/// final b = square(a); // [4.0, 9.0]
/// ```
NDArray<T> square<T>(NDArray<T> a, {NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute square() on a disposed array.');
  }
  final result = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for square.',
      );
    }
  }

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_square_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.float32:
        v_square_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.int64:
        v_square_int64(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.int32:
        v_square_int32(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.complex128:
        v_square_complex128(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      case DType.complex64:
        v_square_complex64(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      case DType.boolean:
        final aBool = a as NDArray<bool>;
        final rBool = result as NDArray<bool>;
        for (var i = 0; i < a.data.length; i++) {
          rBool.data[i] = aBool.data[i];
        }
        return result;
      case DType.uint8:
      case DType.int16:
        final aNum = a as NDArray<num>;
        final rNum = result as NDArray<num>;
        for (var i = 0; i < a.data.length; i++) {
          final val = aNum.data[i];
          rNum.data[i] = val * val;
        }
        return result;
    }
  } else {
    final rank = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }
    switch (a.dtype) {
      case DType.float64:
        s_square_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.float32:
        s_square_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.int64:
        s_square_int64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.int32:
        s_square_int32(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.complex128:
        s_square_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.complex64:
        s_square_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.boolean:
        unaryOp<bool, bool>(
          result.data as List<bool>,
          a.data as List<bool>,
          a.shape,
          a.strides,
          NDArray.computeCStrides(a.shape),
          0,
          0,
          0,
          (x) => x,
        );
        return result;
      case DType.uint8:
      case DType.int16:
        unaryOp<num, num>(
          result.data as List<num>,
          a.data as List<num>,
          a.shape,
          a.strides,
          NDArray.computeCStrides(a.shape),
          0,
          0,
          0,
          (x) => x * x,
        );
        return result;
    }
  }
}

/// Compute the element-wise sine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous array layouts, offloads the loop directly to high-speed native C
///   vector math kernels (`v_sin_double`/`v_sin_float`), bypassing all Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/transcendental_example.dart lang=dart}
///
/// Reference: [Trigonometric Sine Function](https://en.wikipedia.org/wiki/Sine_and_cosine)
NDArray<R> sin<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  final DType<dynamic> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  }

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for sin.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_sin_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_sin_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.complex128) {
      v_sin_complex128(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.complex64) {
      v_sin_complex64(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  } else {
    final rank = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }

    if (a.dtype == DType.float64) {
      s_sin_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      s_sin_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex128) {
      s_sin_complex128(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex64) {
      s_sin_complex64(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    }
  }

  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.sin(x.toDouble()),
    );
  } else {
    unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.sin(x.toDouble()),
    );
  }
  return result;
}

/// Compute the element-wise cosine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous array layouts, offloads the loop directly to high-speed native C
///   vector math kernels (`v_cos_double`/`v_cos_float`), bypassing all Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/transcendental_example.dart lang=dart}
///
/// Reference: [Trigonometric Cosine Function](https://en.wikipedia.org/wiki/Sine_and_cosine)
NDArray<R> cos<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cos() on a disposed array.');
  }
  final DType<dynamic> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  }

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for cos.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }

  if (a.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_cos_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.float32:
        v_cos_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.complex128:
        v_cos_complex128(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      case DType.complex64:
        v_cos_complex64(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      default:
        break;
    }
  } else {
    final rank = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }

    if (a.dtype == DType.float64) {
      s_cos_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      s_cos_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex128) {
      s_cos_complex128(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex64) {
      s_cos_complex64(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    }
  }

  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.cos(x.toDouble()),
    );
  } else {
    unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.cos(x.toDouble()),
    );
  }
  return result;
}

/// Compute the element-wise exponential of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous array layouts, offloads the loop directly to high-speed native C
///   vector math kernels (`v_exp_double`/`v_exp_float`), bypassing all Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/transcendental_example.dart lang=dart}
///
/// Reference: [Exponential Function](https://en.wikipedia.org/wiki/Exponential_function)
NDArray<R> exp<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute exp() on a disposed array.');
  }
  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for exp.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }

  if (a.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_exp_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.float32:
        v_exp_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      default:
        break;
    }
  } else {
    final rank = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }

    if (a.dtype == DType.float64) {
      s_exp_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      s_exp_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    }
  }

  unaryOp<T, R>(
    result.data,
    a.data,
    a.shape,
    a.strides,
    result.strides,
    0,
    a.offsetElements,
    result.offsetElements,
    (x) => math.exp((x as num).toDouble()) as R,
  );
  return result;
}

/// Compute the element-wise natural logarithm of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and the resolved floating-point dtype (Float32 if [a] is Float32, Float64 otherwise).
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous array layouts, offloads the loop directly to high-speed native C
///   vector math kernels (`v_log_double`/`v_log_float`), bypassing all Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/transcendental_example.dart lang=dart}
///
/// Reference: [Natural Logarithm](https://en.wikipedia.org/wiki/Natural_logarithm)
NDArray<R> log<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute log() on a disposed array.');
  }
  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for log.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }

  if (a.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_log_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.float32:
        v_log_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      default:
        break;
    }
  } else {
    final rank = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }
    if (a.dtype == DType.float64) {
      s_log_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      s_log_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    }
  }

  unaryOp<T, R>(
    result.data,
    a.data,
    a.shape,
    a.strides,
    result.strides,
    0,
    a.offsetElements,
    result.offsetElements,
    (x) => math.log((x as num).toDouble()) as R,
  );
  return result;
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

/// Compute the sum of array elements along a specified axis, treating NaNs as zeros.
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
/// final a = `NDArray<double>`.fromList([1.0, double.nan, 3.0, double.nan], [2, 2], DType.float64);
/// final s = nansum(a); // returns 4.0
/// ```
NDArray<T> nansum<T extends Object>(
  NDArray<T> a, {
  int? axis,
  NDArray<T>? out,
}) {
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
    final List<T> elements = size == a.data.length ? a.data : a.toList();
    T acc;
    if (a.dtype == DType.int32 || a.dtype == DType.int64) {
      var sumVal = 0;
      for (var i = 0; i < elements.length; i++) {
        sumVal += elements[i] as int;
      }
      acc = sumVal as T;
    } else if (a.dtype == DType.complex64 || a.dtype == DType.complex128) {
      var sumVal = Complex(0.0, 0.0);
      for (var i = 0; i < elements.length; i++) {
        final val = elements[i] as Complex;
        if (val.real.isNaN || val.imag.isNaN) continue;
        sumVal += val;
      }
      acc = sumVal as T;
    } else {
      var sumVal = 0.0;
      for (var i = 0; i < elements.length; i++) {
        final val = elements[i] as double;
        if (val.isNaN) continue;
        sumVal += val;
      }
      acc = sumVal as T;
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

  reduceRecursive<T, T>(
    a,
    result,
    List<int>.filled(a.shape.length, 0),
    List<int>.filled(newShape.length, 0),
    axis,
    0,
    (current, val) {
      if (val is double && val.isNaN) return current;
      if (val is Complex && (val.real.isNaN || val.imag.isNaN)) return current;
      return ((current as dynamic) + val) as T;
    },
  );
  return result;
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
    final temp = a.isContiguous ? a : a.copy();
    NDArray promotedA;
    if (temp.dtype.isComplex || temp.dtype.isFloating) {
      promotedA = temp;
    } else {
      promotedA = promoteToDouble(temp);
    }

    final List elements = promotedA.data;
    final offset = promotedA.offsetElements;
    var sumVal = (targetDType.isComplex ? Complex(0, 0) : 0.0) as dynamic;
    var count = 0;
    for (var i = 0; i < promotedA.size; i++) {
      final val = elements[offset + i];
      if (val is double && val.isNaN) continue;
      if (val is Complex && (val.real.isNaN || val.imag.isNaN)) continue;
      sumVal += val;
      count++;
    }
    if (promotedA != temp && promotedA != a) {
      promotedA.dispose();
    }
    if (temp != a) {
      temp.dispose();
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

/// Stacks arrays in sequence vertically (row wise).
NDArray<T> vstack<T extends Object>(List<NDArray<T>> arrays) {
  return concatenate(arrays, axis: 0);
}

/// Stacks arrays in sequence horizontally (column wise).
NDArray<T> hstack<T extends Object>(List<NDArray<T>> arrays) {
  return concatenate(arrays, axis: 1);
}

/// Return an array copy of the given object.
///
/// This function corresponds to NumPy's `copy` function. It returns a deep copy
/// of [a] that respects shape, strides, and `DType.`
///
/// **Throws:**
/// - [StateError] if the array [a] is already disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1, 2], [2], DType.int32);
/// final b = copy(a);
/// b.data[0] = 99;
/// print(a.data[0]); // 1 (decoupled memory!)
/// ```
NDArray<T> copy<T extends Object>(NDArray<T> a) {
  return a.copy();
}

/// Compute the element-wise tangent of the array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<R> tan<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute tan() on a disposed array.');
  }
  final DType<dynamic> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  }

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for tan.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }

  if (a.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_tan_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.float32:
        v_tan_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.complex128:
        v_tan_complex128(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      case DType.complex64:
        v_tan_complex64(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      default:
        break;
    }
  } else {
    final rank = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }

    if (a.dtype == DType.float64) {
      s_tan_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      s_tan_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex128) {
      s_tan_complex128(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex64) {
      s_tan_complex64(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    }
  }

  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.tan(x.toDouble()),
    );
  } else {
    unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.tan(x.toDouble()),
    );
  }
  return result;
}

/// Compute the element-wise arc sine (inverse sine) of the array.
///
/// **Preconditions:**
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([0.0, 1.0], [2], DType.float64);
/// final b = asin(a); // [0.0, 1.570796...]
/// ```
NDArray<R> asin<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute asin() on a disposed array.');
  }
  final DType<dynamic> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  }

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for asin.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }

  if (a.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_asin_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.float32:
        v_asin_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.complex128:
        v_asin_complex128(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      case DType.complex64:
        v_asin_complex64(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      default:
        break;
    }
  } else {
    final rank = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }
    if (a.dtype == DType.float64) {
      s_asin_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      s_asin_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex128) {
      s_asin_complex128(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex64) {
      s_asin_complex64(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    }
  }

  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.asin(x.toDouble()),
    );
  } else {
    unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.asin(x),
    );
  }
  return result;
}

/// Compute the element-wise arc cosine (inverse cosine) of the array.
///
/// **Preconditions:**
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 0.0], [2], DType.float64);
/// final b = acos(a); // [0.0, 1.570796...]
/// ```
NDArray<R> acos<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute acos() on a disposed array.');
  }
  final DType<dynamic> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  }

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for acos.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }

  if (a.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_acos_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.float32:
        v_acos_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.complex128:
        v_acos_complex128(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      case DType.complex64:
        v_acos_complex64(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      default:
        break;
    }
  } else {
    final rank = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }
    if (a.dtype == DType.float64) {
      s_acos_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      s_acos_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex128) {
      s_acos_complex128(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex64) {
      s_acos_complex64(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    }
  }

  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.acos(x.toDouble()),
    );
  } else {
    unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.acos(x),
    );
  }
  return result;
}

/// Compute the element-wise arc tangent (inverse tangent) of the array.
///
/// **Preconditions:**
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([0.0, 1.0], [2], DType.float64);
/// final b = atan(a); // [0.0, 0.785398...]
/// ```
NDArray<R> atan<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute atan() on a disposed array.');
  }
  final DType<dynamic> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  }

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for atan.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }

  if (a.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_atan_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.float32:
        v_atan_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.complex128:
        v_atan_complex128(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      case DType.complex64:
        v_atan_complex64(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      default:
        break;
    }
  } else {
    final rank = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }
    if (a.dtype == DType.float64) {
      s_atan_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      s_atan_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex128) {
      s_atan_complex128(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex64) {
      s_atan_complex64(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    }
  }

  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.atan(x.toDouble()),
    );
  } else {
    unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.atan(x),
    );
  }
  return result;
}

/// Compute the element-wise hyperbolic sine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/hyperbolic_example.dart lang=dart}
NDArray<double> sinh<T extends num>(NDArray<T> a, {NDArray<double>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute sinh() on a disposed array.');
  }
  final DType<double> targetDType = a.dtype == DType.float32
      ? DType.float32 as DType<double>
      : DType.float64 as DType<double>;
  final result = out ?? NDArray<double>.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for sinh.',
      );
    }
  }

  if (a.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_sinh_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.float32:
        v_sinh_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      default:
        break;
    }
  } else {
    final rank = a.shape.length;
    final cShape = malloc<ffi.Int>(rank);
    final cStridesA = malloc<ffi.Int>(rank);
    final cStridesRes = malloc<ffi.Int>(rank);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }
    try {
      switch (a.dtype) {
        case DType.float64:
          s_sinh_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        case DType.float32:
          s_sinh_float(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        default:
          break;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesRes);
    }
  }

  unaryOp<T, double>(
    result.data,
    a.data,
    a.shape,
    a.strides,
    result.strides,
    0,
    a.offsetElements,
    result.offsetElements,
    (x) {
      final val = (x as num).toDouble();
      return (math.exp(val) - math.exp(-val)) / 2.0;
    },
  );
  return result;
}

/// Compute the element-wise hyperbolic cosine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/hyperbolic_example.dart lang=dart}
NDArray<double> cosh<T extends num>(NDArray<T> a, {NDArray<double>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cosh() on a disposed array.');
  }
  final DType<double> targetDType = a.dtype == DType.float32
      ? DType.float32 as DType<double>
      : DType.float64 as DType<double>;
  final result = out ?? NDArray<double>.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for cosh.',
      );
    }
  }

  if (a.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_cosh_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.float32:
        v_cosh_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      default:
        break;
    }
  } else {
    final rank = a.shape.length;
    final cShape = malloc<ffi.Int>(rank);
    final cStridesA = malloc<ffi.Int>(rank);
    final cStridesRes = malloc<ffi.Int>(rank);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }
    try {
      switch (a.dtype) {
        case DType.float64:
          s_cosh_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        case DType.float32:
          s_cosh_float(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        default:
          break;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesRes);
    }
  }

  unaryOp<T, double>(
    result.data,
    a.data,
    a.shape,
    a.strides,
    result.strides,
    0,
    a.offsetElements,
    result.offsetElements,
    (x) {
      final val = (x as num).toDouble();
      return (math.exp(val) + math.exp(-val)) / 2.0;
    },
  );
  return result;
}

/// Compute the element-wise hyperbolic tangent of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/hyperbolic_example.dart lang=dart}
NDArray<double> tanh<T extends num>(NDArray<T> a, {NDArray<double>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute tanh() on a disposed array.');
  }
  final DType<double> targetDType = a.dtype == DType.float32
      ? DType.float32 as DType<double>
      : DType.float64 as DType<double>;
  final result = out ?? NDArray<double>.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for tanh.',
      );
    }
  }

  if (a.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_tanh_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.float32:
        v_tanh_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      default:
        break;
    }
  } else {
    final rank = a.shape.length;
    final cShape = malloc<ffi.Int>(rank);
    final cStridesA = malloc<ffi.Int>(rank);
    final cStridesRes = malloc<ffi.Int>(rank);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }
    try {
      switch (a.dtype) {
        case DType.float64:
          s_tanh_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        case DType.float32:
          s_tanh_float(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        default:
          break;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesRes);
    }
  }

  unaryOp<T, double>(
    result.data,
    a.data,
    a.shape,
    a.strides,
    result.strides,
    0,
    a.offsetElements,
    result.offsetElements,
    (x) {
      final val = (x as num).toDouble();
      final exp2val = math.exp(2.0 * val);
      return (exp2val - 1.0) / (exp2val + 1.0);
    },
  );
  return result;
}

/// Compute the element-wise inverse hyperbolic sine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/hyperbolic_example.dart lang=dart}
NDArray<double> asinh<T extends num>(NDArray<T> a, {NDArray<double>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute asinh() on a disposed array.');
  }
  final DType<double> targetDType = a.dtype == DType.float32
      ? DType.float32 as DType<double>
      : DType.float64 as DType<double>;
  final result = out ?? NDArray<double>.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for asinh.',
      );
    }
  }

  if (a.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_asinh_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.float32:
        v_asinh_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      default:
        break;
    }
  } else {
    final rank = a.shape.length;
    final cShape = malloc<ffi.Int>(rank);
    final cStridesA = malloc<ffi.Int>(rank);
    final cStridesRes = malloc<ffi.Int>(rank);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }
    try {
      switch (a.dtype) {
        case DType.float64:
          s_asinh_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        case DType.float32:
          s_asinh_float(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        default:
          break;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesRes);
    }
  }

  unaryOp<T, double>(
    result.data,
    a.data,
    a.shape,
    a.strides,
    result.strides,
    0,
    a.offsetElements,
    result.offsetElements,
    (x) {
      final val = (x as num).toDouble();
      return math.log(val + math.sqrt(val * val + 1.0));
    },
  );
  return result;
}

/// Compute the element-wise inverse hyperbolic cosine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/hyperbolic_example.dart lang=dart}
NDArray<double> acosh<T extends num>(NDArray<T> a, {NDArray<double>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute acosh() on a disposed array.');
  }
  final DType<double> targetDType = a.dtype == DType.float32
      ? DType.float32 as DType<double>
      : DType.float64 as DType<double>;
  final result = out ?? NDArray<double>.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for acosh.',
      );
    }
  }

  if (a.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_acosh_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.float32:
        v_acosh_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      default:
        break;
    }
  } else {
    final rank = a.shape.length;
    final cShape = malloc<ffi.Int>(rank);
    final cStridesA = malloc<ffi.Int>(rank);
    final cStridesRes = malloc<ffi.Int>(rank);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }
    try {
      switch (a.dtype) {
        case DType.float64:
          s_acosh_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        case DType.float32:
          s_acosh_float(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        default:
          break;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesRes);
    }
  }

  unaryOp<T, double>(
    result.data,
    a.data,
    a.shape,
    a.strides,
    result.strides,
    0,
    a.offsetElements,
    result.offsetElements,
    (x) {
      final val = (x as num).toDouble();
      return math.log(val + math.sqrt(val * val - 1.0));
    },
  );
  return result;
}

/// Compute the element-wise inverse hyperbolic tangent of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/hyperbolic_example.dart lang=dart}
NDArray<R> atanh<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute atanh() on a disposed array.');
  }
  final DType<dynamic> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  }

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for atanh.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }

  if (a.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_atanh_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.float32:
        v_atanh_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.complex128:
        v_atanh_complex128(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      case DType.complex64:
        v_atanh_complex64(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      default:
        break;
    }
  } else {
    final rank = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }
    if (a.dtype == DType.float64) {
      s_atanh_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      s_atanh_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex128) {
      s_atanh_complex128(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex64) {
      s_atanh_complex64(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    }
  }

  unaryOp<T, R>(
    result.data,
    a.data,
    a.shape,
    a.strides,
    result.strides,
    0,
    a.offsetElements,
    result.offsetElements,
    (x) {
      final val = (x as num).toDouble();
      return 0.5 * math.log((1.0 + val) / (1.0 - val)) as R;
    },
  );
  return result;
}

/// Compute the element-wise arc tangent of [y] / [x] with full broadcasting support.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray atan2(NDArray y, NDArray x, {NDArray? out}) {
  if (y.isDisposed) {
    throw StateError('Cannot execute atan2() on a disposed array.');
  }
  if (y.dtype == DType.complex128 ||
      y.dtype == DType.complex64 ||
      x.dtype == DType.complex128 ||
      x.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for atan2');
  }
  final broadcastResult = broadcast(y, x);
  final shape = broadcastResult.shape;
  final DType<dynamic> targetDType =
      (y.dtype == DType.float32 && x.dtype == DType.float32)
      ? DType.float32
      : DType.float64;

  final NDArray result;
  if (out != null) {
    if (!listEquals(out.shape, shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for atan2.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(shape, targetDType);
  }

  // 0. Native C Vector Extension Fast-Path Gate for Contiguous Same-Shape arrays
  if (y.isContiguous && x.isContiguous && listEquals(y.shape, x.shape)) {
    if (y.dtype == DType.float64 && x.dtype == DType.float64) {
      v_atan2_double(
        y.pointer.cast(),
        x.pointer.cast(),
        result.pointer.cast(),
        y.data.length,
      );
      return result;
    } else if (y.dtype == DType.float32 && x.dtype == DType.float32) {
      v_atan2_float(
        y.pointer.cast(),
        x.pointer.cast(),
        result.pointer.cast(),
        y.data.length,
      );
      return result;
    }
  }

  final resultStrides = NDArray.computeCStrides(shape);
  final stridesY = broadcastResult.stridesA;
  final stridesX = broadcastResult.stridesB;

  // 0C. General Multidimensional Strided Broadcasting Engine in C (Rank <= 8)
  if (shape.length <= 8) {
    final cShape = malloc<ffi.Int>(shape.length);
    final cStridesY = malloc<ffi.Int>(stridesY.length);
    final cStridesX = malloc<ffi.Int>(stridesX.length);
    final cStridesRes = malloc<ffi.Int>(resultStrides.length);

    for (var i = 0; i < shape.length; i++) {
      cShape[i] = shape[i];
      cStridesY[i] = stridesY[i];
      cStridesX[i] = stridesX[i];
      cStridesRes[i] = resultStrides[i];
    }

    try {
      if (targetDType == DType.float64 &&
          y.dtype == DType.float64 &&
          x.dtype == DType.float64) {
        s_atan2_double(
          y.pointer.cast(),
          cStridesY,
          x.pointer.cast(),
          cStridesX,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          shape.length,
        );
        return result;
      } else if (targetDType == DType.float32 &&
          y.dtype == DType.float32 &&
          x.dtype == DType.float32) {
        s_atan2_float(
          y.pointer.cast(),
          cStridesY,
          x.pointer.cast(),
          cStridesX,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          shape.length,
        );
        return result;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesY);
      malloc.free(cStridesX);
      malloc.free(cStridesRes);
    }
  }

  final rData = result.data as List<double>;

  if (y.dtype == DType.float64 || y.dtype == DType.float32) {
    final yData = y.data as List<double>;
    if (x.dtype == DType.float64 || x.dtype == DType.float32) {
      elementWiseOp<double, double, double>(
        rData,
        yData,
        x.data as List<double>,
        shape,
        broadcastResult.stridesA,
        broadcastResult.stridesB,
        resultStrides,
        0,
        0,
        0,
        0,
        (a, b) => math.atan2(a, b),
      );
    } else {
      elementWiseOp<double, int, double>(
        rData,
        yData,
        x.data as List<int>,
        shape,
        broadcastResult.stridesA,
        broadcastResult.stridesB,
        resultStrides,
        0,
        0,
        0,
        0,
        (a, b) => math.atan2(a, b.toDouble()),
      );
    }
  } else {
    final yData = y.data as List<int>;
    if (x.dtype == DType.float64 || x.dtype == DType.float32) {
      elementWiseOp<int, double, double>(
        rData,
        yData,
        x.data as List<double>,
        shape,
        broadcastResult.stridesA,
        broadcastResult.stridesB,
        resultStrides,
        0,
        0,
        0,
        0,
        (a, b) => math.atan2(a.toDouble(), b),
      );
    } else {
      elementWiseOp<int, int, double>(
        rData,
        yData,
        x.data as List<int>,
        shape,
        broadcastResult.stridesA,
        broadcastResult.stridesB,
        resultStrides,
        0,
        0,
        0,
        0,
        (a, b) => math.atan2(a.toDouble(), b.toDouble()),
      );
    }
  }
  return result;
}

/// Compute the element-wise hypotenuse `sqrt(x1**2 + x2**2)` with broadcasting support.
///
/// **Example:**
/// ```dart
/// final h = hypot(a, b);
/// ```
NDArray<double> hypot(NDArray x1, NDArray x2, {NDArray<double>? out}) {
  if (x1.isDisposed) {
    throw StateError('Cannot execute hypot() on a disposed array.');
  }
  final broadcastResult = broadcast(x1, x2);
  final shape = broadcastResult.shape;
  final DType<double> targetDType =
      (x1.dtype == DType.complex64 || x2.dtype == DType.complex64)
      ? DType.float32 as DType<double>
      : DType.float64 as DType<double>;

  final NDArray<double> result;
  if (out != null) {
    if (!listEquals(out.shape, shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for hypot.',
      );
    }
    result = out;
  } else {
    result = NDArray<double>.create(shape, targetDType);
  }
  final resultStrides = NDArray.computeCStrides(shape);

  if (x1.dtype == DType.complex128 ||
      x2.dtype == DType.complex128 ||
      x1.dtype == DType.complex64 ||
      x2.dtype == DType.complex64) {
    final aCpx = (x1.dtype == DType.complex128 || x1.dtype == DType.complex64)
        ? x1
        : NDArray<Complex>.fromList(
            x1.data.map((e) => Complex((e as num).toDouble(), 0.0)).toList(),
            x1.shape,
            DType.complex128,
          );
    final bCpx = (x2.dtype == DType.complex128 || x2.dtype == DType.complex64)
        ? x2
        : NDArray<Complex>.fromList(
            x2.data.map((e) => Complex((e as num).toDouble(), 0.0)).toList(),
            x2.shape,
            DType.complex128,
          );
    if (listEquals(x1.shape, x2.shape) &&
        x1.isContiguous &&
        x2.isContiguous &&
        result.isContiguous) {
      if (aCpx.dtype == DType.complex128) {
        v_hypot_complex128(
          aCpx.pointer.cast(),
          bCpx.pointer.cast(),
          result.pointer.cast(),
          aCpx.data.length,
        );
        return result;
      } else {
        v_hypot_complex64(
          aCpx.pointer.cast(),
          bCpx.pointer.cast(),
          result.pointer.cast(),
          aCpx.data.length,
        );
        return result;
      }
    } else {
      final rank = shape.length;
      final cShape = malloc<ffi.Int>(rank);
      final cStridesA = malloc<ffi.Int>(rank);
      final cStridesB = malloc<ffi.Int>(rank);
      final cStridesRes = malloc<ffi.Int>(rank);
      for (var i = 0; i < rank; i++) {
        cShape[i] = shape[i];
        cStridesA[i] = broadcastResult.stridesA[i];
        cStridesB[i] = broadcastResult.stridesB[i];
        cStridesRes[i] = resultStrides[i];
      }
      try {
        if (aCpx.dtype == DType.complex128) {
          s_hypot_complex128(
            aCpx.pointer.cast(),
            cStridesA,
            bCpx.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        } else {
          s_hypot_complex64(
            aCpx.pointer.cast(),
            cStridesA,
            bCpx.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        }
      } finally {
        malloc.free(cShape);
        malloc.free(cStridesA);
        malloc.free(cStridesB);
        malloc.free(cStridesRes);
      }
    }
  }

  final rData = result.data;

  double hypotOp(double x, double y) {
    x = x.abs();
    y = y.abs();
    if (x < y) {
      final temp = x;
      x = y;
      y = temp;
    }
    if (x == 0) return 0.0;
    final t = y / x;
    return x * math.sqrt(1.0 + t * t);
  }

  elementWiseOp<num, num, double>(
    rData,
    x1.data as List<num>,
    x2.data as List<num>,
    shape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    0,
    0,
    0,
    0,
    (valA, valB) => hypotOp(valA.toDouble(), valB.toDouble()),
  );

  return result;
}

/// Compute the element-wise power `x1**x2` with broadcasting support.
///
/// **Example:**
/// ```dart
/// final p = power(a, b);
/// ```
NDArray power(NDArray x1, NDArray x2, {NDArray? out}) {
  if (x1.isDisposed) {
    throw StateError('Cannot execute power() on a disposed array.');
  }
  final broadcastResult = broadcast(x1, x2);
  final shape = broadcastResult.shape;
  final DType<dynamic> targetDType;
  if (x1.dtype == DType.complex128 ||
      x2.dtype == DType.complex128 ||
      x1.dtype == DType.complex64 ||
      x2.dtype == DType.complex64) {
    targetDType = (x1.dtype == DType.complex64 || x2.dtype == DType.complex64)
        ? DType.complex64
        : DType.complex128;
  } else {
    targetDType = (x1.dtype == DType.float32 || x2.dtype == DType.float32)
        ? DType.float32
        : DType.float64;
  }

  final NDArray result;
  if (out != null) {
    if (!listEquals(out.shape, shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for power.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(shape, targetDType);
  }
  final resultStrides = NDArray.computeCStrides(shape);

  if (targetDType == DType.complex128 || targetDType == DType.complex64) {
    final aCpx = (x1.dtype == DType.complex128 || x1.dtype == DType.complex64)
        ? x1
        : NDArray<Complex>.fromList(
            x1.data.map((e) => Complex((e as num).toDouble(), 0.0)).toList(),
            x1.shape,
            DType.complex128,
          );
    final bCpx = (x2.dtype == DType.complex128 || x2.dtype == DType.complex64)
        ? x2
        : NDArray<Complex>.fromList(
            x2.data.map((e) => Complex((e as num).toDouble(), 0.0)).toList(),
            x2.shape,
            DType.complex128,
          );
    if (listEquals(x1.shape, x2.shape) &&
        x1.isContiguous &&
        x2.isContiguous &&
        result.isContiguous) {
      if (aCpx.dtype == DType.complex128) {
        v_pow_complex128(
          aCpx.pointer.cast(),
          bCpx.pointer.cast(),
          result.pointer.cast(),
          aCpx.data.length,
        );
        return result;
      } else {
        v_pow_complex64(
          aCpx.pointer.cast(),
          bCpx.pointer.cast(),
          result.pointer.cast(),
          aCpx.data.length,
        );
        return result;
      }
    } else {
      final rank = shape.length;
      final cShape = malloc<ffi.Int>(rank);
      final cStridesA = malloc<ffi.Int>(rank);
      final cStridesB = malloc<ffi.Int>(rank);
      final cStridesRes = malloc<ffi.Int>(rank);
      for (var i = 0; i < rank; i++) {
        cShape[i] = shape[i];
        cStridesA[i] = broadcastResult.stridesA[i];
        cStridesB[i] = broadcastResult.stridesB[i];
        cStridesRes[i] = resultStrides[i];
      }
      try {
        if (aCpx.dtype == DType.complex128) {
          s_pow_complex128(
            aCpx.pointer.cast(),
            cStridesA,
            bCpx.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        } else {
          s_pow_complex64(
            aCpx.pointer.cast(),
            cStridesA,
            bCpx.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        }
      } finally {
        malloc.free(cShape);
        malloc.free(cStridesA);
        malloc.free(cStridesB);
        malloc.free(cStridesRes);
      }
    }
  }

  if (x1.isContiguous &&
      x2.isContiguous &&
      listEquals(x1.shape, x2.shape) &&
      result.isContiguous) {
    if (targetDType == DType.float64 &&
        x1.dtype == DType.float64 &&
        x2.dtype == DType.float64) {
      v_pow_double(
        x1.pointer.cast(),
        x2.pointer.cast(),
        result.pointer.cast(),
        x1.data.length,
      );
      return result;
    } else if (targetDType == DType.float32 &&
        x1.dtype == DType.float32 &&
        x2.dtype == DType.float32) {
      v_pow_float(
        x1.pointer.cast(),
        x2.pointer.cast(),
        result.pointer.cast(),
        x1.data.length,
      );
      return result;
    }
  } else if (shape.length <= 8) {
    final rank = shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesB = cBuffer + (rank * 2);
    final cStridesRes = cBuffer + (rank * 3);
    for (var i = 0; i < rank; i++) {
      cShape[i] = shape[i];
      cStridesA[i] = broadcastResult.stridesA[i];
      cStridesB[i] = broadcastResult.stridesB[i];
      cStridesRes[i] = resultStrides[i];
    }
    if (targetDType == DType.float64 &&
        x1.dtype == DType.float64 &&
        x2.dtype == DType.float64) {
      s_pow_double(
        x1.pointer.cast(),
        cStridesA,
        x2.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (targetDType == DType.float32 &&
        x1.dtype == DType.float32 &&
        x2.dtype == DType.float32) {
      s_pow_float(
        x1.pointer.cast(),
        cStridesA,
        x2.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    }
  }

  elementWiseOp<num, num, double>(
    result.data as List<double>,
    x1.data as List<num>,
    x2.data as List<num>,
    shape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    0,
    0,
    0,
    0,
    (valA, valB) => math.pow(valA.toDouble(), valB.toDouble()).toDouble(),
  );

  return result;
}

/// Numerical negative, element-wise.
///
/// **Example:**
/// ```dart
/// final b = negative(a);
/// ```
NDArray negative(NDArray a, {NDArray? out}) {
  final NDArray result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for negative.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, a.dtype);
  }
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    unaryOp<Complex, Complex>(
      result.data as List<Complex>,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => -x,
    );
  } else if (a.dtype == DType.float64 || a.dtype == DType.float32) {
    unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => -x,
    );
  } else if (a.dtype == DType.int64 || a.dtype == DType.int32) {
    unaryOp<int, int>(
      result.data as List<int>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => -x,
    );
  } else if (a.dtype == DType.boolean) {
    throw UnsupportedError('Boolean arrays do not support negative operator');
  }
  return result;
}

/// Element-wise floor division with broadcasting and dtype upcasting support.
///
/// Corresponds to Dart's `~/` operator.
///
/// **Division by Zero:**
/// - **Integer arrays**: Throws [UnsupportedError] if divisor contains any `0` elements.
///   This upfront safety check prevents a native C integer division by zero which would crash the entire Dart process.
/// - **Floating-point arrays**: Returns `double.nan` silently without throwing exceptions.
///
/// **Example:**
/// ```dart
/// final c = floor_divide(a, b);
/// ```
NDArray floor_divide(NDArray a, NDArray b, {NDArray? out}) {
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final DType<dynamic> targetDType = resolveDType(a.dtype, b.dtype);
  if (targetDType.isComplex) {
    throw UnsupportedError('Complex numbers do not support floor division');
  }

  final NDArray result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for floor_divide.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(commonShape, targetDType);
  }
  final resultStrides = NDArray.computeCStrides(commonShape);

  if (a.isContiguous &&
      b.isContiguous &&
      listEquals(a.shape, b.shape) &&
      result.isContiguous) {
    switch (targetDType) {
      case DType.float64:
        if (a.dtype == DType.float64 && b.dtype == DType.float64) {
          v_floordiv_double(
            a.pointer.cast(),
            b.pointer.cast(),
            result.pointer.cast(),
            a.data.length,
          );
          return result;
        }
      case DType.float32:
        if (a.dtype == DType.float32 && b.dtype == DType.float32) {
          v_floordiv_float(
            a.pointer.cast(),
            b.pointer.cast(),
            result.pointer.cast(),
            a.data.length,
          );
          return result;
        }
      case DType.int64:
        if (a.dtype == DType.int64 && b.dtype == DType.int64) {
          v_floordiv_int64(
            a.pointer.cast(),
            b.pointer.cast(),
            result.pointer.cast(),
            a.data.length,
          );
          final err = get_and_reset_division_error();
          if (err == 1) {
            throw UnsupportedError('Integer division by zero');
          }
          return result;
        }
      case DType.int32:
        if (a.dtype == DType.int32 && b.dtype == DType.int32) {
          v_floordiv_int32(
            a.pointer.cast(),
            b.pointer.cast(),
            result.pointer.cast(),
            a.data.length,
          );
          final err = get_and_reset_division_error();
          if (err == 1) {
            throw UnsupportedError('Integer division by zero');
          }
          return result;
        }
      default:
        break;
    }
  } else if (commonShape.length <= 8) {
    final rank = commonShape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesB = cBuffer + (rank * 2);
    final cStridesRes = cBuffer + (rank * 3);
    for (var i = 0; i < rank; i++) {
      cShape[i] = commonShape[i];
      cStridesA[i] = stridesA[i];
      cStridesB[i] = stridesB[i];
      cStridesRes[i] = resultStrides[i];
    }
    switch (targetDType) {
      case DType.float64:
        if (a.dtype == DType.float64 && b.dtype == DType.float64) {
          s_floordiv_double(
            a.pointer.cast(),
            cStridesA,
            b.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        }
      case DType.float32:
        if (a.dtype == DType.float32 && b.dtype == DType.float32) {
          s_floordiv_float(
            a.pointer.cast(),
            cStridesA,
            b.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        }
      case DType.int64:
        if (a.dtype == DType.int64 && b.dtype == DType.int64) {
          s_floordiv_int64(
            a.pointer.cast(),
            cStridesA,
            b.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          final err = get_and_reset_division_error();
          if (err == 1) {
            throw UnsupportedError('Integer division by zero');
          }
          return result;
        }
      case DType.int32:
        if (a.dtype == DType.int32 && b.dtype == DType.int32) {
          s_floordiv_int32(
            a.pointer.cast(),
            cStridesA,
            b.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          final err = get_and_reset_division_error();
          if (err == 1) {
            throw UnsupportedError('Integer division by zero');
          }
          return result;
        }
      default:
        break;
    }
  }

  double doubleFloorDiv(double x, double y) {
    if (y == 0.0) return double.nan;
    return (x / y).floorToDouble();
  }

  int intFloorDiv(int x, int y) {
    if (y == 0) {
      throw UnsupportedError('Integer division by zero');
    }
    final res = x ~/ y;
    final rem = x % y;
    if (rem != 0 && ((x < 0) ^ (y < 0))) {
      return res - 1;
    }
    return res;
  }

  if (targetDType == DType.float64 || targetDType == DType.float32) {
    elementWiseOp<double, double, double>(
      result.data as List<double>,
      a.data as List<double>,
      b.data as List<double>,
      commonShape,
      stridesA,
      stridesB,
      resultStrides,
      0,
      0,
      0,
      0,
      (x, y) => doubleFloorDiv(x, y),
    );
  } else {
    elementWiseOp<int, int, int>(
      result.data as List<int>,
      a.data as List<int>,
      b.data as List<int>,
      commonShape,
      stridesA,
      stridesB,
      resultStrides,
      0,
      0,
      0,
      0,
      (x, y) => intFloorDiv(x, y),
    );
  }
  return result;
}

/// Element-wise remainder of division with broadcasting and dtype upcasting support.
///
/// Corresponds to Dart's `%` operator.
///
/// **Division by Zero:**
/// - **Integer divisor**: Throws [UnsupportedError] if divisor contains any `0` elements.
///   This upfront safety check prevents a native C integer division by zero which would crash the entire Dart process.
/// - **Floating-point divisor**: Returns `double.nan` silently without throwing exceptions.
///
/// **Example:**
/// ```dart
/// final c = remainder(a, b);
/// ```
NDArray remainder(NDArray a, NDArray b, {NDArray? out}) {
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final DType<dynamic> targetDType = resolveDType(a.dtype, b.dtype);
  if (targetDType.isComplex) {
    throw UnsupportedError('Complex numbers do not support remainder');
  }

  final NDArray result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for remainder.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(commonShape, targetDType);
  }
  final resultStrides = NDArray.computeCStrides(commonShape);

  if (a.isContiguous &&
      b.isContiguous &&
      listEquals(a.shape, b.shape) &&
      result.isContiguous) {
    switch (targetDType) {
      case DType.float64:
        if (a.dtype == DType.float64 && b.dtype == DType.float64) {
          v_remainder_double(
            a.pointer.cast(),
            b.pointer.cast(),
            result.pointer.cast(),
            a.data.length,
          );
          return result;
        }
      case DType.float32:
        if (a.dtype == DType.float32 && b.dtype == DType.float32) {
          v_remainder_float(
            a.pointer.cast(),
            b.pointer.cast(),
            result.pointer.cast(),
            a.data.length,
          );
          return result;
        }
      case DType.int64:
        if (a.dtype == DType.int64 && b.dtype == DType.int64) {
          v_remainder_int64(
            a.pointer.cast(),
            b.pointer.cast(),
            result.pointer.cast(),
            a.data.length,
          );
          final err = get_and_reset_division_error();
          if (err == 1) {
            throw UnsupportedError('Integer division by zero');
          }
          return result;
        }
      case DType.int32:
        if (a.dtype == DType.int32 && b.dtype == DType.int32) {
          v_remainder_int32(
            a.pointer.cast(),
            b.pointer.cast(),
            result.pointer.cast(),
            a.data.length,
          );
          final err = get_and_reset_division_error();
          if (err == 1) {
            throw UnsupportedError('Integer division by zero');
          }
          return result;
        }
      default:
        break;
    }
  } else if (commonShape.length <= 8) {
    final rank = commonShape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesB = cBuffer + (rank * 2);
    final cStridesRes = cBuffer + (rank * 3);
    for (var i = 0; i < rank; i++) {
      cShape[i] = commonShape[i];
      cStridesA[i] = stridesA[i];
      cStridesB[i] = stridesB[i];
      cStridesRes[i] = resultStrides[i];
    }
    switch (targetDType) {
      case DType.float64:
        if (a.dtype == DType.float64 && b.dtype == DType.float64) {
          s_remainder_double(
            a.pointer.cast(),
            cStridesA,
            b.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        }
      case DType.float32:
        if (a.dtype == DType.float32 && b.dtype == DType.float32) {
          s_remainder_float(
            a.pointer.cast(),
            cStridesA,
            b.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        }
      case DType.int64:
        if (a.dtype == DType.int64 && b.dtype == DType.int64) {
          s_remainder_int64(
            a.pointer.cast(),
            cStridesA,
            b.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          final err = get_and_reset_division_error();
          if (err == 1) {
            throw UnsupportedError('Integer division by zero');
          }
          return result;
        }
      case DType.int32:
        if (a.dtype == DType.int32 && b.dtype == DType.int32) {
          s_remainder_int32(
            a.pointer.cast(),
            cStridesA,
            b.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          final err = get_and_reset_division_error();
          if (err == 1) {
            throw UnsupportedError('Integer division by zero');
          }
          return result;
        }
      default:
        break;
    }
  }

  double doubleMod(double x, double y) {
    if (y == 0.0) return double.nan;
    final rem = x % y;
    if (rem != 0.0 && ((rem < 0.0) != (y < 0.0))) {
      return rem + y;
    }
    return rem;
  }

  int intMod(int x, int y) {
    if (y == 0) {
      throw UnsupportedError('Integer division by zero');
    }
    final rem = x % y;
    if (rem != 0 && ((rem < 0) != (y < 0))) {
      return rem + y;
    }
    return rem;
  }

  if (targetDType == DType.float64 || targetDType == DType.float32) {
    elementWiseOp<double, double, double>(
      result.data as List<double>,
      a.data as List<double>,
      b.data as List<double>,
      commonShape,
      stridesA,
      stridesB,
      resultStrides,
      0,
      0,
      0,
      0,
      (x, y) => doubleMod(x, y),
    );
  } else {
    elementWiseOp<int, int, int>(
      result.data as List<int>,
      a.data as List<int>,
      b.data as List<int>,
      commonShape,
      stridesA,
      stridesB,
      resultStrides,
      0,
      0,
      0,
      0,
      (x, y) => intMod(x, y),
    );
  }
  return result;
}

/// Alias for [remainder].
NDArray mod(NDArray a, NDArray b, {NDArray? out}) => remainder(a, b, out: out);

/// Return element-wise quotient and remainder simultaneously.
///
/// Returns a Record `(NDArray, NDArray)` representing `(floor_divide(a, b), remainder(a, b))`.
///
/// **Example:**
/// ```dart
/// final res = divmod(a, b);
/// final q = res.$1;
/// final r = res.$2;
/// ```
(NDArray, NDArray) divmod(NDArray a, NDArray b) {
  return (floor_divide(a, b), remainder(a, b));
}

/// Compute the element-wise absolute value of the array.
///
/// For complex numbers, returns the magnitude as a real array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray abs(NDArray a, {NDArray? out}) {
  final DType<dynamic> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype == DType.complex64 ? DType.float32 : DType.float64;
  } else {
    targetDType = a.dtype;
  }

  final NDArray result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for abs.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType);
  }
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    unaryOp<Complex, double>(
      result.data as List<double>,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (c) => math.sqrt(c.real * c.real + c.imag * c.imag),
    );
    return result;
  }

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    unaryOp<int, int>(
      result.data as List<int>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.abs(),
    );
  } else {
    unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.abs(),
    );
  }
  return result;
}

/// Compute the element-wise sign of the array.
///
/// For real numbers, returns:
/// - -1 if x < 0
/// - 0 if x == 0
/// - 1 if x > 0
/// - nan if x is nan
///
/// For complex numbers, returns `x / |x|` (or 0 if x is 0).
///
/// **Example:**
/// ```dart
/// final s = sign(a);
/// ```
NDArray sign(NDArray a, {NDArray? out}) {
  final NDArray result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for sign.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, a.dtype);
  }
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    unaryOp<Complex, Complex>(
      result.data as List<Complex>,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (c) {
        if (c.real == 0 && c.imag == 0) return Complex(0, 0);
        final mag = math.sqrt(c.real * c.real + c.imag * c.imag);
        return Complex(c.real / mag, c.imag / mag);
      },
    );
    return result;
  }

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    unaryOp<int, int>(
      result.data as List<int>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.sign,
    );
  } else {
    unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.sign,
    );
  }
  return result;
}

/// Compute the element-wise ceiling of the array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray ceil(NDArray a, {NDArray? out}) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for ceil');
  }
  final NDArray result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for ceil.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, a.dtype);
  }

  if (a.isContiguous && result.isContiguous) {
    if (a.dtype == DType.float64) {
      v_ceil_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_ceil_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  }
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    unaryOp<int, int>(
      result.data as List<int>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x,
    );
  } else {
    unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.ceilToDouble(),
    );
  }
  return result;
}

/// Compute the element-wise floor of the array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray floor(NDArray a, {NDArray? out}) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for floor');
  }
  final NDArray result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for floor.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, a.dtype);
  }

  if (a.isContiguous && result.isContiguous) {
    if (a.dtype == DType.float64) {
      v_floor_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_floor_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  }
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    unaryOp<int, int>(
      result.data as List<int>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x,
    );
  } else {
    unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.floorToDouble(),
    );
  }
  return result;
}

/// Compute the element-wise rounding of the array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray round(NDArray a, {NDArray? out}) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for round');
  }
  final NDArray result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for round.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, a.dtype);
  }

  if (a.isContiguous && result.isContiguous) {
    if (a.dtype == DType.float64) {
      v_round_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_round_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  }
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    unaryOp<int, int>(
      result.data as List<int>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x,
    );
  } else {
    unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.roundToDouble(),
    );
  }
  return result;
}

/// Enumerates elements of a multidimensional array yielding coordinates and values.
///
/// Yields records containing the coordinate list and the element value at that coordinate
/// in standard C-contiguous order.
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
///
/// **Throws:**
/// - [StateError] if [a] has been disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.int32);
/// for (final entry in ndenumerate(a)) {
///   print('coord: ${entry.$1}, value: ${entry.$2}');
/// }
/// // Yields:
/// // ([0, 0], 10)
/// // ([0, 1], 20)
/// // ([1, 0], 30)
/// // ([1, 1], 40)
/// ```
Iterable<(List<int> coordinate, T value)> ndenumerate<T>(NDArray<T> a) sync* {
  if (a.isDisposed) {
    throw StateError('Cannot execute ndenumerate() on a disposed array.');
  }

  final shape = a.shape;
  final strides = a.strides;
  final totalSize = shape.isEmpty ? 1 : shape.reduce((x, y) => x * y);

  if (shape.isEmpty) {
    yield ([], a.data[0]);
    return;
  }

  final coord = List<int>.filled(shape.length, 0);
  int offset = 0;

  for (int el = 0; el < totalSize; el++) {
    // Yield a copy of the coordinate list so that users don't receive the same mutated buffer!
    yield (List<int>.from(coord), a.data[offset]);

    // Advance odometer multidimensional coordinate odometer walk!
    for (int d = shape.length - 1; d >= 0; d--) {
      coord[d]++;
      if (coord[d] < shape[d]) {
        offset += strides[d];
        break;
      }
      coord[d] = 0;
      offset -= (shape[d] - 1) * strides[d];
    }
  }
}

/// Returns the real part of a complex array element-wise.
///
/// If the input array [a] is already real (integer or float), returns a zero-copy
/// view of the array [a].
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
/// - If provided, the output recycler [out] must match the expected target shape and float `DType.`
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [ArgumentError] if [out] is provided but has a shape or DType mismatch.
///
/// **Example:**
/// ```dart
/// final a = `NDArray<Complex>`.create([2], `DType.complex128);`
/// a.data[0] = Complex(3.0, 4.0);
/// a.data[1] = Complex(-1.0, 0.0);
/// final r = real(a); // [3.0, -1.0] (`DType.float64)`
/// ```
NDArray<R> real<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute real() on a disposed array.');
  }

  final DType<dynamic> targetDType;
  if (a.dtype == DType.complex64) {
    targetDType = DType.float32;
  } else if (a.dtype == DType.complex128) {
    targetDType = DType.float64;
  } else {
    targetDType = a.dtype;
  }

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for real.',
      );
    }
    result = out;
  } else {
    if (a.dtype != DType.complex128 && a.dtype != DType.complex64) {
      return NDArray.view(a, shape: a.shape, strides: a.strides)
          as NDArray<R>; // Zero-copy view for already real arrays!
    }
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    final resultStrides = NDArray.computeCStrides(a.shape);
    unaryOp<Complex, R>(
      result.data,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.real as R,
    );
    return result;
  } else {
    // This path is taken if out != null and a is not complex.
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    result.data.setRange(0, size, a.toList() as List<R>);
    return result;
  }
}

/// Returns the imaginary part of a complex array element-wise.
///
/// If the input array [a] is already real, returns a zero-filled array of matching shape
/// and target float `DType.`
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
/// - If provided, the output recycler [out] must match the expected target shape and float `DType.`
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [ArgumentError] if [out] is provided but has a shape or DType mismatch.
///
/// **Example:**
/// ```dart
/// final a = `NDArray<Complex>`.create([2], `DType.complex128);`
/// a.data[0] = Complex(3.0, 4.0);
/// a.data[1] = Complex(-1.0, 0.0);
/// final im = imag(a); // [4.0, 0.0] (`DType.float64)`
/// ```
NDArray<R> imag<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute imag() on a disposed array.');
  }

  final DType<dynamic> targetDType = a.dtype == DType.complex64
      ? DType.float32
      : DType.float64;

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for imag.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }

  if (a.dtype != DType.complex128 && a.dtype != DType.complex64) {
    if (out != null) {
      result.data.fillRange(0, result.data.length, 0.0 as R);
      return result;
    }
    return NDArray.zeros(a.shape, targetDType) as NDArray<R>;
  }

  final resultStrides = NDArray.computeCStrides(a.shape);
  unaryOp<Complex, R>(
    result.data,
    a.data as List<Complex>,
    a.shape,
    a.strides,
    resultStrides,
    0,
    0,
    0,
    (x) => x.imag as R,
  );

  return result;
}

/// Converts angles from degrees to radians element-wise.
///
/// **Preconditions:**
/// - Input array [a] must not be disposed.
/// - Input array [a] must not contain complex numbers.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [UnsupportedError] if the array has a complex data type.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([180.0, 90.0, 45.0], [3], DType.float64);
/// final r = deg2rad(a); // [pi, pi / 2.0, pi / 4.0]
/// ```
NDArray<R> deg2rad<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute deg2rad on a disposed array.');
  }
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for deg2rad');
  }

  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;

  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for deg2rad.',
      );
    }
  }

  final factor = NDArray.fromList([0.017453292519943295], [1], targetDType);
  return multiply<T, dynamic, R>(a, factor, out: out);
}

/// Converts angles from radians to degrees element-wise.
///
/// **Preconditions:**
/// - Input array [a] must not be disposed.
/// - Input array [a] must not contain complex numbers.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [UnsupportedError] if the array has a complex data type.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([math.pi, math.pi / 2.0], [2], DType.float64);
/// final d = rad2deg(a); // [180.0, 90.0]
/// ```
NDArray<R> rad2deg<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute rad2deg on a disposed array.');
  }
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for rad2deg');
  }

  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;

  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for rad2deg.',
      );
    }
  }

  final factor = NDArray.fromList([57.29577951308232], [1], targetDType);
  return multiply<T, dynamic, R>(a, factor, out: out);
}

/// Returns an element-wise boolean mask indicating which elements of the array are NaN (Not-a-Number).
///
/// **Preconditions:**
/// - The array must not be disposed.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.nan, 3.0], [3], DType.float64);
/// final mask = isnan(a); // [false, true, false]
/// ```
NDArray<bool> isnan(NDArray a, {NDArray<bool>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute isnan on a disposed array.');
  }
  final NDArray<bool> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for isnan.',
      );
    }
    result = out;
  } else {
    result = NDArray<bool>.create(a.shape, DType.boolean);
  }
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_isnan_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.float32:
        v_isnan_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.complex128:
        v_isnan_complex128(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      case DType.complex64:
        v_isnan_complex64(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      case DType.int32:
      case DType.int64:
        result.fill(false);
        return result;
      default:
        break;
    }
  } else {
    final rank = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = resultStrides[i];
    }
    switch (a.dtype) {
      case DType.float64:
        s_isnan_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.float32:
        s_isnan_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.complex128:
        s_isnan_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.complex64:
        s_isnan_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.int32:
      case DType.int64:
        result.fill(false);
        return result;
      default:
        break;
    }
  }

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    unaryOp<Complex, bool>(
      result.data,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.real.isNaN || x.imag.isNaN,
    );
  } else if (a.dtype.isInteger) {
    result.fill(false);
  } else {
    unaryOp<double, bool>(
      result.data,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.isNaN,
    );
  }
  return result;
}

/// Returns an element-wise boolean mask indicating which elements of the array are positive or negative infinity.
///
/// **Preconditions:**
/// - The array must not be disposed.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.infinity, 3.0], [3], DType.float64);
/// final mask = isinf(a); // [false, true, false]
/// ```
NDArray<bool> isinf(NDArray a, {NDArray<bool>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute isinf on a disposed array.');
  }
  final NDArray<bool> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for isinf.',
      );
    }
    result = out;
  } else {
    result = NDArray<bool>.create(a.shape, DType.boolean);
  }
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_isinf_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.float32:
        v_isinf_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
        return result;
      case DType.complex128:
        v_isinf_complex128(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      case DType.complex64:
        v_isinf_complex64(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      case DType.int32:
      case DType.int64:
        result.fill(false);
        return result;
      default:
        break;
    }
  } else {
    final rank = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = resultStrides[i];
    }
    switch (a.dtype) {
      case DType.float64:
        s_isinf_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.float32:
        s_isinf_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.complex128:
        s_isinf_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.complex64:
        s_isinf_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.int32:
      case DType.int64:
        result.fill(false);
        return result;
      default:
        break;
    }
  }

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    unaryOp<Complex, bool>(
      result.data,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.real.isInfinite || x.imag.isInfinite,
    );
  } else if (a.dtype.isInteger) {
    result.fill(false);
  } else {
    unaryOp<double, bool>(
      result.data,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.isInfinite,
    );
  }
  return result;
}

/// Returns an element-wise boolean mask indicating which elements of the array are finite (neither NaN nor infinite).
///
/// **Preconditions:**
/// - The array must not be disposed.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.nan, double.infinity], [3], DType.float64);
/// final mask = isfinite(a); // [true, false, false]
/// ```
NDArray<bool> isfinite(NDArray a, {NDArray<bool>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute isfinite on a disposed array.');
  }
  final NDArray<bool> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for isfinite.',
      );
    }
    result = out;
  } else {
    result = NDArray<bool>.create(a.shape, DType.boolean);
  }
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_isfinite_double(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      case DType.float32:
        v_isfinite_float(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      case DType.complex128:
        v_isfinite_complex128(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      case DType.complex64:
        v_isfinite_complex64(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      case DType.int32:
      case DType.int64:
        result.fill(true);
        return result;
      default:
        break;
    }
  } else {
    final rank = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = resultStrides[i];
    }
    switch (a.dtype) {
      case DType.float64:
        s_isfinite_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.float32:
        s_isfinite_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.complex128:
        s_isfinite_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.complex64:
        s_isfinite_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.int32:
      case DType.int64:
        result.fill(true);
        return result;
      default:
        break;
    }
  }

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    unaryOp<Complex, bool>(
      result.data,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.real.isFinite && x.imag.isFinite,
    );
  } else if (a.dtype.isInteger) {
    result.fill(true);
  } else {
    unaryOp<double, bool>(
      result.data,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.isFinite,
    );
  }
  return result;
}

/// Return first element-wise argument with the sign of the second element-wise argument.
///
/// **Throws:**
/// - [StateError] if either array has been disposed.
/// - [UnsupportedError] if either array is complex (copysign is not defined for complex numbers).
///
/// **Example:**
/// ```dart
/// final res = copysign(x1, x2);
/// ```
NDArray copysign(NDArray x1, NDArray x2, {NDArray? out}) {
  if (x1.isDisposed || x2.isDisposed) {
    throw StateError('Cannot execute copysign on a disposed array.');
  }
  if (x1.dtype.isComplex || x2.dtype.isComplex) {
    throw UnsupportedError('Complex numbers are not supported for copysign');
  }

  final broadcastResult = broadcast(x1, x2);
  final shape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final DType<dynamic> targetDType =
      (x1.dtype == DType.float32 && x2.dtype == DType.float32)
      ? DType.float32
      : DType.float64;

  final NDArray result;
  if (out != null) {
    if (!listEquals(out.shape, shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for copysign.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(shape, targetDType);
  }
  final resultStrides = NDArray.computeCStrides(shape);

  if (x1.isContiguous &&
      x2.isContiguous &&
      listEquals(x1.shape, x2.shape) &&
      result.isContiguous) {
    if (targetDType == DType.float64 &&
        x1.dtype == DType.float64 &&
        x2.dtype == DType.float64) {
      v_copysign_double(
        x1.pointer.cast(),
        x2.pointer.cast(),
        result.pointer.cast(),
        x1.data.length,
      );
      return result;
    } else if (targetDType == DType.float32 &&
        x1.dtype == DType.float32 &&
        x2.dtype == DType.float32) {
      v_copysign_float(
        x1.pointer.cast(),
        x2.pointer.cast(),
        result.pointer.cast(),
        x1.data.length,
      );
      return result;
    }
  } else if (shape.length <= 8) {
    final rank = shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesB = cBuffer + (rank * 2);
    final cStridesRes = cBuffer + (rank * 3);
    for (var i = 0; i < rank; i++) {
      cShape[i] = shape[i];
      cStridesA[i] = stridesA[i];
      cStridesB[i] = stridesB[i];
      cStridesRes[i] = resultStrides[i];
    }
    if (targetDType == DType.float64 &&
        x1.dtype == DType.float64 &&
        x2.dtype == DType.float64) {
      s_copysign_double(
        x1.pointer.cast(),
        cStridesA,
        x2.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (targetDType == DType.float32 &&
        x1.dtype == DType.float32 &&
        x2.dtype == DType.float32) {
      s_copysign_float(
        x1.pointer.cast(),
        cStridesA,
        x2.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    }
  }

  double copysignOp(double a, double b) {
    if (b == 0.0) {
      return b.isNegative ? -a.abs() : a.abs();
    }
    return b < 0.0 ? -a.abs() : a.abs();
  }

  elementWiseOp<num, num, double>(
    result.data as List<double>,
    x1.data as List<num>,
    x2.data as List<num>,
    shape,
    stridesA,
    stridesB,
    resultStrides,
    0,
    0,
    0,
    0,
    (x, y) => copysignOp(x.toDouble(), y.toDouble()),
  );

  return result;
}

/// Clip (limit) the values in an array using scalar bounds.
///
/// Given an interval `[min, max]`, values outside the interval are clipped
/// to the interval edges.
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
/// - [min] must be less than or equal to [max].
/// - If provided, [out] must have the exact shape and matching [DType] of [a].
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [UnsupportedError] if [a] has a complex [DType] (complex values cannot be ordered).
/// - [ArgumentError] if [out] has an incompatible shape or [DType].
///
/// **Performance considerations:**
/// - Time complexity is $O(N)$ where $N$ is the total number of elements in [a].
/// - For contiguous arrays, offloads directly to C optimized kernels via FFI, executing in $O(N)$ time with $O(1)$ extra memory.
/// - Otherwise, performs element-wise strided iteration in Dart.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
///
/// Reference: [NumPy clip](https://numpy.org/doc/stable/reference/generated/numpy.clip.html)
NDArray<T> clip<T>(NDArray<T> a, {num? min, num? max, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute clip on a disposed array.');
  }
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for clip');
  }
  final result = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for clip.',
      );
    }
    if (out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible DType for clip.',
      );
    }
  }

  final resolvedMin = min ?? _getMinLimit(a.dtype);
  final resolvedMax = max ?? _getMaxLimit(a.dtype);

  final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);

  if (a.isContiguous && result.isContiguous) {
    if (a.dtype == DType.float64) {
      v_clip_double(
        a.pointer.cast(),
        result.pointer.cast(),
        resolvedMin.toDouble(),
        resolvedMax.toDouble(),
        size,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      v_clip_float(
        a.pointer.cast(),
        result.pointer.cast(),
        resolvedMin.toDouble(),
        resolvedMax.toDouble(),
        size,
      );
      return result;
    }
  }

  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype.isInteger) {
    final mn = resolvedMin.toInt();
    final mx = resolvedMax.toInt();
    unaryOp<int, int>(
      result.data as List<int>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => x.clamp(mn, mx),
    );
  } else {
    final mn = resolvedMin.toDouble();
    final mx = resolvedMax.toDouble();
    unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => x.clamp(mn, mx),
    );
  }
  return result;
}

/// Clip (limit) the values in an array using array bounds that broadcast natively against the input array.
///
/// Given array bounds [min] and [max], values outside the interval are clipped
/// to the interval edges.
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
/// - [min] and [max] must not be disposed.
/// - [min] and [max] must be of real/integer numeric types (complex/boolean bounds are not supported).
/// - The shapes of [a], [min], and [max] must be compatible for broadcasting.
/// - If provided, [out] must have the exact broadcasted shape and matching [DType] of [a].
///
/// **Throws:**
/// - [StateError] if [a], [min], or [max] is disposed.
/// - [UnsupportedError] if [a] has a complex [DType] (complex values cannot be ordered).
/// - [ArgumentError] if [min] or [max] is complex or boolean.
/// - [ArgumentError] if shapes are incompatible for broadcasting, or if [out] has an incompatible shape or [DType].
///
/// **Performance considerations:**
/// - Time complexity is $O(N)$ where $N$ is the total number of elements in the broadcasted shape.
/// - Performs element-wise strided iteration in Dart using a ternary walker, requiring zero heap allocations for view creation.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 5.0, 10.0], [3], DType.float64);
/// final minBounds = NDArray.fromList([2.0, 2.0, 2.0], [3], DType.float64);
/// final maxBounds = NDArray.fromList([8.0, 8.0, 8.0], [3], DType.float64);
/// final clipped = clipArray(a, min: minBounds, max: maxBounds); // [2.0, 5.0, 8.0]
/// ```
///
/// Reference: [NumPy clip](https://numpy.org/doc/stable/reference/generated/numpy.clip.html)
NDArray<T> clipArray<T>(
  NDArray<T> a, {
  NDArray<T>? min,
  NDArray<T>? max,
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute clipArray on a disposed array.');
  }
  if (min != null && min.isDisposed) {
    throw StateError(
      'Cannot execute clipArray with a disposed min bounds array.',
    );
  }
  if (max != null && max.isDisposed) {
    throw StateError(
      'Cannot execute clipArray with a disposed max bounds array.',
    );
  }
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for clipArray');
  }
  if (min != null && (min.dtype.isComplex || min.dtype == DType.boolean)) {
    throw ArgumentError(
      'Complex/Boolean bounds are not supported for clipArray',
    );
  }
  if (max != null && (max.dtype.isComplex || max.dtype == DType.boolean)) {
    throw ArgumentError(
      'Complex/Boolean bounds are not supported for clipArray',
    );
  }

  final bool ownsMin = min == null;
  final minArr =
      min ??
      (NDArray<T>.create([], a.dtype)..data[0] = _getMinLimit(a.dtype) as T);

  final bool ownsMax = max == null;
  final maxArr =
      max ??
      (NDArray<T>.create([], a.dtype)..data[0] = _getMaxLimit(a.dtype) as T);

  try {
    NDArray? dummy;
    List<int> commonShape;
    try {
      final b1 = broadcast(a, minArr);
      dummy = NDArray.create(b1.shape, a.dtype);
      final b2 = broadcast(dummy, maxArr);
      commonShape = b2.shape;
    } finally {
      dummy?.dispose();
    }

    final result = out ?? NDArray<T>.create(commonShape, a.dtype);
    if (out != null) {
      if (!listEquals(out.shape, commonShape)) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape for clipArray.',
        );
      }
      if (out.dtype != a.dtype) {
        throw ArgumentError(
          'Provided out buffer has incompatible DType for clipArray.',
        );
      }
    }

    final broadcastA = broadcastTo(a, commonShape);
    final broadcastMin = broadcastTo(minArr, commonShape);
    final broadcastMax = broadcastTo(maxArr, commonShape);
    final resultStrides = NDArray.computeCStrides(commonShape);

    final marker = ScratchArena.marker;
    try {
      final ndim = commonShape.length;
      final cBuffer = ScratchArena.getStridedBuffer(ndim, 5);
      final cShape = cBuffer;
      final cStridesA = cBuffer + ndim;
      final cStridesMin = cBuffer + (ndim * 2);
      final cStridesMax = cBuffer + (ndim * 3);
      final cStridesRes = cBuffer + (ndim * 4);

      for (var i = 0; i < ndim; i++) {
        cShape[i] = commonShape[i];
        cStridesA[i] = broadcastA.strides[i];
        cStridesMin[i] = broadcastMin.strides[i];
        cStridesMax[i] = broadcastMax.strides[i];
        cStridesRes[i] = resultStrides[i];
      }

      switch (a.dtype) {
        case DType.float64:
          s_clip_double(
            (broadcastA.pointer.cast<ffi.Double>() + broadcastA.offsetElements)
                .cast(),
            cStridesA,
            (broadcastMin.pointer.cast<ffi.Double>() +
                    broadcastMin.offsetElements)
                .cast(),
            cStridesMin,
            (broadcastMax.pointer.cast<ffi.Double>() +
                    broadcastMax.offsetElements)
                .cast(),
            cStridesMax,
            (result.pointer.cast<ffi.Double>() + result.offsetElements).cast(),
            cStridesRes,
            cShape,
            ndim,
          );
          return result;
        case DType.float32:
          s_clip_float(
            (broadcastA.pointer.cast<ffi.Float>() + broadcastA.offsetElements)
                .cast(),
            cStridesA,
            (broadcastMin.pointer.cast<ffi.Float>() +
                    broadcastMin.offsetElements)
                .cast(),
            cStridesMin,
            (broadcastMax.pointer.cast<ffi.Float>() +
                    broadcastMax.offsetElements)
                .cast(),
            cStridesMax,
            (result.pointer.cast<ffi.Float>() + result.offsetElements).cast(),
            cStridesRes,
            cShape,
            ndim,
          );
          return result;
        case DType.int64:
          s_clip_int64(
            (broadcastA.pointer.cast<ffi.Int64>() + broadcastA.offsetElements)
                .cast(),
            cStridesA,
            (broadcastMin.pointer.cast<ffi.Int64>() +
                    broadcastMin.offsetElements)
                .cast(),
            cStridesMin,
            (broadcastMax.pointer.cast<ffi.Int64>() +
                    broadcastMax.offsetElements)
                .cast(),
            cStridesMax,
            (result.pointer.cast<ffi.Int64>() + result.offsetElements).cast(),
            cStridesRes,
            cShape,
            ndim,
          );
          return result;
        case DType.int32:
          s_clip_int32(
            (broadcastA.pointer.cast<ffi.Int32>() + broadcastA.offsetElements)
                .cast(),
            cStridesA,
            (broadcastMin.pointer.cast<ffi.Int32>() +
                    broadcastMin.offsetElements)
                .cast(),
            cStridesMin,
            (broadcastMax.pointer.cast<ffi.Int32>() +
                    broadcastMax.offsetElements)
                .cast(),
            cStridesMax,
            (result.pointer.cast<ffi.Int32>() + result.offsetElements).cast(),
            cStridesRes,
            cShape,
            ndim,
          );
          return result;
        case DType.uint8:
          s_clip_uint8(
            (broadcastA.pointer.cast<ffi.Uint8>() + broadcastA.offsetElements)
                .cast(),
            cStridesA,
            (broadcastMin.pointer.cast<ffi.Uint8>() +
                    broadcastMin.offsetElements)
                .cast(),
            cStridesMin,
            (broadcastMax.pointer.cast<ffi.Uint8>() +
                    broadcastMax.offsetElements)
                .cast(),
            cStridesMax,
            (result.pointer.cast<ffi.Uint8>() + result.offsetElements).cast(),
            cStridesRes,
            cShape,
            ndim,
          );
          return result;
        case DType.int16:
          s_clip_int16(
            (broadcastA.pointer.cast<ffi.Int16>() + broadcastA.offsetElements)
                .cast(),
            cStridesA,
            (broadcastMin.pointer.cast<ffi.Int16>() +
                    broadcastMin.offsetElements)
                .cast(),
            cStridesMin,
            (broadcastMax.pointer.cast<ffi.Int16>() +
                    broadcastMax.offsetElements)
                .cast(),
            cStridesMax,
            (result.pointer.cast<ffi.Int16>() + result.offsetElements).cast(),
            cStridesRes,
            cShape,
            ndim,
          );
          return result;
        default:
          break;
      }
    } finally {
      ScratchArena.reset(marker);
    }

    if (a.dtype.isInteger) {
      ternaryOp<int, int, int, int>(
        result.data as List<int>,
        broadcastA.data as List<int>,
        broadcastMin.data as List<int>,
        broadcastMax.data as List<int>,
        commonShape,
        broadcastA.strides,
        broadcastMin.strides,
        broadcastMax.strides,
        resultStrides,
        0,
        broadcastA.offsetElements,
        broadcastMin.offsetElements,
        broadcastMax.offsetElements,
        result.offsetElements,
        (x, mn, mx) => x.clamp(mn, mx),
      );
    } else {
      ternaryOp<double, double, double, double>(
        result.data as List<double>,
        broadcastA.data as List<double>,
        broadcastMin.data as List<double>,
        broadcastMax.data as List<double>,
        commonShape,
        broadcastA.strides,
        broadcastMin.strides,
        broadcastMax.strides,
        resultStrides,
        0,
        broadcastA.offsetElements,
        broadcastMin.offsetElements,
        broadcastMax.offsetElements,
        result.offsetElements,
        (x, mn, mx) => x.clamp(mn, mx),
      );
    }

    return result;
  } finally {
    if (ownsMin) minArr.dispose();
    if (ownsMax) maxArr.dispose();
  }
}

/// Compute the element-wise truth value of NOT [a].
///
/// Returns a boolean array of the same shape as [a] containing the element-wise logical negation.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - If provided, the [out] recycler array must exactly match [a]'s shape and have [DType.boolean] dtype.
///
/// **Throws:**
/// - [StateError] if the input array [a] has been disposed.
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous layout layouts, offloads the loop directly to high-speed native C
///   vector logical kernels (`v_logical_not`), bypassing all Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
///
/// Reference: [NumPy logical_not](https://numpy.org/doc/stable/reference/generated/numpy.logical_not.html)
NDArray<bool> logical_not(NDArray a, {NDArray<bool>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute logical_not on a disposed array.');
  }
  final NDArray<bool> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for logical_not.',
      );
    }
    result = out;
  } else {
    result = NDArray<bool>.create(a.shape, DType.boolean);
  }

  final resultStrides = NDArray.computeCStrides(a.shape);
  final marker = ScratchArena.marker;

  final ffi.Pointer<ffi.Uint8> aBoolPtr;
  final List<int> aBoolStrides;
  if (a.dtype == DType.boolean) {
    aBoolPtr = a.pointer.cast();
    aBoolStrides = a.strides;
  } else {
    aBoolPtr = ScratchArena.allocate<ffi.Uint8>(a.size);
    aBoolStrides = NDArray.computeCStrides(a.shape);
    if (a.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_to_bool_double(a.pointer.cast(), aBoolPtr, a.size);
        case DType.float32:
          v_to_bool_float(a.pointer.cast(), aBoolPtr, a.size);
        case DType.int64:
          v_to_bool_int64(a.pointer.cast(), aBoolPtr, a.size);
        case DType.int32:
          v_to_bool_int32(a.pointer.cast(), aBoolPtr, a.size);
        case DType.uint8:
          v_to_bool_uint8(a.pointer.cast(), aBoolPtr, a.size);
        case DType.int16:
          v_to_bool_int16(a.pointer.cast(), aBoolPtr, a.size);
        case DType.complex128:
          v_to_bool_complex128(a.pointer.cast(), aBoolPtr, a.size);
        case DType.complex64:
          v_to_bool_complex64(a.pointer.cast(), aBoolPtr, a.size);
        case DType.boolean:
          break;
      }
    } else {
      final ndim = a.shape.length;
      final cBuffer = ScratchArena.getStridedBuffer(ndim);
      final cShape = cBuffer;
      final cStridesA = cBuffer + ndim;
      final cStridesTemp = cBuffer + (ndim * 2);
      for (var i = 0; i < ndim; i++) {
        cShape[i] = a.shape[i];
        cStridesA[i] = a.strides[i];
        cStridesTemp[i] = aBoolStrides[i];
      }
      switch (a.dtype) {
        case DType.float64:
          s_to_bool_double(
            a.pointer.cast(),
            cStridesA,
            aBoolPtr,
            cStridesTemp,
            cShape,
            ndim,
          );
        case DType.float32:
          s_to_bool_float(
            a.pointer.cast(),
            cStridesA,
            aBoolPtr,
            cStridesTemp,
            cShape,
            ndim,
          );
        case DType.int64:
          s_to_bool_int64(
            a.pointer.cast(),
            cStridesA,
            aBoolPtr,
            cStridesTemp,
            cShape,
            ndim,
          );
        case DType.int32:
          s_to_bool_int32(
            a.pointer.cast(),
            cStridesA,
            aBoolPtr,
            cStridesTemp,
            cShape,
            ndim,
          );
        case DType.uint8:
          s_to_bool_uint8(
            a.pointer.cast(),
            cStridesA,
            aBoolPtr,
            cStridesTemp,
            cShape,
            ndim,
          );
        case DType.int16:
          s_to_bool_int16(
            a.pointer.cast(),
            cStridesA,
            aBoolPtr,
            cStridesTemp,
            cShape,
            ndim,
          );
        case DType.complex128:
          s_to_bool_complex128(
            a.pointer.cast(),
            cStridesA,
            aBoolPtr,
            cStridesTemp,
            cShape,
            ndim,
          );
        case DType.complex64:
          s_to_bool_complex64(
            a.pointer.cast(),
            cStridesA,
            aBoolPtr,
            cStridesTemp,
            cShape,
            ndim,
          );
        case DType.boolean:
          break;
      }
    }
  }

  if (a.isContiguous && result.isContiguous) {
    v_logical_not(aBoolPtr, result.pointer.cast(), a.size);
  } else {
    final ndim = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(ndim);
    final cShape = cBuffer;
    final cStridesA = cBuffer + ndim;
    final cStridesRes = cBuffer + (ndim * 2);

    for (var i = 0; i < ndim; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = aBoolStrides[i];
      cStridesRes[i] = resultStrides[i];
    }

    s_logical_not(
      aBoolPtr,
      cStridesA,
      result.pointer.cast(),
      cStridesRes,
      cShape,
      ndim,
    );
  }

  ScratchArena.reset(marker);
  return result;
}

/// Element-wise comparison of [a] == [b] with broadcasting and recycling support.
NDArray<bool> equal(NDArray a, NDArray b, {NDArray<bool>? out}) {
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final result = out ?? NDArray<bool>.create(commonShape, DType.boolean);
  final resultStrides = NDArray.computeCStrides(commonShape);

  a.dispatchCompare(
    result.data,
    a,
    b,
    commonShape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    (x, y) => x == y,
  );
  return result;
}

/// Element-wise comparison of [a] != [b] with broadcasting and recycling support.
NDArray<bool> not_equal(NDArray a, NDArray b, {NDArray<bool>? out}) {
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final result = out ?? NDArray<bool>.create(commonShape, DType.boolean);
  final resultStrides = NDArray.computeCStrides(commonShape);

  a.dispatchCompare(
    result.data,
    a,
    b,
    commonShape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    (x, y) => x != y,
  );
  return result;
}

/// Element-wise comparison of [a] > [b] with broadcasting and recycling support.
NDArray<bool> greater(NDArray a, NDArray b, {NDArray<bool>? out}) {
  if (a.dtype.isComplex || b.dtype.isComplex) {
    throw UnsupportedError(
      'Complex numbers do not support inequality comparisons',
    );
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final result = out ?? NDArray<bool>.create(commonShape, DType.boolean);
  final resultStrides = NDArray.computeCStrides(commonShape);

  a.dispatchCompare(
    result.data,
    a,
    b,
    commonShape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    (x, y) => (x as num) > (y as num),
  );
  return result;
}

/// Element-wise comparison of [a] >= [b] with broadcasting and recycling support.
NDArray<bool> greater_equal(NDArray a, NDArray b, {NDArray<bool>? out}) {
  if (a.dtype.isComplex || b.dtype.isComplex) {
    throw UnsupportedError(
      'Complex numbers do not support inequality comparisons',
    );
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final result = out ?? NDArray<bool>.create(commonShape, DType.boolean);
  final resultStrides = NDArray.computeCStrides(commonShape);

  a.dispatchCompare(
    result.data,
    a,
    b,
    commonShape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    (x, y) => (x as num) >= (y as num),
  );
  return result;
}

/// Element-wise comparison of [a] < [b] with broadcasting and recycling support.
NDArray<bool> less(NDArray a, NDArray b, {NDArray<bool>? out}) {
  if (a.dtype.isComplex || b.dtype.isComplex) {
    throw UnsupportedError(
      'Complex numbers do not support inequality comparisons',
    );
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final result = out ?? NDArray<bool>.create(commonShape, DType.boolean);
  final resultStrides = NDArray.computeCStrides(commonShape);

  a.dispatchCompare(
    result.data,
    a,
    b,
    commonShape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    (x, y) => (x as num) < (y as num),
  );
  return result;
}

/// Element-wise comparison of [a] <= [b] with broadcasting and recycling support.
NDArray<bool> less_equal(NDArray a, NDArray b, {NDArray<bool>? out}) {
  if (a.dtype.isComplex || b.dtype.isComplex) {
    throw UnsupportedError(
      'Complex numbers do not support inequality comparisons',
    );
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final result = out ?? NDArray<bool>.create(commonShape, DType.boolean);
  final resultStrides = NDArray.computeCStrides(commonShape);

  a.dispatchCompare(
    result.data,
    a,
    b,
    commonShape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    (x, y) => (x as num) <= (y as num),
  );
  return result;
}

/// Compute the element-wise truth value of [a] AND [b] with broadcasting support.
///
/// Returns a boolean array containing the element-wise logical AND results.
///
/// **Preconditions:**
/// - [a] and [b] must not be disposed.
/// - If provided, the [out] recycler array must exactly match the broadcasted shape of [a] and [b], and have [DType.boolean] dtype.
///
/// **Throws:**
/// - [StateError] if any input array has been disposed.
/// - [ArgumentError] if shapes of [a] and [b] are incompatible for broadcasting.
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous layout layouts, offloads the loop directly to high-speed native C
///   vector logical kernels (`v_logical_and`), bypassing all Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
///
/// Reference: [NumPy logical_and](https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html)
NDArray<bool> logical_and(NDArray a, NDArray b, {NDArray<bool>? out}) {
  return _runBinaryLogical(
    a,
    b,
    out,
    v_logical_and,
    s_logical_and,
    'logical_and',
  );
}

/// Compute the element-wise truth value of [a] OR [b] with broadcasting support.
///
/// Returns a boolean array containing the element-wise logical OR results.
///
/// **Preconditions:**
/// - [a] and [b] must not be disposed.
/// - If provided, the [out] recycler array must exactly match the broadcasted shape of [a] and [b], and have [DType.boolean] dtype.
///
/// **Throws:**
/// - [StateError] if any input array has been disposed.
/// - [ArgumentError] if shapes of [a] and [b] are incompatible for broadcasting.
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous layout layouts, offloads the loop directly to high-speed native C
///   vector logical kernels (`v_logical_or`), bypassing all Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
///
/// Reference: [NumPy logical_or](https://numpy.org/doc/stable/reference/generated/numpy.logical_or.html)
NDArray<bool> logical_or(NDArray a, NDArray b, {NDArray<bool>? out}) {
  return _runBinaryLogical(a, b, out, v_logical_or, s_logical_or, 'logical_or');
}

/// Compute the element-wise truth value of [a] XOR [b] with broadcasting support.
///
/// Returns a boolean array containing the element-wise logical XOR results.
///
/// **Preconditions:**
/// - [a] and [b] must not be disposed.
/// - If provided, the [out] recycler array must exactly match the broadcasted shape of [a] and [b], and have [DType.boolean] dtype.
///
/// **Throws:**
/// - [StateError] if any input array has been disposed.
/// - [ArgumentError] if shapes of [a] and [b] are incompatible for broadcasting.
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous layout layouts, offloads the loop directly to high-speed native C
///   vector logical kernels (`v_logical_xor`), bypassing all Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
///
/// Reference: [NumPy logical_xor](https://numpy.org/doc/stable/reference/generated/numpy.logical_xor.html)
NDArray<bool> logical_xor(NDArray a, NDArray b, {NDArray<bool>? out}) {
  return _runBinaryLogical(
    a,
    b,
    out,
    v_logical_xor,
    s_logical_xor,
    'logical_xor',
  );
}

BroadcastResult _broadcastBinaryStrides(
  List<int> shapeA,
  List<int> stridesA,
  List<int> shapeB,
  List<int> stridesB,
) {
  if (listEquals(shapeA, shapeB)) {
    return BroadcastResult(shapeA, stridesA, stridesB);
  }
  final maxLen = shapeA.length > shapeB.length ? shapeA.length : shapeB.length;
  final commonShape = List<int>.filled(maxLen, 1);
  final newStridesA = List<int>.filled(maxLen, 0);
  final newStridesB = List<int>.filled(maxLen, 0);

  for (var i = 0; i < maxLen; i++) {
    final dimA = i < shapeA.length ? shapeA[shapeA.length - 1 - i] : 1;
    final dimB = i < shapeB.length ? shapeB[shapeB.length - 1 - i] : 1;

    if (dimA == dimB) {
      commonShape[maxLen - 1 - i] = dimA;
      if (i < shapeA.length) {
        newStridesA[maxLen - 1 - i] = stridesA[shapeA.length - 1 - i];
      }
      if (i < shapeB.length) {
        newStridesB[maxLen - 1 - i] = stridesB[shapeB.length - 1 - i];
      }
    } else if (dimA == 1) {
      commonShape[maxLen - 1 - i] = dimB;
      if (i < shapeB.length) {
        newStridesB[maxLen - 1 - i] = stridesB[shapeB.length - 1 - i];
      }
    } else if (dimB == 1) {
      commonShape[maxLen - 1 - i] = dimA;
      if (i < shapeA.length) {
        newStridesA[maxLen - 1 - i] = stridesA[shapeA.length - 1 - i];
      }
    } else {
      throw ArgumentError(
        'Shapes $shapeA and $shapeB are not compatible for broadcasting',
      );
    }
  }
  return BroadcastResult(commonShape, newStridesA, newStridesB);
}

ffi.Pointer<ffi.Uint8> _castToBoolean(
  NDArray x,
  ffi.Pointer<ffi.Uint8> destPtr,
  List<int> destStrides,
) {
  if (x.isContiguous) {
    switch (x.dtype) {
      case DType.float64:
        v_to_bool_double(x.pointer.cast(), destPtr, x.size);
      case DType.float32:
        v_to_bool_float(x.pointer.cast(), destPtr, x.size);
      case DType.int64:
        v_to_bool_int64(x.pointer.cast(), destPtr, x.size);
      case DType.int32:
        v_to_bool_int32(x.pointer.cast(), destPtr, x.size);
      case DType.uint8:
        v_to_bool_uint8(x.pointer.cast(), destPtr, x.size);
      case DType.int16:
        v_to_bool_int16(x.pointer.cast(), destPtr, x.size);
      case DType.complex128:
        v_to_bool_complex128(x.pointer.cast(), destPtr, x.size);
      case DType.complex64:
        v_to_bool_complex64(x.pointer.cast(), destPtr, x.size);
      case DType.boolean:
        break;
    }
  } else {
    final ndim = x.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(ndim);
    final cShape = cBuffer;
    final cStridesX = cBuffer + ndim;
    final cStridesTemp = cBuffer + (ndim * 2);
    for (var i = 0; i < ndim; i++) {
      cShape[i] = x.shape[i];
      cStridesX[i] = x.strides[i];
      cStridesTemp[i] = destStrides[i];
    }
    switch (x.dtype) {
      case DType.float64:
        s_to_bool_double(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
        );
      case DType.float32:
        s_to_bool_float(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
        );
      case DType.int64:
        s_to_bool_int64(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
        );
      case DType.int32:
        s_to_bool_int32(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
        );
      case DType.uint8:
        s_to_bool_uint8(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
        );
      case DType.int16:
        s_to_bool_int16(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
        );
      case DType.complex128:
        s_to_bool_complex128(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
        );
      case DType.complex64:
        s_to_bool_complex64(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
        );
      case DType.boolean:
        break;
    }
  }
  return destPtr;
}

NDArray<bool> _runBinaryLogical(
  NDArray a,
  NDArray b,
  NDArray<bool>? out,
  void Function(
    ffi.Pointer<ffi.Uint8>,
    ffi.Pointer<ffi.Uint8>,
    ffi.Pointer<ffi.Uint8>,
    int,
  )
  contiguousFn,
  void Function(
    ffi.Pointer<ffi.Uint8>,
    ffi.Pointer<ffi.Int>,
    ffi.Pointer<ffi.Uint8>,
    ffi.Pointer<ffi.Int>,
    ffi.Pointer<ffi.Uint8>,
    ffi.Pointer<ffi.Int>,
    ffi.Pointer<ffi.Int>,
    int,
  )
  stridedFn,
  String opName,
) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot perform $opName on disposed arrays.');
  }

  final marker = ScratchArena.marker;

  final ffi.Pointer<ffi.Uint8> aBoolPtr;
  final List<int> aBoolStrides;
  if (a.dtype == DType.boolean) {
    aBoolPtr = a.pointer.cast();
    aBoolStrides = a.strides;
  } else {
    aBoolPtr = ScratchArena.allocate<ffi.Uint8>(a.size);
    aBoolStrides = NDArray.computeCStrides(a.shape);
    _castToBoolean(a, aBoolPtr, aBoolStrides);
  }

  final ffi.Pointer<ffi.Uint8> bBoolPtr;
  final List<int> bBoolStrides;
  if (b.dtype == DType.boolean) {
    bBoolPtr = b.pointer.cast();
    bBoolStrides = b.strides;
  } else {
    bBoolPtr = ScratchArena.allocate<ffi.Uint8>(b.size);
    bBoolStrides = NDArray.computeCStrides(b.shape);
    _castToBoolean(b, bBoolPtr, bBoolStrides);
  }

  final broadcastResult = _broadcastBinaryStrides(
    a.shape,
    aBoolStrides,
    b.shape,
    bBoolStrides,
  );
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final NDArray<bool> result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for $opName.',
      );
    }
    result = out;
  } else {
    result = NDArray<bool>.create(commonShape, DType.boolean);
  }

  final resultStrides = NDArray.computeCStrides(commonShape);

  final isContig =
      a.isContiguous &&
      b.isContiguous &&
      result.isContiguous &&
      listEquals(a.shape, b.shape);

  if (isContig) {
    contiguousFn(aBoolPtr, bBoolPtr, result.pointer.cast(), result.size);
  } else {
    final ndim = commonShape.length;
    final cBuffer = ScratchArena.getStridedBuffer(ndim);
    final cShape = cBuffer;
    final cStridesA = cBuffer + ndim;
    final cStridesB = cBuffer + (ndim * 2);
    final cStridesRes = cBuffer + (ndim * 3);

    for (var i = 0; i < ndim; i++) {
      cShape[i] = commonShape[i];
      cStridesA[i] = stridesA[i];
      cStridesB[i] = stridesB[i];
      cStridesRes[i] = resultStrides[i];
    }

    stridedFn(
      aBoolPtr,
      cStridesA,
      bBoolPtr,
      cStridesB,
      result.pointer.cast(),
      cStridesRes,
      cShape,
      ndim,
    );
  }

  ScratchArena.reset(marker);
  return result;
}

/// Compute the Cholesky decomposition of a square, positive-definite 2D matrix.
///
/// Factorizes a symmetric (or Hermitian for complex), positive-definite matrix [a] into
/// $A = L L^*$ (or $A = L L^T$ for real matrices), where $L$ is a lower triangular matrix
/// factor and $L^*$ is the conjugate transpose of $L$.
///
/// Natively offloads to LAPACK solvers (`dpotrf`, `spotrf`, `cpotrf`, `zpotrf`) depending on precision and complexity.
///
/// **Preconditions:**
/// - The input matrix [a] must not be disposed.
/// - The input matrix [a] must be 2D (shape length exactly 2).
/// - The input matrix [a] must be square (`shape[0] == shape[1]`).
/// - The input matrix [a] must have a floating-point or complex data type (`float32`, `float64`, `complex64`, or `complex128`).
/// - The input matrix [a] must be symmetric/Hermitian positive-definite.
/// - If provided, the [out] destination matrix must have the same shape and dtype as [a], and must be contiguous.
///
/// **Throws:**
/// - [StateError] if the input matrix [a] is disposed.
/// - [ArgumentError] if [a] is not square or not 2D.
/// - [ArgumentError] if [a] has an unsupported dtype (e.g. integer or boolean).
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape, dtype, or is not contiguous.
/// - [ArgumentError] if the matrix is not positive-definite, or if LAPACK returns an error code.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(n^3)$ flops for an $n \times n$ matrix.
/// - Uses LAPACK solvers.
/// - Performs zero memory allocations if a pre-allocated [out] buffer is provided and the input [a] is contiguous.
///
/// **Example:**
/// {@example /example/linalg_example.dart lang=dart}
///
/// Reference: [NumPy linalg.cholesky](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html)
NDArray<T> cholesky<T>(NDArray<T> a, {NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cholesky() on a disposed array.');
  }
  if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
    throw ArgumentError('Matrix must be square and 2D (was ${a.shape})');
  }
  if (!a.dtype.isFloating && !a.dtype.isComplex) {
    throw ArgumentError(
      'Cholesky decomposition is only supported for float and complex dtypes (was ${a.dtype})',
    );
  }
  final n = a.shape[0];
  final targetDType = a.dtype;

  final NDArray<T> src;
  final bool wasCopied;
  if (!a.isContiguous) {
    src = a.copy();
    wasCopied = true;
  } else {
    src = a;
    wasCopied = false;
  }

  final NDArray<T> lMat;
  if (out != null) {
    lMat = out;
    if (!listEquals(lMat.shape, a.shape) || lMat.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out L buffer has incompatible shape or dtype.',
      );
    }
    if (!lMat.isContiguous) {
      throw ArgumentError('Provided out L buffer must be contiguous.');
    }
    src.copy(out: lMat);
  } else {
    lMat = src.copy();
  }

  try {
    // Char 'L' in ASCII is 76
    const uploL = 76;

    final int info;
    switch (targetDType) {
      case DType.float64:
        info = LAPACKE_dpotrf(
          101, // ROW_MAJOR
          uploL,
          n,
          lMat.pointer.cast<ffi.Double>(),
          n,
        );
      case DType.float32:
        info = LAPACKE_spotrf(
          101, // ROW_MAJOR
          uploL,
          n,
          lMat.pointer.cast<ffi.Float>(),
          n,
        );
      case DType.complex128:
        info = LAPACKE_zpotrf(
          101, // ROW_MAJOR
          uploL,
          n,
          lMat.pointer.cast<ffi.Double>(),
          n,
        );
      case DType.complex64:
        info = LAPACKE_cpotrf(
          101, // ROW_MAJOR
          uploL,
          n,
          lMat.pointer.cast<ffi.Float>(),
          n,
        );
      default:
        throw UnimplementedError(
          'Unsupported dtype for Cholesky: $targetDType',
        );
    }

    if (info < 0) {
      throw ArgumentError(
        'Illegal value in call to LAPACKE Cholesky solver: $info',
      );
    }
    if (info > 0) {
      throw ArgumentError(
        'Matrix must be positive-definite for Cholesky decomposition',
      );
    }

    v_zero_upper_triangular(
      lMat.pointer.cast<ffi.Void>(),
      n,
      encodeDType(targetDType),
    );
  } finally {
    if (wasCopied) {
      src.dispose();
    }
  }

  return lMat;
}

/// Computes the QR decomposition of a matrix or a stack of matrices $A = Q R$.
///
/// Decomposes a matrix [a] out an orthogonal matrix `Q` and an upper triangular matrix `R`
/// such that `a = Q * R`.
/// Natively offloads to LAPACK solvers (`dgeqrf` / `sgeqrf` and `dorgqr` / `sorgqr`) depending on precision.
///
/// **Preconditions:**
/// - Input matrix [a] must be at least 2-dimensional.
///
/// **Throws:**
/// - [ArgumentError] if [a] rank is less than 2.
/// - [StateError] if native FFI memory allocation or LAPACK solver initialization fails.
///
/// **Performance considerations:**
/// - Executes at high-speed natively in unmanaged C space.
///
/// **Example:**
/// ```dart
/// final a = `NDArray<double>`.fromList([12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0], [3, 3], DType.float64);
/// final res = qr(a);
/// final q = res.Q;
/// final r = res.R;
/// ```
({NDArray<T> Q, NDArray<T> R}) qr<T>(
  NDArray<T> a, {
  ({NDArray<T> Q, NDArray<T> R})? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute qr() on a disposed array.');
  }
  final rank = a.shape.length;
  if (rank < 2) {
    throw ArgumentError('Matrix must be at least 2D (was ${a.shape})');
  }
  final m = a.shape[rank - 2];
  final n = a.shape[rank - 1];
  final k = m < n ? m : n;
  final stackShape = a.shape.sublist(0, rank - 2);

  final DType<double> targetDType = a.dtype == DType.float32
      ? DType.float32 as DType<double>
      : DType.float64 as DType<double>;

  final qShape = [...stackShape, m, k];
  final rShape = [...stackShape, k, n];

  final NDArray<double> qMat;
  final NDArray<double> rMat;
  if (out != null) {
    qMat = out.Q as NDArray<double>;
    rMat = out.R as NDArray<double>;
    if (!listEquals(qMat.shape, qShape) || qMat.dtype != targetDType) {
      throw ArgumentError(
        'Provided out Q buffer has incompatible shape or dtype.',
      );
    }
    if (!qMat.isContiguous) {
      throw ArgumentError('Provided out Q buffer must be contiguous.');
    }
    if (!listEquals(rMat.shape, rShape) || rMat.dtype != targetDType) {
      throw ArgumentError(
        'Provided out R buffer has incompatible shape or dtype.',
      );
    }
    if (!rMat.isContiguous) {
      throw ArgumentError('Provided out R buffer must be contiguous.');
    }
  } else {
    qMat = NDArray<double>.zeros(qShape, targetDType);
    rMat = NDArray<double>.zeros(rShape, targetDType);
  }

  final NDArray<T> aCast =
      (a.dtype == targetDType ? a : castNDArray(a, targetDType)) as NDArray<T>;
  final bool wasCast = a.dtype != targetDType;

  final aCopy = NDArray.create([m, n], targetDType);
  final marker = ScratchArena.marker;

  try {
    final ffi.Pointer<ffi.Void> tau;
    if (targetDType == DType.float64) {
      tau = ScratchArena.allocate<ffi.Double>(
        k * ffi.sizeOf<ffi.Double>(),
      ).cast<ffi.Void>();
    } else {
      tau = ScratchArena.allocate<ffi.Float>(
        k * ffi.sizeOf<ffi.Float>(),
      ).cast<ffi.Void>();
    }

    walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
      coords,
    ) {
      var offsetA = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetA += coords[i] * aCast.strides[i];
      }

      final sliceView = NDArray.view(
        aCast,
        shape: [m, n],
        strides: aCast.strides.sublist(rank - 2),
        offsetElements: offsetA,
      );
      sliceView.copy(out: aCopy as NDArray<T>);
      sliceView.dispose();

      final r2D = targetDType == DType.float32
          ? NDArray<Float32>.zeros([k, n], DType.float32)
          : NDArray<Float64>.zeros([k, n], DType.float64);
      final q2D = targetDType == DType.float32
          ? NDArray<Float32>.zeros([m, k], DType.float32)
          : NDArray<Float64>.zeros([m, k], DType.float64);

      if (targetDType == DType.float64) {
        final info = LAPACKE_dgeqrf(
          101, // ROW_MAJOR
          m,
          n,
          aCopy.pointer.cast<ffi.Double>(),
          n,
          tau.cast<ffi.Double>(),
        );
        if (info != 0) {
          throw ArgumentError('Illegal value in call to LAPACKE_dgeqrf: $info');
        }

        final r2DData = r2D.data as Float64List;
        final aCopyData = aCopy.data as Float64List;
        for (var i = 0; i < k; i++) {
          for (var j = i; j < n; j++) {
            r2DData[i * n + j] = aCopyData[i * n + j];
          }
        }

        final q2DData = q2D.data as Float64List;
        for (var i = 0; i < m; i++) {
          for (var j = 0; j < k; j++) {
            q2DData[i * k + j] = aCopyData[i * n + j];
          }
        }

        final infoOrg = LAPACKE_dorgqr(
          101, // ROW_MAJOR
          m,
          k,
          k,
          q2D.pointer.cast<ffi.Double>(),
          k,
          tau.cast<ffi.Double>(),
        );
        if (infoOrg != 0) {
          throw ArgumentError(
            'Illegal value in call to LAPACKE_dorgqr: $infoOrg',
          );
        }
      } else {
        final info = LAPACKE_sgeqrf(
          101, // ROW_MAJOR
          m,
          n,
          aCopy.pointer.cast<ffi.Float>(),
          n,
          tau.cast<ffi.Float>(),
        );
        if (info != 0) {
          throw ArgumentError('Illegal value in call to LAPACKE_sgeqrf: $info');
        }

        final r2DData = r2D.data as Float32List;
        final aCopyData = aCopy.data as Float32List;
        for (var i = 0; i < k; i++) {
          for (var j = i; j < n; j++) {
            r2DData[i * n + j] = aCopyData[i * n + j];
          }
        }

        final q2DData = q2D.data as Float32List;
        for (var i = 0; i < m; i++) {
          for (var j = 0; j < k; j++) {
            q2DData[i * k + j] = aCopyData[i * n + j];
          }
        }

        final infoOrg = LAPACKE_sorgqr(
          101, // ROW_MAJOR
          m,
          k,
          k,
          q2D.pointer.cast<ffi.Float>(),
          k,
          tau.cast<ffi.Float>(),
        );
        if (infoOrg != 0) {
          throw ArgumentError(
            'Illegal value in call to LAPACKE_sorgqr: $infoOrg',
          );
        }
      }

      var offsetQ = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetQ += coords[i] * qMat.strides[i];
      }
      var offsetR = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetR += coords[i] * rMat.strides[i];
      }

      final qSlice = NDArray.view(
        qMat,
        shape: [m, k],
        strides: qMat.strides.sublist(rank - 2),
        offsetElements: offsetQ,
      );
      q2D.copy(out: qSlice);
      qSlice.dispose();

      final rSlice = NDArray.view(
        rMat,
        shape: [k, n],
        strides: rMat.strides.sublist(rank - 2),
        offsetElements: offsetR,
      );
      r2D.copy(out: rSlice);
      rSlice.dispose();

      q2D.dispose();
      r2D.dispose();
    });
  } finally {
    ScratchArena.reset(marker);
    aCopy.dispose();
    if (wasCast) {
      aCast.dispose();
    }
  }

  return (Q: qMat as NDArray<T>, R: rMat as NDArray<T>);
}

/// Computes the Singular Value Decomposition (SVD) of a matrix or a stack of matrices $A = U S V^h$.
///
/// Decomposes a matrix [a] out left singular vectors `U`, singular values `S`,
/// and right singular vectors Vh such that `a = U * diag(S) * Vh`.
/// Natively offloads to LAPACK solvers (`dgesdd` / `sgesdd`) depending on precision.
///
/// **Preconditions:**
/// - Input matrix [a] must be at least 2-dimensional.
///
/// **Throws:**
/// - [ArgumentError] if [a] rank is less than 2.
/// - [StateError] if native FFI memory allocation or LAPACK solver initialization fails.
///
/// **Performance considerations:**
/// - Executes at high-speed natively in unmanaged C space.
///
/// **Example:**
/// ```dart
/// final a = `NDArray<double>`.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], DType.float64);
/// final res = svd(a);
/// final u = res.U;
/// final s = res.S;
/// final vh = res.Vh;
/// ```
({NDArray<T> U, NDArray<T> S, NDArray<T> Vh}) svd<T>(NDArray<T> a) {
  if (a.isDisposed) {
    throw StateError('Cannot execute svd() on a disposed array.');
  }
  final rank = a.shape.length;
  if (rank < 2) {
    throw ArgumentError('Matrix must be at least 2D (was ${a.shape})');
  }
  final m = a.shape[rank - 2];
  final n = a.shape[rank - 1];
  final stackShape = a.shape.sublist(0, rank - 2);

  if (m < n) {
    final axes = List<int>.generate(rank, (i) => i);
    axes[rank - 2] = rank - 1;
    axes[rank - 1] = rank - 2;

    final aT = a.transpose(axes);
    final resT = svd(aT);
    final uNew = resT.U;
    final sNew = resT.S;
    final vhNew = resT.Vh;

    final uResult = vhNew.transpose(axes);
    final vhResult = uNew.transpose(axes);

    return (U: uResult, S: sNew, Vh: vhResult);
  }

  final DType<double> targetDType = a.dtype == DType.float32
      ? DType.float32 as DType<double>
      : DType.float64 as DType<double>;

  final uShape = [...stackShape, m, m];
  final sShape = [...stackShape, n];
  final vtShape = [...stackShape, n, n];

  final NDArray<double> uMat = NDArray<double>.zeros(uShape, targetDType);
  final NDArray<double> sMat = NDArray<double>.zeros(sShape, targetDType);
  final NDArray<double> vtMat = NDArray<double>.zeros(vtShape, targetDType);

  final NDArray<T> aCast =
      (a.dtype == targetDType ? a : castNDArray(a, targetDType)) as NDArray<T>;
  final bool wasCast = a.dtype != targetDType;

  final aCopy = NDArray.create([m, n], targetDType);
  final marker = ScratchArena.marker;

  try {
    final ffi.Pointer<ffi.Void> superb;
    final superbLen = math.max(1, n - 1);
    if (targetDType == DType.float64) {
      superb = ScratchArena.allocate<ffi.Double>(
        superbLen * ffi.sizeOf<ffi.Double>(),
      ).cast<ffi.Void>();
    } else {
      superb = ScratchArena.allocate<ffi.Float>(
        superbLen * ffi.sizeOf<ffi.Float>(),
      ).cast<ffi.Void>();
    }

    walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
      coords,
    ) {
      var offsetA = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetA += coords[i] * aCast.strides[i];
      }

      final sliceView = NDArray.view(
        aCast,
        shape: [m, n],
        strides: aCast.strides.sublist(rank - 2),
        offsetElements: offsetA,
      );
      sliceView.copy(out: aCopy as NDArray<T>);
      sliceView.dispose();

      final s2D = targetDType == DType.float32
          ? NDArray<Float32>.zeros([n], DType.float32)
          : NDArray<Float64>.zeros([n], DType.float64);
      final u2D = targetDType == DType.float32
          ? NDArray<Float32>.zeros([m, m], DType.float32)
          : NDArray<Float64>.zeros([m, m], DType.float64);
      final vt2D = targetDType == DType.float32
          ? NDArray<Float32>.zeros([n, n], DType.float32)
          : NDArray<Float64>.zeros([n, n], DType.float64);

      if (targetDType == DType.float64) {
        final info = LAPACKE_dgesvd(
          101, // ROW_MAJOR
          65, // 'A'
          65, // 'A'
          m,
          n,
          aCopy.pointer.cast<ffi.Double>(),
          n,
          s2D.pointer.cast<ffi.Double>(),
          u2D.pointer.cast<ffi.Double>(),
          m,
          vt2D.pointer.cast<ffi.Double>(),
          n,
          superb.cast<ffi.Double>(),
        );
        if (info != 0) {
          throw ArgumentError('Illegal value in call to LAPACKE_dgesvd: $info');
        }
      } else {
        final info = LAPACKE_sgesvd(
          101, // ROW_MAJOR
          65, // 'A'
          65, // 'A'
          m,
          n,
          aCopy.pointer.cast<ffi.Float>(),
          n,
          s2D.pointer.cast<ffi.Float>(),
          u2D.pointer.cast<ffi.Float>(),
          m,
          vt2D.pointer.cast<ffi.Float>(),
          n,
          superb.cast<ffi.Float>(),
        );
        if (info != 0) {
          throw ArgumentError('Illegal value in call to LAPACKE_sgesvd: $info');
        }
      }

      var offsetU = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetU += coords[i] * uMat.strides[i];
      }
      var offsetS = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetS += coords[i] * sMat.strides[i];
      }
      var offsetVt = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetVt += coords[i] * vtMat.strides[i];
      }

      final uSlice = NDArray.view(
        uMat,
        shape: [m, m],
        strides: uMat.strides.sublist(rank - 2),
        offsetElements: offsetU,
      );
      u2D.copy(out: uSlice);
      uSlice.dispose();

      final sSlice = NDArray.view(
        sMat,
        shape: [n],
        strides: sMat.strides.isEmpty ? [1] : [sMat.strides.last],
        offsetElements: offsetS,
      );
      s2D.copy(out: sSlice);
      sSlice.dispose();

      final vtSlice = NDArray.view(
        vtMat,
        shape: [n, n],
        strides: vtMat.strides.sublist(rank - 2),
        offsetElements: offsetVt,
      );
      vt2D.copy(out: vtSlice);
      vtSlice.dispose();

      s2D.dispose();
      u2D.dispose();
      vt2D.dispose();
    });
  } finally {
    ScratchArena.reset(marker);
    aCopy.dispose();
    if (wasCast) {
      aCast.dispose();
    }
  }

  return (
    U: uMat as NDArray<T>,
    S: sMat as NDArray<T>,
    Vh: vtMat as NDArray<T>,
  );
}

/// Extract a diagonal or construct a diagonal array.
///
/// If [v] is a 2D matrix, extracts the k-th diagonal elements vector as a zero-copy 1D view.
/// If [v] is a 1D vector, constructs a 2D square matrix with [v] as the k-th diagonal and zeros elsewhere.
///
/// **Preconditions:**
/// - Input [v] must be a 1D or 2D array.
///
/// **Throws:**
/// - [ArgumentError] if [v] rank is not 1 or 2.
///
/// **Example:**
/// {@example /example/diag_example.dart lang=dart}
///
/// Reference: [Diagonal Matrix](https://en.wikipedia.org/wiki/Diagonal_matrix)
NDArray<T> diag<T>(NDArray<T> v, {int k = 0, NDArray<T>? out}) {
  if (v.shape.length == 2) {
    final m = v.shape[0];
    final n = v.shape[1];

    int startRow;
    int startCol;
    int len;

    if (k >= 0) {
      startRow = 0;
      startCol = k;
      if (startCol >= n) {
        return NDArray<T>.create([0], v.dtype);
      }
      len = math.min(m, n - k);
    } else {
      startRow = -k;
      startCol = 0;
      if (startRow >= m) {
        return NDArray<T>.create([0], v.dtype);
      }
      len = math.min(m + k, n);
    }

    if (len <= 0) {
      return NDArray<T>.create([0], v.dtype);
    }

    final offsetElements = startRow * v.strides[0] + startCol * v.strides[1];
    final diagStride = v.strides[0] + v.strides[1];

    return NDArray<T>.view(
      v,
      shape: [len],
      strides: [diagStride],
      offsetElements: offsetElements,
    );
  } else if (v.shape.length == 1) {
    final n = v.shape[0];
    final size = n + k.abs();
    final targetShape = [size, size];

    final result = out ?? NDArray<T>.zeros(targetShape, v.dtype);
    if (out != null) {
      if (!listEquals(out.shape, targetShape) || out.dtype != v.dtype) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
      for (var i = 0; i < result.data.length; i++) {
        result.data[i] = castValue(0, v.dtype) as T;
      }
    }

    int startRow;
    int startCol;

    if (k >= 0) {
      startRow = 0;
      startCol = k;
    } else {
      startRow = -k;
      startCol = 0;
    }

    final vList = v.toList();
    final resData = result.data;
    final resStrides = result.strides;

    for (var i = 0; i < n; i++) {
      final targetIdx =
          (startRow + i) * resStrides[0] + (startCol + i) * resStrides[1];
      resData[targetIdx] = vList[i];
    }

    return result;
  } else {
    throw ArgumentError('Input array must be 1- or 2-dimensional.');
  }
}

/// Returns a boolean [NDArray] where two arrays are element-wise equal within a tolerance.
///
/// The tolerance relation is defined as:
/// `abs(a - b) <= (atol + rtol * abs(b))`
///
/// **Preconditions:**
/// - Input [a] and [b] must be numeric arrays.
/// - [a] and [b] must have compatible broadcast shapes.
///
/// **Example:**
/// {@example /example/isclose_example.dart lang=dart}
///
/// Reference: [Approximate Equality](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)
NDArray<bool> isclose(
  NDArray a,
  NDArray b, {
  double rtol = 1e-05,
  double atol = 1e-08,
  bool equalNan = false,
}) {
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  final size = commonShape.isEmpty ? 1 : commonShape.reduce((x, y) => x * y);
  final result = NDArray<bool>.zeros(commonShape, DType.boolean);

  final aList = a.toList().cast<num>();
  final bList = b.toList().cast<num>();
  final resData = result.data;

  final broadcastResultRes = broadcast(a, b);
  final stridesA = broadcastResultRes.stridesA;
  final stridesB = broadcastResultRes.stridesB;
  final stridesRes = NDArray.computeCStrides(commonShape);

  final coord = List<int>.filled(commonShape.length, 0);

  for (var el = 0; el < size; el++) {
    var offsetA = 0;
    var offsetB = 0;
    var offsetRes = 0;
    for (var d = 0; d < commonShape.length; d++) {
      offsetA += coord[d] * stridesA[d];
      offsetB += coord[d] * stridesB[d];
      offsetRes += coord[d] * stridesRes[d];
    }

    final valA = aList[offsetA].toDouble();
    final valB = bList[offsetB].toDouble();

    var match = false;
    if (equalNan && valA.isNaN && valB.isNaN) {
      match = true;
    } else if (valA.isInfinite || valB.isInfinite) {
      match = valA == valB;
    } else {
      final diff = (valA - valB).abs();
      final limit = atol + rtol * valB.abs();
      match = diff <= limit;
    }

    resData[offsetRes] = match;

    // Advance coord odometer
    for (var d = commonShape.length - 1; d >= 0; d--) {
      coord[d]++;
      if (coord[d] < commonShape[d]) break;
      coord[d] = 0;
    }
  }

  return result;
}

/// Returns true if two arrays are element-wise equal within a tolerance.
///
/// The tolerance relation is defined as:
/// `abs(a - b) <= (atol + rtol * abs(b))`
///
/// **Preconditions:**
/// - Input [a] and [b] must be numeric arrays.
/// - [a] and [b] must have compatible broadcast shapes.
///
/// **Example:**
/// {@example /example/isclose_example.dart lang=dart}
///
/// Reference: [Approximate Equality](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html)
bool allclose(
  NDArray a,
  NDArray b, {
  double rtol = 1e-05,
  double atol = 1e-08,
  bool equalNan = false,
}) {
  final closeMask = isclose(a, b, rtol: rtol, atol: atol, equalNan: equalNan);
  final maskList = closeMask.toList();
  closeMask.dispose();
  for (final val in maskList) {
    if (!val) return false;
  }
  return true;
}

/// Replace NaN with zero and infinity with large finite numbers.
///
/// By default, maps NaN to [nan] (which defaults to 0.0), maps positive infinity
/// to [posinf] (or the maximum finite float value if null), and maps negative infinity
/// to [neginf] (or the minimum finite float value if null).
///
/// **Preconditions:**
/// - Input [a] must be a numeric array.
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/nan_to_num_example.dart lang=dart}
///
/// Reference: [Replace NaN and Infinities](https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
NDArray nan_to_num(
  NDArray a, {
  double nan = 0.0,
  double? posinf,
  double? neginf,
  NDArray? out,
}) {
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for nan_to_num.',
      );
    }
  }

  final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
  final aList = a.toList();
  final resultCopy = out ?? NDArray.create(a.shape, a.dtype);

  final maxLimit = a.dtype == DType.float32
      ? 3.4028234663852886e+38
      : double.maxFinite;
  final minLimit = -maxLimit;

  final targetPosInf = posinf ?? maxLimit;
  final targetNegInf = neginf ?? minLimit;

  final cleanList = <dynamic>[];

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    final complexList = aList.cast<Complex>();
    for (var i = 0; i < size; i++) {
      var r = complexList[i].real;
      var img = complexList[i].imag;

      if (r.isNaN) r = nan;
      if (r == double.infinity) r = targetPosInf;
      if (r == double.negativeInfinity) r = targetNegInf;

      if (img.isNaN) img = nan;
      if (img == double.infinity) img = targetPosInf;
      if (img == double.negativeInfinity) img = targetNegInf;

      cleanList.add(Complex(r, img));
    }
  } else {
    final numList = aList.cast<num>();
    for (var i = 0; i < size; i++) {
      var val = numList[i].toDouble();

      if (val.isNaN) {
        val = nan;
      } else if (val == double.infinity) {
        val = targetPosInf;
      } else if (val == double.negativeInfinity) {
        val = targetNegInf;
      }

      cleanList.add(val);
    }
  }

  // View-Safe Strided Odometer Write Back!
  final resData = resultCopy.data;
  final resStrides = resultCopy.strides;
  final coord = List<int>.filled(a.shape.length, 0);

  for (var i = 0; i < size; i++) {
    var offsetRes = 0;
    for (var d = 0; d < a.shape.length; d++) {
      offsetRes += coord[d] * resStrides[d];
    }

    resData[offsetRes] = cleanList[i];

    // Advance odometer
    for (var d = a.shape.length - 1; d >= 0; d--) {
      coord[d]++;
      if (coord[d] < a.shape[d]) break;
      coord[d] = 0;
    }
  }

  return resultCopy;
}

/// Compute the broadcasted shape list of two shapes.
List<int> broadcastShapes(List<int> s1, List<int> s2) {
  final len = math.max(s1.length, s2.length);
  final common = List<int>.filled(len, 1);
  for (var i = 0; i < len; i++) {
    final dim1 = s1.length - 1 - i >= 0 ? s1[s1.length - 1 - i] : 1;
    final dim2 = s2.length - 1 - i >= 0 ? s2[s2.length - 1 - i] : 1;

    final target = math.max(dim1, dim2);
    if (dim1 != target && dim1 != 1) {
      throw ArgumentError('Incompatible shapes for broadcasting');
    }
    if (dim2 != target && dim2 != 1) {
      throw ArgumentError('Incompatible shapes for broadcasting');
    }
    common[len - 1 - i] = target;
  }
  return common;
}

/// Return an array drawn from elements in [choicelist], depending on conditions in [condlist].
///
/// This corresponds to NumPy's `select` function.
///
/// **Mathematical Mechanics**:
/// - Evaluates a list of boolean conditions in [condlist] sequentially per cell.
/// - Draws corresponding values from the same-indexed array in [choicelist].
/// - If no condition is met, falls back to [defaultValue].
/// - Leverages zero-copy, zero-allocation $N$-dimensional strides recursive walk in a single pass!
///
/// **Preconditions:**
/// - [condlist] and [choicelist] must have the same length.
/// - All condition and choice arrays must broadcast perfectly to a common shape.
///
/// **Throws:**
/// - [ArgumentError] if [condlist] and [choicelist] lengths mismatch, or if any shape is incompatible.
///
/// **Example:**
/// ```dart
/// final cond1 = NDArray.fromList([true, false], [2], DType.boolean);
/// final cond2 = NDArray.fromList([false, true], [2], DType.boolean);
/// final choice1 = NDArray.fromList([10, 20], [2], DType.int32);
/// final choice2 = NDArray.fromList([100, 200], [2], DType.int32);
/// final res = select([cond1, cond2], [choice1, choice2], defaultValue: 999);
/// print(res.toList()); // [10, 200]
/// ```
NDArray select(
  List<NDArray<bool>> condlist,
  List<NDArray> choicelist, {
  dynamic defaultValue = 0,
}) {
  if (condlist.isEmpty || choicelist.isEmpty) {
    throw ArgumentError('condlist and choicelist must not be empty');
  }
  if (condlist.length != choicelist.length) {
    throw ArgumentError(
      'condlist length (${condlist.length}) must match choicelist length (${choicelist.length})',
    );
  }

  // 1. Calculate common broadcasted shape
  final allShapes = <List<int>>[];
  for (final c in condlist) {
    allShapes.add(c.shape);
  }
  for (final c in choicelist) {
    allShapes.add(c.shape);
  }

  var commonShape = allShapes[0];
  for (var i = 1; i < allShapes.length; i++) {
    commonShape = broadcastShapes(commonShape, allShapes[i]);
  }

  // 2. Determine target upcasted DType
  var targetDType = choicelist[0].dtype;
  for (var i = 1; i < choicelist.length; i++) {
    targetDType = resolveDType(targetDType, choicelist[i].dtype);
  }
  if (defaultValue is double &&
      !targetDType.isFloating &&
      !targetDType.isComplex) {
    targetDType = DType.float64;
  }

  final result = NDArray.create(commonShape, targetDType);

  // 3. Compute strides for all condition and choice operands independently to commonShape
  final stridesCond = condlist
      .map((c) => broadcastStrides(c, commonShape))
      .toList();
  final stridesChoice = choicelist
      .map((c) => broadcastStrides(c, commonShape))
      .toList();
  final resultStrides = NDArray.computeCStrides(commonShape);

  // 4. Execute recursive multi-operand strided walk
  final currentPos = List<int>.filled(commonShape.length, 0);
  final initialOffsetsCond = List<int>.filled(condlist.length, 0);
  final initialOffsetsChoice = List<int>.filled(choicelist.length, 0);

  selectRecursive(
    result,
    condlist,
    choicelist,
    stridesCond,
    stridesChoice,
    resultStrides,
    currentPos,
    0,
    initialOffsetsCond,
    initialOffsetsChoice,
    0,
    defaultValue,
  );

  return result;
}

/// Return the Hanning (Hann) window.
///
/// The Hanning window is a taper formed by using a weighted cosine:
///
/// $$w[n] = 0.5 - 0.5 \cos\left(\frac{2\pi n}{M - 1}\right), \quad 0 \le n \le M-1$$
///
/// Unlike the Hamming window, the Hanning window tapers all the way to exactly
/// **zero** at the boundaries ($w[0] = w[M-1] = 0.0$). It features a fast side-lobe
/// roll-off rate of $18 \text{ dB/octave}$, making it highly suitable for general
/// spectral analysis where suppression of distant side lobes is critical.
///
/// **Example:**
/// ```dart
/// final window = hanning(512);
/// ```
NDArray<T> hanning<T>(int M, {DType<T>? dtype}) {
  final resolvedDType = dtype ?? (DType.float64 as DType<T>);
  if (M < 1) return NDArray.create([0], resolvedDType);
  if (M == 1) {
    return NDArray.fromList(
      [castValue(1.0, resolvedDType)],
      [1],
      resolvedDType,
    );
  }

  if (resolvedDType == DType.float32) {
    final res = NDArray<T>.create([M], resolvedDType);
    v_hanning_float(res.pointer.cast(), M);
    return res;
  } else if (resolvedDType == DType.float64) {
    final res = NDArray<T>.create([M], resolvedDType);
    v_hanning_double(res.pointer.cast(), M);
    return res;
  } else {
    final temp = NDArray<double>.create([M], DType.float64);
    v_hanning_double(temp.pointer.cast(), M);
    final res = castNDArray(temp, resolvedDType);
    temp.dispose();
    return res;
  }
}

/// Return the Hamming window.
///
/// The Hamming window is a taper formed by using an optimized weighted cosine:
///
/// $$w[n] = 0.54 - 0.46 \cos\left(\frac{2\pi n}{M - 1}\right), \quad 0 \le n \le M-1$$
///
/// Unlike the Hanning window, the Hamming window does not taper to zero at the boundaries,
/// leaving a small pedestal/discontinuity ($w[0] = w[M-1] = 0.08$). It is optimized to
/// minimize the maximum side-lobe level (achieving a first side lobe of $-43 \text{ dB}$
/// compared to Hanning's $-32 \text{ dB}$), at the expense of a slower side-lobe roll-off
/// rate of $6 \text{ dB/octave}$.
///
/// **Example:**
/// ```dart
/// final window = hamming(512);
/// ```
NDArray<T> hamming<T>(int M, {DType<T>? dtype}) {
  final resolvedDType = dtype ?? (DType.float64 as DType<T>);
  if (M < 1) return NDArray.create([0], resolvedDType);
  if (M == 1) {
    return NDArray.fromList(
      [castValue(1.0, resolvedDType)],
      [1],
      resolvedDType,
    );
  }

  if (resolvedDType == DType.float32) {
    final res = NDArray<T>.create([M], resolvedDType);
    v_hamming_float(res.pointer.cast(), M);
    return res;
  } else if (resolvedDType == DType.float64) {
    final res = NDArray<T>.create([M], resolvedDType);
    v_hamming_double(res.pointer.cast(), M);
    return res;
  } else {
    final temp = NDArray<double>.create([M], DType.float64);
    v_hamming_double(temp.pointer.cast(), M);
    final res = castNDArray(temp, resolvedDType);
    temp.dispose();
    return res;
  }
}

/// Extract a lower triangular matrix (on and below the k-th diagonal) element-wise.
///
/// **Preconditions:**
/// - Input [a] must be an array with rank >= 2.
/// - If provided, the [out] recycler must have matching shape and dtype.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [a] has rank < 2.
/// - [ArgumentError] if [out] has mismatched shape or dtype.
///
/// **Example:**
/// {@example /example/triangular_example.dart lang=dart}
NDArray<T> tril<T>(NDArray<T> a, {int k = 0, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute tril() on a disposed array.');
  }
  if (a.shape.length < 2) {
    throw ArgumentError('Input array must have rank >= 2.');
  }
  final result = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final rank = a.shape.length;
  final rows = a.shape[rank - 2];
  final cols = a.shape[rank - 1];

  final batchCount = a.shape.isEmpty
      ? 1
      : a.shape.sublist(0, rank - 2).isEmpty
      ? 1
      : a.shape.sublist(0, rank - 2).reduce((x, y) => x * y);

  if (a.isContiguous && result.isContiguous) {
    if (a.dtype == DType.float64) {
      v_tril_double(
        a.pointer.cast(),
        result.pointer.cast(),
        batchCount,
        rows,
        cols,
        k,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      v_tril_float(
        a.pointer.cast(),
        result.pointer.cast(),
        batchCount,
        rows,
        cols,
        k,
      );
      return result;
    }
  }

  final aList = a.isContiguous ? a.data : a.toList();
  final resData = result.data;
  final matrixSize = rows * cols;

  for (var b = 0; b < batchCount; b++) {
    final offset = b * matrixSize;
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        final idx = offset + r * cols + c;
        resData[idx] = (c <= r + k) ? aList[idx] : castValue(0, a.dtype) as T;
      }
    }
  }
  return result;
}

/// Extract an upper triangular matrix (on and above the k-th diagonal) element-wise.
///
/// **Preconditions:**
/// - Input [a] must be an array with rank >= 2.
/// - If provided, the [out] recycler must have matching shape and dtype.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [a] has rank < 2.
/// - [ArgumentError] if [out] has mismatched shape or dtype.
///
/// **Example:**
/// {@example /example/triangular_example.dart lang=dart}
NDArray<T> triu<T>(NDArray<T> a, {int k = 0, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute triu() on a disposed array.');
  }
  if (a.shape.length < 2) {
    throw ArgumentError('Input array must have rank >= 2.');
  }
  final result = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final rank = a.shape.length;
  final rows = a.shape[rank - 2];
  final cols = a.shape[rank - 1];

  final batchCount = a.shape.isEmpty
      ? 1
      : a.shape.sublist(0, rank - 2).isEmpty
      ? 1
      : a.shape.sublist(0, rank - 2).reduce((x, y) => x * y);

  if (a.isContiguous && result.isContiguous) {
    if (a.dtype == DType.float64) {
      v_triu_double(
        a.pointer.cast(),
        result.pointer.cast(),
        batchCount,
        rows,
        cols,
        k,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      v_triu_float(
        a.pointer.cast(),
        result.pointer.cast(),
        batchCount,
        rows,
        cols,
        k,
      );
      return result;
    }
  }

  final aList = a.isContiguous ? a.data : a.toList();
  final resData = result.data;
  final matrixSize = rows * cols;

  for (var b = 0; b < batchCount; b++) {
    final offset = b * matrixSize;
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        final idx = offset + r * cols + c;
        resData[idx] = (c >= r + k) ? aList[idx] : castValue(0, a.dtype) as T;
      }
    }
  }
  return result;
}

/// Result of a least-squares solver [lstsq].
///
/// Reference: [NumPy linalg.lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
final class LstsqResult<T> {
  /// Least-squares solution.
  ///
  /// If the input [b] is 1-dimensional, [x] has shape `[N]`.
  /// If [b] is 2-dimensional, [x] has shape `[N, K]`.
  final NDArray<T> x;

  /// Sums of squared residuals.
  ///
  /// Squared Euclidean 2-norm for each column in $b - a x$.
  /// If the input [b] is 1-dimensional, [residuals] has shape `[1]`.
  /// If [b] is 2-dimensional, [residuals] has shape `[K]`.
  ///
  /// **Note:** Residuals are only computed if the first dimension of the input matrix $a$
  /// is strictly greater than its second dimension ($M > N$) and the effective rank is $N$.
  /// Otherwise, it is returned as an empty array of shape `[0]`.
  final NDArray<double> residuals;

  /// Effective rank of the input matrix $a$.
  final int rank;

  /// Singular values of the input matrix $a$.
  ///
  /// Stored in descending order of magnitude.
  /// Shape is `[min(M, N)]`.
  final NDArray<double> s;

  /// Creates a new [LstsqResult] instance.
  LstsqResult({
    required this.x,
    required this.residuals,
    required this.rank,
    required this.s,
  });
}

/// Computes the least-squares solution to a linear matrix equation $a x = b$.
///
/// Solves the equation $a x = b$ by computing a vector/matrix $x$ that minimizes the
/// Euclidean 2-norm $\|b - a x\|_2^2$.
///
/// Natively offloads to LAPACK divide-and-conquer SVD-based least-squares solvers
/// (`dgelsd`, `sgelsd`, `zgelsd`, `cgelsd`) depending on precision.
///
/// The optional parameter [rcond] acts as the cut-off ratio for small singular values.
/// Singular values smaller than `rcond * largest_singular_value` are treated as zero.
/// If [rcond] is omitted or null, a negative value is passed to the LAPACK solver,
/// which falls back to using the machine precision to determine the effective rank.
///
/// The optional recycler parameter [out] allows reusing an existing array for the output,
/// avoiding new memory allocation.
///
/// **Preconditions:**
/// - Input matrix [a] must be 2-dimensional of shape `[M, N]`.
/// - Input array [b] must be 1-dimensional of shape `[M]` or 2-dimensional of shape `[M, K]`.
/// - The first dimension of [b] must exactly match the first dimension of [a] ($M$).
/// - Input arrays [a] and [b] must have the matching floating-point or complex [DType]. Integers or boolean
///   arrays are not supported.
/// - If provided, the recycler [out] must have the shape `[N]` (if [b] is 1D) or `[N, K]` (if [b] is 2D),
///   and its dtype must exactly match the dtype of [a] and [b].
///
/// **Throws:**
/// - [StateError] if [a] or [b] is disposed.
/// - [ArgumentError] if [a] or [b] does not have a floating-point or complex DType.
/// - [ArgumentError] if [b]'s DType does not match [a]'s DType.
/// - [ArgumentError] if [a] is not 2D, or [b] is not 1D or 2D.
/// - [ArgumentError] if [b]'s first dimension does not match [a]'s first dimension.
/// - [StateError] if [out] is provided but disposed.
/// - [ArgumentError] if [out] has mismatched shape or dtype.
/// - [StateError] if native FFI memory allocation fails or the SVD solver fails to converge.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(M N \min(M, N))$ operations executed in highly-optimized native C space.
///
/// **Example:**
/// {@example /example/linalg_lstsq_example.dart lang=dart}
///
/// Reference: [NumPy linalg.lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
LstsqResult<T> lstsq<T>(
  NDArray<T> a,
  NDArray<T> b, {
  double? rcond,
  NDArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute lstsq() on a disposed array.');
  }
  if (!a.dtype.isFloating && !a.dtype.isComplex) {
    throw ArgumentError(
      'Input array a must have a floating-point or complex DType (was ${a.dtype}).',
    );
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError(
      'Input array b must have the matching DType as a (expected ${a.dtype}, was ${b.dtype}).',
    );
  }
  if (a.shape.length != 2) {
    throw ArgumentError(
      'Input matrix a must be 2-dimensional (was shape ${a.shape}).',
    );
  }
  if (b.shape.length != 1 && b.shape.length != 2) {
    throw ArgumentError(
      'Input right-hand side b must be 1D or 2D (was shape ${b.shape}).',
    );
  }
  final m = a.shape[0];
  final n = a.shape[1];
  if (b.shape[0] != m) {
    throw ArgumentError(
      'First dimension of b (${b.shape[0]}) must match first dimension of a ($m).',
    );
  }

  final nrhs = b.shape.length > 1 ? b.shape[1] : 1;

  if (out != null) {
    if (out.isDisposed) {
      throw StateError('Cannot write to a disposed out buffer.');
    }
    final expectedXShape = b.shape.length > 1 ? [n, nrhs] : [n];
    if (!listEquals(out.shape, expectedXShape) || out.dtype != a.dtype) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  // Create a contiguous copy of `a` (overwrite-safe)
  final aCopy = a.copy();

  // Row-major LAPACKE_gelsd requires b array size to be max(m, n) * nrhs
  final maxMN = m > n ? m : n;
  final bCopyShape = b.shape.length > 1 ? [maxMN, nrhs] : [maxMN];
  final bCopy = NDArray<T>.zeros(bCopyShape, a.dtype);

  // Copy b into bCopy
  final byteCount = b.data.length * a.dtype.byteWidth;
  if (b.isContiguous) {
    ffi.Pointer.fromAddress(bCopy.pointer.address)
        .cast<ffi.Uint8>()
        .asTypedList(byteCount)
        .setAll(
          0,
          ffi.Pointer.fromAddress(
            b.pointer.address,
          ).cast<ffi.Uint8>().asTypedList(byteCount),
        );
  } else {
    final bContig = b.copy();
    ffi.Pointer.fromAddress(bCopy.pointer.address)
        .cast<ffi.Uint8>()
        .asTypedList(byteCount)
        .setAll(
          0,
          ffi.Pointer.fromAddress(
            bContig.pointer.address,
          ).cast<ffi.Uint8>().asTypedList(byteCount),
        );
    bContig.dispose();
  }

  final minMN = m < n ? m : n;
  // Singular values s is always real
  final sDType = (a.dtype == DType.complex64 || a.dtype == DType.float32)
      ? DType.float32
      : DType.float64;
  final s = NDArray<double>.zeros([minMN], sDType as dynamic);

  final marker = ScratchArena.marker;
  final rankPtr = ScratchArena.allocate<ffi.Int>(4);
  final resolvedRcond = rcond ?? -1.0; // negative rcond uses machine precision

  try {
    int info;
    switch (a.dtype) {
      case DType.float64:
        info = LAPACKE_dgelsd(
          101, // ROW_MAJOR
          m,
          n,
          nrhs,
          aCopy.pointer.cast<ffi.Double>(),
          n,
          bCopy.pointer.cast<ffi.Double>(),
          nrhs,
          s.pointer.cast<ffi.Double>(),
          resolvedRcond,
          rankPtr,
        );
      case DType.float32:
        info = LAPACKE_sgelsd(
          101,
          m,
          n,
          nrhs,
          aCopy.pointer.cast<ffi.Float>(),
          n,
          bCopy.pointer.cast<ffi.Float>(),
          nrhs,
          s.pointer.cast<ffi.Float>(),
          resolvedRcond,
          rankPtr,
        );
      case DType.complex128:
        info = LAPACKE_zgelsd(
          101,
          m,
          n,
          nrhs,
          aCopy.pointer.cast<ffi.Double>(),
          n,
          bCopy.pointer.cast<ffi.Double>(),
          nrhs,
          s.pointer.cast<ffi.Double>(),
          resolvedRcond,
          rankPtr,
        );
      case DType.complex64:
        info = LAPACKE_cgelsd(
          101,
          m,
          n,
          nrhs,
          aCopy.pointer.cast<ffi.Float>(),
          n,
          bCopy.pointer.cast<ffi.Float>(),
          nrhs,
          s.pointer.cast<ffi.Float>(),
          resolvedRcond,
          rankPtr,
        );
      default:
        throw UnimplementedError(
          'Unsupported target DType for lstsq: ${a.dtype}',
        );
    }

    if (info < 0) {
      throw ArgumentError('Illegal value in call to LAPACKE gelsd: $info');
    }
    if (info > 0) {
      throw StateError(
        'The SVD algorithm in LAPACKE gelsd failed to converge ($info).',
      );
    }

    final rank = rankPtr.value;

    // Extract solution x: first n rows of bCopy
    final xShape = b.shape.length > 1 ? [n, nrhs] : [n];
    final x = out ?? NDArray<T>.zeros(xShape, a.dtype);
    final elementsToCopy = n * nrhs;
    x.data.setRange(0, elementsToCopy, bCopy.data.sublist(0, elementsToCopy));

    // Extract residuals: sum of squares of elements from row n to m-1 for each column
    final NDArray<double> residuals;
    if (m > n && rank == n) {
      final resShape = b.shape.length > 1 ? [nrhs] : [1];
      residuals = NDArray<double>.zeros(resShape, sDType as dynamic);
      if (a.dtype.isComplex) {
        for (var j = 0; j < nrhs; j++) {
          var sum = 0.0;
          for (var i = n; i < m; i++) {
            final complexVal = bCopy.data[i * nrhs + j] as Complex;
            sum +=
                complexVal.real * complexVal.real +
                complexVal.imag * complexVal.imag;
          }
          residuals.data[j] = sum;
        }
      } else {
        for (var j = 0; j < nrhs; j++) {
          var sum = 0.0;
          for (var i = n; i < m; i++) {
            final val = bCopy.data[i * nrhs + j] as num;
            sum += val * val;
          }
          residuals.data[j] = sum;
        }
      }
    } else {
      residuals = NDArray<double>.zeros([0], sDType as dynamic);
    }

    // Attach to scope or return
    if (out == null) {
      x.detachToParentScope();
    }
    residuals.detachToParentScope();
    s.detachToParentScope();

    return LstsqResult<T>(x: x, residuals: residuals, rank: rank, s: s);
  } finally {
    ScratchArena.reset(marker);
    aCopy.dispose();
    bCopy.dispose();
  }
}

/// Compute the cumulative sum of array elements along a specified axis.
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
/// Calculate the n-th discrete difference along the given axis.
///
/// The first difference is given by `out[i] = a[i+1] - a[i]` along the given axis.
/// Higher differences are calculated recursively.
///
/// **Preconditions:**
/// - Input [a] must not be disposed.
/// - [n] must be >= 0.
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
/// - If provided, [out] must have compatible shape and dtype.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [n] is negative.
/// - [ArgumentError] if [axis] is out of bounds.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1, 2, 4, 7, 0], [5], DType.int64);
/// final res = diff(a); // [1, 2, 3, -7]
/// ```
NDArray<T> diff<T>(NDArray<T> a, {int n = 1, int axis = -1, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute diff() on a disposed array.');
  }
  if (n < 0) {
    throw ArgumentError('Order of difference n must be >= 0 (was $n).');
  }
  if (n == 0) {
    final result = out ?? a.copy();
    if (out != null) {
      for (var i = 0; i < result.data.length; i++) {
        result.data[i] = a.data[i];
      }
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

  if (n >= a.shape[targetAxis]) {
    final emptyShape = List<int>.from(a.shape);
    emptyShape[targetAxis] = 0;
    return out ?? NDArray<T>.create(emptyShape, a.dtype);
  }

  if (n > 1) {
    final step = diff(a, n: n - 1, axis: targetAxis);
    final result = diff(step, n: 1, axis: targetAxis, out: out);
    step.dispose();
    return result;
  }

  final targetShape = List<int>.from(a.shape);
  targetShape[targetAxis] = a.shape[targetAxis] - 1;

  final result = out ?? NDArray<T>.create(targetShape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final rank = a.shape.length;
  final cShape = malloc<ffi.Int>(rank);
  final cStridesA = malloc<ffi.Int>(rank);
  final cStridesRes = malloc<ffi.Int>(rank);

  for (var i = 0; i < rank; i++) {
    cShape[i] = a.shape[i];
    cStridesA[i] = a.strides[i];
    cStridesRes[i] = result.strides[i];
  }

  try {
    final dtype = a.dtype;
    switch (dtype) {
      case DType.float64:
        s_diff_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.float32:
        s_diff_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.int64:
        s_diff_int64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.int32:
        s_diff_int32(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.complex128:
        s_diff_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.complex64:
        s_diff_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.uint8:
      case DType.int16:
      case DType.boolean:
        final doubleA = NDArray<double>.create(a.shape, DType.float64);
        unaryOp<dynamic, double>(
          doubleA.data,
          a.data,
          a.shape,
          a.strides,
          doubleA.strides,
          0,
          a.offsetElements,
          doubleA.offsetElements,
          (x) => (x as num).toDouble(),
        );
        final doubleRes = NDArray<double>.create(targetShape, DType.float64);
        final cStridesDoubleA = malloc<ffi.Int>(rank);
        final cStridesDoubleRes = malloc<ffi.Int>(rank);

        for (var i = 0; i < rank; i++) {
          cStridesDoubleA[i] = doubleA.strides[i];
          cStridesDoubleRes[i] = doubleRes.strides[i];
        }

        try {
          s_diff_double(
            doubleA.pointer.cast(),
            cStridesDoubleA,
            doubleRes.pointer.cast(),
            cStridesDoubleRes,
            cShape,
            rank,
            targetAxis,
          );
        } finally {
          malloc.free(cStridesDoubleA);
          malloc.free(cStridesDoubleRes);
        }

        for (var i = 0; i < result.data.length; i++) {
          result.data[i] = castValue(doubleRes.data[i], a.dtype) as T;
        }
        doubleA.dispose();
        doubleRes.dispose();
    }
  } finally {
    malloc.free(cShape);
    malloc.free(cStridesA);
    malloc.free(cStridesRes);
  }

  return result;
}

/// Compute the element-wise complex conjugate of the array elements.
///
/// **Preconditions:**
/// - The array must not be disposed.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([Complex(1.0, 2.0)], [1], DType.complex128);
/// final c = conj(a); // [Complex(1.0, -2.0)]
/// ```
NDArray conj(NDArray a, {NDArray? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute conj() on a disposed array.');
  }
  final targetDType = a.dtype;
  final result = out ?? NDArray.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for conj.',
      );
    }
  }

  switch (targetDType) {
    case DType.complex128:
      if (a.isContiguous && result.isContiguous) {
        v_conj_complex128(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      } else {
        final rank = a.shape.length;
        final cShape = malloc<ffi.Int>(rank);
        final cStridesA = malloc<ffi.Int>(rank);
        final cStridesRes = malloc<ffi.Int>(rank);
        for (var i = 0; i < rank; i++) {
          cShape[i] = a.shape[i];
          cStridesA[i] = a.strides[i];
          cStridesRes[i] = result.strides[i];
        }
        try {
          s_conj_complex128(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        } finally {
          malloc.free(cShape);
          malloc.free(cStridesA);
          malloc.free(cStridesRes);
        }
      }
    case DType.complex64:
      if (a.isContiguous && result.isContiguous) {
        v_conj_complex64(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      } else {
        final rank = a.shape.length;
        final cShape = malloc<ffi.Int>(rank);
        final cStridesA = malloc<ffi.Int>(rank);
        final cStridesRes = malloc<ffi.Int>(rank);
        for (var i = 0; i < rank; i++) {
          cShape[i] = a.shape[i];
          cStridesA[i] = a.strides[i];
          cStridesRes[i] = result.strides[i];
        }
        try {
          s_conj_complex64(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        } finally {
          malloc.free(cShape);
          malloc.free(cStridesA);
          malloc.free(cStridesRes);
        }
      }
    case DType.float64:
    case DType.float32:
    case DType.int64:
    case DType.int32:
    case DType.uint8:
    case DType.int16:
    case DType.boolean:
      // Real/boolean numbers are their own complex conjugates!
      if (a.isContiguous && result.isContiguous) {
        final totalSize = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
        result.data.setRange(0, totalSize, a.data);
      } else {
        result.fill(0); // initialize
        final rank = a.shape.length;
        final cShape = malloc<ffi.Int>(rank);
        final cStridesA = malloc<ffi.Int>(rank);
        final cStridesRes = malloc<ffi.Int>(rank);
        for (var i = 0; i < rank; i++) {
          cShape[i] = a.shape[i];
          cStridesA[i] = a.strides[i];
          cStridesRes[i] = result.strides[i];
        }
        try {
          switch (targetDType) {
            case DType.float64:
              s_flatten_double(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cShape,
                rank,
              );
            case DType.float32:
              s_flatten_float(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cShape,
                rank,
              );
            case DType.int64:
              s_flatten_int64(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cShape,
                rank,
              );
            case DType.int32:
              s_flatten_int32(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cShape,
                rank,
              );
            case DType.boolean:
              s_flatten_boolean(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cShape,
                rank,
              );
            default:
              // Fallback recursive copy for other strided types
              unaryOp<dynamic, dynamic>(
                result.data,
                a.data,
                a.shape,
                a.strides,
                result.strides,
                0,
                a.offsetElements,
                result.offsetElements,
                (x) => x,
              );
          }
        } finally {
          malloc.free(cShape);
          malloc.free(cStridesA);
          malloc.free(cStridesRes);
        }
      }
      return result;
  }
}

/// Alias for [conj].
NDArray conjugate(NDArray a, {NDArray? out}) => conj(a, out: out);

/// Rolls array elements along a given axis.
///
/// Elements that roll beyond the last position are re-introduced at the first.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - If [axis] is a list, [shift] must be an integer or a list of the same length.
/// - Each axis must be a valid axis index for [a] (within `[-rank, rank - 1]`).
/// - If [axis] is `null`, [shift] must be an integer.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [shift] and [axis] configurations are mismatched or invalid.
/// - [RangeError] if any axis is out of bounds.
///
/// **Performance considerations:**
/// - Algorithmic Time Complexity is $O(N)$ where $N$ is the total number of elements,
///   as it creates a new array and copies elements.
/// - Space Complexity is $O(N)$ for the newly allocated output array.
///
/// **Reference:**
/// Refer to [NumPy roll documentation](https://numpy.org/doc/stable/reference/generated/numpy.roll.html).
///
/// {@example /example/rearranging_example.dart lang=dart}
NDArray<T> roll<T extends Object>(NDArray<T> a, dynamic shift, {dynamic axis}) {
  if (a.isDisposed) {
    throw StateError('Cannot roll a disposed array.');
  }

  // Parse shifts and axes
  final List<int> shifts;
  final List<int>? axes;

  if (axis == null) {
    if (shift is int) {
      shifts = [shift];
      axes = null;
    } else if (shift is List<int>) {
      if (shift.length != 1) {
        throw ArgumentError('shift must be an integer when axis is null');
      }
      shifts = shift;
      axes = null;
    } else {
      throw ArgumentError('shift must be an integer or a list of integers');
    }
  } else if (axis is int) {
    if (shift is int) {
      shifts = [shift];
      axes = [axis];
    } else if (shift is List<int>) {
      if (shift.length != 1) {
        throw ArgumentError(
          'shift and axis must have the same number of elements',
        );
      }
      shifts = shift;
      axes = [axis];
    } else {
      throw ArgumentError('shift must be an integer or a list of integers');
    }
  } else if (axis is List<int>) {
    if (shift is int) {
      shifts = List<int>.filled(axis.length, shift);
      axes = axis;
    } else if (shift is List<int>) {
      if (shift.length != axis.length) {
        throw ArgumentError(
          'shift and axis must have the same number of elements',
        );
      }
      shifts = shift;
      axes = axis;
    } else {
      throw ArgumentError('shift must be an integer or a list of integers');
    }
  } else {
    throw ArgumentError('axis must be null, an integer, or a list of integers');
  }

  if (a.rank == 0) {
    return a.copy();
  }

  return NDArray.scope(() {
    NDArray<T> current = a;

    if (axes == null) {
      final flat = current.ravel();
      final s = shifts[0];
      final rolledFlat = _rollSingle1D(flat, s);
      final result = rolledFlat.reshape(a.shape);
      return result.detachToParentScope();
    } else {
      for (var i = 0; i < axes.length; i++) {
        current = _rollSingle(current, shifts[i], axes[i]);
      }
      return current.copy().detachToParentScope();
    }
  });
}

NDArray<T> _rollSingle1D<T extends Object>(NDArray<T> a, int shift) {
  final size = a.size;
  if (size == 0) return a.copy();
  final s = shift % size;
  if (s == 0) return a.copy();

  final realShift = s < 0 ? size + s : s;

  final part1 = a.slice([Slice(start: size - realShift, stop: size)]);
  final part2 = a.slice([Slice(start: 0, stop: size - realShift)]);
  return concatenate([part1, part2], axis: 0);
}

NDArray<T> _rollSingle<T extends Object>(NDArray<T> a, int shift, int axis) {
  final rank = a.rank;
  final normAx = axis < 0 ? rank + axis : axis;
  if (normAx < 0 || normAx >= rank) {
    throw RangeError.range(normAx, 0, rank - 1, 'axis');
  }

  final dimSize = a.shape[normAx];
  if (dimSize == 0) return a.copy();

  final s = shift % dimSize;
  if (s == 0) return a.copy();

  final realShift = s < 0 ? dimSize + s : s;

  final selectors1 = List<Selector>.generate(
    rank,
    (i) => i == normAx
        ? Slice(start: dimSize - realShift, stop: dimSize)
        : const Slice.all(),
  );
  final selectors2 = List<Selector>.generate(
    rank,
    (i) => i == normAx
        ? Slice(start: 0, stop: dimSize - realShift)
        : const Slice.all(),
  );

  final part1 = a.slice(selectors1);
  final part2 = a.slice(selectors2);

  return concatenate([part1, part2], axis: normAx);
}

/// Signature for C function strided binary operations.
typedef StridedBinaryOp =
    void Function(
      ffi.Pointer<ffi.Void> a,
      ffi.Pointer<ffi.Int> stridesA,
      ffi.Pointer<ffi.Void> b,
      ffi.Pointer<ffi.Int> stridesB,
      ffi.Pointer<ffi.Void> result,
      ffi.Pointer<ffi.Int> stridesResult,
      ffi.Pointer<ffi.Int> shape,
      int rank,
    );

/// Element-wise addition of two arrays.
///
/// Returns a new array with the promoted data type.
NDArray<R> add<Ta, Tb, R>(NDArray<Ta> a, NDArray<Tb> b, {NDArray<R>? out}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot add disposed arrays.');
  }
  final targetDType = resolveDType(a.dtype, b.dtype);
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    result = out;
  } else {
    result = NDArray<R>.create(commonShape, targetDType as DType<R>);
  }

  final resultStrides = NDArray.computeCStrides(commonShape);

  // Specialized paths for Float64 (as in original extensions.dart)
  final isContig =
      a.isContiguous &&
      b.isContiguous &&
      result.isContiguous &&
      listEquals(a.shape, b.shape);

  final ndim = commonShape.length;
  final cBuffer = ScratchArena.getStridedBuffer(ndim);
  final cShape = cBuffer;
  final cStridesA = cBuffer + ndim;
  final cStridesB = cBuffer + (ndim * 2);
  final cStridesRes = cBuffer + (ndim * 3);

  for (var i = 0; i < commonShape.length; i++) {
    cShape[i] = commonShape[i];
    cStridesA[i] = stridesA[i];
    cStridesB[i] = stridesB[i];
    cStridesRes[i] = resultStrides[i];
  }
  switch ((a.dtype, b.dtype)) {
    case (DType.float64, DType.float64) when isContig:
      v_add_double_double_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.float64):
      s_add_double_double_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.float32) when isContig:
      v_add_double_float_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.float32):
      s_add_double_float_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.int64) when isContig:
      v_add_double_int64_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.int64):
      s_add_double_int64_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.int32) when isContig:
      v_add_double_int32_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.int32):
      s_add_double_int32_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.boolean) when isContig:
    case (DType.float64, DType.uint8) when isContig:
      v_add_double_uint8_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.boolean):
    case (DType.float64, DType.uint8):
      s_add_double_uint8_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.int16) when isContig:
      v_add_double_int16_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.int16):
      s_add_double_int16_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.complex128) when isContig:
      v_add_double_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.complex128):
      s_add_double_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.complex64) when isContig:
      v_add_double_cpx64_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.complex64):
      s_add_double_cpx64_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.float64) when isContig:
      v_add_double_float_double(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.float64):
      s_add_double_float_double(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.float32) when isContig:
      v_add_float_float_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.float32):
      s_add_float_float_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.int64) when isContig:
      v_add_float_int64_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.int64):
      s_add_float_int64_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.int32) when isContig:
      v_add_float_int32_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.int32):
      s_add_float_int32_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.boolean) when isContig:
    case (DType.float32, DType.uint8) when isContig:
      v_add_float_uint8_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.boolean):
    case (DType.float32, DType.uint8):
      s_add_float_uint8_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.int16) when isContig:
      v_add_float_int16_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.int16):
      s_add_float_int16_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.complex128) when isContig:
      v_add_float_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.complex128):
      s_add_float_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.complex64) when isContig:
      v_add_float_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.complex64):
      s_add_float_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.float64) when isContig:
      v_add_double_int64_double(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.float64):
      s_add_double_int64_double(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.float32) when isContig:
      v_add_float_int64_float(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.float32):
      s_add_float_int64_float(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.int64) when isContig:
      v_add_int64_int64_int64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.int64):
      s_add_int64_int64_int64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.int32) when isContig:
      v_add_int64_int32_int64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.int32):
      s_add_int64_int32_int64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.boolean) when isContig:
    case (DType.int64, DType.uint8) when isContig:
      v_add_int64_uint8_int64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.boolean):
    case (DType.int64, DType.uint8):
      s_add_int64_uint8_int64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.int16) when isContig:
      v_add_int64_int16_int64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.int16):
      s_add_int64_int16_int64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.complex128) when isContig:
      v_add_int64_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.complex128):
      s_add_int64_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.complex64) when isContig:
      v_add_int64_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.complex64):
      s_add_int64_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.float64) when isContig:
      v_add_double_int32_double(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.float64):
      s_add_double_int32_double(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.float32) when isContig:
      v_add_float_int32_float(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.float32):
      s_add_float_int32_float(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.int64) when isContig:
      v_add_int64_int32_int64(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.int64):
      s_add_int64_int32_int64(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.int32) when isContig:
      v_add_int32_int32_int32(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.int32):
      s_add_int32_int32_int32(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.boolean) when isContig:
    case (DType.int32, DType.uint8) when isContig:
      v_add_int32_uint8_int32(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.boolean):
    case (DType.int32, DType.uint8):
      s_add_int32_uint8_int32(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.int16) when isContig:
      v_add_int32_int16_int32(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.int16):
      s_add_int32_int16_int32(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.complex128) when isContig:
      v_add_int32_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.complex128):
      s_add_int32_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.complex64) when isContig:
      v_add_int32_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.complex64):
      s_add_int32_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.float64) when isContig:
    case (DType.uint8, DType.float64) when isContig:
      v_add_double_uint8_double(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.float64):
    case (DType.uint8, DType.float64):
      s_add_double_uint8_double(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.float32) when isContig:
    case (DType.uint8, DType.float32) when isContig:
      v_add_float_uint8_float(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.float32):
    case (DType.uint8, DType.float32):
      s_add_float_uint8_float(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.int64) when isContig:
    case (DType.uint8, DType.int64) when isContig:
      v_add_int64_uint8_int64(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.int64):
    case (DType.uint8, DType.int64):
      s_add_int64_uint8_int64(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.int32) when isContig:
    case (DType.uint8, DType.int32) when isContig:
      v_add_int32_uint8_int32(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.int32):
    case (DType.uint8, DType.int32):
      s_add_int32_uint8_int32(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.boolean) when isContig:
    case (DType.boolean, DType.uint8) when isContig:
    case (DType.uint8, DType.boolean) when isContig:
    case (DType.uint8, DType.uint8) when isContig:
      v_add_uint8_uint8_uint8(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.boolean):
    case (DType.boolean, DType.uint8):
    case (DType.uint8, DType.boolean):
    case (DType.uint8, DType.uint8):
      s_add_uint8_uint8_uint8(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.int16) when isContig:
    case (DType.uint8, DType.int16) when isContig:
      v_add_uint8_int16_int16(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.int16):
    case (DType.uint8, DType.int16):
      s_add_uint8_int16_int16(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.complex128) when isContig:
    case (DType.uint8, DType.complex128) when isContig:
      v_add_uint8_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.complex128):
    case (DType.uint8, DType.complex128):
      s_add_uint8_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.complex64) when isContig:
    case (DType.uint8, DType.complex64) when isContig:
      v_add_uint8_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.complex64):
    case (DType.uint8, DType.complex64):
      s_add_uint8_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.float64) when isContig:
      v_add_double_int16_double(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.float64):
      s_add_double_int16_double(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.float32) when isContig:
      v_add_float_int16_float(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.float32):
      s_add_float_int16_float(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.int64) when isContig:
      v_add_int64_int16_int64(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.int64):
      s_add_int64_int16_int64(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.int32) when isContig:
      v_add_int32_int16_int32(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.int32):
      s_add_int32_int16_int32(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.boolean) when isContig:
    case (DType.int16, DType.uint8) when isContig:
      v_add_uint8_int16_int16(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.boolean):
    case (DType.int16, DType.uint8):
      s_add_uint8_int16_int16(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.int16) when isContig:
      v_add_int16_int16_int16(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.int16):
      s_add_int16_int16_int16(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.complex128) when isContig:
      v_add_int16_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.complex128):
      s_add_int16_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.complex64) when isContig:
      v_add_int16_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.complex64):
      s_add_int16_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.float64) when isContig:
      v_add_double_cpx_cpx(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.float64):
      s_add_double_cpx_cpx(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.float32) when isContig:
      v_add_float_cpx_cpx(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.float32):
      s_add_float_cpx_cpx(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.int64) when isContig:
      v_add_int64_cpx_cpx(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.int64):
      s_add_int64_cpx_cpx(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.int32) when isContig:
      v_add_int32_cpx_cpx(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.int32):
      s_add_int32_cpx_cpx(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.boolean) when isContig:
    case (DType.complex128, DType.uint8) when isContig:
      v_add_uint8_cpx_cpx(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.boolean):
    case (DType.complex128, DType.uint8):
      s_add_uint8_cpx_cpx(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.int16) when isContig:
      v_add_int16_cpx_cpx(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.int16):
      s_add_int16_cpx_cpx(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.complex128) when isContig:
      v_add_cpx_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.complex128):
      s_add_cpx_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.complex64) when isContig:
      v_add_cpx_cpx64_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.complex64):
      s_add_cpx_cpx64_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.float64) when isContig:
      v_add_double_cpx64_cpx(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.float64):
      s_add_double_cpx64_cpx(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.float32) when isContig:
      v_add_float_cpx64_cpx64(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.float32):
      s_add_float_cpx64_cpx64(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.int64) when isContig:
      v_add_int64_cpx64_cpx64(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.int64):
      s_add_int64_cpx64_cpx64(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.int32) when isContig:
      v_add_int32_cpx64_cpx64(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.int32):
      s_add_int32_cpx64_cpx64(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.boolean) when isContig:
    case (DType.complex64, DType.uint8) when isContig:
      v_add_uint8_cpx64_cpx64(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.boolean):
    case (DType.complex64, DType.uint8):
      s_add_uint8_cpx64_cpx64(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.int16) when isContig:
      v_add_int16_cpx64_cpx64(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.int16):
      s_add_int16_cpx64_cpx64(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.complex128) when isContig:
      v_add_cpx_cpx64_cpx(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.complex128):
      s_add_cpx_cpx64_cpx(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.complex64) when isContig:
      v_add_cpx64_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.complex64):
      s_add_cpx64_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
  }
  // ignore: dead_code
  throw UnsupportedError('Unsupported operand types');
}

/// Element-wise subtraction of two arrays.
NDArray<R> subtract<Ta, Tb, R>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<R>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot subtract disposed arrays.');
  }
  final targetDType = resolveDType(a.dtype, b.dtype);
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    result = out;
  } else {
    result = NDArray<R>.create(commonShape, targetDType as DType<R>);
  }

  final resultStrides = NDArray.computeCStrides(commonShape);

  final isContig =
      a.isContiguous &&
      b.isContiguous &&
      result.isContiguous &&
      listEquals(a.shape, b.shape);

  final ndim = commonShape.length;
  final cBuffer = ScratchArena.getStridedBuffer(ndim);
  final cShape = cBuffer;
  final cStridesA = cBuffer + ndim;
  final cStridesB = cBuffer + (ndim * 2);
  final cStridesRes = cBuffer + (ndim * 3);

  for (var i = 0; i < commonShape.length; i++) {
    cShape[i] = commonShape[i];
    cStridesA[i] = stridesA[i];
    cStridesB[i] = stridesB[i];
    cStridesRes[i] = resultStrides[i];
  }
  switch ((a.dtype, b.dtype)) {
    case (DType.float64, DType.float64) when isContig:
      v_sub_double_double_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.float64):
      s_sub_double_double_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.float32) when isContig:
      v_sub_double_float_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.float32):
      s_sub_double_float_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.int64) when isContig:
      v_sub_double_int64_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.int64):
      s_sub_double_int64_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.int32) when isContig:
      v_sub_double_int32_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.int32):
      s_sub_double_int32_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.boolean) when isContig:
    case (DType.float64, DType.uint8) when isContig:
      v_sub_double_uint8_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.boolean):
    case (DType.float64, DType.uint8):
      s_sub_double_uint8_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.int16) when isContig:
      v_sub_double_int16_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.int16):
      s_sub_double_int16_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.complex128) when isContig:
      v_sub_double_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.complex128):
      s_sub_double_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.complex64) when isContig:
      v_sub_double_cpx64_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.complex64):
      s_sub_double_cpx64_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.float64) when isContig:
      v_sub_float_double_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.float64):
      s_sub_float_double_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.float32) when isContig:
      v_sub_float_float_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.float32):
      s_sub_float_float_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.int64) when isContig:
      v_sub_float_int64_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.int64):
      s_sub_float_int64_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.int32) when isContig:
      v_sub_float_int32_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.int32):
      s_sub_float_int32_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.boolean) when isContig:
    case (DType.float32, DType.uint8) when isContig:
      v_sub_float_uint8_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.boolean):
    case (DType.float32, DType.uint8):
      s_sub_float_uint8_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.int16) when isContig:
      v_sub_float_int16_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.int16):
      s_sub_float_int16_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.complex128) when isContig:
      v_sub_float_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.complex128):
      s_sub_float_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.complex64) when isContig:
      v_sub_float_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.complex64):
      s_sub_float_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.float64) when isContig:
      v_sub_int64_double_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.float64):
      s_sub_int64_double_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.float32) when isContig:
      v_sub_int64_float_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.float32):
      s_sub_int64_float_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.int64) when isContig:
      v_sub_int64_int64_int64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.int64):
      s_sub_int64_int64_int64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.int32) when isContig:
      v_sub_int64_int32_int64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.int32):
      s_sub_int64_int32_int64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.boolean) when isContig:
    case (DType.int64, DType.uint8) when isContig:
      v_sub_int64_uint8_int64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.boolean):
    case (DType.int64, DType.uint8):
      s_sub_int64_uint8_int64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.int16) when isContig:
      v_sub_int64_int16_int64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.int16):
      s_sub_int64_int16_int64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.complex128) when isContig:
      v_sub_int64_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.complex128):
      s_sub_int64_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.complex64) when isContig:
      v_sub_int64_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.complex64):
      s_sub_int64_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.float64) when isContig:
      v_sub_int32_double_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.float64):
      s_sub_int32_double_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.float32) when isContig:
      v_sub_int32_float_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.float32):
      s_sub_int32_float_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.int64) when isContig:
      v_sub_int32_int64_int64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.int64):
      s_sub_int32_int64_int64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.int32) when isContig:
      v_sub_int32_int32_int32(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.int32):
      s_sub_int32_int32_int32(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.boolean) when isContig:
    case (DType.int32, DType.uint8) when isContig:
      v_sub_int32_uint8_int32(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.boolean):
    case (DType.int32, DType.uint8):
      s_sub_int32_uint8_int32(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.int16) when isContig:
      v_sub_int32_int16_int32(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.int16):
      s_sub_int32_int16_int32(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.complex128) when isContig:
      v_sub_int32_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.complex128):
      s_sub_int32_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.complex64) when isContig:
      v_sub_int32_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.complex64):
      s_sub_int32_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.float64) when isContig:
    case (DType.uint8, DType.float64) when isContig:
      v_sub_uint8_double_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.float64):
    case (DType.uint8, DType.float64):
      s_sub_uint8_double_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.float32) when isContig:
    case (DType.uint8, DType.float32) when isContig:
      v_sub_uint8_float_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.float32):
    case (DType.uint8, DType.float32):
      s_sub_uint8_float_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.int64) when isContig:
    case (DType.uint8, DType.int64) when isContig:
      v_sub_uint8_int64_int64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.int64):
    case (DType.uint8, DType.int64):
      s_sub_uint8_int64_int64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.int32) when isContig:
    case (DType.uint8, DType.int32) when isContig:
      v_sub_uint8_int32_int32(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.int32):
    case (DType.uint8, DType.int32):
      s_sub_uint8_int32_int32(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.boolean) when isContig:
    case (DType.boolean, DType.uint8) when isContig:
    case (DType.uint8, DType.boolean) when isContig:
    case (DType.uint8, DType.uint8) when isContig:
      v_sub_uint8_uint8_uint8(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.boolean):
    case (DType.boolean, DType.uint8):
    case (DType.uint8, DType.boolean):
    case (DType.uint8, DType.uint8):
      s_sub_uint8_uint8_uint8(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.int16) when isContig:
    case (DType.uint8, DType.int16) when isContig:
      v_sub_uint8_int16_int16(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.int16):
    case (DType.uint8, DType.int16):
      s_sub_uint8_int16_int16(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.complex128) when isContig:
    case (DType.uint8, DType.complex128) when isContig:
      v_sub_uint8_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.complex128):
    case (DType.uint8, DType.complex128):
      s_sub_uint8_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.complex64) when isContig:
    case (DType.uint8, DType.complex64) when isContig:
      v_sub_uint8_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.complex64):
    case (DType.uint8, DType.complex64):
      s_sub_uint8_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.float64) when isContig:
      v_sub_int16_double_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.float64):
      s_sub_int16_double_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.float32) when isContig:
      v_sub_int16_float_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.float32):
      s_sub_int16_float_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.int64) when isContig:
      v_sub_int16_int64_int64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.int64):
      s_sub_int16_int64_int64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.int32) when isContig:
      v_sub_int16_int32_int32(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.int32):
      s_sub_int16_int32_int32(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.boolean) when isContig:
    case (DType.int16, DType.uint8) when isContig:
      v_sub_int16_uint8_int16(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.boolean):
    case (DType.int16, DType.uint8):
      s_sub_int16_uint8_int16(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.int16) when isContig:
      v_sub_int16_int16_int16(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.int16):
      s_sub_int16_int16_int16(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.complex128) when isContig:
      v_sub_int16_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.complex128):
      s_sub_int16_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.complex64) when isContig:
      v_sub_int16_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.complex64):
      s_sub_int16_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.float64) when isContig:
      v_sub_cpx_double_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.float64):
      s_sub_cpx_double_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.float32) when isContig:
      v_sub_cpx_float_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.float32):
      s_sub_cpx_float_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.int64) when isContig:
      v_sub_cpx_int64_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.int64):
      s_sub_cpx_int64_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.int32) when isContig:
      v_sub_cpx_int32_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.int32):
      s_sub_cpx_int32_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.boolean) when isContig:
    case (DType.complex128, DType.uint8) when isContig:
      v_sub_cpx_uint8_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.boolean):
    case (DType.complex128, DType.uint8):
      s_sub_cpx_uint8_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.int16) when isContig:
      v_sub_cpx_int16_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.int16):
      s_sub_cpx_int16_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.complex128) when isContig:
      v_sub_cpx_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.complex128):
      s_sub_cpx_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.complex64) when isContig:
      v_sub_cpx_cpx64_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.complex64):
      s_sub_cpx_cpx64_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.float64) when isContig:
      v_sub_cpx64_double_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.float64):
      s_sub_cpx64_double_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.float32) when isContig:
      v_sub_cpx64_float_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.float32):
      s_sub_cpx64_float_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.int64) when isContig:
      v_sub_cpx64_int64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.int64):
      s_sub_cpx64_int64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.int32) when isContig:
      v_sub_cpx64_int32_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.int32):
      s_sub_cpx64_int32_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.boolean) when isContig:
    case (DType.complex64, DType.uint8) when isContig:
      v_sub_cpx64_uint8_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.boolean):
    case (DType.complex64, DType.uint8):
      s_sub_cpx64_uint8_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.int16) when isContig:
      v_sub_cpx64_int16_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.int16):
      s_sub_cpx64_int16_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.complex128) when isContig:
      v_sub_cpx64_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.complex128):
      s_sub_cpx64_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.complex64) when isContig:
      v_sub_cpx64_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.complex64):
      s_sub_cpx64_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
  }
  // ignore: dead_code
  throw UnsupportedError('Unsupported operand types');
}

/// Element-wise multiplication of two arrays with full broadcasting support.
///
/// **Overflow behavior:**
/// - **Integer arrays** (`int32`, `int64`, etc.) overflow silently wrapping around via standard two's complement.
/// - **Floating-point arrays** (`float32`, `float64`) overflow silently to `double.infinity` or `double.negativeInfinity` per IEEE 754.
NDArray<R> multiply<Ta, Tb, R>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<R>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot multiply disposed arrays.');
  }
  final targetDType = resolveDType(a.dtype, b.dtype);
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    result = out;
  } else {
    result = NDArray<R>.create(commonShape, targetDType as DType<R>);
  }

  final resultStrides = NDArray.computeCStrides(commonShape);

  final isContig =
      a.isContiguous &&
      b.isContiguous &&
      result.isContiguous &&
      listEquals(a.shape, b.shape);

  final ndim = commonShape.length;
  final cBuffer = ScratchArena.getStridedBuffer(ndim);
  final cShape = cBuffer;
  final cStridesA = cBuffer + ndim;
  final cStridesB = cBuffer + (ndim * 2);
  final cStridesRes = cBuffer + (ndim * 3);

  for (var i = 0; i < commonShape.length; i++) {
    cShape[i] = commonShape[i];
    cStridesA[i] = stridesA[i];
    cStridesB[i] = stridesB[i];
    cStridesRes[i] = resultStrides[i];
  }

  switch ((a.dtype, b.dtype)) {
    case (DType.float64, DType.float64) when isContig:
      v_mul_double_double_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.float64):
      s_mul_double_double_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.float32) when isContig:
      v_mul_double_float_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.float32):
      s_mul_double_float_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.int64) when isContig:
      v_mul_double_int64_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.int64):
      s_mul_double_int64_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.int32) when isContig:
      v_mul_double_int32_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.int32):
      s_mul_double_int32_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.boolean) when isContig:
    case (DType.float64, DType.uint8) when isContig:
      v_mul_double_uint8_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.boolean):
    case (DType.float64, DType.uint8):
      s_mul_double_uint8_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.int16) when isContig:
      v_mul_double_int16_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.int16):
      s_mul_double_int16_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.complex128) when isContig:
      v_mul_double_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.complex128):
      s_mul_double_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.complex64) when isContig:
      v_mul_double_cpx64_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.complex64):
      s_mul_double_cpx64_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.float64) when isContig:
      v_mul_double_float_double(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.float64):
      s_mul_double_float_double(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.float32) when isContig:
      v_mul_float_float_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.float32):
      s_mul_float_float_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.int64) when isContig:
      v_mul_float_int64_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.int64):
      s_mul_float_int64_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.int32) when isContig:
      v_mul_float_int32_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.int32):
      s_mul_float_int32_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.boolean) when isContig:
    case (DType.float32, DType.uint8) when isContig:
      v_mul_float_uint8_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.boolean):
    case (DType.float32, DType.uint8):
      s_mul_float_uint8_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.int16) when isContig:
      v_mul_float_int16_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.int16):
      s_mul_float_int16_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.complex128) when isContig:
      v_mul_float_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.complex128):
      s_mul_float_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.complex64) when isContig:
      v_mul_float_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.complex64):
      s_mul_float_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.float64) when isContig:
      v_mul_double_int64_double(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.float64):
      s_mul_double_int64_double(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.float32) when isContig:
      v_mul_float_int64_float(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.float32):
      s_mul_float_int64_float(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.int64) when isContig:
      v_mul_int64_int64_int64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.int64):
      s_mul_int64_int64_int64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.int32) when isContig:
      v_mul_int64_int32_int64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.int32):
      s_mul_int64_int32_int64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.boolean) when isContig:
    case (DType.int64, DType.uint8) when isContig:
      v_mul_int64_uint8_int64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.boolean):
    case (DType.int64, DType.uint8):
      s_mul_int64_uint8_int64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.int16) when isContig:
      v_mul_int64_int16_int64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.int16):
      s_mul_int64_int16_int64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.complex128) when isContig:
      v_mul_int64_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.complex128):
      s_mul_int64_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.complex64) when isContig:
      v_mul_int64_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.complex64):
      s_mul_int64_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.float64) when isContig:
      v_mul_double_int32_double(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.float64):
      s_mul_double_int32_double(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.float32) when isContig:
      v_mul_float_int32_float(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.float32):
      s_mul_float_int32_float(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.int64) when isContig:
      v_mul_int64_int32_int64(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.int64):
      s_mul_int64_int32_int64(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.int32) when isContig:
      v_mul_int32_int32_int32(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.int32):
      s_mul_int32_int32_int32(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.boolean) when isContig:
    case (DType.int32, DType.uint8) when isContig:
      v_mul_int32_uint8_int32(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.boolean):
    case (DType.int32, DType.uint8):
      s_mul_int32_uint8_int32(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.int16) when isContig:
      v_mul_int32_int16_int32(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.int16):
      s_mul_int32_int16_int32(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.complex128) when isContig:
      v_mul_int32_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.complex128):
      s_mul_int32_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.complex64) when isContig:
      v_mul_int32_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.complex64):
      s_mul_int32_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.float64) when isContig:
    case (DType.uint8, DType.float64) when isContig:
      v_mul_double_uint8_double(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.float64):
    case (DType.uint8, DType.float64):
      s_mul_double_uint8_double(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.float32) when isContig:
    case (DType.uint8, DType.float32) when isContig:
      v_mul_float_uint8_float(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.float32):
    case (DType.uint8, DType.float32):
      s_mul_float_uint8_float(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.int64) when isContig:
    case (DType.uint8, DType.int64) when isContig:
      v_mul_int64_uint8_int64(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.int64):
    case (DType.uint8, DType.int64):
      s_mul_int64_uint8_int64(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.int32) when isContig:
    case (DType.uint8, DType.int32) when isContig:
      v_mul_int32_uint8_int32(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.int32):
    case (DType.uint8, DType.int32):
      s_mul_int32_uint8_int32(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.boolean) when isContig:
    case (DType.boolean, DType.uint8) when isContig:
    case (DType.uint8, DType.boolean) when isContig:
    case (DType.uint8, DType.uint8) when isContig:
      v_mul_uint8_uint8_uint8(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.boolean):
    case (DType.boolean, DType.uint8):
    case (DType.uint8, DType.boolean):
    case (DType.uint8, DType.uint8):
      s_mul_uint8_uint8_uint8(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.int16) when isContig:
    case (DType.uint8, DType.int16) when isContig:
      v_mul_uint8_int16_int16(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.int16):
    case (DType.uint8, DType.int16):
      s_mul_uint8_int16_int16(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.complex128) when isContig:
    case (DType.uint8, DType.complex128) when isContig:
      v_mul_uint8_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.complex128):
    case (DType.uint8, DType.complex128):
      s_mul_uint8_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.complex64) when isContig:
    case (DType.uint8, DType.complex64) when isContig:
      v_mul_uint8_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.complex64):
    case (DType.uint8, DType.complex64):
      s_mul_uint8_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.float64) when isContig:
      v_mul_double_int16_double(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.float64):
      s_mul_double_int16_double(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.float32) when isContig:
      v_mul_float_int16_float(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.float32):
      s_mul_float_int16_float(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.int64) when isContig:
      v_mul_int64_int16_int64(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.int64):
      s_mul_int64_int16_int64(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.int32) when isContig:
      v_mul_int32_int16_int32(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.int32):
      s_mul_int32_int16_int32(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.boolean) when isContig:
    case (DType.int16, DType.uint8) when isContig:
      v_mul_uint8_int16_int16(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.boolean):
    case (DType.int16, DType.uint8):
      s_mul_uint8_int16_int16(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.int16) when isContig:
      v_mul_int16_int16_int16(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.int16):
      s_mul_int16_int16_int16(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.complex128) when isContig:
      v_mul_int16_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.complex128):
      s_mul_int16_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.complex64) when isContig:
      v_mul_int16_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.complex64):
      s_mul_int16_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.float64) when isContig:
      v_mul_double_cpx_cpx(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.float64):
      s_mul_double_cpx_cpx(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.float32) when isContig:
      v_mul_float_cpx_cpx(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.float32):
      s_mul_float_cpx_cpx(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.int64) when isContig:
      v_mul_int64_cpx_cpx(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.int64):
      s_mul_int64_cpx_cpx(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.int32) when isContig:
      v_mul_int32_cpx_cpx(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.int32):
      s_mul_int32_cpx_cpx(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.boolean) when isContig:
    case (DType.complex128, DType.uint8) when isContig:
      v_mul_uint8_cpx_cpx(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.boolean):
    case (DType.complex128, DType.uint8):
      s_mul_uint8_cpx_cpx(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.int16) when isContig:
      v_mul_int16_cpx_cpx(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.int16):
      s_mul_int16_cpx_cpx(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.complex128) when isContig:
      v_mul_cpx_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.complex128):
      s_mul_cpx_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.complex64) when isContig:
      v_mul_cpx_cpx64_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.complex64):
      s_mul_cpx_cpx64_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.float64) when isContig:
      v_mul_double_cpx64_cpx(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.float64):
      s_mul_double_cpx64_cpx(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.float32) when isContig:
      v_mul_float_cpx64_cpx64(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.float32):
      s_mul_float_cpx64_cpx64(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.int64) when isContig:
      v_mul_int64_cpx64_cpx64(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.int64):
      s_mul_int64_cpx64_cpx64(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.int32) when isContig:
      v_mul_int32_cpx64_cpx64(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.int32):
      s_mul_int32_cpx64_cpx64(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.boolean) when isContig:
    case (DType.complex64, DType.uint8) when isContig:
      v_mul_uint8_cpx64_cpx64(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.boolean):
    case (DType.complex64, DType.uint8):
      s_mul_uint8_cpx64_cpx64(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.int16) when isContig:
      v_mul_int16_cpx64_cpx64(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.int16):
      s_mul_int16_cpx64_cpx64(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.complex128) when isContig:
      v_mul_cpx_cpx64_cpx(
        b.pointer.cast(),
        a.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.complex128):
      s_mul_cpx_cpx64_cpx(
        b.pointer.cast(),
        cStridesB,
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.complex64) when isContig:
      v_mul_cpx64_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.complex64):
      s_mul_cpx64_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
  }
  // ignore: dead_code
  throw UnsupportedError('Unsupported operand types');
}

/// Element-wise division of two arrays with full broadcasting support.
///
/// Always upcasts integer operands to [DType.float64] and performs floating-point division.
///
/// **Division by Zero:**
/// Division by zero is handled silently under IEEE 754 floating-point rules:
/// - Dividing a non-zero value by zero results in `double.infinity` or `double.negativeInfinity`.
/// - Dividing zero by zero results in `double.nan`.
/// No exception is thrown.
NDArray<R> divide<Ta, Tb, R>(NDArray<Ta> a, NDArray<Tb> b, {NDArray<R>? out}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot divide disposed arrays.');
  }
  var targetDType = resolveDType(a.dtype, b.dtype);
  if (targetDType.isInteger) {
    targetDType = DType.float64;
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    result = out;
  } else {
    result = NDArray<R>.create(commonShape, targetDType as DType<R>);
  }

  final resultStrides = NDArray.computeCStrides(commonShape);

  final isContig =
      a.isContiguous &&
      b.isContiguous &&
      result.isContiguous &&
      listEquals(a.shape, b.shape);

  final ndim = commonShape.length;
  final cBuffer = ScratchArena.getStridedBuffer(ndim);
  final cShape = cBuffer;
  final cStridesA = cBuffer + ndim;
  final cStridesB = cBuffer + (ndim * 2);
  final cStridesRes = cBuffer + (ndim * 3);

  for (var i = 0; i < commonShape.length; i++) {
    cShape[i] = commonShape[i];
    cStridesA[i] = stridesA[i];
    cStridesB[i] = stridesB[i];
    cStridesRes[i] = resultStrides[i];
  }
  switch ((a.dtype, b.dtype)) {
    // DIV cases
    case (DType.float64, DType.float64) when isContig:
      v_div_double_double_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.float64):
      s_div_double_double_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.float32) when isContig:
      v_div_double_float_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.float32):
      s_div_double_float_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.int64) when isContig:
      v_div_double_int64_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.int64):
      s_div_double_int64_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.int32) when isContig:
      v_div_double_int32_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.int32):
      s_div_double_int32_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.boolean) when isContig:
    case (DType.float64, DType.uint8) when isContig:
      v_div_double_uint8_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.boolean):
    case (DType.float64, DType.uint8):
      s_div_double_uint8_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.int16) when isContig:
      v_div_double_int16_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.int16):
      s_div_double_int16_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.complex128) when isContig:
      v_div_double_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.complex128):
      s_div_double_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float64, DType.complex64) when isContig:
      v_div_double_cpx64_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float64, DType.complex64):
      s_div_double_cpx64_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.float64) when isContig:
      v_div_float_double_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.float64):
      s_div_float_double_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.float32) when isContig:
      v_div_float_float_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.float32):
      s_div_float_float_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.int64) when isContig:
      v_div_float_int64_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.int64):
      s_div_float_int64_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.int32) when isContig:
      v_div_float_int32_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.int32):
      s_div_float_int32_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.boolean) when isContig:
    case (DType.float32, DType.uint8) when isContig:
      v_div_float_uint8_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.boolean):
    case (DType.float32, DType.uint8):
      s_div_float_uint8_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.int16) when isContig:
      v_div_float_int16_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.int16):
      s_div_float_int16_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.complex128) when isContig:
      v_div_float_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.complex128):
      s_div_float_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.float32, DType.complex64) when isContig:
      v_div_float_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.float32, DType.complex64):
      s_div_float_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.float64) when isContig:
      v_div_int64_double_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.float64):
      s_div_int64_double_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.float32) when isContig:
      v_div_int64_float_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.float32):
      s_div_int64_float_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.int64) when isContig:
      v_div_int64_int64_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.int64):
      s_div_int64_int64_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.int32) when isContig:
      v_div_int64_int32_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.int32):
      s_div_int64_int32_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.boolean) when isContig:
    case (DType.int64, DType.uint8) when isContig:
      v_div_int64_uint8_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.boolean):
    case (DType.int64, DType.uint8):
      s_div_int64_uint8_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.int16) when isContig:
      v_div_int64_int16_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.int16):
      s_div_int64_int16_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.complex128) when isContig:
      v_div_int64_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.complex128):
      s_div_int64_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int64, DType.complex64) when isContig:
      v_div_int64_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int64, DType.complex64):
      s_div_int64_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.float64) when isContig:
      v_div_int32_double_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.float64):
      s_div_int32_double_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.float32) when isContig:
      v_div_int32_float_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.float32):
      s_div_int32_float_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.int64) when isContig:
      v_div_int32_int64_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.int64):
      s_div_int32_int64_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.int32) when isContig:
      v_div_int32_int32_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.int32):
      s_div_int32_int32_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.boolean) when isContig:
    case (DType.int32, DType.uint8) when isContig:
      v_div_int32_uint8_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.boolean):
    case (DType.int32, DType.uint8):
      s_div_int32_uint8_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.int16) when isContig:
      v_div_int32_int16_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.int16):
      s_div_int32_int16_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.complex128) when isContig:
      v_div_int32_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.complex128):
      s_div_int32_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int32, DType.complex64) when isContig:
      v_div_int32_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int32, DType.complex64):
      s_div_int32_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.float64) when isContig:
    case (DType.uint8, DType.float64) when isContig:
      v_div_uint8_double_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.float64):
    case (DType.uint8, DType.float64):
      s_div_uint8_double_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.float32) when isContig:
    case (DType.uint8, DType.float32) when isContig:
      v_div_uint8_float_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.float32):
    case (DType.uint8, DType.float32):
      s_div_uint8_float_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.int64) when isContig:
    case (DType.uint8, DType.int64) when isContig:
      v_div_uint8_int64_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.int64):
    case (DType.uint8, DType.int64):
      s_div_uint8_int64_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.int32) when isContig:
    case (DType.uint8, DType.int32) when isContig:
      v_div_uint8_int32_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.int32):
    case (DType.uint8, DType.int32):
      s_div_uint8_int32_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.boolean) when isContig:
    case (DType.boolean, DType.uint8) when isContig:
    case (DType.uint8, DType.boolean) when isContig:
    case (DType.uint8, DType.uint8) when isContig:
      v_div_uint8_uint8_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.boolean):
    case (DType.boolean, DType.uint8):
    case (DType.uint8, DType.boolean):
    case (DType.uint8, DType.uint8):
      s_div_uint8_uint8_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.int16) when isContig:
    case (DType.uint8, DType.int16) when isContig:
      v_div_uint8_int16_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.int16):
    case (DType.uint8, DType.int16):
      s_div_uint8_int16_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.complex128) when isContig:
    case (DType.uint8, DType.complex128) when isContig:
      v_div_uint8_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.complex128):
    case (DType.uint8, DType.complex128):
      s_div_uint8_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.boolean, DType.complex64) when isContig:
    case (DType.uint8, DType.complex64) when isContig:
      v_div_uint8_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.boolean, DType.complex64):
    case (DType.uint8, DType.complex64):
      s_div_uint8_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.float64) when isContig:
      v_div_int16_double_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.float64):
      s_div_int16_double_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.float32) when isContig:
      v_div_int16_float_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.float32):
      s_div_int16_float_float(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.int64) when isContig:
      v_div_int16_int64_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.int64):
      s_div_int16_int64_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.int32) when isContig:
      v_div_int16_int32_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.int32):
      s_div_int16_int32_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.boolean) when isContig:
    case (DType.int16, DType.uint8) when isContig:
      v_div_int16_uint8_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.boolean):
    case (DType.int16, DType.uint8):
      s_div_int16_uint8_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.int16) when isContig:
      v_div_int16_int16_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.int16):
      s_div_int16_int16_double(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.complex128) when isContig:
      v_div_int16_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.complex128):
      s_div_int16_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.int16, DType.complex64) when isContig:
      v_div_int16_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.int16, DType.complex64):
      s_div_int16_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.float64) when isContig:
      v_div_cpx_double_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.float64):
      s_div_cpx_double_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.float32) when isContig:
      v_div_cpx_float_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.float32):
      s_div_cpx_float_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.int64) when isContig:
      v_div_cpx_int64_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.int64):
      s_div_cpx_int64_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.int32) when isContig:
      v_div_cpx_int32_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.int32):
      s_div_cpx_int32_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.boolean) when isContig:
    case (DType.complex128, DType.uint8) when isContig:
      v_div_cpx_uint8_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.boolean):
    case (DType.complex128, DType.uint8):
      s_div_cpx_uint8_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.int16) when isContig:
      v_div_cpx_int16_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.int16):
      s_div_cpx_int16_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.complex128) when isContig:
      v_div_cpx_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.complex128):
      s_div_cpx_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex128, DType.complex64) when isContig:
      v_div_cpx_cpx64_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex128, DType.complex64):
      s_div_cpx_cpx64_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.float64) when isContig:
      v_div_cpx64_double_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.float64):
      s_div_cpx64_double_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.float32) when isContig:
      v_div_cpx64_float_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.float32):
      s_div_cpx64_float_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.int64) when isContig:
      v_div_cpx64_int64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.int64):
      s_div_cpx64_int64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.int32) when isContig:
      v_div_cpx64_int32_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.int32):
      s_div_cpx64_int32_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.boolean) when isContig:
    case (DType.complex64, DType.uint8) when isContig:
      v_div_cpx64_uint8_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.boolean):
    case (DType.complex64, DType.uint8):
      s_div_cpx64_uint8_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.int16) when isContig:
      v_div_cpx64_int16_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.int16):
      s_div_cpx64_int16_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.complex128) when isContig:
      v_div_cpx64_cpx_cpx(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.complex128):
      s_div_cpx64_cpx_cpx(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
    case (DType.complex64, DType.complex64) when isContig:
      v_div_cpx64_cpx64_cpx64(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.size,
      );
      return result;
    case (DType.complex64, DType.complex64):
      s_div_cpx64_cpx64_cpx64(
        a.pointer.cast(),
        cStridesA,
        b.pointer.cast(),
        cStridesB,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        commonShape.length,
      );
      return result;
  }
  // ignore: dead_code
  throw UnsupportedError('Unsupported operand types');
}

/// Compute the bitwise AND of two arrays, element-wise.
///
/// Calculates the bitwise AND of two integer arrays, element-wise.
///
/// **Preconditions:**
/// - [a] and [b] must be integer-typed arrays (`int32`, `int64`, `uint8`, `int16`).
/// - [a] and [b] must not be disposed.
/// - [a] and [b] must be broadcast-compatible.
/// - If provided, [out] must match the broadcasted shape and resolved integer dtype.
///
/// **Throws:**
/// - [StateError] if [a] or [b] is disposed.
/// - [ArgumentError] if [a] or [b] is not integer-typed.
/// - [ArgumentError] if shapes are incompatible for broadcasting.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For contiguous layouts, offloads the loop directly to optimized native C vector bitwise kernels, bypassing Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy bitwise_and](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_and.html)
NDArray<Tr> bitwise_and<Ta, Tb, Tr>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<Tr>? out,
}) {
  final prep = _prepareBinaryBitwise<Ta, Tb, Tr>(a, b, out, 'bitwise_and');
  final aCast = prep.aCast;
  final bCast = prep.bCast;
  final result = prep.result;

  try {
    if (prep.isContig) {
      final size = aCast.size;
      switch (result.dtype) {
        case DType.int32:
          v_bitwise_and_int32(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int64:
          v_bitwise_and_int64(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.uint8:
          v_bitwise_and_uint8(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int16:
          v_bitwise_and_int16(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    } else {
      final rank = prep.commonShape.length;
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesB = cBuffer + (rank * 2);
      final cStridesRes = cBuffer + (rank * 3);

      for (var i = 0; i < rank; i++) {
        cShape[i] = prep.commonShape[i];
        cStridesA[i] = prep.stridesA[i];
        cStridesB[i] = prep.stridesB[i];
        cStridesRes[i] = prep.resultStrides[i];
      }

      switch (result.dtype) {
        case DType.int32:
          s_bitwise_and_int32(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int64:
          s_bitwise_and_int64(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.uint8:
          s_bitwise_and_uint8(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int16:
          s_bitwise_and_int16(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    }
  } finally {
    if (aCast != a) {
      aCast.dispose();
    }
    if (bCast != b) {
      bCast.dispose();
    }
  }

  return result;
}

/// Compute the bitwise OR of two arrays, element-wise.
///
/// Calculates the bitwise OR of two integer arrays, element-wise.
///
/// **Preconditions:**
/// - [a] and [b] must be integer-typed arrays (`int32`, `int64`, `uint8`, `int16`).
/// - [a] and [b] must not be disposed.
/// - [a] and [b] must be broadcast-compatible.
/// - If provided, [out] must match the broadcasted shape and resolved integer dtype.
///
/// **Throws:**
/// - [StateError] if [a] or [b] is disposed.
/// - [ArgumentError] if [a] or [b] is not integer-typed.
/// - [ArgumentError] if shapes are incompatible for broadcasting.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For contiguous layouts, offloads the loop directly to optimized native C vector bitwise kernels, bypassing Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy bitwise_or](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_or.html)
NDArray<Tr> bitwise_or<Ta, Tb, Tr>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<Tr>? out,
}) {
  final prep = _prepareBinaryBitwise<Ta, Tb, Tr>(a, b, out, 'bitwise_or');
  final aCast = prep.aCast;
  final bCast = prep.bCast;
  final result = prep.result;

  try {
    if (prep.isContig) {
      final size = aCast.size;
      switch (result.dtype) {
        case DType.int32:
          v_bitwise_or_int32(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int64:
          v_bitwise_or_int64(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.uint8:
          v_bitwise_or_uint8(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int16:
          v_bitwise_or_int16(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    } else {
      final rank = prep.commonShape.length;
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesB = cBuffer + (rank * 2);
      final cStridesRes = cBuffer + (rank * 3);

      for (var i = 0; i < rank; i++) {
        cShape[i] = prep.commonShape[i];
        cStridesA[i] = prep.stridesA[i];
        cStridesB[i] = prep.stridesB[i];
        cStridesRes[i] = prep.resultStrides[i];
      }

      switch (result.dtype) {
        case DType.int32:
          s_bitwise_or_int32(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int64:
          s_bitwise_or_int64(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.uint8:
          s_bitwise_or_uint8(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int16:
          s_bitwise_or_int16(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    }
  } finally {
    if (aCast != a) {
      aCast.dispose();
    }
    if (bCast != b) {
      bCast.dispose();
    }
  }

  return result;
}

/// Compute the bitwise XOR of two arrays, element-wise.
///
/// Calculates the bitwise XOR of two integer arrays, element-wise.
///
/// **Preconditions:**
/// - [a] and [b] must be integer-typed arrays (`int32`, `int64`, `uint8`, `int16`).
/// - [a] and [b] must not be disposed.
/// - [a] and [b] must be broadcast-compatible.
/// - If provided, [out] must match the broadcasted shape and resolved integer dtype.
///
/// **Throws:**
/// - [StateError] if [a] or [b] is disposed.
/// - [ArgumentError] if [a] or [b] is not integer-typed.
/// - [ArgumentError] if shapes are incompatible for broadcasting.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For contiguous layouts, offloads the loop directly to optimized native C vector bitwise kernels, bypassing Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy bitwise_xor](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_xor.html)
NDArray<Tr> bitwise_xor<Ta, Tb, Tr>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<Tr>? out,
}) {
  final prep = _prepareBinaryBitwise<Ta, Tb, Tr>(a, b, out, 'bitwise_xor');
  final aCast = prep.aCast;
  final bCast = prep.bCast;
  final result = prep.result;

  try {
    if (prep.isContig) {
      final size = aCast.size;
      switch (result.dtype) {
        case DType.int32:
          v_bitwise_xor_int32(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int64:
          v_bitwise_xor_int64(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.uint8:
          v_bitwise_xor_uint8(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int16:
          v_bitwise_xor_int16(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    } else {
      final rank = prep.commonShape.length;
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesB = cBuffer + (rank * 2);
      final cStridesRes = cBuffer + (rank * 3);

      for (var i = 0; i < rank; i++) {
        cShape[i] = prep.commonShape[i];
        cStridesA[i] = prep.stridesA[i];
        cStridesB[i] = prep.stridesB[i];
        cStridesRes[i] = prep.resultStrides[i];
      }

      switch (result.dtype) {
        case DType.int32:
          s_bitwise_xor_int32(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int64:
          s_bitwise_xor_int64(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.uint8:
          s_bitwise_xor_uint8(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int16:
          s_bitwise_xor_int16(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    }
  } finally {
    if (aCast != a) {
      aCast.dispose();
    }
    if (bCast != b) {
      bCast.dispose();
    }
  }

  return result;
}

/// Shift the bits of an integer to the left, element-wise.
///
/// Bits are shifted to the left by appending 0s at the right.
///
/// **Preconditions:**
/// - [a] and [b] must be integer-typed arrays (`int32`, `int64`, `uint8`, `int16`).
/// - [a] and [b] must not be disposed.
/// - [a] and [b] must be broadcast-compatible.
/// - If provided, [out] must match the broadcasted shape and resolved integer dtype.
///
/// **Throws:**
/// - [StateError] if [a] or [b] is disposed.
/// - [ArgumentError] if [a] or [b] is not integer-typed.
/// - [ArgumentError] if shapes are incompatible for broadcasting.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For contiguous layouts, offloads the loop directly to optimized native C vector bitwise kernels, bypassing Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy left_shift](https://numpy.org/doc/stable/reference/generated/numpy.left_shift.html)
NDArray<Tr> left_shift<Ta, Tb, Tr>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<Tr>? out,
}) {
  final prep = _prepareBinaryBitwise<Ta, Tb, Tr>(a, b, out, 'left_shift');
  final aCast = prep.aCast;
  final bCast = prep.bCast;
  final result = prep.result;

  try {
    if (prep.isContig) {
      final size = aCast.size;
      switch (result.dtype) {
        case DType.int32:
          v_left_shift_int32(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int64:
          v_left_shift_int64(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.uint8:
          v_left_shift_uint8(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int16:
          v_left_shift_int16(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    } else {
      final rank = prep.commonShape.length;
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesB = cBuffer + (rank * 2);
      final cStridesRes = cBuffer + (rank * 3);

      for (var i = 0; i < rank; i++) {
        cShape[i] = prep.commonShape[i];
        cStridesA[i] = prep.stridesA[i];
        cStridesB[i] = prep.stridesB[i];
        cStridesRes[i] = prep.resultStrides[i];
      }

      switch (result.dtype) {
        case DType.int32:
          s_left_shift_int32(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int64:
          s_left_shift_int64(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.uint8:
          s_left_shift_uint8(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int16:
          s_left_shift_int16(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    }
  } finally {
    if (aCast != a) {
      aCast.dispose();
    }
    if (bCast != b) {
      bCast.dispose();
    }
  }

  return result;
}

/// Shift the bits of an integer to the right, element-wise.
///
/// Bits are shifted to the right.
///
/// **Preconditions:**
/// - [a] and [b] must be integer-typed arrays (`int32`, `int64`, `uint8`, `int16`).
/// - [a] and [b] must not be disposed.
/// - [a] and [b] must be broadcast-compatible.
/// - If provided, [out] must match the broadcasted shape and resolved integer dtype.
///
/// **Throws:**
/// - [StateError] if [a] or [b] is disposed.
/// - [ArgumentError] if [a] or [b] is not integer-typed.
/// - [ArgumentError] if shapes are incompatible for broadcasting.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For contiguous layouts, offloads the loop directly to optimized native C vector bitwise kernels, bypassing Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy right_shift](https://numpy.org/doc/stable/reference/generated/numpy.right_shift.html)
NDArray<Tr> right_shift<Ta, Tb, Tr>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<Tr>? out,
}) {
  final prep = _prepareBinaryBitwise<Ta, Tb, Tr>(a, b, out, 'right_shift');
  final aCast = prep.aCast;
  final bCast = prep.bCast;
  final result = prep.result;

  try {
    if (prep.isContig) {
      final size = aCast.size;
      switch (result.dtype) {
        case DType.int32:
          v_right_shift_int32(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int64:
          v_right_shift_int64(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.uint8:
          v_right_shift_uint8(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int16:
          v_right_shift_int16(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    } else {
      final rank = prep.commonShape.length;
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesB = cBuffer + (rank * 2);
      final cStridesRes = cBuffer + (rank * 3);

      for (var i = 0; i < rank; i++) {
        cShape[i] = prep.commonShape[i];
        cStridesA[i] = prep.stridesA[i];
        cStridesB[i] = prep.stridesB[i];
        cStridesRes[i] = prep.resultStrides[i];
      }

      switch (result.dtype) {
        case DType.int32:
          s_right_shift_int32(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int64:
          s_right_shift_int64(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.uint8:
          s_right_shift_uint8(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int16:
          s_right_shift_int16(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    }
  } finally {
    if (aCast != a) {
      aCast.dispose();
    }
    if (bCast != b) {
      bCast.dispose();
    }
  }

  return result;
}

/// Compute bitwise inversion, or bitwise NOT, element-wise.
///
/// Calculates the bitwise NOT of an integer array, element-wise.
///
/// **Preconditions:**
/// - [a] must be an integer-typed array (`int32`, `int64`, `uint8`, `int16`).
/// - [a] must not be disposed.
/// - If provided, [out] must match the shape and dtype of [a].
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [a] is not integer-typed.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For contiguous layouts, offloads the loop directly to optimized native C vector bitwise kernels, bypassing Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy invert](https://numpy.org/doc/stable/reference/generated/numpy.invert.html)
NDArray<Tr> invert<Ta, Tr>(NDArray<Ta> a, {NDArray<Tr>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot perform invert on a disposed array.');
  }

  if (!a.dtype.isInteger) {
    throw ArgumentError(
      'Bitwise operations are only supported for integer data types.',
    );
  }

  final NDArray<Tr> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for invert.',
      );
    }
    result = out;
  } else {
    result = NDArray<Tr>.create(a.shape, a.dtype as DType<Tr>);
  }

  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.isContiguous && result.isContiguous) {
    final size = a.size;
    switch (a.dtype) {
      case DType.int32:
        v_invert_int32(a.pointer.cast(), result.pointer.cast(), size);
      case DType.int64:
        v_invert_int64(a.pointer.cast(), result.pointer.cast(), size);
      case DType.uint8:
        v_invert_uint8(a.pointer.cast(), result.pointer.cast(), size);
      case DType.int16:
        v_invert_int16(a.pointer.cast(), result.pointer.cast(), size);
      default:
        throw UnsupportedError('Unsupported integer DType: ${a.dtype}');
    }
  } else {
    final rank = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesSrc = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);

    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesSrc[i] = a.strides[i];
      cStridesRes[i] = resultStrides[i];
    }

    switch (a.dtype) {
      case DType.int32:
        s_invert_int32(
          a.pointer.cast(),
          cStridesSrc,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
      case DType.int64:
        s_invert_int64(
          a.pointer.cast(),
          cStridesSrc,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
      case DType.uint8:
        s_invert_uint8(
          a.pointer.cast(),
          cStridesSrc,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
      case DType.int16:
        s_invert_int16(
          a.pointer.cast(),
          cStridesSrc,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
      default:
        throw UnsupportedError('Unsupported integer DType: ${a.dtype}');
    }
  }

  return result;
}

({
  NDArray aCast,
  NDArray bCast,
  NDArray<Tr> result,
  List<int> commonShape,
  List<int> stridesA,
  List<int> stridesB,
  List<int> resultStrides,
  bool isContig,
})
_prepareBinaryBitwise<Ta, Tb, Tr>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  NDArray<Tr>? out,
  String opName,
) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot perform $opName on disposed arrays.');
  }

  if (!a.dtype.isInteger || !b.dtype.isInteger) {
    throw ArgumentError(
      'Bitwise operations are only supported for integer data types.',
    );
  }

  final DType targetDType = resolveDType(a.dtype, b.dtype);

  // Upcast inputs if they do not match the resolved target integer type
  NDArray aCast = a;
  if (a.dtype != targetDType) {
    aCast = NDArray.fromList(a.toList(), a.shape, targetDType);
  }

  NDArray bCast = b;
  if (b.dtype != targetDType) {
    bCast = NDArray.fromList(b.toList(), b.shape, targetDType);
  }

  final broadcastResult = broadcast(aCast, bCast);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final NDArray<Tr> result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for $opName.',
      );
    }
    result = out;
  } else {
    result = NDArray<Tr>.create(commonShape, targetDType as DType<Tr>);
  }

  final resultStrides = NDArray.computeCStrides(commonShape);

  final isContig =
      aCast.isContiguous &&
      bCast.isContiguous &&
      result.isContiguous &&
      listEquals(aCast.shape, bCast.shape);

  return (
    aCast: aCast,
    bCast: bCast,
    result: result,
    commonShape: commonShape,
    stridesA: stridesA,
    stridesB: stridesB,
    resultStrides: resultStrides,
    isContig: isContig,
  );
}

/// Kronecker product of two arrays.
///
/// Computes the Kronecker product, a composite matrix of the two input arrays,
/// which is defined as the block matrix where each element of [a] is multiplied by the entire array [b].
///
/// **Preconditions:**
/// - Both arrays [a] and [b] must not be disposed.
/// - If provided, the recycler [out] must have the correct shape and dtype.
///
/// **Throws:**
/// - [StateError] if [a] or [b] is disposed.
/// - [ArgumentError] if [out] has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N_a \times N_b)$ using optimized sequential native heap FFI memory sweeps.
///
/// **Example:**
/// {@example /example/linalg_advanced_example.dart lang=dart}
///
/// Reference: [NumPy kron](https://numpy.org/doc/stable/reference/generated/numpy.kron.html)
NDArray<R> kron<Ta, Tb, R>(NDArray<Ta> a, NDArray<Tb> b, {NDArray<R>? out}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute kron() on disposed arrays.');
  }

  final rankA = a.rank;
  final rankB = b.rank;
  final maxRank = math.max(rankA, rankB);

  final paddedShapeA = List<int>.filled(maxRank, 1);
  final paddedStridesA = List<int>.filled(maxRank, 0);
  for (var i = 0; i < rankA; i++) {
    paddedShapeA[maxRank - rankA + i] = a.shape[i];
    paddedStridesA[maxRank - rankA + i] = a.strides[i];
  }

  final paddedShapeB = List<int>.filled(maxRank, 1);
  final paddedStridesB = List<int>.filled(maxRank, 0);
  for (var i = 0; i < rankB; i++) {
    paddedShapeB[maxRank - rankB + i] = b.shape[i];
    paddedStridesB[maxRank - rankB + i] = b.strides[i];
  }

  final expectedShape = List<int>.filled(maxRank, 0);
  for (var i = 0; i < maxRank; i++) {
    expectedShape[i] = paddedShapeA[i] * paddedShapeB[i];
  }

  final targetDType = resolveDType(a.dtype, b.dtype);
  if (out != null) {
    if (!listEquals(out.shape, expectedShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out recycler has incompatible shape or dtype (expected shape $expectedShape and dtype $targetDType).',
      );
    }
  }

  final result =
      out ?? NDArray<R>.create(expectedShape, targetDType as DType<R>);

  final aCast = castNDArray(a, targetDType);
  final bCast = castNDArray(b, targetDType);

  final marker = ScratchArena.marker;
  final cStridesA = ScratchArena.allocate<ffi.Int>(
    maxRank * ffi.sizeOf<ffi.Int>(),
  );
  final cShapeA = ScratchArena.allocate<ffi.Int>(
    maxRank * ffi.sizeOf<ffi.Int>(),
  );
  final cStridesB = ScratchArena.allocate<ffi.Int>(
    maxRank * ffi.sizeOf<ffi.Int>(),
  );
  final cShapeB = ScratchArena.allocate<ffi.Int>(
    maxRank * ffi.sizeOf<ffi.Int>(),
  );
  final cStridesRes = ScratchArena.allocate<ffi.Int>(
    maxRank * ffi.sizeOf<ffi.Int>(),
  );
  final cShapeRes = ScratchArena.allocate<ffi.Int>(
    maxRank * ffi.sizeOf<ffi.Int>(),
  );

  for (var i = 0; i < maxRank; i++) {
    cStridesA[i] = paddedStridesA[i];
    cShapeA[i] = paddedShapeA[i];
    cStridesB[i] = paddedStridesB[i];
    cShapeB[i] = paddedShapeB[i];
    cStridesRes[i] = result.strides[i];
    cShapeRes[i] = result.shape[i];
  }

  try {
    switch (targetDType) {
      case DType.float64:
        s_kron_double(
          aCast.pointer.cast(),
          cStridesA,
          cShapeA,
          bCast.pointer.cast(),
          cStridesB,
          cShapeB,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          maxRank,
        );
      case DType.float32:
        s_kron_float(
          aCast.pointer.cast(),
          cStridesA,
          cShapeA,
          bCast.pointer.cast(),
          cStridesB,
          cShapeB,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          maxRank,
        );
      case DType.int64:
        s_kron_int64(
          aCast.pointer.cast(),
          cStridesA,
          cShapeA,
          bCast.pointer.cast(),
          cStridesB,
          cShapeB,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          maxRank,
        );
      case DType.int32:
        s_kron_int32(
          aCast.pointer.cast(),
          cStridesA,
          cShapeA,
          bCast.pointer.cast(),
          cStridesB,
          cShapeB,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          maxRank,
        );
      case DType.uint8:
        s_kron_uint8(
          aCast.pointer.cast(),
          cStridesA,
          cShapeA,
          bCast.pointer.cast(),
          cStridesB,
          cShapeB,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          maxRank,
        );
      case DType.int16:
        s_kron_int16(
          aCast.pointer.cast(),
          cStridesA,
          cShapeA,
          bCast.pointer.cast(),
          cStridesB,
          cShapeB,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          maxRank,
        );
      case DType.complex128:
        s_kron_complex128(
          aCast.pointer.cast(),
          cStridesA,
          cShapeA,
          bCast.pointer.cast(),
          cStridesB,
          cShapeB,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          maxRank,
        );
      case DType.complex64:
        s_kron_complex64(
          aCast.pointer.cast(),
          cStridesA,
          cShapeA,
          bCast.pointer.cast(),
          cStridesB,
          cShapeB,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          maxRank,
        );
      case DType.boolean:
        s_kron_boolean(
          aCast.pointer.cast(),
          cStridesA,
          cShapeA,
          bCast.pointer.cast(),
          cStridesB,
          cShapeB,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          maxRank,
        );
    }
  } finally {
    ScratchArena.reset(marker);
    if (aCast != a) aCast.dispose();
    if (bCast != b) bCast.dispose();
  }

  if (out == null) {
    result.detachToParentScope();
  }
  return result;
}

/// Compute the outer product of two vectors.
///
/// Given two input vectors [a] and [b], computes the outer product matrix:
/// `res[i, j] = a[i] * b[j]`.
/// If the input arrays are not 1-dimensional, they are flattened first.
///
/// **Preconditions:**
/// - Both arrays [a] and [b] must not be disposed.
/// - If provided, the [out] recycler must have shape `[size(a), size(b)]` and the correct dtype.
///
/// **Throws:**
/// - [StateError] if [a] or [b] is disposed.
/// - [ArgumentError] if [out] has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N_a \times N_b)$ using highly optimized native strided loops.
///
/// **Example:**
/// {@example /example/linalg_advanced_example.dart lang=dart}
///
/// Reference: [NumPy outer](https://numpy.org/doc/stable/reference/generated/numpy.outer.html)
NDArray<R> outer<Ta, Tb, R>(NDArray<Ta> a, NDArray<Tb> b, {NDArray<R>? out}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute outer() on disposed arrays.');
  }

  final sizeA = a.size;
  final sizeB = b.size;
  final expectedShape = [sizeA, sizeB];
  final targetDType = resolveDType(a.dtype, b.dtype);

  if (out != null) {
    if (!listEquals(out.shape, expectedShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out recycler has incompatible shape or dtype (expected shape $expectedShape and dtype $targetDType).',
      );
    }
  }

  final result =
      out ?? NDArray<R>.create(expectedShape, targetDType as DType<R>);

  final flatA = a.rank == 1 ? a : a.ravel();
  final flatB = b.rank == 1 ? b : b.ravel();

  final aCast = castNDArray(flatA, targetDType);
  final bCast = castNDArray(flatB, targetDType);

  try {
    switch (targetDType) {
      case DType.float64:
        s_outer_double(
          aCast.pointer.cast(),
          aCast.strides.isEmpty ? 1 : aCast.strides[0],
          sizeA,
          bCast.pointer.cast(),
          bCast.strides.isEmpty ? 1 : bCast.strides[0],
          sizeB,
          result.pointer.cast(),
          result.strides[0],
          result.strides[1],
        );
      case DType.float32:
        s_outer_float(
          aCast.pointer.cast(),
          aCast.strides.isEmpty ? 1 : aCast.strides[0],
          sizeA,
          bCast.pointer.cast(),
          bCast.strides.isEmpty ? 1 : bCast.strides[0],
          sizeB,
          result.pointer.cast(),
          result.strides[0],
          result.strides[1],
        );
      case DType.int64:
        s_outer_int64(
          aCast.pointer.cast(),
          aCast.strides.isEmpty ? 1 : aCast.strides[0],
          sizeA,
          bCast.pointer.cast(),
          bCast.strides.isEmpty ? 1 : bCast.strides[0],
          sizeB,
          result.pointer.cast(),
          result.strides[0],
          result.strides[1],
        );
      case DType.int32:
        s_outer_int32(
          aCast.pointer.cast(),
          aCast.strides.isEmpty ? 1 : aCast.strides[0],
          sizeA,
          bCast.pointer.cast(),
          bCast.strides.isEmpty ? 1 : bCast.strides[0],
          sizeB,
          result.pointer.cast(),
          result.strides[0],
          result.strides[1],
        );
      case DType.uint8:
        s_outer_uint8(
          aCast.pointer.cast(),
          aCast.strides.isEmpty ? 1 : aCast.strides[0],
          sizeA,
          bCast.pointer.cast(),
          bCast.strides.isEmpty ? 1 : bCast.strides[0],
          sizeB,
          result.pointer.cast(),
          result.strides[0],
          result.strides[1],
        );
      case DType.int16:
        s_outer_int16(
          aCast.pointer.cast(),
          aCast.strides.isEmpty ? 1 : aCast.strides[0],
          sizeA,
          bCast.pointer.cast(),
          bCast.strides.isEmpty ? 1 : bCast.strides[0],
          sizeB,
          result.pointer.cast(),
          result.strides[0],
          result.strides[1],
        );
      case DType.complex128:
        s_outer_complex128(
          aCast.pointer.cast(),
          aCast.strides.isEmpty ? 1 : aCast.strides[0],
          sizeA,
          bCast.pointer.cast(),
          bCast.strides.isEmpty ? 1 : bCast.strides[0],
          sizeB,
          result.pointer.cast(),
          result.strides[0],
          result.strides[1],
        );
      case DType.complex64:
        s_outer_complex64(
          aCast.pointer.cast(),
          aCast.strides.isEmpty ? 1 : aCast.strides[0],
          sizeA,
          bCast.pointer.cast(),
          bCast.strides.isEmpty ? 1 : bCast.strides[0],
          sizeB,
          result.pointer.cast(),
          result.strides[0],
          result.strides[1],
        );
      case DType.boolean:
        s_outer_boolean(
          aCast.pointer.cast(),
          aCast.strides.isEmpty ? 1 : aCast.strides[0],
          sizeA,
          bCast.pointer.cast(),
          bCast.strides.isEmpty ? 1 : bCast.strides[0],
          sizeB,
          result.pointer.cast(),
          result.strides[0],
          result.strides[1],
        );
    }
  } finally {
    if (flatA != a) flatA.dispose();
    if (flatB != b) flatB.dispose();
    if (aCast != flatA) aCast.dispose();
    if (bCast != flatB) bCast.dispose();
  }

  if (out == null) {
    result.detachToParentScope();
  }
  return result;
}

/// Compute the cross product of two (arrays of) vectors.
///
/// The cross product of two vectors is defined in 3D (and 2D, where it returns the z-component as a scalar).
/// If the inputs are multidimensional, the cross product is computed along the specified axes.
///
/// **Preconditions:**
/// - Both arrays [a] and [b] must not be disposed.
/// - The size of the cross product axes must be 2 or 3.
/// - If provided, the recycler [out] must have the correct shape and dtype.
///
/// **Throws:**
/// - [StateError] if [a] or [b] is disposed.
/// - [ArgumentError] if axes sizes are not 2 or 3, or are mismatched.
/// - [ArgumentError] if [out] has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Uses optimized native FFI vector cross loops, bypassing stack dimensions sequentially.
///
/// **Example:**
/// {@example /example/linalg_advanced_example.dart lang=dart}
///
/// Reference: [NumPy cross](https://numpy.org/doc/stable/reference/generated/numpy.cross.html)
NDArray<R> cross<Ta, Tb, R>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  int? axisa,
  int? axisb,
  int? axisc,
  int? axis,
  NDArray<R>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute cross() on disposed arrays.');
  }

  var axisA = axis ?? axisa ?? -1;
  var axisB = axis ?? axisb ?? -1;
  var axisC = axis ?? axisc ?? -1;

  if (axisA < 0) axisA = a.rank + axisA;
  if (axisB < 0) axisB = b.rank + axisB;

  if (axisA < 0 || axisA >= a.rank) {
    throw ArgumentError('axisa $axisA out of bounds for shape ${a.shape}');
  }
  if (axisB < 0 || axisB >= b.rank) {
    throw ArgumentError('axisb $axisB out of bounds for shape ${b.shape}');
  }

  final lenA = a.shape[axisA];
  final lenB = b.shape[axisB];

  if ((lenA != 2 && lenA != 3) || (lenB != 2 && lenB != 3)) {
    throw ArgumentError(
      'Cross product axes sizes must be 2 or 3 (got axisa size $lenA and axisb size $lenB).',
    );
  }
  if (lenA != lenB) {
    throw ArgumentError(
      'Mismatched cross product axes sizes: axisa size $lenA != axisb size $lenB.',
    );
  }

  final is3D = lenA == 3;

  final stackA = List<int>.from(a.shape)..removeAt(axisA);
  final stackB = List<int>.from(b.shape)..removeAt(axisB);
  final broadcastStack = broadcastStackShapes(stackA, stackB);

  final expectedShape = List<int>.from(broadcastStack);
  if (is3D) {
    var finalAxisC = axisC;
    if (finalAxisC < 0) finalAxisC = expectedShape.length + 1 + finalAxisC;
    if (finalAxisC < 0 || finalAxisC > expectedShape.length) {
      finalAxisC = expectedShape.length;
    }
    expectedShape.insert(finalAxisC, 3);
    axisC = finalAxisC;
  }

  final targetDType = resolveDType(a.dtype, b.dtype);
  if (out != null) {
    if (!listEquals(out.shape, expectedShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out recycler has incompatible shape or dtype (expected shape $expectedShape and dtype $targetDType).',
      );
    }
  }

  final result =
      out ?? NDArray<R>.create(expectedShape, targetDType as DType<R>);

  final aCast = castNDArray(a, targetDType);
  final bCast = castNDArray(b, targetDType);

  final lenResult = broadcastStack.length;
  final walkStridesA = List<int>.filled(lenResult, 0);
  final walkStridesB = List<int>.filled(lenResult, 0);
  final walkStridesRes = List<int>.filled(lenResult, 0);

  for (var i = 0; i < lenResult; i++) {
    final resAxis = lenResult - 1 - i;
    final axisIdxA = stackA.length - 1 - i;
    final axisIdxB = stackB.length - 1 - i;

    var resAxisIdx = resAxis;
    if (is3D && resAxis >= axisC) {
      resAxisIdx = resAxis + 1;
    }

    if (axisIdxA >= 0) {
      final origAxisA = axisIdxA < axisA ? axisIdxA : axisIdxA + 1;
      walkStridesA[resAxis] = (stackA[axisIdxA] == broadcastStack[resAxis])
          ? aCast.strides[origAxisA]
          : 0;
    }
    if (axisIdxB >= 0) {
      final origAxisB = axisIdxB < axisB ? axisIdxB : axisIdxB + 1;
      walkStridesB[resAxis] = (stackB[axisIdxB] == broadcastStack[resAxis])
          ? bCast.strides[origAxisB]
          : 0;
    }
    walkStridesRes[resAxis] = result.strides[resAxisIdx];
  }

  final strideVecA = aCast.strides[axisA];
  final strideVecB = bCast.strides[axisB];
  final strideVecRes = is3D ? result.strides[axisC] : 0;

  void walk(int dim, int offsetA, int offsetB, int offsetRes) {
    if (dim == lenResult) {
      switch (targetDType) {
        case DType.float64:
          if (is3D) {
            s_cross_3d_double(
              aCast.pointer.cast<ffi.Double>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Double>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Double>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_double(
              aCast.pointer.cast<ffi.Double>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Double>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Double>() + offsetRes,
            );
          }
        case DType.float32:
          if (is3D) {
            s_cross_3d_float(
              aCast.pointer.cast<ffi.Float>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Float>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Float>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_float(
              aCast.pointer.cast<ffi.Float>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Float>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Float>() + offsetRes,
            );
          }
        case DType.int64:
          if (is3D) {
            s_cross_3d_int64(
              aCast.pointer.cast<ffi.Int64>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Int64>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Int64>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_int64(
              aCast.pointer.cast<ffi.Int64>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Int64>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Int64>() + offsetRes,
            );
          }
        case DType.int32:
          if (is3D) {
            s_cross_3d_int32(
              aCast.pointer.cast<ffi.Int32>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Int32>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Int32>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_int32(
              aCast.pointer.cast<ffi.Int32>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Int32>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Int32>() + offsetRes,
            );
          }
        case DType.uint8:
          if (is3D) {
            s_cross_3d_uint8(
              aCast.pointer.cast<ffi.Uint8>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Uint8>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Uint8>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_uint8(
              aCast.pointer.cast<ffi.Uint8>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Uint8>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Uint8>() + offsetRes,
            );
          }
        case DType.int16:
          if (is3D) {
            s_cross_3d_int16(
              aCast.pointer.cast<ffi.Int16>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Int16>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Int16>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_int16(
              aCast.pointer.cast<ffi.Int16>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Int16>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Int16>() + offsetRes,
            );
          }
        case DType.complex128:
          if (is3D) {
            s_cross_3d_complex128(
              aCast.pointer.cast<cpx_t>() + offsetA,
              strideVecA,
              bCast.pointer.cast<cpx_t>() + offsetB,
              strideVecB,
              result.pointer.cast<cpx_t>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_complex128(
              aCast.pointer.cast<cpx_t>() + offsetA,
              strideVecA,
              bCast.pointer.cast<cpx_t>() + offsetB,
              strideVecB,
              result.pointer.cast<cpx_t>() + offsetRes,
            );
          }
        case DType.complex64:
          if (is3D) {
            s_cross_3d_complex64(
              aCast.pointer.cast<cpx_f_t>() + offsetA,
              strideVecA,
              bCast.pointer.cast<cpx_f_t>() + offsetB,
              strideVecB,
              result.pointer.cast<cpx_f_t>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_complex64(
              aCast.pointer.cast<cpx_f_t>() + offsetA,
              strideVecA,
              bCast.pointer.cast<cpx_f_t>() + offsetB,
              strideVecB,
              result.pointer.cast<cpx_f_t>() + offsetRes,
            );
          }
        case DType.boolean:
          if (is3D) {
            s_cross_3d_boolean(
              aCast.pointer.cast<ffi.Uint8>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Uint8>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Uint8>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_boolean(
              aCast.pointer.cast<ffi.Uint8>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Uint8>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Uint8>() + offsetRes,
            );
          }
      }
      return;
    }

    final size = broadcastStack[dim];
    final strideA = walkStridesA[dim];
    final strideB = walkStridesB[dim];
    final strideRes = walkStridesRes[dim];

    for (var i = 0; i < size; i++) {
      walk(
        dim + 1,
        offsetA + i * strideA,
        offsetB + i * strideB,
        offsetRes + i * strideRes,
      );
    }
  }

  walk(0, 0, 0, 0);

  if (aCast != a) aCast.dispose();
  if (bCast != b) bCast.dispose();

  if (out == null) {
    result.detachToParentScope();
  }
  return result;
}

/// Compute a vector or matrix norm.
///
/// Computes one of the standard vector or matrix norms (magnitude) along the specified axis/axes.
/// The result is always a real-valued floating-point array.
///
/// **Preconditions:**
/// - Input [a] must not be disposed.
/// - If provided, the recycler [out] must have the correct shape and dtype.
/// - If provided, [axis] must be within bounds.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [axis] or [ord] combinations are invalid.
///
/// **Performance considerations:**
/// - Uses fast native vector reductions for Chebyshev, L1, and L2 vector calculations.
///
/// **Example:**
/// {@example /example/linalg_advanced_example.dart lang=dart}
///
/// Reference: [NumPy linalg.norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)
NDArray norm(
  NDArray a, {
  dynamic ord,
  dynamic axis,
  bool keepdims = false,
  NDArray? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute norm() on a disposed array.');
  }

  final targetDType = (a.dtype == DType.float32 || a.dtype == DType.complex64)
      ? DType.float32
      : DType.float64;

  List<int> normAxes;
  if (axis == null) {
    if (ord == 'fro' || ord == 'nuc' || (ord != null && a.rank == 2)) {
      normAxes = [0, 1];
    } else {
      normAxes = List<int>.generate(a.rank, (i) => i);
    }
  } else if (axis is int) {
    var ax = axis < 0 ? a.rank + axis : axis;
    if (ax < 0 || ax >= a.rank) {
      throw ArgumentError('axis $axis out of bounds for rank ${a.rank}.');
    }
    normAxes = [ax];
  } else if (axis is List) {
    normAxes = axis.map((e) {
      var ax = (e as int) < 0 ? a.rank + e : e;
      if (ax < 0 || ax >= a.rank) {
        throw ArgumentError('axis $e out of bounds for rank ${a.rank}.');
      }
      return ax;
    }).toList();
  } else {
    throw ArgumentError('axis must be null, an int, or a List of ints.');
  }

  if (normAxes.length > 2) {
    throw ArgumentError(
      'Improper number of axes to norm(): ${normAxes.length}.',
    );
  }

  final expectedShape = <int>[];
  for (var i = 0; i < a.rank; i++) {
    if (!normAxes.contains(i)) {
      expectedShape.add(a.shape[i]);
    } else if (keepdims) {
      expectedShape.add(1);
    }
  }

  if (out != null) {
    if (!listEquals(out.shape, expectedShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out recycler has incompatible shape or dtype (expected shape $expectedShape and dtype $targetDType).',
      );
    }
  }

  if (a.rank == 1 && axis == null) {
    final result = out ?? NDArray.zeros(expectedShape, targetDType);
    final val = _vectorNorm(a, ord, targetDType);
    result.data[0] = castValue(val, targetDType);
    if (out == null) {
      result.detachToParentScope();
    }
    return result;
  }

  final result = out ?? NDArray.zeros(expectedShape, targetDType);

  final stackShape = <int>[];
  for (var i = 0; i < a.rank; i++) {
    if (!normAxes.contains(i)) {
      stackShape.add(a.shape[i]);
    }
  }

  walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
    coords,
  ) {
    var offsetRes = 0;
    var stackIdx = 0;
    for (var i = 0; i < result.rank; i++) {
      if (result.shape[i] == 1) continue;
      offsetRes += coords[stackIdx] * result.strides[i];
      stackIdx++;
    }

    var offsetA = 0;
    stackIdx = 0;
    for (var i = 0; i < a.rank; i++) {
      if (!normAxes.contains(i)) {
        offsetA += coords[stackIdx] * a.strides[i];
        stackIdx++;
      }
    }

    if (normAxes.length == 1) {
      final axisIdx = normAxes[0];
      final slice = NDArray.view(
        a,
        shape: [a.shape[axisIdx]],
        strides: [a.strides[axisIdx]],
        offsetElements: offsetA,
      );
      final normVal = _vectorNorm(slice, ord, targetDType);
      result.data[offsetRes] = castValue(normVal, targetDType);
      slice.dispose();
    } else {
      final ax0 = normAxes[0];
      final ax1 = normAxes[1];
      final slice = NDArray.view(
        a,
        shape: [a.shape[ax0], a.shape[ax1]],
        strides: [a.strides[ax0], a.strides[ax1]],
        offsetElements: offsetA,
      );
      final normVal = _matrixNorm(slice, ord, targetDType);
      result.data[offsetRes] = castValue(normVal, targetDType);
      slice.dispose();
    }
  });

  if (out == null) {
    result.detachToParentScope();
  }
  return result;
}

double _vectorNorm(NDArray a, dynamic ord, DType targetDType) {
  final needsCast = a.dtype != targetDType;
  final castedA = needsCast ? castNDArray(a, targetDType) : a;

  final size = castedA.size;
  final stride = castedA.strides.isEmpty ? 1 : castedA.strides[0];

  try {
    if (ord == null || ord == 2) {
      double sum;
      if (targetDType == DType.float32) {
        if (castedA.dtype.isComplex) {
          sum = r_norm_l2_complex64(castedA.pointer.cast(), stride, size);
        } else {
          sum = r_norm_l2_float(castedA.pointer.cast(), stride, size);
        }
      } else {
        if (castedA.dtype.isComplex) {
          sum = r_norm_l2_complex128(castedA.pointer.cast(), stride, size);
        } else {
          sum = r_norm_l2_double(castedA.pointer.cast(), stride, size);
        }
      }
      return math.sqrt(sum);
    } else if (ord == 1) {
      if (targetDType == DType.float32) {
        if (castedA.dtype.isComplex) {
          return r_norm_l1_complex64(castedA.pointer.cast(), stride, size);
        } else {
          return r_norm_l1_float(castedA.pointer.cast(), stride, size);
        }
      } else {
        if (castedA.dtype.isComplex) {
          return r_norm_l1_complex128(castedA.pointer.cast(), stride, size);
        } else {
          return r_norm_l1_double(castedA.pointer.cast(), stride, size);
        }
      }
    } else if (ord == double.infinity) {
      if (targetDType == DType.float32) {
        if (castedA.dtype.isComplex) {
          return r_norm_inf_complex64(castedA.pointer.cast(), stride, size);
        } else {
          return r_norm_inf_float(castedA.pointer.cast(), stride, size);
        }
      } else {
        if (castedA.dtype.isComplex) {
          return r_norm_inf_complex128(castedA.pointer.cast(), stride, size);
        } else {
          return r_norm_inf_double(castedA.pointer.cast(), stride, size);
        }
      }
    } else if (ord == double.negativeInfinity) {
      if (targetDType == DType.float32) {
        if (castedA.dtype.isComplex) {
          return r_norm_neg_inf_complex64(castedA.pointer.cast(), stride, size);
        } else {
          return r_norm_neg_inf_float(castedA.pointer.cast(), stride, size);
        }
      } else {
        if (castedA.dtype.isComplex) {
          return r_norm_neg_inf_complex128(
            castedA.pointer.cast(),
            stride,
            size,
          );
        } else {
          return r_norm_neg_inf_double(castedA.pointer.cast(), stride, size);
        }
      }
    } else if (ord is num) {
      double sum;
      final p = ord.toDouble();
      if (targetDType == DType.float32) {
        if (castedA.dtype.isComplex) {
          sum = r_norm_lp_complex64(castedA.pointer.cast(), stride, size, p);
        } else {
          sum = r_norm_lp_float(castedA.pointer.cast(), stride, size, p);
        }
      } else {
        if (castedA.dtype.isComplex) {
          sum = r_norm_lp_complex128(castedA.pointer.cast(), stride, size, p);
        } else {
          sum = r_norm_lp_double(castedA.pointer.cast(), stride, size, p);
        }
      }
      return math.pow(sum, 1.0 / p).toDouble();
    } else {
      throw ArgumentError('Invalid vector norm order: $ord');
    }
  } finally {
    if (needsCast) castedA.dispose();
  }
}

double _matrixNorm(NDArray a, dynamic ord, DType targetDType) {
  final rows = a.shape[0];
  final cols = a.shape[1];

  if (ord == null || ord == 'fro') {
    final flat = a.ravel();
    final res = _vectorNorm(flat, 2, targetDType);
    flat.dispose();
    return res;
  } else if (ord == 1) {
    var maxColSum = 0.0;
    for (var c = 0; c < cols; c++) {
      final colSlice = NDArray.view(
        a,
        shape: [rows],
        strides: [a.strides[0]],
        offsetElements: c * a.strides[1],
      );
      final sum = _vectorNorm(colSlice, 1, targetDType);
      if (sum > maxColSum) maxColSum = sum;
      colSlice.dispose();
    }
    return maxColSum;
  } else if (ord == -1) {
    double? minColSum;
    for (var c = 0; c < cols; c++) {
      final colSlice = NDArray.view(
        a,
        shape: [rows],
        strides: [a.strides[0]],
        offsetElements: c * a.strides[1],
      );
      final sum = _vectorNorm(colSlice, 1, targetDType);
      if (minColSum == null || sum < minColSum) minColSum = sum;
      colSlice.dispose();
    }
    return minColSum ?? 0.0;
  } else if (ord == double.infinity) {
    var maxRowSum = 0.0;
    for (var r = 0; r < rows; r++) {
      final rowSlice = NDArray.view(
        a,
        shape: [cols],
        strides: [a.strides[1]],
        offsetElements: r * a.strides[0],
      );
      final sum = _vectorNorm(rowSlice, 1, targetDType);
      if (sum > maxRowSum) maxRowSum = sum;
      rowSlice.dispose();
    }
    return maxRowSum;
  } else if (ord == double.negativeInfinity) {
    double? minRowSum;
    for (var r = 0; r < rows; r++) {
      final rowSlice = NDArray.view(
        a,
        shape: [cols],
        strides: [a.strides[1]],
        offsetElements: r * a.strides[0],
      );
      final sum = _vectorNorm(rowSlice, 1, targetDType);
      if (minRowSum == null || sum < minRowSum) minRowSum = sum;
      rowSlice.dispose();
    }
    return minRowSum ?? 0.0;
  } else if (ord == 2 || ord == -2) {
    final castedA =
        (a.dtype == DType.float64 ||
            a.dtype == DType.float32 ||
            a.dtype.isComplex)
        ? a
        : castNDArray(a, DType.float64);
    try {
      final svdRes = svd(castedA);
      final s = svdRes.S;
      final val = ord == 2 ? s.data[0] : s.data[s.data.length - 1];
      svdRes.dispose();
      return val.toDouble();
    } finally {
      if (castedA != a) castedA.dispose();
    }
  } else {
    throw ArgumentError('Invalid matrix norm order: $ord');
  }
}

num _getMinLimit(DType dtype) {
  switch (dtype) {
    case DType.float64:
    case DType.float32:
      return double.negativeInfinity;
    case DType.int64:
      return -9223372036854775808;
    case DType.int32:
      return -2147483648;
    case DType.int16:
      return -32768;
    case DType.uint8:
      return 0;
    default:
      return double.negativeInfinity;
  }
}

num _getMaxLimit(DType dtype) {
  switch (dtype) {
    case DType.float64:
    case DType.float32:
      return double.infinity;
    case DType.int64:
      return 9223372036854775807;
    case DType.int32:
      return 2147483647;
    case DType.int16:
      return 32767;
    case DType.uint8:
      return 255;
    default:
      return double.infinity;
  }
}

extension QRRecordDispose on ({NDArray Q, NDArray R}) {
  void dispose() {
    Q.dispose();
    R.dispose();
  }
}

extension SVDRecordDispose on ({NDArray U, NDArray S, NDArray Vh}) {
  void dispose() {
    U.dispose();
    S.dispose();
    Vh.dispose();
  }
}
