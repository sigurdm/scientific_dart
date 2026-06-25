// ignore_for_file: non_constant_identifier_names
import 'dart:math' as math;
import 'dart:ffi' as ffi;
import '../../ndarray.dart';
import '../../ndarray_bindings.dart';
import '../../scratch_arena.dart';
import '../helpers.dart';
import '../broadcasting.dart';
import 'arithmetic.dart';

/// Computes the element-wise sine of the array.
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
/// - For C-contiguous array layouts, uses native C vector math kernels (`v_sin_double`/`v_sin_float`).
///
/// **Example:**
/// {@example /example/transcendental_example.dart lang=dart}
///
/// Reference: [Trigonometric Sine Function](https://en.wikipedia.org/wiki/Sine_and_cosine)
NDArray<R> sin<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute sin() on a disposed array.');
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
        'Provided out buffer has incompatible shape or dtype for sin.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_sin_double(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.float32:
        v_sin_float(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex128:
        v_sin_complex128(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex64:
        v_sin_complex64(a.pointer.cast(), result.pointer.cast(), a.size);
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

  if (a.dtype.isInteger) {
    unaryOp<num, double>(
      result.data as List<double>,
      a.data as List<num>,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => math.sin(x),
    );
  } else {
    unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => math.sin(x.toDouble()),
    );
  }
  return result;
}

/// Computes the element-wise sinc of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`) or [Complex].
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous array layouts, uses native C vector math kernels (`v_sinc_double`/`v_sinc_float` etc).
NDArray<R> sinc<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute sinc() on a disposed array.');
  }

  // Handle integer types by promoting to float64 (double)
  if (a.dtype.isInteger) {
    final promoted = promoteToDouble(a);
    final res = sinc<double, double>(promoted, out: out as NDArray<double>?);
    promoted.dispose();
    return res as NDArray<R>;
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
        'Provided out buffer has incompatible shape or dtype for sinc.',
      );
    }
    result = out;
  } else {
    result = NDArray<R>.create(a.shape, targetDType as DType<R>);
  }

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_sinc_double(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.float32:
        v_sinc_float(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex128:
        v_sinc_complex128(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex64:
        v_sinc_complex64(a.pointer.cast(), result.pointer.cast(), a.size);
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
      s_sinc_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      s_sinc_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex128) {
      s_sinc_complex128(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex64) {
      s_sinc_complex64(
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

  throw ArgumentError('Unsupported DType for sinc: ${a.dtype}');
}

/// Computes the element-wise cosine of the array.
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
/// - For C-contiguous array layouts, uses native C vector math kernels (`v_cos_double`/`v_cos_float`).
///
/// **Example:**
/// {@example /example/transcendental_example.dart lang=dart}
///
/// Reference: [Trigonometric Cosine Function](https://en.wikipedia.org/wiki/Sine_and_cosine)
NDArray<R> cos<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
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

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_cos_double(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.float32:
        v_cos_float(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex128:
        v_cos_complex128(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex64:
        v_cos_complex64(a.pointer.cast(), result.pointer.cast(), a.size);
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

    switch (a.dtype) {
      case DType.float64:
        s_cos_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.float32:
        s_cos_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.complex128:
        s_cos_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.complex64:
        s_cos_complex64(
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
  }

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => math.cos(x.toDouble()),
    );
  } else {
    unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => math.cos(x.toDouble()),
    );
  }
  return result;
}

/// Computes the element-wise tangent of the array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<R> tan<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
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

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_tan_double(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.float32:
        v_tan_float(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex128:
        v_tan_complex128(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex64:
        v_tan_complex64(a.pointer.cast(), result.pointer.cast(), a.size);
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

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => math.tan(x.toDouble()),
    );
  } else {
    unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => math.tan(x.toDouble()),
    );
  }
  return result;
}

/// Computes the element-wise arc sine (inverse sine) of the array.
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
  if (a.isDisposed || (out != null && out.isDisposed)) {
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

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_asin_double(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.float32:
        v_asin_float(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex128:
        v_asin_complex128(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex64:
        v_asin_complex64(a.pointer.cast(), result.pointer.cast(), a.size);
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

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => math.asin(x.toDouble()),
    );
  } else {
    unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => math.asin(x),
    );
  }
  return result;
}

/// Computes the element-wise arc cosine (inverse cosine) of the array.
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
  if (a.isDisposed || (out != null && out.isDisposed)) {
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

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_acos_double(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.float32:
        v_acos_float(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex128:
        v_acos_complex128(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex64:
        v_acos_complex64(a.pointer.cast(), result.pointer.cast(), a.size);
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

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => math.acos(x.toDouble()),
    );
  } else {
    unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => math.acos(x),
    );
  }
  return result;
}

/// Computes the element-wise arc tangent (inverse tangent) of the array.
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
  if (a.isDisposed || (out != null && out.isDisposed)) {
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

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_atan_double(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.float32:
        v_atan_float(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex128:
        v_atan_complex128(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex64:
        v_atan_complex64(a.pointer.cast(), result.pointer.cast(), a.size);
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

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => math.atan(x.toDouble()),
    );
  } else {
    unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
      (x) => math.atan(x),
    );
  }
  return result;
}

/// Computes the element-wise hyperbolic sine of the array.
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
NDArray<R> sinh<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute sinh() on a disposed array.');
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
        'Provided out buffer has incompatible shape or dtype for sinh.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_sinh_double(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.float32:
        v_sinh_float(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex128:
        v_sinh_complex128(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex64:
        v_sinh_complex64(a.pointer.cast(), result.pointer.cast(), a.size);
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
      s_sinh_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      s_sinh_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex128) {
      s_sinh_complex128(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex64) {
      s_sinh_complex64(
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
      return (math.exp(val) - math.exp(-val)) / 2.0 as R;
    },
  );
  return result;
}

/// Computes the element-wise hyperbolic cosine of the array.
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
NDArray<R> cosh<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute cosh() on a disposed array.');
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
        'Provided out buffer has incompatible shape or dtype for cosh.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_cosh_double(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.float32:
        v_cosh_float(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex128:
        v_cosh_complex128(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex64:
        v_cosh_complex64(a.pointer.cast(), result.pointer.cast(), a.size);
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
      s_cosh_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      s_cosh_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex128) {
      s_cosh_complex128(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex64) {
      s_cosh_complex64(
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
      return (math.exp(val) + math.exp(-val)) / 2.0 as R;
    },
  );
  return result;
}

/// Computes the element-wise hyperbolic tangent of the array.
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
NDArray<R> tanh<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute tanh() on a disposed array.');
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
        'Provided out buffer has incompatible shape or dtype for tanh.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_tanh_double(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.float32:
        v_tanh_float(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex128:
        v_tanh_complex128(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex64:
        v_tanh_complex64(a.pointer.cast(), result.pointer.cast(), a.size);
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
      s_tanh_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      s_tanh_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex128) {
      s_tanh_complex128(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex64) {
      s_tanh_complex64(
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
      final exp2val = math.exp(2.0 * val);
      return (exp2val - 1.0) / (exp2val + 1.0) as R;
    },
  );
  return result;
}

/// Computes the element-wise inverse hyperbolic sine of the array.
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
NDArray<R> asinh<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute asinh() on a disposed array.');
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
        'Provided out buffer has incompatible shape or dtype for asinh.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType) as NDArray<R>;
  }

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_asinh_double(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.float32:
        v_asinh_float(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex128:
        v_asinh_complex128(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex64:
        v_asinh_complex64(a.pointer.cast(), result.pointer.cast(), a.size);
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
      s_asinh_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      s_asinh_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex128) {
      s_asinh_complex128(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
      );
      return result;
    } else if (a.dtype == DType.complex64) {
      s_asinh_complex64(
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
      return math.log(val + math.sqrt(val * val + 1.0)) as R;
    },
  );
  return result;
}

/// Computes the element-wise inverse hyperbolic cosine of the array.
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
NDArray<R> acosh<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute acosh() on a disposed array.');
  }
  final DType<R> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype as DType<R>;
  } else {
    targetDType =
        (a.dtype == DType.float32 ? DType.float32 : DType.float64) as DType<R>;
  }

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for acosh.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType);
  }

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_acosh_double(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.float32:
        v_acosh_float(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex128:
        v_acosh_complex128(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex64:
        v_acosh_complex64(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      default:
        break;
    }
  } else {
    final rank = a.shape.length;
    final marker = ScratchArena.marker;
    final cShape = ScratchArena.copyInts(a.shape);
    final cStridesA = ScratchArena.copyInts(a.strides);
    final cStridesRes = ScratchArena.copyInts(result.strides);
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
        case DType.complex128:
          s_acosh_complex128(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        case DType.complex64:
          s_acosh_complex64(
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
      ScratchArena.reset(marker);
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
      return math.log(val + math.sqrt(val * val - 1.0)) as R;
    },
  );
  return result;
}

/// Computes the element-wise inverse hyperbolic tangent of the array.
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
  if (a.isDisposed || (out != null && out.isDisposed)) {
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

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_atanh_double(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.float32:
        v_atanh_float(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex128:
        v_atanh_complex128(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex64:
        v_atanh_complex64(a.pointer.cast(), result.pointer.cast(), a.size);
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

/// Computes the element-wise arc tangent of [y] / [x] with full broadcasting support.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray<double> atan2<Ty, Tx>(
  NDArray<Ty> y,
  NDArray<Tx> x, {
  NDArray<double>? out,
}) {
  if (y.isDisposed || x.isDisposed || (out != null && out.isDisposed)) {
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
  final DType<double> targetDType =
      (y.dtype == DType.float32 && x.dtype == DType.float32)
      ? DType.float32
      : DType.float64;

  final NDArray<double> result;
  if (out != null) {
    if (!listEquals(out.shape, shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for atan2.',
      );
    }
    result = out;
  } else {
    result = NDArray<double>.create(shape, targetDType);
  }

  // 0. Native C Vector Extension Fast-Path Gate for Contiguous Same-Shape arrays
  if (y.isContiguous &&
      x.isContiguous &&
      result.isContiguous &&
      listEquals(y.shape, x.shape)) {
    switch ((y.dtype, x.dtype)) {
      case (DType.float64, DType.float64):
        v_atan2_double(
          y.pointer.cast(),
          x.pointer.cast(),
          result.pointer.cast(),
          y.size,
        );
        return result;
      case (DType.float32, DType.float32):
        v_atan2_float(
          y.pointer.cast(),
          x.pointer.cast(),
          result.pointer.cast(),
          y.size,
        );
        return result;
      default:
        break;
    }
  }
  final stridesY = broadcastResult.stridesA;
  final stridesX = broadcastResult.stridesB;

  // 0C. General Multidimensional Strided Broadcasting Engine in C (Rank <= 8)
  if (shape.length <= 8) {
    final marker = ScratchArena.marker;
    final cShape = ScratchArena.copyInts(shape);
    final cStridesY = ScratchArena.copyInts(stridesY);
    final cStridesX = ScratchArena.copyInts(stridesX);
    final cStridesRes = ScratchArena.copyInts(result.strides);
    try {
      switch ((targetDType, y.dtype, x.dtype)) {
        case (DType.float64, DType.float64, DType.float64):
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
        case (DType.float32, DType.float32, DType.float32):
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
        default:
          break;
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }

  final rData = result.data;

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
        result.strides,
        0,
        y.offsetElements,
        x.offsetElements,
        result.offsetElements,
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
        result.strides,
        0,
        y.offsetElements,
        x.offsetElements,
        result.offsetElements,
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
        result.strides,
        0,
        y.offsetElements,
        x.offsetElements,
        result.offsetElements,
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
        result.strides,
        0,
        y.offsetElements,
        x.offsetElements,
        result.offsetElements,
        (a, b) => math.atan2(a.toDouble(), b.toDouble()),
      );
    }
  }
  return result;
}

/// Computes the element-wise hypotenuse `sqrt(x1**2 + x2**2)` with broadcasting support.
///
/// **Example:**
/// ```dart
/// final h = hypot(a, b);
/// ```
NDArray<R> hypot<Ta, Tb, R>(NDArray<Ta> a, NDArray<Tb> b, {NDArray<R>? out}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute hypot() on a disposed array.');
  }
  final broadcastResult = broadcast(a, b);
  final shape = broadcastResult.shape;
  final DType<R> targetDType =
      ((a.dtype == DType.complex64 || b.dtype == DType.complex64)
              ? DType.float32
              : DType.float64)
          as DType<R>;

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for hypot.',
      );
    }
    result = out;
  } else {
    result = NDArray<R>.create(shape, targetDType);
  }

  if (a.dtype == DType.complex128 ||
      b.dtype == DType.complex128 ||
      a.dtype == DType.complex64 ||
      b.dtype == DType.complex64) {
    final aCpx = (a.dtype == DType.complex128 || a.dtype == DType.complex64)
        ? a as NDArray<Complex>
        : NDArray<Complex>.fromList(
            a.data.map((e) => Complex((e as num).toDouble(), 0.0)).toList(),
            a.shape,
            DType.complex128,
          );
    final bCpx = (b.dtype == DType.complex128 || b.dtype == DType.complex64)
        ? b as NDArray<Complex>
        : NDArray<Complex>.fromList(
            b.data.map((e) => Complex((e as num).toDouble(), 0.0)).toList(),
            b.shape,
            DType.complex128,
          );
    if (listEquals(a.shape, b.shape) &&
        a.isContiguous &&
        b.isContiguous &&
        result.isContiguous) {
      if (aCpx.dtype == DType.complex128) {
        v_hypot_complex128(
          aCpx.pointer.cast(),
          bCpx.pointer.cast(),
          result.pointer.cast(),
          aCpx.size,
        );
        return result;
      } else {
        v_hypot_complex64(
          aCpx.pointer.cast(),
          bCpx.pointer.cast(),
          result.pointer.cast(),
          aCpx.size,
        );
        return result;
      }
    } else {
      final rank = shape.length;
      final marker = ScratchArena.marker;
      final cShape = ScratchArena.copyInts(shape);
      final cStridesA = ScratchArena.copyInts(broadcastResult.stridesA);
      final cStridesB = ScratchArena.copyInts(broadcastResult.stridesB);
      final cStridesRes = ScratchArena.copyInts(result.strides);
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
        ScratchArena.reset(marker);
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

  elementWiseOp<num, num, R>(
    rData,
    a.data as List<num>,
    b.data as List<num>,
    shape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    result.strides,
    0,
    a.offsetElements,
    b.offsetElements,
    result.offsetElements,
    (valA, valB) => hypotOp(valA.toDouble(), valB.toDouble()) as R,
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
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute deg2rad() on a disposed array.');
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
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute rad2deg() on a disposed array.');
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
