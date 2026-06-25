// ignore_for_file: non_constant_identifier_names
import 'dart:ffi' as ffi;
import '../../ndarray.dart';
import '../../ndarray_bindings.dart';
import '../../scratch_arena.dart';
import '../helpers.dart';
import '../broadcasting.dart';


/// Returns an element-wise boolean mask indicating which elements of the array are NaN.
///
/// **Preconditions:**
/// - Input array [a] must not be disposed.
/// - If provided, the [out] recycler array must match the shape and have boolean dtype.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [ArgumentError] if [out] has incompatible shape or dtype.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.nan, 3.0], [3], DType.float64);
/// final mask = isnan(a); // [false, true, false]
/// ```
NDArray<bool> isnan<T>(NDArray<T> a, {NDArray<bool>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute isnan() on a disposed array.');
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

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_isnan_double(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.float32:
        v_isnan_float(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex128:
        v_isnan_complex128(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex64:
        v_isnan_complex64(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.int32:
      case DType.int64:
      case DType.int16:
      case DType.uint8:
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
      cStridesRes[i] = result.strides[i];
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
      case DType.int16:
      case DType.uint8:
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
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
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
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
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
NDArray<bool> isinf<T>(NDArray<T> a, {NDArray<bool>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute isinf() on a disposed array.');
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

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_isinf_double(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.float32:
        v_isinf_float(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex128:
        v_isinf_complex128(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex64:
        v_isinf_complex64(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.int32:
      case DType.int64:
      case DType.int16:
      case DType.uint8:
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
      cStridesRes[i] = result.strides[i];
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
      case DType.int16:
      case DType.uint8:
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
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
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
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
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
NDArray<bool> isfinite<T extends Object>(NDArray<T> a, {NDArray<bool>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute isfinite() on a disposed array.');
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

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.float64:
        v_isfinite_double(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.float32:
        v_isfinite_float(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex128:
        v_isfinite_complex128(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex64:
        v_isfinite_complex64(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.int32:
      case DType.int64:
      case DType.int16:
      case DType.uint8:
      case DType.boolean:
        result.fill(true);
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
      case DType.int16:
      case DType.uint8:
      case DType.boolean:
        result.fill(true);
        return result;
    }
  }
}

/// Returns first element-wise argument with the sign of the second element-wise argument.
///
/// **Throws:**
/// - [StateError] if either array has been disposed.
/// - [UnsupportedError] if either array is complex (copysign is not defined for complex numbers).
///
/// **Example:**
/// ```dart
/// final res = copysign(x1, x2);
/// ```
NDArray<T> copysign<T extends Object>(
  NDArray<T> x1,
  NDArray<T> x2, {
  NDArray<T>? out,
}) {
  if (x1.isDisposed || x2.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute copysign() on a disposed array.');
  }
  if (x1.dtype.isComplex || x2.dtype.isComplex) {
    throw UnsupportedError('Complex numbers are not supported for copysign');
  }

  final broadcastResult = broadcast(x1, x2);
  final shape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final DType<T> targetDType = x1.dtype;

  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for copysign.',
      );
    }
    result = out;
  } else {
    result = NDArray<T>.create(shape, targetDType);
  }

  if (x1.isContiguous &&
      x2.isContiguous &&
      listEquals(x1.shape, x2.shape) &&
      result.isContiguous) {
    if (targetDType == DType.float64) {
      v_copysign_double(
        x1.pointer.cast(),
        x2.pointer.cast(),
        result.pointer.cast(),
        x1.data.length,
      );
      return result;
    } else if (targetDType == DType.float32) {
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
      cStridesRes[i] = result.strides[i];
    }
    if (targetDType == DType.float64) {
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
    } else if (targetDType == DType.float32) {
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

  if (targetDType == DType.float64 || targetDType == DType.float32) {
    elementWiseOp<double, double, double>(
      result.data as List<double>,
      x1.data as List<double>,
      x2.data as List<double>,
      shape,
      stridesA,
      stridesB,
      result.strides,
      0,
      x1.offsetElements,
      x2.offsetElements,
      result.offsetElements,
      (x, y) => copysignOp(x, y),
    );
  } else {
    elementWiseOp<num, num, int>(
      result.data as List<int>,
      x1.data as List<num>,
      x2.data as List<num>,
      shape,
      stridesA,
      stridesB,
      result.strides,
      0,
      x1.offsetElements,
      x2.offsetElements,
      result.offsetElements,
      (x, y) => copysignOp(x.toDouble(), y.toDouble()).toInt(),
    );
  }

  return result;
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
NDArray<bool> isClose<Ta, Tb>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  double rtol = 1e-05,
  double atol = 1e-08,
  bool equalNan = false,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute isClose() on a disposed array.');
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  final size = commonShape.isEmpty ? 1 : commonShape.reduce((x, y) => x * y);
  final result = NDArray<bool>.zeros(commonShape, DType.boolean);

  final aList = a.toList();
  final bList = b.toList();
  final resData = result.data;

  final broadcastResultRes = broadcast(a, b);
  final stridesA = broadcastResultRes.stridesA;
  final stridesB = broadcastResultRes.stridesB;

  final coord = List<int>.filled(commonShape.length, 0);

  bool isNaNVal(dynamic val) {
    if (val is Complex) return val.real.isNaN || val.imag.isNaN;
    if (val is num) return val.isNaN;
    return false;
  }

  bool isInfiniteVal(dynamic val) {
    if (val is Complex) return val.real.isInfinite || val.imag.isInfinite;
    if (val is num) return val.isInfinite;
    return false;
  }

  double absVal(dynamic val) {
    if (val is Complex) return val.abs;
    if (val is num) return val.abs().toDouble();
    return 0.0;
  }

  dynamic diffVal(dynamic a, dynamic b) {
    if (a is Complex) {
      if (b is Complex) return a - b;
      return a - (b as num);
    }
    if (b is Complex) {
      return Complex((a as num).toDouble(), 0.0) - b;
    }
    return (a as num) - (b as num);
  }

  for (var el = 0; el < size; el++) {
    var offsetA = 0;
    var offsetB = 0;
    var offsetRes = 0;
    for (var d = 0; d < commonShape.length; d++) {
      offsetA += coord[d] * stridesA[d];
      offsetB += coord[d] * stridesB[d];
      offsetRes += coord[d] * result.strides[d];
    }

    final valA = aList[offsetA];
    final valB = bList[offsetB];

    var match = false;
    if (equalNan && isNaNVal(valA) && isNaNVal(valB)) {
      match = true;
    } else if (isInfiniteVal(valA) || isInfiniteVal(valB)) {
      match = valA == valB;
    } else {
      final diff = absVal(diffVal(valA, valB));
      final limit = atol + rtol * absVal(valB);
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
bool allClose<Ta, Tb>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  double rtol = 1e-05,
  double atol = 1e-08,
  bool equalNan = false,
}) {
  final closeMask = isClose(a, b, rtol: rtol, atol: atol, equalNan: equalNan);
  final maskList = closeMask.toList();
  closeMask.dispose();
  for (final val in maskList) {
    if (!val) return false;
  }
  return true;
}
