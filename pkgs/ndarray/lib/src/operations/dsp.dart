import 'dart:math' as math;
import 'dart:ffi' as ffi;
import '../ndarray.dart';
import '../ndarray_bindings.dart';
import '../scratch_arena.dart';

/// Computes the element-wise phase/argument of complex numbers.
///
/// Returns an array of double/float (matching the precision of the complex
/// input, i.e., Float64 for Complex128, Float32 for Complex64) with values
/// in $[-\pi, \pi]$.
///
/// Throws [ArgumentError] if the input array is not complex.
NDArray<R> angle<T extends Complex, R extends double>(
  NDArray<T> a, {
  NDArray<R>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute angle() on a disposed array.');
  }

  if (a.dtype != DType.complex128 && a.dtype != DType.complex64) {
    throw ArgumentError('Input array must be complex for angle().');
  }

  final DType<R> targetDType;
  switch (a.dtype) {
    case DType.complex128:
      targetDType = DType.float64 as DType<R>;
      break;
    case DType.complex64:
      targetDType = DType.float32 as DType<R>;
      break;
  }

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for angle.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType);
  }

  if (a.isContiguous && result.isContiguous) {
    switch (a.dtype) {
      case DType.complex128:
        v_angle_complex128(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case DType.complex64:
        v_angle_complex64(a.pointer.cast(), result.pointer.cast(), a.size);
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
      case DType.complex128:
        s_angle_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      case DType.complex64:
        s_angle_complex64(
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
}

/// Unwraps radian phase angles by changing absolute jumps greater than [discont]
/// to their $2\pi$ complement along the given [axis].
///
/// Throws [ArgumentError] if the input array is not float32 or float64.
NDArray<T> unwrap<T extends double>(
  NDArray<T> a, {
  double discont = math.pi,
  int axis = -1,
  NDArray<T>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute unwrap() on a disposed array.');
  }

  if (!a.dtype.isFloating) {
    throw ArgumentError('Input array must be float32 or float64 for unwrap().');
  }

  final rank = a.shape.length;
  final resolvedAxis = axis < 0 ? rank + axis : axis;
  if (resolvedAxis < 0 || resolvedAxis >= rank) {
    throw ArgumentError('Invalid axis $axis for shape ${a.shape}');
  }

  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for unwrap.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, a.dtype);
  }

  final rankForBuffer = a.shape.length;
  final cBuffer = ScratchArena.getStridedBuffer(rankForBuffer);
  final cShape = cBuffer;
  final cStridesA = cBuffer + rankForBuffer;
  final cStridesRes = cBuffer + (rankForBuffer * 2);
  for (var i = 0; i < rankForBuffer; i++) {
    cShape[i] = a.shape[i];
    cStridesA[i] = a.strides[i];
    cStridesRes[i] = result.strides[i];
  }

  switch (a.dtype) {
    case DType.float64:
      s_unwrap_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rankForBuffer,
        resolvedAxis,
        discont,
      );
      return result;
    case DType.float32:
      s_unwrap_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rankForBuffer,
        resolvedAxis,
        discont,
      );
      return result;
  }
}
