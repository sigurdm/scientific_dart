// ignore_for_file: non_constant_identifier_names
import '../../ndarray.dart';
import '../../ndarray_bindings.dart';
import '../../scratch_arena.dart';
import '../helpers.dart';

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
/// final a = NDArray<Complex>.create([2], `DType.complex128);`
/// a.data[0] = Complex(3.0, 4.0);
/// a.data[1] = Complex(-1.0, 0.0);
/// final r = real(a); // [3.0, -1.0] (`DType.float64)`
/// ```
NDArray<R> real<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
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
    unaryOp<Complex, R>(
      result.data,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      result.strides,
      0,
      a.offsetElements,
      result.offsetElements,
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
/// final a = NDArray<Complex>.create([2], `DType.complex128);`
/// a.data[0] = Complex(3.0, 4.0);
/// a.data[1] = Complex(-1.0, 0.0);
/// final im = imag(a); // [4.0, 0.0] (`DType.float64)`
/// ```
NDArray<R> imag<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
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
  unaryOp<Complex, R>(
    result.data,
    a.data as List<Complex>,
    a.shape,
    a.strides,
    result.strides,
    0,
    a.offsetElements,
    result.offsetElements,
    (x) => x.imag as R,
  );

  return result;
}

/// Computes the element-wise complex conjugate of the array elements.
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
NDArray<T> conj<T>(NDArray<T> a, {NDArray<T>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute conj() on a disposed array.');
  }
  final targetDType = a.dtype;
  final result = out ?? NDArray<T>.create(a.shape, targetDType);
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
        v_conj_complex128(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      } else {
        final rank = a.shape.length;
        final marker = ScratchArena.marker;
        final cShape = ScratchArena.copyInts(a.shape);
        final cStridesA = ScratchArena.copyInts(a.strides);
        final cStridesRes = ScratchArena.copyInts(result.strides);
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
          ScratchArena.reset(marker);
        }
      }
    case DType.complex64:
      if (a.isContiguous && result.isContiguous) {
        v_conj_complex64(a.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      } else {
        final rank = a.shape.length;
        final marker = ScratchArena.marker;
        final cShape = ScratchArena.copyInts(a.shape);
        final cStridesA = ScratchArena.copyInts(a.strides);
        final cStridesRes = ScratchArena.copyInts(result.strides);
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
          ScratchArena.reset(marker);
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
      a.copy(out: result);
      return result;
  }
}

/// Alias for [conj].
NDArray<T> conjugate<T>(NDArray<T> a, {NDArray<T>? out}) => conj(a, out: out);
