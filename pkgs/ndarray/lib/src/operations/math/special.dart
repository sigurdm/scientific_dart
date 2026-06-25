// ignore_for_file: non_constant_identifier_names
import '../../ndarray.dart';
import '../../ndarray_bindings.dart';
import '../../scratch_arena.dart';
import '../helpers.dart';

/// Computes the zeroth order modified Bessel function of the first kind, $I_0(x)$ element-wise.
///
/// Supports float32, float64, complex64, complex128.
/// Integer and boolean types are promoted to float64.
///
/// **Mathematical Details:**
/// - For real types (float32, float64):
///   - For $|x| \le 3.75$, uses the Abramowitz-Stegun polynomial approximation.
///   - For $|x| > 3.75$, uses the asymptotic expansion.
/// - For complex types (complex64, complex128):
///   - For $|z| \le 15$, uses the power series expansion:
///     $$I_0(z) = \sum_{k=0}^{n} \frac{(z^2/4)^k}{(k!)^2}$$
///   - For $|z| > 15$, uses the two-term asymptotic expansion:
///     $$I_0(z) \approx \frac{e^z}{\sqrt{2\pi z}} A(z) + i \frac{e^{-z}}{\sqrt{2\pi z}} A(-z)$$
///     where $A(z) = 1 + \frac{1}{8z} + \frac{9}{128z^2} + \frac{75}{1024z^3} + \frac{1225}{32768z^4}$,
///     mapped to the first quadrant and conjugated appropriately to handle all quadrants.
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
/// - If [out] is provided, it must not be disposed and must have the same shape
///   and compatible dtype as the result.
///
/// **Throws:**
/// - [StateError] if [a] or [out] is disposed.
/// - [ArgumentError] if [out] has incompatible shape or dtype.
/// - [ArgumentError] if the dtype of [a] is not supported.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([0.0, 1.0, 2.0], [3], DType.float64);
/// final b = i0(a);
/// print(b.data); // [1.0, ~1.266066, ~2.279585]
/// ```
NDArray<R> i0<T, R>(NDArray<T> a, {NDArray<R>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute i0() on a disposed array.');
  }

  // Handle integer and boolean types by promoting to float64 (double)
  if (a.dtype.isInteger || a.dtype == DType.boolean) {
    final promoted = promoteToDouble(a);
    final res = i0<double, double>(promoted, out: out as NDArray<double>?);
    promoted.dispose();
    return res as NDArray<R>;
  }

  final DType<R> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype as DType<R>;
  } else if (a.dtype == DType.float32) {
    targetDType = DType.float32 as DType<R>;
  } else {
    targetDType = DType.float64 as DType<R>;
  }

  final NDArray<R> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for i0.',
      );
    }
    result = out;
  } else {
    result = NDArray.create(a.shape, targetDType);
  }

  void dispatchContiguous(NDArray src, NDArray dest) {
    switch (src.dtype) {
      case DType.float64:
        v_i0_double(src.pointer.cast(), dest.pointer.cast(), src.size);
        break;
      case DType.float32:
        v_i0_float(src.pointer.cast(), dest.pointer.cast(), src.size);
        break;
      case DType.complex128:
        v_i0_complex128(src.pointer.cast(), dest.pointer.cast(), src.size);
        break;
      case DType.complex64:
        v_i0_complex64(src.pointer.cast(), dest.pointer.cast(), src.size);
        break;
      default:
        throw UnsupportedError(
          'Unsupported dtype for i0 contiguous dispatch: ${src.dtype}',
        );
    }
  }

  void dispatchStrided(NDArray src, NDArray dest) {
    final rank = src.shape.length;
    final marker = ScratchArena.marker;
    final cShape = ScratchArena.copyInts(src.shape);
    final cStridesSrc = ScratchArena.copyInts(src.strides);
    final cStridesDest = ScratchArena.copyInts(dest.strides);
    try {
      switch (src.dtype) {
        case DType.float64:
          s_i0_double(
            src.pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cStridesDest,
            cShape,
            rank,
          );
          break;
        case DType.float32:
          s_i0_float(
            src.pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cStridesDest,
            cShape,
            rank,
          );
          break;
        case DType.complex128:
          s_i0_complex128(
            src.pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cStridesDest,
            cShape,
            rank,
          );
          break;
        case DType.complex64:
          s_i0_complex64(
            src.pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cStridesDest,
            cShape,
            rank,
          );
          break;
        default:
          throw UnsupportedError(
            'Unsupported dtype for i0 strided dispatch: ${src.dtype}',
          );
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }

  if (a.isContiguous && result.isContiguous) {
    dispatchContiguous(a, result);
  } else {
    final rank = a.shape.length;
    if (rank <= 8) {
      dispatchStrided(a, result);
    } else {
      final tempA = a.isContiguous ? a : a.copy();
      final tempResult = result.isContiguous
          ? result
          : NDArray.create(result.shape, result.dtype);

      dispatchContiguous(tempA, tempResult);

      if (!identical(tempResult, result)) {
        tempResult.copy(out: result);
        tempResult.dispose();
      }
      if (!identical(tempA, a)) {
        tempA.dispose();
      }
    }
  }

  return result;
}
