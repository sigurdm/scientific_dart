import os

workspace_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
path = os.path.join(workspace_root, "lib/src/operations/math/windows.dart")

new_content = """// ignore_for_file: non_constant_identifier_names
import '../../ndarray.dart';
import '../../ndarray_bindings.dart';
import '../helpers.dart';

/// Returns the Hanning (Hann) window.
///
/// The Hanning window is a taper formed by using a weighted cosine:
///
/// $$w[n] = 0.5 - 0.5 \\cos\\left(\\frac{2\\pi n}{M - 1}\\right), \\quad 0 \\le n \\le M-1$$
///
/// Unlike the Hamming window, the Hanning window tapers all the way to exactly
/// **zero** at the boundaries ($w[0] = w[M-1] = 0.0$). It features a fast side-lobe
/// roll-off rate of $18 \\text{ dB/octave}$, making it highly suitable for general
/// spectral analysis where suppression of distant side lobes is critical.
///
/// **Preconditions:**
/// - If [out] is provided, it must not be disposed and must have shape `[M]` (or `[0]` if `M < 1`)
///   and matching [DType].
///
/// **Throws:**
/// - It is an error if [out] is disposed.
/// - It is an error if [out] has incompatible shape or [DType].
///
/// **Example:**
/// ```dart
/// final window = hanning(512);
/// ```
NDArray<T> hanning<T>(int M, {DType<T>? dtype, NDArray<T>? out}) {
  if (out != null && out.isDisposed) {
    throw StateError('Cannot execute hanning() on a disposed out buffer.');
  }
  final resolvedDType = dtype ?? (out?.dtype ?? (DType.float64 as DType<T>));
  final targetShape = [M < 1 ? 0 : M];
  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != resolvedDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for hanning.',
      );
    }
    result = out;
  } else {
    result = NDArray<T>.create(targetShape, resolvedDType);
  }

  if (M < 1) return result;
  if (M == 1) {
    result.fill(castValue(1.0, resolvedDType));
    return result;
  }

  if (resolvedDType == DType.float32) {
    if (result.isContiguous) {
      v_hanning_float(result.pointer.cast(), M);
    } else {
      final temp = NDArray<Float32>.create([M], DType.float32);
      v_hanning_float(temp.pointer.cast(), M);
      temp.copy(out: result as NDArray<Float32>);
      temp.dispose();
    }
    return result;
  } else if (resolvedDType == DType.float64) {
    if (result.isContiguous) {
      v_hanning_double(result.pointer.cast(), M);
    } else {
      final temp = NDArray<Float64>.create([M], DType.float64);
      v_hanning_double(temp.pointer.cast(), M);
      temp.copy(out: result as NDArray<Float64>);
      temp.dispose();
    }
    return result;
  } else {
    final temp = NDArray<Float64>.create([M], DType.float64);
    v_hanning_double(temp.pointer.cast(), M);
    final casted = castNDArray(temp, resolvedDType);
    casted.copy(out: result);
    temp.dispose();
    casted.dispose();
    return result;
  }
}

/// Returns the Hamming window.
///
/// The Hamming window is a taper formed by using an optimized weighted cosine:
///
/// $$w[n] = 0.54 - 0.46 \\cos\\left(\\frac{2\\pi n}{M - 1}\\right), \\quad 0 \\le n \\le M-1$$
///
/// Unlike the Hanning window, the Hamming window does not taper to zero at the boundaries,
/// leaving a small pedestal/discontinuity ($w[0] = w[M-1] = 0.08$). It is optimized to
/// minimize the maximum side-lobe level (achieving a first side lobe of $-43 \\text{ dB}$
/// compared to Hanning's $-32 \\text{ dB}$), at the expense of a slower side-lobe roll-off
/// rate of $6 \\text{ dB/octave}$.
///
/// **Preconditions:**
/// - If [out] is provided, it must not be disposed and must have shape `[M]` (or `[0]` if `M < 1`)
///   and matching [DType].
///
/// **Throws:**
/// - It is an error if [out] is disposed.
/// - It is an error if [out] has incompatible shape or [DType].
///
/// **Example:**
/// ```dart
/// final window = hamming(512);
/// ```
NDArray<T> hamming<T>(int M, {DType<T>? dtype, NDArray<T>? out}) {
  if (out != null && out.isDisposed) {
    throw StateError('Cannot execute hamming() on a disposed out buffer.');
  }
  final resolvedDType = dtype ?? (out?.dtype ?? (DType.float64 as DType<T>));
  final targetShape = [M < 1 ? 0 : M];
  final NDArray<T> result;
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != resolvedDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for hamming.',
      );
    }
    result = out;
  } else {
    result = NDArray<T>.create(targetShape, resolvedDType);
  }

  if (M < 1) return result;
  if (M == 1) {
    result.fill(castValue(1.0, resolvedDType));
    return result;
  }

  if (resolvedDType == DType.float32) {
    if (result.isContiguous) {
      v_hamming_float(result.pointer.cast(), M);
    } else {
      final temp = NDArray<Float32>.create([M], DType.float32);
      v_hamming_float(temp.pointer.cast(), M);
      temp.copy(out: result as NDArray<Float32>);
      temp.dispose();
    }
    return result;
  } else if (resolvedDType == DType.float64) {
    if (result.isContiguous) {
      v_hamming_double(result.pointer.cast(), M);
    } else {
      final temp = NDArray<Float64>.create([M], DType.float64);
      v_hamming_double(temp.pointer.cast(), M);
      temp.copy(out: result as NDArray<Float64>);
      temp.dispose();
    }
    return result;
  } else {
    final temp = NDArray<Float64>.create([M], DType.float64);
    v_hamming_double(temp.pointer.cast(), M);
    final casted = castNDArray(temp, resolvedDType);
    casted.copy(out: result);
    temp.dispose();
    casted.dispose();
    return result;
  }
}
"""

with open(path, "w") as f:
    f.write(new_content)
print("Updated windows.dart")
