// ignore_for_file: non_constant_identifier_names
import '../../ndarray.dart';
import '../../ndarray_bindings.dart';
import '../helpers.dart';

/// Returns the Hanning (Hann) window.
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

/// Returns the Hamming window.
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
