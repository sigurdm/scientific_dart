// ignore_for_file: non_constant_identifier_names
import 'dart:math' as math;
import '../dtype.dart';
import '../gpu_array.dart';
import '../slice.dart';
import '../backend/compute_engine.dart';
import '../operations/manipulation.dart' as manip;

/// Internal helper for 1D Complex FFT (Cooley-Tukey / Bluestein).
void _fft1d(
  List<double> real,
  List<double> imag,
  int n, {
  required bool inverse,
}) {
  if (n <= 1) return;

  // Check if power of 2
  if ((n & (n - 1)) == 0) {
    // Bit reversal permutation
    var j = 0;
    for (var i = 0; i < n - 1; i++) {
      if (i < j) {
        final tr = real[i];
        real[i] = real[j];
        real[j] = tr;
        final ti = imag[i];
        imag[i] = imag[j];
        imag[j] = ti;
      }
      var k = n >> 1;
      while (k <= j) {
        j -= k;
        k >>= 1;
      }
      j += k;
    }

    // Cooley-Tukey radix-2 butterfly
    for (var len = 2; len <= n; len <<= 1) {
      final half = len >> 1;
      final angle = (inverse ? 2.0 : -2.0) * math.pi / len;
      final wstepR = math.cos(angle);
      final wstepI = math.sin(angle);

      for (var i = 0; i < n; i += len) {
        var wr = 1.0;
        var wi = 0.0;
        for (var k = 0; k < half; k++) {
          final uR = real[i + k];
          final uI = imag[i + k];
          final vR = real[i + k + half] * wr - imag[i + k + half] * wi;
          final vI = real[i + k + half] * wi + imag[i + k + half] * wr;

          real[i + k] = uR + vR;
          imag[i + k] = uI + vI;
          real[i + k + half] = uR - vR;
          imag[i + k + half] = uI - vI;

          final nextWr = wr * wstepR - wi * wstepI;
          wi = wr * wstepI + wi * wstepR;
          wr = nextWr;
        }
      }
    }
  } else {
    // Direct DFT for arbitrary N
    final outR = List<double>.filled(n, 0.0);
    final outI = List<double>.filled(n, 0.0);
    final sign = inverse ? 1.0 : -1.0;

    for (var k = 0; k < n; k++) {
      var sumR = 0.0;
      var sumI = 0.0;
      for (var t = 0; t < n; t++) {
        final angle = sign * 2.0 * math.pi * t * k / n;
        final cosA = math.cos(angle);
        final sinA = math.sin(angle);
        sumR += real[t] * cosA - imag[t] * sinA;
        sumI += real[t] * sinA + imag[t] * cosA;
      }
      outR[k] = sumR;
      outI[k] = sumI;
    }

    for (var i = 0; i < n; i++) {
      real[i] = outR[i];
      imag[i] = outI[i];
    }
  }

  if (inverse) {
    for (var i = 0; i < n; i++) {
      real[i] /= n;
      imag[i] /= n;
    }
  }
}

(double, double) _readComplexOrReal(dynamic val) {
  if (val is Complex) {
    return (val.real, val.imag);
  }
  if (val is num) {
    return (val.toDouble(), 0.0);
  }
  return (0.0, 0.0);
}

/// Computes the 1D Discrete Fourier Transform of [a] along [axis].
GpuArray<T> fft<T>(
  GpuArray a, {
  int? n,
  int axis = -1,
  String? norm,
}) {
  final normAxis = axis < 0 ? axis + a.rank : axis;
  final length = n ?? a.shape[normAxis];

  final outDType = (a.dtype == DType.complex64) ? DType.complex64 : DType.complex128;
  final outShape = List<int>.from(a.shape);
  outShape[normAxis] = length;

  final result = GpuArray<T>.empty(outShape, outDType as DType<T>, device: a.device);

  final inLength = a.shape[normAxis];
  final outerSize = a.shape.sublist(0, normAxis).fold(1, (r, e) => r * e);
  final innerSize = a.shape.sublist(normAxis + 1).fold(1, (r, e) => r * e);

  final aFlat = a.toNDArray();
  final aList = aFlat.toList();

  final realBuf = List<double>.filled(length, 0.0);
  final imagBuf = List<double>.filled(length, 0.0);

  for (var o = 0; o < outerSize; o++) {
    for (var i = 0; i < innerSize; i++) {
      for (var k = 0; k < length; k++) {
        if (k < inLength) {
          final srcIdx = (o * inLength + k) * innerSize + i;
          final (r, im) = _readComplexOrReal(aList[srcIdx]);
          realBuf[k] = r;
          imagBuf[k] = im;
        } else {
          realBuf[k] = 0.0;
          imagBuf[k] = 0.0;
        }
      }

      _fft1d(realBuf, imagBuf, length, inverse: false);

      if (norm == 'ortho') {
        final scale = 1.0 / math.sqrt(length);
        for (var k = 0; k < length; k++) {
          realBuf[k] *= scale;
          imagBuf[k] *= scale;
        }
      }

      for (var k = 0; k < length; k++) {
        final dstIdx = (o * length + k) * innerSize + i;
        ComputeEngine.writeAny(
          result.buffer,
          outDType,
          dstIdx,
          Complex(realBuf[k], imagBuf[k]),
        );
      }
    }
  }

  return result;
}

/// Computes the 1D Inverse Discrete Fourier Transform of [a] along [axis].
GpuArray<T> ifft<T>(
  GpuArray a, {
  int? n,
  int axis = -1,
  String? norm,
}) {
  final normAxis = axis < 0 ? axis + a.rank : axis;
  final length = n ?? a.shape[normAxis];

  final outDType =
      (a.dtype == DType.complex64) ? DType.complex64 : DType.complex128;
  final outShape = List<int>.from(a.shape);
  outShape[normAxis] = length;

  final result =
      GpuArray<T>.empty(outShape, outDType as DType<T>, device: a.device);

  final inLength = a.shape[normAxis];
  final outerSize = a.shape.sublist(0, normAxis).fold(1, (r, e) => r * e);
  final innerSize = a.shape.sublist(normAxis + 1).fold(1, (r, e) => r * e);

  final aFlat = a.toNDArray();
  final aList = aFlat.toList();

  final realBuf = List<double>.filled(length, 0.0);
  final imagBuf = List<double>.filled(length, 0.0);

  for (var o = 0; o < outerSize; o++) {
    for (var i = 0; i < innerSize; i++) {
      for (var k = 0; k < length; k++) {
        if (k < inLength) {
          final srcIdx = (o * inLength + k) * innerSize + i;
          final (r, im) = _readComplexOrReal(aList[srcIdx]);
          realBuf[k] = r;
          imagBuf[k] = im;
        } else {
          realBuf[k] = 0.0;
          imagBuf[k] = 0.0;
        }
      }

      _fft1d(realBuf, imagBuf, length, inverse: true);

      if (norm == 'ortho') {
        final scale = math.sqrt(length);
        for (var k = 0; k < length; k++) {
          realBuf[k] *= scale;
          imagBuf[k] *= scale;
        }
      }

      for (var k = 0; k < length; k++) {
        final dstIdx = (o * length + k) * innerSize + i;
        ComputeEngine.writeAny(
          result.buffer,
          outDType,
          dstIdx,
          Complex(realBuf[k], imagBuf[k]),
        );
      }
    }
  }

  return result;
}

/// Computes 1D real Discrete Fourier Transform of [a].
GpuArray<T> rfft<T>(
  GpuArray a, {
  int? n,
  int axis = -1,
  String? norm,
}) {
  final normAxis = axis < 0 ? axis + a.rank : axis;
  final length = n ?? a.shape[normAxis];
  final outLen = (length ~/ 2) + 1;

  final fullFft = fft(a, n: length, axis: normAxis, norm: norm);
  final sliceSpecs = List<dynamic>.generate(
    a.rank,
    (d) => d == normAxis ? Slice(0, outLen) : const All(),
  );
  return fullFft.slice(sliceSpecs) as GpuArray<T>;
}

/// Computes inverse of 1D real Discrete Fourier Transform.
GpuArray<T> irfft<T>(
  GpuArray a, {
  int? n,
  int axis = -1,
  String? norm,
}) {
  final normAxis = axis < 0 ? axis + a.rank : axis;
  final inLen = a.shape[normAxis];
  final outLen = n ?? ((inLen - 1) * 2);

  final fullComplexShape = List<int>.from(a.shape);
  fullComplexShape[normAxis] = outLen;

  final fullComplex =
      GpuArray.zeros(fullComplexShape, DType.complex128, device: a.device);

  final aFlat = a.toNDArray();
  final aList = aFlat.toList();
  final outerSize = a.shape.sublist(0, normAxis).fold(1, (r, e) => r * e);
  final innerSize = a.shape.sublist(normAxis + 1).fold(1, (r, e) => r * e);

  for (var o = 0; o < outerSize; o++) {
    for (var i = 0; i < innerSize; i++) {
      for (var k = 0; k < inLen; k++) {
        final srcIdx = (o * inLen + k) * innerSize + i;
        final (r, im) = _readComplexOrReal(aList[srcIdx]);
        final dstIdx = (o * outLen + k) * innerSize + i;
        ComputeEngine.writeAny(
          fullComplex.buffer,
          DType.complex128,
          dstIdx,
          Complex(r, im),
        );
      }
      for (var k = inLen; k < outLen; k++) {
        final mirror = outLen - k;
        final srcIdx = (o * inLen + mirror) * innerSize + i;
        final (r, im) = _readComplexOrReal(aList[srcIdx]);
        final dstIdx = (o * outLen + k) * innerSize + i;
        ComputeEngine.writeAny(
          fullComplex.buffer,
          DType.complex128,
          dstIdx,
          Complex(r, -im),
        );
      }
    }
  }

  aFlat.dispose();

  final ifftRes = ifft(fullComplex, n: outLen, axis: normAxis, norm: norm);
  final outDType = (a.dtype == DType.complex64) ? DType.float32 : DType.float64;
  final result = GpuArray<T>.empty(fullComplexShape, outDType as DType<T>, device: a.device);

  final ifftFlat = ifftRes.toNDArray();
  final ifftList = ifftFlat.toList();
  final totalElements = ShapeUtils.computeSize(fullComplexShape);

  for (var i = 0; i < totalElements; i++) {
    final (r, _) = _readComplexOrReal(ifftList[i]);
    ComputeEngine.writeValue(result.buffer, outDType, i, r);
  }

  ifftFlat.dispose();
  return result;
}

/// Computes 2D Discrete Fourier Transform of [a].
GpuArray<T> fft2<T>(
  GpuArray a, {
  List<int>? s,
  List<int> axes = const [-2, -1],
  String? norm,
}) {
  final ax1 = axes[0] < 0 ? axes[0] + a.rank : axes[0];
  final ax2 = axes[1] < 0 ? axes[1] + a.rank : axes[1];
  final s1 = s != null ? s[0] : null;
  final s2 = s != null ? s[1] : null;

  var res = fft(a, n: s1, axis: ax1, norm: norm);
  res = fft(res, n: s2, axis: ax2, norm: norm);
  return res as GpuArray<T>;
}

/// Computes 2D Inverse Discrete Fourier Transform of [a].
GpuArray<T> ifft2<T>(
  GpuArray a, {
  List<int>? s,
  List<int> axes = const [-2, -1],
  String? norm,
}) {
  final ax1 = axes[0] < 0 ? axes[0] + a.rank : axes[0];
  final ax2 = axes[1] < 0 ? axes[1] + a.rank : axes[1];
  final s1 = s != null ? s[0] : null;
  final s2 = s != null ? s[1] : null;

  var res = ifft(a, n: s1, axis: ax1, norm: norm);
  res = ifft(res, n: s2, axis: ax2, norm: norm);
  return res as GpuArray<T>;
}

/// Returns the DFT sample frequencies.
GpuArray<Float64> fftfreq(int n, {double d = 1.0}) {
  final freqs = List<double>.filled(n, 0.0);
  final factor = 1.0 / (d * n);
  final half = (n - 1) ~/ 2 + 1;

  for (var i = 0; i < half; i++) {
    freqs[i] = i * factor;
  }
  for (var i = half; i < n; i++) {
    freqs[i] = -(n - i) * factor;
  }

  return GpuArray.fromList(freqs, [n], DType.float64);
}

/// Returns the Real DFT sample frequencies.
GpuArray<Float64> rfftfreq(int n, {double d = 1.0}) {
  final outLen = (n ~/ 2) + 1;
  final freqs = List<double>.filled(outLen, 0.0);
  final factor = 1.0 / (d * n);

  for (var i = 0; i < outLen; i++) {
    freqs[i] = i * factor;
  }

  return GpuArray.fromList(freqs, [outLen], DType.float64);
}

/// Shifts zero-frequency component to center of spectrum.
GpuArray<T> fftshift<T>(GpuArray<T> x, {dynamic axes}) {
  if (axes == null) {
    var curr = x;
    for (var dim = 0; dim < x.rank; dim++) {
      final shift = x.shape[dim] ~/ 2;
      curr = manip.roll(curr, shift, axis: dim);
    }
    return curr;
  }
  if (axes is int) {
    final shift = x.shape[axes < 0 ? axes + x.rank : axes] ~/ 2;
    return manip.roll(x, shift, axis: axes);
  }
  if (axes is List<int>) {
    var curr = x;
    for (final ax in axes) {
      final normAx = ax < 0 ? ax + x.rank : ax;
      final shift = x.shape[normAx] ~/ 2;
      curr = manip.roll(curr, shift, axis: normAx);
    }
    return curr;
  }
  throw ArgumentError('Invalid axes for fftshift: $axes');
}

/// Inverse of [fftshift].
GpuArray<T> ifftshift<T>(GpuArray<T> x, {dynamic axes}) {
  if (axes == null) {
    var curr = x;
    for (var dim = 0; dim < x.rank; dim++) {
      final shift = -((x.shape[dim]) ~/ 2);
      curr = manip.roll(curr, shift, axis: dim);
    }
    return curr;
  }
  if (axes is int) {
    final shift = -((x.shape[axes < 0 ? axes + x.rank : axes]) ~/ 2);
    return manip.roll(x, shift, axis: axes);
  }
  if (axes is List<int>) {
    var curr = x;
    for (final ax in axes) {
      final normAx = ax < 0 ? ax + x.rank : ax;
      final shift = -((x.shape[normAx]) ~/ 2);
      curr = manip.roll(curr, shift, axis: normAx);
    }
    return curr;
  }
  throw ArgumentError('Invalid axes for ifftshift: $axes');
}
