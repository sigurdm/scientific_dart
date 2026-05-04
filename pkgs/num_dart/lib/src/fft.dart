import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'package:pocketfft/pocketfft.dart';
import 'ndarray.dart';

/// Compute the 1D discrete Fourier Transform (FFT) along the last axis using native PocketFFT/KissFFT.
///
/// Transforms time-domain discrete sequences into frequency coefficients.
/// Output is **always a Complex array** (`DType.complex128` or `DType.complex64` depending on precision).
///
/// Supports arbitrary sequence lengths via high-speed mixed-radix prime factoring natively.
///
/// **Arguments:**
/// - [a]: The input signal [NDArray] (real or complex).
/// - [n]: Optional target transform length. If provided, the sequence along the last axis is truncated
///        or padded with zeros to length [n]. If omitted, defaults to the last axis size.
///
/// **Example:**
/// {@example /example/fft_example.dart lang=dart}
NDArray fft(NDArray a, {int? n}) {
  if (a.shape.isEmpty) {
    throw ArgumentError(
      'Cannot compute FFT on a 0-dimensional or empty scalar array',
    );
  }

  final lastAxisDim = a.shape.last;
  final targetLen = n ?? lastAxisDim;
  if (targetLen <= 0) {
    throw ArgumentError(
      'Target transform length [n] must be greater than 0 (was $n)',
    );
  }

  // Formulate output shape by replacing the last axis with targetLen
  final outShape = List<int>.from(a.shape);
  outShape[outShape.length - 1] = targetLen;

  final targetDType = (a.dtype == DType.float32 || a.dtype == DType.complex64)
      ? DType.complex64
      : DType.complex128;

  final result = NDArray.zeros(outShape, targetDType);

  // Count how many 1D row sub-signals exist to execute strided walks
  final totalElements = a.shape.reduce((x, y) => x * y);
  final signalsCount = totalElements ~/ lastAxisDim;

  // 1. Allocate the native mixed-radix FFT plan configuration (0 for forward transform)
  final cfg = kiss_fft_alloc(targetLen, 0, ffi.nullptr, ffi.nullptr);
  if (cfg.address == 0) {
    throw StateError(
      'Failed to allocate native FFT plan for length $targetLen',
    );
  }

  // 2. Allocate C-heap unmanaged struct array buffers
  final pin = malloc<kiss_fft_cpx>(targetLen);
  final pout = malloc<kiss_fft_cpx>(targetLen);

  try {
    // Walk every independent 1D row signal along the last dimension axis
    for (var s = 0; s < signalsCount; s++) {
      final srcStart = s * lastAxisDim;
      final destStart = s * targetLen;

      // Populate input buffer, applying zero-padding or truncation if n is specified
      for (var i = 0; i < targetLen; i++) {
        if (i < lastAxisDim) {
          final val = a.data[srcStart + i];
          if (val is Complex) {
            pin[i].r = val.real;
            pin[i].i = val.imag;
          } else {
            pin[i].r = (val as num).toDouble();
            pin[i].i = 0.0;
          }
        } else {
          // Zero coefficient padding
          pin[i].r = 0.0;
          pin[i].i = 0.0;
        }
      }

      // 3. Fire high-speed native FFT on the C heap components
      kiss_fft(cfg, pin, pout);

      // 4. Collect results from pout back into Dart Complex tensor list buffer
      for (var i = 0; i < targetLen; i++) {
        result.data[destStart + i] = Complex(pout[i].r, pout[i].i);
      }
    }
  } finally {
    // 5. Release C heap allocations and planners to prevent memory leaks
    free(cfg.cast<ffi.Void>());
    malloc.free(pin);
    malloc.free(pout);
  }

  return result;
}

/// Compute the 1D Inverse discrete Fourier Transform (IFFT) along the last axis using native PocketFFT/KissFFT.
///
/// Transforms frequency domain coefficients back into time-domain complex signals.
/// Exposes standard `1 / N` normalization scaling automatically.
///
/// **Example:**
/// {@example /example/fft_example.dart lang=dart}
NDArray ifft(NDArray a, {int? n}) {
  if (a.shape.isEmpty) {
    throw ArgumentError(
      'Cannot compute IFFT on a 0-dimensional or empty scalar array',
    );
  }

  final lastAxisDim = a.shape.last;
  final targetLen = n ?? lastAxisDim;
  if (targetLen <= 0) {
    throw ArgumentError(
      'Target transform length [n] must be greater than 0 (was $n)',
    );
  }

  final outShape = List<int>.from(a.shape);
  outShape[outShape.length - 1] = targetLen;

  final targetDType = (a.dtype == DType.float32 || a.dtype == DType.complex64)
      ? DType.complex64
      : DType.complex128;

  final result = NDArray.zeros(outShape, targetDType);

  final totalElements = a.shape.reduce((x, y) => x * y);
  final signalsCount = totalElements ~/ lastAxisDim;

  // 1. Allocate native inverse plan configuration (1 for inverse transform)
  final cfg = kiss_fft_alloc(targetLen, 1, ffi.nullptr, ffi.nullptr);
  if (cfg.address == 0) {
    throw StateError(
      'Failed to allocate native IFFT plan for length $targetLen',
    );
  }

  final pin = malloc<kiss_fft_cpx>(targetLen);
  final pout = malloc<kiss_fft_cpx>(targetLen);

  try {
    for (var s = 0; s < signalsCount; s++) {
      final srcStart = s * lastAxisDim;
      final destStart = s * targetLen;

      for (var i = 0; i < targetLen; i++) {
        if (i < lastAxisDim) {
          final val = a.data[srcStart + i];
          if (val is Complex) {
            pin[i].r = val.real;
            pin[i].i = val.imag;
          } else {
            pin[i].r = (val as num).toDouble();
            pin[i].i = 0.0;
          }
        } else {
          pin[i].r = 0.0;
          pin[i].i = 0.0;
        }
      }

      // 2. Fire high-speed native inverse transform
      kiss_fft(cfg, pin, pout);

      // 3. Apply standard 1/N scaling factor normalization (KissFFT leaves it unscaled)
      final scaleFactor = 1.0 / targetLen;

      for (var i = 0; i < targetLen; i++) {
        result.data[destStart + i] = Complex(
          pout[i].r * scaleFactor,
          pout[i].i * scaleFactor,
        );
      }
    }
  } finally {
    free(cfg.cast<ffi.Void>());
    malloc.free(pin);
    malloc.free(pout);
  }

  return result;
}
