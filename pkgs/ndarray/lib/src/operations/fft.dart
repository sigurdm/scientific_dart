// ignore_for_file: non_constant_identifier_names
import 'package:pocketfft/pocketfft.dart';
import '../ndarray.dart';
import 'dart:ffi' as ffi;
import '../scratch_arena.dart';

// Standalone operational relative cross-imports
import 'math.dart';

/// Computes the 1D discrete Fourier Transform (FFT) along the specified [axis].
///
/// Transforms discrete sequences from the time/space domain into frequency coefficients
/// using the standard Discrete Fourier Transform (DFT) formula:
///
///   X_k = ∑_{n=0}^{N-1} x_n * e^(-i * 2 * π * k * n / N)
///
/// The resulting array is **always complex** (DType.complex128 or DType.complex64 depending on precision).
///
/// Natively offloads computation to pocketfft's pocketfft/KissFFT mixed-radix prime factoring, supporting
/// arbitrary non-power-of-two sequence lengths at high speeds.
///
/// **Preconditions:**
/// - Input [a] must have rank $\ge 1$ (not empty or 0-dimensional).
/// - The specified [axis] must be within valid rank boundaries `[-a.rank, a.rank - 1]`.
/// - If provided, the target length [n] must be greater than 0.
///
/// **Throws:**
/// - [ArgumentError] if the input array shape is empty (scalar 0D).
/// - [RangeError] if the specified [axis] is out of bounds.
/// - [ArgumentError] if [n] is provided but is less than or equal to 0.
/// - [StateError] if native FFI memory allocations or KissFFT plan allocation (`kiss_fft_alloc`) fail.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N \log N)$ where $N$ is the transform length, scaling linearly even for prime lengths.
/// - Swapping the selected [axis] to the final dimension is a zero-copy, zero-allocation transpose view ($O(1)$ complexity).
///
/// **Example:**
/// {@example /example/fft_example.dart lang=dart}
///
/// Reference: [Cooley-Tukey FFT Algorithm](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm)
NDArray fft(NDArray a, {int? n, int axis = -1}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute fft() on a disposed array.');
  }
  if (a.shape.isEmpty) {
    throw ArgumentError(
      'Cannot compute FFT on a 0-dimensional or empty scalar array',
    );
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw RangeError.range(axis, -rank, rank - 1, 'axis');
  }

  if (normAxis != rank - 1) {
    final axes = List.generate(rank, (i) => i);
    axes[normAxis] = rank - 1;
    axes[rank - 1] = normAxis;

    final transposedInput = a.transpose(axes);
    final transposedResult = fft(transposedInput, n: n);
    return transposedResult.transpose(axes);
  }

  final NDArray inputA;
  final bool wasCopied;
  if (!a.isContiguous) {
    inputA = a.copy();
    wasCopied = true;
  } else {
    inputA = a;
    wasCopied = false;
  }

  final lastAxisDim = inputA.shape.last;
  final targetLen = n ?? lastAxisDim;
  if (targetLen <= 0) {
    if (wasCopied) {
      inputA.dispose();
    }
    throw ArgumentError(
      'Target transform length [n] must be greater than 0 (was $n)',
    );
  }

  // Formulate output shape by replacing the last axis with targetLen
  final outShape = List<int>.from(inputA.shape);
  outShape[outShape.length - 1] = targetLen;

  final targetDType =
      (inputA.dtype == DType.float32 || inputA.dtype == DType.complex64)
      ? DType.complex64
      : DType.complex128;

  final result = NDArray.zeros(outShape, targetDType);

  // Count how many 1D row sub-signals exist to execute strided walks
  final totalElements = inputA.shape.reduce((x, y) => x * y);
  final signalsCount = totalElements ~/ lastAxisDim;

  final isZeroCopyFastPath =
      inputA.dtype == DType.complex128 &&
      targetLen == lastAxisDim &&
      inputA.isContiguous;

  kiss_fft_cfg cfg = ffi.nullptr.cast();

  if (isZeroCopyFastPath) {
    final marker = ScratchArena.marker;
    try {
      final lenmem = ScratchArena.allocate<ffi.Size>(ffi.sizeOf<ffi.Size>());
      lenmem[0] = 0;
      kiss_fft_alloc(targetLen, 0, ffi.nullptr, lenmem);

      final mem = ScratchArena.allocate<ffi.Void>(lenmem[0]);
      cfg = kiss_fft_alloc(targetLen, 0, mem, lenmem);
      if (cfg.address == 0) {
        throw StateError(
          'Failed to allocate native FFT plan for length $targetLen',
        );
      }

      for (var s = 0; s < signalsCount; s++) {
        final rowPin = inputA.pointer.cast<kiss_fft_cpx>() + s * lastAxisDim;
        final rowPout = result.pointer.cast<kiss_fft_cpx>() + s * targetLen;
        kiss_fft(cfg, rowPin, rowPout);
      }
    } finally {
      ScratchArena.reset(marker);
      if (wasCopied) {
        inputA.dispose();
      }
    }
    return result;
  }

  final marker = ScratchArena.marker;
  ffi.Pointer<kiss_fft_cpx> pin = ffi.nullptr.cast();
  ffi.Pointer<kiss_fft_cpx> pout = ffi.nullptr.cast();

  try {
    cfg = kiss_fft_alloc(targetLen, 0, ffi.nullptr, ffi.nullptr);
    if (cfg.address == 0) {
      throw StateError(
        'Failed to allocate native FFT plan for length $targetLen',
      );
    }
    pin = ScratchArena.allocate<kiss_fft_cpx>(
      targetLen * ffi.sizeOf<kiss_fft_cpx>(),
    );
    pout = ScratchArena.allocate<kiss_fft_cpx>(
      targetLen * ffi.sizeOf<kiss_fft_cpx>(),
    );

    final copyLen = targetLen < lastAxisDim ? targetLen : lastAxisDim;
    for (var s = 0; s < signalsCount; s++) {
      final srcStart = s * lastAxisDim;
      final destStart = s * targetLen;

      // Populate input buffer, applying zero-padding or truncation if n is specified
      if (inputA.data is ComplexList) {
        final compList = inputA.data as ComplexList;
        for (var i = 0; i < copyLen; i++) {
          pin[i].r = compList.getReal(srcStart + i);
          pin[i].i = compList.getImag(srcStart + i);
        }
        final zeroPaddingStart = copyLen;
        for (var i = zeroPaddingStart; i < targetLen; i++) {
          pin[i].r = 0.0;
          pin[i].i = 0.0;
        }
      } else {
        for (var i = 0; i < copyLen; i++) {
          final val = inputA.data[srcStart + i];
          pin[i].r = (val as num).toDouble();
          pin[i].i = 0.0;
        }
        final zeroPaddingStart = copyLen;
        for (var i = zeroPaddingStart; i < targetLen; i++) {
          pin[i].r = 0.0;
          pin[i].i = 0.0;
        }
      }

      // 3. Fire high-speed native FFT on the C heap components
      kiss_fft(cfg, pin, pout);

      // 4. Collect results from pout back into Dart Complex tensor list buffer
      final compList = result.data as ComplexList;
      for (var i = 0; i < targetLen; i++) {
        compList.setRealImag(destStart + i, pout[i].r, pout[i].i);
      }
    }
  } finally {
    // 5. Release C heap allocations and planners to prevent memory leaks
    if (cfg.address != 0) {
      free(cfg.cast<ffi.Void>());
    }

    ScratchArena.reset(marker);
    if (wasCopied) {
      inputA.dispose();
    }
  }

  return result;
}

/// Computes the 1D inverse discrete Fourier Transform (IFFT) along the specified [axis].
///
/// Transforms frequency domain coefficients back into complex time/space domain signals, applying
/// standard `1 / N` normalization scaling automatically:
///
///   x_n = (1 / N) * ∑_{k=0}^{N-1} X_k * e^(i * 2 * π * k * n / N)
///
/// The resulting array is **always complex** (DType.complex128 or DType.complex64 depending on precision).
///
/// **Preconditions:**
/// - Input [a] must have rank $\ge 1$ (not empty or 0-dimensional).
/// - The specified [axis] must be within valid rank boundaries `[-a.rank, a.rank - 1]`.
/// - If provided, the target length [n] must be greater than 0.
///
/// **Throws:**
/// - [ArgumentError] if the input array shape is empty (scalar 0D).
/// - [RangeError] if the specified [axis] is out of bounds.
/// - [ArgumentError] if [n] is provided but is less than or equal to 0.
/// - [StateError] if native FFI memory allocations or KissFFT plan allocation (`kiss_fft_alloc`) fail.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N \log N)$ where $N$ is the transform length.
/// - Swapping the selected [axis] to the final dimension is a zero-copy, zero-allocation transpose view ($O(1)$ complexity).
///
/// **Example:**
/// {@example /example/fft_example.dart lang=dart}
NDArray ifft(NDArray a, {int? n, int axis = -1}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute ifft() on a disposed array.');
  }
  if (a.shape.isEmpty) {
    throw ArgumentError(
      'Cannot compute IFFT on a 0-dimensional or empty scalar array',
    );
  }

  final rank = a.shape.length;
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw RangeError.range(axis, -rank, rank - 1, 'axis');
  }

  if (normAxis != rank - 1) {
    final axes = List.generate(rank, (i) => i);
    axes[normAxis] = rank - 1;
    axes[rank - 1] = normAxis;

    final transposedInput = a.transpose(axes);
    final transposedResult = ifft(transposedInput, n: n);
    return transposedResult.transpose(axes);
  }

  final NDArray inputA;
  final bool wasCopied;
  if (!a.isContiguous) {
    inputA = a.copy();
    wasCopied = true;
  } else {
    inputA = a;
    wasCopied = false;
  }

  final lastAxisDim = inputA.shape.last;
  final targetLen = n ?? lastAxisDim;
  if (targetLen <= 0) {
    if (wasCopied) {
      inputA.dispose();
    }
    throw ArgumentError(
      'Target transform length [n] must be greater than 0 (was $n)',
    );
  }

  final outShape = List<int>.from(inputA.shape);
  outShape[outShape.length - 1] = targetLen;

  final targetDType =
      (inputA.dtype == DType.float32 || inputA.dtype == DType.complex64)
      ? DType.complex64
      : DType.complex128;

  final result = NDArray.zeros(outShape, targetDType);

  final totalElements = inputA.shape.reduce((x, y) => x * y);
  final signalsCount = totalElements ~/ lastAxisDim;

  final isZeroCopyFastPath =
      inputA.dtype == DType.complex128 &&
      targetLen == lastAxisDim &&
      inputA.isContiguous;

  kiss_fft_cfg cfg = ffi.nullptr.cast();

  if (isZeroCopyFastPath) {
    final marker = ScratchArena.marker;
    try {
      final lenmem = ScratchArena.allocate<ffi.Size>(ffi.sizeOf<ffi.Size>());
      lenmem[0] = 0;
      kiss_fft_alloc(targetLen, 1, ffi.nullptr, lenmem);

      final mem = ScratchArena.allocate<ffi.Void>(lenmem[0]);
      cfg = kiss_fft_alloc(targetLen, 1, mem, lenmem);
      if (cfg.address == 0) {
        throw StateError(
          'Failed to allocate native IFFT plan for length $targetLen',
        );
      }

      final scaleFactor = 1.0 / targetLen;
      for (var s = 0; s < signalsCount; s++) {
        final rowPin = inputA.pointer.cast<kiss_fft_cpx>() + s * lastAxisDim;
        final rowPout = result.pointer.cast<kiss_fft_cpx>() + s * targetLen;
        kiss_fft(cfg, rowPin, rowPout);

        for (var i = 0; i < targetLen; i++) {
          rowPout[i].r *= scaleFactor;
          rowPout[i].i *= scaleFactor;
        }
      }
    } finally {
      ScratchArena.reset(marker);
      if (wasCopied) {
        inputA.dispose();
      }
    }
    return result;
  }

  final marker = ScratchArena.marker;
  ffi.Pointer<kiss_fft_cpx> pin = ffi.nullptr.cast();
  ffi.Pointer<kiss_fft_cpx> pout = ffi.nullptr.cast();

  try {
    cfg = kiss_fft_alloc(targetLen, 1, ffi.nullptr, ffi.nullptr);
    if (cfg.address == 0) {
      throw StateError(
        'Failed to allocate native IFFT plan for length $targetLen',
      );
    }
    pin = ScratchArena.allocate<kiss_fft_cpx>(
      targetLen * ffi.sizeOf<kiss_fft_cpx>(),
    );
    pout = ScratchArena.allocate<kiss_fft_cpx>(
      targetLen * ffi.sizeOf<kiss_fft_cpx>(),
    );

    final copyLen = targetLen < lastAxisDim ? targetLen : lastAxisDim;
    for (var s = 0; s < signalsCount; s++) {
      final srcStart = s * lastAxisDim;
      final destStart = s * targetLen;

      if (inputA.data is ComplexList) {
        final compList = inputA.data as ComplexList;
        for (var i = 0; i < copyLen; i++) {
          pin[i].r = compList.getReal(srcStart + i);
          pin[i].i = compList.getImag(srcStart + i);
        }
        final zeroPaddingStart = copyLen;
        for (var i = zeroPaddingStart; i < targetLen; i++) {
          pin[i].r = 0.0;
          pin[i].i = 0.0;
        }
      } else {
        for (var i = 0; i < copyLen; i++) {
          final val = inputA.data[srcStart + i];
          pin[i].r = (val as num).toDouble();
          pin[i].i = 0.0;
        }
        final zeroPaddingStart = copyLen;
        for (var i = zeroPaddingStart; i < targetLen; i++) {
          pin[i].r = 0.0;
          pin[i].i = 0.0;
        }
      }

      // 2. Fire high-speed native inverse transform
      kiss_fft(cfg, pin, pout);

      // 3. Apply standard 1/N scaling factor normalization (KissFFT leaves it unscaled)
      final scaleFactor = 1.0 / targetLen;

      final compList = result.data as ComplexList;
      for (var i = 0; i < targetLen; i++) {
        compList.setRealImag(
          destStart + i,
          pout[i].r * scaleFactor,
          pout[i].i * scaleFactor,
        );
      }
    }
  } finally {
    if (cfg.address != 0) {
      free(cfg.cast<ffi.Void>());
    }

    ScratchArena.reset(marker);
    if (wasCopied) {
      inputA.dispose();
    }
  }

  return result;
}

/// Shifts the zero-frequency component to the center of the spectrum.
///
/// Swaps half-spaces along all specified [axes].
/// This is extremely useful for re-positioning frequency components so that the zero-frequency (DC)
/// component is centered in the middle of the grid/spectrum.
///
/// **Preconditions:**
/// - Input [a] must not be disposed.
/// - If [axes] is a list of integers, all elements must be unique and within valid range `[-a.rank, a.rank - 1]`.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [axes] contains duplicate indices or invalid types.
/// - [RangeError] if any axis index is out of bounds.
///
/// **Performance considerations:**
/// - Algorithmic Time Complexity is $O(N)$ where $N$ is the total number of elements.
/// - Space Complexity is $O(N)$ as it allocates a new array and copies elements to construct the shifted spectrum.
///
/// **Example:**
/// {@example /example/fftshift_example.dart lang=dart}
///
/// Reference: [NumPy fftshift](https://numpy.org/doc/stable/reference/generated/numpy.fft.fftshift.html)
NDArray<T> fftshift<T extends Object>(NDArray<T> a, {dynamic axes}) {
  if (a.isDisposed) {
    throw StateError('Cannot shift a disposed array.');
  }

  final rank = a.rank;
  if (rank == 0) {
    return a.copy();
  }

  // Resolve axes
  final List<int> resolvedAxes;
  if (axes == null) {
    resolvedAxes = List.generate(rank, (i) => i);
  } else if (axes is int) {
    final norm = axes < 0 ? rank + axes : axes;
    if (norm < 0 || norm >= rank) {
      throw RangeError.range(axes, -rank, rank - 1, 'axes');
    }
    resolvedAxes = [norm];
  } else if (axes is List<int>) {
    resolvedAxes = [];
    for (var axis in axes) {
      final norm = axis < 0 ? rank + axis : axis;
      if (norm < 0 || norm >= rank) {
        throw RangeError.range(axis, -rank, rank - 1, 'axes');
      }
      if (resolvedAxes.contains(norm)) {
        throw ArgumentError('Duplicate axis $norm specified in axes.');
      }
      resolvedAxes.add(norm);
    }
  } else {
    throw ArgumentError(
      'axes must be null, an integer, or a list of integers.',
    );
  }

  return NDArray.scope(() {
    NDArray<T> current = a;
    for (final axis in resolvedAxes) {
      final dimSize = current.shape[axis];
      final shift = dimSize ~/ 2;
      current = roll(current, shift, axis: axis);
    }
    return current.copy().detachToParentScope();
  });
}

/// Inverse of [fftshift].
///
/// Shifts the zero-frequency component back to the beginning of the spectrum.
///
/// **Preconditions:**
/// - Input [a] must not be disposed.
/// - If [axes] is a list of integers, all elements must be unique and within valid range `[-a.rank, a.rank - 1]`.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [axes] contains duplicate indices or invalid types.
/// - [RangeError] if any axis index is out of bounds.
///
/// **Performance considerations:**
/// - Algorithmic Time Complexity is $O(N)$ where $N$ is the total number of elements.
/// - Space Complexity is $O(N)$ as it allocates a new array and copies elements to construct the shifted spectrum.
///
/// **Example:**
/// {@example /example/fftshift_example.dart lang=dart}
///
/// Reference: [NumPy ifftshift](https://numpy.org/doc/stable/reference/generated/numpy.fft.ifftshift.html)
NDArray<T> ifftshift<T extends Object>(NDArray<T> a, {dynamic axes}) {
  if (a.isDisposed) {
    throw StateError('Cannot shift a disposed array.');
  }

  final rank = a.rank;
  if (rank == 0) {
    return a.copy();
  }

  // Resolve axes
  final List<int> resolvedAxes;
  if (axes == null) {
    resolvedAxes = List.generate(rank, (i) => i);
  } else if (axes is int) {
    final norm = axes < 0 ? rank + axes : axes;
    if (norm < 0 || norm >= rank) {
      throw RangeError.range(axes, -rank, rank - 1, 'axes');
    }
    resolvedAxes = [norm];
  } else if (axes is List<int>) {
    resolvedAxes = [];
    for (var axis in axes) {
      final norm = axis < 0 ? rank + axis : axis;
      if (norm < 0 || norm >= rank) {
        throw RangeError.range(axis, -rank, rank - 1, 'axes');
      }
      if (resolvedAxes.contains(norm)) {
        throw ArgumentError('Duplicate axis $norm specified in axes.');
      }
      resolvedAxes.add(norm);
    }
  } else {
    throw ArgumentError(
      'axes must be null, an integer, or a list of integers.',
    );
  }

  return NDArray.scope(() {
    NDArray<T> current = a;
    for (final axis in resolvedAxes) {
      final dimSize = current.shape[axis];
      final shift = (dimSize + 1) ~/ 2;
      current = roll(current, shift, axis: axis);
    }
    return current.copy().detachToParentScope();
  });
}
