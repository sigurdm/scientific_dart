# num_dart

[![Pub Version](https://img.shields.io/pub/v/num_dart)](https://pub.dev/packages/num_dart)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**num_dart** is a high-performance, FFI-accelerated N-dimensional array and numerical computing library for Dart, heavily inspired by NumPy. It features unmanaged C-heap memory allocations, blazingly fast vectorised ufuncs, parallelized builds, and native OpenBLAS-backed linear algebra solvers.

---

## Key Features

- 🚀 **FFI Native Acceleration**: Directly offloads vector operations, transcendental functions, and reductions to optimized C kernels, bypassing JIT overhead.
- 🎛️ **Zero-Copy Slices & Transposes**: Supports stride-based multi-dimensional array views, transposes, dimensions expansion (`expand_dims`), and squeezing (`squeeze`) with zero memory copies.
- 🧮 **OpenBLAS Linear Algebra**: Accelerated determinant (`det`), matrix inversion (`inv`), QR, SVD, and solvers (`solve`) powered by raw LAPACK.
- 📶 **Mixed-Radix pocketfft**: Fast Fourier Transform (`fft`) and Inverse FFT (`ifft`) mapped to PocketFFT unmanaged kernels.
- 🎲 **RNG Distributions**: High-performance Normal (Gaussian), Poisson, and Binomial sample generation.
- 🔍 **Data Sanitation & Tolerance Comparisons**: Approximate floating-point equality comparisons (`isclose`/`allclose`) and NaN/Infinities cleaning (`nan_to_num`).

---

## Getting Started: Scientific DSP Denoising Tutorial

Here is a comprehensive end-to-end example demonstrating how to generate a signal, inject RNG Gaussian noise, execute FFI Mixed-Radix FFT spectral mapping, apply a frequency low-pass filter, restore the time-domain signal via FFI IFFT, and verify reconstructed correctness using tolerance comparisons:

```dart
import 'dart:math' as math;
import 'package:num_dart/num_dart.dart';

void main() {
  print('=== NDArray Master Scientific & ML Signal Denoising Tutorial ===\n');

  // 1. Generate pure time-domain signal
  // Sample a 10 Hz sine wave at 128 Hz for 1 second (128 points)
  const samplingRate = 128.0;
  const numPoints = 128;

  final time = NDArray.create([numPoints], DType.float64);
  final pureSignal = NDArray.create([numPoints], DType.float64);

  final timeData = time.data;
  final pureData = pureSignal.data;

  for (var i = 0; i < numPoints; i++) {
    final t = i / samplingRate;
    timeData[i] = t;
    pureData[i] = 5.0 * math.sin(2.0 * math.pi * 10.0 * t); // 10Hz sine wave, amplitude 5
  }
  print('1. Generated pure 10 Hz sine wave signal.');

  // 2. Inject RNG Gaussian Noise to simulate measurement sensors
  final noise = normal([numPoints], loc: 0.0, scale: 0.5, random: math.Random(42));
  
  // Combine pure signal + noise element-wise!
  final noisySignal = add(pureSignal, noise);
  print('2. Injected RNG Gaussian noise (loc = 0.0, scale = 0.5) element-wise.');

  // 3. Execute FFI-Accelerated mixed-radix FFT to map signal to frequency space
  final fftCoeffs = fft(noisySignal);
  print('3. Executed mixed-radix FFI FFT to transform signal to frequency space.');

  // 4. Low-pass Filter: Filter out high frequencies (noise) above 15 Hz!
  final halfLen = numPoints ~/ 2;
  var filteredBinsCount = 0;

  for (var i = 0; i <= halfLen; i++) {
    final freq = i * samplingRate / numPoints;
    if (freq > 15.0) {
      // Symmetrically zero out positive and corresponding negative high frequency bins!
      fftCoeffs.data[i] = Complex(0.0, 0.0);
      if (i > 0 && i < halfLen) {
        fftCoeffs.data[numPoints - i] = Complex(0.0, 0.0);
      }
      filteredBinsCount++;
    }
  }
  print('4. Low-pass filter applied: zeroed out $filteredBinsCount noise bins (> 15 Hz).');

  // 5. Restoration: Inverse Fourier Transform (IFFT) back to time domain!
  final restoredComplex = ifft(fftCoeffs);

  // Extract real component of the reconstructed signal
  final reconstructed = NDArray.create([numPoints], DType.float64);
  final reconstructedData = reconstructed.data;
  for (var i = 0; i < numPoints; i++) {
    reconstructedData[i] = restoredComplex.data[i].real;
  }
  print('5. Executed FFI IFFT to restore signal back to time-domain.');

  // 6. Compare reconstructed signal against original pure signal!
  final isApproximatelyClose = allclose(pureSignal, reconstructed, rtol: 0.3, atol: 0.5);
  print('\n=== Verification Results ===');
  print('Is restored signal approximately close to pure signal? $isApproximatelyClose');

  // Cleanup unmanaged FFI heap memory!
  time.dispose();
  pureSignal.dispose();
  noise.dispose();
  noisySignal.dispose();
  fftCoeffs.dispose();
  restoredComplex.dispose();
  reconstructed.dispose();
}
```

---

## Installation

Add `num_dart` to your `pubspec.yaml`:

```yaml
dependencies:
  num_dart: ^0.0.1
```

### Compiling Native Libraries

`num_dart` relies on native compiled shared libraries for PocketFFT and custom universal ufuncs kernels.

To automatically build native bindings for your platform (Linux/macOS/Windows), run the automatic build script:
```bash
dart run openblas:build
```
This parallelizes compilation over available hardware CPU cores to complete setup in seconds.

---

## License

This project is licensed under the Apache License, Version 2.0 - see the [LICENSE](LICENSE) file for details.
