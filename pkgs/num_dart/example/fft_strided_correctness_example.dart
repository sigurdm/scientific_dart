import 'package:num_dart/num_dart.dart';
import 'dart:typed_data';

void main() {
  print(
    '=== NDArray Discrete Fourier Transform (FFT) Strides Correctness Examples ===\n',
  );

  runFFTStridedCorrectnessExample();
}

void runFFTStridedCorrectnessExample() {
  print('--- 1. Allocating Contiguous Signals Parent Array ---');
  // We create a 2D array of shape [2, 2]
  final parent = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
  print('Parent array data:\n$parent');
  print(
    'Parent strides: ${parent.strides}, isContiguous: ${parent.isContiguous}',
  );

  print('\n--- 2. Generating Transposed Non-Contiguous View ---');
  // Transposing swaps axis 0 and axis 1 -> yields [[1, 3], [2, 4]]
  final transposed = parent.transposed;
  print('Transposed array data:\n$transposed');
  print(
    'Transposed strides: ${transposed.strides}, isContiguous: ${transposed.isContiguous}',
  );

  print('\n--- 3. Executing FFT on Transposed Signals View ---');
  // Prior to our premium duplication fix, this silently corrupted FFI results!
  // Now, it automatically duplicates and processes transposed coordinates perfectly!
  final freqTransposed = fft(transposed);
  print('Transposed FFT results DType: ${freqTransposed.dtype}');
  print('Transposed FFT results flat data: ${freqTransposed.data}');

  print('\n--- 4. Comparing against Contiguous Equivalent Array ---');
  // Standard contiguous equivalent array [[1, 3], [2, 4]]
  final contiguous = NDArray.fromList(
    [1.0, 3.0, 2.0, 4.0],
    [2, 2],
    DType.float64,
  );
  final freqContig = fft(contiguous);
  print('Contiguous equivalent FFT flat data: ${freqContig.data}');

  // Let's check mathematical parity!
  var match = true;
  for (var i = 0; i < 4; i++) {
    final diffReal = (freqTransposed.data[i].real - freqContig.data[i].real)
        .abs();
    final diffImag = (freqTransposed.data[i].imag - freqContig.data[i].imag)
        .abs();
    if (diffReal > 1e-5 || diffImag > 1e-5) {
      match = false;
    }
  }
  print(
    '\n🏆 Numerical parity check between transposed and contiguous FFT: ${match ? "SUCCESS" : "FAILED"}',
  );
}
