import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'package:pocketfft/pocketfft.dart';

void main() {
  const nfft = 4;
  print('=== PocketFFT 1D Complex FFT Example (N=$nfft) ===');

  // 1. Allocate forward transform configuration plan (0 = forward)
  final cfg = kiss_fft_alloc(nfft, 0, ffi.nullptr, ffi.nullptr);
  if (cfg.address == 0) {
    throw StateError('Failed to allocate FFT configuration plan.');
  }

  // 2. Allocate native memory for input and output complex arrays
  final fin = malloc<kiss_fft_cpx>(nfft);
  final fout = malloc<kiss_fft_cpx>(nfft);

  try {
    // 3. Initialize input signal: Kronecker delta impulse [1, 0, 0, 0]
    for (var i = 0; i < nfft; i++) {
      fin[i].r = (i == 0) ? 1.0 : 0.0;
      fin[i].i = 0.0;
    }

    print('Input signal:');
    for (var i = 0; i < nfft; i++) {
      print('  x[$i] = ${fin[i].r} + ${fin[i].i}i');
    }

    // 4. Compute Fast Fourier Transform
    kiss_fft(cfg, fin, fout);

    print('\nFFT Frequency Spectrum Output:');
    for (var i = 0; i < nfft; i++) {
      print('  X[$i] = ${fout[i].r.toStringAsFixed(2)} + ${fout[i].i.toStringAsFixed(2)}i');
    }
  } finally {
    // 5. Clean up allocated native memory
    malloc.free(cfg);
    malloc.free(fin);
    malloc.free(fout);
  }
}
