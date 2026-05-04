import 'package:num_dart/num_dart.dart';
import 'package:test/test.dart';
import 'dart:typed_data';

void main() {
  group('NDArray Fast Fourier Transform (FFT & IFFT) Tensor Tests', () {
    test(
      'Kronecker Delta Spike 1D FFT yields perfectly uniform flat spectrum coefficients',
      () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 0.0, 0.0, 0.0]), [
          4,
        ], DType.float64);
        final freq = fft(a);

        expect(freq.shape, [4]);
        expect(freq.dtype, DType.complex128);

        for (var i = 0; i < 4; i++) {
          final Complex c = freq.data[i];
          expect(c.real, closeTo(1.0, 1e-9));
          expect(c.imag, closeTo(0.0, 1e-9));
        }
      },
    );

    test(
      'Flat Constant DC Signal 1D FFT yields single spectral spike at zero frequency index',
      () {
        final a = NDArray.fromList(Float64List.fromList([2.0, 2.0, 2.0, 2.0]), [
          4,
        ], DType.float64);
        final freq = fft(a);

        final Complex dcComponent = freq.data[0];
        expect(dcComponent.real, closeTo(8.0, 1e-9));
        expect(dcComponent.imag, closeTo(0.0, 1e-9));

        for (var i = 1; i < 4; i++) {
          final Complex c = freq.data[i];
          expect(c.real, closeTo(0.0, 1e-9));
          expect(c.imag, closeTo(0.0, 1e-9));
        }
      },
    );

    test(
      'Verify complete FFT and IFFT full round-trip signal restoration (Power of Two)',
      () {
        final a = NDArray.fromList(
          Float64List.fromList([1.5, -2.0, 3.5, 4.0, 0.5, -1.0, 2.0, -0.5]),
          [8],
          DType.float64,
        );

        final freq = fft(a);
        final restored = ifft(freq);

        expect(restored.shape, [8]);
        expect(restored.dtype, DType.complex128);

        for (var i = 0; i < 8; i++) {
          final Complex c = restored.data[i];
          expect(c.real, closeTo(a.data[i], 1e-9));
          expect(c.imag, closeTo(0.0, 1e-9));
        }
      },
    );

    test(
      'Verify Mixed-Radix core support for non-power-of-two lengths (N=6 and N=5) without crashes!',
      () {
        final a6 = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
          [6],
          DType.float64,
        );
        final freq6 = fft(a6);
        expect(freq6.shape, [6]);

        final restored6 = ifft(freq6);
        for (var i = 0; i < 6; i++) {
          expect(restored6.data[i].real, closeTo(a6.data[i], 1e-9));
        }

        final a5 = NDArray.fromList(
          Float64List.fromList([1.0, 0.0, 1.0, 0.0, 1.0]),
          [5],
          DType.float64,
        );
        final freq5 = fft(a5);
        expect(freq5.shape, [5]);

        final restored5 = ifft(freq5);
        expect(restored5.shape, [5]);
        expect(restored5.data[4].real, closeTo(1.0, 1e-9));
      },
    );

    test(
      'Verify standard parameter [n] zero-coefficient padding and truncation behavior',
      () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          2,
        ], DType.float64);
        final paddedFreq = fft(a, n: 4);
        expect(paddedFreq.shape, [4]);

        final restored = ifft(paddedFreq);
        expect(restored.shape, [4]);
        expect(restored.data[0].real, closeTo(1.0, 1e-9));
        expect(restored.data[1].real, closeTo(2.0, 1e-9));
        expect(restored.data[2].real, closeTo(0.0, 1e-9));
        expect(restored.data[3].real, closeTo(0.0, 1e-9));
      },
    );
  });
}
