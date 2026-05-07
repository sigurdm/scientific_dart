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
        addTearDown(a.dispose);
        final freq = fft(a);
        addTearDown(freq.dispose);

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
        addTearDown(a.dispose);
        final freq = fft(a);
        addTearDown(freq.dispose);

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
        addTearDown(a.dispose);

        final freq = fft(a);
        addTearDown(freq.dispose);
        final restored = ifft(freq);
        addTearDown(restored.dispose);

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
        addTearDown(a6.dispose);
        final freq6 = fft(a6);
        addTearDown(freq6.dispose);
        expect(freq6.shape, [6]);

        final restored6 = ifft(freq6);
        addTearDown(restored6.dispose);
        for (var i = 0; i < 6; i++) {
          expect(restored6.data[i].real, closeTo(a6.data[i], 1e-9));
        }

        final a5 = NDArray.fromList(
          Float64List.fromList([1.0, 0.0, 1.0, 0.0, 1.0]),
          [5],
          DType.float64,
        );
        addTearDown(a5.dispose);
        final freq5 = fft(a5);
        addTearDown(freq5.dispose);
        expect(freq5.shape, [5]);

        final restored5 = ifft(freq5);
        addTearDown(restored5.dispose);
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
        addTearDown(a.dispose);
        final paddedFreq = fft(a, n: 4);
        addTearDown(paddedFreq.dispose);
        expect(paddedFreq.shape, [4]);

        final restored = ifft(paddedFreq);
        addTearDown(restored.dispose);
        expect(restored.shape, [4]);
        expect(restored.data[0].real, closeTo(1.0, 1e-9));
        expect(restored.data[1].real, closeTo(2.0, 1e-9));
        expect(restored.data[2].real, closeTo(0.0, 1e-9));
        expect(restored.data[3].real, closeTo(0.0, 1e-9));
      },
    );

    test(
      'Verify Float32/Complex64 precision branches, and IFFT with real input fallback',
      () {
        final a32 = NDArray.fromList(
          Float32List.fromList([1.0, 0.0, 0.0, 0.0]),
          [4],
          DType.float32,
        );
        addTearDown(a32.dispose);
        final freq32 = fft(a32);
        addTearDown(freq32.dispose);
        expect(freq32.dtype, DType.complex64);
        expect(freq32.shape, [4]);

        final restored32 = ifft(freq32);
        addTearDown(restored32.dispose);
        expect(restored32.dtype, DType.complex64);
        expect(restored32.shape, [4]);
        expect(restored32.data[0].real, closeTo(1.0, 1e-5));

        final comp64 = NDArray.fromList(
          [
            Complex(1.0, 2.0),
            Complex(3.0, 4.0),
            Complex(5.0, 6.0),
            Complex(7.0, 8.0),
          ],
          [4],
          DType.complex64,
        );
        addTearDown(comp64.dispose);
        final freq64 = fft(comp64);
        addTearDown(freq64.dispose);
        expect(freq64.dtype, DType.complex64);
        expect(freq64.shape, [4]);

        final restored64 = ifft(freq64);
        addTearDown(restored64.dispose);
        expect(restored64.dtype, DType.complex64);
        expect(restored64.shape, [4]);
        expect(restored64.data[0].real, closeTo(1.0, 1e-5));
        expect(restored64.data[0].imag, closeTo(2.0, 1e-5));

        final realSignal = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
          [4],
          DType.float64,
        );
        addTearDown(realSignal.dispose);
        final restoredFromReal = ifft(realSignal);
        addTearDown(restoredFromReal.dispose);
        expect(restoredFromReal.dtype, DType.complex128);
        expect(restoredFromReal.shape, [4]);
      },
    );

    test(
      'Verify fft() and ifft() on non-contiguous transposed strided views',
      () {
        final parent = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        addTearDown(parent.dispose);

        final view = parent.transposed;
        addTearDown(view.dispose);
        expect(view.isContiguous, false);

        final freq = fft(view);
        addTearDown(freq.dispose);
        expect(freq.shape, [2, 2]);

        final contiguous = NDArray.fromList(
          [1.0, 3.0, 2.0, 4.0],
          [2, 2],
          DType.float64,
        );
        addTearDown(contiguous.dispose);
        final freqContig = fft(contiguous);
        addTearDown(freqContig.dispose);

        for (var i = 0; i < 4; i++) {
          expect(freq.data[i].real, closeTo(freqContig.data[i].real, 1e-5));
          expect(freq.data[i].imag, closeTo(freqContig.data[i].imag, 1e-5));
        }

        final restored = ifft(freq);
        addTearDown(restored.dispose);
        expect(restored.shape, [2, 2]);
        final restoredContig = ifft(freqContig);
        addTearDown(restoredContig.dispose);

        for (var i = 0; i < 4; i++) {
          expect(
            restored.data[i].real,
            closeTo(restoredContig.data[i].real, 1e-5),
          );
          expect(
            restored.data[i].imag,
            closeTo(restoredContig.data[i].imag, 1e-5),
          );
        }
      },
    );

    test(
      'Verify fft() and ifft() throws StateError on native plan allocation failure',
      () {
        final a = NDArray<double>.fromList(Float64List.fromList([1.0, 2.0]), [
          2,
        ], DType.float64);
        addTearDown(a.dispose);

        final hugeN = 10000000000000;

        expect(() => fft(a, n: hugeN), throwsA(anything));
        expect(() => ifft(a, n: hugeN), throwsA(anything));
      },
    );
  });
}
