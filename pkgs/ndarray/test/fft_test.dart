import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';
import 'dart:typed_data';

void main() {
  group('NDArray Fast Fourier Transform (FFT & IFFT) Tensor Tests', () {
    test(
      'Kronecker Delta Spike 1D FFT yields perfectly uniform flat spectrum coefficients',
      () => NDArray.scope(() {
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
      }),
    );

    test(
      'Flat Constant DC Signal 1D FFT yields single spectral spike at zero frequency index',
      () => NDArray.scope(() {
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
      }),
    );

    test(
      'Verify complete FFT and IFFT full round-trip signal restoration (Power of Two)',
      () => NDArray.scope(() {
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
      }),
    );

    test(
      'Verify Mixed-Radix core support for non-power-of-two lengths (N=6 and N=5) without crashes!',
      () => NDArray.scope(() {
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
      }),
    );

    test(
      'Verify standard parameter [n] zero-coefficient padding and truncation behavior',
      () => NDArray.scope(() {
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
      }),
    );

    test(
      'Verify Float32/Complex64 precision branches, and IFFT with real input fallback',
      () => NDArray.scope(() {
        final a32 = NDArray.fromList(
          Float32List.fromList([1.0, 0.0, 0.0, 0.0]),
          [4],
          DType.float32,
        );
        final freq32 = fft(a32);
        expect(freq32.dtype, DType.complex64);
        expect(freq32.shape, [4]);

        final restored32 = ifft(freq32);
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
        final freq64 = fft(comp64);
        expect(freq64.dtype, DType.complex64);
        expect(freq64.shape, [4]);

        final restored64 = ifft(freq64);
        expect(restored64.dtype, DType.complex64);
        expect(restored64.shape, [4]);
        expect(restored64.data[0].real, closeTo(1.0, 1e-5));
        expect(restored64.data[0].imag, closeTo(2.0, 1e-5));

        final realSignal = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
          [4],
          DType.float64,
        );
        final restoredFromReal = ifft(realSignal);
        expect(restoredFromReal.dtype, DType.complex128);
        expect(restoredFromReal.shape, [4]);
      }),
    );

    test(
      'Verify fft() and ifft() on non-contiguous transposed strided views',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );

        final view = parent.transposed;
        expect(view.isContiguous, false);

        final freq = fft(view);
        expect(freq.shape, [2, 2]);

        final contiguous = NDArray.fromList(
          [1.0, 3.0, 2.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final freqContig = fft(contiguous);

        for (var i = 0; i < 4; i++) {
          expect(freq.data[i].real, closeTo(freqContig.data[i].real, 1e-5));
          expect(freq.data[i].imag, closeTo(freqContig.data[i].imag, 1e-5));
        }

        final restored = ifft(freq);
        expect(restored.shape, [2, 2]);
        final restoredContig = ifft(freqContig);

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
      }),
    );

    test(
      'Verify fft() and ifft() throws StateError on native plan allocation failure',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          2,
        ], DType.float64);

        final hugeN = 10000000000000;

        expect(() => fft(a, n: hugeN), throwsA(anything));
        expect(() => ifft(a, n: hugeN), throwsA(anything));
      }),
    );

    group('Multi-dimensional FFT with axis parameter', () {
      test(
        '2D FFT along axis 0 (columns)',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float64,
          );
          final freq = fft(a, axis: 0).copy();
          expect(freq.shape, [2, 2]);
          expect(freq.dtype, DType.complex128);
          expect(freq.data[0].real, closeTo(4.0, 1e-9));
          expect(freq.data[1].real, closeTo(6.0, 1e-9));
          expect(freq.data[2].real, closeTo(-2.0, 1e-9));
          expect(freq.data[3].real, closeTo(-2.0, 1e-9));
        }),
      );

      test(
        '2D FFT along axis 1 (rows)',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float64,
          );
          final freq = fft(a, axis: 1).copy();
          expect(freq.shape, [2, 2]);
          expect(freq.dtype, DType.complex128);
          expect(freq.data[0].real, closeTo(3.0, 1e-9));
          expect(freq.data[1].real, closeTo(-1.0, 1e-9));
          expect(freq.data[2].real, closeTo(7.0, 1e-9));
          expect(freq.data[3].real, closeTo(-1.0, 1e-9));
        }),
      );

      test(
        '2D IFFT along axis 0',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(4.0, 0.0),
              Complex(6.0, 0.0),
              Complex(-2.0, 0.0),
              Complex(-2.0, 0.0),
            ],
            [2, 2],
            DType.complex128,
          );
          final restored = ifft(a, axis: 0).copy();
          expect(restored.shape, [2, 2]);
          expect(restored.data[0].real, closeTo(1.0, 1e-9));
          expect(restored.data[1].real, closeTo(2.0, 1e-9));
          expect(restored.data[2].real, closeTo(3.0, 1e-9));
          expect(restored.data[3].real, closeTo(4.0, 1e-9));
        }),
      );
    });

    group('fftshift and ifftshift spectrum shifting tests', () {
      test(
        '1D Odd Length (N=5) fftshift & ifftshift round-trip',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5],
            DType.float64,
          );

          final shifted = fftshift(a);
          expect(shifted.toList(), [3.0, 4.0, 0.0, 1.0, 2.0]);

          final restored = ifftshift(shifted);
          expect(restored.toList(), [0.0, 1.0, 2.0, 3.0, 4.0]);
        }),
      );

      test(
        '1D Even Length (N=6) fftshift & ifftshift round-trip',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [6],
            DType.float64,
          );

          final shifted = fftshift(a);
          expect(shifted.toList(), [3.0, 4.0, 5.0, 0.0, 1.0, 2.0]);

          final restored = ifftshift(shifted);
          expect(restored.toList(), [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        }),
      );

      test(
        '2D Matrix shifting along all axes',
        () => NDArray.scope(() {
          final grid = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );

          final shifted = fftshift(grid);
          expect(shifted.toList(), [6.0, 4.0, 5.0, 3.0, 1.0, 2.0]);

          final restored = ifftshift(shifted);
          expect(restored.toList(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        }),
      );

      test(
        '2D Matrix shifting along specific axis (axis 0)',
        () => NDArray.scope(() {
          final grid = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );

          final shifted = fftshift(grid, axes: 0);
          expect(shifted.toList(), [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
        }),
      );

      test(
        'fftshift / ifftshift extension methods on NDArray',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [0.0, 1.0, 2.0, 3.0, 4.0],
            [5],
            DType.float64,
          );

          final shifted = fftshift(a);
          expect(shifted.toList(), [3.0, 4.0, 0.0, 1.0, 2.0]);

          final restored = ifftshift(shifted);
          expect(restored.toList(), [0.0, 1.0, 2.0, 3.0, 4.0]);
        }),
      );

      test(
        'Preconditions & error handling',
        () => NDArray.scope(() {
          final a = NDArray.fromList([0.0, 1.0], [2], DType.float64);

          expect(() => fftshift(a, axes: 2), throwsRangeError);
          expect(() => fftshift(a, axes: -3), throwsRangeError);
          expect(() => fftshift(a, axes: [0, 0]), throwsArgumentError);
          expect(() => fftshift(a, axes: 'invalid'), throwsArgumentError);

          a.dispose();
          expect(() => fftshift(a), throwsStateError);
        }),
      );
    });
  });
}
