import 'package:test/test.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/fft.dart' as fft;
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Fast Fourier Transforms (gpuarray.fft)', () {
    test('1D Complex FFT and IFFT roundtrip (fft, ifft)', () {
      ResourceScope.scope(() {
        // [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        final signal = GpuArray.fromList(
          [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
          [8],
          DType.float64,
        );

        final spectrum = fft.fft(signal);
        expect(spectrum.shape, equals([8]));
        expect(spectrum.dtype, equals(DType.complex128));

        // Invert back
        final recovered = fft.ifft(spectrum);
        expect(recovered.shape, equals([8]));

        final recFlat = recovered.toNDArray();
        final recData = recFlat.toList().cast<Complex>();
        final origData = signal.toList().cast<double>();

        for (var i = 0; i < 8; i++) {
          expect(recData[i].real, closeTo(origData[i], 1e-4)); // Real part matches
          expect(recData[i].imag, closeTo(0.0, 1e-4)); // Imag part is 0
        }
      });
    });

    test('1D Real FFT and IRFFT (rfft, irfft)', () {
      ResourceScope.scope(() {
        final signal = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [8],
          DType.float64,
        );

        final rspec = fft.rfft(signal);
        expect(rspec.shape, equals([5])); // N/2 + 1

        final recovered = fft.irfft(rspec, n: 8);
        expect(recovered.shape, equals([8]));

        final recData = recovered.toList().cast<double>();
        final origData = signal.toList().cast<double>();

        for (var i = 0; i < 8; i++) {
          expect(recData[i], closeTo(origData[i], 1e-4));
        }
      });
    });

    test('2D FFT and IFFT (fft2, ifft2)', () {
      ResourceScope.scope(() {
        final matrix = GpuArray.fromList([
          1.0, 2.0,
          3.0, 4.0,
        ], [2, 2], DType.float64);

        final spec2d = fft.fft2(matrix);
        expect(spec2d.shape, equals([2, 2]));

        final rec2d = fft.ifft2(spec2d);
        expect(rec2d.shape, equals([2, 2]));

        final recFlat = rec2d.toNDArray();
        final recData = recFlat.toList().cast<Complex>();
        final origData = matrix.toList().cast<double>();

        for (var i = 0; i < 4; i++) {
          expect(recData[i].real, closeTo(origData[i], 1e-4));
          expect(recData[i].imag, closeTo(0.0, 1e-4));
        }
      });
    });

    test('Frequency utilities (fftfreq, rfftfreq, fftshift, ifftshift)', () {
      ResourceScope.scope(() {
        final freqs = fft.fftfreq(8, d: 0.1);
        expect(freqs.shape, equals([8]));
        final fList = freqs.toList().cast<double>();
        expect(fList[0], equals(0.0));
        expect(fList[1], closeTo(1.25, 1e-4));

        final rfreqs = fft.rfftfreq(8, d: 0.1);
        expect(rfreqs.shape, equals([5]));

        final x = GpuArray.fromList([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], [8], DType.float64);
        final shifted = fft.fftshift(x);
        final unshifted = fft.ifftshift(shifted);
        expect(unshifted.toList(), equals(x.toList()));
      });
    });
  });
}
