import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';
import 'dart:math' as math;

void main() {
  group('linspace', () {
    test('standard linspace', () {
      final a = linspace(0.0, 1.0, 5);
      expect(a.shape, [5]);
      for (var i = 0; i < 5; i++) {
        expect(a.data[i], closeTo(i * 0.25, 1e-10));
      }
    });

    test('linspace without endpoint', () {
      final a = linspace(0.0, 1.0, 5, endpoint: false);
      expect(a.shape, [5]);
      for (var i = 0; i < 5; i++) {
        expect(a.data[i], closeTo(i * 0.2, 1e-10));
      }
    });

    test('integer linspace', () {
      final a = linspace<int>(0, 10, 5, dtype: DType.int64);
      expect(a.dtype, DType.int64);
      expect(a.data, [0, 2, 5, 7, 10]); // interpolated: 0, 2.5, 5, 7.5, 10 -> truncated to int
    });
  });

  group('logspace', () {
    test('standard logspace (base 10)', () {
      final a = logspace(0.0, 2.0, 3); // 10^0, 10^1, 10^2
      expect(a.shape, [3]);
      expect(a.data[0], closeTo(1.0, 1e-10));
      expect(a.data[1], closeTo(10.0, 1e-10));
      expect(a.data[2], closeTo(100.0, 1e-10));
    });

    test('logspace base 2', () {
      final a = logspace(0.0, 4.0, 5, base: 2.0); // 2^0, 2^1, 2^2, 2^3, 2^4
      expect(a.data, [1.0, 2.0, 4.0, 8.0, 16.0]);
    });

    test('logspace without endpoint', () {
      final a = logspace(0.0, 3.0, 3, endpoint: false); // 10^0, 10^1, 10^2
      expect(a.data[0], closeTo(1.0, 1e-10));
      expect(a.data[1], closeTo(10.0, 1e-10));
      expect(a.data[2], closeTo(100.0, 1e-10));
    });
  });

  group('geomspace', () {
    test('standard geomspace', () {
      final a = geomspace(1.0, 1000.0, 4);
      expect(a.data[0], closeTo(1.0, 1e-10));
      expect(a.data[1], closeTo(10.0, 1e-10));
      expect(a.data[2], closeTo(100.0, 1e-10));
      expect(a.data[3], closeTo(1000.0, 1e-10));
    });

    test('negative geomspace', () {
      final a = geomspace(-1.0, -1000.0, 4);
      expect(a.data[0], closeTo(-1.0, 1e-10));
      expect(a.data[1], closeTo(-10.0, 1e-10));
      expect(a.data[2], closeTo(-100.0, 1e-10));
      expect(a.data[3], closeTo(-1000.0, 1e-10));
    });

    test('geomspace without endpoint', () {
      final a = geomspace(1.0, 1000.0, 3, endpoint: false);
      expect(a.data[0], closeTo(1.0, 1e-10));
      expect(a.data[1], closeTo(10.0, 1e-10));
      expect(a.data[2], closeTo(100.0, 1e-10));
    });

    test('geomspace invalid signs', () {
      expect(() => geomspace(-1.0, 10.0, 5), throwsArgumentError);
      expect(() => geomspace(1.0, -10.0, 5), throwsArgumentError);
      expect(() => geomspace(0.0, 10.0, 5), throwsArgumentError);
    });
  });

  group('Complex spacers', () {
    test('linspace complex', () {
      final a = linspace<Complex>(Complex(0, 0), Complex(1, 1), 3);
      expect(a.dtype, DType.complex128);
      expect(a.data[0], Complex(0, 0));
      expect(a.data[1], Complex(0.5, 0.5));
      expect(a.data[2], Complex(1, 1));
    });

    test('logspace complex', () {
      final a = logspace<Complex>(
        Complex(0, 0),
        Complex(0, 2),
        3,
        base: Complex(10.0, 0.0),
      );
      expect(a.data[0], Complex(1, 0));
      final expected1 = Complex(math.cos(math.log(10)), math.sin(math.log(10)));
      final val1 = a.data[1];
      expect(val1.real, closeTo(expected1.real, 1e-10));
      expect(val1.imag, closeTo(expected1.imag, 1e-10));
    });

    test('geomspace complex', () {
      final a = geomspace<Complex>(Complex(1, 0), Complex(-1, 0), 3);
      expect(a.data[0], Complex(1, 0));
      final val1 = a.data[1];
      expect(val1.real, closeTo(0, 1e-10));
      expect(val1.imag, closeTo(1, 1e-10));
      final val2 = a.data[2];
      expect(val2.real, closeTo(-1, 1e-10));
      expect(val2.imag, closeTo(0, 1e-10));
    });
  });

  group('Broadcasting spacers', () {
    test('linspace broadcasting', () {
      final start = NDArray.fromList([0.0, 10.0], [2], DType.float64);
      final stop = NDArray.fromList([1.0, 11.0], [2], DType.float64);
      // axis=0 (default) -> [numSamples, 2]
      final a = linspaceGrid(start, stop, 3);
      expect(a.shape, [3, 2]);
      // row 0: [0, 10]
      // row 1: [0.5, 10.5]
      // row 2: [1, 11]
      expect(a.data, [0.0, 10.0, 0.5, 10.5, 1.0, 11.0]);
    });

    test('linspace axis support', () {
      final start = NDArray.fromList([0.0, 10.0], [2], DType.float64);
      final stop = NDArray.fromList([1.0, 11.0], [2], DType.float64);
      // axis=1 -> [2, numSamples]
      final a = linspaceGrid(start, stop, 3, axis: 1);
      expect(a.shape, [2, 3]);
      // row 0: [0, 0.5, 1]
      // row 1: [10, 10.5, 11]
      expect(a.data, [0.0, 0.5, 1.0, 10.0, 10.5, 11.0]);
    });

    test('linspaceGridWithStep', () {
      final start = NDArray.fromList([0.0, 10.0], [2], DType.float64);
      final stop = NDArray.fromList([1.0, 12.0], [2], DType.float64);
      final (a, step) = linspaceGridWithStep(start, stop, 3);
      expect(a.shape, [3, 2]);
      expect(step.shape, [2]);
      expect(step.data, [0.5, 1.0]);
    });
  });

  group('Step retrieval', () {
    test('linspaceWithStep', () {
      final (a, step) = linspaceWithStep(0.0, 10.0, 5);
      expect(step, 2.5);
      expect(a.data, [0.0, 2.5, 5.0, 7.5, 10.0]);
    });
  });
}
