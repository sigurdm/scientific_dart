import 'dart:math' as math;
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('angle', () {
    test('contiguous complex128', () {
      NDArray.scope(() {
        final a = NDArray<Complex128>.fromList(
          [
            Complex(1.0, 0.0),
            Complex(0.0, 1.0),
            Complex(-1.0, 0.0),
            Complex(0.0, -1.0),
            Complex(1.0, 1.0),
            Complex(-1.0, -1.0),
          ],
          [6],
          DType.complex128,
        );

        final res = angle(a);
        expect(res.dtype, DType.float64);
        expect(res.isContiguous, true);

        final expected = [
          0.0,
          math.pi / 2,
          math.pi,
          -math.pi / 2,
          math.pi / 4,
          -3 * math.pi / 4,
        ];

        for (var i = 0; i < res.size; i++) {
          expect(res.data[i], closeTo(expected[i], 1e-9));
        }
      });
    });

    test('contiguous complex64', () {
      NDArray.scope(() {
        final a = NDArray<Complex64>.fromList(
          [Complex(1.0, 0.0), Complex(0.0, 1.0)],
          [2],
          DType.complex64,
        );

        final res = angle(a);
        expect(res.dtype, DType.float32);
        expect(res.isContiguous, true);

        final expected = [0.0, math.pi / 2];
        for (var i = 0; i < res.size; i++) {
          expect(res.data[i], closeTo(expected[i], 1e-5));
        }
      });
    });

    test('strided complex128', () {
      NDArray.scope(() {
        final a = NDArray<Complex128>.fromList(
          [
            Complex(1.0, 0.0),
            Complex(99.0, 99.0),
            Complex(0.0, 1.0),
            Complex(99.0, 99.0),
          ],
          [2, 2],
          DType.complex128,
        );

        final sliced = a.slice([Slice.all(), Index(0)]);
        expect(sliced.isContiguous, false);

        final res = angle(sliced);
        expect(res.dtype, DType.float64);

        final expected = [0.0, math.pi / 2];
        expect(res.size, 2);
        for (var i = 0; i < res.size; i++) {
          expect(res.getCell([i]), closeTo(expected[i], 1e-9));
        }
      });
    });

    test('out parameter', () {
      NDArray.scope(() {
        final a = NDArray<Complex128>.fromList(
          [Complex(1.0, 0.0), Complex(0.0, 1.0)],
          [2],
          DType.complex128,
        );

        final out = NDArray<Float64>.zeros([2], DType.float64);
        final res = angle(a, out: out);
        expect(identical(res, out), true);
        expect(res.getCell([0]), closeTo(0.0, 1e-9));
        expect(res.getCell([1]), closeTo(math.pi / 2, 1e-9));
      });
    });

    test('invalid type', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList([1.0, 2.0], [2], DType.float64);
        expect(() => angle(a as dynamic), throwsArgumentError);
      });
    });
  });

  group('unwrap', () {
    test('1D float64 default discont', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [
            0.0,
            0.5 * math.pi,
            1.1 * math.pi,
            1.6 * math.pi,
            2.2 * math.pi,
            0.1 * math.pi,
          ],
          [6],
          DType.float64,
        );

        final res = unwrap(a);
        expect(res.dtype, DType.float64);

        final expected = [
          0.0,
          0.5 * math.pi,
          1.1 * math.pi,
          1.6 * math.pi,
          2.2 * math.pi,
          2.1 * math.pi,
        ];

        for (var i = 0; i < res.size; i++) {
          expect(res.data[i], closeTo(expected[i], 1e-9));
        }
      });
    });

    test('1D float64 custom discont', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [0.0, 1.2 * math.pi],
          [2],
          DType.float64,
        );

        final res1 = unwrap(a, discont: 1.5 * math.pi);
        expect(res1.getCell([1]), closeTo(1.2 * math.pi, 1e-9));

        final res2 = unwrap(a);
        expect(res2.getCell([1]), closeTo(-0.8 * math.pi, 1e-9));
      });
    });

    test('1D float32 custom discont', () {
      NDArray.scope(() {
        final a = NDArray<Float32>.fromList(
          [0.0, 1.2 * math.pi],
          [2],
          DType.float32,
        );

        final res1 = unwrap(a, discont: 1.5 * math.pi);
        expect(res1.getCell([1]), closeTo(1.2 * math.pi, 1e-6));

        final res2 = unwrap(a);
        expect(res2.getCell([1]), closeTo(-0.8 * math.pi, 1e-6));
      });
    });

    test('2D unwrap along axis 0', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [0.0, 0.0, 1.5 * math.pi, 1.5 * math.pi, 0.0, 0.0],
          [3, 2],
          DType.float64,
        );

        final res = unwrap(a, axis: 0);

        final expected = [0.0, 0.0, -0.5 * math.pi, -0.5 * math.pi, 0.0, 0.0];

        for (var i = 0; i < res.size; i++) {
          expect(res.data[i], closeTo(expected[i], 1e-9));
        }
      });
    });

    test('2D unwrap along axis 1', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [0.0, 1.5 * math.pi, 1.6 * math.pi, 0.0, 0.0, 0.0],
          [2, 3],
          DType.float64,
        );

        final res = unwrap(a, axis: 1);
        expect(res.shape, equals([2, 3]));

        expect(res.getCell([0, 0]), closeTo(0.0, 1e-9));
        expect(res.getCell([0, 1]), closeTo(-0.5 * math.pi, 1e-9));
        expect(res.getCell([0, 2]), closeTo(-0.4 * math.pi, 1e-9));

        expect(res.getCell([1, 0]), closeTo(0.0, 1e-9));
        expect(res.getCell([1, 1]), closeTo(0.0, 1e-9));
        expect(res.getCell([1, 2]), closeTo(0.0, 1e-9));
      });
    });

    test('unwrap boundary cases (pi and -pi)', () {
      NDArray.scope(() {
        // numpy.unwrap([0.0, pi]) -> [0.0, pi]
        // numpy.unwrap([0.0, -pi]) -> [0.0, -pi]
        final a = NDArray<Float64>.fromList([0.0, math.pi], [2], DType.float64);
        final res = unwrap(a);
        expect(res.data[1], closeTo(math.pi, 1e-9));

        final b = NDArray<Float64>.fromList(
          [0.0, -math.pi],
          [2],
          DType.float64,
        );
        final res2 = unwrap(b);
        expect(res2.data[1], closeTo(-math.pi, 1e-9));
      });
    });
  });
}
