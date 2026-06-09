import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Percentiles, Medians, and Quantiles Tests', () {
    group('Median Tests', () {
      test('Flat float64 median (odd size)', () {
        NDArray.scope(() {
          final a = NDArray.fromList([3.0, 1.0, 2.0], [3], DType.float64);
          final m = median(a);
          expect(m.shape, <int>[]);
          expect(m.dtype, DType.float64);
          expect(m.data[0], 2.0);
        });
      });

      test('Flat float64 median (even size)', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 3.0, 2.0, 4.0], [4], DType.float64);
          final m = median(a);
          expect(m.shape, <int>[]);
          expect(m.data[0], 2.5);
        });
      });

      test('2D float64 median along axis 0', () {
        NDArray.scope(() {
          // [[1, 5, 3],
          //  [4, 2, 6]]
          final a = NDArray.fromList(
            [1.0, 5.0, 3.0, 4.0, 2.0, 6.0],
            [2, 3],
            DType.float64,
          );
          final m = median(a, axis: 0);
          expect(m.shape, [3]);
          expect(m.toList(), [
            2.5,
            3.5,
            4.5,
          ]); // averages: (1+4)/2, (5+2)/2, (3+6)/2
        });
      });

      test('2D float64 median along axis 1', () {
        NDArray.scope(() {
          // [[1, 5, 3], -> sorted: 1, 3, 5 -> median: 3
          //  [4, 2, 6]] -> sorted: 2, 4, 6 -> median: 4
          final a = NDArray.fromList(
            [1.0, 5.0, 3.0, 4.0, 2.0, 6.0],
            [2, 3],
            DType.float64,
          );
          final m = median(a, axis: 1);
          expect(m.shape, [2]);
          expect(m.toList(), [3.0, 4.0]);
        });
      });

      test('Median with out parameter', () {
        NDArray.scope(() {
          final a = NDArray.fromList([3.0, 1.0, 2.0], [3], DType.float64);
          final out = NDArray<double>.zeros([], DType.float64);
          final m = median(a, out: out);
          expect(identical(m, out), true);
          expect(out.data[0], 2.0);
        });
      });

      test('Median integer types (int32)', () {
        NDArray.scope(() {
          final a = NDArray.fromList([3, 1, 2, 4], [4], DType.int32);
          final m = median(a);
          // 1, 2, 3, 4 -> (2+3)/2 = 2 (integer division in C kernel: (2+3)/2 = 2)
          expect(m.dtype, DType.int32);
          expect(m.data[0], 2);
        });
      });

      test('Median integer types (int64)', () {
        NDArray.scope(() {
          final a = NDArray.fromList([10, 30, 20], [3], DType.int64);
          final m = median(a);
          expect(m.dtype, DType.int64);
          expect(m.data[0], 20);
        });
      });

      test('Median integer types (uint8)', () {
        NDArray.scope(() {
          final a = NDArray.fromList([5, 1, 3, 4], [4], DType.uint8);
          final m = median(a);
          // 1, 3, 4, 5 -> (3+4)/2 = 3
          expect(m.dtype, DType.uint8);
          expect(m.data[0], 3);
        });
      });

      test('Median complex128 (independent real/imag)', () {
        NDArray.scope(() {
          final a = NDArray<Complex>.create([3], DType.complex128);
          a.data[0] = Complex(3.0, 1.0);
          a.data[1] = Complex(1.0, 9.0);
          a.data[2] = Complex(2.0, 5.0);
          // Reals: 3, 1, 2 -> median is 2
          // Imags: 1, 9, 5 -> median is 5
          final m = median(a);
          expect(m.dtype, DType.complex128);
          expect(m.data[0], Complex(2.0, 5.0));
        });
      });

      test('Median complex64 (independent real/imag)', () {
        NDArray.scope(() {
          final a = NDArray<Complex>.create([4], DType.complex64);
          a.data[0] = Complex(1.0, 4.0);
          a.data[1] = Complex(3.0, 1.0);
          a.data[2] = Complex(2.0, 2.0);
          a.data[3] = Complex(4.0, 3.0);
          // Reals: 1, 2, 3, 4 -> median: (2+3)/2 = 2.5
          // Imags: 1, 2, 3, 4 -> median: (2+3)/2 = 2.5
          final m = median(a);
          expect(m.dtype, DType.complex64);
          expect(m.data[0], Complex(2.5, 2.5));
        });
      });

      test('Median with negative axis', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 5.0, 3.0, 4.0, 2.0, 6.0],
            [2, 3],
            DType.float64,
          );
          // axis -1 is equivalent to axis 1
          final m = median(a, axis: -1);
          expect(m.shape, [2]);
          expect(m.toList(), [3.0, 4.0]);
        });
      });
    });

    group('Percentile Tests', () {
      test('Flat float64 percentile (interpolation)', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [15.0, 20.0, 35.0, 40.0, 50.0],
            [5],
            DType.float64,
          );
          // 0-indexed indices: 0: 15, 1: 20, 2: 35, 3: 40, 4: 50
          // q = 40 -> idx = 4 * 0.4 = 1.6
          // low = 1, high = 2
          // val = 20 + 0.6 * (35 - 20) = 20 + 0.6 * 15 = 20 + 9 = 29.0
          final p = percentile(a, 40.0);
          expect(p.shape, <int>[]);
          expect(p.dtype, DType.float64);
          expect(p.data[0], closeTo(29.0, 1e-9));
        });
      });

      test('Percentile 0 and 100', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [15.0, 20.0, 35.0, 40.0, 50.0],
            [5],
            DType.float64,
          );
          expect(percentile(a, 0.0).data[0], 15.0);
          expect(percentile(a, 100.0).data[0], 50.0);
        });
      });

      test('2D float32 percentile along axis 1', () {
        NDArray.scope(() {
          // [[1, 5, 3], -> sorted: 1, 3, 5. idx = 2 * 0.75 = 1.5. low=1, high=2. val = 3 + 0.5 * (5-3) = 4.0
          //  [4, 2, 6]] -> sorted: 2, 4, 6. idx = 1.5. val = 4 + 0.5 * (6-4) = 5.0
          final a = NDArray.fromList(
            [1.0, 5.0, 3.0, 4.0, 2.0, 6.0],
            [2, 3],
            DType.float32,
          );
          final p = percentile(a, 75.0, axis: 1);
          expect(p.shape, [2]);
          expect(p.dtype, DType.float64); // returns double
          expect(p.toList(), [closeTo(4.0, 1e-6), closeTo(5.0, 1e-6)]);
        });
      });

      test('Percentile integer types returns double', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
          // q = 50 -> idx = 3 * 0.5 = 1.5. low=1, high=2. val = 2 + 0.5 * (3 - 2) = 2.5
          final p = percentile(a, 50.0);
          expect(p.dtype, DType.float64);
          expect(p.data[0], 2.5);
        });
      });
    });

    group('Quantile Tests', () {
      test('Quantile basic', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [15.0, 20.0, 35.0, 40.0, 50.0],
            [5],
            DType.float64,
          );
          // q = 0.4 is same as 40th percentile
          final q = quantile(a, 0.4);
          expect(q.shape, <int>[]);
          expect(q.dtype, DType.float64);
          expect(q.data[0], closeTo(29.0, 1e-9));
        });
      });
    });

    group('Validation & Edge Cases', () {
      test('Empty array throws', () {
        NDArray.scope(() {
          final a = NDArray<double>.zeros([0], DType.float64);
          expect(() => median(a), throwsArgumentError);
          expect(() => percentile(a, 50.0), throwsArgumentError);
        });
      });

      test('Invalid q throws', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          expect(() => percentile(a, -1.0), throwsArgumentError);
          expect(() => percentile(a, 101.0), throwsArgumentError);
          expect(() => quantile(a, -0.1), throwsArgumentError);
          expect(() => quantile(a, 1.1), throwsArgumentError);
        });
      });

      test('Axis out of bounds throws', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          expect(() => median(a, axis: 1), throwsArgumentError);
          expect(() => median(a, axis: -2), throwsArgumentError);
        });
      });
    });
  });
}
