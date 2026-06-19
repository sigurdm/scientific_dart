import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';
import 'dart:math' as math;

void main() {
  group('Percentiles, Medians, and Quantiles Tests', () {
    group('Median Tests', () {
      test('Flat float64 median (odd size)', () {
        NDArray.scope(() {
          final a = NDArray.fromList([3.0, 1.0, 2.0], [3], DType.float64);
          final m = median(a);
          expect(m.shape, <int>[]);
          expect(m.dtype, DType.float64);
          expect(m.toList()[0], 2.0);
        });
      });

      test('Flat float64 median (even size)', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1.0, 3.0, 2.0, 4.0], [4], DType.float64);
          final m = median(a);
          expect(m.shape, <int>[]);
          expect(m.toList()[0], 2.5);
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
          expect(out.toList()[0], 2.0);
        });
      });

      test('Median integer types (int32)', () {
        NDArray.scope(() {
          final a = NDArray.fromList([3, 1, 2, 4], [4], DType.int32);
          final m = median(a);
          // 1, 2, 3, 4 -> (2+3)/2 = 2 (integer division in C kernel: (2+3)/2 = 2)
          expect(m.dtype, DType.int32);
          expect(m.toList()[0], 2);
        });
      });

      test('Median integer types (int64)', () {
        NDArray.scope(() {
          final a = NDArray.fromList([10, 30, 20], [3], DType.int64);
          final m = median(a);
          expect(m.dtype, DType.int64);
          expect(m.toList()[0], 20);
        });
      });

      test('Median integer types (uint8)', () {
        NDArray.scope(() {
          final a = NDArray.fromList([5, 1, 3, 4], [4], DType.uint8);
          final m = median(a);
          // 1, 3, 4, 5 -> (3+4)/2 = 3
          expect(m.dtype, DType.uint8);
          expect(m.toList()[0], 3);
        });
      });

      test('Median complex128 (independent real/imag)', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [Complex128(3.0, 1.0), Complex128(1.0, 9.0), Complex128(2.0, 5.0)],
            [3],
            DType.complex128,
          );
          // Reals: 3, 1, 2 -> median is 2
          // Imags: 1, 9, 5 -> median is 5
          final m = median(a);
          expect(m.dtype, DType.complex128);
          expect(m.scalar, Complex128(2.0, 5.0));
        });
      });

      test('Median complex64 (independent real/imag)', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex64(1.0, 4.0),
              Complex64(3.0, 1.0),
              Complex64(2.0, 2.0),
              Complex64(4.0, 3.0),
            ],
            [4],
            DType.complex64,
          );
          // Reals: 1, 2, 3, 4 -> median: (2+3)/2 = 2.5
          // Imags: 1, 2, 3, 4 -> median: (2+3)/2 = 2.5
          final m = median(a);
          expect(m.dtype, DType.complex64);
          expect(m.scalar, Complex64(2.5, 2.5));
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
          expect(p.toList()[0], closeTo(29.0, 1e-9));
        });
      });

      test('Percentile 0 and 100', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [15.0, 20.0, 35.0, 40.0, 50.0],
            [5],
            DType.float64,
          );
          expect(percentile(a, 0.0).toList()[0], 15.0);
          expect(percentile(a, 100.0).toList()[0], 50.0);
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
          expect(p.toList()[0], 2.5);
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
          expect(q.toList()[0], closeTo(29.0, 1e-9));
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

    group('Percentile Methods Tests (NumPy compatibility)', () {
      final a = NDArray.fromList([10.0, 20.0, 30.0, 40.0], [4], DType.float64);

      test('invertedCdf', () {
        NDArray.scope(() {
          expect(
            percentile(a, 35.0, method: QuantileMethod.invertedCdf).scalar,
            closeTo(20.0, 1e-9),
          );
          expect(
            percentile(a, 50.0, method: QuantileMethod.invertedCdf).scalar,
            closeTo(20.0, 1e-9),
          );
        });
      });

      test('averagedInvertedCdf', () {
        NDArray.scope(() {
          expect(
            percentile(
              a,
              35.0,
              method: QuantileMethod.averagedInvertedCdf,
            ).scalar,
            closeTo(20.0, 1e-9),
          );
          expect(
            percentile(
              a,
              50.0,
              method: QuantileMethod.averagedInvertedCdf,
            ).scalar,
            closeTo(25.0, 1e-9),
          );
        });
      });

      test('closestObservation', () {
        NDArray.scope(() {
          expect(
            percentile(
              a,
              35.0,
              method: QuantileMethod.closestObservation,
            ).scalar,
            closeTo(10.0, 1e-9),
          );
          expect(
            percentile(
              a,
              50.0,
              method: QuantileMethod.closestObservation,
            ).scalar,
            closeTo(20.0, 1e-9),
          );
        });
      });

      test('interpolatedInvertedCdf', () {
        NDArray.scope(() {
          expect(
            percentile(
              a,
              35.0,
              method: QuantileMethod.interpolatedInvertedCdf,
            ).scalar,
            closeTo(14.0, 1e-9),
          );
          expect(
            percentile(
              a,
              50.0,
              method: QuantileMethod.interpolatedInvertedCdf,
            ).scalar,
            closeTo(20.0, 1e-9),
          );
        });
      });

      test('hazen', () {
        NDArray.scope(() {
          expect(
            percentile(a, 35.0, method: QuantileMethod.hazen).scalar,
            closeTo(19.0, 1e-9),
          );
          expect(
            percentile(a, 50.0, method: QuantileMethod.hazen).scalar,
            closeTo(25.0, 1e-9),
          );
        });
      });

      test('weibull', () {
        NDArray.scope(() {
          expect(
            percentile(a, 35.0, method: QuantileMethod.weibull).scalar,
            closeTo(17.5, 1e-9),
          );
          expect(
            percentile(a, 50.0, method: QuantileMethod.weibull).scalar,
            closeTo(25.0, 1e-9),
          );
        });
      });

      test('linear', () {
        NDArray.scope(() {
          expect(
            percentile(a, 35.0, method: QuantileMethod.linear).scalar,
            closeTo(20.5, 1e-9),
          );
          expect(
            percentile(a, 50.0, method: QuantileMethod.linear).scalar,
            closeTo(25.0, 1e-9),
          );
        });
      });

      test('medianUnbiased', () {
        NDArray.scope(() {
          expect(
            percentile(a, 35.0, method: QuantileMethod.medianUnbiased).scalar,
            closeTo(18.5, 1e-9),
          );
          expect(
            percentile(a, 50.0, method: QuantileMethod.medianUnbiased).scalar,
            closeTo(25.0, 1e-9),
          );
        });
      });

      test('normalUnbiased', () {
        NDArray.scope(() {
          expect(
            percentile(a, 35.0, method: QuantileMethod.normalUnbiased).scalar,
            closeTo(18.625, 1e-9),
          );
          expect(
            percentile(a, 50.0, method: QuantileMethod.normalUnbiased).scalar,
            closeTo(25.0, 1e-9),
          );
        });
      });

      test('lower', () {
        NDArray.scope(() {
          expect(
            percentile(a, 35.0, method: QuantileMethod.lower).scalar,
            closeTo(20.0, 1e-9),
          );
          expect(
            percentile(a, 50.0, method: QuantileMethod.lower).scalar,
            closeTo(20.0, 1e-9),
          );
        });
      });

      test('higher', () {
        NDArray.scope(() {
          expect(
            percentile(a, 35.0, method: QuantileMethod.higher).scalar,
            closeTo(30.0, 1e-9),
          );
          expect(
            percentile(a, 50.0, method: QuantileMethod.higher).scalar,
            closeTo(30.0, 1e-9),
          );
        });
      });

      test('midpoint', () {
        NDArray.scope(() {
          expect(
            percentile(a, 35.0, method: QuantileMethod.midpoint).scalar,
            closeTo(25.0, 1e-9),
          );
          expect(
            percentile(a, 50.0, method: QuantileMethod.midpoint).scalar,
            closeTo(25.0, 1e-9),
          );
        });
      });

      test('nearest', () {
        NDArray.scope(() {
          expect(
            percentile(a, 35.0, method: QuantileMethod.nearest).scalar,
            closeTo(20.0, 1e-9),
          );
          expect(
            percentile(a, 50.0, method: QuantileMethod.nearest).scalar,
            closeTo(30.0, 1e-9),
          );
        });
      });
    });
  });

  group('Distance Tests', () {
    group('Pairwise Distance (pdist)', () {
      test('Euclidean metric - Float64', () {
        NDArray.scope(() {
          final x = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
            DType.float64,
          );

          final res = pdist(x, metric: DistanceMetric.euclidean);

          expect(res.shape, [3]); // 3 * 2 / 2 = 3
          expect(res.dtype, DType.float64);
          // dist(0, 1) = sqrt((1-3)^2 + (2-4)^2) = sqrt(8) approx 2.82842712
          // dist(0, 2) = sqrt((1-5)^2 + (2-6)^2) = sqrt(32) approx 5.65685425
          // dist(1, 2) = sqrt((3-5)^2 + (4-6)^2) = sqrt(8) approx 2.82842712
          expect(res.getCell([0]), closeTo(2.8284271247461903, 1e-9));
          expect(res.getCell([1]), closeTo(5.656854249492381, 1e-9));
          expect(res.getCell([2]), closeTo(2.8284271247461903, 1e-9));
        });
      });

      test('Cosine metric - Float32', () {
        NDArray.scope(() {
          final x = NDArray<Float32>.fromList(
            [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [3, 2],
            DType.float32,
          );

          final res = pdist(x, metric: DistanceMetric.cosine);

          expect(res.shape, [3]);
          // dist(0, 1) = 1 - 0 / (1 * 1) = 1.0
          // dist(0, 2) = 1 - 1 / (1 * sqrt(2)) = 1 - 0.70710678 = 0.29289322
          // dist(1, 2) = 1 - 1 / (1 * sqrt(2)) = 0.29289322
          expect(res.getCell([0]), closeTo(1.0, 1e-6));
          expect(res.getCell([1]), closeTo(0.2928932188134524, 1e-6));
          expect(res.getCell([2]), closeTo(0.2928932188134524, 1e-6));
        });
      });

      test('Hamming metric - Int32', () {
        NDArray.scope(() {
          final x = NDArray<Int32>.fromList(
            [1, 0, 1, 1, 1, 0, 0, 0, 1],
            [3, 3],
            DType.int32,
          );

          final res = pdist(x, metric: DistanceMetric.hamming);

          expect(res.shape, [3]);
          // dist(0, 1) = diff( [1,0,1], [1,1,0] ) / 3 = 2/3 approx 0.66666667
          // dist(0, 2) = diff( [1,0,1], [0,0,1] ) / 3 = 1/3 approx 0.33333333
          // dist(1, 2) = diff( [1,1,0], [0,0,1] ) / 3 = 3/3 = 1.0
          expect(res.getCell([0]), closeTo(2.0 / 3.0, 1e-9));
          expect(res.getCell([1]), closeTo(1.0 / 3.0, 1e-9));
          expect(res.getCell([2]), closeTo(1.0, 1e-9));
        });
      });

      test('Hamming metric - Bool', () {
        NDArray.scope(() {
          final x = NDArray<bool>.fromList(
            [true, false, true, true, true, false, false, false, true],
            [3, 3],
            DType.boolean,
          );

          final res = pdist(x, metric: DistanceMetric.hamming);

          expect(res.shape, [3]);
          expect(res.getCell([0]), closeTo(2.0 / 3.0, 1e-9));
          expect(res.getCell([1]), closeTo(1.0 / 3.0, 1e-9));
          expect(res.getCell([2]), closeTo(1.0, 1e-9));
        });
      });

      test('Strided views (non-contiguous)', () {
        NDArray.scope(() {
          // Create a larger array and slice it to get non-contiguous views
          final xFull = NDArray<Float64>.fromList(
            [
              1.0,
              99.0,
              2.0,
              99.0,
              99.0,
              99.0,
              3.0,
              99.0,
              4.0,
              99.0,
              99.0,
              99.0,
              5.0,
              99.0,
              6.0,
            ],
            [5, 3],
            DType.float64,
          );

          // Slice every 2nd row and columns 0 and 2
          final x = xFull.slice([
            const Slice(start: 0, stop: 5, step: 2), // rows 0, 2, 4
            const Slice(start: 0, stop: 3, step: 2), // cols 0, 2
          ]);

          expect(x.shape, [3, 2]);
          expect(x.isContiguous, false);
          // x should be:
          // [1.0, 2.0]
          // [3.0, 4.0]
          // [5.0, 6.0]

          final res = pdist(x, metric: DistanceMetric.euclidean);

          expect(res.shape, [3]);
          expect(res.getCell([0]), closeTo(2.8284271247461903, 1e-9));
          expect(res.getCell([1]), closeTo(5.656854249492381, 1e-9));
          expect(res.getCell([2]), closeTo(2.8284271247461903, 1e-9));
        });
      });

      test('Edge case - M < 2', () {
        NDArray.scope(() {
          final x1 = NDArray<Float64>.fromList(
            [1.0, 2.0],
            [1, 2],
            DType.float64,
          );
          final res1 = pdist(x1);
          expect(res1.shape, [0]);
          expect(res1.dtype, DType.float64);

          final x0 = NDArray<Float64>.create([0, 2], DType.float64);
          final res0 = pdist(x0);
          expect(res0.shape, [0]);
        });
      });

      test('Edge case - empty features (N = 0)', () {
        NDArray.scope(() {
          final x = NDArray<Float64>.create([3, 0], DType.float64);

          final resEuc = pdist(x, metric: DistanceMetric.euclidean);
          expect(resEuc.shape, [3]);
          expect(resEuc.getCell([0]), 0.0);
          expect(resEuc.getCell([1]), 0.0);
          expect(resEuc.getCell([2]), 0.0);

          final resCos = pdist(x, metric: DistanceMetric.cosine);
          expect(resCos.shape, [3]);
          expect(resCos.getCell([0]).isNaN, true);
          expect(resCos.getCell([1]).isNaN, true);
          expect(resCos.getCell([2]).isNaN, true);

          final resHam = pdist(x, metric: DistanceMetric.hamming);
          expect(resHam.shape, [3]);
          expect(resHam.getCell([0]).isNaN, true);
          expect(resHam.getCell([1]).isNaN, true);
          expect(resHam.getCell([2]).isNaN, true);
        });
      });

      test('Edge case - Cosine zero vectors (division by zero)', () {
        NDArray.scope(() {
          final x = NDArray<Float64>.fromList(
            [0.0, 0.0, 1.0, 2.0, 0.0, 0.0],
            [3, 2],
            DType.float64,
          );

          final res = pdist(x, metric: DistanceMetric.cosine);

          expect(res.shape, [3]);
          expect(res.getCell([0]).isNaN, true); // dist(zero, non-zero)
          expect(res.getCell([1]).isNaN, true); // dist(zero, zero)
          expect(res.getCell([2]).isNaN, true); // dist(non-zero, zero)
        });
      });

      test('Recycling out parameter', () {
        NDArray.scope(() {
          final x = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
            DType.float64,
          );

          final out = NDArray<Float64>.create([3], DType.float64);
          final res = pdist(x, metric: DistanceMetric.euclidean, out: out);

          expect(identical(res, out), true);
          expect(res.getCell([0]), closeTo(2.8284271247461903, 1e-9));
        });
      });

      test('Recycling out parameter - shape mismatch throws', () {
        NDArray.scope(() {
          final x = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [3, 2],
            DType.float64,
          );

          final out = NDArray<Float64>.create([
            4,
          ], DType.float64); // Wrong size, expected 3
          expect(() => pdist(x, out: out), throwsArgumentError);
        });
      });
    });

    group('Pairwise Distance between two sets (cdist)', () {
      test('Euclidean metric - Float64', () {
        NDArray.scope(() {
          final xa = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final xb = NDArray<Float64>.fromList(
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [3, 2],
            DType.float64,
          );

          final res = cdist(xa, xb, metric: DistanceMetric.euclidean);

          expect(res.shape, [2, 3]); // M x K = 2 x 3
          expect(res.dtype, DType.float64);

          // dist(xa[0], xb[0]) = sqrt((1-5)^2 + (2-6)^2) = sqrt(32) approx 5.65685425
          // dist(xa[0], xb[1]) = sqrt((1-7)^2 + (2-8)^2) = sqrt(72) approx 8.48528137
          // dist(xa[0], xb[2]) = sqrt((1-9)^2 + (2-10)^2) = sqrt(128) approx 11.3137085
          // dist(xa[1], xb[0]) = sqrt((3-5)^2 + (4-6)^2) = sqrt(8) approx 2.82842712
          // dist(xa[1], xb[1]) = sqrt((3-7)^2 + (4-8)^2) = sqrt(32) approx 5.65685425
          // dist(xa[1], xb[2]) = sqrt((3-9)^2 + (4-10)^2) = sqrt(72) approx 8.48528137

          expect(res.getCell([0, 0]), closeTo(5.656854249492381, 1e-9));
          expect(res.getCell([0, 1]), closeTo(8.48528137423857, 1e-9));
          expect(res.getCell([0, 2]), closeTo(11.31370849898476, 1e-9));
          expect(res.getCell([1, 0]), closeTo(2.8284271247461903, 1e-9));
          expect(res.getCell([1, 1]), closeTo(5.656854249492381, 1e-9));
          expect(res.getCell([1, 2]), closeTo(8.48528137423857, 1e-9));
        });
      });

      test('Mixed dtypes (promotion)', () {
        NDArray.scope(() {
          final xa = NDArray<Int32>.fromList([1, 2], [1, 2], DType.int32);
          final xb = NDArray<Float64>.fromList(
            [3.0, 4.0],
            [1, 2],
            DType.float64,
          );

          final res = cdist(xa, xb, metric: DistanceMetric.euclidean);
          expect(res.shape, [1, 1]);
          expect(res.getCell([0, 0]), closeTo(2.8284271247461903, 1e-9));
        });
      });

      test('Strided views (non-contiguous)', () {
        NDArray.scope(() {
          final xaFull = NDArray<Float64>.fromList(
            [1.0, 99.0, 2.0, 99.0, 99.0, 99.0, 3.0, 99.0, 4.0],
            [3, 3],
            DType.float64,
          );
          final xa = xaFull.slice([
            const Slice(start: 0, stop: 3, step: 2), // rows 0, 2
            const Slice(start: 0, stop: 3, step: 2), // cols 0, 2
          ]); // [1, 2], [3, 4]

          final xbFull = NDArray<Float64>.fromList(
            [
              5.0,
              99.0,
              6.0,
              99.0,
              99.0,
              99.0,
              7.0,
              99.0,
              8.0,
              99.0,
              99.0,
              99.0,
              9.0,
              99.0,
              10.0,
            ],
            [5, 3],
            DType.float64,
          );
          final xb = xbFull.slice([
            const Slice(start: 0, stop: 5, step: 2), // rows 0, 2, 4
            const Slice(start: 0, stop: 3, step: 2), // cols 0, 2
          ]); // [5, 6], [7, 8], [9, 10]

          final res = cdist(xa, xb, metric: DistanceMetric.euclidean);

          expect(res.shape, [2, 3]);
          expect(res.getCell([0, 0]), closeTo(5.656854249492381, 1e-9));
          expect(res.getCell([0, 1]), closeTo(8.48528137423857, 1e-9));
          expect(res.getCell([0, 2]), closeTo(11.31370849898476, 1e-9));
          expect(res.getCell([1, 0]), closeTo(2.8284271247461903, 1e-9));
          expect(res.getCell([1, 1]), closeTo(5.656854249492381, 1e-9));
          expect(res.getCell([1, 2]), closeTo(8.48528137423857, 1e-9));
        });
      });

      test('Edge case - empty inputs (M = 0 or K = 0)', () {
        NDArray.scope(() {
          final xa = NDArray<Float64>.create([0, 2], DType.float64);
          final xb = NDArray<Float64>.create([3, 2], DType.float64);

          final res1 = cdist(xa, xb);
          expect(res1.shape, [0, 3]);

          final res2 = cdist(xb, xa);
          expect(res2.shape, [3, 0]);
        });
      });

      test('Edge case - empty features (N = 0)', () {
        NDArray.scope(() {
          final xa = NDArray<Float64>.create([2, 0], DType.float64);
          final xb = NDArray<Float64>.create([3, 0], DType.float64);

          final res = cdist(xa, xb, metric: DistanceMetric.euclidean);
          expect(res.shape, [2, 3]);
          for (var i = 0; i < 2; i++) {
            for (var j = 0; j < 3; j++) {
              expect(res.getCell([i, j]), 0.0);
            }
          }

          final resCos = cdist(xa, xb, metric: DistanceMetric.cosine);
          expect(resCos.shape, [2, 3]);
          for (var i = 0; i < 2; i++) {
            for (var j = 0; j < 3; j++) {
              expect(resCos.getCell([i, j]).isNaN, true);
            }
          }
        });
      });

      test('Mismatched features dimension throws', () {
        NDArray.scope(() {
          final xa = NDArray<Float64>.create([2, 2], DType.float64);
          final xb = NDArray<Float64>.create([
            3,
            3,
          ], DType.float64); // N=3 instead of 2

          expect(() => cdist(xa, xb), throwsArgumentError);
        });
      });

      test('Recycling out parameter', () {
        NDArray.scope(() {
          final xa = NDArray<Float64>.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final xb = NDArray<Float64>.fromList(
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [3, 2],
            DType.float64,
          );

          final out = NDArray<Float64>.create([2, 3], DType.float64);
          final res = cdist(xa, xb, metric: DistanceMetric.euclidean, out: out);

          expect(identical(res, out), true);
          expect(res.getCell([0, 0]), closeTo(5.656854249492381, 1e-9));
        });
      });
    });
  });

  group('1D Linear Interpolation (interp)', () {
    test('Basic interpolation', () {
      final xp = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final fp = NDArray.fromList([10.0, 20.0, 40.0], [3], DType.float64);
      final x = NDArray.fromList([1.5, 2.5], [2], DType.float64);

      final res = interp(x, xp, fp);

      expect(res.shape, [2]);
      expect(res.dtype, DType.float64);
      expect(res.getCell([0]), closeTo(15.0, 1e-9));
      expect(res.getCell([1]), closeTo(30.0, 1e-9));

      xp.dispose();
      fp.dispose();
      x.dispose();
      res.dispose();
    });

    test('Boundary values (default left/right)', () {
      final xp = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final fp = NDArray.fromList([10.0, 20.0, 40.0], [3], DType.float64);
      final x = NDArray.fromList([0.0, 4.0], [2], DType.float64);

      final res = interp(x, xp, fp);

      expect(res.getCell([0]), closeTo(10.0, 1e-9)); // fp[0]
      expect(res.getCell([1]), closeTo(40.0, 1e-9)); // fp[last]

      xp.dispose();
      fp.dispose();
      x.dispose();
      res.dispose();
    });

    test('Boundary values (custom left/right)', () {
      final xp = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final fp = NDArray.fromList([10.0, 20.0, 40.0], [3], DType.float64);
      final x = NDArray.fromList([0.0, 4.0], [2], DType.float64);

      final res = interp(x, xp, fp, left: -1.0, right: -2.0);

      expect(res.getCell([0]), closeTo(-1.0, 1e-9));
      expect(res.getCell([1]), closeTo(-2.0, 1e-9));

      xp.dispose();
      fp.dispose();
      x.dispose();
      res.dispose();
    });

    test('Exact data points', () {
      final xp = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final fp = NDArray.fromList([10.0, 20.0, 40.0], [3], DType.float64);
      final x = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);

      final res = interp(x, xp, fp);

      expect(res.getCell([0]), closeTo(10.0, 1e-9));
      expect(res.getCell([1]), closeTo(20.0, 1e-9));
      expect(res.getCell([2]), closeTo(40.0, 1e-9));

      xp.dispose();
      fp.dispose();
      x.dispose();
      res.dispose();
    });

    test('Multi-dimensional x', () {
      final xp = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final fp = NDArray.fromList([10.0, 20.0, 40.0], [3], DType.float64);
      final x = NDArray.fromList([1.5, 2.5, 0.0, 4.0], [2, 2], DType.float64);

      final res = interp(x, xp, fp, left: -1.0, right: -2.0);

      expect(res.shape, [2, 2]);
      expect(res.getCell([0, 0]), closeTo(15.0, 1e-9));
      expect(res.getCell([0, 1]), closeTo(30.0, 1e-9));
      expect(res.getCell([1, 0]), closeTo(-1.0, 1e-9));
      expect(res.getCell([1, 1]), closeTo(-2.0, 1e-9));

      xp.dispose();
      fp.dispose();
      x.dispose();
      res.dispose();
    });

    test('Integer input promotion', () {
      final xp = NDArray.fromList([1, 2, 3], [3], DType.int32);
      final fp = NDArray.fromList([10, 20, 40], [3], DType.int32);
      final x = NDArray.fromList([1.5, 2.5], [2], DType.float64);

      final res = interp(x, xp, fp);

      expect(res.dtype, DType.float64);
      expect(res.getCell([0]), closeTo(15.0, 1e-9));
      expect(res.getCell([1]), closeTo(30.0, 1e-9));

      xp.dispose();
      fp.dispose();
      x.dispose();
      res.dispose();
    });

    test('Single point xp/fp', () {
      final xp = NDArray.fromList([2.0], [1], DType.float64);
      final fp = NDArray.fromList([20.0], [1], DType.float64);
      final x = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);

      final res = interp(x, xp, fp);

      expect(res.getCell([0]), closeTo(20.0, 1e-9)); // left fallback to fp[0]
      expect(res.getCell([1]), closeTo(20.0, 1e-9)); // exact
      expect(res.getCell([2]), closeTo(20.0, 1e-9)); // right fallback to fp[0]

      final resCustom = interp(x, xp, fp, left: -1.0, right: -2.0);
      expect(resCustom.getCell([0]), closeTo(-1.0, 1e-9));
      expect(resCustom.getCell([1]), closeTo(20.0, 1e-9));
      expect(resCustom.getCell([2]), closeTo(-2.0, 1e-9));

      xp.dispose();
      fp.dispose();
      x.dispose();
      res.dispose();
      resCustom.dispose();
    });

    test('Unsorted xp throws ArgumentError', () {
      final xp = NDArray.fromList([2.0, 1.0, 3.0], [3], DType.float64);
      final fp = NDArray.fromList([20.0, 10.0, 40.0], [3], DType.float64);
      final x = NDArray.fromList([1.5], [1], DType.float64);

      expect(() => interp(x, xp, fp), throwsArgumentError);

      xp.dispose();
      fp.dispose();
      x.dispose();
    });

    test('Duplicate xp throws ArgumentError (strictly increasing)', () {
      final xp = NDArray.fromList([1.0, 2.0, 2.0, 3.0], [4], DType.float64);
      final fp = NDArray.fromList([10.0, 20.0, 30.0, 40.0], [4], DType.float64);
      final x = NDArray.fromList([1.5], [1], DType.float64);

      expect(() => interp(x, xp, fp), throwsArgumentError);

      xp.dispose();
      fp.dispose();
      x.dispose();
    });

    test('Mismatched xp and fp lengths throws ArgumentError', () {
      final xp = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final fp = NDArray.fromList([10.0, 20.0], [2], DType.float64);
      final x = NDArray.fromList([1.5], [1], DType.float64);

      expect(() => interp(x, xp, fp), throwsArgumentError);

      xp.dispose();
      fp.dispose();
      x.dispose();
    });

    test('Non-1D xp or fp throws ArgumentError', () {
      final xp = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
      final fp = NDArray.fromList([10.0, 20.0, 30.0, 40.0], [4], DType.float64);
      final x = NDArray.fromList([1.5], [1], DType.float64);

      expect(() => interp(x, xp, fp), throwsArgumentError);

      final xp2 = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
      final fp2 = NDArray.fromList(
        [10.0, 20.0, 30.0, 40.0],
        [2, 2],
        DType.float64,
      );

      expect(() => interp(x, xp2, fp2), throwsArgumentError);

      xp.dispose();
      fp.dispose();
      x.dispose();
      xp2.dispose();
      fp2.dispose();
    });

    test('Empty xp throws ArgumentError', () {
      final xp = NDArray.fromList(<double>[], [0], DType.float64);
      final fp = NDArray.fromList(<double>[], [0], DType.float64);
      final x = NDArray.fromList([1.5], [1], DType.float64);

      expect(() => interp(x, xp, fp), throwsArgumentError);

      xp.dispose();
      fp.dispose();
      x.dispose();
    });

    test('Strided inputs (views)', () {
      final xpFull = NDArray.fromList(
        [1.0, 99.0, 2.0, 99.0, 3.0],
        [5],
        DType.float64,
      );
      final xp = xpFull.slice([
        Slice(start: 0, stop: 5, step: 2),
      ]); // [1.0, 2.0, 3.0]
      expect(xp.isContiguous, false);
      expect(xp.shape, [3]);

      final fpFull = NDArray.fromList(
        [10.0, 99.0, 20.0, 99.0, 40.0],
        [5],
        DType.float64,
      );
      final fp = fpFull.slice([
        Slice(start: 0, stop: 5, step: 2),
      ]); // [10.0, 20.0, 40.0]
      expect(fp.isContiguous, false);
      expect(fp.shape, [3]);

      final xFull = NDArray.fromList(
        [1.5, 99.0, 2.5, 99.0],
        [4],
        DType.float64,
      );
      final x = xFull.slice([Slice(start: 0, stop: 4, step: 2)]); // [1.5, 2.5]
      expect(x.isContiguous, false);
      expect(x.shape, [2]);

      final res = interp(x, xp, fp);

      expect(res.shape, [2]);
      expect(res.getCell([0]), closeTo(15.0, 1e-9));
      expect(res.getCell([1]), closeTo(30.0, 1e-9));

      xpFull.dispose();
      fpFull.dispose();
      xFull.dispose();
      res.dispose();
    });

    test('0D x (scalar) contiguous', () {
      final xp = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final fp = NDArray.fromList([10.0, 20.0, 40.0], [3], DType.float64);
      final x = NDArray.fromList([1.5], [], DType.float64);

      final res = interp(x, xp, fp);

      expect(res.shape, []);
      expect(res.getCell([]), closeTo(15.0, 1e-9));

      xp.dispose();
      fp.dispose();
      x.dispose();
      res.dispose();
    });

    test('0D x (scalar) strided xp/fp', () {
      final xpFull = NDArray.fromList(
        [1.0, 99.0, 2.0, 99.0, 3.0],
        [5],
        DType.float64,
      );
      final xp = xpFull.slice([
        Slice(start: 0, stop: 5, step: 2),
      ]); // [1.0, 2.0, 3.0]
      final fpFull = NDArray.fromList(
        [10.0, 99.0, 20.0, 99.0, 40.0],
        [5],
        DType.float64,
      );
      final fp = fpFull.slice([
        Slice(start: 0, stop: 5, step: 2),
      ]); // [10.0, 20.0, 40.0]

      final x = NDArray.fromList([1.5], [], DType.float64);

      final res = interp(x, xp, fp);

      expect(res.shape, []);
      expect(res.getCell([]), closeTo(15.0, 1e-9));

      xpFull.dispose();
      fpFull.dispose();
      x.dispose();
      res.dispose();
    });
  });

  group('Quantitative Financial ufuncs Tests', () {
    group('Future Value (fv) Tests', () {
      test('Scalar-like 0D arrays', () {
        NDArray.scope(() {
          final rate = NDArray.fromList([0.05 / 12], [], DType.float64);
          final nper = NDArray.fromList([10.0 * 12], [], DType.float64);
          final pmt = NDArray.fromList([-100.0], [], DType.float64);
          final pvVal = NDArray.fromList([-100.0], [], DType.float64);

          final res = fv(rate, nper, pmt, pvVal);
          expect(res.shape, <int>[]);
          expect(res.dtype, DType.float64);
          expect(res.scalar, closeTo(15692.92889433575, 1e-5));
        });
      });

      test('Rate = 0 case', () {
        NDArray.scope(() {
          final rate = NDArray.fromList([0.0], [], DType.float64);
          final nper = NDArray.fromList([120.0], [], DType.float64);
          final pmt = NDArray.fromList([-100.0], [], DType.float64);
          final pvVal = NDArray.fromList([-100.0], [], DType.float64);

          final res = fv(rate, nper, pmt, pvVal);
          // fv = - (pv + pmt * nper) = - (-100 + -100 * 120) = - (-12100) = 12100
          expect(res.scalar, closeTo(12100.0, 1e-5));
        });
      });

      test('Broadcasting rate array', () {
        NDArray.scope(() {
          final rate = NDArray.fromList(
            [0.05 / 12, 0.06 / 12, 0.07 / 12],
            [3],
            DType.float64,
          );
          final nper = NDArray.fromList([10.0 * 12], [], DType.float64);
          final pmt = NDArray.fromList([-100.0], [], DType.float64);
          final pvVal = NDArray.fromList([-100.0], [], DType.float64);

          final res = fv(rate, nper, pmt, pvVal);
          expect(res.shape, [3]);
          expect(res.getCell([0]), closeTo(15692.92889434, 1e-5));
          expect(res.getCell([1]), closeTo(16569.87435405, 1e-5));
          expect(res.getCell([2]), closeTo(17509.44688102, 1e-5));
        });
      });

      test('when = "begin" (1) vs "end" (0)', () {
        NDArray.scope(() {
          final rate = NDArray.fromList([0.05 / 12], [], DType.float64);
          final nper = NDArray.fromList([10.0 * 12], [], DType.float64);
          final pmt = NDArray.fromList([-100.0], [], DType.float64);
          final pvVal = NDArray.fromList([-100.0], [], DType.float64);

          final resEnd = fv(rate, nper, pmt, pvVal, when: 'end');
          final resBegin = fv(rate, nper, pmt, pvVal, when: 'begin');

          expect(resEnd.scalar, closeTo(15692.92889433575, 1e-5));
          // with when=1, payments are at beginning, so more interest
          expect(resBegin.scalar, greaterThan(resEnd.scalar));
          expect(resBegin.scalar, closeTo(15757.629844104778, 1e-5));
        });
      });

      test('out parameter recycling', () {
        NDArray.scope(() {
          final rate = NDArray.fromList([0.05 / 12], [], DType.float64);
          final nper = NDArray.fromList([10.0 * 12], [], DType.float64);
          final pmt = NDArray.fromList([-100.0], [], DType.float64);
          final pvVal = NDArray.fromList([-100.0], [], DType.float64);
          final out = NDArray<Float64>.zeros([], DType.float64);

          final res = fv(rate, nper, pmt, pvVal, out: out);
          expect(identical(res, out), true);
          expect(out.scalar, closeTo(15692.92889433575, 1e-5));
        });
      });
    });

    group('Present Value (pv) Tests', () {
      test('Scalar-like 0D arrays', () {
        NDArray.scope(() {
          final rate = NDArray.fromList([0.05 / 12], [], DType.float64);
          final nper = NDArray.fromList([10.0 * 12], [], DType.float64);
          final pmt = NDArray.fromList([-100.0], [], DType.float64);
          final fvVal = NDArray.fromList(
            [15692.92889433575],
            [],
            DType.float64,
          );

          final res = pv(rate, nper, pmt, fvVal);
          expect(res.shape, <int>[]);
          expect(res.scalar, closeTo(-100.0, 1e-5));
        });
      });

      test('Rate = 0 case', () {
        NDArray.scope(() {
          final rate = NDArray.fromList([0.0], [], DType.float64);
          final nper = NDArray.fromList([120.0], [], DType.float64);
          final pmt = NDArray.fromList([-100.0], [], DType.float64);
          final fvVal = NDArray.fromList([12100.0], [], DType.float64);

          final res = pv(rate, nper, pmt, fvVal);
          // pv = - (fv + pmt * nper) = - (12100 + -100 * 120) = - (12100 - 12000) = -100
          expect(res.scalar, closeTo(-100.0, 1e-5));
        });
      });

      test('Broadcasting rate array', () {
        NDArray.scope(() {
          final rate = NDArray.fromList(
            [0.05 / 12, 0.04 / 12, 0.03 / 12],
            [3],
            DType.float64,
          );
          final nper = NDArray.fromList([10.0 * 12], [], DType.float64);
          final pmt = NDArray.fromList([-100.0], [], DType.float64);
          final fvVal = NDArray.fromList([15692.93], [], DType.float64);

          final res = pv(rate, nper, pmt, fvVal);
          expect(res.shape, [3]);
          expect(res.getCell([0]), closeTo(-100.00067132, 1e-5));
          expect(res.getCell([1]), closeTo(-649.26771385, 1e-5));
          expect(res.getCell([2]), closeTo(-1273.78633713, 1e-5));
        });
      });
    });

    group('Net Present Value (npv) Tests', () {
      test('1D cash flows, scalar rate', () {
        NDArray.scope(() {
          final rate = NDArray.fromList([0.08], [], DType.float64);
          final cashflows = NDArray.fromList(
            [-40000.0, 5000.0, 8000.0, 12000.0, 30000.0],
            [5],
            DType.float64,
          );

          final res = npv(rate, cashflows);
          expect(res.shape, <int>[]);
          expect(res.scalar, closeTo(3065.22267, 1e-5));
        });
      });

      test('Broadcasting multiple rates and multiple cash flows', () {
        NDArray.scope(() {
          final rates = NDArray.fromList(
            [0.00, 0.05, 0.10],
            [3],
            DType.float64,
          );
          final cashflows = NDArray.fromList(
            [-4000.0, 500.0, 800.0, -5000.0, 600.0, 900.0],
            [2, 3],
            DType.float64,
          );

          final res = npv(rates, cashflows);
          expect(res.shape, [3, 2]);

          expect(res.getCell([0, 0]), closeTo(-2700.0, 1e-2));
          expect(res.getCell([0, 1]), closeTo(-3500.0, 1e-2));
          expect(res.getCell([1, 0]), closeTo(-2798.19, 1e-2));
          expect(res.getCell([1, 1]), closeTo(-3612.24, 1e-2));
          expect(res.getCell([2, 0]), closeTo(-2884.30, 1e-2));
          expect(res.getCell([2, 1]), closeTo(-3710.74, 1e-2));
        });
      });
    });

    group('Internal Rate of Return (irr) Tests', () {
      test('Simple cash flows 1D', () {
        NDArray.scope(() {
          final cashflows = NDArray.fromList(
            [-100.0, 39.0, 59.0, 55.0, 20.0],
            [5],
            DType.float64,
          );
          final res = irr(cashflows);
          expect(res.shape, <int>[]);
          expect(res.scalar, closeTo(0.28095, 1e-5));
        });
      });

      test('Cash flows with negative rate result', () {
        NDArray.scope(() {
          final cashflows = NDArray.fromList(
            [-100.0, 0.0, 0.0, 74.0],
            [4],
            DType.float64,
          );
          final res = irr(cashflows);
          expect(res.scalar, closeTo(-0.0955, 1e-4));
        });
      });

      test('Same sign cash flows returns NaN', () {
        NDArray.scope(() {
          final cashflows = NDArray.fromList(
            [-100.0, -50.0, -20.0],
            [3],
            DType.float64,
          );
          final res = irr(cashflows);
          expect(res.scalar.isNaN, true);
        });
      });

      test('Same sign cash flows throws exception when requested', () {
        NDArray.scope(() {
          final cashflows = NDArray.fromList(
            [-100.0, -50.0, -20.0],
            [3],
            DType.float64,
          );
          expect(
            () => irr(cashflows, raiseExceptions: true),
            throwsA(isA<NoRealSolutionException>()),
          );
        });
      });
    });
  });

  group('Set Operations Tests', () {
    group('unique', () {
      test('double with duplicates and NaNs', () {
        final a = NDArray<double>.fromList(
          [3.0, 1.0, 2.0, 1.0, double.nan, 2.0, double.nan],
          [7],
          DType.float64,
        );
        final res = unique(a);
        expect(res.shape, [4]);
        // NaNs should be sorted to the end
        expectListEqualsWithNaNs(res.toList(), [1.0, 2.0, 3.0, double.nan]);
        a.dispose();
        res.dispose();
      });

      test('int32 flat', () {
        final a = NDArray<int>.fromList([1, 2, 2, 3, 1, 4], [6], DType.int32);
        final res = unique(a);
        expect(res.shape, [4]);
        expect(res.toList(), [1, 2, 3, 4]);
        a.dispose();
        res.dispose();
      });

      test('int32 2D (flattened)', () {
        final a = NDArray<int>.fromList([1, 2, 2, 3], [2, 2], DType.int32);
        final res = unique(a);
        expect(res.shape, [3]);
        expect(res.toList(), [1, 2, 3]);
        a.dispose();
        res.dispose();
      });

      test('uint8 empty', () {
        final a = NDArray<int>.fromList([], [0], DType.uint8);
        final res = unique(a);
        expect(res.shape, [0]);
        expect(res.toList(), <int>[]);
        a.dispose();
        res.dispose();
      });

      test('uint8 non-empty unique', () {
        final a = NDArray<int>.fromList([3, 1, 2, 1, 3], [5], DType.uint8);
        final res = unique(a);
        expect(res.dtype, DType.uint8);
        expect(res.toList(), [1, 2, 3]);
        a.dispose();
        res.dispose();
      });

      test('complex128', () {
        final a = NDArray<Complex>.fromList(
          [Complex(1, 2), Complex(3, 4), Complex(1, 2), Complex(2, 3)],
          [4],
          DType.complex128,
        );
        final res = unique(a);
        expect(res.shape, [3]);
        // Sorted lexicographically: (1,2), (2,3), (3,4)
        expect(res.toList(), [Complex(1, 2), Complex(2, 3), Complex(3, 4)]);
        a.dispose();
        res.dispose();
      });

      test('complex128 with duplicates and NaNs', () {
        final a = NDArray<Complex>.fromList(
          [
            Complex(3.0, 4.0),
            Complex(1.0, 2.0),
            Complex(double.nan, 2.0),
            Complex(1.0, double.nan),
            Complex(double.nan, double.nan),
            Complex(1.0, 2.0),
            Complex(double.nan, 2.0),
          ],
          [7],
          DType.complex128,
        );
        final res = unique(a);
        expect(res.shape, [5]);
        expectListEqualsWithNaNs(res.toList(), [
          Complex(1.0, 2.0),
          Complex(1.0, double.nan),
          Complex(3.0, 4.0),
          Complex(double.nan, 2.0),
          Complex(double.nan, double.nan),
        ]);
        a.dispose();
        res.dispose();
      });

      test('int32 non-contiguous 1D', () {
        final a = NDArray<int>.fromList([1, 2, 3, 4, 5, 6], [6], DType.int32);
        // Slice with step 2: [1, 3, 5]
        final slice = a.slice([Slice(step: 2)]);
        expect(slice.isContiguous, false);
        expect(slice.toList(), [1, 3, 5]);

        final res = unique(slice);
        expect(res.shape, [3]);
        expect(res.toList(), [1, 3, 5]);

        a.dispose();
        res.dispose();
      });

      test('optional returns int32', () {
        final a = NDArray<int>.fromList([1, 2, 2, 3, 1, 4], [6], DType.int32);

        final (u, index: idx, inverse: inv, counts: cnt) = unique(
          a,
          returnIndex: true,
          returnInverse: true,
          returnCounts: true,
        );

        expect(u.toList(), [1, 2, 3, 4]);
        expect(idx!.toList(), [0, 1, 3, 5]);
        expect(inv!.toList(), [0, 1, 1, 2, 0, 3]);
        expect(cnt!.toList(), [2, 2, 1, 1]);

        a.dispose();
        u.dispose();
        idx.dispose();
        inv.dispose();
        cnt.dispose();
      });

      test('optional returns double with NaNs', () {
        final a = NDArray<double>.fromList(
          [3.0, 1.0, 2.0, 1.0, double.nan, 2.0, double.nan],
          [7],
          DType.float64,
        );

        final (u, index: idx, inverse: inv, counts: cnt) = unique(
          a,
          returnIndex: true,
          returnInverse: true,
          returnCounts: true,
        );

        expectListEqualsWithNaNs(u.toList(), [1.0, 2.0, 3.0, double.nan]);
        expect(idx!.toList(), [1, 2, 0, 4]);
        expect(inv!.toList(), [2, 0, 1, 0, 3, 1, 3]);
        expect(cnt!.toList(), [2, 2, 1, 2]);

        a.dispose();
        u.dispose();
        idx.dispose();
        inv.dispose();
        cnt.dispose();
      });
    });

    group('intersect1d', () {
      test('int32 basic', () {
        final a = NDArray<int>.fromList([1, 3, 4, 3], [4], DType.int32);
        final b = NDArray<int>.fromList([3, 1, 2, 1], [4], DType.int32);
        final res = intersect1d(a, b);
        expect(res.shape, [2]);
        expect(res.toList(), [1, 3]);
        a.dispose();
        b.dispose();
        res.dispose();
      });

      test('int32 assumeUnique', () {
        final a = NDArray<int>.fromList([1, 3, 4], [3], DType.int32);
        final b = NDArray<int>.fromList([1, 2, 3], [3], DType.int32);
        final res = intersect1d(a, b, assumeUnique: true);
        expect(res.shape, [2]);
        expect(res.toList(), [1, 3]);
        a.dispose();
        b.dispose();
        res.dispose();
      });

      test('int32 assumeUnique with unsorted inputs', () {
        final a = NDArray<int>.fromList([3, 1, 4], [3], DType.int32);
        final b = NDArray<int>.fromList([2, 1, 3], [3], DType.int32);
        final res = intersect1d(a, b, assumeUnique: true);
        expect(res.shape, [2]);
        expect(res.toList(), [1, 3]);
        a.dispose();
        b.dispose();
        res.dispose();
      });

      test('uint8 basic intersect1d', () {
        final a = NDArray<int>.fromList([1, 2, 3, 2], [4], DType.uint8);
        final b = NDArray<int>.fromList([2, 3, 4, 3], [4], DType.uint8);
        final res = intersect1d(a, b);
        expect(res.dtype, DType.uint8);
        expect(res.toList(), [2, 3]);
        a.dispose();
        b.dispose();
        res.dispose();
      });

      test('double with NaNs', () {
        final a = NDArray<double>.fromList(
          [double.nan, 1.0, 2.0],
          [3],
          DType.float64,
        );
        final b = NDArray<double>.fromList(
          [2.0, double.nan, 3.0],
          [3],
          DType.float64,
        );
        final res = intersect1d(a, b);
        // NaNs should match
        expect(res.shape, [2]);
        expectListEqualsWithNaNs(res.toList(), [2.0, double.nan]);
        a.dispose();
        b.dispose();
        res.dispose();
      });

      test('int32 non-contiguous 1D', () {
        final a = NDArray<int>.fromList([1, 2, 3, 4, 5, 6], [6], DType.int32);
        final b = NDArray<int>.fromList([3, 0, 5, 0], [4], DType.int32);
        final sliceA = a.slice([Slice(step: 2)]); // [1, 3, 5]
        final sliceB = b.slice([Slice(step: 2)]); // [3, 5]

        final res = intersect1d(sliceA, sliceB);
        expect(res.toList(), [3, 5]);

        a.dispose();
        b.dispose();
        res.dispose();
      });
    });

    group('setdiff1d', () {
      test('int32 basic', () {
        final a = NDArray<int>.fromList([1, 2, 3, 2, 4], [5], DType.int32);
        final b = NDArray<int>.fromList([2, 3, 5], [3], DType.int32);
        final res = setdiff1d(a, b);
        expect(res.shape, [2]);
        expect(res.toList(), [1, 4]);
        a.dispose();
        b.dispose();
        res.dispose();
      });

      test('int32 non-contiguous 1D', () {
        final a = NDArray<int>.fromList([1, 2, 3, 4, 5, 6], [6], DType.int32);
        final b = NDArray<int>.fromList([3, 0, 5, 0], [4], DType.int32);
        final sliceA = a.slice([Slice(step: 2)]); // [1, 3, 5]
        final sliceB = b.slice([Slice(step: 2)]); // [3, 5]

        final res = setdiff1d(sliceA, sliceB);
        expect(res.toList(), [1]);

        a.dispose();
        b.dispose();
        res.dispose();
      });

      test('uint8 setdiff1d', () {
        final a = NDArray<int>.fromList([1, 2, 3, 2], [4], DType.uint8);
        final b = NDArray<int>.fromList([2, 4], [2], DType.uint8);
        final res = setdiff1d(a, b);
        expect(res.dtype, DType.uint8);
        expect(res.toList(), [1, 3]);
        a.dispose();
        b.dispose();
        res.dispose();
      });
    });

    group('setxor1d', () {
      test('int32 basic', () {
        final a = NDArray<int>.fromList([1, 2, 3], [3], DType.int32);
        final b = NDArray<int>.fromList([2, 3, 4], [3], DType.int32);
        final res = setxor1d(a, b);
        expect(res.shape, [2]);
        expect(res.toList(), [1, 4]);
        a.dispose();
        b.dispose();
        res.dispose();
      });

      test('uint8 setxor1d', () {
        final a = NDArray<int>.fromList([1, 2, 3], [3], DType.uint8);
        final b = NDArray<int>.fromList([2, 3, 4], [3], DType.uint8);
        final res = setxor1d(a, b);
        expect(res.dtype, DType.uint8);
        expect(res.toList(), [1, 4]);
        a.dispose();
        b.dispose();
        res.dispose();
      });
    });

    group('union1d', () {
      test('int32 basic', () {
        final a = NDArray<int>.fromList([1, 2, 3], [3], DType.int32);
        final b = NDArray<int>.fromList([2, 3, 4, 5], [4], DType.int32);
        final res = union1d(a, b);
        expect(res.shape, [5]);
        expect(res.toList(), [1, 2, 3, 4, 5]);
        a.dispose();
        b.dispose();
        res.dispose();
      });

      test('uint8 union1d', () {
        final a = NDArray<int>.fromList([1, 2, 3], [3], DType.uint8);
        final b = NDArray<int>.fromList([2, 3, 4, 5], [4], DType.uint8);
        final res = union1d(a, b);
        expect(res.dtype, DType.uint8);
        expect(res.toList(), [1, 2, 3, 4, 5]);
        a.dispose();
        b.dispose();
        res.dispose();
      });
    });

    group('isin', () {
      test('int32 basic', () {
        final element = NDArray<int>.fromList(
          [1, 2, 3, 4, 2, 1],
          [6],
          DType.int32,
        );
        final testElements = NDArray<int>.fromList([2, 4], [2], DType.int32);
        final res = isin(element, testElements);
        expect(res.shape, [6]);
        expect(res.dtype, DType.boolean);
        expect(res.toList(), [false, true, false, true, true, false]);
        element.dispose();
        testElements.dispose();
        res.dispose();
      });

      test('int32 2D element shape preserved', () {
        final element = NDArray<int>.fromList(
          [1, 2, 3, 4],
          [2, 2],
          DType.int32,
        );
        final testElements = NDArray<int>.fromList([2, 3], [2], DType.int32);
        final res = isin(element, testElements);
        expect(res.shape, [2, 2]);
        expect(res.dtype, DType.boolean);
        expect(res.toList(), [false, true, true, false]);
        element.dispose();
        testElements.dispose();
        res.dispose();
      });

      test('invert', () {
        final element = NDArray<int>.fromList([1, 2, 3], [3], DType.int32);
        final testElements = NDArray<int>.fromList([2], [1], DType.int32);
        final res = isin(element, testElements, invert: true);
        expect(res.toList(), [true, false, true]);
        element.dispose();
        testElements.dispose();
        res.dispose();
      });

      test('uint8 isin', () {
        final element = NDArray<int>.fromList([1, 2, 3, 2], [4], DType.uint8);
        final testElements = NDArray<int>.fromList([2, 4], [2], DType.uint8);
        final res = isin(element, testElements);
        expect(res.toList(), [false, true, false, true]);
        element.dispose();
        testElements.dispose();
        res.dispose();
      });

      test('isin assumeUnique with unsorted testElements', () {
        final element = NDArray<int>.fromList([1, 2, 3], [3], DType.int32);
        final testElements = NDArray<int>.fromList([3, 1], [2], DType.int32);
        final res = isin(element, testElements, assumeUnique: true);
        expect(res.toList(), [true, false, true]);
        element.dispose();
        testElements.dispose();
        res.dispose();
      });

      test('non-contiguous elements and testElements', () {
        final element = NDArray<int>.fromList(
          [1, 2, 3, 4, 5, 6],
          [6],
          DType.int32,
        );
        final testElements = NDArray<int>.fromList(
          [3, 0, 5, 0],
          [4],
          DType.int32,
        );

        final sliceElement = element.slice([Slice(step: 2)]); // [1, 3, 5]
        final sliceTest = testElements.slice([Slice(step: 2)]); // [3, 5]

        final res = isin(sliceElement, sliceTest);
        expect(res.toList(), [false, true, true]);

        element.dispose();
        testElements.dispose();
        res.dispose();
      });
    });
  });

  group('Shaping & Meshes Tests', () {
    group('asStrided Tests', () {
      test('basic 1D to 2D asStrided', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
          final view = asStrided(a, shape: [2, 2], strides: [2, 1]);

          expect(view.shape, [2, 2]);
          expect(view.toList(), [1, 2, 3, 4]);

          // Mutating view must affect original
          view.setCell([0, 1], Int32(99));
          expect(a.getCell([1]).value, 99);
        });
      });

      test('asStrided keeps strides if null', () {
        NDArray.scope(() {
          final a = NDArray.fromList([10, 20, 30], [3], DType.int32);
          final view = asStrided(a);
          expect(view.shape, [3]);
          expect(view.strides, [1]);
          expect(view.toList(), [10, 20, 30]);
        });
      });

      test('asStrided invalid shape/strides throws', () {
        NDArray.scope(() {
          final a = NDArray.create([4], DType.int32);
          expect(
            () => asStrided(a, shape: [2, 2], strides: [1]),
            throwsArgumentError,
          );
        });
      });
    });

    group('GridRange Tests', () {
      test('GridRange standard step', () {
        final r = GridRange(0, 5, step: 2);
        expect(r.start, 0.0);
        expect(r.stop, 5.0);
        expect(r.step, 2.0);
        expect(r.numPoints, isNull);
      });

      test('GridRange numPoints', () {
        final r = GridRange(0, 5, numPoints: 5);
        expect(r.start, 0.0);
        expect(r.stop, 5.0);
        expect(r.numPoints, 5);
      });

      test('GridRange.numpy standard step', () {
        final r = GridRange.numpy(0, 5, 2);
        expect(r.start, 0.0);
        expect(r.stop, 5.0);
        expect(r.step, 2.0);
        expect(r.numPoints, isNull);
      });

      test('GridRange.numpy complex step', () {
        final r = GridRange.numpy(0, 5, Complex(0, 5));
        expect(r.start, 0.0);
        expect(r.stop, 5.0);
        expect(r.numPoints, 5);
      });
    });

    group('ogrid Tests', () {
      test('basic 2D ogrid', () {
        NDArray.scope(() {
          final grid = ogrid([
            GridRange(0, 3), // 0, 1, 2
            GridRange(0, 2), // 0, 1
          ]);

          expect(grid.length, 2);
          expect(grid[0].shape, [3, 1]);
          expect(grid[0].toList(), [0.0, 1.0, 2.0]);

          expect(grid[1].shape, [1, 2]);
          expect(grid[1].toList(), [0.0, 1.0]);
        });
      });

      test('3D ogrid with numPoints and step', () {
        NDArray.scope(() {
          final grid = ogrid([
            GridRange(0, 2, numPoints: 3), // 0, 1, 2 (inclusive)
            GridRange(0, 2, step: 1), // 0, 1 (exclusive)
            GridRange.numpy(0, 1, Complex(0, 2)), // 0, 1 (inclusive, 2 points)
          ]);

          expect(grid.length, 3);
          expect(grid[0].shape, [3, 1, 1]);
          expect(grid[0].toList(), [0.0, 1.0, 2.0]);

          expect(grid[1].shape, [1, 2, 1]);
          expect(grid[1].toList(), [0.0, 1.0]);

          expect(grid[2].shape, [1, 1, 2]);
          expect(grid[2].toList(), [0.0, 1.0]);
        });
      });

      test('empty ogrid throws', () {
        expect(() => ogrid([]), throwsArgumentError);
      });
    });

    group('mgrid Tests', () {
      test('basic 2D mgrid', () {
        NDArray.scope(() {
          final grid = mgrid([GridRange(0, 3), GridRange(0, 2)]);

          expect(grid.shape, [2, 3, 2]);
          expect(grid.toList(), [
            0.0, 0.0, 1.0, 1.0, 2.0, 2.0, // grid 0
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, // grid 1
          ]);
        });
      });

      test('3D mgrid', () {
        NDArray.scope(() {
          final grid = mgrid([
            GridRange(0, 2, step: 1),
            GridRange(0, 2, step: 1),
            GridRange(0, 2, step: 1),
          ]);

          expect(grid.shape, [3, 2, 2, 2]);
          expect(grid.slice([Index(0)]).toList(), [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
          ]);
        });
      });

      test('empty mgrid throws', () {
        expect(() => mgrid([]), throwsArgumentError);
      });
    });
    test(
      'nansum() and nanmean() ufunc correctness across float dtypes',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, double.nan, 3.0, double.nan, 5.0],
          [5],
          DType.float64,
        );

        // 1. nansum
        final sF64 = nansum(a);
        expect(sF64.shape, <int>[]);
        expect(sF64.dtype, DType.float64);
        expect(sF64.scalar, 9.0); // 1 + 3 + 5 = 9

        // 2. nanmean
        final mF64 = nanmean(a);
        expect(mF64.shape, <int>[]);
        expect(mF64.dtype, DType.float64);
        expect(mF64.scalar, 3.0); // 9 / 3 = 3
      }),
    );

    test(
      'nansum() and nanmean() along axis check',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, double.nan, 3.0, 4.0, 5.0, double.nan],
          [2, 3],
          DType.float64,
        );

        // Integrate columns (axis = 1)
        final sAxis1 = nansum(a, axis: 1);
        expect(sAxis1.shape, [2]);
        expect(sAxis1.toList(), [4.0, 9.0]); // row 0: 1+3=4, row 1: 4+5=9

        final mAxis1 = nanmean(a, axis: 1);
        expect(mAxis1.shape, [2]);
        expect(mAxis1.toList(), [2.0, 4.5]); // row 0: 4/2=2.0, row 1: 9/2=4.5

        // Integrate rows (axis = 0)
        final sAxis0 = nansum(a, axis: 0);
        expect(sAxis0.shape, [3]);
        expect(sAxis0.toList(), [
          5.0,
          5.0,
          3.0,
        ]); // col 0: 1+4=5, col 1: 5, col 2: 3

        final mAxis0 = nanmean(a, axis: 0);
        expect(mAxis0.shape, [3]);
        expect(mAxis0.toList(), [
          2.5,
          5.0,
          3.0,
        ]); // col 0: 5/2=2.5, col 1: 5/1=5, col 2: 3/1=3
      }),
    );

    test(
      'nansum() and nanmean() on empty/disposed/preconditions',
      () => NDArray.scope(() {
        final empty = NDArray.zeros([0], DType.float64);
        expect(nansum(empty).scalar, 0.0);
        expect(nanmean<double>(empty).scalar.isNaN, true);

        final a = NDArray.fromList([1.0, double.nan], [2], DType.float64);
        a.dispose();
        expect(() => nansum(a), throwsStateError);
        expect(() => nanmean(a), throwsStateError);
      }),
    );

    test(
      'variance() and std() on non-contiguous view calculation',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );

        final viewT = parent.transposed;

        expect(variance(viewT).scalar, closeTo(17.5 / 6.0, 1e-9));
        expect(std(viewT).scalar, closeTo(math.sqrt(17.5 / 6.0), 1e-9));
      }),
    );

    test(
      'variance() and std() on sliced view calculation',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );

        final view = parent.slice([Slice(start: 0, stop: 2), Slice.all()]);
        expect(variance(view).scalar, closeTo(1.25, 1e-9));
        expect(std(view).scalar, closeTo(math.sqrt(1.25), 1e-9));
      }),
    );

    test(
      'percentile() and quantile() statistical estimation correctness',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [15.0, 20.0, 35.0, 40.0, 50.0],
          [5],
          DType.float64,
        );

        // 1. percentile (q = 30) -> idx = 4 * 0.3 = 1.2 -> 20 + 0.2*(35-20) = 23.0
        final p30 = percentile(a, 30.0);
        expect(p30.scalar, closeTo(23.0, 1e-9));

        // 2. quantile (q = 0.30)
        final q30 = quantile(a, 0.3);
        expect(q30.scalar, closeTo(23.0, 1e-9));
      }),
    );

    test(
      'percentile() and quantile() invalid inputs and limits checks',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);

        expect(() => percentile(a, -5.0), throwsArgumentError);
        expect(() => percentile(a, 105.0), throwsArgumentError);
        expect(() => quantile(a, -0.1), throwsArgumentError);
        expect(() => quantile(a, 1.1), throwsArgumentError);
      }),
    );

    test(
      'percentile() along columns and rows axis integration correctness',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 5.0, 3.0, 4.0, 2.0, 6.0],
          [2, 3],
          DType.float64,
        );

        // axis 1: row 0: 1, 3, 5 -> q=50 -> 3.0; row 1: 2, 4, 6 -> q=50 -> 4.0
        final p50Axis1 = percentile(a, 50.0, axis: 1);
        expect(p50Axis1.shape, [2]);
        expect(p50Axis1.toList(), [3.0, 4.0]);

        // axis 0: col 0: 1, 4 -> q=50 -> 2.5; col 1: 5, 2 -> 3.5; col 2: 3, 6 -> 4.5
        final p50Axis0 = percentile(a, 50.0, axis: 0);
        expect(p50Axis0.shape, [3]);
        expect(p50Axis0.toList(), [2.5, 3.5, 4.5]);
      }),
    );

    test(
      'percentile() quantile() contiguous float32 coverage',
      () => NDArray.scope(() {
        final a = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float32);

        final p = percentile(a, 50.0);
        expect(p.dtype, DType.float64); // percentile always returns float64
        expect(p.scalar, closeTo(20.0, 1e-9));

        final q = quantile(a, 0.5);
        expect(q.dtype, DType.float64);
        expect(q.scalar, closeTo(20.0, 1e-9));
      }),
    );

    test(
      'percentile() quantile() non-contiguous transposed views',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float64,
        );
        final transposed = parent.transposed; // non-contiguous [2, 2]
        expect(transposed.isContiguous, false);

        // transposed: [[10.0, 30.0], [20.0, 40.0]]
        final p = percentile(transposed, 50.0, axis: 1);
        expect(p.shape, [2]);
        expect(p.toList(), [20.0, 30.0]); // row 0 median: 20, row 1 median: 30
      }),
    );

    test(
      'percentile() quantile() with out parameter recycler',
      () => NDArray.scope(() {
        final a = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
        final outPercentile = NDArray<double>.zeros([], DType.float64);

        final resP = percentile(a, 50.0, out: outPercentile);
        expect(identical(resP, outPercentile), true);
        expect(outPercentile.scalar, 20.0);

        final outQuantile = NDArray<double>.zeros([], DType.float64);
        final resQ = quantile(a, 0.5, out: outQuantile);
        expect(identical(resQ, outQuantile), true);
        expect(outQuantile.scalar, 20.0);

        // Incompatible shape throws ArgumentError
        final badOut = NDArray<double>.zeros([2], DType.float64);
        expect(() => percentile(a, 50.0, out: badOut), throwsArgumentError);
        expect(() => quantile(a, 0.5, out: badOut), throwsArgumentError);
      }),
    );
  });
}

void expectListEqualsWithNaNs(List actual, List expected) {
  expect(actual.length, expected.length);
  for (var i = 0; i < actual.length; i++) {
    final a = actual[i];
    final e = expected[i];
    if (a is double && e is double) {
      if (a.isNaN && e.isNaN) continue;
    }
    if (a is Complex && e is Complex) {
      if ((a.real.isNaN && e.real.isNaN || a.real == e.real) &&
          (a.imag.isNaN && e.imag.isNaN || a.imag == e.imag)) {
        continue;
      }
    }
    expect(a, e);
  }
}
