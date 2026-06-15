import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
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
        final x1 = NDArray<Float64>.fromList([1.0, 2.0], [1, 2], DType.float64);
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
        final xb = NDArray<Float64>.fromList([3.0, 4.0], [1, 2], DType.float64);

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
}
