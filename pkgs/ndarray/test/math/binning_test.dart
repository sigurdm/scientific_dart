import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Binning Operations Tests', () {
    group('bincount', () {
      test('simple bincount', () {
        NDArray.scope(() {
          final x = NDArray.fromList([1, 1, 2, 4], [4], DType.int32);
          final res = bincount(x);
          expect(res.dtype, DType.int64);
          expect(res.shape, [5]);
          expect(res.toList(), [0, 2, 1, 0, 1]);
        });
      });

      test('bincount with minlength', () {
        NDArray.scope(() {
          final x = NDArray.fromList([1, 1, 2], [3], DType.int32);
          final res = bincount(x, minlength: 5);
          expect(res.shape, [5]);
          expect(res.toList(), [0, 2, 1, 0, 0]);
        });
      });

      test('bincount with weights (double)', () {
        NDArray.scope(() {
          final x = NDArray.fromList([1, 1, 2], [3], DType.int32);
          final w = NDArray.fromList([0.5, 1.5, 2.0], [3], DType.float64);
          final res = bincount(x, weights: w);
          expect(res.dtype, DType.float64);
          expect(res.shape, [3]);
          expect(res.toList(), [0.0, 2.0, 2.0]);
        });
      });

      test('bincount with weights (float)', () {
        NDArray.scope(() {
          final x = NDArray.fromList([1, 1, 2], [3], DType.int32);
          final w = NDArray.fromList([0.5, 1.5, 2.0], [3], DType.float32);
          final res = bincount(x, weights: w);
          expect(res.dtype, DType.float32);
          expect(res.shape, [3]);
          expect(res.toList(), [0.0, 2.0, 2.0]);
        });
      });

      test('bincount strided input', () {
        NDArray.scope(() {
          final x = NDArray.fromList([1, 2, 3, 1, 4, 1], [2, 3], DType.int32);
          final col0 = x.slice([Slice.all(), Index(0)]);
          expect(col0.isContiguous, false);
          final res = bincount(col0 as NDArray<int>);
          expect(res.shape, [2]);
          expect(res.toList(), [0, 2]);
        });
      });

      test('bincount error cases', () {
        NDArray.scope(() {
          final x = NDArray.fromList([-1, 1], [2], DType.int32);
          expect(() => bincount(x), throwsArgumentError);

          final x2D = NDArray.fromList([1, 2], [1, 2], DType.int32);
          expect(() => bincount(x2D), throwsArgumentError);

          final xUint8 = NDArray.fromList([1, 2], [2], DType.uint8);
          expect(() => bincount(xUint8), throwsArgumentError);
        });
      });
    });

    group('digitize', () {
      test('digitize increasing bins', () {
        NDArray.scope(() {
          final x = NDArray.fromList([0.2, 6.4, 3.0, 1.6], [4], DType.float64);
          final bins = NDArray.fromList(
            [0.0, 1.0, 2.5, 4.0, 10.0],
            [5],
            DType.float64,
          );
          final res = digitize(x, bins);
          expect(res.dtype, DType.int32);
          expect(res.toList(), [1, 4, 3, 2]);
        });
      });

      test('digitize decreasing bins', () {
        NDArray.scope(() {
          final x = NDArray.fromList([0.2, 6.4, 3.0, 1.6], [4], DType.float64);
          final bins = NDArray.fromList(
            [10.0, 4.0, 2.5, 1.0, 0.0],
            [5],
            DType.float64,
          );
          final res = digitize(x, bins);
          expect(res.toList(), [4, 1, 2, 3]);
        });
      });

      test('digitize right=true', () {
        NDArray.scope(() {
          final x = NDArray.fromList([1.0, 2.5, 4.0], [3], DType.float64);
          final bins = NDArray.fromList([1.0, 2.5, 4.0], [3], DType.float64);
          final res = digitize(x, bins, right: true);
          expect(res.toList(), [0, 1, 2]);
        });
      });

      test('digitize right=false (default)', () {
        NDArray.scope(() {
          final x = NDArray.fromList([1.0, 2.5, 4.0], [3], DType.float64);
          final bins = NDArray.fromList([1.0, 2.5, 4.0], [3], DType.float64);
          final res = digitize(x, bins, right: false);
          expect(res.toList(), [1, 2, 3]);
        });
      });

      test('digitize error cases', () {
        NDArray.scope(() {
          final x = NDArray.fromList([1.0], [1], DType.float64);
          final binsNonMono = NDArray.fromList(
            [1.0, 3.0, 2.0],
            [3],
            DType.float64,
          );
          expect(() => digitize(x, binsNonMono), throwsArgumentError);
        });
      });
    });

    group('histogram', () {
      test('uniform bins', () {
        NDArray.scope(() {
          final x = NDArray.fromList([1.0, 2.0, 1.0], [3], DType.float64);
          final res = histogram(x, bins: 2, range: (1.0, 2.0));
          expect(res.hist.toList(), [2, 1]);
          expect(res.binEdges.toList(), [1.0, 1.5, 2.0]);
        });
      });

      test('custom bins', () {
        NDArray.scope(() {
          final x = NDArray.fromList([0.5, 1.5, 2.5, 3.5], [4], DType.float64);
          final bins = NDArray.fromList(
            [0.0, 1.0, 3.0, 4.0],
            [4],
            DType.float64,
          );
          final res = histogram(x, bins: bins);
          expect(res.hist.toList(), [1, 2, 1]);
        });
      });

      test('histogram with weights', () {
        NDArray.scope(() {
          final x = NDArray.fromList([1.0, 2.0, 1.0], [3], DType.float64);
          final w = NDArray.fromList([0.5, 1.0, 1.5], [3], DType.float64);
          final res = histogram(x, bins: 2, range: (1.0, 2.0), weights: w);
          expect(res.hist.toList(), [2.0, 1.0]);
        });
      });

      test('histogram density=true', () {
        NDArray.scope(() {
          final x = NDArray.fromList([1.0, 2.0, 1.0], [3], DType.float64);
          final res = histogram(x, bins: 2, range: (1.0, 2.0), density: true);
          expect(res.hist.toList()[0], closeTo(1.3333333333333333, 1e-7));
          expect(res.hist.toList()[1], closeTo(0.6666666666666666, 1e-7));
        });
      });
    });
  });
}
