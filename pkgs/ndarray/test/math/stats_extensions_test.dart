import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('ptp tests', () {
    test('ptp flat contiguous axis=null', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 5.0, 3.0, 10.0, 2.0],
          [5],
          DType.float64,
        );
        final res = ptp(a);
        expect(res.shape, <int>[]);
        expect(res.dtype, DType.float64);
        expect(res.scalar, 9.0);
      });
    });

    test('ptp 2D contiguous axis=0', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final res = ptp(a, axis: 0);
        expect(res.shape, [3]);
        expect(res.dtype, DType.float64);
        expect(res.toList(), [3.0, 3.0, 3.0]);
      });
    });

    test('ptp 2D contiguous axis=1', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final res = ptp(a, axis: 1);
        expect(res.shape, [2]);
        expect(res.dtype, DType.float64);
        expect(res.toList(), [2.0, 2.0]);
      });
    });

    test('ptp strided (non-contiguous) array', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );
        final transposed = a.transpose(); // shape [2, 3], non-contiguous
        // transposed is:
        // [[1.0, 3.0, 5.0],
        //  [2.0, 4.0, 6.0]]

        final res = ptp(transposed, axis: 1); // ptp along rows
        expect(res.shape, [2]);
        expect(res.toList(), [4.0, 4.0]); // (5-1), (6-2)
      });
    });

    test('ptp with out parameter', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 5.0, 2.0, 10.0], [4], DType.float64);
        final out = NDArray<double>.zeros([], DType.float64);
        final res = ptp(a, out: out);
        expect(identical(res, out), true);
        expect(out.scalar, 9.0);
      });
    });

    test('ptp integer types', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1, 5, 2, 10], [4], DType.int32);
        final res = ptp(a);
        expect(res.dtype, DType.int32);
        expect(res.scalar, 9);
      });
    });

    test('ptp validation', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        a.dispose();
        expect(() => ptp(a), throwsStateError);
      });

      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final out = NDArray<double>.zeros([
          2,
        ], DType.float64); // incompatible shape, should be []
        expect(() => ptp(a, out: out), throwsArgumentError);
      });

      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final out = NDArray<double>.zeros(
          [],
          DType.float32,
        ); // incompatible dtype
        expect(() => ptp(a, out: out), throwsArgumentError);
      });
    });
  });

  group('average tests', () {
    test('average without weights (same as mean)', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final res = average(a);
        expect(res.average.shape, <int>[]);
        expect(res.average.scalar, 2.5);
        expect(res.sumOfWeights, null);
      });
    });

    test('average with 1D weights axis=0', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final w = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final res = average(a, weights: w, returned: true);

        // weighted sum = 1*1 + 2*2 + 3*3 + 4*4 = 1 + 4 + 9 + 16 = 30
        // sum of weights = 1 + 2 + 3 + 4 = 10
        // avg = 30 / 10 = 3.0
        expect(res.average.shape, <int>[]);
        expect(res.average.scalar, 3.0);
        expect(res.sumOfWeights, isNotNull);
        expect(res.sumOfWeights!.shape, <int>[]);
        expect(res.sumOfWeights!.scalar, 10.0);
      });
    });

    test('average with 1D weights axis=1 (2D input)', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final w = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final res = average(a, axis: 1, weights: w, returned: true);

        // Row 1: (1*1 + 2*2 + 3*3) / 6 = (1 + 4 + 9) / 6 = 14 / 6 = 2.3333333333333335
        // Row 2: (4*1 + 5*2 + 6*3) / 6 = (4 + 10 + 18) / 6 = 32 / 6 = 5.333333333333333
        // sum of weights = 1 + 2 + 3 = 6
        expect(res.average.shape, [2]);
        expect(res.average.toList()[0], closeTo(2.3333333333333335, 1e-9));
        expect(res.average.toList()[1], closeTo(5.333333333333333, 1e-9));
        expect(res.sumOfWeights, isNotNull);
        expect(res.sumOfWeights!.shape, [2]);
        expect(res.sumOfWeights!.toList(), [6.0, 6.0]);
      });
    });

    test('average with ND weights', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final w = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);

        final res = average(a, weights: w, returned: true); // global average
        // weighted sum = 1*1 + 2*2 + 3*3 + 4*4 = 30
        // sum of weights = 1 + 2 + 3 + 4 = 10
        // avg = 3.0
        expect(res.average.shape, <int>[]);
        expect(res.average.scalar, 3.0);
        expect(res.sumOfWeights!.scalar, 10.0);
      });
    });

    test('average with ND weights along axis=0', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final w = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);

        final res = average(a, axis: 0, weights: w, returned: true);
        // col 0: (1*1 + 3*3) / (1+3) = 10 / 4 = 2.5
        // col 1: (2*2 + 4*4) / (2+4) = 20 / 6 = 3.3333333333333335
        // sum of weights: [4.0, 6.0]
        expect(res.average.shape, [2]);
        expect(res.average.toList()[0], 2.5);
        expect(res.average.toList()[1], closeTo(3.3333333333333335, 1e-9));
        expect(res.sumOfWeights!.toList(), [4.0, 6.0]);
      });
    });

    test('average with out parameter', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final w = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final out = NDArray<double>.zeros([], DType.float64);
        final res = average(a, weights: w, out: out);
        expect(identical(res.average, out), true);
        expect(out.scalar, 3.0);
      });
    });

    test('average integer types promotion', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
        final w = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
        final res = average<int, int, double>(a, weights: w, returned: true);

        expect(res.average.dtype, DType.float64);
        expect(res.average.scalar, 3.0);
        expect(
          res.sumOfWeights!.dtype,
          DType.float64,
        ); // should be promoted to R (double)
        expect(res.sumOfWeights!.scalar, 10.0);
      });
    });

    test('average validation', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        a.dispose();
        expect(() => average(a), throwsStateError);
      });

      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final w = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        // length mismatch
        expect(() => average(a, weights: w), throwsArgumentError);
      });

      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final w = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        // axis is null, but weights is 1-D and a is 2-D
        expect(() => average(a, weights: w), throwsArgumentError);
      });
    });
  });
}
