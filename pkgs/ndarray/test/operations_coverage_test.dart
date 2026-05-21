import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Operations Extra Coverage Tests', () {
    test('setNumThreads coverage', () {
      NDArray.scope(() {
        // Valid thread sets
        setNumThreads(1);
        setNumThreads(2);

        // Invalid thread sets
        expect(() => setNumThreads(0), throwsArgumentError);
        expect(() => setNumThreads(-1), throwsArgumentError);
      });
    });

    test('Float32 global ufuncs multiply and divide contiguous fast paths', () {
      NDArray.scope(() {
        final a = NDArray.fromList([2.0, 4.0, 6.0, 8.0], [2, 2], DType.float32);
        final b = NDArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float32,
        );

        // Global multiply
        final prod = multiply(a, b);
        expect(prod.dtype, DType.float32);
        expect(prod.toList(), [20.0, 80.0, 180.0, 320.0]);

        // Global divide
        final quot = divide(b, a);
        expect(quot.dtype, DType.float32);
        expect(quot.toList(), [5.0, 5.0, 5.0, 5.0]);
      });
    });

    test('Mixed divisions strided view fallback coverage', () {
      NDArray.scope(() {
        // Float64 / Int64
        final a = NDArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float64,
        ).transpose(); // view
        final b = NDArray.fromList(
          [2, 4, 6, 8],
          [2, 2],
          DType.int64,
        ).transpose(); // view
        final res1 = divide(a, b);
        expect(res1.toList(), [5.0, 5.0, 5.0, 5.0]);

        // Int64 / Float64
        final res2 = divide(b, a);
        expect(res2.toList(), [0.2, 0.2, 0.2, 0.2]);

        // Int32 / Int32
        final c = NDArray.fromList(
          [10, 20, 30, 40],
          [2, 2],
          DType.int32,
        ).transpose(); // view
        final d = NDArray.fromList(
          [2, 4, 5, 8],
          [2, 2],
          DType.int32,
        ).transpose(); // view
        final res3 = divide(c, d);
        expect(res3.toList(), [5.0, 6.0, 5.0, 5.0]);
      });
    });

    test('High-dimensional stack broadcasted matmul coverage', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        // Shape [2, 2, 2]
        final b = NDArray.fromList(
          [1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0],
          [2, 2, 2],
          DType.float64,
        );

        // matmul a [2, 2] and b [2, 2, 2] -> result [2, 2, 2]
        final res1 = matmul(a, b);
        expect(res1.shape, [2, 2, 2]);
        expect(res1.toList(), [1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0]);

        // matmul b [2, 2, 2] and a [2, 2] -> result [2, 2, 2]
        final res2 = matmul(b, a);
        expect(res2.shape, [2, 2, 2]);
        expect(res2.toList(), [1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0]);
      });
    });

    test(
      'hanning and hamming window functions type safety and calculation',
      () {
        NDArray.scope(() {
          // Hanning window
          final h1 = hanning(5, dtype: DType.float64);
          expect(h1.shape, [5]);
          expect(h1.dtype, DType.float64);
          expect(h1.data[0], 0.0);
          expect(h1.data[2], 1.0); // center value
          expect(h1.data[4], 0.0);

          final h1_32 = hanning(5, dtype: DType.float32);
          expect(h1_32.dtype, DType.float32);
          expect(h1_32.data[2], 1.0);

          // Hamming window
          final h2 = hamming(5, dtype: DType.float64);
          expect(h2.shape, [5]);
          expect(h2.dtype, DType.float64);
          expect(h2.data[0], closeTo(0.08, 1e-9));
          expect(h2.data[2], 1.0); // center value
          expect(h2.data[4], closeTo(0.08, 1e-9));

          final h2_32 = hamming(5, dtype: DType.float32);
          expect(h2_32.dtype, DType.float32);
          expect(h2_32.data[2], 1.0);

          // M <= 1 edge cases
          final zeroH = hanning(0);
          expect(zeroH.shape, [0]);
          final oneH = hanning(1);
          expect(oneH.shape, [1]);
          expect(oneH.data[0], 1.0);
        });
      },
    );
  });
}
