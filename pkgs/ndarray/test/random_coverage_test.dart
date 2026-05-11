import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Random Module Extra Coverage Tests', () {
    test('uniform exception and recycler mismatch coverage', () {
      NDArray.scope(() {
        // Invalid dtype
        expect(() => uniform([2, 2], dtype: DType.int32), throwsArgumentError);

        // Incompatible into shape
        expect(
          () => uniform([
            2,
            2,
          ], into: NDArray<Float64>.create([3], DType.float64)),
          throwsArgumentError,
        );
        // Incompatible into dtype
        expect(
          () => uniform(
            [2, 2],
            dtype: DType.float64,
            into: NDArray<Float32>.create([2, 2], DType.float32) as dynamic,
          ),
          throwsArgumentError,
        );
      });
    });

    test('randint exception and recycler mismatch coverage', () {
      NDArray.scope(() {
        // Invalid low >= high
        expect(() => randint([2, 2], low: 10, high: 10), throwsArgumentError);
        // Invalid dtype
        expect(
          () => randint([2, 2], low: 0, high: 5, dtype: DType.float64),
          throwsArgumentError,
        );

        // Incompatible into shape
        expect(
          () => randint(
            [2, 2],
            low: 0,
            high: 5,
            into: NDArray<Int64>.create([3], DType.int64),
          ),
          throwsArgumentError,
        );
      });
    });

    test('normal exception and recycler mismatch coverage', () {
      NDArray.scope(() {
        // Invalid scale <= 0.0
        expect(() => normal([2, 2], scale: 0.0), throwsArgumentError);
        // Invalid dtype
        expect(() => normal([2, 2], dtype: DType.int64), throwsArgumentError);

        // Incompatible into shape
        expect(
          () =>
              normal([2, 2], into: NDArray<Float64>.create([3], DType.float64)),
          throwsArgumentError,
        );
      });
    });

    test('exponential exception and recycler mismatch coverage', () {
      NDArray.scope(() {
        // Invalid scale <= 0.0
        expect(() => exponential([2, 2], scale: 0.0), throwsArgumentError);
        // Invalid dtype
        expect(
          () => exponential([2, 2], dtype: DType.int64),
          throwsArgumentError,
        );

        // Incompatible into shape
        expect(
          () => exponential([
            2,
            2,
          ], into: NDArray<Float64>.create([3], DType.float64)),
          throwsArgumentError,
        );
      });
    });

    test('poisson exception and recycler mismatch coverage', () {
      NDArray.scope(() {
        // Invalid lam <= 0.0
        expect(() => poisson([2, 2], lam: 0.0), throwsArgumentError);
        // Invalid dtype
        expect(
          () => poisson([2, 2], dtype: DType.float64),
          throwsArgumentError,
        );

        // Incompatible into shape
        expect(
          () => poisson([2, 2], into: NDArray<Int64>.create([3], DType.int64)),
          throwsArgumentError,
        );
      });
    });

    test('binomial exception and recycler mismatch coverage', () {
      NDArray.scope(() {
        // Invalid n < 0
        expect(() => binomial([2, 2], n: -1, p: 0.5), throwsArgumentError);
        // Invalid p out of [0, 1]
        expect(() => binomial([2, 2], n: 10, p: 1.5), throwsArgumentError);
        // Invalid dtype
        expect(
          () => binomial([2, 2], n: 10, p: 0.5, dtype: DType.float64),
          throwsArgumentError,
        );

        // Incompatible into shape
        expect(
          () => binomial(
            [2, 2],
            n: 10,
            p: 0.5,
            into: NDArray<Int64>.create([3], DType.int64),
          ),
          throwsArgumentError,
        );
      });
    });

    test('multivariateNormal exception and recycler mismatch coverage', () {
      NDArray.scope(() {
        final mean = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final cov = NDArray.fromList(
          [1.0, 0.0, 0.0, 1.0],
          [2, 2],
          DType.float64,
        );

        // Invalid cov length/dimensions (non-2D or non-square)
        expect(
          () => multivariateNormal(mean, NDArray.zeros([3], DType.float64)),
          throwsArgumentError,
        );
        // Invalid dtype
        expect(
          () => multivariateNormal(mean, cov, dtype: DType.int64),
          throwsArgumentError,
        );

        // Incompatible into shape
        expect(
          () => multivariateNormal(
            mean,
            cov,
            size: [2],
            into: NDArray<Float64>.create([3], DType.float64),
          ),
          throwsArgumentError,
        );

        // Valid into recycler buffer
        final validInto = NDArray<Float64>.create([2, 2], DType.float64);
        final res = multivariateNormal(mean, cov, size: [2], into: validInto);
        expect(res == validInto, true);
        expect(res.shape, [2, 2]);
      });
    });

    test('multinomial exception and recycler mismatch coverage', () {
      NDArray.scope(() {
        final pvals = NDArray.fromList([0.2, 0.5, 0.3], [3], DType.float64);

        // Invalid dtype
        expect(
          () => multinomial(10, pvals, dtype: DType.float64),
          throwsArgumentError,
        );

        // Incompatible into shape
        expect(
          () => multinomial(
            10,
            pvals,
            size: [2],
            into: NDArray<Int32>.create([3], DType.int32),
          ),
          throwsArgumentError,
        );

        // Valid into recycler buffer
        final validInto = NDArray<Int32>.create([2, 3], DType.int32);
        final res = multinomial(10, pvals, size: [2], into: validInto);
        expect(res == validInto, true);
        expect(res.shape, [2, 3]);
      });
    });
  });
}
