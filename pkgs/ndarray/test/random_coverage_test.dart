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

    test('FFT float32 and exponential FFI acceleration correctness', () {
      NDArray.scope(() {
        // 1. uniform float32 (FFI)
        final u32 = uniform([100], dtype: DType.float32);
        expect(u32.dtype, DType.float32);
        expect(u32.shape, [100]);
        for (var i = 0; i < 100; i++) {
          expect(u32.data[i], greaterThanOrEqualTo(0.0));
          expect(u32.data[i], lessThan(1.0));
        }

        // 2. exponential float64
        final exp64 = exponential([50], scale: 2.0, dtype: DType.float64);
        expect(exp64.dtype, DType.float64);
        expect(exp64.shape, [50]);
        for (var i = 0; i < 50; i++) {
          expect(exp64.data[i], greaterThanOrEqualTo(0.0));
        }

        // 3. exponential float32
        final exp32 = exponential([50], scale: 1.5, dtype: DType.float32);
        expect(exp32.dtype, DType.float32);
        expect(exp32.shape, [50]);
        for (var i = 0; i < 50; i++) {
          expect(exp32.data[i], greaterThanOrEqualTo(0.0));
        }
      });
    });

    test('secure random generation across all 8 distributions', () {
      NDArray.scope(() {
        // 1. uniform
        final uniSec = uniform([10], dtype: DType.float64, secure: true);
        expect(uniSec.shape, [10]);
        for (var i = 0; i < 10; i++) {
          expect(uniSec.data[i], greaterThanOrEqualTo(0.0));
          expect(uniSec.data[i], lessThan(1.0));
        }

        // 2. randint
        final riSec = randint(
          [10],
          low: -5,
          high: 15,
          dtype: DType.int32,
          secure: true,
        );
        expect(riSec.shape, [10]);
        for (var i = 0; i < 10; i++) {
          expect(riSec.data[i], greaterThanOrEqualTo(-5));
          expect(riSec.data[i], lessThan(15));
        }

        // 2b. randint with huge range (> 2^32)
        final riSecHuge = randint(
          [5],
          low: 0,
          high: 100000000000000,
          dtype: DType.int64,
          secure: true,
        );
        expect(riSecHuge.shape, [5]);

        // 3. normal
        final normSec = normal(
          [10],
          loc: 5.0,
          scale: 2.0,
          dtype: DType.float32,
          secure: true,
        );
        expect(normSec.shape, [10]);
        expect(normSec.dtype, DType.float32);

        // 4. exponential
        final expSec = exponential(
          [10],
          scale: 3.0,
          dtype: DType.float64,
          secure: true,
        );
        expect(expSec.shape, [10]);
        for (var i = 0; i < 10; i++) {
          expect(expSec.data[i], greaterThanOrEqualTo(0.0));
        }

        // 5. poisson
        final poiSec = poisson(
          [10],
          lam: 4.0,
          dtype: DType.int64,
          secure: true,
        );
        expect(poiSec.shape, [10]);
        for (var i = 0; i < 10; i++) {
          expect(poiSec.data[i], greaterThanOrEqualTo(0));
        }

        // 6. binomial (small n)
        final binSecSmall = binomial(
          [10],
          n: 10,
          p: 0.5,
          dtype: DType.int32,
          secure: true,
        );
        expect(binSecSmall.shape, [10]);
        for (var i = 0; i < 10; i++) {
          expect(binSecSmall.data[i], greaterThanOrEqualTo(0));
          expect(binSecSmall.data[i], lessThanOrEqualTo(10));
        }

        // 7. binomial (large n)
        final binSecLarge = binomial(
          [10],
          n: 100,
          p: 0.4,
          dtype: DType.int64,
          secure: true,
        );
        expect(binSecLarge.shape, [10]);
        for (var i = 0; i < 10; i++) {
          expect(binSecLarge.data[i], greaterThanOrEqualTo(0));
          expect(binSecLarge.data[i], lessThanOrEqualTo(100));
        }

        // 7b. binomial (large n) with int32
        final binSecLarge32 = binomial(
          [5],
          n: 100,
          p: 0.35,
          dtype: DType.int32,
          secure: true,
        );
        expect(binSecLarge32.shape, [5]);
        expect(binSecLarge32.dtype, DType.int32);

        // 8. multivariateNormal
        final mean = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final cov = NDArray.fromList(
          [1.0, 0.0, 0.0, 1.0],
          [2, 2],
          DType.float64,
        );
        final mvSec = multivariateNormal(mean, cov, size: [5], secure: true);
        expect(mvSec.shape, [5, 2]);

        // 9. multinomial
        final pvals = NDArray.fromList([0.2, 0.5, 0.3], [3], DType.float64);
        final multiSec = multinomial(10, pvals, size: [5], secure: true);
        expect(multiSec.shape, [5, 3]);
      });
    });
  });
}
