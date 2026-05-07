import 'package:num_dart/num_dart.dart';
import 'package:test/test.dart';
import 'dart:math' as math;
import 'dart:typed_data';

void main() {
  group('NDArray Random Distributions Tests', () {
    group('normal distribution tests', () {
      test('Shape, type, and standard parameter checks', () {
        final a = normal([2, 5], loc: 0.0, scale: 1.0, dtype: DType.float64);
        expect(a.shape, [2, 5]);
        expect(a.dtype, DType.float64);
        expect(a.data.length, 10);
      });

      test('Statistical mean and variance validation (Law of Large Numbers)', () {
        const loc = 10.0;
        const scale = 3.0;
        // Draw a large sample size to confirm sample parameters match target distributions
        final a = normal([10000], loc: loc, scale: scale, dtype: DType.float64);

        final sampleMean = mean(a) as double;
        final sampleStd = std(a) as double;

        // Over 10,000 draws, sample mean should be very close to loc, and stddev close to scale!
        expect(sampleMean, closeTo(loc, 0.1));
        expect(sampleStd, closeTo(scale, 0.1));
      });

      test('Throws on invalid parameters', () {
        expect(() => normal([2], scale: 0.0), throwsArgumentError);
        expect(() => normal([2], scale: -1.5), throwsArgumentError);
        expect(() => normal([2], dtype: DType.int32), throwsArgumentError);
      });
    });

    group('exponential distribution tests', () {
      test('Basic property and statistics validation', () {
        const scale = 2.5;
        final a = exponential([5000], scale: scale, dtype: DType.float64);

        expect(a.shape, [5000]);
        // All values in an exponential distribution are strictly non-negative
        for (final val in a.data) {
          expect(val, greaterThanOrEqualTo(0.0));
        }

        final sampleMean = mean(a) as double;
        // Exponential distribution mean is exactly equal to its scale parameter (beta)
        expect(sampleMean, closeTo(scale, 0.15));
      });
    });

    group('poisson distribution tests', () {
      test('Small lambda exact Knuth path checks', () {
        const lam = 4.0;
        final a = poisson([5000], lam: lam, dtype: DType.int64);
        expect(a.dtype, DType.int64);

        final sampleMean = mean(a) as double;
        final sampleVar = variance(a) as double;

        // In a Poisson distribution, both Mean and Variance are exactly equal to lambda!
        expect(sampleMean, closeTo(lam, 0.15));
        expect(sampleVar, closeTo(lam, 0.25));
      });

      test('Large lambda Gaussian Approximation path triggers safely', () {
        const lam = 50.0;
        // Large lam triggers Track B to avoid numerical underflow or infinite loops!
        final a = poisson([5000], lam: lam, dtype: DType.int32);
        expect(a.dtype, DType.int32);

        final sampleMean = mean(a) as double;
        expect(sampleMean, closeTo(lam, 0.3));
        for (final val in a.data) {
          expect(val, greaterThanOrEqualTo(0));
        }
      });
    });

    group('binomial distribution tests', () {
      test('Small n Bernoulli trials path checks', () {
        const n = 20;
        const p = 0.4;
        final a = binomial([5000], n: n, p: p, dtype: DType.int64);

        expect(a.dtype, DType.int64);
        for (final val in a.data) {
          expect(val, greaterThanOrEqualTo(0));
          expect(val, lessThanOrEqualTo(n));
        }

        final sampleMean = mean(a) as double;
        final expectedMean = n * p; // 20 * 0.4 = 8.0
        expect(sampleMean, closeTo(expectedMean, 0.2));
      });

      test('Large n Normal Approximation triggers safely', () {
        const n = 1000;
        const p = 0.3;
        final a = binomial([5000], n: n, p: p, dtype: DType.int32);

        expect(a.dtype, DType.int32);
        final sampleMean = mean(a) as double;
        final expectedMean = n * p; // 1000 * 0.3 = 300.0
        expect(sampleMean, closeTo(expectedMean, 1.0));
      });
    });

    group('Seeded Reproducibility and Determinism checks', () {
      test('Seeding Random(seed) guarantees exact reproducible draws', () {
        final shape = [2, 3];

        // Create two separate Random engines with matching seeds!
        final r1 = math.Random(12345);
        final r2 = math.Random(12345);

        final n1 = normal(shape, loc: 5.0, scale: 2.0, random: r1);
        final n2 = normal(shape, loc: 5.0, scale: 2.0, random: r2);
        expect(n1.toList(), n2.toList()); // exact matching doubles!

        final e1 = exponential([10], scale: 1.5, random: math.Random(42));
        final e2 = exponential([10], scale: 1.5, random: math.Random(42));
        expect(e1.toList(), e2.toList());

        final p1 = poisson([10], lam: 12.0, random: math.Random(7));
        final p2 = poisson([10], lam: 12.0, random: math.Random(7));
        expect(p1.toList(), p2.toList());

        final b1 = binomial([10], n: 100, p: 0.2, random: math.Random(999));
        final b2 = binomial([10], n: 100, p: 0.2, random: math.Random(999));
        expect(b1.toList(), b2.toList());
      });
    });
  });
}
