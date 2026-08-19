import 'package:criterion/criterion.dart';
import 'package:ndarray/ndarray.dart';

void main() async {
  setNumThreads(1);
  const size = 100000;
  const gridDim = 500;

  await criterion(
    'NDArray Calculus & Numerical Integration Benchmark Suite',
    (c) {
      final vec1d = linspace<double>(0.0, 100.0, size, dtype: DType.float64);

      final grid2d = linspace<double>(
        0.0,
        100.0,
        gridDim * gridDim,
        dtype: DType.float64,
      ).reshape([gridDim, gridDim]);

      c.group('1. Numerical Differentiation', () {
        c.bench('diff(a, n=1) [100k points]', () {
          final res = diff(vec1d);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('diff(a, n=2) [100k points]', () {
          final res = diff(vec1d, n: 2);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench(
          'gradientArray(grid2D) [500x500]',
          () {
            final grads = gradientArray(grid2d);
            for (final g in grads) {
              blackhole(g);
              g.dispose();
            }
          },
          throughput: Throughput.elements(gridDim * gridDim),
        );
      });

      c.group('2. Numerical Integration', () {
        c.bench(
          'trapz(y, spacing=0.01) [100k points]',
          () {
            final res = trapz(vec1d, spacing: const Spacing.step(0.01));
            blackhole(res);
            res.dispose();
          },
          throughput: Throughput.elements(size),
        );

        c.bench(
          'trapz(grid2D, axis=0) [500x500]',
          () {
            final res = trapz(
              grid2d,
              axis: 0,
              spacing: const Spacing.step(0.01),
            );
            blackhole(res);
            res.dispose();
          },
          throughput: Throughput.elements(gridDim * gridDim),
        );
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/calculus_integration',
    ),
  );
}
