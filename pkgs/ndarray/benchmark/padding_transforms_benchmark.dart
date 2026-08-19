import 'package:criterion/criterion.dart';
import 'package:ndarray/ndarray.dart';

void main() async {
  setNumThreads(1);

  await criterion(
    'NDArray Padding, Rotations, Rolling & Splitting Benchmark Suite',
    (c) {
      const dim = 500;
      final mat = linspace<double>(
        0.0,
        100.0,
        dim * dim,
        dtype: DType.float64,
      ).reshape([dim, dim]);

      c.group('1. Multidimensional Array Padding', () {
        c.bench(
          'pad(constant, pad_width=10) [500x500 -> 520x520]',
          () {
            final res = pad(mat, PadWidth.all(10), mode: PadMode.constant);
            blackhole(res);
            res.dispose();
          },
          throughput: Throughput.elements(520 * 520),
        );

        c.bench(
          'pad(edge, pad_width=10) [500x500 -> 520x520]',
          () {
            final res = pad(mat, PadWidth.all(10), mode: PadMode.edge);
            blackhole(res);
            res.dispose();
          },
          throughput: Throughput.elements(520 * 520),
        );

        c.bench(
          'pad(reflect, pad_width=10) [500x500 -> 520x520]',
          () {
            final res = pad(mat, PadWidth.all(10), mode: PadMode.reflect);
            blackhole(res);
            res.dispose();
          },
          throughput: Throughput.elements(520 * 520),
        );
      });

      c.group('2. Array Flipping & Rolling', () {
        c.bench('roll([20, 20]) [500x500]', () {
          final res = roll(mat, [20, 20], axis: [0, 1]);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(dim * dim));

        c.bench('flip(axis=0) [500x500]', () {
          final res = flip(mat, axis: 0);
          blackhole(res);
        }, throughput: Throughput.elements(dim * dim));

        c.bench('fliplr() [500x500]', () {
          final res = fliplr(mat);
          blackhole(res);
        }, throughput: Throughput.elements(dim * dim));
      });

      c.group('3. Array Splitting & Chunking', () {
        const largeRows = 1000;
        final largeMat = linspace<double>(
          0.0,
          100.0,
          largeRows * dim,
          dtype: DType.float64,
        ).reshape([largeRows, dim]);

        c.bench(
          'split(10 chunks, axis=0) [1000x500]',
          () {
            final parts = split(largeMat, 10, axis: 0);
            for (final p in parts) {
              blackhole(p);
            }
          },
          throughput: Throughput.elements(largeRows * dim),
        );
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/padding_transforms',
    ),
  );
}
