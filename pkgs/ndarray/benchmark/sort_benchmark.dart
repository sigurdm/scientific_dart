import 'dart:math' as math;
import 'dart:typed_data';
import 'package:criterion/criterion.dart';
import 'package:ndarray/ndarray.dart';

void main() async {
  setNumThreads(1);

  final sizes = [1000, 10000, 50000];

  await criterion(
    'NDArray Timsort & Argsort Comprehensive Benchmark Suite',
    (c) {
      void registerTrack(
        String label,
        Float64List Function(int size) templateGen,
      ) {
        c.group(label, () {
          for (final size in sizes) {
            final template = templateGen(size);

            c.bench<NDArray<double>>(
              'Direct sort() [$size]',
              (arr) {
                final res = sort(arr);
                blackhole(res);
                res.dispose();
                arr.dispose();
              },
              setup: () =>
                  NDArray<double>.fromList(template, [size], DType.float64),
              batchSize: BatchSize.largeInput,
              throughput: Throughput.elements(size),
            );

            c.bench<NDArray<double>>(
              'Indirect argsort() [$size]',
              (arr) {
                final res = argsort(arr);
                blackhole(res);
                res.dispose();
                arr.dispose();
              },
              setup: () =>
                  NDArray<double>.fromList(template, [size], DType.float64),
              batchSize: BatchSize.largeInput,
              throughput: Throughput.elements(size),
            );
          }
        });
      }

      registerTrack('Random Array', (size) {
        final rand = math.Random(42);
        return Float64List.fromList(
          List.generate(size, (_) => rand.nextDouble() * 1000.0),
        );
      });

      registerTrack('Already Sorted', (size) {
        return Float64List.fromList(List.generate(size, (i) => i.toDouble()));
      });

      registerTrack('Reverse Sorted', (size) {
        return Float64List.fromList(
          List.generate(size, (i) => (size - i).toDouble()),
        );
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/sort',
    ),
  );
}
