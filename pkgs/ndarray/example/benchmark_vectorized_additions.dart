import 'package:criterion/criterion.dart';
import 'package:ndarray/ndarray.dart';

void main() async {
  const arraySize = 100000;

  await criterion(
    'NDArray Float32 SIMD Additions Fast Path Benchmark',
    (c) {
      final a = linspace<double>(1.0, 100.0, arraySize, dtype: DType.float32);
      final b = linspace<double>(1.0, 100.0, arraySize, dtype: DType.float32);

      final viewA = a.reshape([arraySize ~/ 2, 2]).transposed;
      final viewB = b.reshape([arraySize ~/ 2, 2]).transposed;
      final outStrided = NDArray.create([2, arraySize ~/ 2], DType.float32);

      c.group('Float32 Addition Paths', () {
        c.bench(
          '1. Vectorized SIMD Additions Fast Path (Float32x4List)',
          () {
            final res = add(a, b);
            blackhole(res);
            res.dispose();
          },
          throughput: Throughput.elements(arraySize),
        );

        c.bench(
          '2. Non-Contiguous Strided Additions Fallback (Pure loops)',
          () {
            add(viewA, viewB, out: outStrided);
          },
          throughput: Throughput.elements(arraySize),
        );
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/vectorized_additions',
    ),
  );
}
