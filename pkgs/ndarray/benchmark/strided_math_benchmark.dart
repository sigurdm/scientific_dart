import 'package:criterion/criterion.dart';
import 'package:ndarray/ndarray.dart';

void main() async {
  await criterion(
    'NDArray Non-Contiguous Strided Math Benchmark',
    (c) {
      final mat = NDArray<double>.zeros([1500, 1500], DType.float64);
      for (var i = 0; i < mat.data.length; i++) {
        mat.data[i] = i.toDouble() / 100000.0;
      }
      final matT = mat.transpose();
      final out = NDArray<double>.create([1500, 1500], DType.float64);

      c.bench(
        'strided tan(matT) [shape=1500x1500 transposed]',
        () {
          tan(matT, out: out);
        },
        throughput: Throughput.elements(1500 * 1500),
      );

      c.bench(
        'strided exp(matT) [shape=1500x1500 transposed]',
        () {
          exp(matT, out: out);
        },
        throughput: Throughput.elements(1500 * 1500),
      );
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/strided_math',
    ),
  );
}
