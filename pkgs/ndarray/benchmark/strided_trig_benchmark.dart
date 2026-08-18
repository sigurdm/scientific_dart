import 'package:criterion/criterion.dart';
import 'package:ndarray/ndarray.dart';

void main() async {
  await criterion(
    'NDArray Non-Contiguous sin() Performance Benchmark',
    (c) {
      final mat = NDArray<double>.zeros([2000, 2000], DType.float64);
      for (var i = 0; i < mat.data.length; i++) {
        mat.data[i] = i.toDouble() / 10000.0;
      }
      final matT = mat.transpose();
      final out = NDArray<double>.create([2000, 2000], DType.float64);

      c.bench(
        'strided sin(matT) [shape=2000x2000 transposed]',
        () {
          sin(matT, out: out);
        },
        throughput: Throughput.elements(2000 * 2000),
      );
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/strided_trig',
    ),
  );
}
