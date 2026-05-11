import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray Non-Contiguous Strided Math performance Benchmark ===\n');

  NDArray.scope(() {
    // Create a large 1500x1500 double matrix
    final mat = NDArray<double>.zeros([1500, 1500], DType.float64);
    for (var i = 0; i < mat.data.length; i++) {
      mat.data[i] = i.toDouble() / 100000.0;
    }

    // Transpose it to make it highly strided non-contiguous
    final matT = mat.transpose();

    // 1. tan() benchmark
    final stopwatchTan = Stopwatch()..start();
    const iterations = 10;
    for (var i = 0; i < iterations; i++) {
      final res = tan(matT);
      res.dispose();
    }
    stopwatchTan.stop();
    print(
      'Average execution time for strided tan(): ${(stopwatchTan.elapsedMilliseconds / iterations).toStringAsFixed(2)} ms',
    );

    // 2. exp() benchmark
    final stopwatchExp = Stopwatch()..start();
    for (var i = 0; i < iterations; i++) {
      final res = exp(matT);
      res.dispose();
    }
    stopwatchExp.stop();
    print(
      'Average execution time for strided exp(): ${(stopwatchExp.elapsedMilliseconds / iterations).toStringAsFixed(2)} ms',
    );
  });
}
