import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray Non-Contiguous sin() Performance Benchmark ===\n');

  NDArray.scope(() {
    // Create a large 2000x2000 Float64 matrix
    final mat = NDArray<double>.zeros([2000, 2000], DType.float64);
    for (var i = 0; i < mat.data.length; i++) {
      mat.data[i] = i.toDouble() / 10000.0;
    }

    // Transpose it to make it non-contiguous
    final matT = mat.transpose();
    print('Matrix contiguous check: ${mat.isContiguous} (shape: ${mat.shape})');
    print(
      'Transposed view contiguous check: ${matT.isContiguous} (shape: ${matT.shape})',
    );

    // Warm up
    final warm = sin(matT);
    warm.dispose();

    // Benchmark loop
    final stopwatch = Stopwatch()..start();
    const iterations = 10;
    for (var i = 0; i < iterations; i++) {
      final res = sin(matT);
      res.dispose();
    }
    stopwatch.stop();

    final elapsedMs = stopwatch.elapsedMilliseconds;
    final avgMs = elapsedMs / iterations;
    print(
      '\nAverage execution time for sin() on non-contiguous transposed matrix:',
    );
    print(' => ${avgMs.toStringAsFixed(2)} ms per iteration');
  });
}
