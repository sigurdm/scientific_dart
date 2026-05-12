import 'dart:math' as math;
import 'dart:typed_data';
import 'package:ndarray/ndarray.dart';

void main() {
  print(
    '============================================================================',
  );
  // Ensure OpenBLAS threads are configured to 1
  setNumThreads(1);

  print(
    '        ndarray TIMSORT & ARGSORT COMPREHENSIVE BENCHMARK SUITE            ',
  );
  print(
    '============================================================================\n',
  );

  final sizes = [1000, 10000, 50000];
  for (final size in sizes) {
    print(
      '----------------------------------------------------------------------------',
    );
    print(' BENCHMARK METRICS FOR ARRAY SIZE: $size elements');
    print(
      '----------------------------------------------------------------------------',
    );

    // Generate templates
    final random = math.Random(42);
    final randomData = Float64List(size);
    final sortedData = Float64List(size);
    final reverseData = Float64List(size);

    for (var i = 0; i < size; i++) {
      randomData[i] = random.nextDouble() * 1000.0;
      sortedData[i] = i.toDouble();
      reverseData[i] = (size - i).toDouble();
    }

    benchmarkType('Random Array', size, randomData);
    benchmarkType('Already Sorted', size, sortedData);
    benchmarkType('Reverse Sorted', size, reverseData);
    print('');
  }
}

void benchmarkType(String label, int size, Float64List template) {
  final target = NDArray.zeros([size], DType.float64);
  final iterations = size > 10000 ? 100 : 500;

  // --- 1. Direct sort() ---
  // Warmup
  for (var i = 0; i < 10; i++) {
    target.data.setRange(0, size, template.cast<Float64>());
    final res = sort(target);
    res.dispose();
  }

  var stopwatch = Stopwatch()..start();
  for (var i = 0; i < iterations; i++) {
    target.data.setRange(0, size, template.cast<Float64>());
    final res = sort(target);
    res.dispose();
  }
  stopwatch.stop();
  final avgSortUs = stopwatch.elapsedMicroseconds / iterations;

  // --- 2. Indirect argsort() ---
  // Warmup
  for (var i = 0; i < 10; i++) {
    target.data.setRange(0, size, template.cast<Float64>());
    final res = argsort(target);
    res.dispose();
  }

  stopwatch = Stopwatch()..start();
  for (var i = 0; i < iterations; i++) {
    target.data.setRange(0, size, template.cast<Float64>());
    final res = argsort(target);
    res.dispose();
  }
  stopwatch.stop();
  final avgArgsortUs = stopwatch.elapsedMicroseconds / iterations;

  // Print formatted results
  print(
    '${label.padRight(16)} | Direct sort(): ${avgSortUs.toStringAsFixed(2).padLeft(10)} us | Argsort(): ${avgArgsortUs.toStringAsFixed(2).padLeft(10)} us',
  );
  target.dispose();
}
