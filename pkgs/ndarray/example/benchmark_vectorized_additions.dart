import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray Float32 SIMD additions Fast Path Benchmark ===\n');

  runAdditionsBenchmark();
}

void runAdditionsBenchmark() {
  const arraySize = 100000;
  const iterations = 200;

  print('Generating Float32 arrays of size $arraySize...');
  final a = linspace<double>(1.0, 100.0, arraySize, dtype: DType.float32);
  final b = linspace<double>(1.0, 100.0, arraySize, dtype: DType.float32);

  print('Warming up VM compiler...');
  for (var i = 0; i < 10; i++) {
    final temp = add(a, b);
    temp.dispose();
  }

  print('\n--- 1. Vectorized SIMD additions Fast Path (Float32x4List) ---');
  final stopwatchSIMD = Stopwatch()..start();
  for (var i = 0; i < iterations; i++) {
    final res = add(a, b);
    res.dispose();
  }
  stopwatchSIMD.stop();
  final timeSIMD = stopwatchSIMD.elapsedMicroseconds / 1000.0;
  print(
    'SIMD additions: ${timeSIMD.toStringAsFixed(2)} ms (average ${(timeSIMD / iterations).toStringAsFixed(3)} ms/iter)',
  );

  print('\n--- 2. Non-Contiguous Strided additions Fallback (Pure loops) ---');
  // Swap axes slightly to produce non-contiguous views, bypassing the SIMD path:
  final viewA = a.reshape([arraySize ~/ 2, 2]).transposed;
  final viewB = b.reshape([arraySize ~/ 2, 2]).transposed;

  final stopwatchPure = Stopwatch()..start();
  for (var i = 0; i < iterations; i++) {
    final res = add(viewA, viewB);
    res.dispose();
  }
  stopwatchPure.stop();
  final timePure = stopwatchPure.elapsedMicroseconds / 1000.0;
  print(
    'Pure loops additions: ${timePure.toStringAsFixed(2)} ms (average ${(timePure / iterations).toStringAsFixed(3)} ms/iter)',
  );

  final speedup = timePure / timeSIMD;
  print(
    '\n🏆 SIMD additions speedup over pure loops: ${speedup.toStringAsFixed(2)}x faster!',
  );
}
