import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray Non-Zero coordinates Search Benchmark ===\n');

  runNonzeroBenchmark();
}

void runNonzeroBenchmark() {
  const size = 80; // 80 x 80 matrix
  const iterations = 300;

  print(
    'Allocating dense Double matrix of shape [$size, $size] with sparse non-zeros...',
  );
  final a = NDArray.zeros([size, size], DType.float64);
  // Populate some indices with non-zero entries
  for (var i = 0; i < size; i += 5) {
    for (var j = 0; j < size; j += 3) {
      a.setCell([i, j], 9.9);
    }
  }

  print('Warming up VM compiler...');
  for (var i = 0; i < 10; i++) {
    final temp = nonzero(a);
    for (final list in temp) {
      list.dispose();
    }
  }

  print('\n--- 1. Optimized running Flat offset nonzero() ---');
  final stopwatchOptimized = Stopwatch()..start();
  for (var i = 0; i < iterations; i++) {
    final res = nonzero(a);
    for (final list in res) {
      list.dispose();
    }
  }
  stopwatchOptimized.stop();
  final timeOptimized = stopwatchOptimized.elapsedMicroseconds / 1000.0;
  print(
    'Optimized nonzero(): ${timeOptimized.toStringAsFixed(2)} ms (average ${(timeOptimized / iterations).toStringAsFixed(3)} ms/iter)',
  );

  print('\n--- 2. Slow Bracket Selector Sweep fallback ---');
  final stopwatchSlow = Stopwatch()..start();
  for (var step = 0; step < iterations; step++) {
    final coordinateLists = List.generate(2, (_) => <int>[]);
    for (var i = 0; i < size; i++) {
      for (var j = 0; j < size; j++) {
        // Invokes slow bracket checks and RangeErrors cell lookup sweeps recursively
        final val = a[[i, j]];
        if (val != 0.0) {
          coordinateLists[0].add(i);
          coordinateLists[1].add(j);
        }
      }
    }
    final res = coordinateLists.map((list) {
      return NDArray<int>.fromList(list, [list.length], DType.int32);
    }).toList();
    for (final list in res) {
      list.dispose();
    }
  }
  stopwatchSlow.stop();
  final timeSlow = stopwatchSlow.elapsedMicroseconds / 1000.0;
  print(
    'Slow bracket sweep: ${timeSlow.toStringAsFixed(2)} ms (average ${(timeSlow / iterations).toStringAsFixed(3)} ms/iter)',
  );

  final speedup = timeSlow / timeOptimized;
  print(
    '\n🏆 Optimized nonzero() speedup over bracket loops: ${speedup.toStringAsFixed(2)}x faster!',
  );
}
