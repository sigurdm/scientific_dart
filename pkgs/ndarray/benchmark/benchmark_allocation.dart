import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';

void main() {
  const iterations = 1000000;
  const size = 4; // Simulating 4D shape/strides

  print('--- Benchmarking Allocation Overhead ---');
  print('Iterations: $iterations');
  print('Array Size: $size integers');
  print('----------------------------------------');

  // 1. Benchmark Malloc + Free
  final swMalloc = Stopwatch()..start();
  for (var i = 0; i < iterations; i++) {
    final ptr = malloc<ffi.Int>(size);
    // Simulate filling
    ptr[0] = 1;
    ptr[1] = 2;
    ptr[2] = 3;
    ptr[3] = 4;
    malloc.free(ptr);
  }
  swMalloc.stop();
  final mallocTime = swMalloc.elapsedMicroseconds / iterations;
  print('Malloc + Free: ${mallocTime.toStringAsFixed(3)} us per iteration');

  // 2. Benchmark Buffer Reuse
  final buffer = malloc<ffi.Int>(32); // Reusable buffer
  final swBuffer = Stopwatch()..start();
  for (var i = 0; i < iterations; i++) {
    final ptr = buffer;
    // Simulate filling
    ptr[0] = 1;
    ptr[1] = 2;
    ptr[2] = 3;
    ptr[3] = 4;
  }
  swBuffer.stop();
  final bufferTime = swBuffer.elapsedMicroseconds / iterations;
  print('Buffer Reuse : ${bufferTime.toStringAsFixed(3)} us per iteration');

  malloc.free(buffer);

  print('----------------------------------------');
  final speedup = mallocTime / bufferTime;
  print('Speedup: ${speedup.toStringAsFixed(1)}x');
}
