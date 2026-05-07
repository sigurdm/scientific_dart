import 'package:ndarray/ndarray.dart';
import 'dart:ffi' as ffi;
import 'package:openblas/openblas.dart';

void main() {
  print('--- num_dart Benchmark ---');

  final sizes = [10, 100, 1000, 10000, 100000, 1000000];

  print('\n--- Float64 Addition (FFI vs Dart) ---');
  print('| Size | Dart (us) | FFI (us) | Winner |');
  print('|------|-----------|----------|--------|');

  for (final size in sizes) {
    final iterations = (1000000 / size).clamp(100, 100000).toInt();
    final warmup = (iterations / 10).clamp(10, 1000).toInt();

    final a = NDArray<double>.ones([size], DType.float64);
    final b = NDArray<double>.ones([size], DType.float64);

    // 1. Benchmark Dart addition
    ffiThresholds[Operation.add] = size + 1;
    for (var i = 0; i < warmup; i++) {
      final r = add(a, b);
      r.dispose();
    }
    final stopwatchDart = Stopwatch()..start();
    for (var i = 0; i < iterations; i++) {
      final r = add(a, b);
      r.dispose();
    }
    stopwatchDart.stop();
    final dartTime = stopwatchDart.elapsedMicroseconds / iterations;

    // 2. Benchmark FFI addition (calling cblas_daxpy directly)
    // Warmup FFI
    for (var i = 0; i < warmup; i++) {
      cblas_daxpy(
        size,
        1.0,
        b.pointer.cast<ffi.Double>(),
        1,
        a.pointer.cast<ffi.Double>(),
        1,
      );
    }

    final stopwatchFFI = Stopwatch()..start();
    for (var i = 0; i < iterations; i++) {
      // To be fair with non-destructive Dart addition, we include a copy step!
      final resultFFI = NDArray<double>.create([size], DType.float64);
      resultFFI.data.setRange(0, size, a.data);

      cblas_daxpy(
        size,
        1.0,
        b.pointer.cast<ffi.Double>(),
        1,
        resultFFI.pointer.cast<ffi.Double>(),
        1,
      );
      resultFFI.dispose();
    }
    stopwatchFFI.stop();
    final ffiTime = stopwatchFFI.elapsedMicroseconds / iterations;

    final winner = dartTime < ffiTime ? 'Dart' : 'FFI';
    print(
      '| $size | ${dartTime.toStringAsFixed(2)} | ${ffiTime.toStringAsFixed(2)} | $winner |',
    );

    a.dispose();
    b.dispose();
  }

  print('\n--- Float32 Addition (Scalar vs SIMD vs FFI) ---');
  print('| Size | Scalar (us) | SIMD (us) | FFI (us) | Winner |');
  print('|------|-------------|-----------|----------|--------|');

  for (final size in sizes) {
    if (size < 10) continue; // Skip very small sizes for simplicity

    final iterations = (1000000 / size).clamp(100, 100000).toInt();
    final warmup = (iterations / 10).clamp(10, 1000).toInt();

    final a = NDArray<double>.ones([size], DType.float32);
    final b = NDArray<double>.ones([size], DType.float32);

    final sizeHalf = size ~/ 2;
    final aNonContig = NDArray<double>.view(a, shape: [sizeHalf], strides: [2]);
    final bNonContig = NDArray<double>.view(b, shape: [sizeHalf], strides: [2]);

    // 1. Warmup & Benchmark scalar
    for (var i = 0; i < warmup; i++) {
      final r = add(aNonContig, bNonContig);
      r.dispose();
    }
    final stopwatchScalar = Stopwatch()..start();
    for (var i = 0; i < iterations; i++) {
      final r = add(aNonContig, bNonContig);
      r.dispose();
    }
    stopwatchScalar.stop();
    final scalarTime = stopwatchScalar.elapsedMicroseconds / iterations;

    // 2. SIMD path (contiguous arrays of the SAME length as non-contig)
    final aSIMD = NDArray<double>.ones([sizeHalf], DType.float32);
    final bSIMD = NDArray<double>.ones([sizeHalf], DType.float32);

    for (var i = 0; i < warmup; i++) {
      final r = add(aSIMD, bSIMD);
      r.dispose();
    }
    final stopwatchSIMD = Stopwatch()..start();
    for (var i = 0; i < iterations; i++) {
      final r = add(aSIMD, bSIMD);
      r.dispose();
    }
    stopwatchSIMD.stop();
    final simdTime = stopwatchSIMD.elapsedMicroseconds / iterations;

    // 3. FFI path (contiguous arrays of the SAME length as non-contig)
    final aFFI = NDArray<double>.ones([sizeHalf], DType.float32);
    final bFFI = NDArray<double>.ones([sizeHalf], DType.float32);

    // Warmup FFI
    for (var i = 0; i < warmup; i++) {
      cblas_saxpy(
        sizeHalf,
        1.0,
        bFFI.pointer.cast<ffi.Float>(),
        1,
        aFFI.pointer.cast<ffi.Float>(),
        1,
      );
    }

    final stopwatchFFI = Stopwatch()..start();
    for (var i = 0; i < iterations; i++) {
      final resultFFI = NDArray<double>.create([sizeHalf], DType.float32);
      resultFFI.data.setRange(0, sizeHalf, aFFI.data);

      cblas_saxpy(
        sizeHalf,
        1.0,
        bFFI.pointer.cast<ffi.Float>(),
        1,
        resultFFI.pointer.cast<ffi.Float>(),
        1,
      );
      resultFFI.dispose();
    }
    stopwatchFFI.stop();
    final ffiTime = stopwatchFFI.elapsedMicroseconds / iterations;

    String winner = 'Scalar';
    if (simdTime < scalarTime && simdTime < ffiTime) winner = 'SIMD';
    if (ffiTime < scalarTime && ffiTime < simdTime) winner = 'FFI';

    print(
      '| $sizeHalf | ${scalarTime.toStringAsFixed(2)} | ${simdTime.toStringAsFixed(2)} | ${ffiTime.toStringAsFixed(2)} | $winner |',
    );

    a.dispose();
    b.dispose();
    aSIMD.dispose();
    bSIMD.dispose();
    aFFI.dispose();
    bFFI.dispose();
  }
}
