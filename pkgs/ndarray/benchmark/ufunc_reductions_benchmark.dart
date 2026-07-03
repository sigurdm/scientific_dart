import 'dart:ffi';
import 'package:ndarray/ndarray.dart';

void main() {
  const int size1d = 1000000;
  const int dim2d = 1000;
  const int iterations = 100;
  const int warmup = 10;

  print('================================================================');
  print('NDArray (Dart) Ufunc Reductions & Masked Functions Benchmark');
  print('================================================================');
  print('Array Size 1D: $size1d elements');
  print('Array Size 2D: ${dim2d}x$dim2d elements');
  print('Iterations: $iterations (after $warmup warmup runs)\n');

  // Prepare test arrays
  final a1d = NDArray.fromList(List.generate(size1d, (i) => i % 100 * 1.0), [
    size1d,
  ], DType.float64);
  final b1d = NDArray.fromList(
    List.generate(size1d, (i) => (i % 50 + 1) * 1.0),
    [size1d],
    DType.float64,
  );
  final mask1d = NDArray.fromList(
    List.generate(size1d, (i) => i % 2 == 0 ? 1 : 0),
    [size1d],
    DType.uint8,
  );

  final a2d = NDArray.fromList(
    List.generate(dim2d * dim2d, (i) => (i % 100) * 1.0),
    [dim2d, dim2d],
    DType.float64,
  );
  final b2d = NDArray.fromList(
    List.generate(dim2d * dim2d, (i) => (i % 50 + 1) * 1.0),
    [dim2d, dim2d],
    DType.float64,
  );

  final aOuter = NDArray.fromList(List.generate(1000, (i) => i * 1.0), [
    1000,
  ], DType.float64);
  final bOuter = NDArray.fromList(List.generate(1000, (i) => (i + 1) * 1.0), [
    1000,
  ], DType.float64);

  final indicesAt = NDArray.fromList(
    List.generate(10000, (i) => (i * 97) % size1d),
    [10000],
    DType.int64,
  );
  final valsAt = NDArray.fromList(List.generate(10000, (i) => i * 1.0), [
    10000,
  ], DType.float64);

  final indicesReduceat = NDArray.fromList(
    List.generate(1000, (i) => i * 1000),
    [1000],
    DType.int64,
  );

  final out1d = NDArray.zeros([size1d], DType.float64);
  final out2d = NDArray.zeros([dim2d, dim2d], DType.float64);

  double bench(String name, void Function() fn) {
    // Warmup
    for (int i = 0; i < warmup; i++) {
      fn();
    }
    final sw = Stopwatch()..start();
    for (int i = 0; i < iterations; i++) {
      fn();
    }
    sw.stop();
    final avgUs = sw.elapsedMicroseconds / iterations;
    final avgMs = avgUs / 1000.0;
    print(
      '${name.padRight(42)}: ${avgUs.toStringAsFixed(2).padLeft(10)} us (${avgMs.toStringAsFixed(3)} ms)',
    );
    return avgUs;
  }

  // 1. Reductions
  bench('reduce(add) [1D 1M global]', () {
    a1d.reduce(op: BinaryOp.add);
  });

  bench('reduce(add) [2D 1000x1000 axis:0]', () {
    a2d.reduce(op: BinaryOp.add, axis: 0);
  });

  bench('reduce(multiply) [1D 100K global]', () {
    a1d.slice([Slice(stop: 100000)]).reduce(op: BinaryOp.multiply);
  });

  // 2. Accumulate
  bench('accumulate(add) [1D 1M cumsum]', () {
    a1d.accumulate(op: BinaryOp.add);
  });

  bench('accumulate(add) [2D 1000x1000 axis:0]', () {
    a2d.accumulate(op: BinaryOp.add, axis: 0);
  });

  // 3. Outer
  bench('outer(add) [1000 x 1000]', () {
    aOuter.outer(bOuter, op: BinaryOp.add);
  });

  bench('outer(multiply) [1000 x 1000]', () {
    aOuter.outer(bOuter, op: BinaryOp.multiply);
  });

  // 4. Reduceat
  bench('reduceat(add) [1M array, 1000 segments]', () {
    a1d.reduceat(indicesReduceat, op: BinaryOp.add);
  });

  // 5. At (unbuffered scatter update)
  bench('at(add) [1M array, 10K indices]', () {
    a1d.at(indicesAt, valsAt, op: BinaryOp.add);
  });

  // 6. Masked Functions (where=)
  bench('add(where=mask) [1D 1M contiguous]', () {
    add(a1d, b1d, where: mask1d, out: out1d);
  });

  bench('multiply(where=mask) [1D 1M contiguous]', () {
    multiply(a1d, b1d, where: mask1d, out: out1d);
  });

  final aStrided = a2d.flatten();
  final bStrided = b2d.flatten();
  bench('add(where=mask) [1D 1M strided view]', () {
    add(aStrided.slice([Slice(step: 2)]), bStrided.slice([Slice(step: 2)]));
  });

  // Dispose test arrays
  a1d.dispose();
  b1d.dispose();
  mask1d.dispose();
  a2d.dispose();
  b2d.dispose();
  aOuter.dispose();
  bOuter.dispose();
  indicesAt.dispose();
  valsAt.dispose();
  indicesReduceat.dispose();
  out1d.dispose();
  out2d.dispose();
  aStrided.dispose();
  bStrided.dispose();
}
