import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray Element-Wise Comparisons & Mask Recycling Examples ===\n');

  runBasicComparisonsExample();
  runMaskRecyclingExample();
}

void runBasicComparisonsExample() {
  print('--- 1. Broadcasted Comparisons ---');
  final a = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
  final b = NDArray.fromList([10.0, 99.0, 30.0], [3], DType.float64);

  print('Array A: ${a.data}');
  print('Array B: ${b.data}');

  // equal
  final eq = equal(a, b);
  print('equal(A, B): ${eq.toList()}');

  // greater
  final gt = greater(b, a);
  print('greater(B, A): ${gt.toList()}');

  // less_equal
  final lte = less_equal(a, b);
  print('less_equal(A, B): ${lte.toList()}\n');
}

void runMaskRecyclingExample() {
  print('--- 2. Allocation-Free Mask Recycling inside Loops ---');
  final dataset = NDArray.fromList(
    [0.5, 1.2, -0.8, 2.5, 0.1],
    [5],
    DType.float64,
  );
  final threshold = NDArray.fromList(
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [5],
    DType.float64,
  );

  print('Dataset: ${dataset.data}');
  print('Threshold bounds: ${threshold.data}');

  // Pre-allocate a boolean mask result buffer once!
  final recycledMask = NDArray<bool>.create([5], DType.boolean);

  print('\nIteratively comparing dataset against thresholds bounds...');
  const iterations = 5;
  for (var step = 1; step <= iterations; step++) {
    // Write the comparison result directly in-place into recycledMask!
    // No intermediate memory is allocated!
    greater(dataset, threshold, out: recycledMask);

    print('Step $step -> recycledMask data: ${recycledMask.toList()}');
  }

  print(
    '\n🏆 Mask buffer recycled successfully in-place with 0 memory allocations!',
  );
}
