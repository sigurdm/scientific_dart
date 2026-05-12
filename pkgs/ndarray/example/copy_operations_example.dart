import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray Deep Copy (copy) ufunc Examples ===\n');

  runContiguousCopyExample();
  runStridedViewCopyExample();
}

void runContiguousCopyExample() {
  print('--- 1. Deep Copying a Contiguous Array ---');
  // Allocate standard 1D array
  final parent = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
  print('Parent array: ${parent.data}');

  // Perform deep copy using top-level copy() ufunc (equivalent to np.copy(a))
  final duplicate = copy(parent);
  print('Copied duplicate: ${duplicate.data}');

  // Let\'s verify memory decoupling (modifying copy does not affect parent!)
  print('\nModifying index 0 of duplicate to 99.0...');
  duplicate.data[0] = Float64(99.0);

  print('Parent array index 0: ${parent.data[0]}');
  print('Duplicate array index 0: ${duplicate.data[0]}');
  print('🏆 Memory blocks are successfully decoupled!');
}

void runStridedViewCopyExample() {
  print('\n--- 2. Deep Copying a Strided Transposed Array View ---');
  // Allocate 2D array [[1, 2], [3, 4]]
  final parent = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
  print('Parent 2D array data:\n$parent');

  // Swapping axes creates a strided, non-contiguous view
  final transposedView = parent.transposed;
  print('Transposed view data:\n$transposedView');
  print('Transposed view isContiguous: ${transposedView.isContiguous}');

  // copy() automatically detects strided view, and duplicates a contiguous equivalent copy!
  final duplicate = copy(transposedView);
  print('\nCopied duplicate flat data: ${duplicate.data}');
  print('Copied duplicate isContiguous: ${duplicate.isContiguous}');

  // Decoupled memory verification
  duplicate.data[0] = Float64(99.0);
  print('\nModifying copy data[0] to 99.0...');
  print('Original parent data[0] (still 1.0): ${parent.data[0]}');
  print('🏆 Strided coordinates recursively deep copied successfully!');
}
