import 'package:num_dart/num_dart.dart';

void main() {
  print('=== NDArray Non-Zero Elements coordinates Extraction ===\n');

  runNonzeroExtractionExample();
}

void runNonzeroExtractionExample() {
  // Allocate a 3x3 grid representing sparse spatial labels/features
  // [[0, 5, 0],
  //  [2, 0, 0],
  //  [0, 0, 9]]
  print('Allocating 3x3 feature matrix:');
  final grid = NDArray.fromList(
    [0.0, 5.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 9.0],
    [3, 3],
    DType.float64,
  );

  print('Grid:');
  print(grid);

  // nonzero() extracts coordinate indices along each axis:
  // For a 2D array, it returns a list of two 1D NDArrays:
  // indexList[0]: row indices
  // indexList[1]: column indices
  final indices = nonzero(grid);

  final rows = indices[0];
  final cols = indices[1];

  print('\nExtracted Rows Indices: ${rows.toList()}');
  print('Extracted Columns Indices: ${cols.toList()}');

  print('\nMapping coordinates to non-zero element values:');
  final count = rows.shape[0];
  for (var i = 0; i < count; i++) {
    final r = rows.data[i];
    final c = cols.data[i];

    // getCell retrieves the element in-place at the coordinates
    final val = grid.getCell([r, c]);
    print('🏆 Non-Zero Element found at coordinate [$r, $c] -> value: $val');
  }

  // Dispose FFI memory blocks cleanly!
  grid.dispose();
  for (final list in indices) {
    list.dispose();
  }
}
