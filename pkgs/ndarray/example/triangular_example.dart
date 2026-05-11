import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray tril() and triu() Triangular Extractions Examples ===\n');

  NDArray.scope(() {
    // 1. Create a 3x3 matrix
    final a = NDArray.fromList(
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
      [3, 3],
      DType.float64,
    );

    print('Original 3x3 Matrix:');
    _printMatrix(a);

    // 2. tril: Lower triangular extraction (k = 0)
    final lower = tril(a);
    print('\nLower Triangular (tril, k=0):');
    _printMatrix(lower);

    // 3. triu: Upper triangular extraction (k = 0)
    final upper = triu(a);
    print('\nUpper Triangular (triu, k=0):');
    _printMatrix(upper);

    // 4. Diagonal offsets (k = 1 and k = -1)
    final lowerK1 = tril(a, k: 1);
    print('\nLower Triangular with positive offset (tril, k=1):');
    _printMatrix(lowerK1);

    final upperKM1 = triu(a, k: -1);
    print('\nUpper Triangular with negative offset (triu, k=-1):');
    _printMatrix(upperKM1);

    // 5. Memory-efficient Recycling Buffer Reuse
    final recycler = NDArray<double>.zeros([3, 3], DType.float64);
    final recycledLower = tril(a, k: 0, out: recycler);
    print(
      '\nRecycled Output Buffer (identical check): ${identical(recycledLower, recycler) ? "PASS" : "FAIL"}',
    );
    _printMatrix(recycledLower);
  });
}

void _printMatrix(NDArray a) {
  final rows = a.shape[0];
  final cols = a.shape[1];
  for (var r = 0; r < rows; r++) {
    final rowStr = [];
    for (var c = 0; c < cols; c++) {
      rowStr.add(a.data[r * cols + c].toStringAsFixed(1).padLeft(5));
    }
    print(' [ ${rowStr.join(', ')} ]');
  }
}
