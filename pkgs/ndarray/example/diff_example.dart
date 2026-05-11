import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray diff() Discrete Difference Examples ===\n');

  NDArray.scope(() {
    // 1. Flat sequence difference
    final a = NDArray.fromList([1, 2, 4, 7, 0], [5], DType.int64);
    print('Original flat array:');
    print(' [ ${a.data.join(", ")} ]');

    final d1 = diff(a);
    print('\n1st Difference (diff, n=1):');
    print(' [ ${d1.data.join(", ")} ]');

    final d2 = diff(a, n: 2);
    print('\n2nd Difference (diff, n=2):');
    print(' [ ${d2.data.join(", ")} ]');

    // 2. 2D Matrix difference along specified axes
    final mat = NDArray.fromList(
      [1.0, 3.0, 9.0, 2.0, 5.0, 15.0],
      [2, 3],
      DType.float64,
    );

    print('\nOriginal 2D Matrix (2x3):');
    _printMatrix(mat);

    final dAxis0 = diff(mat, axis: 0);
    print('\nDifference along columns (axis=0):');
    _printMatrix(dAxis0);

    final dAxis1 = diff(mat, axis: 1);
    print('\nDifference along rows (axis=1):');
    _printMatrix(dAxis1);

    // 3. Complex number difference
    final aComp = NDArray.fromList(
      [Complex(1.0, 2.0), Complex(3.0, 5.0), Complex(6.0, 10.0)],
      [3],
      DType.complex128,
    );
    print('\nOriginal Complex array:');
    print(' [ ${aComp.data.join(", ")} ]');

    final dComp = diff(aComp);
    print('\nComplex array difference:');
    print(' [ ${dComp.data.join(", ")} ]');
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
    print(' [ ${rowStr.join(", ")} ]');
  }
}
