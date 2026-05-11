import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray linalg.pinv() and linalg.matrix_power() Examples ===\n');

  NDArray.scope(() {
    // 1. Compute Moore-Penrose pseudo-inverse of a singular rectangular 2x3 matrix
    final a = NDArray.fromList(
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
      [2, 3],
      DType.float64,
    );

    print('Original 2x3 Matrix (A):');
    _printMatrix(a);

    final aPlus = pinv(a);
    print('\nMoore-Penrose Pseudo-Inverse (A+):');
    _printMatrix(aPlus);

    // Verify pseudo-inverse property: A * A+ * A = A
    final temp = matmul(a, aPlus);
    final recovery = matmul(temp, a);
    print('\nReconstructed Matrix (A * A+ * A):');
    _printMatrix(recovery);
    temp.dispose();
    recovery.dispose();

    // 2. Compute Matrix Power of a 2x2 transition matrix
    final t = NDArray.fromList([0.8, 0.2, 0.1, 0.9], [2, 2], DType.float64);

    print('\nTransition Matrix (T):');
    _printMatrix(t);

    final t10 = matrix_power(t, 10);
    print('\nTransition Matrix raised to 10th power (T^10):');
    _printMatrix(t10);

    // Raise to negative power (T^-1)
    final tInv = matrix_power(t, -1);
    print('\nTransition Matrix Inverse (T^-1):');
    _printMatrix(tInv);

    t10.dispose();
    tInv.dispose();
  });
}

void _printMatrix(NDArray a) {
  final rows = a.shape[0];
  final cols = a.shape[1];
  for (var r = 0; r < rows; r++) {
    final rowStr = [];
    for (var c = 0; c < cols; c++) {
      rowStr.add(a.data[r * cols + c].toStringAsFixed(4).padLeft(9));
    }
    print(' [ ${rowStr.join(', ')} ]');
  }
}
