import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray linalg.lstsq() Least-Squares Example ===\n');

  NDArray.scope(() {
    // We want to fit a straight line y = c + m * x to the following data points:
    // (1, 2), (2, 3.9), (3, 6.1)
    //
    // The system of equations is A * [c, m]^T = B:
    // [1, 1] * [c, m]^T = 2
    // [1, 2] * [c, m]^T = 3.9
    // [1, 3] * [c, m]^T = 6.1
    final a = NDArray.fromList(
      [1.0, 1.0, 1.0, 2.0, 1.0, 3.0],
      [3, 2],
      DType.float64,
    );
    final b = NDArray.fromList([2.0, 3.9, 6.1], [3], DType.float64);

    print('Matrix (A):');
    _printMatrix(a);

    print('\nRight-Hand Side (b):');
    print('  ${b.toList()}');

    final res = lstsq<double>(a, b);

    print('\nLeast-Squares Solution (x):');
    print('  Intercept (c): ${res.x.data[0].toStringAsFixed(4)}');
    print('  Slope (m):     ${res.x.data[1].toStringAsFixed(4)}');

    print('\nSums of Squared Residuals:');
    print('  ${res.residuals.toList()}');

    print('\nEffective Rank of A:');
    print('  ${res.rank}');

    print('\nSingular Values of A:');
    print('  ${res.s.toList()}');
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
