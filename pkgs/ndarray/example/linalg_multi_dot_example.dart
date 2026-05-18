import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray linalg.multi_dot() Chain Optimizer Example ===\n');

  NDArray.scope(() {
    // Create 3 matrices with shapes: [2, 10] * [10, 5] * [5, 3]
    // The product will have shape [2, 3].
    // multi_dot dynamically calculates that (A * B) * C is mathematically much faster
    // to evaluate than A * (B * C) and evaluates it in the optimal order automatically.
    final a = NDArray.ones([2, 10], DType.float64);
    final b = NDArray.ones([10, 5], DType.float64);
    final c = NDArray.ones([5, 3], DType.float64);

    print('Matrix A shape: ${a.shape}');
    print('Matrix B shape: ${b.shape}');
    print('Matrix C shape: ${c.shape}');

    final res = multi_dot([a, b, c]);

    print('\nResult of multi_dot(A, B, C) shape: ${res.shape}');
    print('Result elements:');
    _printMatrix(res);
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
