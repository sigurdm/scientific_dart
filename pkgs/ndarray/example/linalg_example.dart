import 'dart:typed_data';
import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray LAPACK Linear Algebra Decompositions Examples ===\n');
  runMatrixInversionExample();
  runCholeskyDecompositionExample();
  runQRDecompositionExample();
  runSVDDecompositionExample();
  runBatchedMatmulStackExample();
}

void runMatrixInversionExample() {
  print('--- High-Speed LAPACK Matrix Inversion (inv) ---');
  final a = NDArray.fromList(Float64List.fromList([4.0, 7.0, 2.0, 6.0]), [
    2,
    2,
  ], DType.float64);
  print('Matrix A:\n[4.0, 7.0]\n[2.0, 6.0]');

  // inv() now executes high-performance LAPACK dgetrf + dgetri directly on the heap!
  final aInv = inv(a);
  print(
    'Inverse Matrix A^-1:\n[${aInv.data.sublist(0, 2)}]\n[${aInv.data.sublist(2, 4)}]',
  );
  // Expected approx: [0.6, -0.7] and [-0.2, 0.4]
}

void runCholeskyDecompositionExample() {
  print('\n--- Cholesky Decomposition (np.linalg.cholesky equivalent) ---');
  // Create a symmetric, positive-definite matrix
  final a = NDArray.fromList(
    Float64List.fromList([
      4.0,
      12.0,
      -16.0,
      12.0,
      37.0,
      -43.0,
      -16.0,
      -43.0,
      98.0,
    ]),
    [3, 3],
    DType.float64,
  );
  print('Symmetric Positive-Definite Matrix A:');
  print('[4.0, 12.0, -16.0]\n[12.0, 37.0, -43.0]\n[-16.0, -43.0, 98.0]');

  // cholesky factorizes into A = L * L^T, returns Lower triangular L matrix
  final res = cholesky(a);
  final l = res['L']!;
  print('Cholesky Lower Triangular Factor L:');
  print(
    '[${l.data.sublist(0, 3)}]\n[${l.data.sublist(3, 6)}]\n[${l.data.sublist(6, 9)}]',
  );
  // Expected: L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
}

void runQRDecompositionExample() {
  print('\n--- QR Decomposition (Orthogonal Factorization) ---');
  final a = NDArray.fromList(
    Float64List.fromList([
      12.0,
      -51.0,
      4.0,
      6.0,
      167.0,
      -68.0,
      -4.0,
      24.0,
      -41.0,
    ]),
    [3, 3],
    DType.float64,
  );
  print('Matrix A (3x3):');
  print('[12.0, -51.0, 4.0]\n[6.0, 167.0, -68.0]\n[-4.0, 24.0, -41.0]');

  // Factorizes A = Q * R. Q is an orthogonal/unitary matrix, R is upper triangular.
  final res = qr(a);
  final q = res['Q']!;
  final r = res['R']!;

  print('Orthogonal Matrix Q:');
  print(
    '[${q.data.sublist(0, 3)}]\n[${q.data.sublist(3, 6)}]\n[${q.data.sublist(6, 9)}]',
  );

  print('Upper Triangular Matrix R:');
  print(
    '[${r.data.sublist(0, 3)}]\n[${r.data.sublist(3, 6)}]\n[${r.data.sublist(6, 9)}]',
  );
}

void runSVDDecompositionExample() {
  print('\n--- Singular Value Decomposition (SVD) ---');
  final a = NDArray.fromList(
    Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    [3, 2],
    DType.float64,
  ); // 3x2 non-square matrix
  print('Non-Square Matrix A (3x2):');
  print('[1.0, 2.0]\n[3.0, 4.0]\n[5.0, 6.0]');

  // Computes A = U * S * Vh, where S is 1D vector of singular values
  final res = svd(a);
  final u = res['U']!;
  final s = res['S']!;
  final vh = res['Vh']!;

  print('Left Singular Vectors U (3x3 matrix):');
  print(
    '[${u.data.sublist(0, 3)}]\n[${u.data.sublist(3, 6)}]\n[${u.data.sublist(6, 9)}]',
  );

  print('Singular Values S (1D vector of length min(m,n)):');
  print('${s.data}'); // length 2

  print('Right Singular Vectors V^T (Vh, 2x2 matrix):');
  print('[${vh.data.sublist(0, 2)}]\n[${vh.data.sublist(2, 4)}]');
}

void runBatchedMatmulStackExample() {
  print('\n--- High-Dimensional (ND) Broadcasted Matmul Stacks ---');
  // A neural net batch of inputs with shape [2, 3] (batch_size=2, input_dim=3)
  final inputs = NDArray.fromList(
    Float64List.fromList([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
    [2, 3],
    DType.float64,
  );

  // A stacked multi-head weight tensor with shape [2, 3, 2] (num_heads=2, input_dim=3, output_dim=2)
  // Head 0: all 0.5s, Head 1: all 2.0s
  final weights = NDArray.fromList(
    Float64List.fromList([
      0.5, 0.5, 0.5, 0.5, 0.5, 0.5, // Head 0
      2.0, 2.0, 2.0, 2.0, 2.0, 2.0, // Head 1
    ]),
    [2, 3, 2],
    DType.float64,
  );

  print('Inputs Tensor Shape: ${inputs.shape}');
  print('Weights Matrix Stack Shape: ${weights.shape} (2 independent heads)');

  // matmul automatically broadcasts inputs shape [2, 3] up to [2, 2, 3] to multiply against weights [2, 3, 2]!
  // This fires 2 parallel BLAS dgemm matrix operations natively, producing output shape [2, 2, 2]!
  final outputs = matmul(inputs, weights);
  print(
    'Broadcasted Output Stack Shape -> expected [2, 2, 2]: ${outputs.shape}',
  );

  print('Output Head 0 (Expected all 1.5s and 3.0s):');
  print('Row 0: [${outputs.data.sublist(0, 2)}]');
  print('Row 1: [${outputs.data.sublist(2, 4)}]');

  print('Output Head 1 (Expected all 6.0s and 12.0s):');
  print('Row 0: [${outputs.data.sublist(4, 6)}]');
  print('Row 1: [${outputs.data.sublist(6, 8)}]');
}
