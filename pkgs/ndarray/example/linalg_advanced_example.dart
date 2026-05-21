import 'package:ndarray/ndarray.dart';

void main() {
  // 1. Kronecker Product
  final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
  final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);
  final k = kron(a, b);
  print('Kronecker Product: ${k.toList()}'); // [3.0, 4.0, 6.0, 8.0]
  k.dispose();

  // 2. Vector Outer Product
  final u = NDArray.fromList([1, 2], [2], DType.int32);
  final v = NDArray.fromList([3, 4], [2], DType.int32);
  final o = outer(u, v);
  print('Outer Product Matrix:\n${o.toList()}'); // [3, 4, 6, 8]
  o.dispose();

  // 3. Vector Cross Product (3D)
  final v1 = NDArray.fromList([1.0, 0.0, 0.0], [3], DType.float64);
  final v2 = NDArray.fromList([0.0, 1.0, 0.0], [3], DType.float64);
  final c = cross(v1, v2);
  print('Cross Product Vector: ${c.toList()}'); // [0.0, 0.0, 1.0]
  c.dispose();

  // 4. Vector and Matrix Norms
  final x = NDArray.fromList([1.0, -2.0, 3.0, -4.0], [4], DType.float64);
  final l1 = norm(x, ord: 1);
  print('Vector L1 Norm: ${l1.data[0]}'); // 10.0
  l1.dispose();

  final l2 = norm(x, ord: 2);
  print('Vector L2 Norm: ${l2.data[0]}'); // sqrt(30)
  l2.dispose();

  final m = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
  final fro = norm(m, ord: 'fro');
  print('Matrix Frobenius Norm: ${fro.data[0]}'); // sqrt(30)
  fro.dispose();
}
