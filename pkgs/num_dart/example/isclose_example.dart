import 'dart:typed_data';
import 'package:num_dart/num_dart.dart';

void main() {
  print('=== NDArray isclose() and allclose() Tolerance Comparisons ===\n');

  // 1. Main floating-point tolerance verification
  final a = NDArray.fromList(Float64List.fromList([1.0, 1.00001, 2.0]), [
    3,
  ], DType.float64);
  final b = NDArray.fromList(Float64List.fromList([1.0, 1.00002, 2.0]), [
    3,
  ], DType.float64);

  print('a: ${a.toList()}');
  print('b: ${b.toList()}');

  // Default tolerances: rtol = 1e-05, atol = 1e-08
  final closeDefault = isclose(a, b);
  print('\nisclose (default): ${closeDefault.toList()}'); // [true, false, true]

  // Stretch tolerances: rtol = 1e-04
  final closeStretched = isclose(a, b, rtol: 1e-04);
  print(
    'isclose (rtol = 1e-04): ${closeStretched.toList()}',
  ); // [true, true, true]

  // 2. allclose check
  final allCloseDefault = allclose(a, b);
  print('\nallclose (default): $allCloseDefault'); // false

  final allCloseStretched = allclose(a, b, rtol: 1e-04);
  print('allclose (rtol = 1e-04): $allCloseStretched'); // true

  // Cleanup unmanaged heap memory
  a.dispose();
  b.dispose();
  closeDefault.dispose();
  closeStretched.dispose();
}
