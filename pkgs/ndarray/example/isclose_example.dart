import 'dart:typed_data';
import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray isClose() and allClose() Tolerance Comparisons ===\n');

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
  final closeDefault = isClose(a, b);
  print('\nisClose (default): ${closeDefault.toList()}'); // [true, false, true]

  // Stretch tolerances: rtol = 1e-04
  final closeStretched = isClose(a, b, rtol: 1e-04);
  print(
    'isClose (rtol = 1e-04): ${closeStretched.toList()}',
  ); // [true, true, true]

  // 2. allClose check
  final allCloseDefault = allClose(a, b);
  print('\nallClose (default): $allCloseDefault'); // false

  final allCloseStretched = allClose(a, b, rtol: 1e-04);
  print('allClose (rtol = 1e-04): $allCloseStretched'); // true

  // Cleanup unmanaged heap memory
  a.dispose();
  b.dispose();
  closeDefault.dispose();
  closeStretched.dispose();
}
