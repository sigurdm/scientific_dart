import 'package:ndarray/ndarray.dart';

void main() {
  // Define data points
  final xp = NDArray.fromList([0.0, 1.0, 2.0, 5.0], [4], DType.float64);
  final fp = NDArray.fromList([0.0, 2.0, 3.0, 10.0], [4], DType.float64);

  // Define points to evaluate
  final x = NDArray.fromList([-1.0, 0.5, 1.5, 3.0, 6.0], [5], DType.float64);

  print('xp (data points):');
  print(xp.toList());
  print('fp (data values):');
  print(fp.toList());
  print('x (eval points):');
  print(x.toList());

  // Perform interpolation
  final y = interp(x, xp, fp);
  print('Interpolated values:');
  print(y.toList());

  // Perform interpolation with custom boundary values
  final yCustom = interp(x, xp, fp, left: -99.0, right: 99.0);
  print('Interpolated values (custom boundaries):');
  print(yCustom.toList());

  // Clean up
  xp.dispose();
  fp.dispose();
  x.dispose();
  y.dispose();
  yCustom.dispose();
}
