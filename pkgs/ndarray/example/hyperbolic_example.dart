import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray sinh(), cosh(), and tanh() Hyperbolic Examples ===\n');

  NDArray.scope(() {
    // 1. Hyperbolic tangent activation loop
    final a = NDArray.fromList([-2.0, -1.0, 0.0, 1.0, 2.0], [5], DType.float64);
    print('Input activation values:');
    print(' [ ${a.data.join(", ")} ]');

    final tanhAct = tanh(a);
    print('\nTanh Activation results (tanh):');
    print(' [ ${tanhAct.data.map((e) => e.toStringAsFixed(4)).join(", ")} ]');

    final sinhVal = sinh(a);
    print('\nHyperbolic sine results (sinh):');
    print(' [ ${sinhVal.data.map((e) => e.toStringAsFixed(4)).join(", ")} ]');

    final coshVal = cosh(a);
    print('\nHyperbolic cosine results (cosh):');
    print(' [ ${coshVal.data.map((e) => e.toStringAsFixed(4)).join(", ")} ]');

    // 2. Inverse Hyperbolic sweeps
    final asinhVal = asinh(sinhVal);
    print('\nReconstructed values via inverse (asinh(sinh(x))):');
    print(' [ ${asinhVal.data.map((e) => e.toStringAsFixed(4)).join(", ")} ]');
  });
}
