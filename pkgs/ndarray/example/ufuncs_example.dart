import 'dart:typed_data';
import 'package:ndarray/ndarray.dart';

void main() {
  print(
    '=== NDArray Universal Functions (ufuncs) & Broadcasting Examples ===\n',
  );
  runMixedTypeArithmeticExample();
  runTrigAndRoundingExample();
  runComplexAbsoluteValueExample();
  runComparisonBroadcastingExample();
  runLogicalOperationsExample();
  runLogicalReductionsExample();
  runAngleConvertersExample();
  runEnumerateAndComplexComponentsExample();
}

void runMixedTypeArithmeticExample() {
  print('--- Mixed Real & Complex Arithmetic Upcasting ---');
  // Create a 1D complex array
  final a = NDArray<Complex>.create([2], DType.complex128);
  a.data[0] = Complex(1.0, 2.0);
  a.data[1] = Complex(3.0, 4.0);

  // Create a 1D real array
  final b = NDArray.fromList(Float64List.fromList([10.0, 20.0]), [
    2,
  ], DType.float64);

  // Add them together. The real array automatically upcasts to interact with the complex array!
  final c = add(a, b);
  print('a: ${a.data}');
  print('b: ${b.data}');
  print(
    'a + b (Upcasted to Complex): ${c.data}',
  ); // [Complex(11, 2), Complex(23, 4)]
}

void runTrigAndRoundingExample() {
  print('\n--- New Math Ufuncs (Trig, Hyperbolic, Rounding, Clip) ---');
  final a = NDArray.fromList(Float64List.fromList([0.0, 0.5, 1.0]), [
    3,
  ], DType.float64);

  // Tangent and Atan2 with broadcasting
  final t = tan(a);
  print('tan(a): ${t.data}');

  final y = NDArray.fromList(Float64List.fromList([1.0]), [1], DType.float64);
  final x = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
    2,
  ], DType.float64);
  final at = atan2(y, x); // Broadcasts y to match x
  print('atan2(1.0, [1.0, 2.0]) shape: ${at.shape}, data: ${at.data}');

  // Hyperbolic trig functions
  print('sinh(a): ${sinh(a).data}');
  print('cosh(a): ${cosh(a).data}');
  print('tanh(a): ${tanh(a).data}');

  // Rounding operations
  final b = NDArray.fromList(Float64List.fromList([-1.6, 1.2, 2.7]), [
    3,
  ], DType.float64);
  print('b: ${b.data}');
  print('ceil(b): ${ceil(b).data}');
  print('floor(b): ${floor(b).data}');
  print('round(b): ${round(b).data}');

  // Clipping array elements
  final cl = clip(b, min: -1.0, max: 2.0);
  print('clip(b, -1.0, 2.0): ${cl.data}'); // [-1.0, 1.2, 2.0]
}

void runComplexAbsoluteValueExample() {
  print('\n--- Complex Absolute Value (Magnitude) ---');
  final a = NDArray<Complex>.create([2], DType.complex128);
  a.data[0] = Complex(3.0, 4.0); // magnitude = sqrt(3^2 + 4^2) = 5.0
  a.data[1] = Complex(0.0, -12.0); // magnitude = 12.0

  // abs() on complex produces a real Float64 array containing magnitudes
  final magnitudes = abs(a);
  print('Complex array: ${a.data}');
  print(
    'Magnitudes (Real array): ${magnitudes.data} with dtype ${magnitudes.dtype}',
  );
}

void runComparisonBroadcastingExample() {
  print('\n--- Full NumPy-style Comparison Operator Broadcasting ---');
  // Matrix of shape [2, 3]
  final mat = NDArray.fromList(
    Float64List.fromList([1.0, 5.0, 2.0, 4.0, 2.0, 6.0]),
    [2, 3],
    DType.float64,
  );

  // Row vector of shape [1, 3]
  final rowVec = NDArray.fromList(Float64List.fromList([3.0, 2.0, 4.0]), [
    1,
    3,
  ], DType.float64);

  // Perform broadcasting comparison: mat > rowVec
  // rowVec will be stretched vertically to compare each row!
  final mask = mat > rowVec;
  print('mat shape: ${mat.shape}');
  print('rowVec shape: ${rowVec.shape}');
  print('mat > rowVec mask shape: ${mask.shape}');
  // [1.0>3.0, 5.0>2.0, 2.0>4.0] -> [0, 1, 0]
  // [4.0>3.0, 2.0>2.0, 6.0>4.0] -> [1, 0, 1]
  print('mat > rowVec mask data: ${mask.data}');
}

void runLogicalOperationsExample() {
  print('\n--- Logical Operations (Combining Boolean Masks) ---');
  final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0]), [
    5,
  ], DType.float64);

  // Create two boolean masks (represented as int32 arrays of 1 and 0)
  final maskGT2 = a > 2.0; // [0, 0, 1, 1, 1]
  final maskLT5 = a < 5.0; // [1, 1, 1, 1, 0]

  // Combine them element-wise using logical_and
  final combined = logical_and(maskGT2, maskLT5);
  print('a: ${a.data}');
  print('a > 2.0: ${maskGT2.data}');
  print('a < 5.0: ${maskLT5.data}');
  print('logical_and(maskGT2, maskLT5): ${combined.data}'); // [0, 0, 1, 1, 0]
}

void runLogicalReductionsExample() {
  print('\n--- Logical Reductions (all, any) ---');
  final a = NDArray.fromList([true, true, false], [3], DType.boolean);
  print('a: ${a.data}');
  print('all(a): ${all(a)}'); // false
  print('any(a): ${any(a)}'); // true

  final mat = NDArray.fromList(
    [true, true, false, true, false, false],
    [2, 3],
    DType.boolean,
  );
  print('2D matrix:\n$mat');
  print('all(mat, axis: 0): ${all(mat, axis: 0).data}'); // [true, false, false]
}

void runAngleConvertersExample() {
  print('\n--- Angle Converters (deg2rad, rad2deg) ---');
  final deg = NDArray.fromList([180.0, 90.0, 45.0], [3], DType.float64);
  final rad = deg2rad(deg);
  print('Degrees: ${deg.data}');
  print('Radians: ${rad.data}'); // [pi, pi/2, pi/4]

  final back = rad2deg(rad);
  print('Radians: ${rad.data}');
  print('Back to Degrees: ${back.data}'); // [180, 90, 45]
}

void runEnumerateAndComplexComponentsExample() {
  print('\n--- Multidimensional Enumerator (ndenumerate) ---');
  final a = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.int32);
  for (final entry in ndenumerate(a)) {
    print('  Coordinate: ${entry.$1}, Value: ${entry.$2}');
  }

  print('\n--- Complex Components Extractors (real, imag) ---');
  final c = NDArray<Complex>.create([2], DType.complex128);
  c.data[0] = Complex(3.0, 4.0);
  c.data[1] = Complex(-1.0, 0.0);

  final r = real(c);
  final im = imag(c);

  print('Complex array: ${c.data}');
  print('Real component (Float64): ${r.data}'); // [3.0, -1.0]
  print('Imaginary component (Float64): ${im.data}'); // [4.0, 0.0]

  print('\n--- Zero-Copy view Demonstration ---');
  final realArr = NDArray.fromList([1.5, 2.5], [2], DType.float64);
  final realView = real(realArr); // Zero-copy view!
  print('Original Array: ${realArr.data}');
  print('Real View: ${realView.data}');

  // Mutating view updates parent automatically!
  realView.data[0] = 99.9;
  print('Mutated Real View: ${realView.data}');
  print('Propagated Parent Array: ${realArr.data}');
}
