import 'dart:typed_data';
import 'package:num_dart/num_dart.dart';

void main() {
  print(
    '=== NDArray Universal Functions (ufuncs) & Broadcasting Examples ===\n',
  );
  runMixedTypeArithmeticExample();
  runTrigAndRoundingExample();
  runComplexAbsoluteValueExample();
  runComparisonBroadcastingExample();
  runLogicalOperationsExample();
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
  final cl = clip(b, -1.0, 2.0);
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
