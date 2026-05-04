import 'dart:typed_data';
import 'package:num_dart/num_dart.dart';

void main() {
  print('=== NDArray Shape Manipulation Examples ===\n');
  runExpandDimsExample();
  runSqueezeExample();
  runSwapaxesExample();
  runMoveaxisExample();
  runTileExample();
  runRepeatExample();
}

void runExpandDimsExample() {
  print('--- expandDims Example ---');
  final a = NDArray.fromList(Float64List.fromList([1, 2, 3, 4]), [
    2,
    2,
  ], DType.float64);
  print('Original shape: ${a.shape}'); // [2, 2]

  // Insert a new dimension at axis 0
  final b = a.expandDims(0);
  print('Expanded axis 0 shape: ${b.shape}'); // [1, 2, 2]
  print('Expanded axis 0 data: ${b.data}');

  // Insert a new dimension at the end (axis 2 or -1)
  final c = a.expandDims(-1);
  print('Expanded last axis shape: ${c.shape}'); // [2, 2, 1]
}

void runSqueezeExample() {
  print('\n--- squeeze Example ---');
  final a = NDArray.fromList(Float64List.fromList([1, 2]), [
    1,
    2,
    1,
  ], DType.float64);
  print('Original shape: ${a.shape}'); // [1, 2, 1]

  // Squeeze all dimensions of size 1
  final b = a.squeeze();
  print('Squeezed all shape: ${b.shape}'); // [2]

  // Squeeze only a specific axis
  final c = a.squeeze(axis: 0);
  print('Squeezed axis 0 shape: ${c.shape}'); // [2, 1]
}

void runSwapaxesExample() {
  print('\n--- swapaxes Example ---');
  final a = NDArray.fromList(Float64List.fromList([1, 2, 3, 4, 5, 6]), [
    2,
    3,
  ], DType.float64);
  print('Original shape: ${a.shape}'); // [2, 3]

  // Swap axis 0 and axis 1
  final b = a.swapaxes(0, 1);
  print('Swapped (0, 1) shape: ${b.shape}'); // [3, 2]
}

void runMoveaxisExample() {
  print('\n--- moveaxis Example ---');
  final a = NDArray.create([2, 3, 4], DType.float64);
  print('Original shape: ${a.shape}'); // [2, 3, 4]

  // Move axis 0 to the end (axis 2)
  final b = a.moveaxis(0, 2);
  print('Moved axis 0 to 2 shape: ${b.shape}'); // [3, 4, 2]
}

void runTileExample() {
  print('\n--- tile Example ---');
  final a = NDArray.fromList(Float64List.fromList([1, 2]), [2], DType.float64);
  print('Original array: ${a.data} with shape ${a.shape}');

  // Tile 3 times along the single axis
  final b = tile(a, 3);
  print('Tiled 3 times shape: ${b.shape}'); // [6]
  print('Tiled 3 times data: ${b.data}'); // [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]

  // Tile a 2D block reps
  final c = tile(a, [2, 2]);
  print('Tiled with [2, 2] shape: ${c.shape}'); // [2, 4]
}

void runRepeatExample() {
  print('\n--- repeat Example ---');
  final a = NDArray.fromList(Float64List.fromList([1, 2, 3, 4]), [
    2,
    2,
  ], DType.float64);
  print('Original array:\n${a.data} shape ${a.shape}');

  // Repeat elements 2 times along axis 0
  final b = repeat(a, 2, axis: 0);
  print('Repeated 2x axis 0 shape: ${b.shape}'); // [4, 2]
  print('Repeated 2x axis 0 data: ${b.data}');

  // Repeat with a custom list per element along axis 1
  final c = repeat(a, [1, 3], axis: 1);
  print('Repeated with [1, 3] axis 1 shape: ${c.shape}'); // [2, 4]
}
