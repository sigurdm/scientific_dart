import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray fftshift and ifftshift Spectrum Shifting Examples ===\n');

  // 1D Example (Odd Length)
  print('--- 1D Spectrum Shifting (Odd Length N=5) ---');
  final signal1D = NDArray.fromList(
    [0.0, 1.0, 2.0, 3.0, 4.0],
    [5],
    DType.float64,
  );
  print('Original 1D Array:   ${signal1D.toList()}');

  final shifted1D = fftshift(signal1D);
  print('After fftshift:      ${shifted1D.toList()}');

  final restored1D = ifftshift(shifted1D);
  print('After ifftshift:     ${restored1D.toList()}\n');

  // 1D Example (Even Length)
  print('--- 1D Spectrum Shifting (Even Length N=6) ---');
  final signalEven = NDArray.fromList(
    [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
    [6],
    DType.float64,
  );
  print('Original 1D Array:   ${signalEven.toList()}');

  final shiftedEven = fftshift(signalEven);
  print('After fftshift:      ${shiftedEven.toList()}');

  final restoredEven = ifftshift(shiftedEven);
  print('After ifftshift:     ${restoredEven.toList()}\n');

  // 2D Grid Example
  print('--- 2D Grid Spectrum Shifting (2D Shape: [2, 3]) ---');
  final grid = NDArray.fromList(
    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    [2, 3],
    DType.float64,
  );
  print('Original 2D Grid:');
  _print2DGrid(grid);

  final shifted2D = fftshift(grid);
  print('\nAfter fftshift (both axes):');
  _print2DGrid(shifted2D);

  final shifted2DAxis0 = fftshift(grid, axes: 0);
  print('\nAfter fftshift (axis 0 only):');
  _print2DGrid(shifted2DAxis0);

  final restored2D = ifftshift(shifted2D);
  print('\nAfter ifftshift (both axes, restored):');
  _print2DGrid(restored2D);

  // Cleanup scope
  signal1D.dispose();
  shifted1D.dispose();
  restored1D.dispose();
  signalEven.dispose();
  shiftedEven.dispose();
  restoredEven.dispose();
  grid.dispose();
  shifted2D.dispose();
  shifted2DAxis0.dispose();
  restored2D.dispose();
}

void _print2DGrid(NDArray a) {
  final list = a.toList();
  final cols = a.shape[1];
  for (var r = 0; r < a.shape[0]; r++) {
    final row = list.sublist(r * cols, (r + 1) * cols);
    print('  $row');
  }
}
