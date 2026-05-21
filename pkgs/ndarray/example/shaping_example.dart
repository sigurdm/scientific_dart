import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray Shaping & Meshes Examples ===\n');
  runAsStridedExamples();
  runMGridExamples();
  runOGridExamples();
}

void runAsStridedExamples() {
  print('--- 1. Low-level Striding View (asStrided) ---');
  NDArray.scope(() {
    final a = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
    print('Original 1D array: ${a.toList()}');

    // Create a 2x2 strided view pointing to the same memory
    final view = asStrided(a, shape: [2, 2], strides: [2, 1]);
    print('2x2 strided view: ${view.toList()}');

    // Mutating the view affects the original array
    view.setCell([0, 0], Int32(99));
    print('After modifying view[0, 0] to 99:');
    print('View: ${view.toList()}');
    print('Original: ${a.toList()}\n');
  });
}

void runMGridExamples() {
  print('--- 2. Dense Meshgrid (mgrid) ---');
  NDArray.scope(() {
    // Generate dense meshgrid of shape [2, 3, 2]
    final grid = mgrid([
      GridRange(0, 3), // 0 to 2 inclusive
      GridRange(0, 2), // 0 to 1 inclusive
    ]);

    print('Grid shape: ${grid.shape}');
    print('Grid coordinates: ${grid.toList()}');

    // Check individual coordinate arrays
    final x = grid.slice([Index(0)]);
    final y = grid.slice([Index(1)]);
    print('X coordinate grid:\n$x');
    print('Y coordinate grid:\n$y\n');
  });
}

void runOGridExamples() {
  print('--- 3. Open Meshgrid (ogrid) ---');
  NDArray.scope(() {
    // Generate open meshgrid of two arrays
    final grid = ogrid([GridRange(0, 3), GridRange(0, 2)]);

    print('X grid shape: ${grid[0].shape}');
    print('X grid: ${grid[0].toList()}');
    print('Y grid shape: ${grid[1].shape}');
    print('Y grid: ${grid[1].toList()}\n');
  });
}
