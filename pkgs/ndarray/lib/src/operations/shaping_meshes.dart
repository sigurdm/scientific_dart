// ignore_for_file: non_constant_identifier_names
import '../ndarray.dart';

// Standalone operational relative cross-imports
import 'spacers.dart';

/// Represents a range specification for meshgrid operations.
///
/// Generates evenly-spaced coordinate values along a grid dimension.
///
/// **Preconditions:**
/// - [step] must be non-zero.
/// - If [numPoints] is specified, it must be strictly positive ($\ge 1$).
///
/// **Throws:**
/// - [ArgumentError] if [step] is 0.
/// - [ArgumentError] if [numPoints] is negative or 0.
///
/// **Example:**
/// {@example /example/shaping_example.dart lang=dart}
final class GridRange {
  /// The starting value of the range (inclusive).
  final double start;

  /// The ending value of the range.
  ///
  /// If [numPoints] is specified, [stop] is inclusive.
  /// Otherwise, [stop] is exclusive (standard half-open range).
  final double stop;

  /// The step size between adjacent points.
  /// Only active if [numPoints] is null. Defaults to 1.0.
  final double step;

  /// The total number of points to generate.
  /// If not null, the points are generated evenly spaced between [start]
  /// and [stop] (inclusive), and [step] is ignored.
  final int? numPoints;

  /// Creates a new grid range specification.
  GridRange(this.start, this.stop, {this.step = 1.0, this.numPoints}) {
    if (step == 0.0) {
      throw ArgumentError('Step cannot be zero');
    }
    if (numPoints != null && numPoints! <= 0) {
      throw ArgumentError('numPoints must be positive');
    }
  }

  /// Creates a grid range specification using NumPy-style parameters.
  ///
  /// If [step] is a [Complex] number with a non-zero imaginary part,
  /// the absolute value of the imaginary part is treated as the [numPoints] count.
  /// Otherwise, the real part of [step] (or a real [num]) is treated as the step size.
  factory GridRange.numpy(double start, double stop, dynamic step) {
    if (step is Complex && step.imag != 0.0) {
      return GridRange(start, stop, numPoints: step.imag.abs().toInt());
    } else if (step is num) {
      return GridRange(start, stop, step: step.toDouble());
    } else {
      throw ArgumentError('step must be a num or a Complex number');
    }
  }
}

/// Helper to generate a 1D coordinate array from a [GridRange].
NDArray<double> _generate1DCoordinate(GridRange range, DType<double> dtype) {
  if (range.numPoints != null) {
    return linspace<double>(
      range.start,
      range.stop,
      range.numPoints!,
      dtype: dtype,
    );
  } else {
    return NDArray<double>.arange(
      range.start,
      range.stop,
      step: range.step,
      dtype: dtype,
    );
  }
}

/// Creates a view of [x] with the given [shape] and [strides].
///
/// [shape] and [strides] default to the array's shape and strides if not provided.
///
/// **Memory Safety Warning:**
/// This function is extremely low-level and does not perform bounds safety checks
/// on memory accesses. The user must ensure that the specified shape and strides
/// do not reference memory outside of the underlying C buffer. Accessing elements
/// out of bounds will result in memory corruption, undefined behavior, or crashes.
///
/// **Preconditions:**
/// - [x] must not be disposed.
/// - If [shape] is provided, all its dimensions must be positive.
/// - If [strides] is provided, its length must match the length of [shape].
///
/// **Throws:**
/// - [StateError] if [x] has been disposed.
/// - [ArgumentError] if [shape] and [strides] lengths do not match.
///
/// **Memory Ownership & Lifetime View Warning:**
/// > [!WARNING]
/// > This operation returns a **zero-copy metadata view** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned view will **silently mutate the original array**. Disposing of the parent array [x] will invalidate the returned view. Calling [dispose()] on the returned view does nothing.
///
/// **Example:**
/// {@example /example/shaping_example.dart lang=dart}
///
/// Refer to the [NumPy as_strided reference](https://numpy.org/doc/stable/reference/generated/numpy.lib.stride_tricks.as_strided.html)
/// for details.
NDArray<T> asStrided<T>(NDArray<T> x, {List<int>? shape, List<int>? strides}) {
  if (x.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  final targetShape = shape ?? x.shape;
  final targetStrides = strides ?? x.strides;

  if (targetShape.length != targetStrides.length) {
    throw ArgumentError(
      'Shape length (${targetShape.length}) must match strides length (${targetStrides.length}).',
    );
  }

  return NDArray<T>.view(
    x,
    shape: targetShape,
    strides: targetStrides,
    offsetElements: x.offsetElements,
  );
}

/// Returns an open multi-dimensional mesh-grid.
///
/// Returns a list of zero-allocation, zero-copy, broadcastable [NDArray]s,
/// one for each dimension range in [ranges].
///
/// **Preconditions:**
/// - [ranges] must not be empty.
///
/// **Throws:**
/// - [ArgumentError] if [ranges] is empty.
///
/// **Example:**
/// {@example /example/shaping_example.dart lang=dart}
///
/// Refer to the [NumPy ogrid reference](https://numpy.org/doc/stable/reference/generated/numpy.ogrid.html)
/// for details.
List<NDArray<double>> ogrid(
  List<GridRange> ranges, {
  DType<double> dtype = DType.float64,
}) {
  if (ranges.isEmpty) {
    throw ArgumentError('ranges must not be empty.');
  }

  final k = ranges.length;
  final results = <NDArray<double>>[];

  for (var i = 0; i < k; i++) {
    final arr1D = _generate1DCoordinate(ranges[i], dtype);
    final shape = List<int>.filled(k, 1);
    shape[i] = arr1D.size;

    final reshaped = arr1D.reshape(shape);
    reshaped.detachToParentScope();
    results.add(reshaped);
  }

  return results;
}

/// Returns a dense multi-dimensional mesh-grid.
///
/// Returns a single contiguous [NDArray] containing the coordinate grids.
///
/// **Preconditions:**
/// - [ranges] must not be empty.
///
/// **Throws:**
/// - [ArgumentError] if [ranges] is empty.
///
/// **Example:**
/// {@example /example/shaping_example.dart lang=dart}
///
/// Refer to the [NumPy mgrid reference](https://numpy.org/doc/stable/reference/generated/numpy.mgrid.html)
/// for details.
NDArray<double> mgrid(
  List<GridRange> ranges, {
  DType<double> dtype = DType.float64,
}) {
  if (ranges.isEmpty) {
    throw ArgumentError('ranges must not be empty.');
  }

  final k = ranges.length;

  // 1. Generate 1D coordinates to determine shape of the grid
  final allCoords = <List<double>>[];
  final gridShape = <int>[];

  for (var i = 0; i < k; i++) {
    final arr1D = _generate1DCoordinate(ranges[i], dtype);
    allCoords.add(arr1D.toList());
    gridShape.add(arr1D.size);
    arr1D.dispose();
  }

  // 2. Allocate the dense result array of shape [k, d1, d2, ..., dk]
  final outputShape = [k, ...gridShape];
  final result = NDArray<double>.create(outputShape, dtype);
  final gridStrides = NDArray.computeCStrides(gridShape);
  final gridSize = gridShape.isEmpty ? 1 : gridShape.reduce((a, b) => a * b);

  // 3. Walk recursively to fill coordinates in-place
  for (var i = 0; i < k; i++) {
    final coords = allCoords[i];
    final gridOffset = i * gridSize;

    void walk(int dim, int currentOffset, double? val) {
      if (dim == k) {
        result.data[gridOffset + currentOffset] = val!;
        return;
      }

      final size = gridShape[dim];
      final stride = gridStrides[dim];

      if (dim == i) {
        for (var c = 0; c < size; c++) {
          walk(dim + 1, currentOffset + c * stride, coords[c]);
        }
      } else {
        for (var c = 0; c < size; c++) {
          walk(dim + 1, currentOffset + c * stride, val);
        }
      }
    }

    walk(0, 0, null);
  }

  result.detachToParentScope();
  return result;
}
