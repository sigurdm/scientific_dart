// ignore_for_file: non_constant_identifier_names
import '../ndarray.dart';

// Standalone operational relative cross-imports
import 'helpers.dart';

/// Concatenates a list of arrays along a specified axis.
NDArray<T> concatenate<T extends Object>(
  List<NDArray<T>> arrays, {
  int axis = 0,
}) {
  if (arrays.isEmpty) {
    throw ArgumentError('List of arrays must not be empty');
  }

  for (final arr in arrays) {
    if (arr.isDisposed) {
      throw StateError('Cannot concatenate a disposed array.');
    }
  }

  final first = arrays.first;
  final rank = first.shape.length;
  final dtype = first.dtype;

  if (axis < 0 || axis >= rank) {
    throw RangeError.index(axis, first.shape, 'axis out of range');
  }

  for (var i = 1; i < arrays.length; i++) {
    final arr = arrays[i];
    if (arr.dtype != dtype) {
      throw ArgumentError('All arrays must have the same DType');
    }
    if (arr.shape.length != rank) {
      throw ArgumentError('All arrays must have the same rank');
    }
    for (var j = 0; j < rank; j++) {
      if (j != axis && arr.shape[j] != first.shape[j]) {
        throw ArgumentError('Shapes must match except in dimension $axis');
      }
    }
  }

  final targetShape = List<int>.from(first.shape);
  var totalAxisSize = 0;
  for (final arr in arrays) {
    totalAxisSize += arr.shape[axis];
  }
  targetShape[axis] = totalAxisSize;

  final result = NDArray<T>.create(targetShape, dtype);

  var allContiguous = true;
  for (final arr in arrays) {
    if (!arr.isContiguous) {
      allContiguous = false;
      break;
    }
  }

  if (allContiguous && axis == 0) {
    var destOffset = 0;
    for (final arr in arrays) {
      final size = arr.shape.isEmpty ? 1 : arr.shape.reduce((a, b) => a * b);
      copyContiguousFlat(arr, result, destOffset, size);
      destOffset += size;
    }
    return result;
  }

  var axisOffset = 0;
  for (final arr in arrays) {
    copyConcatenateRecursive(
      arr,
      result,
      axis,
      axisOffset,
      List<int>.filled(rank, 0),
      0,
    );
    axisOffset += arr.shape[axis];
  }

  return result;
}

/// Join a sequence of arrays along a new axis.
///
/// Stacks the input [arrays] along a new dimension at [axis]. All arrays in the
/// list must have the exact same shape and `DType.`
///
/// **Preconditions:**
/// - Input list [arrays] must be non-empty.
/// - All arrays in [arrays] must not be disposed.
/// - All arrays in [arrays] must share identical shapes and DTypes.
///
/// **Throws:**
/// - [ArgumentError] if [arrays] is empty, or shape/DType mismatches occur.
/// - [RangeError] if [axis] is out of bounds.
/// - [StateError] if any array is disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1, 2], [2], DType.int32);
/// final b = NDArray.fromList([3, 4], [2], DType.int32);
/// final s = stack([a, b], axis: 0); // shape [2, 2], values [[1, 2], [3, 4]]
/// ```
NDArray<T> stack<T extends Object>(List<NDArray<T>> arrays, {int axis = 0}) {
  if (arrays.isEmpty) {
    throw ArgumentError('List of arrays to stack must not be empty.');
  }

  for (final arr in arrays) {
    if (arr.isDisposed) {
      throw StateError('Cannot execute stack() on a disposed array.');
    }
  }

  final first = arrays.first;
  final rank = first.shape.length;
  final dtype = first.dtype;

  // Validate identical shapes and DTypes
  for (var i = 1; i < arrays.length; i++) {
    final arr = arrays[i];
    if (arr.dtype != dtype) {
      throw ArgumentError('All arrays in stack must have identical DTypes.');
    }
    if (!listEquals(arr.shape, first.shape)) {
      throw ArgumentError('All arrays in stack must have identical shapes.');
    }
  }

  // Target axis can range from -(rank + 1) to rank
  final targetAxis = axis < 0 ? rank + 1 + axis : axis;
  if (targetAxis < 0 || targetAxis > rank) {
    throw RangeError.range(targetAxis, 0, rank, 'axis');
  }

  // Build stacked output shape by inserting arrays.length at targetAxis index
  final stackedShape = List<int>.from(first.shape);
  stackedShape.insert(targetAxis, arrays.length);

  final result = NDArray.zeros(stackedShape, dtype);

  for (var i = 0; i < arrays.length; i++) {
    final currentIndices = List<int>.filled(first.shape.length, 0);
    copyStackRecursive(arrays[i], result, targetAxis, i, currentIndices, 0);
  }

  return result;
}

/// Expand the shape of an array by inserting a new axis of size 1.
///
/// Inserts a new dimension of size 1 at the specified [axis] position.
/// If [axis] is negative, it is normalized relative to the rank of [a] plus 1.
///
/// **Preconditions:**
/// - [axis] normalized value must be between `0` and the rank of [a] inclusive.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Performance considerations:**
/// - This is a zero-copy view manipulation executing in absolute $O(1)$ time complexity.
///
/// **Example:**
/// {@example /example/shape_view_example.dart lang=dart}
///
/// Reference: [Expand Dimensions](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html)
NDArray<T> expand_dims<T extends Object>(NDArray<T> a, int axis) {
  final rank = a.shape.length;
  var targetAxis = axis < 0 ? rank + 1 + axis : axis;

  if (targetAxis < 0 || targetAxis > rank) {
    throw ArgumentError(
      'Axis $axis is out of bounds for array of rank $rank (valid bounds: [${-rank - 1}, $rank])',
    );
  }

  final newShape = List<int>.from(a.shape);
  final newStrides = List<int>.from(a.strides);

  newShape.insert(targetAxis, 1);
  // Insert stride mapping: copy sibling stride or default to 1 if rank is 0
  final siblingStride = targetAxis < rank ? a.strides[targetAxis] : 1;
  newStrides.insert(targetAxis, siblingStride);

  return NDArray.view(
    a,
    shape: newShape,
    strides: newStrides,
    offsetElements: 0,
  );
}

/// Remove axes of size 1 from the shape of an array.
///
/// If [axis] is provided, only removes the specified dimensions (which must have size 1).
/// If [axis] is null, removes all dimensions of size 1.
///
/// **Preconditions:**
/// - Specified [axis] entries must indeed correspond to dimensions of size 1.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds or targets a dimension whose size is not 1.
///
/// **Performance considerations:**
/// - This is a zero-copy view manipulation executing in absolute $O(1)$ time complexity.
///
/// **Example:**
/// {@example /example/shape_view_example.dart lang=dart}
///
/// Reference: [Squeeze Dimensions](https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html)
NDArray<T> squeeze<T extends Object>(NDArray<T> a, {List<int>? axis}) {
  final rank = a.shape.length;
  final shape = a.shape;
  final strides = a.strides;

  final newShape = <int>[];
  final newStrides = <int>[];

  final squeezeAxes = <int>{};
  if (axis != null) {
    for (final ax in axis) {
      final targetAx = ax < 0 ? rank + ax : ax;
      if (targetAx < 0 || targetAx >= rank) {
        throw ArgumentError('Axis $ax is out of bounds for rank $rank');
      }
      if (shape[targetAx] != 1) {
        throw ArgumentError(
          'Cannot squeeze axis $ax because its dimension size is ${shape[targetAx]} (must be 1)',
        );
      }
      squeezeAxes.add(targetAx);
    }
  } else {
    for (var i = 0; i < rank; i++) {
      if (shape[i] == 1) {
        squeezeAxes.add(i);
      }
    }
  }

  for (var i = 0; i < rank; i++) {
    if (!squeezeAxes.contains(i)) {
      newShape.add(shape[i]);
      newStrides.add(strides[i]);
    }
  }

  // Squeezing all dimensions of a 1D unit tensor (e.g. shape [1]) yields a 0D scalar shape []
  return NDArray.view(
    a,
    shape: newShape,
    strides: newStrides,
    offsetElements: 0,
  );
}

/// Create a sliding window view over several dimensions of an array.
///
/// This corresponds to NumPy's `lib.stride_tricks.sliding_window_view` function.
///
/// **Mathematical Mechanics**:
/// By manipulating strides, a sliding window of shape `windowShape` can be created
/// without copying any element data (completely zero-copy, copy-free, and zero-allocation).
///
/// For an input array `a` with shape $S = (s_0, \dots, s_{D-1})$ and strides $V = (v_0, \dots, v_{D-1})$:
/// - The axes to apply sliding windows are specified by [axis] (defaults to all axes).
/// - The output shape has the original dimensions reduced by `windowShape - 1`, with the window
///   dimensions appended at the end:
///   $$S_{\text{out}} = (s_0 - w_0 + 1, \dots, s_{k} - w_k + 1, \dots, w_0, \dots, w_k)$$
/// - The output strides has the original strides, with the original strides of the window axes appended at the end:
///   $$V_{\text{out}} = (v_0, \dots, v_{D-1}, v_{\text{axis}_0}, \dots, v_{\text{axis}_k})$$
///
/// **Preconditions:**
/// - [windowShape] length must match [axis] length.
/// - Each window dimension must be strictly positive and less than or equal to the corresponding axis size.
///
/// **Throws:**
/// - [ArgumentError] if axes are out of range, shapes mismatch, or window dimensions exceed axis sizes.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0, 5.0], [5], DType.float64);
/// final view = slidingWindowView(a, [3]);
/// print(view.shape); // [3, 3]
/// print(view.toList()); // [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]
/// ```
NDArray<T> slidingWindowView<T extends Object>(
  NDArray<T> a,
  List<int> windowShape, {
  List<int>? axis,
}) {
  final rank = a.shape.length;

  // 1. Resolve and validate target axes
  final targetAxes = <int>[];
  if (axis != null) {
    for (var ax in axis) {
      final resolved = ax < 0 ? rank + ax : ax;
      if (resolved < 0 || resolved >= rank) {
        throw RangeError.range(resolved, 0, rank - 1, 'axis');
      }
      if (targetAxes.contains(resolved)) {
        throw ArgumentError(
          'Duplicate axis specified in sliding window: $resolved',
        );
      }
      targetAxes.add(resolved);
    }
  } else {
    // Default is all axes
    for (var i = 0; i < rank; i++) {
      targetAxes.add(i);
    }
  }

  if (windowShape.length != targetAxes.length) {
    throw ArgumentError(
      'windowShape length (${windowShape.length}) must match axes length (${targetAxes.length})',
    );
  }

  // 2. Calculate output shape and strides
  final outShape = List<int>.from(a.shape);
  final outStrides = List<int>.from(a.strides);

  for (var i = 0; i < targetAxes.length; i++) {
    final ax = targetAxes[i];
    final wSize = windowShape[i];
    final aSize = a.shape[ax];

    if (wSize <= 0) {
      throw ArgumentError(
        'windowShape dimensions must be strictly positive (was $wSize)',
      );
    }
    if (wSize > aSize) {
      throw ArgumentError(
        'windowShape dimension ($wSize) cannot exceed axis size ($aSize) for axis $ax',
      );
    }

    outShape[ax] = aSize - wSize + 1;
  }

  // Append window dimensions and their corresponding strides at the end
  for (var i = 0; i < targetAxes.length; i++) {
    final ax = targetAxes[i];
    outShape.add(windowShape[i]);
    outStrides.add(a.strides[ax]);
  }

  // 3. Return the zero-copy NDArray view sharing backing unmanaged C memory
  return NDArray.view(a, shape: outShape, strides: outStrides);
}

/// Reverses the order of elements along the given axis/axes.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - If [axis] is a list, all elements must be unique.
/// - Each axis must be a valid axis index for [a] (within `[-rank, rank - 1]`).
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [RangeError] if any axis is out of bounds.
/// - [ArgumentError] if [axis] contains duplicate indices.
///
/// **Performance considerations:**
/// - Algorithmic Time Complexity is $O(1)$ because it returns a zero-copy strided view.
/// - Space Complexity is $O(1)$ as no new array data is allocated.
///
/// **Reference:**
/// Refer to [NumPy flip documentation](https://numpy.org/doc/stable/reference/generated/numpy.flip.html).
///
/// {@example /example/rearranging_example.dart lang=dart}
NDArray<T> flip<T extends Object>(NDArray<T> a, {dynamic axis}) {
  if (a.isDisposed) {
    throw StateError('Cannot flip a disposed array.');
  }

  final rank = a.rank;
  if (rank == 0) {
    return NDArray.view(
      a,
      shape: [],
      strides: [],
      offsetElements: a.offsetElements,
    );
  }

  final List<int> axesToFlip;
  if (axis == null) {
    axesToFlip = List.generate(rank, (i) => i);
  } else if (axis is int) {
    final normAx = axis < 0 ? rank + axis : axis;
    if (normAx < 0 || normAx >= rank) {
      throw RangeError.range(normAx, 0, rank - 1, 'axis');
    }
    axesToFlip = [normAx];
  } else if (axis is List<int>) {
    final uniqueAxes = <int>{};
    for (final ax in axis) {
      final normAx = ax < 0 ? rank + ax : ax;
      if (normAx < 0 || normAx >= rank) {
        throw RangeError.range(normAx, 0, rank - 1, 'axis');
      }
      if (!uniqueAxes.add(normAx)) {
        throw ArgumentError('axes must be unique');
      }
    }
    axesToFlip = uniqueAxes.toList();
  } else {
    throw ArgumentError('axis must be null, an integer, or a list of integers');
  }

  final newStrides = List<int>.from(a.strides);
  var offset = a.offsetElements;

  for (final ax in axesToFlip) {
    newStrides[ax] = a.strides[ax] * -1;
    offset += (a.shape[ax] - 1) * a.strides[ax];
  }

  return NDArray.view(
    a,
    shape: a.shape,
    strides: newStrides,
    offsetElements: offset,
  );
}

/// Flips the array in the left/right direction (column-wise).
///
/// Equivalent to `flip(a, axis: 1)`.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [a] must have a rank of at least 2.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [a] rank is less than 2.
///
/// **Performance considerations:**
/// - Algorithmic Time Complexity is $O(1)$ because it returns a zero-copy strided view.
/// - Space Complexity is $O(1)$ as no new array data is allocated.
///
/// **Reference:**
/// Refer to [NumPy fliplr documentation](https://numpy.org/doc/stable/reference/generated/numpy.fliplr.html).
///
/// {@example /example/rearranging_example.dart lang=dart}
NDArray<T> fliplr<T extends Object>(NDArray<T> a) {
  if (a.isDisposed) {
    throw StateError('Cannot fliplr a disposed array.');
  }
  if (a.rank < 2) {
    throw ArgumentError('Input must be >= 2-D.');
  }
  return flip(a, axis: 1);
}

/// Flips the array in the up/down direction (row-wise).
///
/// Equivalent to `flip(a, axis: 0)`.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [a] must have a rank of at least 1.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [a] rank is less than 1.
///
/// **Performance considerations:**
/// - Algorithmic Time Complexity is $O(1)$ because it returns a zero-copy strided view.
/// - Space Complexity is $O(1)$ as no new array data is allocated.
///
/// **Reference:**
/// Refer to [NumPy flipud documentation](https://numpy.org/doc/stable/reference/generated/numpy.flipud.html).
///
/// {@example /example/rearranging_example.dart lang=dart}
NDArray<T> flipud<T extends Object>(NDArray<T> a) {
  if (a.isDisposed) {
    throw StateError('Cannot flipud a disposed array.');
  }
  if (a.rank < 1) {
    throw ArgumentError('Input must be >= 1-D.');
  }
  return flip(a, axis: 0);
}
