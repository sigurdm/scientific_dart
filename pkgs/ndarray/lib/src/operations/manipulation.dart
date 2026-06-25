// ignore_for_file: non_constant_identifier_names
import 'dart:math' as math;
import '../ndarray.dart';
import '../ndarray_bindings.dart';
import '../scratch_arena.dart';


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
///
/// **Memory Ownership & Lifetime View Warning:**
/// > [!WARNING]
/// > This operation returns a **zero-copy metadata view** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned view will **silently mutate the original array**. Disposing of the parent array [a] will invalidate the returned view. Calling [dispose] on the returned view does nothing.
NDArray<T> expand_dims<T extends Object>(NDArray<T> a, int axis) {
  if (a.isDisposed) {
    throw StateError('Cannot execute expand_dims() on a disposed array.');
  }
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
///
/// **Memory Ownership & Lifetime View Warning:**
/// > [!WARNING]
/// > This operation returns a **zero-copy metadata view** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned view will **silently mutate the original array**. Disposing of the parent array [a] will invalidate the returned view. Calling [dispose] on the returned view does nothing.
NDArray<T> squeeze<T extends Object>(NDArray<T> a, {List<int>? axis}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute squeeze() on a disposed array.');
  }
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
///
/// **Memory Ownership & Lifetime View Warning:**
/// > [!WARNING]
/// > This operation returns a **zero-copy metadata view** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned view will **silently mutate the original array**. Disposing of the parent array [a] will invalidate the returned view. Calling [dispose] on the returned view does nothing.
NDArray<T> slidingWindowView<T extends Object>(
  NDArray<T> a,
  List<int> windowShape, {
  List<int>? axis,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute slidingWindowView() on a disposed array.');
  }
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
///
/// **Memory Ownership & Lifetime View Warning:**
/// > [!WARNING]
/// > This operation returns a **zero-copy metadata view** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned view will **silently mutate the original array**. Disposing of the parent array [a] will invalidate the returned view. Calling [dispose] on the returned view does nothing.
NDArray<T> flip<T extends Object>(NDArray<T> a, {dynamic axis}) {
  if (a.isDisposed) {
    throw StateError('Cannot flip a disposed array.');
  }

  final rank = a.rank;
  if (rank == 0) {
    return NDArray.view(a, shape: [], strides: [], offsetElements: 0);
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
  var offset = 0;

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

/// Stacks arrays in sequence vertically (row wise).
NDArray<T> vstack<T extends Object>(List<NDArray<T>> arrays) {
  return concatenate(arrays, axis: 0);
}

/// Stacks arrays in sequence horizontally (column wise).
NDArray<T> hstack<T extends Object>(List<NDArray<T>> arrays) {
  return concatenate(arrays, axis: 1);
}

/// Returns a deep, C-contiguous copy of the given array.
///
/// The copy preserves the logical order and values of the elements defined by
/// [a]'s shape and strides. However, the physical memory layout of the returned
/// array is always contiguous (and its strides are reset to standard C-contiguous
/// strides).
///
/// This function corresponds to NumPy's `copy` function.
///
/// **Throws:**
/// - [StateError] if the array [a] is already disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1, 2], [2], DType.int32);
/// final b = copy(a);
/// b.data[0] = 99;
/// print(a.data[0]); // 1 (decoupled memory!)
/// ```
NDArray<T> copy<T extends Object>(NDArray<T> a) {
  if (a.isDisposed) {
    throw StateError('Cannot execute copy() on a disposed array.');
  }
  return a.copy();
}

/// Extracts a diagonal or constructs a diagonal array.
///
/// **Memory Ownership & Lifetime Warning:**
/// - **If [v] is 2D (extracting diagonal):** Returns a **zero-copy metadata view** sharing the underlying C memory page. Mutating elements inside the returned view will **silently mutate the original array**. Disposing of the parent array [v] will invalidate the returned view. Calling [dispose] on the view does nothing.
/// - **If [v] is 1D (constructing diagonal):** Allocates a new 2D array on the unmanaged C heap (unless [out] is provided). **The caller takes full ownership** of this memory and **must explicitly call [dispose]** to prevent native leaks.
///
/// **Example:**
/// {@example /example/diag_example.dart lang=dart}
///
/// Reference: [Diagonal Matrix](https://en.wikipedia.org/wiki/Diagonal_matrix)
NDArray<T> diag<T>(NDArray<T> v, {int k = 0, NDArray<T>? out}) {
  if (v.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute diag() on a disposed array.');
  }
  if (v.shape.length == 2) {
    final m = v.shape[0];
    final n = v.shape[1];

    int startRow;
    int startCol;
    int len;

    if (k >= 0) {
      startRow = 0;
      startCol = k;
      if (startCol >= n) {
        return NDArray<T>.create([0], v.dtype);
      }
      len = math.min(m, n - k);
    } else {
      startRow = -k;
      startCol = 0;
      if (startRow >= m) {
        return NDArray<T>.create([0], v.dtype);
      }
      len = math.min(m + k, n);
    }

    if (len <= 0) {
      return NDArray<T>.create([0], v.dtype);
    }

    final offsetElements = startRow * v.strides[0] + startCol * v.strides[1];
    final diagStride = v.strides[0] + v.strides[1];

    return NDArray<T>.view(
      v,
      shape: [len],
      strides: [diagStride],
      offsetElements: offsetElements,
    );
  } else if (v.shape.length == 1) {
    final n = v.shape[0];
    final size = n + k.abs();
    final targetShape = [size, size];

    final result = out ?? NDArray<T>.zeros(targetShape, v.dtype);
    if (out != null) {
      if (!listEquals(out.shape, targetShape) || out.dtype != v.dtype) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
      for (var i = 0; i < result.data.length; i++) {
        result.data[i] = castValue(0, v.dtype) as T;
      }
    }

    int startRow;
    int startCol;

    if (k >= 0) {
      startRow = 0;
      startCol = k;
    } else {
      startRow = -k;
      startCol = 0;
    }

    final vList = v.toList();
    final resData = result.data;
    final resStrides = result.strides;

    for (var i = 0; i < n; i++) {
      final targetIdx =
          (startRow + i) * resStrides[0] + (startCol + i) * resStrides[1];
      resData[targetIdx] = vList[i];
    }

    return result;
  } else {
    throw ArgumentError('Input array must be 1- or 2-dimensional.');
  }
}

/// Extract a lower triangular matrix (on and below the k-th diagonal) element-wise.
///
/// **Preconditions:**
/// - Input [a] must be an array with rank >= 2.
/// - If provided, the [out] recycler must have matching shape and dtype.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [a] has rank < 2.
/// - [ArgumentError] if [out] has mismatched shape or dtype.
///
/// **Example:**
/// {@example /example/triangular_example.dart lang=dart}
NDArray<T> tril<T>(NDArray<T> a, {int k = 0, NDArray<T>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute tril() on a disposed array.');
  }
  if (a.shape.length < 2) {
    throw ArgumentError('Input array must have rank >= 2.');
  }
  final result = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final rank = a.shape.length;
  final rows = a.shape[rank - 2];
  final cols = a.shape[rank - 1];

  final batchCount = a.shape.isEmpty
      ? 1
      : a.shape.sublist(0, rank - 2).isEmpty
      ? 1
      : a.shape.sublist(0, rank - 2).reduce((x, y) => x * y);

  if (a.isContiguous && result.isContiguous) {
    if (a.dtype == DType.float64) {
      v_tril_double(
        a.pointer.cast(),
        result.pointer.cast(),
        batchCount,
        rows,
        cols,
        k,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      v_tril_float(
        a.pointer.cast(),
        result.pointer.cast(),
        batchCount,
        rows,
        cols,
        k,
      );
      return result;
    }
  }

  final aList = a.isContiguous ? a.data : a.toList();
  final resData = result.data;
  final matrixSize = rows * cols;

  for (var b = 0; b < batchCount; b++) {
    final offset = b * matrixSize;
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        final idx = offset + r * cols + c;
        resData[idx] = (c <= r + k) ? aList[idx] : castValue(0, a.dtype) as T;
      }
    }
  }
  return result;
}

/// Extract an upper triangular matrix (on and above the k-th diagonal) element-wise.
///
/// **Preconditions:**
/// - Input [a] must be an array with rank >= 2.
/// - If provided, the [out] recycler must have matching shape and dtype.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [a] has rank < 2.
/// - [ArgumentError] if [out] has mismatched shape or dtype.
///
/// **Example:**
/// {@example /example/triangular_example.dart lang=dart}
NDArray<T> triu<T>(NDArray<T> a, {int k = 0, NDArray<T>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute triu() on a disposed array.');
  }
  if (a.shape.length < 2) {
    throw ArgumentError('Input array must have rank >= 2.');
  }
  final result = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final rank = a.shape.length;
  final rows = a.shape[rank - 2];
  final cols = a.shape[rank - 1];

  final batchCount = a.shape.isEmpty
      ? 1
      : a.shape.sublist(0, rank - 2).isEmpty
      ? 1
      : a.shape.sublist(0, rank - 2).reduce((x, y) => x * y);

  if (a.isContiguous && result.isContiguous) {
    if (a.dtype == DType.float64) {
      v_triu_double(
        a.pointer.cast(),
        result.pointer.cast(),
        batchCount,
        rows,
        cols,
        k,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      v_triu_float(
        a.pointer.cast(),
        result.pointer.cast(),
        batchCount,
        rows,
        cols,
        k,
      );
      return result;
    }
  }

  final aList = a.isContiguous ? a.data : a.toList();
  final resData = result.data;
  final matrixSize = rows * cols;

  for (var b = 0; b < batchCount; b++) {
    final offset = b * matrixSize;
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        final idx = offset + r * cols + c;
        resData[idx] = (c >= r + k) ? aList[idx] : castValue(0, a.dtype) as T;
      }
    }
  }
  return result;
}

/// Calculate the n-th discrete difference along the given axis.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1, 2, 4, 7, 0], [5], DType.int32);
/// final res = diff(a); // [1, 2, 3, -7]
/// ```
NDArray<T> diff<T>(NDArray<T> a, {int n = 1, int axis = -1, NDArray<T>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute diff() on a disposed array.');
  }
  if (n < 0) {
    throw ArgumentError('Order of difference n must be >= 0 (was $n).');
  }
  if (n == 0) {
    final result = out ?? a.copy();
    if (out != null) {
      for (var i = 0; i < result.data.length; i++) {
        result.data[i] = a.data[i];
      }
    }
    return result;
  }

  var targetAxis = axis;
  if (targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }
  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  if (n >= a.shape[targetAxis]) {
    final emptyShape = List<int>.from(a.shape);
    emptyShape[targetAxis] = 0;
    return out ?? NDArray<T>.create(emptyShape, a.dtype);
  }

  if (n > 1) {
    final step = diff(a, n: n - 1, axis: targetAxis);
    final result = diff(step, n: 1, axis: targetAxis, out: out);
    step.dispose();
    return result;
  }

  final targetShape = List<int>.from(a.shape);
  targetShape[targetAxis] = a.shape[targetAxis] - 1;

  final result = out ?? NDArray<T>.create(targetShape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final rank = a.shape.length;
  final marker = ScratchArena.marker;
  final cShape = ScratchArena.copyInts(a.shape);
  final cStridesA = ScratchArena.copyInts(a.strides);
  final cStridesRes = ScratchArena.copyInts(result.strides);
  try {
    final dtype = a.dtype;
    switch (dtype) {
      case DType.float64:
        s_diff_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.float32:
        s_diff_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.int64:
        s_diff_int64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.int32:
        s_diff_int32(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.complex128:
        s_diff_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.complex64:
        s_diff_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.uint8:
      case DType.int16:
      case DType.boolean:
        final doubleA = NDArray<double>.create(a.shape, DType.float64);
        unaryOp<dynamic, double>(
          doubleA.data,
          a.data,
          a.shape,
          a.strides,
          doubleA.strides,
          0,
          a.offsetElements,
          doubleA.offsetElements,
          (x) => (x as num).toDouble(),
        );
        final doubleRes = NDArray<double>.create(targetShape, DType.float64);
        final cStridesDoubleA = ScratchArena.copyInts(doubleA.strides);
        final cStridesDoubleRes = ScratchArena.copyInts(doubleRes.strides);

        s_diff_double(
          doubleA.pointer.cast(),
          cStridesDoubleA,
          doubleRes.pointer.cast(),
          cStridesDoubleRes,
          cShape,
          rank,
          targetAxis,
        );

        for (var i = 0; i < result.data.length; i++) {
          result.data[i] = castValue(doubleRes.data[i], a.dtype) as T;
        }
        doubleA.dispose();
        doubleRes.dispose();
    }
  } finally {
    ScratchArena.reset(marker);
  }

  return result;
}

/// Roll array elements along a given axis.
///
/// Elements that roll beyond the last position are re-introduced at the first.
///
/// **Preconditions:**
/// - The array [a] must not be disposed.
/// - [shift] must be an integer, or a list of integers if [axis] is a list of integers.
/// - [axis] must be null, an integer, or a list of integers.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if the shift/axis arguments are invalid or mismatched.
///
/// **Performance considerations:**
/// - Algorithmic Time Complexity is $O(N)$ where $N$ is the total number of elements,
///   as it creates a new array and copies elements.
/// - Space Complexity is $O(N)$ for the newly allocated output array.
///
/// **Reference:**
/// Refer to [NumPy roll documentation](https://numpy.org/doc/stable/reference/generated/numpy.roll.html).
///
/// {@example /example/rearranging_example.dart lang=dart}
NDArray<T> roll<T extends Object>(NDArray<T> a, dynamic shift, {dynamic axis}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute roll() on a disposed array.');
  }

  // Parse shifts and axes
  final List<int> shifts;
  final List<int>? axes;

  if (axis == null) {
    if (shift is int) {
      shifts = [shift];
      axes = null;
    } else if (shift is List<int>) {
      if (shift.length != 1) {
        throw ArgumentError('shift must be an integer when axis is null');
      }
      shifts = shift;
      axes = null;
    } else {
      throw ArgumentError('shift must be an integer or a list of integers');
    }
  } else if (axis is int) {
    if (shift is int) {
      shifts = [shift];
      axes = [axis];
    } else if (shift is List<int>) {
      if (shift.length != 1) {
        throw ArgumentError(
          'shift and axis must have the same number of elements',
        );
      }
      shifts = shift;
      axes = [axis];
    } else {
      throw ArgumentError('shift must be an integer or a list of integers');
    }
  } else if (axis is List<int>) {
    if (shift is int) {
      shifts = List<int>.filled(axis.length, shift);
      axes = axis;
    } else if (shift is List<int>) {
      if (shift.length != axis.length) {
        throw ArgumentError(
          'shift and axis must have the same number of elements',
        );
      }
      shifts = shift;
      axes = axis;
    } else {
      throw ArgumentError('shift must be an integer or a list of integers');
    }
  } else {
    throw ArgumentError('axis must be null, an integer, or a list of integers');
  }

  if (a.rank == 0) {
    return a.copy();
  }

  return NDArray.scope(() {
    NDArray<T> current = a;

    if (axes == null) {
      final flat = current.ravel();
      final s = shifts[0];
      final rolledFlat = _rollSingle1D(flat, s);
      final result = rolledFlat.reshape(a.shape);
      return result.detachToParentScope();
    } else {
      for (var i = 0; i < axes.length; i++) {
        current = _rollSingle(current, shifts[i], axes[i]);
      }
      return current.copy().detachToParentScope();
    }
  });
}

NDArray<T> _rollSingle1D<T extends Object>(NDArray<T> a, int shift) {
  if (a.isDisposed) {
    throw StateError('Cannot execute _rollSingle1D() on a disposed array.');
  }
  final size = a.size;
  if (size == 0) return a.copy();
  final s = shift % size;
  if (s == 0) return a.copy();

  final realShift = s < 0 ? size + s : s;

  final part1 = a.slice([Slice(start: size - realShift, stop: size)]);
  final part2 = a.slice([Slice(start: 0, stop: size - realShift)]);
  return concatenate([part1, part2], axis: 0);
}

NDArray<T> _rollSingle<T extends Object>(NDArray<T> a, int shift, int axis) {
  if (a.isDisposed) {
    throw StateError('Cannot execute _rollSingle() on a disposed array.');
  }
  final rank = a.rank;
  final normAx = axis < 0 ? rank + axis : axis;
  if (normAx < 0 || normAx >= rank) {
    throw RangeError.range(normAx, 0, rank - 1, 'axis');
  }

  final dimSize = a.shape[normAx];
  if (dimSize == 0) return a.copy();

  final s = shift % dimSize;
  if (s == 0) return a.copy();

  final realShift = s < 0 ? dimSize + s : s;

  final selectors1 = List<Selector>.generate(
    rank,
    (i) => i == normAx
        ? Slice(start: dimSize - realShift, stop: dimSize)
        : const Slice.all(),
  );
  final selectors2 = List<Selector>.generate(
    rank,
    (i) => i == normAx
        ? Slice(start: 0, stop: dimSize - realShift)
        : const Slice.all(),
  );

  final part1 = a.slice(selectors1);
  final part2 = a.slice(selectors2);

  return concatenate([part1, part2], axis: normAx);
}
