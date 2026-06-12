import '../ndarray.dart';

// Standalone operational relative cross-imports

/// Repeats elements of an array.
///
/// Repeats each element of an array [a] along the given [axis] a number of times
/// specified by [repeats].
///
/// If [axis] is null, [a] is flattened first.
///
/// **Preconditions:**
/// - If [axis] is specified, it must be within the range `[-rank, rank - 1]`.
/// - [repeats] must be an `NDArray<int, Int64Marker>`.
/// - Its length must match the size of the
///   dimension along [axis].
/// - All values in [repeats] must be non-negative ($\ge 0$).
/// - If [out] is provided, it must have the correct shape and [DType] to store
///   the result.
///
/// **Throws:**
/// - [RangeError] if [axis] is out of bounds.
/// - [ArgumentError] if [repeats] length does not match
///   the dimension along [axis], or it contains negative values.
/// - [ArgumentError] if [out] shape or [DType] is incompatible.
///
/// **Performance Considerations:**
/// - Time Complexity: $O(N)$ where $N$ is the total number of elements in the
///   output array.
/// - Space Complexity: $O(N)$ for the output array (unless [out] is provided).
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1, 2], [2], DType.int32);
/// final r = repeat(a, [3]);
/// print(r.toList()); // [1, 1, 1, 2, 2, 2]
/// ```
NDArray<T, MT> repeat<T, MT extends Marker>(
  NDArray<T, MT> a,
  List<int> repeats, {
  int? axis,
  NDArray<T, MT>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }

  NDArray<T, MT> src = a;
  int normAxis;
  bool ownsSrc;

  if (axis == null) {
    src = a.flatten();
    normAxis = 0;
    ownsSrc = true;
  } else {
    final rank = a.rank;
    if (axis < -rank || axis >= rank) {
      throw RangeError.range(axis, -rank, rank - 1, 'axis');
    }
    normAxis = axis < 0 ? rank + axis : axis;
    if (!a.isContiguous) {
      src = a.copy();
      ownsSrc = true;
    } else {
      ownsSrc = false;
    }
  }

  try {
    List<int> repsList = repeats;
    if (repsList.length == 1) {
      repsList = List<int>.filled(src.shape[normAxis], repsList[0]);
    }

    if (repsList.length != src.shape[normAxis]) {
      throw ArgumentError(
        'repeats length (${repsList.length}) must match the dimension along axis ($normAxis) which is ${src.shape[normAxis]}',
      );
    }

    final bool hasNegative = repsList.any((x) => x < 0);
    if (hasNegative) {
      throw ArgumentError('repeats values must be non-negative');
    }

    final outputShape = List<int>.from(src.shape);
    final newDimSize = repsList.isEmpty ? 0 : repsList.reduce((x, y) => x + y);
    outputShape[normAxis] = newDimSize;

    final NDArray<T, MT> result;
    if (out != null) {
      if (out.isDisposed) {
        throw StateError('Cannot access a disposed out NDArray.');
      }
      if (out.dtype != src.dtype) {
        throw ArgumentError('out buffer must have the same dtype as input');
      }
      if (out.shape.length != outputShape.length) {
        throw ArgumentError('out buffer shape length must match output shape');
      }
      for (var i = 0; i < outputShape.length; i++) {
        if (out.shape[i] != outputShape[i]) {
          throw ArgumentError('out buffer shape must match output shape');
        }
      }
      result = out;
    } else {
      result = NDArray.create(outputShape, src.dtype);
    }

    if (result.size == 0) {
      return result;
    }

    final outer = src.shape.sublist(0, normAxis).fold<int>(1, (a, b) => a * b);
    final dim = src.shape[normAxis];
    final inner = src.shape.sublist(normAxis + 1).fold<int>(1, (a, b) => a * b);

    final destDim = result.shape[normAxis];

    var destOffset = 0;
    for (var i = 0; i < dim; i++) {
      final rep = repsList[i];
      if (rep == 0) continue;

      for (var o = 0; o < outer; o++) {
        final srcStart = (o * dim + i) * inner;
        final destStart = (o * destDim + destOffset) * inner;

        final srcView = NDArray.view(
          src,
          shape: [rep, inner],
          strides: [0, 1],
          offsetElements: srcStart,
        );

        final destView = NDArray.view(
          result,
          shape: [rep, inner],
          strides: [inner, 1],
          offsetElements: destStart,
        );

        srcView.copy(out: destView);
      }
      destOffset += rep;
    }

    return result;
  } finally {
    if (ownsSrc && !identical(src, a)) {
      src.dispose();
    }
  }
}

/// Constructs an array by repeating [a] the number of times given by [reps].
///
/// If [reps] has length `d`, the result will have dimension of `max(d, a.ndim)`.
/// If `a.ndim < d`, [a] is promoted to be d-dimensional by prepending new axes.
/// If `a.ndim > d`, [reps] is promoted to `a.ndim` by pre-pending 1's to it.
///
/// **Preconditions:**
/// - [reps] must be an `NDArray<int, Int64Marker>`.
/// - All values in [reps] must be non-negative ($\ge 0$).
/// - If [out] is provided, it must have the correct shape and [DType] to store
///   the result.
///
/// **Throws:**
/// - [ArgumentError] if [reps] contains negative values.
/// - [ArgumentError] if [out] shape or [DType] is incompatible.
///
/// **Performance Considerations:**
/// - Time Complexity: $O(N)$ where $N$ is the total number of elements in the
///   output array.
/// - Space Complexity: $O(N)$ for the output array (unless [out] is provided).
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1, 2], [2], DType.int32);
/// final t = tile(a, [2]);
/// print(t.toList()); // [1, 2, 1, 2]
/// ```
NDArray<T, MT> tile<T, MT extends Marker>(
  NDArray<T, MT> a,
  List<int> reps, {
  NDArray<T, MT>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }

  final bool hasNegative = reps.any((x) => x < 0);
  if (hasNegative) {
    throw ArgumentError('reps values must be non-negative');
  }

  NDArray<T, MT> src = a;
  bool ownsSrc = false;
  List<int> tileReps = List<int>.from(reps);

  try {
    // Align dimensions
    if (src.rank < tileReps.length) {
      final newShape = [
        ...List<int>.filled(tileReps.length - src.rank, 1),
        ...src.shape,
      ];
      src = src.reshape(newShape);
      ownsSrc = !identical(src, a);
    } else if (src.rank > tileReps.length) {
      tileReps = [
        ...List<int>.filled(src.rank - tileReps.length, 1),
        ...tileReps,
      ];
    }

    final outputShape = List<int>.filled(src.rank, 0);
    for (var i = 0; i < src.rank; i++) {
      outputShape[i] = src.shape[i] * tileReps[i];
    }

    final NDArray<T, MT> result;
    if (out != null) {
      if (out.isDisposed) {
        throw StateError('Cannot access a disposed out NDArray.');
      }
      if (out.dtype != src.dtype) {
        throw ArgumentError('out buffer must have the same dtype as input');
      }
      if (out.shape.length != outputShape.length) {
        throw ArgumentError('out buffer shape length must match output shape');
      }
      for (var i = 0; i < outputShape.length; i++) {
        if (out.shape[i] != outputShape[i]) {
          throw ArgumentError('out buffer shape must match output shape');
        }
      }
      result = out;
    } else {
      result = NDArray.create(outputShape, src.dtype);
    }

    if (result.size == 0) {
      return result;
    }

    final rank = src.rank;
    final viewShape = <int>[];
    final srcStrides = <int>[];

    for (var i = 0; i < rank; i++) {
      viewShape.add(tileReps[i]);
      viewShape.add(src.shape[i]);
      srcStrides.add(0);
      srcStrides.add(src.strides[i]);
    }

    final srcView = NDArray.view(
      src,
      shape: viewShape,
      strides: srcStrides,
      offsetElements: 0,
    );

    final destView = NDArray.view(
      result,
      shape: viewShape,
      strides: NDArray.computeCStrides(viewShape),
      offsetElements: 0,
    );

    srcView.copy(out: destView);

    return result;
  } finally {
    if (ownsSrc && !identical(src, a)) {
      src.dispose();
    }
  }
}
