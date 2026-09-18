import '../ndarray.dart';
import '../ndarray_extensions_bindings.dart';
import '../scratch_arena.dart';

import 'helpers.dart';

/// Repeats elements of an array.
///
/// Repeats each element of an array [a] along the given [axis] a number of times
/// specified by [repeats].
///
/// If [axis] is null, [a] is flattened first.
///
/// **Preconditions:**
/// - If [axis] is specified, it must be within the range `[-rank, rank - 1]`.
/// - [repeats] must be an `int` or `List<int>`.
/// - Its length must match the size of the
///   dimension along [axis] (or be 1 / a scalar `int`).
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
/// **Performance considerations:**
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
NDArray<T> repeat<T>(
  NDArray<T> a,
  Object repeats, {
  int? axis,
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot access a disposed out NDArray.');
  }

  final List<int> rawRepeats;
  if (repeats is int) {
    rawRepeats = [repeats];
  } else if (repeats is List<int>) {
    rawRepeats = repeats;
  } else {
    throw ArgumentError('repeats must be an int or a List<int>');
  }

  return NDArray.scope(() {
    NDArray<T> src = a;
    int normAxis;

    if (axis == null) {
      src = a.flatten();
      normAxis = 0;
    } else {
      final rank = a.rank;
      if (axis < -rank || axis >= rank) {
        throw RangeError.range(axis, -rank, rank - 1, 'axis');
      }
      normAxis = axis < 0 ? rank + axis : axis;
      if (!a.isContiguous) {
        src = a.copy();
      }
    }

    List<int> repsList = rawRepeats;
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

    if (out != null) {
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
    }

    final bool useTempOut =
        out != null && (!out.isContiguous || sharesMemory(a, out));
    final NDArray<T> target = (out != null && !useTempOut)
        ? out
        : NDArray<T>.create(outputShape, src.dtype);

    if (target.size == 0) {
      if (out != null) {
        return out;
      }
      return target.detachToParentScope();
    }

    final outer = src.shape.sublist(0, normAxis).fold<int>(1, (x, y) => x * y);
    final dim = src.shape[normAxis];
    final inner = src.shape.sublist(normAxis + 1).fold<int>(1, (x, y) => x * y);

    final destDim = target.shape[normAxis];

    var destOffset = 0;
    for (var i = 0; i < dim; i++) {
      final rep = repsList[i];
      if (rep == 0) continue;

      for (var o = 0; o < outer; o++) {
        final srcStart = (o * dim + i) * inner;
        final destStart = (o * destDim + destOffset) * inner;

        final srcView = NDArray<T>.view(
          src,
          shape: [rep, inner],
          strides: [0, 1],
          offsetElements: srcStart,
        );

        final destView = NDArray<T>.view(
          target,
          shape: [rep, inner],
          strides: [inner, 1],
          offsetElements: destStart,
        );

        srcView.copy(out: destView);
      }
      destOffset += rep;
    }

    if (out != null) {
      if (useTempOut) {
        target.copy(out: out);
      }
      return out;
    }
    return target.detachToParentScope();
  });
}

/// Constructs an array by repeating [a] the number of times given by [reps].
///
/// If [reps] has length `d`, the result will have dimension of `max(d, a.rank)`.
/// If `a.rank < d`, [a] is promoted to be d-dimensional by prepending new axes.
/// If `a.rank > d`, [reps] is promoted to `a.rank` by pre-pending 1's to it.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - All values in [reps] must be non-negative ($\ge 0$).
/// - If [out] is provided, it must not be disposed and must have the correct shape and [DType] to store the result.
/// - It is an error if [a] is disposed or [out] is disposed.
/// - It is an error if [reps] contains negative values.
/// - It is an error if [out] shape or [DType] is incompatible.
///
/// **Performance considerations:**
/// - Time Complexity: $O(N)$ where $N$ is the total number of elements in the
///   output array.
/// - Contiguous blocks are replicated directly in unmanaged C memory using fast exponential doubling `memcpy`.
/// - Space Complexity: $O(N)$ for the output array (unless [out] is provided).
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1, 2], [2], DType.int32);
/// final t = tile(a, [2]);
/// print(t.toList()); // [1, 2, 1, 2]
/// ```
///
/// Refer to the [NumPy tile reference](https://numpy.org/doc/stable/reference/generated/numpy.tile.html)
/// for details.
NDArray<T> tile<T extends Object>(
  NDArray<T> a,
  List<int> reps, {
  NDArray<T>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot access a disposed NDArray.');
  }

  final List<int> rawReps = reps;

  final bool hasNegative = rawReps.any((x) => x < 0);
  if (hasNegative) {
    throw ArgumentError('reps values must be non-negative');
  }

  return NDArray.scope(() {
    NDArray<T> src = a;
    List<int> tileReps = List<int>.from(rawReps);

    // Align dimensions
    if (src.rank < tileReps.length) {
      final newShape = [
        ...List<int>.filled(tileReps.length - src.rank, 1),
        ...src.shape,
      ];
      src = src.reshape(newShape);
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

    if (out != null) {
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
    }

    final bool useTempOut =
        out != null && (!out.isContiguous || sharesMemory(a, out));
    final NDArray<T> target = (out != null && !useTempOut)
        ? out
        : NDArray<T>.create(outputShape, src.dtype);

    if (target.size == 0) {
      if (out != null) {
        return out;
      }
      return target.detachToParentScope();
    }

    final rank = src.rank;
    if (rank == 0) {
      final marker = ScratchArena.marker;
      try {
        final cSrcShape = ScratchArena.copyInt64s(const <int>[]);
        final cReps = ScratchArena.copyInt64s(const <int>[]);
        final cOutShape = ScratchArena.copyInt64s(const <int>[]);
        native_tile_contiguous(
          src.dtype.index,
          src.pointer.cast(),
          cSrcShape,
          cReps,
          target.pointer.cast(),
          cOutShape,
          0,
        );
      } finally {
        ScratchArena.reset(marker);
      }
      if (out != null) {
        if (useTempOut) {
          target.copy(out: out);
        }
        return out;
      }
      return target.detachToParentScope();
    }

    final marker = ScratchArena.marker;
    try {
      final cSrcShape = ScratchArena.copyInt64s(src.shape);
      final cReps = ScratchArena.copyInt64s(tileReps);
      final cOutShape = ScratchArena.copyInt64s(outputShape);

      if (src.isContiguous && target.isContiguous) {
        native_tile_contiguous(
          src.dtype.index,
          src.pointer.cast(),
          cSrcShape,
          cReps,
          target.pointer.cast(),
          cOutShape,
          rank,
        );
      } else if (target.isContiguous) {
        final contigSrc = src.copy();
        native_tile_contiguous(
          contigSrc.dtype.index,
          contigSrc.pointer.cast(),
          cSrcShape,
          cReps,
          target.pointer.cast(),
          cOutShape,
          rank,
        );
      } else {
        final cSrcStrides = ScratchArena.copyInt64s(src.strides);
        final cOutStrides = ScratchArena.copyInt64s(target.strides);
        native_tile_strided(
          src.dtype.index,
          src.pointer.cast(),
          cSrcShape,
          cSrcStrides,
          cReps,
          target.pointer.cast(),
          cOutShape,
          cOutStrides,
          rank,
        );
      }
    } finally {
      ScratchArena.reset(marker);
    }

    if (out != null) {
      if (useTempOut) {
        target.copy(out: out);
      }
      return out;
    }
    return target.detachToParentScope();
  });
}
