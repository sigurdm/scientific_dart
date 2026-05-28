// ignore_for_file: non_constant_identifier_names
import 'dart:typed_data';
import 'dart:math' as math;
import 'dart:math' show Random;
import 'dart:io';
import 'package:archive/archive.dart';
import 'package:pocketfft/pocketfft.dart';
import '../ndarray.dart';
import 'package:openblas/openblas.dart';
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import '../ndarray_bindings.dart';
import '../scratch_arena.dart';

// Standalone operational relative cross-imports
import 'math.dart';
import 'stats.dart';
import 'sorting.dart';
import 'linalg.dart';
import 'spacers.dart';
import 'manipulation.dart';
import 'broadcasting.dart';
import 'splitting.dart';
import 'shaping_meshes.dart';
import 'repeating_tiling.dart';
import 'io.dart';
import 'random.dart';
import 'fft.dart';
import 'calculus.dart';
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
/// - [repeats] must be an [int], [List<int>], or `NDArray<int>`.
/// - If [repeats] is a list or array, its length must match the size of the
///   dimension along [axis].
/// - All values in [repeats] must be non-negative ($\ge 0$).
/// - If [out] is provided, it must have the correct shape and [DType] to store
///   the result.
///
/// **Throws:**
/// - [RangeError] if [axis] is out of bounds.
/// - [ArgumentError] if [repeats] type is invalid, or its length does not match
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
/// final r = repeat(a, 3);
/// print(r.toList()); // [1, 1, 1, 2, 2, 2]
/// ```
NDArray<T> repeat<T>(
  NDArray<T> a,
  dynamic repeats, {
  int? axis,
  NDArray<T>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }

  NDArray<T> src = a;
  int normAxis;
  final bool ownsSrc;

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
    ownsSrc = false;
  }

  try {
    List<int> repsList;
    if (repeats is int) {
      if (repeats < 0) {
        throw ArgumentError('repeats must be non-negative');
      }
      repsList = List<int>.filled(src.shape[normAxis], repeats);
    } else if (repeats is List<int>) {
      repsList = repeats;
    } else if (repeats is NDArray<int>) {
      repsList = repeats.toList();
    } else {
      throw ArgumentError('repeats must be int, List<int>, or NDArray<int>');
    }

    if (repsList.length != src.shape[normAxis]) {
      throw ArgumentError(
        'repeats length (${repsList.length}) must match the dimension along axis ($normAxis) which is ${src.shape[normAxis]}',
      );
    }

    for (var i = 0; i < repsList.length; i++) {
      if (repsList[i] < 0) {
        throw ArgumentError('repeats values must be non-negative');
      }
    }

    final outputShape = List<int>.from(src.shape);
    final newDimSize = repsList.isEmpty ? 0 : repsList.reduce((x, y) => x + y);
    outputShape[normAxis] = newDimSize;

    final NDArray<T> result;
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
      result = NDArray<T>.create(outputShape, src.dtype);
    }

    if (result.size == 0) {
      return result;
    }

    _copyRepeat(src, result, repsList, normAxis);

    return result;
  } finally {
    if (ownsSrc && !identical(src, a)) {
      src.dispose();
    }
  }
}

void _copyRepeat<T>(
  NDArray<T> src,
  NDArray<T> dest,
  List<int> repeats,
  int axis,
) {
  final rank = src.rank;
  final srcCoords = List<int>.filled(rank, 0);
  final destCoords = List<int>.filled(rank, 0);

  void walk(int dim) {
    if (dim == rank) {
      dest.setCell(destCoords, src.getCell(srcCoords));
      return;
    }

    if (dim == axis) {
      var destIdx = 0;
      for (var srcIdx = 0; srcIdx < src.shape[dim]; srcIdx++) {
        final rep = repeats[srcIdx];
        srcCoords[dim] = srcIdx;
        for (var r = 0; r < rep; r++) {
          destCoords[dim] = destIdx + r;
          walk(dim + 1);
        }
        destIdx += rep;
      }
    } else {
      for (var i = 0; i < src.shape[dim]; i++) {
        srcCoords[dim] = i;
        destCoords[dim] = i;
        walk(dim + 1);
      }
    }
  }

  walk(0);
}

/// Constructs an array by repeating [a] the number of times given by [reps].
///
/// If [reps] has length `d`, the result will have dimension of `max(d, a.ndim)`.
/// If `a.ndim < d`, [a] is promoted to be d-dimensional by prepending new axes.
/// If `a.ndim > d`, [reps] is promoted to `a.ndim` by pre-pending 1's to it.
///
/// **Preconditions:**
/// - [reps] must be an [int], [List<int>], or `NDArray<int>`.
/// - All values in [reps] must be non-negative ($\ge 0$).
/// - If [out] is provided, it must have the correct shape and [DType] to store
///   the result.
///
/// **Throws:**
/// - [ArgumentError] if [reps] type is invalid, or contains negative values.
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
/// final t = tile(a, 2);
/// print(t.toList()); // [1, 2, 1, 2]
/// ```
NDArray<T> tile<T>(NDArray<T> a, dynamic reps, {NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }

  List<int> repsList;
  if (reps is int) {
    if (reps < 0) {
      throw ArgumentError('reps must be non-negative');
    }
    repsList = [reps];
  } else if (reps is List<int>) {
    repsList = reps;
  } else if (reps is NDArray<int>) {
    repsList = reps.toList();
  } else {
    throw ArgumentError('reps must be int, List<int>, or NDArray<int>');
  }

  for (var i = 0; i < repsList.length; i++) {
    if (repsList[i] < 0) {
      throw ArgumentError('reps values must be non-negative');
    }
  }

  NDArray<T> src = a;
  bool ownsSrc = false;
  List<int> tileReps = List<int>.from(repsList);

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

    final NDArray<T> result;
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
      result = NDArray<T>.create(outputShape, src.dtype);
    }

    if (result.size == 0) {
      return result;
    }

    _copyTile(src, result);

    return result;
  } finally {
    if (ownsSrc && !identical(src, a)) {
      src.dispose();
    }
  }
}

void _copyTile<T>(NDArray<T> src, NDArray<T> dest) {
  final rank = src.rank;
  final destCoords = List<int>.filled(rank, 0);
  final srcCoords = List<int>.filled(rank, 0);

  void walk(int dim) {
    if (dim == rank) {
      dest.setCell(destCoords, src.getCell(srcCoords));
      return;
    }

    final srcDimSize = src.shape[dim];
    for (var i = 0; i < dest.shape[dim]; i++) {
      destCoords[dim] = i;
      srcCoords[dim] = i % srcDimSize;
      walk(dim + 1);
    }
  }

  walk(0);
}
