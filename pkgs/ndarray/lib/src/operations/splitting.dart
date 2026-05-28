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

/// Helper to slice an array along a specific axis between [start] and [stop].
NDArray<T> _sliceAlongAxis<T>(NDArray<T> a, int axis, int start, int stop) {
  final selectors = List<Selector>.filled(a.shape.length, Slice.all());
  selectors[axis] = Slice(start: start, stop: stop);
  return a.slice(selectors);
}

/// Splits an array into multiple sub-arrays of as equal size as possible.
///
/// The array is divided into [sections] along [axis]. If such a split is not possible,
/// the first $L \pmod{\text{sections}}$ sub-arrays will have size $L // \text{sections} + 1$,
/// and the rest will have size $L // \text{sections}$.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [sections] must be strictly positive ($\ge 1$).
/// - [axis] must be within bounds `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [sections] is less than or equal to 0.
/// - [RangeError] if [axis] is out of bounds.
///
/// **Memory Safety & zero-copy View Warning:**
/// > [!WARNING]
/// > This operation returns a list of **zero-copy metadata views** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned sub-arrays will **silently mutate the original array [a]**.
///
/// **Example:**
/// {@example /example/splitting_example.dart region=array_split lang=dart}
///
/// Refer to the [NumPy array_split reference](https://numpy.org/doc/stable/reference/generated/numpy.array_split.html)
/// for details.
List<NDArray<T>> array_split<T>(NDArray<T> a, int sections, {int axis = 0}) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  if (sections <= 0) {
    throw ArgumentError('Number of sections must be positive.');
  }

  final rank = a.rank;
  if (axis < -rank || axis >= rank) {
    throw RangeError.range(axis, -rank, rank - 1, 'axis');
  }
  final normAxis = axis < 0 ? rank + axis : axis;
  final L = a.shape[normAxis];

  final List<NDArray<T>> results = [];

  final S_0 = L ~/ sections;
  final rem = L % sections;
  var currentIdx = 0;

  for (var i = 0; i < sections; i++) {
    final size = i < rem ? S_0 + 1 : S_0;
    final start = currentIdx;
    final stop = currentIdx + size;
    currentIdx = stop;

    final sub = _sliceAlongAxis(a, normAxis, start, stop);
    sub.detachToParentScope();
    results.add(sub);
  }

  return results;
}

/// Splits an array along [axis] at the coordinate points specified in [indices].
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [axis] must be within bounds `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [RangeError] if [axis] is out of bounds.
///
/// **Memory Safety & zero-copy View Warning:**
/// > [!WARNING]
/// > This operation returns a list of **zero-copy metadata views** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned sub-arrays will **silently mutate the original array [a]**.
///
/// **Example:**
/// {@example /example/splitting_example.dart region=array_split lang=dart}
///
/// Refer to the [NumPy array_split reference](https://numpy.org/doc/stable/reference/generated/numpy.array_split.html)
/// for details.
List<NDArray<T>> array_split_at<T>(
  NDArray<T> a,
  List<int> indices, {
  int axis = 0,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }

  final rank = a.rank;
  if (axis < -rank || axis >= rank) {
    throw RangeError.range(axis, -rank, rank - 1, 'axis');
  }
  final normAxis = axis < 0 ? rank + axis : axis;
  final L = a.shape[normAxis];

  final List<NDArray<T>> results = [];

  final boundaries = <int>[0];
  for (var p in indices) {
    boundaries.add(p.clamp(0, L));
  }
  boundaries.add(L);

  for (var i = 0; i < boundaries.length - 1; i++) {
    final start = boundaries[i];
    final stop = boundaries[i + 1];
    final sub = _sliceAlongAxis(a, normAxis, start, stop);
    sub.detachToParentScope();
    results.add(sub);
  }

  return results;
}

/// Splits an array into multiple sub-arrays of strictly equal size.
///
/// The array is divided into [sections] along [axis]. Throws [ArgumentError]
/// if the split does not result in an equal division.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [sections] must be strictly positive ($\ge 1$).
/// - [axis] must be within bounds `[-rank, rank - 1]`.
/// - The dimension size along [axis] must be divisible by [sections].
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [RangeError] if [axis] is out of bounds.
/// - [ArgumentError] if [sections] is less than or equal to 0.
/// - [ArgumentError] if division is not equal.
///
/// **Memory Safety & zero-copy View Warning:**
/// > [!WARNING]
/// > This operation returns a list of **zero-copy metadata views** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned sub-arrays will **silently mutate the original array [a]**.
///
/// **Example:**
/// {@example /example/splitting_example.dart region=split lang=dart}
///
/// Refer to the [NumPy split reference](https://numpy.org/doc/stable/reference/generated/numpy.split.html)
/// for details.
List<NDArray<T>> split<T>(NDArray<T> a, int sections, {int axis = 0}) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }

  final rank = a.rank;
  if (axis < -rank || axis >= rank) {
    throw RangeError.range(axis, -rank, rank - 1, 'axis');
  }
  final normAxis = axis < 0 ? rank + axis : axis;
  final L = a.shape[normAxis];

  if (sections <= 0) {
    throw ArgumentError('Number of sections must be positive.');
  }
  if (L % sections != 0) {
    throw ArgumentError(
      'array split does not result in an equal division: '
      'dimension size along axis $normAxis is $L, which is not divisible by $sections.',
    );
  }

  return array_split(a, sections, axis: normAxis);
}

/// Splits an array along [axis] at the coordinate points specified in [indices].
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [axis] must be within bounds `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [RangeError] if [axis] is out of bounds.
///
/// **Memory Safety & zero-copy View Warning:**
/// > [!WARNING]
/// > This operation returns a list of **zero-copy metadata views** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned sub-arrays will **silently mutate the original array [a]**.
///
/// **Example:**
/// {@example /example/splitting_example.dart region=split lang=dart}
///
/// Refer to the [NumPy split reference](https://numpy.org/doc/stable/reference/generated/numpy.split.html)
/// for details.
List<NDArray<T>> split_at<T>(NDArray<T> a, List<int> indices, {int axis = 0}) {
  return array_split_at(a, indices, axis: axis);
}

/// Splits an array horizontally (column-wise) into [sections] equal sub-arrays.
///
/// Equivalent to [split] with `axis = 1` for arrays of rank $\ge 2$, and
/// `axis = 0` for 1D arrays.
///
/// **Preconditions:**
/// - [a] must not be disposed.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if the split is invalid or rank is $< 1$.
///
/// **Memory Safety & zero-copy View Warning:**
/// > [!WARNING]
/// > This operation returns a list of **zero-copy metadata views** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned sub-arrays will **silently mutate the original array [a]**.
///
/// **Example:**
/// {@example /example/splitting_example.dart region=hsplit lang=dart}
///
/// Refer to the [NumPy hsplit reference](https://numpy.org/doc/stable/reference/generated/numpy.hsplit.html)
/// for details.
List<NDArray<T>> hsplit<T>(NDArray<T> a, int sections) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  if (a.rank == 0) {
    throw ArgumentError('Cannot hsplit a 0D array.');
  }
  final axis = a.rank == 1 ? 0 : 1;
  return split(a, sections, axis: axis);
}

/// Splits an array horizontally (column-wise) at the coordinate points specified in [indices].
///
/// Equivalent to [split_at] with `axis = 1` for arrays of rank $\ge 2$, and
/// `axis = 0` for 1D arrays.
///
/// **Preconditions:**
/// - [a] must not be disposed.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if the split is invalid or rank is $< 1$.
///
/// **Memory Safety & zero-copy View Warning:**
/// > [!WARNING]
/// > This operation returns a list of **zero-copy metadata views** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned sub-arrays will **silently mutate the original array [a]**.
///
/// **Example:**
/// {@example /example/splitting_example.dart region=hsplit lang=dart}
///
/// Refer to the [NumPy hsplit reference](https://numpy.org/doc/stable/reference/generated/numpy.hsplit.html)
/// for details.
List<NDArray<T>> hsplit_at<T>(NDArray<T> a, List<int> indices) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  if (a.rank == 0) {
    throw ArgumentError('Cannot hsplit a 0D array.');
  }
  final axis = a.rank == 1 ? 0 : 1;
  return split_at(a, indices, axis: axis);
}

/// Splits an array vertically (row-wise) into [sections] equal sub-arrays.
///
/// Equivalent to [split] with `axis = 0`. Requires rank $\ge 2$.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [a] must have rank $\ge 2$.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [a] has rank $< 2$.
///
/// **Memory Safety & zero-copy View Warning:**
/// > [!WARNING]
/// > This operation returns a list of **zero-copy metadata views** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned sub-arrays will **silently mutate the original array [a]**.
///
/// **Example:**
/// {@example /example/splitting_example.dart region=vsplit lang=dart}
///
/// Refer to the [NumPy vsplit reference](https://numpy.org/doc/stable/reference/generated/numpy.vsplit.html)
/// for details.
List<NDArray<T>> vsplit<T>(NDArray<T> a, int sections) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  if (a.rank < 2) {
    throw ArgumentError('vsplit only supports arrays of rank >= 2.');
  }
  return split(a, sections, axis: 0);
}

/// Splits an array vertically (row-wise) at the coordinate points specified in [indices].
///
/// Equivalent to [split_at] with `axis = 0`. Requires rank $\ge 2$.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [a] must have rank $\ge 2$.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [a] has rank $< 2$.
///
/// **Memory Safety & zero-copy View Warning:**
/// > [!WARNING]
/// > This operation returns a list of **zero-copy metadata views** sharing the underlying unmanaged C heap memory page with the input array. Mutating elements inside the returned sub-arrays will **silently mutate the original array [a]**.
///
/// **Example:**
/// {@example /example/splitting_example.dart region=vsplit lang=dart}
///
/// Refer to the [NumPy vsplit reference](https://numpy.org/doc/stable/reference/generated/numpy.vsplit.html)
/// for details.
List<NDArray<T>> vsplit_at<T>(NDArray<T> a, List<int> indices) {
  if (a.isDisposed) {
    throw StateError('Cannot access a disposed NDArray.');
  }
  if (a.rank < 2) {
    throw ArgumentError('vsplit only supports arrays of rank >= 2.');
  }
  return split_at(a, indices, axis: 0);
}
