import '../ndarray.dart';
import '../ndarray_bindings.dart';
import 'helpers.dart';
import 'sorting.dart';

/// Find the unique elements of an array.
///
/// Returns the sorted unique elements of an array.
///
/// If [ar] is not 1D, it is flattened first.
///
/// **Preconditions:**
/// - [ar] must not be disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1, 2, 2, 3, 1], [5], DType.int32);
/// final u = unique(a); // [1, 2, 3]
/// ```
NDArray<T> unique<T extends Object>(
  NDArray<T> ar, {
  bool returnIndex = false,
  bool returnInverse = false,
  bool returnCounts = false,
}) {
  if (ar.isDisposed) {
    throw StateError('Cannot execute unique on a disposed array.');
  }

  if (returnIndex || returnInverse || returnCounts) {
    throw UnimplementedError(
      'Optional returns for unique are not yet implemented.',
    );
  }

  final flat = (ar.rank == 1 && ar.isContiguous) ? ar : ar.flatten();
  final dest = NDArray<T>.create(flat.shape, flat.dtype);

  final uniqueCount = ndarray_unique(
    flat.pointer.cast(),
    dest.pointer.cast(),
    flat.size,
    encodeDType(flat.dtype),
  );

  if (uniqueCount == 0) {
    dest.dispose();
    if (flat != ar) {
      flat.dispose();
    }
    return NDArray<T>.create([0], flat.dtype);
  }

  final view = dest.slice([Slice(start: 0, stop: uniqueCount)]);
  final result = view.copy();

  view.dispose();
  dest.dispose();

  if (flat != ar) {
    flat.dispose();
  }

  return result;
}

/// Find the intersection of two arrays.
///
/// Returns the sorted, unique values that are in both of the input arrays.
///
/// **Preconditions:**
/// - Both arrays must not be disposed.
/// - Both arrays must have the same dtype.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
/// final b = NDArray.fromList([2, 3, 4], [3], DType.int32);
/// final res = intersect1d(a, b); // [2, 3]
/// ```
NDArray<T> intersect1d<T extends Object>(
  NDArray<T> ar1,
  NDArray<T> ar2, {
  bool assumeUnique = false,
}) {
  if (ar1.isDisposed || ar2.isDisposed) {
    throw StateError('Cannot execute intersect1d on disposed array(s).');
  }

  if (ar1.dtype != ar2.dtype) {
    throw ArgumentError('Input arrays must have the same dtype.');
  }

  final flat1 = (ar1.rank == 1 && ar1.isContiguous) ? ar1 : ar1.flatten();
  final flat2 = (ar2.rank == 1 && ar2.isContiguous) ? ar2 : ar2.flatten();

  final NDArray<T> u1;
  final NDArray<T> u2;

  if (assumeUnique) {
    u1 = sort<T>(flat1);
    u2 = sort<T>(flat2);
  } else {
    u1 = unique(flat1);
    u2 = unique(flat2);
  }

  final maxDstSize = u1.size < u2.size ? u1.size : u2.size;

  if (maxDstSize == 0) {
    u1.dispose();
    u2.dispose();
    if (flat1 != ar1) flat1.dispose();
    if (flat2 != ar2) flat2.dispose();
    return NDArray<T>.create([0], ar1.dtype);
  }

  final dest = NDArray<T>.create([maxDstSize], ar1.dtype);

  final intersectionCount = ndarray_intersect1d(
    u1.pointer.cast(),
    u1.size,
    u2.pointer.cast(),
    u2.size,
    dest.pointer.cast(),
    encodeDType(ar1.dtype),
  );

  final NDArray<T> result;
  if (intersectionCount == 0) {
    result = NDArray<T>.create([0], ar1.dtype);
    dest.dispose();
  } else if (intersectionCount < maxDstSize) {
    final view = dest.slice([Slice(start: 0, stop: intersectionCount)]);
    result = view.copy();
    view.dispose();
    dest.dispose();
  } else {
    result = dest;
  }

  u1.dispose();
  u2.dispose();
  if (flat1 != ar1) flat1.dispose();
  if (flat2 != ar2) flat2.dispose();

  return result;
}

/// Find the set difference of two arrays.
///
/// Returns the unique values in [ar1] that are not in [ar2].
///
/// **Preconditions:**
/// - Both arrays must not be disposed.
/// - Both arrays must have the same dtype.
NDArray<T> setdiff1d<T extends Object>(
  NDArray<T> ar1,
  NDArray<T> ar2, {
  bool assumeUnique = false,
}) {
  if (ar1.isDisposed || ar2.isDisposed) {
    throw StateError('Cannot execute setdiff1d on disposed array(s).');
  }

  if (ar1.dtype != ar2.dtype) {
    throw ArgumentError('Input arrays must have the same dtype.');
  }

  final flat1 = (ar1.rank == 1 && ar1.isContiguous) ? ar1 : ar1.flatten();
  final flat2 = (ar2.rank == 1 && ar2.isContiguous) ? ar2 : ar2.flatten();

  final NDArray<T> u1;
  final NDArray<T> u2;

  if (assumeUnique) {
    u1 = sort<T>(flat1);
    u2 = sort<T>(flat2);
  } else {
    u1 = unique(flat1);
    u2 = unique(flat2);
  }

  final maxDstSize = u1.size;

  if (maxDstSize == 0) {
    u1.dispose();
    u2.dispose();
    if (flat1 != ar1) flat1.dispose();
    if (flat2 != ar2) flat2.dispose();
    return NDArray<T>.create([0], ar1.dtype);
  }

  final dest = NDArray<T>.create([maxDstSize], ar1.dtype);

  final diffCount = ndarray_setdiff1d(
    u1.pointer.cast(),
    u1.size,
    u2.pointer.cast(),
    u2.size,
    dest.pointer.cast(),
    encodeDType(ar1.dtype),
  );

  final NDArray<T> result;
  if (diffCount == 0) {
    result = NDArray<T>.create([0], ar1.dtype);
    dest.dispose();
  } else if (diffCount < maxDstSize) {
    final view = dest.slice([Slice(start: 0, stop: diffCount)]);
    result = view.copy();
    view.dispose();
    dest.dispose();
  } else {
    result = dest;
  }

  u1.dispose();
  u2.dispose();
  if (flat1 != ar1) flat1.dispose();
  if (flat2 != ar2) flat2.dispose();

  return result;
}

/// Find the set exclusive-or of two arrays.
///
/// Returns the sorted, unique values that are in only one (not both) of the input arrays.
///
/// **Preconditions:**
/// - Both arrays must not be disposed.
/// - Both arrays must have the same dtype.
NDArray<T> setxor1d<T extends Object>(
  NDArray<T> ar1,
  NDArray<T> ar2, {
  bool assumeUnique = false,
}) {
  if (ar1.isDisposed || ar2.isDisposed) {
    throw StateError('Cannot execute setxor1d on disposed array(s).');
  }

  if (ar1.dtype != ar2.dtype) {
    throw ArgumentError('Input arrays must have the same dtype.');
  }

  final flat1 = (ar1.rank == 1 && ar1.isContiguous) ? ar1 : ar1.flatten();
  final flat2 = (ar2.rank == 1 && ar2.isContiguous) ? ar2 : ar2.flatten();

  final NDArray<T> u1;
  final NDArray<T> u2;

  if (assumeUnique) {
    u1 = sort<T>(flat1);
    u2 = sort<T>(flat2);
  } else {
    u1 = unique(flat1);
    u2 = unique(flat2);
  }

  final maxDstSize = u1.size + u2.size;

  if (maxDstSize == 0) {
    u1.dispose();
    u2.dispose();
    if (flat1 != ar1) flat1.dispose();
    if (flat2 != ar2) flat2.dispose();
    return NDArray<T>.create([0], ar1.dtype);
  }

  final dest = NDArray<T>.create([maxDstSize], ar1.dtype);

  final xorCount = ndarray_setxor1d(
    u1.pointer.cast(),
    u1.size,
    u2.pointer.cast(),
    u2.size,
    dest.pointer.cast(),
    encodeDType(ar1.dtype),
  );

  final NDArray<T> result;
  if (xorCount == 0) {
    result = NDArray<T>.create([0], ar1.dtype);
    dest.dispose();
  } else if (xorCount < maxDstSize) {
    final view = dest.slice([Slice(start: 0, stop: xorCount)]);
    result = view.copy();
    view.dispose();
    dest.dispose();
  } else {
    result = dest;
  }

  u1.dispose();
  u2.dispose();
  if (flat1 != ar1) flat1.dispose();
  if (flat2 != ar2) flat2.dispose();

  return result;
}

/// Find the union of two arrays.
///
/// Returns the unique, sorted array of values that are in either of the two input arrays.
///
/// **Preconditions:**
/// - Both arrays must not be disposed.
/// - Both arrays must have the same dtype.
NDArray<T> union1d<T extends Object>(NDArray<T> ar1, NDArray<T> ar2) {
  if (ar1.isDisposed || ar2.isDisposed) {
    throw StateError('Cannot execute union1d on disposed array(s).');
  }

  if (ar1.dtype != ar2.dtype) {
    throw ArgumentError('Input arrays must have the same dtype.');
  }

  final flat1 = (ar1.rank == 1 && ar1.isContiguous) ? ar1 : ar1.flatten();
  final flat2 = (ar2.rank == 1 && ar2.isContiguous) ? ar2 : ar2.flatten();

  final u1 = unique(flat1);
  final u2 = unique(flat2);

  final maxDstSize = u1.size + u2.size;

  if (maxDstSize == 0) {
    u1.dispose();
    u2.dispose();
    if (flat1 != ar1) flat1.dispose();
    if (flat2 != ar2) flat2.dispose();
    return NDArray<T>.create([0], ar1.dtype);
  }

  final dest = NDArray<T>.create([maxDstSize], ar1.dtype);

  final unionCount = ndarray_union1d(
    u1.pointer.cast(),
    u1.size,
    u2.pointer.cast(),
    u2.size,
    dest.pointer.cast(),
    encodeDType(ar1.dtype),
  );

  final NDArray<T> result;
  if (unionCount == 0) {
    result = NDArray<T>.create([0], ar1.dtype);
    dest.dispose();
  } else if (unionCount < maxDstSize) {
    final view = dest.slice([Slice(start: 0, stop: unionCount)]);
    result = view.copy();
    view.dispose();
    dest.dispose();
  } else {
    result = dest;
  }

  u1.dispose();
  u2.dispose();
  if (flat1 != ar1) flat1.dispose();
  if (flat2 != ar2) flat2.dispose();

  return result;
}

/// Test whether each element of a 1D array is also present in a second array.
///
/// Returns a boolean array the same shape as [element] that is True where an element of [element] is in [testElements] and False otherwise.
///
/// **Preconditions:**
/// - Both arrays must not be disposed.
/// - Both arrays must have the same dtype.
NDArray<bool> isin<T extends Object>(
  NDArray<T> element,
  NDArray<T> testElements, {
  bool assumeUnique = false,
  bool invert = false,
}) {
  if (element.isDisposed || testElements.isDisposed) {
    throw StateError('Cannot execute isin on disposed array(s).');
  }

  if (element.dtype != testElements.dtype) {
    throw ArgumentError('Input arrays must have the same dtype.');
  }

  final flatTest = (testElements.rank == 1 && testElements.isContiguous)
      ? testElements
      : testElements.flatten();
  final NDArray<T> uTest;

  if (assumeUnique) {
    uTest = sort<T>(flatTest);
  } else {
    uTest = unique(flatTest);
  }

  final NDArray<T> contigElement;
  if (element.isContiguous) {
    contigElement = element;
  } else {
    contigElement = element.copy();
  }

  final dest = NDArray<bool>.create(element.shape, DType.boolean);

  ndarray_isin(
    contigElement.pointer.cast(),
    element.size,
    uTest.pointer.cast(),
    uTest.size,
    dest.pointer.cast(),
    encodeDType(element.dtype),
    invert ? 1 : 0,
  );

  if (contigElement != element) {
    contigElement.dispose();
  }
  uTest.dispose();
  if (flatTest != testElements) {
    flatTest.dispose();
  }

  return dest;
}
