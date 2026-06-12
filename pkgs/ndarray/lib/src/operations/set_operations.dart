import 'dart:ffi' as ffi;
import '../ndarray.dart';
import '../ndarray_bindings.dart';
import 'helpers.dart'; // for encodeDType
import 'sorting.dart';

/// Find the unique elements of an array.
///
/// Returns the sorted unique elements of an array.
///
/// If [ar] is not 1D, it is flattened first.
dynamic unique<T extends Object, MT extends Marker>(
  NDArray<T, MT> ar, {
  bool returnIndex = false,
  bool returnInverse = false,
  bool returnCounts = false,
}) {
  if (ar.isDisposed) {
    throw StateError('Cannot execute unique on a disposed array.');
  }

  final flat = (ar.rank == 1 && ar.isContiguous) ? ar : ar.flatten();
  final dest = NDArray.create(flat.shape, flat.dtype);

  NDArray<int, Int64Marker>? outIndex;
  NDArray<int, Int64Marker>? outInverse;
  NDArray<int, Int64Marker>? outCounts;

  if (returnIndex) {
    outIndex = NDArray.create([flat.size], DType.int64);
  }
  if (returnInverse) {
    outInverse = NDArray.create([flat.size], DType.int64);
  }
  if (returnCounts) {
    outCounts = NDArray.create([flat.size], DType.int64);
  }

  final pIndex = outIndex != null
      ? outIndex.pointer.cast<ffi.Int64>()
      : ffi.Pointer<ffi.Int64>.fromAddress(0);
  final pInverse = outInverse != null
      ? outInverse.pointer.cast<ffi.Int64>()
      : ffi.Pointer<ffi.Int64>.fromAddress(0);
  final pCounts = outCounts != null
      ? outCounts.pointer.cast<ffi.Int64>()
      : ffi.Pointer<ffi.Int64>.fromAddress(0);

  final uniqueCount = ndarray_unique(
    flat.pointer.cast(),
    dest.pointer.cast(),
    flat.size,
    encodeDType(flat.dtype),
    pIndex,
    pInverse,
    pCounts,
  );

  if (uniqueCount == 0) {
    dest.dispose();
    outIndex?.dispose();
    outInverse?.dispose();
    outCounts?.dispose();
    final empty = NDArray.create([0], flat.dtype);
    if (flat != ar) flat.dispose();

    if (returnIndex || returnInverse || returnCounts) {
      return (
        empty,
        index: returnIndex ? NDArray.create([0], DType.int64) : null,
        inverse: returnInverse ? NDArray.create([0], DType.int64) : null,
        counts: returnCounts ? NDArray.create([0], DType.int64) : null,
      );
    }
    return empty;
  }

  final view = dest.slice([Slice(start: 0, stop: uniqueCount)]);
  final result = view.copy();
  view.dispose();
  dest.dispose();

  NDArray<int, Int64Marker>? indexResult;
  if (outIndex != null) {
    final v = outIndex.slice([Slice(start: 0, stop: uniqueCount)]);
    indexResult = v.copy();
    v.dispose();
    outIndex.dispose();
  }

  NDArray<int, Int64Marker>? inverseResult;
  if (outInverse != null) {
    inverseResult = outInverse;
  }

  NDArray<int, Int64Marker>? countsResult;
  if (outCounts != null) {
    final v = outCounts.slice([Slice(start: 0, stop: uniqueCount)]);
    countsResult = v.copy();
    v.dispose();
    outCounts.dispose();
  }

  if (flat != ar) {
    flat.dispose();
  }

  if (returnIndex || returnInverse || returnCounts) {
    return (
      result,
      index: indexResult,
      inverse: inverseResult,
      counts: countsResult,
    );
  }

  return result;
}

/// Find the intersection of two arrays.
///
/// Returns the sorted, unique values that are in both of the input arrays.
NDArray<T, MT> intersect1d<T extends Object, MT extends Marker>(
  NDArray<T, MT> ar1,
  NDArray<T, MT> ar2, {
  bool assumeUnique = false,
}) {
  if (ar1.isDisposed || ar2.isDisposed) {
    throw StateError('Cannot execute intersect1d on disposed array(s).');
  }

  final flat1 = (ar1.rank == 1 && ar1.isContiguous) ? ar1 : ar1.flatten();
  final flat2 = (ar2.rank == 1 && ar2.isContiguous) ? ar2 : ar2.flatten();

  final NDArray<T, MT> u1;
  final NDArray<T, MT> u2;

  if (assumeUnique) {
    u1 = sort(flat1);
    u2 = sort(flat2);
  } else {
    u1 = unique(flat1);
    u2 = unique(flat2);
  }

  final maxDstSize = u1.size < u2.size ? u1.size : u2.size;

  if (maxDstSize == 0) {
    if (!assumeUnique) {
      if (u1 != flat1) u1.dispose();
      if (u2 != flat2) u2.dispose();
    }
    if (ar1.rank != 1) flat1.dispose();
    if (ar2.rank != 1) flat2.dispose();
    return NDArray.create([0], ar1.dtype);
  }

  final dest = NDArray.create([maxDstSize], ar1.dtype);

  final intersectionCount = ndarray_intersect1d(
    u1.pointer.cast(),
    u1.size,
    u2.pointer.cast(),
    u2.size,
    dest.pointer.cast(),
    encodeDType(ar1.dtype),
  );

  final NDArray<T, MT> result;
  if (intersectionCount == 0) {
    result = NDArray.create([0], ar1.dtype);
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
NDArray<T, MT> setdiff1d<T extends Object, MT extends Marker>(
  NDArray<T, MT> ar1,
  NDArray<T, MT> ar2, {
  bool assumeUnique = false,
}) {
  if (ar1.isDisposed || ar2.isDisposed) {
    throw StateError('Cannot execute setdiff1d on disposed array(s).');
  }

  final flat1 = (ar1.rank == 1 && ar1.isContiguous) ? ar1 : ar1.flatten();
  final flat2 = (ar2.rank == 1 && ar2.isContiguous) ? ar2 : ar2.flatten();

  final NDArray<T, MT> u1;
  final NDArray<T, MT> u2;

  if (assumeUnique) {
    u1 = sort(flat1);
    u2 = sort(flat2);
  } else {
    u1 = unique(flat1);
    u2 = unique(flat2);
  }

  final maxDstSize = u1.size;

  if (maxDstSize == 0) {
    if (!assumeUnique) {
      if (u1 != flat1) u1.dispose();
      if (u2 != flat2) u2.dispose();
    }
    if (ar1.rank != 1) flat1.dispose();
    if (ar2.rank != 1) flat2.dispose();
    return NDArray.create([0], ar1.dtype);
  }

  final dest = NDArray.create([maxDstSize], ar1.dtype);

  final diffCount = ndarray_setdiff1d(
    u1.pointer.cast(),
    u1.size,
    u2.pointer.cast(),
    u2.size,
    dest.pointer.cast(),
    encodeDType(ar1.dtype),
  );

  final NDArray<T, MT> result;
  if (diffCount == 0) {
    result = NDArray.create([0], ar1.dtype);
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
NDArray<T, MT> setxor1d<T extends Object, MT extends Marker>(
  NDArray<T, MT> ar1,
  NDArray<T, MT> ar2, {
  bool assumeUnique = false,
}) {
  if (ar1.isDisposed || ar2.isDisposed) {
    throw StateError('Cannot execute setxor1d on disposed array(s).');
  }

  final flat1 = (ar1.rank == 1 && ar1.isContiguous) ? ar1 : ar1.flatten();
  final flat2 = (ar2.rank == 1 && ar2.isContiguous) ? ar2 : ar2.flatten();

  final NDArray<T, MT> u1;
  final NDArray<T, MT> u2;

  if (assumeUnique) {
    u1 = sort(flat1);
    u2 = sort(flat2);
  } else {
    u1 = unique(flat1);
    u2 = unique(flat2);
  }

  final maxDstSize = u1.size + u2.size;

  if (maxDstSize == 0) {
    if (!assumeUnique) {
      if (u1 != flat1) u1.dispose();
      if (u2 != flat2) u2.dispose();
    }
    if (ar1.rank != 1) flat1.dispose();
    if (ar2.rank != 1) flat2.dispose();
    return NDArray.create([0], ar1.dtype);
  }

  final dest = NDArray.create([maxDstSize], ar1.dtype);

  final xorCount = ndarray_setxor1d(
    u1.pointer.cast(),
    u1.size,
    u2.pointer.cast(),
    u2.size,
    dest.pointer.cast(),
    encodeDType(ar1.dtype),
  );

  final NDArray<T, MT> result;
  if (xorCount == 0) {
    result = NDArray.create([0], ar1.dtype);
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
NDArray<T, MT> union1d<T extends Object, MT extends Marker>(
  NDArray<T, MT> ar1,
  NDArray<T, MT> ar2,
) {
  if (ar1.isDisposed || ar2.isDisposed) {
    throw StateError('Cannot execute union1d on disposed array(s).');
  }

  final flat1 = (ar1.rank == 1 && ar1.isContiguous) ? ar1 : ar1.flatten();
  final flat2 = (ar2.rank == 1 && ar2.isContiguous) ? ar2 : ar2.flatten();

  final u1 = unique(flat1);
  final u2 = unique(flat2);

  final maxDstSize = u1.size + u2.size;

  if (maxDstSize == 0) {
    u1.dispose();
    u2.dispose();
    if (ar1.rank != 1) flat1.dispose();
    if (ar2.rank != 1) flat2.dispose();
    return NDArray.create([0], ar1.dtype);
  }

  final dest = NDArray.create([maxDstSize], ar1.dtype);

  final unionCount = ndarray_union1d(
    u1.pointer.cast(),
    u1.size,
    u2.pointer.cast(),
    u2.size,
    dest.pointer.cast(),
    encodeDType(ar1.dtype),
  );

  final NDArray<T, MT> result;
  if (unionCount == 0) {
    result = NDArray.create([0], ar1.dtype);
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
NDArray<bool, BooleanMarker> isin<T extends Object, MT extends Marker>(
  NDArray<T, MT> element,
  NDArray<T, MT> testElements, {
  bool assumeUnique = false,
  bool invert = false,
}) {
  if (element.isDisposed || testElements.isDisposed) {
    throw StateError('Cannot execute isin on disposed array(s).');
  }

  final flatTest = (testElements.rank == 1 && testElements.isContiguous)
      ? testElements
      : testElements.flatten();
  final NDArray<T, MT> uTest;

  if (assumeUnique) {
    uTest = sort(flatTest);
  } else {
    uTest = unique(flatTest);
  }

  final NDArray<T, MT> contigElement;
  if (element.isContiguous) {
    contigElement = element;
  } else {
    contigElement = element.copy();
  }

  final dest = NDArray.create(element.shape, DType.boolean);

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
