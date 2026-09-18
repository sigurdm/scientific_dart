import 'dart:ffi' as ffi;
import '../ndarray.dart';
import '../ndarray_bindings.dart';
import 'helpers.dart';
import 'sorting.dart';

/// Finds the unique elements of an array.
///
/// Returns the sorted unique elements of an array.
///
/// If [ar] is not 1D, it is flattened first.
///
/// It is an error if [ar] has an unsupported dtype.
///
/// It is an error if [ar] is disposed.
dynamic unique<T extends Object>(
  NDArray<T> ar, {
  bool returnIndex = false,
  bool returnInverse = false,
  bool returnCounts = false,
  NDArray<T>? out,
}) {
  if (ar.isDisposed) {
    throw StateError('Cannot execute unique on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write unique result to a disposed output array.');
  }
  if (out != null && out.dtype != ar.dtype) {
    throw ArgumentError('Incompatible out buffer dtype.');
  }

  return NDArray.scope(() {
    final flat = (ar.rank == 1 && ar.isContiguous) ? ar : ar.flatten();
    final dest = NDArray<T>.create(flat.shape, flat.dtype);
    final outIndex = returnIndex
        ? NDArray<int>.create([flat.size], DType.int64)
        : null;
    final outInverse = returnInverse
        ? NDArray<int>.create([flat.size], DType.int64)
        : null;
    final outCounts = returnCounts
        ? NDArray<int>.create([flat.size], DType.int64)
        : null;

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
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError('Incompatible out buffer shape.');
      }
      final empty =
          out ?? (NDArray<T>.create([0], flat.dtype)..detachToParentScope());

      if (returnIndex || returnInverse || returnCounts) {
        return (
          values: empty,
          index: returnIndex
              ? (NDArray<int>.create([0], DType.int64)..detachToParentScope())
              : null,
          inverse: returnInverse
              ? (NDArray<int>.create([0], DType.int64)..detachToParentScope())
              : null,
          counts: returnCounts
              ? (NDArray<int>.create([0], DType.int64)..detachToParentScope())
              : null,
        );
      }
      return empty;
    }

    if (out != null && !listEquals(out.shape, [uniqueCount])) {
      throw ArgumentError('Incompatible out buffer shape.');
    }

    final validView = dest.slice([Slice(start: 0, stop: uniqueCount)]);
    final NDArray<T> result;
    if (out != null) {
      validView.copy(out: out);
      result = out;
    } else {
      result = validView.copy()..detachToParentScope();
    }

    NDArray<int>? indexResult;
    if (outIndex != null) {
      indexResult = outIndex.slice([Slice(start: 0, stop: uniqueCount)]).copy()
        ..detachToParentScope();
    }

    NDArray<int>? inverseResult;
    if (outInverse != null) {
      inverseResult = outInverse.copy()..detachToParentScope();
    }

    NDArray<int>? countsResult;
    if (outCounts != null) {
      countsResult = outCounts.slice([
        Slice(start: 0, stop: uniqueCount),
      ]).copy()..detachToParentScope();
    }

    if (returnIndex || returnInverse || returnCounts) {
      return (
        values: result,
        index: indexResult,
        inverse: inverseResult,
        counts: countsResult,
      );
    }

    return result;
  });
}

/// Finds the intersection of two arrays.
///
/// Returns the sorted, unique values that are in both of the input arrays.
///
/// It is an error if [ar1] or [ar2] is disposed.
NDArray<T> intersect1d<T extends Object>(
  NDArray<T> ar1,
  NDArray<T> ar2, {
  bool assumeUnique = false,
  NDArray<T>? out,
}) {
  if (ar1.isDisposed || ar2.isDisposed) {
    throw StateError('Cannot execute intersect1d on disposed array(s).');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write intersect1d result to a disposed output array.',
    );
  }
  final DType<T> commonDType =
      (ar1.dtype == ar2.dtype ? ar1.dtype : resolveDType(ar1.dtype, ar2.dtype))
          as DType<T>;
  if (out != null && out.dtype != commonDType) {
    throw ArgumentError('Incompatible out buffer dtype.');
  }

  return NDArray.scope(() {
    final NDArray<T> c1 = ar1.dtype == commonDType
        ? ar1
        : castNDArray<T>(ar1, commonDType);
    final NDArray<T> c2 = ar2.dtype == commonDType
        ? ar2
        : castNDArray<T>(ar2, commonDType);
    final NDArray<T> flat1 = (c1.rank == 1 && c1.isContiguous)
        ? c1
        : c1.flatten();
    final NDArray<T> flat2 = (c2.rank == 1 && c2.isContiguous)
        ? c2
        : c2.flatten();

    final NDArray u1 = assumeUnique ? sort(flat1) : unique(flat1) as NDArray;
    final NDArray u2 = assumeUnique ? sort(flat2) : unique(flat2) as NDArray;

    final maxDstSize = u1.size < u2.size ? u1.size : u2.size;

    if (maxDstSize == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError('Incompatible out buffer shape.');
      }
      return out ??
          (NDArray<T>.create([0], commonDType)..detachToParentScope());
    }

    final dest = NDArray<T>.create([maxDstSize], commonDType);

    final intersectionCount = ndarray_intersect1d(
      u1.pointer.cast(),
      u1.size,
      u2.pointer.cast(),
      u2.size,
      dest.pointer.cast(),
      encodeDType(commonDType),
    );

    if (intersectionCount == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError('Incompatible out buffer shape.');
      }
      return out ??
          (NDArray<T>.create([0], commonDType)..detachToParentScope());
    }

    if (out != null && !listEquals(out.shape, [intersectionCount])) {
      throw ArgumentError('Incompatible out buffer shape.');
    }

    final validView = dest.slice([Slice(start: 0, stop: intersectionCount)]);
    if (out != null) {
      validView.copy(out: out);
      return out;
    } else {
      return validView.copy()..detachToParentScope();
    }
  });
}

/// Finds the set difference of two arrays.
///
/// Returns the unique values in [ar1] that are not in [ar2].
///
/// It is an error if [ar1] or [ar2] is disposed.
NDArray<T> setdiff1d<T extends Object>(
  NDArray<T> ar1,
  NDArray<T> ar2, {
  bool assumeUnique = false,
  NDArray<T>? out,
}) {
  if (ar1.isDisposed || ar2.isDisposed) {
    throw StateError('Cannot execute setdiff1d on disposed array(s).');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write setdiff1d result to a disposed output array.',
    );
  }
  final DType<T> commonDType =
      (ar1.dtype == ar2.dtype ? ar1.dtype : resolveDType(ar1.dtype, ar2.dtype))
          as DType<T>;
  if (out != null && out.dtype != commonDType) {
    throw ArgumentError('Incompatible out buffer dtype.');
  }

  return NDArray.scope(() {
    final NDArray<T> c1 = ar1.dtype == commonDType
        ? ar1
        : castNDArray<T>(ar1, commonDType);
    final NDArray<T> c2 = ar2.dtype == commonDType
        ? ar2
        : castNDArray<T>(ar2, commonDType);
    final NDArray<T> flat1 = (c1.rank == 1 && c1.isContiguous)
        ? c1
        : c1.flatten();
    final NDArray<T> flat2 = (c2.rank == 1 && c2.isContiguous)
        ? c2
        : c2.flatten();

    final NDArray u1 = assumeUnique ? sort(flat1) : unique(flat1) as NDArray;
    final NDArray u2 = assumeUnique ? sort(flat2) : unique(flat2) as NDArray;

    final maxDstSize = u1.size;

    if (maxDstSize == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError('Incompatible out buffer shape.');
      }
      return out ??
          (NDArray<T>.create([0], commonDType)..detachToParentScope());
    }

    final dest = NDArray<T>.create([maxDstSize], commonDType);

    final diffCount = ndarray_setdiff1d(
      u1.pointer.cast(),
      u1.size,
      u2.pointer.cast(),
      u2.size,
      dest.pointer.cast(),
      encodeDType(commonDType),
    );

    if (diffCount == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError('Incompatible out buffer shape.');
      }
      return out ??
          (NDArray<T>.create([0], commonDType)..detachToParentScope());
    }

    if (out != null && !listEquals(out.shape, [diffCount])) {
      throw ArgumentError('Incompatible out buffer shape.');
    }

    final validView = dest.slice([Slice(start: 0, stop: diffCount)]);
    if (out != null) {
      validView.copy(out: out);
      return out;
    } else {
      return validView.copy()..detachToParentScope();
    }
  });
}

/// Finds the set exclusive-or of two arrays.
///
/// Returns the sorted, unique values that are in only one (not both) of the input arrays.
///
/// It is an error if [ar1] or [ar2] is disposed.
NDArray<T> setxor1d<T extends Object>(
  NDArray<T> ar1,
  NDArray<T> ar2, {
  bool assumeUnique = false,
  NDArray<T>? out,
}) {
  if (ar1.isDisposed || ar2.isDisposed) {
    throw StateError('Cannot execute setxor1d on disposed array(s).');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write setxor1d result to a disposed output array.',
    );
  }
  final DType<T> commonDType =
      (ar1.dtype == ar2.dtype ? ar1.dtype : resolveDType(ar1.dtype, ar2.dtype))
          as DType<T>;
  if (out != null && out.dtype != commonDType) {
    throw ArgumentError('Incompatible out buffer dtype.');
  }

  return NDArray.scope(() {
    final NDArray<T> c1 = ar1.dtype == commonDType
        ? ar1
        : castNDArray<T>(ar1, commonDType);
    final NDArray<T> c2 = ar2.dtype == commonDType
        ? ar2
        : castNDArray<T>(ar2, commonDType);
    final NDArray<T> flat1 = (c1.rank == 1 && c1.isContiguous)
        ? c1
        : c1.flatten();
    final NDArray<T> flat2 = (c2.rank == 1 && c2.isContiguous)
        ? c2
        : c2.flatten();

    final NDArray u1 = assumeUnique ? sort(flat1) : unique(flat1) as NDArray;
    final NDArray u2 = assumeUnique ? sort(flat2) : unique(flat2) as NDArray;

    final maxDstSize = u1.size + u2.size;

    if (maxDstSize == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError('Incompatible out buffer shape.');
      }
      return out ??
          (NDArray<T>.create([0], commonDType)..detachToParentScope());
    }

    final dest = NDArray<T>.create([maxDstSize], commonDType);

    final xorCount = ndarray_setxor1d(
      u1.pointer.cast(),
      u1.size,
      u2.pointer.cast(),
      u2.size,
      dest.pointer.cast(),
      encodeDType(commonDType),
    );

    if (xorCount == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError('Incompatible out buffer shape.');
      }
      return out ??
          (NDArray<T>.create([0], commonDType)..detachToParentScope());
    }

    if (out != null && !listEquals(out.shape, [xorCount])) {
      throw ArgumentError('Incompatible out buffer shape.');
    }

    final validView = dest.slice([Slice(start: 0, stop: xorCount)]);
    if (out != null) {
      validView.copy(out: out);
      return out;
    } else {
      return validView.copy()..detachToParentScope();
    }
  });
}

/// Finds the union of two arrays.
///
/// Returns the unique, sorted array of values that are in either of the two input arrays.
///
/// It is an error if [ar1] or [ar2] is disposed.
NDArray<T> union1d<T extends Object>(
  NDArray<T> ar1,
  NDArray<T> ar2, {
  NDArray<T>? out,
}) {
  if (ar1.isDisposed || ar2.isDisposed) {
    throw StateError('Cannot execute union1d on disposed array(s).');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write union1d result to a disposed output array.');
  }
  final DType<T> commonDType =
      (ar1.dtype == ar2.dtype ? ar1.dtype : resolveDType(ar1.dtype, ar2.dtype))
          as DType<T>;
  if (out != null && out.dtype != commonDType) {
    throw ArgumentError('Incompatible out buffer dtype.');
  }

  return NDArray.scope(() {
    final NDArray<T> c1 = ar1.dtype == commonDType
        ? ar1
        : castNDArray<T>(ar1, commonDType);
    final NDArray<T> c2 = ar2.dtype == commonDType
        ? ar2
        : castNDArray<T>(ar2, commonDType);
    final NDArray<T> flat1 = (c1.rank == 1 && c1.isContiguous)
        ? c1
        : c1.flatten();
    final NDArray<T> flat2 = (c2.rank == 1 && c2.isContiguous)
        ? c2
        : c2.flatten();

    final NDArray u1 = unique(flat1) as NDArray;
    final NDArray u2 = unique(flat2) as NDArray;

    final maxDstSize = u1.size + u2.size;

    if (maxDstSize == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError('Incompatible out buffer shape.');
      }
      return out ??
          (NDArray<T>.create([0], commonDType)..detachToParentScope());
    }

    final dest = NDArray<T>.create([maxDstSize], commonDType);

    final unionCount = ndarray_union1d(
      u1.pointer.cast(),
      u1.size,
      u2.pointer.cast(),
      u2.size,
      dest.pointer.cast(),
      encodeDType(commonDType),
    );

    if (unionCount == 0) {
      if (out != null && !listEquals(out.shape, [0])) {
        throw ArgumentError('Incompatible out buffer shape.');
      }
      return out ??
          (NDArray<T>.create([0], commonDType)..detachToParentScope());
    }

    if (out != null && !listEquals(out.shape, [unionCount])) {
      throw ArgumentError('Incompatible out buffer shape.');
    }

    final validView = dest.slice([Slice(start: 0, stop: unionCount)]);
    if (out != null) {
      validView.copy(out: out);
      return out;
    } else {
      return validView.copy()..detachToParentScope();
    }
  });
}

/// Tests whether each element of an array is also present in a second array.
///
/// Returns a boolean array of the same shape as [element] that is `true` where an element of [element] is in [testElements] and `false` otherwise.
///
/// It is an error if [element] or [testElements] is disposed.
NDArray<bool> isin<T extends Object>(
  NDArray<T> element,
  NDArray<T> testElements, {
  bool assumeUnique = false,
  bool invert = false,
  NDArray<bool>? out,
}) {
  if (element.isDisposed || testElements.isDisposed) {
    throw StateError('Cannot execute isin on disposed array(s).');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write isin result to a disposed output array.');
  }
  if (out != null) {
    if (!listEquals(out.shape, element.shape) || out.dtype != DType.boolean) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  final DType<T> commonDType =
      (element.dtype == testElements.dtype
              ? element.dtype
              : resolveDType(element.dtype, testElements.dtype))
          as DType<T>;

  final bool useTempOut =
      out != null &&
      (!out.isContiguous ||
          sharesMemory(element, out) ||
          sharesMemory(testElements, out));

  return NDArray.scope(() {
    final NDArray<T> cElement = element.dtype == commonDType
        ? element
        : castNDArray<T>(element, commonDType);
    final NDArray<T> cTest = testElements.dtype == commonDType
        ? testElements
        : castNDArray<T>(testElements, commonDType);

    final NDArray<T> flatTest = (cTest.rank == 1 && cTest.isContiguous)
        ? cTest
        : cTest.flatten();
    final NDArray uTest = assumeUnique
        ? sort(flatTest)
        : unique(flatTest) as NDArray;
    final NDArray<T> contigElement = cElement.isContiguous
        ? cElement
        : cElement.copy();

    final dest = (out != null && !useTempOut)
        ? out
        : NDArray<bool>.create(element.shape, DType.boolean);

    ndarray_isin(
      contigElement.pointer.cast(),
      element.size,
      uTest.pointer.cast(),
      uTest.size,
      dest.pointer.cast(),
      encodeDType(commonDType),
      invert ? 1 : 0,
    );

    if (useTempOut) {
      dest.copy(out: out);
      return out;
    }
    if (out == null) {
      dest.detachToParentScope();
    }
    return dest;
  });
}
