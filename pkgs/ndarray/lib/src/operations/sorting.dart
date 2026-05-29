// ignore_for_file: non_constant_identifier_names
import 'dart:typed_data';
import '../ndarray.dart';
import 'dart:ffi' as ffi;
import '../ndarray_bindings.dart';
import '../scratch_arena.dart';

// Standalone operational relative cross-imports
import 'spacers.dart';
import 'helpers.dart';

/// Returns a sorted copy of an array along a specified [axis].
///
/// This function corresponds to NumPy's `sort` function.
///
/// It uses native ANSI C `qsort` via FFI to perform zero-copy, high-speed
/// in-place sorting straight on the C heap for contiguous last-axis rows, completely
/// bypassing Dart memory marshalling.
///
/// Complex numbers are sorted lexicographically: by their real parts first,
/// and by their imaginary parts if the real parts are equal.
///
/// **Preconditions:**
/// - [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [RangeError] if [axis] is out of bounds.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
NDArray sort(NDArray a, {int axis = -1, SortKind kind = SortKind.quicksort}) {
  if (a.size == 0) {
    return NDArray.create(a.shape, a.dtype);
  }
  final rank = a.shape.length;
  if (rank == 0) {
    return NDArray.fromList(List.from(a.data), [], a.dtype);
  }

  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(targetAxis, 0, rank - 1, 'axis');
  }

  if (targetAxis != rank - 1) {
    final swappedView = a.swapaxes(targetAxis, rank - 1);
    final sortedView = sort(swappedView, axis: rank - 1, kind: kind);
    return sortedView.swapaxes(targetAxis, rank - 1);
  }

  NDArray src = a;
  if (!a.isContiguous) {
    src = NDArray.fromList(a.toList(), a.shape, a.dtype);
  }

  final result = NDArray.create(src.shape, src.dtype);
  if (src.dtype == DType.float64) {
    (result.data as Float64List).setRange(
      0,
      src.data.length,
      src.data as Float64List,
    );
  } else if (src.dtype == DType.float32) {
    (result.data as Float32List).setRange(
      0,
      src.data.length,
      src.data as Float32List,
    );
  } else if (src.dtype == DType.int64) {
    (result.data as Int64List).setRange(
      0,
      src.data.length,
      src.data as Int64List,
    );
  } else if (src.dtype == DType.int32) {
    (result.data as Int32List).setRange(
      0,
      src.data.length,
      src.data as Int32List,
    );
  } else if (src.dtype == DType.complex128 || src.dtype == DType.complex64) {
    final srcBacking = (src.data as ComplexList).backingList;
    final resBacking = (result.data as ComplexList).backingList;
    if (srcBacking is Float64List && resBacking is Float64List) {
      resBacking.setRange(0, srcBacking.length, srcBacking);
    } else if (srcBacking is Float32List && resBacking is Float32List) {
      resBacking.setRange(0, srcBacking.length, srcBacking);
    }
  } else if (src.dtype == DType.boolean) {
    final srcBacking = (src.data as BoolList).backingList;
    final resBacking = (result.data as BoolList).backingList;
    resBacking.setRange(0, srcBacking.length, srcBacking);
  }

  final n = src.shape.last;
  final totalSize = src.shape.isEmpty ? 1 : src.shape.reduce((x, y) => x * y);
  final numRows = totalSize ~/ n;

  if (src.dtype == DType.boolean) {
    // Sort boolean rows in $O(N)$
    for (var r = 0; r < numRows; r++) {
      final rowStart = r * n;
      final rowEnd = rowStart + n;
      final resBacking = (result.data as BoolList).backingList;
      var falses = 0;
      for (var i = rowStart; i < rowEnd; i++) {
        if (resBacking[i] == 0) falses++;
      }
      for (var i = rowStart; i < rowStart + falses; i++) {
        resBacking[i] = 0;
      }
      for (var i = rowStart + falses; i < rowEnd; i++) {
        resBacking[i] = 1;
      }
    }
    return result;
  }

  int elementSizeInBytes;
  if (src.dtype == DType.float64 || src.dtype == DType.int64) {
    elementSizeInBytes = 8;
  } else if (src.dtype == DType.float32 || src.dtype == DType.int32) {
    elementSizeInBytes = 4;
  } else if (src.dtype == DType.complex64) {
    elementSizeInBytes = 8;
  } else if (src.dtype == DType.complex128) {
    elementSizeInBytes = 16;
  } else {
    throw UnimplementedError('Unsupported dtype for sort: ${src.dtype}');
  }

  final baseCast = result.pointer.cast<ffi.Uint8>();
  final rowSizeInBytes = n * elementSizeInBytes;
  final nativeKind = mapSortKind(kind);

  for (var r = 0; r < numRows; r++) {
    final rowPtr = baseCast + (r * rowSizeInBytes);

    // High-speed direct C sorters bypassing FFI context switches
    if (src.dtype == DType.float64) {
      native_sort_double(rowPtr.cast<ffi.Double>(), n, nativeKind);
    } else if (src.dtype == DType.float32) {
      native_sort_float(rowPtr.cast<ffi.Float>(), n, nativeKind);
    } else if (src.dtype == DType.int64) {
      native_sort_int64(rowPtr.cast<ffi.LongLong>(), n, nativeKind);
    } else if (src.dtype == DType.int32) {
      native_sort_int32(rowPtr.cast<ffi.Int>(), n, nativeKind);
    } else if (src.dtype == DType.complex128) {
      native_sort_complex128(rowPtr.cast<ffi.Double>(), n, nativeKind);
    } else if (src.dtype == DType.complex64) {
      native_sort_complex64(rowPtr.cast<ffi.Float>(), n, nativeKind);
    }
  }

  return result;
}

/// Returns the indices that would sort an array along a specified [axis].
///
/// This function corresponds to NumPy's `argsort` function.
///
/// It performs indirect index sorting, fully supporting complex numbers
/// lexicographical logic and internal axes reorientation via the axis-swapping pipeline.
///
/// **Preconditions:**
/// - [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [RangeError] if [axis] is out of bounds.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
NDArray<int> argsort(
  NDArray a, {
  int axis = -1,
  SortKind kind = SortKind.quicksort,
}) {
  if (a.size == 0) {
    return NDArray<int>.create(a.shape, DType.int32);
  }
  final rank = a.shape.length;
  if (rank == 0) {
    return NDArray.fromList(<int>[0], [], DType.int32);
  }

  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(targetAxis, 0, rank - 1, 'axis');
  }

  if (targetAxis != rank - 1) {
    final swappedView = a.swapaxes(targetAxis, rank - 1);
    final sortedIndicesView = argsort(swappedView, axis: rank - 1, kind: kind);
    return sortedIndicesView.swapaxes(targetAxis, rank - 1);
  }

  NDArray src = a;
  bool needsDispose = false;
  if (!a.isContiguous) {
    src = NDArray.create(a.shape, a.dtype);
    src.data.setRange(0, src.data.length, a.toList());
    needsDispose = true;
  }

  try {
    final n = src.shape.last;
    final totalSize = src.shape.isEmpty ? 1 : src.shape.reduce((x, y) => x * y);
    final numRows = totalSize ~/ n;

    final result = NDArray<int>.create(src.shape, DType.int32);
    final nativeKind = mapSortKind(kind);

    if (src.dtype == DType.float64) {
      final dataPtr = src.pointer.cast<ffi.Double>();
      final resPtr = result.pointer.cast<ffi.Int>();
      for (var r = 0; r < numRows; r++) {
        native_argsort_double(dataPtr + r * n, resPtr + r * n, n, nativeKind);
      }
      return result;
    } else if (src.dtype == DType.float32) {
      final dataPtr = src.pointer.cast<ffi.Float>();
      final resPtr = result.pointer.cast<ffi.Int>();
      for (var r = 0; r < numRows; r++) {
        native_argsort_float(dataPtr + r * n, resPtr + r * n, n, nativeKind);
      }
      return result;
    } else if (src.dtype == DType.int64) {
      final dataPtr = src.pointer.cast<ffi.LongLong>();
      final resPtr = result.pointer.cast<ffi.Int>();
      for (var r = 0; r < numRows; r++) {
        native_argsort_int64(dataPtr + r * n, resPtr + r * n, n, nativeKind);
      }
      return result;
    } else if (src.dtype == DType.int32) {
      final dataPtr = src.pointer.cast<ffi.Int>();
      final resPtr = result.pointer.cast<ffi.Int>();
      for (var r = 0; r < numRows; r++) {
        native_argsort_int32(dataPtr + r * n, resPtr + r * n, n, nativeKind);
      }
      return result;
    }

    for (var r = 0; r < numRows; r++) {
      final rowStart = r * n;
      final indices = List<int>.generate(n, (i) => i);

      if (src.dtype == DType.complex128 || src.dtype == DType.complex64) {
        final dataList = src.data as List<Complex>;
        indices.sort((i, j) {
          final cA = dataList[rowStart + i];
          final cB = dataList[rowStart + j];
          if (cA.real != cB.real) return cA.real.compareTo(cB.real);
          return cA.imag.compareTo(cB.imag);
        });
      } else if (src.dtype == DType.boolean) {
        final dataList = src.data as List<bool>;
        indices.sort((i, j) {
          final bA = dataList[rowStart + i];
          final bB = dataList[rowStart + j];
          if (bA == bB) return 0;
          return bA ? 1 : -1;
        });
      } else {
        throw UnimplementedError('Unsupported dtype for argsort: ${src.dtype}');
      }

      for (var i = 0; i < n; i++) {
        result.data[rowStart + i] = indices[i];
      }
    }

    return result;
  } finally {
    if (needsDispose) {
      src.dispose();
    }
  }
}

/// Rearranges the elements of the array along a specified [axis] such that
/// the value of the element at [kth] position is in the position it would be
/// in a sorted array.
///
/// This function corresponds to NumPy's `partition` function.
///
/// **Preconditions:**
/// - [axis] must be within `[-rank, rank - 1]`.
/// - [kth] must be an `int` or `List<int>` containing indices within `[0, axis_size - 1]`.
///
/// **Throws:**
/// - [RangeError] if [axis] is out of bounds.
/// - [ArgumentError] if [kth] is invalid.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
NDArray partition(NDArray a, dynamic kth, {int axis = -1}) {
  final rank = a.shape.length;
  if (rank == 0) {
    return NDArray.fromList(List.from(a.data), [], a.dtype);
  }

  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(targetAxis, 0, rank - 1, 'axis');
  }

  final n = a.shape[targetAxis];

  // Parse and validate kth
  final List<int> kList = [];
  if (kth is int) {
    final val = kth < 0 ? n + kth : kth;
    if (val < 0 || val >= n) {
      throw RangeError.value(kth, 'kth');
    }
    kList.add(val);
  } else if (kth is Iterable<int>) {
    for (final k in kth) {
      final val = k < 0 ? n + k : k;
      if (val < 0 || val >= n) {
        throw RangeError.value(k, 'kth element');
      }
      kList.add(val);
    }
  } else {
    throw ArgumentError('kth must be an int or an Iterable of ints.');
  }

  final uniqueK = kList.toSet().toList()..sort();

  if (targetAxis != rank - 1) {
    final swappedView = a.swapaxes(targetAxis, rank - 1);
    final partitionedView = partition(swappedView, uniqueK, axis: rank - 1);
    return partitionedView.swapaxes(targetAxis, rank - 1);
  }

  NDArray src = a;
  if (!a.isContiguous) {
    src = NDArray.fromList(a.toList(), a.shape, a.dtype);
  }

  final result = NDArray.create(src.shape, src.dtype);

  // Copy data to result
  if (src.dtype == DType.float64) {
    (result.data as Float64List).setRange(
      0,
      src.data.length,
      src.data as Float64List,
    );
  } else if (src.dtype == DType.float32) {
    (result.data as Float32List).setRange(
      0,
      src.data.length,
      src.data as Float32List,
    );
  } else if (src.dtype == DType.int64) {
    (result.data as Int64List).setRange(
      0,
      src.data.length,
      src.data as Int64List,
    );
  } else if (src.dtype == DType.int32) {
    (result.data as Int32List).setRange(
      0,
      src.data.length,
      src.data as Int32List,
    );
  } else if (src.dtype == DType.complex128 || src.dtype == DType.complex64) {
    final srcBacking = (src.data as ComplexList).backingList;
    final resBacking = (result.data as ComplexList).backingList;
    if (srcBacking is Float64List && resBacking is Float64List) {
      resBacking.setRange(0, srcBacking.length, srcBacking);
    } else if (srcBacking is Float32List && resBacking is Float32List) {
      resBacking.setRange(0, srcBacking.length, srcBacking);
    }
  } else if (src.dtype == DType.boolean) {
    final srcBacking = (src.data as BoolList).backingList;
    final resBacking = (result.data as BoolList).backingList;
    resBacking.setRange(0, srcBacking.length, srcBacking);
  }

  if (uniqueK.isEmpty) {
    return result;
  }

  final totalSize = src.shape.isEmpty ? 1 : src.shape.reduce((x, y) => x * y);
  final numRows = totalSize ~/ n;

  if (src.dtype == DType.boolean) {
    // A boolean partition is sorted
    return sort(result, axis: rank - 1);
  }

  int elementSizeInBytes;
  if (src.dtype == DType.float64 || src.dtype == DType.int64) {
    elementSizeInBytes = 8;
  } else if (src.dtype == DType.float32 || src.dtype == DType.int32) {
    elementSizeInBytes = 4;
  } else if (src.dtype == DType.complex64) {
    elementSizeInBytes = 8;
  } else if (src.dtype == DType.complex128) {
    elementSizeInBytes = 16;
  } else {
    throw UnimplementedError('Unsupported dtype for partition: ${src.dtype}');
  }

  final baseCast = result.pointer.cast<ffi.Uint8>();
  final rowSizeInBytes = n * elementSizeInBytes;

  final marker = ScratchArena.marker;
  final cKList = ScratchArena.allocate<ffi.Int>(
    uniqueK.length * ffi.sizeOf<ffi.Int>(),
  );
  for (var i = 0; i < uniqueK.length; i++) {
    cKList[i] = uniqueK[i];
  }

  try {
    for (var r = 0; r < numRows; r++) {
      final rowPtr = baseCast + (r * rowSizeInBytes);

      switch (src.dtype) {
        case DType.float64:
          native_partition_double(
            rowPtr.cast<ffi.Double>(),
            n,
            cKList,
            uniqueK.length,
          );
        case DType.float32:
          native_partition_float(
            rowPtr.cast<ffi.Float>(),
            n,
            cKList,
            uniqueK.length,
          );
        case DType.int64:
          native_partition_int64(
            rowPtr.cast<ffi.LongLong>(),
            n,
            cKList,
            uniqueK.length,
          );
        case DType.int32:
          native_partition_int32(
            rowPtr.cast<ffi.Int>(),
            n,
            cKList,
            uniqueK.length,
          );
        case DType.complex128:
          native_partition_complex128(
            rowPtr.cast<ffi.Double>(),
            n,
            cKList,
            uniqueK.length,
          );
        case DType.complex64:
          native_partition_complex64(
            rowPtr.cast<ffi.Float>(),
            n,
            cKList,
            uniqueK.length,
          );
        default:
          break;
      }
    }
  } finally {
    ScratchArena.reset(marker);
  }

  return result;
}

/// Returns the indices that would partition an array along a specified [axis].
///
/// This function corresponds to NumPy's `argpartition` function.
///
/// **Preconditions:**
/// - [axis] must be within `[-rank, rank - 1]`.
/// - [kth] must be an `int` or `List<int>` containing indices within `[0, axis_size - 1]`.
///
/// **Throws:**
/// - [RangeError] if [axis] is out of bounds.
/// - [ArgumentError] if [kth] is invalid.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
NDArray<int> argpartition(NDArray a, dynamic kth, {int axis = -1}) {
  final rank = a.shape.length;
  if (rank == 0) {
    return NDArray.fromList(<int>[0], [], DType.int32);
  }

  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(targetAxis, 0, rank - 1, 'axis');
  }

  final n = a.shape[targetAxis];

  // Parse and validate kth
  final List<int> kList = [];
  if (kth is int) {
    final val = kth < 0 ? n + kth : kth;
    if (val < 0 || val >= n) {
      throw RangeError.value(kth, 'kth');
    }
    kList.add(val);
  } else if (kth is Iterable<int>) {
    for (final k in kth) {
      final val = k < 0 ? n + k : k;
      if (val < 0 || val >= n) {
        throw RangeError.value(k, 'kth element');
      }
      kList.add(val);
    }
  } else {
    throw ArgumentError('kth must be an int or an Iterable of ints.');
  }

  final uniqueK = kList.toSet().toList()..sort();

  if (targetAxis != rank - 1) {
    final swappedView = a.swapaxes(targetAxis, rank - 1);
    final partitionedIndicesView = argpartition(
      swappedView,
      uniqueK,
      axis: rank - 1,
    );
    return partitionedIndicesView.swapaxes(targetAxis, rank - 1);
  }

  NDArray src = a;
  bool needsDispose = false;
  if (!a.isContiguous) {
    src = NDArray.create(a.shape, a.dtype);
    src.data.setRange(0, src.data.length, a.toList());
    needsDispose = true;
  }

  try {
    final totalSize = src.shape.isEmpty ? 1 : src.shape.reduce((x, y) => x * y);
    final numRows = totalSize ~/ n;

    final result = NDArray<int>.create(src.shape, DType.int32);

    if (uniqueK.isEmpty) {
      for (var i = 0; i < result.data.length; i++) {
        result.data[i] = i % n;
      }
      return result;
    }

    final marker = ScratchArena.marker;
    final cKList = ScratchArena.allocate<ffi.Int>(
      uniqueK.length * ffi.sizeOf<ffi.Int>(),
    );
    for (var i = 0; i < uniqueK.length; i++) {
      cKList[i] = uniqueK[i];
    }

    try {
      if (src.dtype == DType.float64) {
        final dataPtr = src.pointer.cast<ffi.Double>();
        final resPtr = result.pointer.cast<ffi.Int>();
        for (var r = 0; r < numRows; r++) {
          native_argpartition_double(
            dataPtr + r * n,
            resPtr + r * n,
            n,
            cKList,
            uniqueK.length,
          );
        }
        return result;
      } else if (src.dtype == DType.float32) {
        final dataPtr = src.pointer.cast<ffi.Float>();
        final resPtr = result.pointer.cast<ffi.Int>();
        for (var r = 0; r < numRows; r++) {
          native_argpartition_float(
            dataPtr + r * n,
            resPtr + r * n,
            n,
            cKList,
            uniqueK.length,
          );
        }
        return result;
      } else if (src.dtype == DType.int64) {
        final dataPtr = src.pointer.cast<ffi.LongLong>();
        final resPtr = result.pointer.cast<ffi.Int>();
        for (var r = 0; r < numRows; r++) {
          native_argpartition_int64(
            dataPtr + r * n,
            resPtr + r * n,
            n,
            cKList,
            uniqueK.length,
          );
        }
        return result;
      } else if (src.dtype == DType.int32) {
        final dataPtr = src.pointer.cast<ffi.Int>();
        final resPtr = result.pointer.cast<ffi.Int>();
        for (var r = 0; r < numRows; r++) {
          native_argpartition_int32(
            dataPtr + r * n,
            resPtr + r * n,
            n,
            cKList,
            uniqueK.length,
          );
        }
        return result;
      } else if (src.dtype == DType.complex128) {
        final dataPtr = src.pointer.cast<ffi.Double>();
        final resPtr = result.pointer.cast<ffi.Int>();
        for (var r = 0; r < numRows; r++) {
          native_argpartition_complex128(
            dataPtr + r * n * 2,
            resPtr + r * n,
            n,
            cKList,
            uniqueK.length,
          );
        }
        return result;
      } else if (src.dtype == DType.complex64) {
        final dataPtr = src.pointer.cast<ffi.Float>();
        final resPtr = result.pointer.cast<ffi.Int>();
        for (var r = 0; r < numRows; r++) {
          native_argpartition_complex64(
            dataPtr + r * n * 2,
            resPtr + r * n,
            n,
            cKList,
            uniqueK.length,
          );
        }
        return result;
      } else if (src.dtype == DType.boolean) {
        for (var r = 0; r < numRows; r++) {
          final rowStart = r * n;
          final indices = List<int>.generate(n, (i) => i);
          final dataList = src.data as List<bool>;
          indices.sort((i, j) {
            final bA = dataList[rowStart + i];
            final bB = dataList[rowStart + j];
            if (bA == bB) return 0;
            return bA ? 1 : -1;
          });
          for (var i = 0; i < n; i++) {
            result.data[rowStart + i] = indices[i];
          }
        }
        return result;
      } else {
        throw UnimplementedError(
          'Unsupported dtype for argpartition: ${src.dtype}',
        );
      }
    } finally {
      ScratchArena.reset(marker);
    }
  } finally {
    if (needsDispose) {
      src.dispose();
    }
  }
}

/// Find indices where elements of [v] should be inserted to maintain order in a sorted 1-D array [a].
///
/// This function corresponds to NumPy's `searchsorted` function.
///
/// Binary search is performed at C-speed using native pointers, fully supporting
/// arbitrary multi-dimensional shapes for the query array [v]. The returned index
/// array will have the exact same shape as [v].
///
/// ### Preconditions
/// - [a] must be a 1-D array. If [sorter] is `null`, [a] must be sorted in ascending order.
/// - [v] must have a matching data type to [a].
/// - [sorter] (optional) must be a 1-D integer array of the same size as [a]
///   containing indices that sort [a] into ascending order. If provided, binary search
///   is performed indirectly using the sorter indices, completely copy-free.
///
/// ### Throws
/// - [ArgumentError] if [a] is not 1-D, [sorter] shape/size is invalid, or if data types mismatch.
/// - [StateError] if any input array is already disposed.
///
/// ### Performance Considerations
/// - **Time Complexity**: $O(M \log N)$ where $N$ is the size of [a] and $M$ is the size of [v].
/// - **Memory Complexity**: $O(M)$ to hold the returned N-dimensional shape. The C search runs in $O(1)$ auxiliary space.
///
/// ### Inline Example:
/// ```dart
/// import 'package:ndarray/ndarray.dart';
///
/// void main() {
///   final a = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
///   final v = NDArray.fromList([15.0, 30.0, 5.0, 35.0], [2, 2], DType.float64);
///
///   // Finds insertion indices for the entire multi-dimensional grid v:
///   final indices = searchsorted(a, v, side: SearchSide.left);
///
///   print(indices.shape); // [2, 2] (matches shape of v)
///   print(indices.toList()); // [[1, 2], [0, 3]]
/// }
/// ```
NDArray<int> searchsorted(
  NDArray a,
  NDArray v, {
  SearchSide side = SearchSide.left,
  NDArray<int>? sorter,
}) {
  if (a.shape.length != 1) {
    throw ArgumentError('a must be a 1-D array.');
  }

  if (sorter != null &&
      (sorter.shape.length != 1 || sorter.shape[0] != a.shape[0])) {
    throw ArgumentError('sorter must be a 1-D array of the same size as a.');
  }

  final result = NDArray<int>.create(v.shape, DType.int32);

  if (v.size == 0) {
    return result;
  }

  NDArray srcA = a;
  if (!a.isContiguous) {
    srcA = NDArray.fromList(a.toList(), a.shape, a.dtype);
  }

  NDArray srcV = v;
  if (!v.isContiguous) {
    srcV = NDArray.fromList(v.toList(), v.shape, v.dtype);
  }

  NDArray<int>? srcSorter = sorter;
  if (sorter != null && !sorter.isContiguous) {
    srcSorter = NDArray<int>.fromList(
      sorter.toList(),
      sorter.shape,
      DType.int32,
    );
  }

  final size = srcA.shape[0];
  final numValues = srcV.size;
  final sideLeft = side == SearchSide.left ? 1 : 0;

  final ffi.Pointer<ffi.Int> cSorter = (srcSorter != null)
      ? srcSorter.pointer.cast<ffi.Int>()
      : ffi.Pointer<ffi.Int>.fromAddress(0);

  try {
    switch (srcA.dtype) {
      case DType.float64:
        if (srcV.dtype != DType.float64) {
          throw ArgumentError(
            'v and a must have matching dtypes (expected float64, got ${v.dtype})',
          );
        }
        native_searchsorted_double(
          srcA.pointer.cast<ffi.Double>(),
          size,
          srcV.pointer.cast<ffi.Double>(),
          result.pointer.cast<ffi.Int>(),
          numValues,
          sideLeft,
          cSorter,
        );
      case DType.float32:
        if (srcV.dtype != DType.float32) {
          throw ArgumentError(
            'v and a must have matching dtypes (expected float32, got ${v.dtype})',
          );
        }
        native_searchsorted_float(
          srcA.pointer.cast<ffi.Float>(),
          size,
          srcV.pointer.cast<ffi.Float>(),
          result.pointer.cast<ffi.Int>(),
          numValues,
          sideLeft,
          cSorter,
        );
      case DType.int64:
        if (srcV.dtype != DType.int64) {
          throw ArgumentError(
            'v and a must have matching dtypes (expected int64, got ${v.dtype})',
          );
        }
        native_searchsorted_int64(
          srcA.pointer.cast<ffi.LongLong>(),
          size,
          srcV.pointer.cast<ffi.LongLong>(),
          result.pointer.cast<ffi.Int>(),
          numValues,
          sideLeft,
          cSorter,
        );
      case DType.int32:
        if (srcV.dtype != DType.int32) {
          throw ArgumentError(
            'v and a must have matching dtypes (expected int32, got ${v.dtype})',
          );
        }
        native_searchsorted_int32(
          srcA.pointer.cast<ffi.Int>(),
          size,
          srcV.pointer.cast<ffi.Int>(),
          result.pointer.cast<ffi.Int>(),
          numValues,
          sideLeft,
          cSorter,
        );
      case DType.complex128:
        if (srcV.dtype != DType.complex128) {
          throw ArgumentError(
            'v and a must have matching dtypes (expected complex128, got ${v.dtype})',
          );
        }
        native_searchsorted_complex128(
          srcA.pointer.cast<ffi.Double>(),
          size,
          srcV.pointer.cast<ffi.Double>(),
          result.pointer.cast<ffi.Int>(),
          numValues,
          sideLeft,
          cSorter,
        );
      case DType.complex64:
        if (srcV.dtype != DType.complex64) {
          throw ArgumentError(
            'v and a must have matching dtypes (expected complex64, got ${v.dtype})',
          );
        }
        native_searchsorted_complex64(
          srcA.pointer.cast<ffi.Float>(),
          size,
          srcV.pointer.cast<ffi.Float>(),
          result.pointer.cast<ffi.Int>(),
          numValues,
          sideLeft,
          cSorter,
        );
      case DType.boolean:
        final dataA = srcA.data as List<bool>;
        final dataV = srcV.data as List<bool>;
        final sortedIndices = srcSorter?.toList();

        bool getElement(int idx) {
          return sortedIndices != null ? dataA[sortedIndices[idx]] : dataA[idx];
        }

        for (var vIdx = 0; vIdx < numValues; vIdx++) {
          final val = dataV[vIdx];
          var low = 0;
          var high = size;
          while (low < high) {
            final mid = low + ((high - low) >> 1);
            final midVal = getElement(mid);

            int comp;
            if (midVal == val) {
              comp = 0;
            } else {
              comp = midVal ? 1 : -1; // false < true
            }

            if (side == SearchSide.left) {
              if (comp < 0) {
                low = mid + 1;
              } else {
                high = mid;
              }
            } else {
              if (comp <= 0) {
                low = mid + 1;
              } else {
                high = mid;
              }
            }
          }
          result.data[vIdx] = low;
        }
      default:
        throw UnimplementedError(
          'Unsupported dtype for searchsorted: ${srcA.dtype}',
        );
    }
  } finally {
    if (srcA != a) srcA.dispose();
    if (srcV != v) srcV.dispose();
    if (srcSorter != sorter) srcSorter?.dispose();
  }

  return result;
}

/// Return elements chosen from [x] or [y] depending on [condition], or the indices where [condition] is true.
///
/// This function implements a ternary conditional selector or coordinate selector depending on its arguments:
/// 1. **Ternary Select (Three Arguments):**
///    If both [x] and [y] are provided, returns a new array with elements from [x] where [condition] is true, and
///    elements from [y] where it is false. Broadcasting rules apply across [condition], [x], and [y].
/// 2. **Coordinate Selector (One Argument):**
///    If both [x] and [y] are omitted, returns a list of 1-D index coordinate arrays representing the indices
///    where [condition] evaluates to true.
///
/// **Preconditions:**
/// - [condition] must be a boolean array (`condition.dtype == DType.boolean`).
/// - Either both or neither of [x] and [y] must be provided.
/// - If provided, [condition], [x], and [y] shapes must be mutually broadcast-compatible.
///
/// **Throws:**
/// - [ArgumentError] if only one of [x] or [y] is provided.
/// - [ArgumentError] if [out] is specified when [x] and [y] are omitted.
/// - [ArgumentError] if the shapes are not broadcast-compatible.
/// - [ArgumentError] if the [out] recycler has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic time complexity is $O(N)$ where $N$ is the broadcasted result size.
/// - If all arrays are contiguous, of `float32`/`float64`/`int32`/`int64` types, and C-contiguous,
///   leverages high-speed C FFI vector operations (`s_where_double`/`s_where_float`) for ultra-high performance.
///
/// **Memory Ownership & Lifetime:**
/// - Allocates a new array (or list of arrays) on the unmanaged C heap. **The caller takes full ownership** of this memory and **must explicitly call [dispose()]** on all returned arrays to prevent native leaks, unless executing inside a managed [NDArray.scope()].
///
/// **NumPy Counterpart:**
/// - Maps directly to NumPy's `np.where`.
///
/// **Example:**
/// ```dart
/// final cond = NDArray.fromList([true, false, true], [3], DType.boolean);
/// final x = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
/// final y = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
/// final result = where(cond, x, y) as NDArray<double>;
/// print(result.toList()); // [1.0, 20.0, 3.0]
/// result.dispose();
/// ```
dynamic where(NDArray condition, [NDArray? x, NDArray? y, NDArray? out]) {
  if (x == null && y == null) {
    if (out != null) {
      throw ArgumentError(
        'out buffer cannot be provided when x and y are omitted.',
      );
    }
    return nonzero(condition);
  }

  if ((x == null && y != null) || (x != null && y == null)) {
    throw ArgumentError('Either both or neither of x and y must be given');
  }

  // Calculate target common shape via high-speed 3-way broadcast matching
  final commonShape = broadcast3Shapes(condition.shape, x!.shape, y!.shape);

  final DType<dynamic> targetDType = resolveDType(x.dtype, y.dtype);

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for where() result.',
      );
    }
  }

  // Compute precise broadcasted strides for each operand independently to commonShape
  final stridesCond = broadcastStrides(condition, commonShape);
  final stridesX = broadcastStrides(x, commonShape);
  final stridesY = broadcastStrides(y, commonShape);

  final result = out ?? NDArray.create(commonShape, targetDType);
  final resultStrides = NDArray.computeCStrides(commonShape);

  // 0. Advanced ND Odometer Ternary Broadcasting Engine in C (Rank <= 8)
  if (commonShape.length <= 8 && condition.dtype == DType.boolean) {
    final marker = ScratchArena.marker;
    final cShape = ScratchArena.allocate<ffi.Int>(
      commonShape.length * ffi.sizeOf<ffi.Int>(),
    );
    final cStridesCond = ScratchArena.allocate<ffi.Int>(
      stridesCond.length * ffi.sizeOf<ffi.Int>(),
    );
    final cStridesX = ScratchArena.allocate<ffi.Int>(
      stridesX.length * ffi.sizeOf<ffi.Int>(),
    );
    final cStridesY = ScratchArena.allocate<ffi.Int>(
      stridesY.length * ffi.sizeOf<ffi.Int>(),
    );
    final cStridesRes = ScratchArena.allocate<ffi.Int>(
      resultStrides.length * ffi.sizeOf<ffi.Int>(),
    );

    for (var i = 0; i < commonShape.length; i++) {
      cShape[i] = commonShape[i];
      cStridesCond[i] = stridesCond[i];
      cStridesX[i] = stridesX[i];
      cStridesY[i] = stridesY[i];
      cStridesRes[i] = resultStrides[i];
    }

    try {
      if (targetDType == DType.float64 &&
          x.dtype == DType.float64 &&
          y.dtype == DType.float64) {
        s_where_double(
          condition.pointer.cast(),
          cStridesCond,
          x.pointer.cast(),
          cStridesX,
          y.pointer.cast(),
          cStridesY,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
        );
        return result;
      } else if (targetDType == DType.float32 &&
          x.dtype == DType.float32 &&
          y.dtype == DType.float32) {
        s_where_float(
          condition.pointer.cast(),
          cStridesCond,
          x.pointer.cast(),
          cStridesX,
          y.pointer.cast(),
          cStridesY,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
        );
        return result;
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }

  dispatchWhere(
    result.data,
    condition,
    x,
    y,
    commonShape,
    stridesCond,
    stridesX,
    stridesY,
    resultStrides,
  );

  return result;
}

/// Returns the indices of the elements that are non-zero.
///
/// Returns a `List<NDArray<int>>` containing 1D integer arrays, one for each dimension
/// of [a], which give the coordinates of the non-zero elements along that dimension.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
List<NDArray<int>> nonzero(NDArray a) {
  final rank = a.shape.length;
  final count = count_nonzero<Object>(a as NDArray<Object>).scalar;
  final results = List.generate(
    rank,
    (_) => NDArray<int>.create([count], DType.int32, zeroInit: true),
  );

  if (count == 0 || rank == 0) {
    return results;
  }

  // Convert input to a contiguous boolean mask using type-specialized native C intrinsics!
  final cond = NDArray.scope(() {
    final res = NDArray<bool>.create(a.shape, DType.boolean);
    final marker = ScratchArena.marker;
    try {
      final cShape = ScratchArena.allocate<ffi.Int>(
        rank * ffi.sizeOf<ffi.Int>(),
      );
      final cStrides = ScratchArena.allocate<ffi.Int>(
        rank * ffi.sizeOf<ffi.Int>(),
      );
      for (var i = 0; i < rank; i++) {
        cShape[i] = a.shape[i];
        cStrides[i] = a.strides[i];
      }

      final isContiguousVal = a.isContiguous ? 1 : 0;

      switch (a.dtype) {
        case DType.float64:
          native_to_bool_mask_double(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            res.pointer.cast(),
          );
        case DType.float32:
          native_to_bool_mask_float(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            res.pointer.cast(),
          );
        case DType.int64:
          native_to_bool_mask_int64(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            res.pointer.cast(),
          );
        case DType.int32:
          native_to_bool_mask_int32(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            res.pointer.cast(),
          );
        case DType.complex128:
          native_to_bool_mask_complex128(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            res.pointer.cast(),
          );
        case DType.complex64:
          native_to_bool_mask_complex64(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            res.pointer.cast(),
          );
        case DType.boolean:
        case DType.uint8:
          native_to_bool_mask_uint8(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            res.pointer.cast(),
          );
        case DType.int16:
          native_to_bool_mask_int16(
            a.pointer.cast(),
            a.size,
            cShape,
            cStrides,
            rank,
            isContiguousVal,
            res.pointer.cast(),
          );
      }
    } finally {
      ScratchArena.reset(marker);
    }
    return res.detachToParentScope();
  });

  final marker = ScratchArena.marker;
  try {
    final cShape = ScratchArena.allocate<ffi.Int>(rank * ffi.sizeOf<ffi.Int>());
    final cStrides = ScratchArena.allocate<ffi.Int>(
      rank * ffi.sizeOf<ffi.Int>(),
    );
    for (var i = 0; i < rank; i++) {
      cShape[i] = cond.shape[i];
      cStrides[i] = cond.strides[i];
    }

    final outCoords = ScratchArena.allocate<ffi.Pointer<ffi.Int>>(
      rank * ffi.sizeOf<ffi.Pointer<ffi.Int>>(),
    );
    for (var d = 0; d < rank; d++) {
      outCoords[d] = results[d].pointer.cast<ffi.Int>();
    }

    native_collect_nonzero_coords(
      cond.pointer.cast(),
      cond.size,
      cShape,
      cStrides,
      rank,
      outCoords,
    );
  } finally {
    ScratchArena.reset(marker);
    cond.dispose();
  }

  return results;
}

void _dispatchCountNonzeroFFI(
  ffi.Pointer<ffi.Void> src,
  ffi.Pointer<ffi.Int> stridesSrc,
  ffi.Pointer<ffi.Int> dest,
  ffi.Pointer<ffi.Int> stridesDest,
  ffi.Pointer<ffi.Int> shape,
  int rank,
  int axis,
  int isContig,
  DType dtype,
) {
  switch (dtype) {
    case DType.float64:
      native_count_nonzero_double(
        src,
        stridesSrc,
        dest.cast(),
        stridesDest,
        shape,
        rank,
        axis,
        isContig,
      );
    case DType.float32:
      native_count_nonzero_float(
        src,
        stridesSrc,
        dest.cast(),
        stridesDest,
        shape,
        rank,
        axis,
        isContig,
      );
    case DType.int64:
      native_count_nonzero_int64(
        src,
        stridesSrc,
        dest.cast(),
        stridesDest,
        shape,
        rank,
        axis,
        isContig,
      );
    case DType.int32:
      native_count_nonzero_int32(
        src,
        stridesSrc,
        dest.cast(),
        stridesDest,
        shape,
        rank,
        axis,
        isContig,
      );
    case DType.boolean:
    case DType.uint8:
      native_count_nonzero_uint8(
        src,
        stridesSrc,
        dest.cast(),
        stridesDest,
        shape,
        rank,
        axis,
        isContig,
      );
    case DType.int16:
      native_count_nonzero_int16(
        src,
        stridesSrc,
        dest.cast(),
        stridesDest,
        shape,
        rank,
        axis,
        isContig,
      );
    default:
      throw UnsupportedError('Unsupported data type for count_nonzero: $dtype');
  }
}

/// Count the number of non-zero elements in the array [a].
///
/// If [axis] is provided, counts along that axis and returns a new array.
/// Otherwise, counts all elements globally and returns a 0-dimensional [NDArray]
/// whose value can be accessed via [scalar].
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
NDArray<int> count_nonzero<T>(NDArray<T> a, {int? axis, NDArray<int>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot count non-zero elements on a disposed array.');
  }

  final rank = a.shape.length;

  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));

  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.int32) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (axis == null) {
    // Global flat reduction
    final isContig = a.isContiguous;
    final NDArray<T> src;
    if (!isContig) {
      src = a.copy();
    } else {
      src = a;
    }

    final result = out ?? NDArray<int>.create([], DType.int32);
    final marker = ScratchArena.marker;
    try {
      final cShape = ScratchArena.allocate<ffi.Int>(ffi.sizeOf<ffi.Int>());
      cShape[0] = src.size;
      final cStrides = ScratchArena.allocate<ffi.Int>(ffi.sizeOf<ffi.Int>());
      cStrides[0] = 1;

      _dispatchCountNonzeroFFI(
        src.pointer,
        cStrides,
        result.pointer.cast<ffi.Int>(),
        cStrides,
        cShape,
        1,
        -1,
        1,
        src.dtype,
      );
    } finally {
      ScratchArena.reset(marker);
      if (!isContig) src.dispose();
    }

    return result;
  }

  // Axis reduction
  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(targetAxis, 0, rank - 1, 'axis');
  }

  final result = out ?? NDArray<int>.create(targetShape, DType.int32);

  final marker = ScratchArena.marker;
  try {
    final cShape = ScratchArena.allocate<ffi.Int>(rank * ffi.sizeOf<ffi.Int>());
    final cStridesSrc = ScratchArena.allocate<ffi.Int>(
      rank * ffi.sizeOf<ffi.Int>(),
    );
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesSrc[i] = a.strides[i];
    }

    final rankDest = targetShape.length;
    final cStridesDest = ScratchArena.allocate<ffi.Int>(
      (rankDest > 0 ? rankDest : 1) * ffi.sizeOf<ffi.Int>(),
    );
    for (var i = 0; i < rankDest; i++) {
      cStridesDest[i] = result.strides[i];
    }

    _dispatchCountNonzeroFFI(
      a.pointer,
      cStridesSrc,
      result.pointer.cast<ffi.Int>(),
      cStridesDest,
      cShape,
      rank,
      targetAxis,
      0,
      a.dtype,
    );
  } finally {
    ScratchArena.reset(marker);
  }

  if (out == null) {
    result.detachToParentScope();
  }
  return result;
}

void _dispatchArgMinMaxFFI(
  ffi.Pointer<ffi.Void> src,
  ffi.Pointer<ffi.Int> stridesSrc,
  ffi.Pointer<ffi.Int> dest,
  ffi.Pointer<ffi.Int> stridesDest,
  ffi.Pointer<ffi.Int> shape,
  int rank,
  int axis,
  int isMax,
  int isContig,
  DType dtype,
) {
  switch (dtype) {
    case DType.float64:
      native_argminmax_double(
        src,
        stridesSrc,
        dest.cast(),
        stridesDest,
        shape,
        rank,
        axis,
        isMax,
        isContig,
      );
    case DType.float32:
      native_argminmax_float(
        src,
        stridesSrc,
        dest.cast(),
        stridesDest,
        shape,
        rank,
        axis,
        isMax,
        isContig,
      );
    case DType.int64:
      native_argminmax_int64(
        src,
        stridesSrc,
        dest.cast(),
        stridesDest,
        shape,
        rank,
        axis,
        isMax,
        isContig,
      );
    case DType.int32:
      native_argminmax_int32(
        src,
        stridesSrc,
        dest.cast(),
        stridesDest,
        shape,
        rank,
        axis,
        isMax,
        isContig,
      );
    case DType.boolean:
    case DType.uint8:
      native_argminmax_uint8(
        src,
        stridesSrc,
        dest.cast(),
        stridesDest,
        shape,
        rank,
        axis,
        isMax,
        isContig,
      );
    case DType.int16:
      native_argminmax_int16(
        src,
        stridesSrc,
        dest.cast(),
        stridesDest,
        shape,
        rank,
        axis,
        isMax,
        isContig,
      );
    default:
      throw UnsupportedError('Unsupported data type for argmin/argmax: $dtype');
  }
}

NDArray<int> _argminmaxFFI<T>(
  NDArray<T> a,
  int? axis,
  bool isMax, {
  NDArray<int>? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot calculate reduction on a disposed array.');
  }
  if (a.size == 0) {
    throw ArgumentError('Cannot compute reduction on an empty array.');
  }
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported.');
  }

  final rank = a.shape.length;
  final isMaxVal = isMax ? 1 : 0;

  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis < 0 ? rank + axis : axis));

  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.int32) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (axis == null) {
    // Global flat reduction
    final isContig = a.isContiguous;
    final NDArray<T> src;
    if (!isContig) {
      src = a.copy();
    } else {
      src = a;
    }

    final result = out ?? NDArray<int>.create([], DType.int32);
    final marker = ScratchArena.marker;
    try {
      final cShape = ScratchArena.allocate<ffi.Int>(ffi.sizeOf<ffi.Int>());
      cShape[0] = src.size;
      final cStrides = ScratchArena.allocate<ffi.Int>(ffi.sizeOf<ffi.Int>());
      cStrides[0] = 1;

      _dispatchArgMinMaxFFI(
        src.pointer,
        cStrides,
        result.pointer.cast<ffi.Int>(),
        cStrides, // dummy contiguous dest strides
        cShape,
        1, // dummy rank
        -1, // global reduction flag
        isMaxVal,
        1, // isContig flat
        src.dtype,
      );
    } finally {
      ScratchArena.reset(marker);
      if (!isContig) src.dispose();
    }

    return result;
  }

  // Axis reduction
  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(targetAxis, 0, rank - 1, 'axis');
  }

  final result = out ?? NDArray<int>.create(targetShape, DType.int32);

  final marker = ScratchArena.marker;
  try {
    final cShape = ScratchArena.allocate<ffi.Int>(rank * ffi.sizeOf<ffi.Int>());
    final cStridesSrc = ScratchArena.allocate<ffi.Int>(
      rank * ffi.sizeOf<ffi.Int>(),
    );
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesSrc[i] = a.strides[i];
    }

    final rankDest = targetShape.length;
    final cStridesDest = ScratchArena.allocate<ffi.Int>(
      (rankDest > 0 ? rankDest : 1) * ffi.sizeOf<ffi.Int>(),
    );
    for (var i = 0; i < rankDest; i++) {
      cStridesDest[i] = result.strides[i];
    }

    _dispatchArgMinMaxFFI(
      a.pointer,
      cStridesSrc,
      result.pointer.cast<ffi.Int>(),
      cStridesDest,
      cShape,
      rank,
      targetAxis,
      isMaxVal,
      0, // axis-based strided reduction
      a.dtype,
    );
  } finally {
    ScratchArena.reset(marker);
  }

  if (out == null) {
    result.detachToParentScope();
  }
  return result;
}

/// Returns the indices of the maximum values along an [axis].
///
/// If [axis] is null, flattens the array and returns a flat 0-dimensional [NDArray]
/// whose value can be accessed via [scalar].
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
NDArray<int> argmax<T>(NDArray<T> a, {int? axis, NDArray<int>? out}) {
  return _argminmaxFFI<T>(a, axis, true, out: out);
}

/// Returns the indices of the minimum values along an [axis].
///
/// If [axis] is null, flattens the array and returns a flat 0-dimensional [NDArray]
/// whose value can be accessed via [scalar].
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
NDArray<int> argmin<T>(NDArray<T> a, {int? axis, NDArray<int>? out}) {
  return _argminmaxFFI<T>(a, axis, false, out: out);
}
