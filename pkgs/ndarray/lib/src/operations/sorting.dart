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

  final cKList = malloc<ffi.Int>(uniqueK.length);
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
    malloc.free(cKList);
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

    final cKList = malloc<ffi.Int>(uniqueK.length);
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
      malloc.free(cKList);
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
    final cShape = malloc<ffi.Int>(commonShape.length);
    final cStridesCond = malloc<ffi.Int>(stridesCond.length);
    final cStridesX = malloc<ffi.Int>(stridesX.length);
    final cStridesY = malloc<ffi.Int>(stridesY.length);
    final cStridesRes = malloc<ffi.Int>(resultStrides.length);

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
      malloc.free(cShape);
      malloc.free(cStridesCond);
      malloc.free(cStridesX);
      malloc.free(cStridesY);
      malloc.free(cStridesRes);
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
  final count = count_nonzero<Object>(a as NDArray<Object>) as int;
  final results = List.generate(
    rank,
    (_) => NDArray<int>.create([count], DType.int32, zeroInit: true),
  );

  if (count == 0) {
    return results;
  }

  final shape = a.shape;
  final strides = a.strides;
  final totalSize = shape.isEmpty ? 1 : shape.reduce((x, y) => x * y);

  final coord = List<int>.filled(rank, 0);
  int offset = 0;
  int writeIdx = 0;

  for (int el = 0; el < totalSize; el++) {
    final val = a.data[offset];
    if (isTrueHelper(val)) {
      for (var d = 0; d < rank; d++) {
        results[d].data[writeIdx] = coord[d];
      }
      writeIdx++;
    }

    // Advance odometer multidimensional coordinate odometer walk!
    for (int d = rank - 1; d >= 0; d--) {
      coord[d]++;
      if (coord[d] < shape[d]) {
        offset += strides[d];
        break;
      }
      coord[d] = 0;
      offset -= (shape[d] - 1) * strides[d];
    }
  }

  return results;
}

/// Count the number of non-zero elements in the array [a].
///
/// If [axis] is provided, counts along that axis and returns a new array.
/// Otherwise, counts all elements globally and returns a single scalar integer.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
dynamic count_nonzero<T>(NDArray<T> a, {int? axis}) {
  if (axis == null) {
    var count = 0;
    if (a.isContiguous) {
      for (var i = 0; i < a.data.length; i++) {
        if (isTrueHelper(a.data[i])) count++;
      }
      return count;
    }
    final rank = a.shape.length;
    final pos = List<int>.filled(rank, 0);
    int countWalk(int dim) {
      if (dim == rank) return isTrueHelper(a[pos]) ? 1 : 0;
      var subCount = 0;
      for (var i = 0; i < a.shape[dim]; i++) {
        pos[dim] = i;
        subCount += countWalk(dim + 1);
      }
      return subCount;
    }

    return countWalk(0);
  }

  final rank = a.shape.length;
  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(targetAxis, 0, rank - 1, 'axis');
  }

  final targetShape = List<int>.from(a.shape)..removeAt(targetAxis);
  final result = NDArray<int>.zeros(targetShape, DType.int32);

  countNonzeroRecursive<T>(a, result, List<int>.filled(rank, 0), targetAxis, 0);

  return result;
}

/// Returns the indices of the maximum values along an [axis].
///
/// If [axis] is null, flattens the array and returns a flat scalar integer index.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
dynamic argmax<T>(NDArray<T> a, {int? axis}) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for argmax');
  }

  if (axis == null) {
    NDArray<T> src = a;
    if (!a.isContiguous) {
      src = NDArray<T>.fromList(a.toList(), a.shape, a.dtype);
    }

    var maxIdx = 0;
    if (src.dtype == DType.int32 || src.dtype == DType.int64) {
      final dataList = src.data as List<int>;
      var maxVal = dataList[0];
      for (var i = 1; i < dataList.length; i++) {
        if (dataList[i] > maxVal) {
          maxVal = dataList[i];
          maxIdx = i;
        }
      }
    } else if (src.dtype == DType.boolean) {
      final dataList = src.data as List<bool>;
      var maxVal = dataList[0] ? 1 : 0;
      for (var i = 1; i < dataList.length; i++) {
        final val = dataList[i] ? 1 : 0;
        if (val > maxVal) {
          maxVal = val;
          maxIdx = i;
        }
      }
    } else {
      final dataList = src.data as List<double>;
      var maxVal = dataList[0];
      for (var i = 1; i < dataList.length; i++) {
        if (dataList[i] > maxVal) {
          maxVal = dataList[i];
          maxIdx = i;
        }
      }
    }
    if (src != a) src.dispose();
    return maxIdx;
  }

  final rank = a.shape.length;
  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(targetAxis, 0, rank - 1, 'axis');
  }

  final targetShape = List<int>.from(a.shape)..removeAt(targetAxis);
  final result = NDArray<int>.create(targetShape, DType.int32);
  result.data.fillRange(0, result.data.length, 0);

  argMinMaxRecursive<T>(
    a,
    result,
    List<int>.filled(rank, 0),
    targetAxis,
    0,
    true,
  );
  return result;
}

/// Returns the indices of the minimum values along an [axis].
///
/// If [axis] is null, flattens the array and returns a flat scalar integer index.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
dynamic argmin<T>(NDArray<T> a, {int? axis}) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for argmin');
  }

  if (axis == null) {
    NDArray<T> src = a;
    if (!a.isContiguous) {
      src = NDArray<T>.fromList(a.toList(), a.shape, a.dtype);
    }

    var minIdx = 0;
    if (src.dtype == DType.int32 || src.dtype == DType.int64) {
      final dataList = src.data as List<int>;
      var minVal = dataList[0];
      for (var i = 1; i < dataList.length; i++) {
        if (dataList[i] < minVal) {
          minVal = dataList[i];
          minIdx = i;
        }
      }
    } else if (src.dtype == DType.boolean) {
      final dataList = src.data as List<bool>;
      var minVal = dataList[0] ? 1 : 0;
      for (var i = 1; i < dataList.length; i++) {
        final val = dataList[i] ? 1 : 0;
        if (val < minVal) {
          minVal = val;
          minIdx = i;
        }
      }
    } else {
      final dataList = src.data as List<double>;
      var minVal = dataList[0];
      for (var i = 1; i < dataList.length; i++) {
        if (dataList[i] < minVal) {
          minVal = dataList[i];
          minIdx = i;
        }
      }
    }
    if (src != a) src.dispose();
    return minIdx;
  }

  final rank = a.shape.length;
  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(targetAxis, 0, rank - 1, 'axis');
  }

  final targetShape = List<int>.from(a.shape)..removeAt(targetAxis);
  final result = NDArray<int>.create(targetShape, DType.int32);
  result.data.fillRange(0, result.data.length, 0);

  argMinMaxRecursive<T>(
    a,
    result,
    List<int>.filled(rank, 0),
    targetAxis,
    0,
    false,
  );
  return result;
}
