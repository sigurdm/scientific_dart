// ignore_for_file: non_constant_identifier_names
import 'dart:math' as math;
import 'package:openblas/openblas.dart';
import '../../ndarray.dart';
import '../helpers.dart';


/// Configure the number of parallel execution threads used by OpenBLAS at runtime.
///
/// **Preconditions:**
/// - [numThreads] must be greater than or equal to 1.
///
/// **Throws:**
/// - [ArgumentError] if [numThreads] is less than 1.
///
/// **Example:**
/// ```dart
/// setNumThreads(1); // Disable multi-threading to bypass overhead on small matrices
/// ```
void setNumThreads(int numThreads) {
  if (numThreads < 1) {
    throw ArgumentError(
      'Number of threads must be at least 1 (was $numThreads)',
    );
  }
  openblas_set_num_threads(numThreads);
}

/// Enumerates elements of a multidimensional array yielding coordinates and values.
///
/// Yields records containing the coordinate list and the element value at that coordinate
/// in standard C-contiguous order.
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
///
/// **Throws:**
/// - [StateError] if [a] has been disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.int32);
/// for (final entry in ndenumerate(a)) {
///   print('coord: ${entry.$1}, value: ${entry.$2}');
/// }
/// // Yields:
/// // ([0, 0], 10)
/// // ([0, 1], 20)
/// // ([1, 0], 30)
/// // ([1, 1], 40)
/// ```
Iterable<(List<int> coordinate, T value)> ndenumerate<T>(NDArray<T> a) sync* {
  if (a.isDisposed) {
    throw StateError('Cannot execute ndenumerate() on a disposed array.');
  }

  final shape = a.shape;
  final strides = a.strides;
  final totalSize = shape.isEmpty ? 1 : shape.reduce((x, y) => x * y);

  if (shape.isEmpty) {
    yield ([], a.data[0]);
    return;
  }

  final coord = List<int>.filled(shape.length, 0);
  int offset = 0;

  for (int el = 0; el < totalSize; el++) {
    // Yield a copy of the coordinate list so that users don't receive the same mutated buffer!
    yield (List<int>.from(coord), a.data[offset]);

    // Advance odometer multidimensional coordinate odometer walk!
    for (int d = shape.length - 1; d >= 0; d--) {
      coord[d]++;
      if (coord[d] < shape[d]) {
        offset += strides[d];
        break;
      }
      coord[d] = 0;
      offset -= (shape[d] - 1) * strides[d];
    }
  }
}

/// Replace NaN with zero and infinity with large finite numbers.
///
/// By default, maps NaN to [nan] (which defaults to 0.0), maps positive infinity
/// to [posinf] (or the maximum finite float value if null), and maps negative infinity
/// to [neginf] (or the minimum finite float value if null).
///
/// **Preconditions:**
/// - Input [a] must be a numeric array.
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/nan_to_num_example.dart lang=dart}
///
/// Reference: [Replace NaN and Infinities](https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
NDArray nan_to_num(
  NDArray a, {
  double nan = 0.0,
  double? posinf,
  double? neginf,
  NDArray? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute nan_to_num() on a disposed array.');
  }
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for nan_to_num.',
      );
    }
  }

  final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
  final aList = a.toList();
  final resultCopy = out ?? NDArray.create(a.shape, a.dtype);

  final maxLimit = a.dtype == DType.float32
      ? 3.4028234663852886e+38
      : double.maxFinite;
  final minLimit = -maxLimit;

  final targetPosInf = posinf ?? maxLimit;
  final targetNegInf = neginf ?? minLimit;

  final cleanList = <dynamic>[];

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    final complexList = aList.cast<Complex>();
    for (var i = 0; i < size; i++) {
      var r = complexList[i].real;
      var img = complexList[i].imag;

      if (r.isNaN) r = nan;
      if (r == double.infinity) r = targetPosInf;
      if (r == double.negativeInfinity) r = targetNegInf;

      if (img.isNaN) img = nan;
      if (img == double.infinity) img = targetPosInf;
      if (img == double.negativeInfinity) img = targetNegInf;

      cleanList.add(Complex(r, img));
    }
  } else {
    final numList = aList.cast<num>();
    for (var i = 0; i < size; i++) {
      var val = numList[i].toDouble();

      if (val.isNaN) {
        val = nan;
      } else if (val == double.infinity) {
        val = targetPosInf;
      } else if (val == double.negativeInfinity) {
        val = targetNegInf;
      }

      cleanList.add(val);
    }
  }

  // View-Safe Strided Odometer Write Back!
  final resData = resultCopy.data;
  final resStrides = resultCopy.strides;
  final coord = List<int>.filled(a.shape.length, 0);

  for (var i = 0; i < size; i++) {
    var offsetRes = 0;
    for (var d = 0; d < a.shape.length; d++) {
      offsetRes += coord[d] * resStrides[d];
    }

    resData[offsetRes] = cleanList[i];

    // Advance odometer
    for (var d = a.shape.length - 1; d >= 0; d--) {
      coord[d]++;
      if (coord[d] < a.shape[d]) break;
      coord[d] = 0;
    }
  }

  return resultCopy;
}

/// Computes the broadcasted shape list of two shapes.
List<int> broadcastShapes(List<int> s1, List<int> s2) {
  final len = math.max(s1.length, s2.length);
  final common = List<int>.filled(len, 1);
  for (var i = 0; i < len; i++) {
    final dim1 = s1.length - 1 - i >= 0 ? s1[s1.length - 1 - i] : 1;
    final dim2 = s2.length - 1 - i >= 0 ? s2[s2.length - 1 - i] : 1;

    final target = math.max(dim1, dim2);
    if (dim1 != target && dim1 != 1) {
      throw ArgumentError('Incompatible shapes for broadcasting');
    }
    if (dim2 != target && dim2 != 1) {
      throw ArgumentError('Incompatible shapes for broadcasting');
    }
    common[len - 1 - i] = target;
  }
  return common;
}

/// Returns an array drawn from elements in [choicelist], depending on conditions in [condlist].
///
/// This corresponds to NumPy's `select` function.
///
/// **Mathematical Mechanics**:
/// - Evaluates a list of boolean conditions in [condlist] sequentially per cell.
/// - Draws corresponding values from the same-indexed array in [choicelist].
/// - If no condition is met, falls back to [defaultValue].
/// - Leverages zero-copy, zero-allocation $N$-dimensional strides recursive walk in a single pass!
///
/// **Preconditions:**
/// - [condlist] and [choicelist] must have the same length.
/// - All condition and choice arrays must broadcast perfectly to a common shape.
///
/// **Throws:**
/// - [ArgumentError] if [condlist] and [choicelist] lengths mismatch, or if any shape is incompatible.
///
/// **Example:**
/// ```dart
/// final cond1 = NDArray.fromList([true, false], [2], DType.boolean);
/// final cond2 = NDArray.fromList([false, true], [2], DType.boolean);
/// final choice1 = NDArray.fromList([10, 20], [2], DType.int32);
/// final choice2 = NDArray.fromList([100, 200], [2], DType.int32);
/// final res = select([cond1, cond2], [choice1, choice2], defaultValue: 999);
/// print(res.toList()); // [10, 200]
/// ```
NDArray select(
  List<NDArray<bool>> condlist,
  List<NDArray> choicelist, {
  dynamic defaultValue = 0,
}) {
  if (condlist.isEmpty || choicelist.isEmpty) {
    throw ArgumentError('condlist and choicelist must not be empty');
  }
  if (condlist.length != choicelist.length) {
    throw ArgumentError(
      'condlist length (${condlist.length}) must match choicelist length (${choicelist.length})',
    );
  }

  // 1. Calculate common broadcasted shape
  final allShapes = <List<int>>[];
  for (final c in condlist) {
    allShapes.add(c.shape);
  }
  for (final c in choicelist) {
    allShapes.add(c.shape);
  }

  var commonShape = allShapes[0];
  for (var i = 1; i < allShapes.length; i++) {
    commonShape = broadcastShapes(commonShape, allShapes[i]);
  }

  // 2. Determine target upcasted DType
  var targetDType = choicelist[0].dtype;
  for (var i = 1; i < choicelist.length; i++) {
    targetDType = resolveDType(targetDType, choicelist[i].dtype);
  }
  if (defaultValue is double &&
      !targetDType.isFloating &&
      !targetDType.isComplex) {
    targetDType = DType.float64;
  }

  final result = NDArray.create(commonShape, targetDType);

  // 3. Compute strides for all condition and choice operands independently to commonShape
  final stridesCond = condlist
      .map((c) => broadcastStrides(c, commonShape))
      .toList();
  final stridesChoice = choicelist
      .map((c) => broadcastStrides(c, commonShape))
      .toList();

  // 4. Execute recursive multi-operand strided walk
  final currentPos = List<int>.filled(commonShape.length, 0);
  final initialOffsetsCond = List<int>.filled(condlist.length, 0);
  final initialOffsetsChoice = List<int>.filled(choicelist.length, 0);

  selectRecursive(
    result,
    condlist,
    choicelist,
    stridesCond,
    stridesChoice,
    result.strides,
    currentPos,
    0,
    initialOffsetsCond,
    initialOffsetsChoice,
    0,
    defaultValue,
  );

  return result;
}
