// ignore_for_file: non_constant_identifier_names
import '../ndarray.dart';
import '../nditer.dart';
import '../scratch_arena.dart';
import 'helpers.dart';

/// Modes for handling out-of-bounds choice indices in [choose].
enum ChooseMode {
  /// Raises a [RangeError] if an index is out of bounds (default).
  raise,

  /// Wraps indices using modulo arithmetic (`(idx % N + N) % N`).
  wrap,

  /// Clamps indices to the valid choice range `[0, N - 1]`.
  clip,
}

/// Helper function to broadcast a list of shapes into a common compatible shape.
List<int> _broadcastMultiShapes(List<List<int>> shapes) {
  if (shapes.isEmpty) return [];
  var maxLen = 0;
  for (final s in shapes) {
    if (s.length > maxLen) maxLen = s.length;
  }
  final result = List<int>.filled(maxLen, 1);
  for (var i = 0; i < maxLen; i++) {
    var maxDim = 1;
    for (final s in shapes) {
      final dim = i < s.length ? s[s.length - 1 - i] : 1;
      if (dim != 1) {
        if (maxDim != 1 && maxDim != dim) {
          throw ArgumentError('Incompatible shapes for broadcasting: $shapes');
        }
        maxDim = dim;
      }
    }
    result[maxLen - 1 - i] = maxDim;
  }
  return result;
}

/// Helper function to map target coordinate to array coordinate based on broadcasting in-place.
void _mapCoordInPlace(
  List<int> targetCoord,
  List<int> arrShape,
  List<int> outCoord,
) {
  final rank = arrShape.length;
  final targetRank = targetCoord.length;
  for (var i = 0; i < rank; i++) {
    final dim = arrShape[rank - 1 - i];
    final targetDim = targetCoord[targetRank - 1 - i];
    outCoord[rank - 1 - i] = dim == 1 ? 0 : targetDim;
  }
}

/// Extracts elements from an array along a specified [axis] using coordinate index arrays.
///
/// This function corresponds to NumPy's `take_along_axis`.
///
/// **Preconditions:**
/// - It is an error if [arr], [indices], or [out] (if provided) is disposed.
/// - It is an error if [arr] and [indices] do not have the same rank (`arr.rank == indices.rank`).
/// - It is an error if [axis] is not within `[-arr.rank, arr.rank - 1]`.
/// - It is an error if non-axis dimensions of [arr] and [indices] are not broadcast-compatible.
/// - It is an error if index values in [indices] are invalid 1D indices along [axis] of [arr].
/// - It is an error if [out] is provided and its shape does not match the target broadcast shape or its dtype does not match [arr.dtype].
///
/// **Throws:**
/// - Throws [StateError] if [arr], [indices], or [out] is disposed.
/// - Throws [ArgumentError] if ranks don't match, shapes are incompatible, or [out] shape/dtype is invalid.
/// - Throws [RangeError] if [axis] or an index value in [indices] is out of bounds.
///
/// **Example:**
/// ```dart
/// final a = NDArray<Float64>.fromList([10, 20, 30, 40, 50, 60], [2, 3], DType.float64);
/// final indices = NDArray<int>.fromList([2, 0, 1, 1], [2, 2], DType.int32);
/// final result = take_along_axis(a, indices, 1);
/// ```
NDArray<T> take_along_axis<T extends Object>(
  NDArray<T> arr,
  NDArray<int> indices,
  int axis, {
  NDArray<T>? out,
}) {
  if (arr.isDisposed || indices.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute take_along_axis on a disposed array.');
  }
  final rank = arr.shape.length;
  if (indices.shape.length != rank) {
    throw ArgumentError(
      'arr and indices must have the same rank (arr.ndim=${arr.shape.length}, indices.ndim=${indices.shape.length})',
    );
  }
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw RangeError.range(normAxis, 0, rank - 1, 'axis');
  }

  final targetShape = List<int>.filled(rank, 0);
  for (var i = 0; i < rank; i++) {
    if (i == normAxis) {
      targetShape[i] = indices.shape[i];
    } else {
      final dimA = arr.shape[i];
      final dimI = indices.shape[i];
      if (dimA != dimI && dimA != 1 && dimI != 1) {
        throw ArgumentError(
          'Incompatible shapes along dimension $i: arr.shape[i]=$dimA vs indices.shape[i]=$dimI',
        );
      }
      targetShape[i] = dimA > dimI ? dimA : dimI;
    }
  }

  if (out != null) {
    if (out.dtype != arr.dtype) {
      throw ArgumentError(
        'out dtype (${out.dtype}) must match arr dtype (${arr.dtype})',
      );
    }
    if (!listEquals(out.shape, targetShape)) {
      throw ArgumentError(
        'out shape (${out.shape}) must match target shape ($targetShape)',
      );
    }
  }

  return NDArray.scope(() {
    final result = out ?? NDArray<T>.create(targetShape, arr.dtype);
    final marker = ScratchArena.marker;
    final idxCoord = List<int>.filled(rank, 0);
    final arrCoord = List<int>.filled(rank, 0);
    final axisSize = arr.shape[normAxis];

    switch (arr.dtype) {
      case DType.float64:
      case DType.float32:
      case DType.int64:
      case DType.int32:
      case DType.int16:
      case DType.uint8:
      case DType.boolean:
      case DType.complex128:
      case DType.complex64:
        final iter = NDIter(result);
        while (iter.moveNext()) {
          final coords = iter.coords;
          _mapCoordInPlace(coords, indices.shape, idxCoord);
          final idxVal = indices.getCell(idxCoord);
          var targetIdx = idxVal < 0 ? idxVal + axisSize : idxVal;
          if (targetIdx < 0 || targetIdx >= axisSize) {
            ScratchArena.reset(marker);
            throw RangeError.range(
              targetIdx,
              0,
              axisSize - 1,
              'index along axis $normAxis',
            );
          }
          _mapCoordInPlace(coords, arr.shape, arrCoord);
          arrCoord[normAxis] = targetIdx;
          final val = arr.getCell(arrCoord);
          result.setCell(coords, val);
        }
        break;
    }

    ScratchArena.reset(marker);
    return result.detachToParentScope();
  });
}

/// Puts values into an array along a specified [axis] using 1D index arrays.
///
/// This function corresponds to NumPy's `put_along_axis`.
/// Modifies [arr] in-place (or writes to [out] if provided).
///
/// **Preconditions:**
/// - It is an error if [arr], [indices], or [values] (or [out] if provided) is disposed.
/// - It is an error if [arr], [indices], and [values] do not have compatible ranks and shapes.
/// - It is an error if [axis] is not within `[-arr.rank, arr.rank - 1]`.
/// - It is an error if index values in [indices] are invalid indices along [axis] of [arr].
/// - It is an error if [out] is provided and its shape does not match [arr.shape] or its dtype does not match [arr.dtype].
///
/// **Throws:**
/// - Throws [StateError] if any input array is disposed.
/// - Throws [ArgumentError] if shapes are incompatible or [out] is invalid.
/// - Throws [RangeError] if [axis] or index values in [indices] are out of bounds.
///
/// **Example:**
/// ```dart
/// final a = NDArray<Float64>.fromList([10, 20, 30, 40, 50, 60], [2, 3], DType.float64);
/// final indices = NDArray<int>.fromList([2, 0, 1, 1], [2, 2], DType.int32);
/// final values = NDArray<Float64>.fromList([99, 88, 77, 66], [2, 2], DType.float64);
/// put_along_axis(a, indices, values, 1);
/// ```
NDArray<T> put_along_axis<T extends Object>(
  NDArray<T> arr,
  NDArray<int> indices,
  Object values,
  int axis, {
  NDArray<T>? out,
}) {
  if (arr.isDisposed || indices.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute put_along_axis on a disposed array.');
  }
  final rank = arr.shape.length;
  if (indices.shape.length != rank) {
    throw ArgumentError('arr and indices must have the same rank');
  }
  final normAxis = axis < 0 ? rank + axis : axis;
  if (normAxis < 0 || normAxis >= rank) {
    throw RangeError.range(normAxis, 0, rank - 1, 'axis');
  }

  return NDArray.scope(() {
    final valuesArr = values is NDArray<T>
        ? values
        : toNDArray(values, arr.dtype);
    if (valuesArr.isDisposed) {
      throw StateError('Cannot execute put_along_axis with disposed values.');
    }

    final NDArray<T> target;
    if (out != null) {
      if (out.dtype != arr.dtype) {
        throw ArgumentError('out dtype must match arr dtype');
      }
      if (!listEquals(out.shape, arr.shape)) {
        throw ArgumentError('out shape must match arr shape');
      }
      if (!identical(out, arr)) {
        arr.copy(out: out);
      }
      target = out;
    } else {
      target = arr;
    }

    final marker = ScratchArena.marker;
    final valCoord = List<int>.filled(valuesArr.shape.length, 0);
    final targetCoord = List<int>.filled(target.shape.length, 0);
    final axisSize = target.shape[normAxis];

    switch (arr.dtype) {
      case DType.float64:
      case DType.float32:
      case DType.int64:
      case DType.int32:
      case DType.int16:
      case DType.uint8:
      case DType.boolean:
      case DType.complex128:
      case DType.complex64:
        final iter = NDIter(indices);
        while (iter.moveNext()) {
          final coords = iter.coords;
          final idxVal = indices.getCell(coords);
          var targetIdx = idxVal < 0 ? idxVal + axisSize : idxVal;
          if (targetIdx < 0 || targetIdx >= axisSize) {
            ScratchArena.reset(marker);
            throw RangeError.range(
              targetIdx,
              0,
              axisSize - 1,
              'index along axis $normAxis',
            );
          }
          _mapCoordInPlace(coords, valuesArr.shape, valCoord);
          final val = valuesArr.getCell(valCoord);
          _mapCoordInPlace(coords, target.shape, targetCoord);
          targetCoord[normAxis] = targetIdx;
          target.setCell(targetCoord, val);
        }
        break;
    }

    ScratchArena.reset(marker);
    return target.detachToParentScope();
  });
}

/// Constructs an array from an index array ([a]) and a list of arrays or scalars ([choices]).
///
/// This function corresponds to NumPy's `choose`.
///
/// **Preconditions:**
/// - It is an error if [a] or any choice item in [choices] (or [out] if provided) is disposed.
/// - It is an error if [choices] is empty.
/// - It is an error if shapes of [a] and all choice items are not broadcast-compatible.
/// - It is an error if index values in [a] are out of bounds and [mode] is [ChooseMode.raise].
/// - It is an error if [out] is provided and its shape does not match the broadcast shape or its dtype does not match resolved choices dtype.
///
/// **Throws:**
/// - Throws [StateError] if any input array is disposed.
/// - Throws [ArgumentError] if [choices] is empty or shapes cannot be broadcast.
/// - Throws [RangeError] if index values in [a] are out of bounds and [mode] is [ChooseMode.raise].
///
/// **Example:**
/// ```dart
/// final choices = [
///   NDArray<Float64>.fromList([0, 1, 2, 3], [2, 2], DType.float64),
///   NDArray<Float64>.fromList([10, 11, 12, 13], [2, 2], DType.float64),
/// ];
/// final a = NDArray<int>.fromList([0, 1, 1, 0], [2, 2], DType.int32);
/// final result = choose(a, choices);
/// ```
NDArray<T> choose<T extends Object>(
  NDArray<int> a,
  List<Object> choices, {
  NDArray<T>? out,
  ChooseMode mode = ChooseMode.raise,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute choose on a disposed array.');
  }
  if (choices.isEmpty) {
    throw ArgumentError('choices list must not be empty');
  }

  return NDArray.scope(() {
    final resolvedDType =
        (out?.dtype) ??
        (() {
          final first = choices.first;
          DType dt = first is NDArray
              ? first.dtype
              : toNDArray(first, DType.float64).dtype;
          for (var i = 1; i < choices.length; i++) {
            final item = choices[i];
            final itemDt = item is NDArray
                ? item.dtype
                : toNDArray(item, DType.float64).dtype;
            dt = resolveDType(dt, itemDt);
          }
          return dt as DType<T>;
        })();

    final choiceArrays = choices
        .map((c) => c is NDArray<T> ? c : toNDArray(c, resolvedDType))
        .toList();

    for (var i = 0; i < choiceArrays.length; i++) {
      if (choiceArrays[i].isDisposed) {
        throw StateError(
          'Cannot execute choose with a disposed choice array at index $i.',
        );
      }
    }

    final allShapes = <List<int>>[a.shape, ...choiceArrays.map((c) => c.shape)];
    final targetShape = _broadcastMultiShapes(allShapes);

    if (out != null) {
      if (out.dtype != resolvedDType) {
        throw ArgumentError('out dtype must match resolved choices dtype');
      }
      if (!listEquals(out.shape, targetShape)) {
        throw ArgumentError(
          'out shape must match broadcast shape ($targetShape)',
        );
      }
    }

    final result = out ?? NDArray<T>.create(targetShape, resolvedDType);
    final nChoices = choiceArrays.length;
    final marker = ScratchArena.marker;
    final aCoord = List<int>.filled(a.shape.length, 0);
    final choiceCoords = choiceArrays
        .map((c) => List<int>.filled(c.shape.length, 0))
        .toList();

    switch (resolvedDType) {
      case DType.float64:
      case DType.float32:
      case DType.int64:
      case DType.int32:
      case DType.int16:
      case DType.uint8:
      case DType.boolean:
      case DType.complex128:
      case DType.complex64:
        final iter = NDIter(result);
        while (iter.moveNext()) {
          final coords = iter.coords;
          _mapCoordInPlace(coords, a.shape, aCoord);
          var idxVal = a.getCell(aCoord);

          switch (mode) {
            case ChooseMode.raise:
              if (idxVal < 0 || idxVal >= nChoices) {
                ScratchArena.reset(marker);
                throw RangeError.range(idxVal, 0, nChoices - 1, 'choice index');
              }
              break;
            case ChooseMode.wrap:
              idxVal = idxVal % nChoices;
              if (idxVal < 0) idxVal += nChoices;
              break;
            case ChooseMode.clip:
              if (idxVal < 0) {
                idxVal = 0;
              } else if (idxVal >= nChoices) {
                idxVal = nChoices - 1;
              }
              break;
          }

          final choiceArr = choiceArrays[idxVal];
          final choiceCoord = choiceCoords[idxVal];
          _mapCoordInPlace(coords, choiceArr.shape, choiceCoord);
          final val = choiceArr.getCell(choiceCoord);
          result.setCell(coords, val);
        }
        break;
    }

    ScratchArena.reset(marker);
    return result.detachToParentScope();
  });
}

/// Returns an array drawn from elements in [choicelist], depending on conditions in [condlist].
///
/// This function corresponds to NumPy's `select`.
///
/// **Preconditions:**
/// - It is an error if any array in [condlist] or item in [choicelist] (or [out] if provided) is disposed.
/// - It is an error if [condlist] and [choicelist] do not have the same non-zero length.
/// - It is an error if [condlist] and [choicelist] items are not broadcast-compatible.
/// - It is an error if [out] is provided and its shape does not match the broadcast shape or its dtype does not match resolved dtype.
///
/// **Throws:**
/// - Throws [StateError] if any input array is disposed.
/// - Throws [ArgumentError] if list lengths don't match, lists are empty, or shapes/dtypes are incompatible.
///
/// **Example:**
/// ```dart
/// final x = NDArray<Float64>.fromList([1, 2, 3, 4, 5], [5], DType.float64);
/// final conds = [x < 2, x > 3];
/// final choices = [x * 10, x * 100];
/// final result = select(conds, choices, defaultValue: -1.0);
/// ```
NDArray<T> select<T extends Object>(
  List<NDArray<bool>> condlist,
  List<Object> choicelist, {
  Object? defaultValue,
  DType<T>? dtype,
  NDArray<T>? out,
}) {
  if (condlist.isEmpty || choicelist.isEmpty) {
    throw ArgumentError('condlist and choicelist must not be empty');
  }
  if (condlist.length != choicelist.length) {
    throw ArgumentError(
      'condlist (${condlist.length}) and choicelist (${choicelist.length}) must have the same length',
    );
  }

  for (var i = 0; i < condlist.length; i++) {
    if (condlist[i].isDisposed) {
      throw StateError(
        'Cannot execute select with a disposed condition array at index $i.',
      );
    }
  }

  return NDArray.scope(() {
    final resolvedDType =
        dtype ??
        (out?.dtype) ??
        (() {
          DType getItemDType(Object item) {
            if (item is NDArray) return item.dtype;
            if (item is int) return DType.int32;
            if (item is bool) return DType.boolean;
            if (item is Complex) return DType.complex128;
            return DType.float64;
          }

          DType dt = getItemDType(choicelist.first);
          for (var i = 1; i < choicelist.length; i++) {
            dt = resolveDType(dt, getItemDType(choicelist[i]));
          }
          if (defaultValue != null) {
            dt = resolveDType(dt, getItemDType(defaultValue));
          }
          return dt as DType<T>;
        })();

    final choiceArrays = choicelist
        .map((c) => c is NDArray<T> ? c : toNDArray(c, resolvedDType))
        .toList();
    for (var i = 0; i < choiceArrays.length; i++) {
      if (choiceArrays[i].isDisposed) {
        throw StateError(
          'Cannot execute select with a disposed choice array at index $i.',
        );
      }
    }

    final defaultValObj = defaultValue ?? 0;
    final defaultArr = defaultValObj is NDArray<T>
        ? defaultValObj
        : toNDArray(defaultValObj, resolvedDType);
    if (defaultArr.isDisposed) {
      throw StateError('Cannot execute select with a disposed default array.');
    }

    final allShapes = <List<int>>[
      ...condlist.map((c) => c.shape),
      ...choiceArrays.map((c) => c.shape),
      defaultArr.shape,
    ];
    final targetShape = _broadcastMultiShapes(allShapes);

    if (out != null) {
      if (out.isDisposed) {
        throw StateError('Cannot use a disposed out array.');
      }
      if (out.dtype != resolvedDType) {
        throw ArgumentError('out dtype must match resolved dtype');
      }
      if (!listEquals(out.shape, targetShape)) {
        throw ArgumentError(
          'out shape must match broadcast shape ($targetShape)',
        );
      }
    }

    final result = out ?? NDArray<T>.create(targetShape, resolvedDType);
    final nConds = condlist.length;
    final marker = ScratchArena.marker;
    final condCoords = condlist
        .map((c) => List<int>.filled(c.shape.length, 0))
        .toList();
    final choiceCoords = choiceArrays
        .map((c) => List<int>.filled(c.shape.length, 0))
        .toList();
    final defaultCoord = List<int>.filled(defaultArr.shape.length, 0);

    switch (resolvedDType) {
      case DType.float64:
      case DType.float32:
      case DType.int64:
      case DType.int32:
      case DType.int16:
      case DType.uint8:
      case DType.boolean:
      case DType.complex128:
      case DType.complex64:
        final iter = NDIter(result);
        while (iter.moveNext()) {
          final coords = iter.coords;
          var selectedIdx = -1;
          for (var i = 0; i < nConds; i++) {
            final condArr = condlist[i];
            final condCoord = condCoords[i];
            _mapCoordInPlace(coords, condArr.shape, condCoord);
            if (condArr.getCell(condCoord)) {
              selectedIdx = i;
              break;
            }
          }

          if (selectedIdx != -1) {
            final choiceArr = choiceArrays[selectedIdx];
            final choiceCoord = choiceCoords[selectedIdx];
            _mapCoordInPlace(coords, choiceArr.shape, choiceCoord);
            final val = choiceArr.getCell(choiceCoord);
            result.setCell(coords, castValue(val, result.dtype));
          } else {
            _mapCoordInPlace(coords, defaultArr.shape, defaultCoord);
            final val = defaultArr.getCell(defaultCoord);
            result.setCell(coords, castValue(val, result.dtype));
          }
        }
        break;
    }

    ScratchArena.reset(marker);
    return result.detachToParentScope();
  });
}
