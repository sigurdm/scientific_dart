// ignore_for_file: non_constant_identifier_names
import "dart:math" as math;
import "../ndarray.dart";
import "math.dart";
import "linalg.dart";
import "stats.dart";
import "helpers.dart";

NDArray<R> _asTyped<R>(NDArray arr) {
  if (arr is NDArray<R>) return arr;
  return NDArray<R>.view(
    arr as dynamic,
    shape: arr.shape,
    strides: arr.strides,
    offsetElements: arr.offsetElements,
  );
}

NDArray<T> _diagonalView<T>(NDArray<T> arr, int ax1, int ax2) {
  final minAx = math.min(ax1, ax2);
  final maxAx = math.max(ax1, ax2);

  final newShape = <int>[];
  final newStrides = <int>[];

  for (var i = 0; i < arr.shape.length; i++) {
    if (i == minAx) {
      newShape.add(arr.shape[minAx]);
      newStrides.add(arr.strides[minAx] + arr.strides[maxAx]);
    } else if (i != maxAx) {
      newShape.add(arr.shape[i]);
      newStrides.add(arr.strides[i]);
    }
  }

  return NDArray<T>.view(arr, shape: newShape, strides: newStrides);
}

/// Computes tensor dot product along specified axes for arrays [a] and [b].
///
/// Given two tensors [a] and [b] and an array of axes (or integer number of axes),
/// sums products of elements over the specified axes.
///
/// **Axes specification:**
/// Represents contracted axis specifications for tensor dot products ([tensordot]).
///
/// Use one of the three primary constructors to create a [TensordotAxes] instance:
/// - [TensordotAxes.count]: Contracts the last `N` axes of array A with the first `N` axes of array B.
/// - [TensordotAxes.explicit]: Contracts explicit axis lists of array A with corresponding axes of array B.
/// - [TensordotAxes.pair]: Contracts a single pair of axes between array A and array B.
final class TensordotAxes {
  /// The number of contracted axes, or `null` if explicit axis lists are provided.
  final int? count;

  /// Explicit contracted axis indices for array A, or `null` if [count] is used.
  final List<int>? explicitAxesA;

  /// Explicit contracted axis indices for array B, or `null` if [count] is used.
  final List<int>? explicitAxesB;

  /// 1. Contracts the last [n] axes of array A with the first [n] axes of array B.
  ///
  /// It is an error if [n] is negative.
  const TensordotAxes.count(int n)
    : count = n,
      explicitAxesA = null,
      explicitAxesB = null;

  /// 2. Contracts specified axis lists [axesA] of array A with corresponding axes [axesB] of array B.
  ///
  /// It is an error if [axesA] and [axesB] have different lengths.
  const TensordotAxes.explicit(List<int> axesA, List<int> axesB)
    : count = null,
      explicitAxesA = axesA,
      explicitAxesB = axesB;

  /// 3. Contracts a single axis pair [axisA] of array A with [axisB] of array B.
  TensordotAxes.pair(int axisA, int axisB)
    : count = null,
      explicitAxesA = [axisA],
      explicitAxesB = [axisB];

  /// Resolves contracted axis index lists for tensors of rank [rankA] and [rankB].
  (List<int>, List<int>) resolve(int rankA, int rankB) {
    if (count != null) {
      final n = count!;
      if (n < 0 || n > rankA || n > rankB) {
        throw ArgumentError(
          "Invalid number of contracted axes $n for tensor ranks $rankA and $rankB.",
        );
      }
      final axesA = List.generate(n, (i) => rankA - n + i);
      final axesB = List.generate(n, (i) => i);
      return (axesA, axesB);
    }

    final axesA = explicitAxesA!;
    final axesB = explicitAxesB!;
    if (axesA.length != axesB.length) {
      throw ArgumentError(
        "Axes length mismatch: ${axesA.length} vs ${axesB.length}.",
      );
    }
    return (List<int>.from(axesA), List<int>.from(axesB));
  }
}

/// Computes tensor dot product along specified contracted axes for arrays [a] and [b].
///
/// Given two tensors [a] and [b], sums products of elements over specified contracted axes.
///
/// ### Axes Specification
/// Axes are specified using a [TensordotAxes] object created via one of its constructors:
/// - [TensordotAxes.count]: Contracts the last `N` axes of [a] and first `N` axes of [b].
/// - [TensordotAxes.explicit]: Contracts explicit axis index lists (`axesA`, `axesB`).
/// - [TensordotAxes.pair]: Contracts a single pair of axes (`axisA`, `axisB`).
///
/// It is an error if any input or output array is disposed, if axes specifications are invalid,
/// if axis dimensions mismatch, or if a specified axis index is out of bounds.
///
/// ### References & Further Reading
/// - [NumPy tensordot Documentation](https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html)
NDArray<R> tensordot<Ta, Tb, R>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  TensordotAxes axes = const TensordotAxes.count(2),
  NDArray<R>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError("Cannot execute tensordot() on a disposed array.");
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      "Cannot write tensordot result to a disposed output array.",
    );
  }

  final (axesA, axesB) = axes.resolve(a.shape.length, b.shape.length);

  final normAxesA = axesA
      .map((ax) => ax < 0 ? a.shape.length + ax : ax)
      .toList();
  final normAxesB = axesB
      .map((ax) => ax < 0 ? b.shape.length + ax : ax)
      .toList();

  for (var i = 0; i < normAxesA.length; i++) {
    final axA = normAxesA[i];
    final axB = normAxesB[i];
    if (axA < 0 || axA >= a.shape.length) {
      throw RangeError.range(axA, 0, a.shape.length - 1, "axisA");
    }
    if (axB < 0 || axB >= b.shape.length) {
      throw RangeError.range(axB, 0, b.shape.length - 1, "axisB");
    }
    if (a.shape[axA] != b.shape[axB]) {
      throw ArgumentError(
        "Dimension mismatch at contracted axis: ${a.shape[axA]} != ${b.shape[axB]}",
      );
    }
  }

  final freeA = List.generate(
    a.shape.length,
    (i) => i,
  ).where((i) => !normAxesA.contains(i)).toList();
  final freeB = List.generate(
    b.shape.length,
    (i) => i,
  ).where((i) => !normAxesB.contains(i)).toList();

  final targetShape = [
    ...freeA.map((i) => a.shape[i]),
    ...freeB.map((i) => b.shape[i]),
  ];

  final targetDType = resolveDType(a.dtype, b.dtype);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != targetDType) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype (expected shape $targetShape and dtype $targetDType, got shape ${out.shape} and dtype ${out.dtype}).",
      );
    }
  }

  return NDArray.scope(() {
    final aPerm = a.transpose([...freeA, ...normAxesA]);
    final bPerm = b.transpose([...normAxesB, ...freeB]);

    final m = freeA.map((i) => a.shape[i]).fold(1, (x, y) => x * y);
    final k = normAxesA.map((i) => a.shape[i]).fold(1, (x, y) => x * y);
    final n = freeB.map((i) => b.shape[i]).fold(1, (x, y) => x * y);

    final a2D = aPerm.reshape([m, k]);
    final b2D = bPerm.reshape([k, n]);

    final res2D = matmul<Object, Object, R>(
      a2D as NDArray<Object>,
      b2D as NDArray<Object>,
    );
    final res = res2D.reshape(targetShape);

    if (out != null) {
      res.copy(out: out);
      return out;
    }
    return _asTyped<R>(res.detachToParentScope());
  });
}

/// Represents index subscript specifications for Einstein summation ([einsum]).
///
/// Subscripts define how input tensor axes map to contracted or output dimensions using numeric index identifiers.
///
/// Use one of the three primary constructors to create an [EinsumSubscripts] instance:
/// - [EinsumSubscripts.fromIndices]: Construct directly using integer axis IDs. Ellipsis `...` is represented by `-1`.
/// - [EinsumSubscripts.parse]: Parses standard Einstein summation notation strings (e.g. `'ij,jk->ik'`).
/// - [EinsumSubscripts.fromLabels]: Construct using string label lists for input operands and optional output.
final class EinsumSubscripts {
  /// The numeric index IDs for each operand tensor axis. `-1` represents an ellipsis (`...`).
  final List<List<int>> operandIndices;

  /// The numeric index IDs for output tensor axes, or `null` if implicit output.
  final List<int>? outputIndices;

  /// Whether any operand or output subscript contains an ellipsis (`-1`).
  final bool hasEllipsis;

  const EinsumSubscripts._(
    this.operandIndices,
    this.outputIndices,
    this.hasEllipsis,
  );

  /// 1. Creates an [EinsumSubscripts] from explicit integer index lists for operands and optional output.
  ///
  /// Each operand's axes are assigned integer IDs. Contracted axes share matching integer IDs across operands.
  /// Use `-1` to represent an ellipsis (`...`) for batch broadcasting.
  ///
  /// It is an error if [inputIndices] is empty.
  factory EinsumSubscripts.fromIndices(
    List<List<int>> inputIndices, [
    List<int>? outputIndices,
  ]) {
    if (inputIndices.isEmpty) {
      throw ArgumentError('inputIndices cannot be empty.');
    }

    var containsEllipsis = false;
    for (final op in inputIndices) {
      if (op.contains(-1)) containsEllipsis = true;
    }
    if (outputIndices != null && outputIndices.contains(-1)) {
      containsEllipsis = true;
    }

    final copyInputs = inputIndices
        .map((l) => List<int>.unmodifiable(l))
        .toList();
    final copyOutput = outputIndices != null
        ? List<int>.unmodifiable(outputIndices)
        : null;

    return EinsumSubscripts._(
      List<List<int>>.unmodifiable(copyInputs),
      copyOutput,
      containsEllipsis,
    );
  }

  /// 2. Parses a subscript string in Einstein summation notation (e.g. `'ij,jk->ik'`, `'...ij,jk->...ik'`, or `'ii->'`).
  ///
  /// Maps character labels (e.g. `'i'`, `'j'`) to numeric index IDs internally.
  ///
  /// It is an error if subscript syntax is invalid.
  factory EinsumSubscripts.parse(String subscripts) {
    final cleanSub = subscripts.replaceAll(' ', '');
    final parts = cleanSub.split('->');
    if (parts.length > 2) {
      throw ArgumentError(
        'Invalid einsum subscript: multiple "->" delimiters found.',
      );
    }

    final inStr = parts[0];
    final outStr = parts.length == 2 ? parts[1] : null;

    final rawOperandTokens = inStr.isEmpty
        ? <List<String>>[]
        : inStr.split(',').map((s) => _tokenizeTerm(s)).toList();

    final rawOutTokens = outStr != null ? _tokenizeTerm(outStr) : null;

    return EinsumSubscripts.fromLabels(rawOperandTokens, rawOutTokens);
  }

  /// 3. Creates an [EinsumSubscripts] from string label lists for input operands and optional output.
  ///
  /// String labels (e.g. `'i'`, `'j'`, `'k'`) are automatically assigned unique numeric index IDs.
  /// Use `'...'` to represent an ellipsis for batch broadcasting.
  ///
  /// It is an error if [inputLabels] is empty.
  factory EinsumSubscripts.fromLabels(
    List<List<String>> inputLabels, [
    List<String>? outputLabels,
  ]) {
    if (inputLabels.isEmpty) {
      throw ArgumentError('inputLabels cannot be empty.');
    }

    final labelToId = <String, int>{};
    var nextId = 0;

    int getId(String label) {
      if (label == '...') return -1;
      return labelToId.putIfAbsent(label, () => nextId++);
    }

    final numericInputs = inputLabels
        .map((list) => list.map((lbl) => getId(lbl)).toList())
        .toList();

    final numericOutput = outputLabels?.map((lbl) => getId(lbl)).toList();

    return EinsumSubscripts.fromIndices(numericInputs, numericOutput);
  }

  static List<String> _tokenizeTerm(String term) {
    final tokens = <String>[];
    for (var i = 0; i < term.length; i++) {
      if (i + 2 < term.length && term.substring(i, i + 3) == '...') {
        tokens.add('...');
        i += 2;
      } else {
        tokens.add(term[i]);
      }
    }
    return tokens;
  }
}

/// Evaluates Einstein summation convention over multiple multi-dimensional array operands.
///
/// Einstein summation provides a concise notation for expressing complex tensor contractions,
/// matrix multiplications, transpositions, traces, and array operations by specifying index labels
/// for each input operand and the resulting output.
///
/// Subscripts are specified using an [EinsumSubscripts] object created via one of its constructors:
/// - [EinsumSubscripts.fromIndices]: Construct directly using integer axis IDs.
/// - [EinsumSubscripts.parse]: Parses a standard notation string (e.g. `EinsumSubscripts.parse('ij,jk->ik')`).
/// - [EinsumSubscripts.fromLabels]: Construct using string label lists.
///
/// It is an error if operands is empty, if any operand or [out] is disposed, or if subscript syntax or shapes are invalid.
///
/// ### References & Further Reading
/// - [NumPy einsum Documentation](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)

NDArray<R> einsum<T extends Object, R extends Object>(
  EinsumSubscripts subscripts,
  List<NDArray<T>> operands, {
  NDArray<R>? out,
}) {
  if (operands.isEmpty) {
    throw ArgumentError("einsum requires at least one operand.");
  }
  for (var i = 0; i < operands.length; i++) {
    if (operands[i].isDisposed) {
      throw StateError(
        "Cannot execute einsum on disposed operand at index $i.",
      );
    }
  }
  if (out != null && out.isDisposed) {
    throw StateError("Cannot write einsum result to a disposed output array.");
  }

  if (subscripts.operandIndices.length != operands.length) {
    throw ArgumentError(
      "Number of subscript terms (${subscripts.operandIndices.length}) does not match number of operands (${operands.length}).",
    );
  }

  var operandSubs = subscripts.operandIndices;
  List<int>? outSub = subscripts.outputIndices;

  if (subscripts.hasEllipsis) {
    int maxEllipsisDims = 0;
    for (var i = 0; i < operands.length; i++) {
      final sub = operandSubs[i];
      final explicitCount = sub.where((id) => id != -1).length;
      final ellipsisDims = operands[i].shape.length - explicitCount;
      if (ellipsisDims < 0) {
        throw ArgumentError(
          "Operand $i shape ${operands[i].shape} has fewer dimensions than explicit labels in $sub.",
        );
      }
      if (ellipsisDims > maxEllipsisDims) {
        maxEllipsisDims = ellipsisDims;
      }
    }

    final ellipsisIds = List<int>.generate(maxEllipsisDims, (i) => 10000 + i);

    final resolvedOperandSubs = <List<int>>[];
    for (var i = 0; i < operands.length; i++) {
      final sub = operandSubs[i];
      if (sub.contains(-1)) {
        final explicitCount = sub.where((id) => id != -1).length;
        final count = operands[i].shape.length - explicitCount;
        final activeEllipsis = ellipsisIds.sublist(maxEllipsisDims - count);
        final newSub = <int>[];
        for (final id in sub) {
          if (id == -1) {
            newSub.addAll(activeEllipsis);
          } else {
            newSub.add(id);
          }
        }
        resolvedOperandSubs.add(newSub);
      } else {
        resolvedOperandSubs.add(sub);
      }
    }

    List<int> resolvedOutSub;
    if (outSub != null) {
      if (outSub.contains(-1)) {
        final newOut = <int>[];
        for (final id in outSub) {
          if (id == -1) {
            newOut.addAll(ellipsisIds);
          } else {
            newOut.add(id);
          }
        }
        resolvedOutSub = newOut;
      } else {
        resolvedOutSub = outSub;
      }
    } else {
      final labelCounts = <int, int>{};
      for (final sub in resolvedOperandSubs) {
        for (final id in sub) {
          labelCounts[id] = (labelCounts[id] ?? 0) + 1;
        }
      }
      final singleLabels =
          labelCounts.entries
              .where((e) => e.value == 1 && !ellipsisIds.contains(e.key))
              .map((e) => e.key)
              .toList()
            ..sort();
      resolvedOutSub = [...ellipsisIds, ...singleLabels];
    }

    operandSubs = resolvedOperandSubs;
    outSub = resolvedOutSub;
  } else {
    if (outSub == null) {
      final labelCounts = <int, int>{};
      for (final sub in operandSubs) {
        for (final id in sub) {
          labelCounts[id] = (labelCounts[id] ?? 0) + 1;
        }
      }
      final singleLabels =
          labelCounts.entries
              .where((e) => e.value == 1)
              .map((e) => e.key)
              .toList()
            ..sort();
      outSub = singleLabels;
    }
  }

  final finalOutSub = outSub;

  for (var i = 0; i < operands.length; i++) {
    if (operandSubs[i].length != operands[i].shape.length) {
      throw ArgumentError(
        "Operand $i shape ${operands[i].shape} rank does not match subscript ${operandSubs[i]} length (${operandSubs[i].length}).",
      );
    }
  }

  return NDArray.scope(() {
    final labelSizes = <int, int>{};
    for (var i = 0; i < operands.length; i++) {
      final sub = operandSubs[i];
      final shape = operands[i].shape;
      for (var j = 0; j < sub.length; j++) {
        final id = sub[j];
        final size = shape[j];
        if (labelSizes.containsKey(id)) {
          if (labelSizes[id] != size) {
            throw ArgumentError(
              "Dimension mismatch for label ID $id: ${labelSizes[id]} vs $size",
            );
          }
        } else {
          labelSizes[id] = size;
        }
      }
    }

    if (operands.length == 1) {
      var op = _asTyped<Object>(operands[0]);
      var sub = operandSubs[0];

      final seenInOp = <int, int>{};
      for (var j = 0; j < sub.length; j++) {
        final id = sub[j];
        if (seenInOp.containsKey(id)) {
          final firstAx = seenInOp[id]!;
          final secondAx = j;
          op = _diagonalView(op, firstAx, secondAx);
          final newSub = List<int>.from(sub)..removeAt(secondAx);
          sub = newSub;
          j--;
        } else {
          seenInOp[id] = j;
        }
      }

      final remainingIds = sub;
      final axesToSum = <int>[];
      final keptIds = <int>[];

      for (var j = 0; j < remainingIds.length; j++) {
        final id = remainingIds[j];
        if (!finalOutSub.contains(id)) {
          axesToSum.add(j);
        } else {
          keptIds.add(id);
        }
      }

      NDArray res = op;
      axesToSum.sort((a, b) => b.compareTo(a));
      for (final ax in axesToSum) {
        res = sum<Object>(res as NDArray<Object>, axis: ax);
      }

      if (keptIds.length > 1) {
        final perm = <int>[];
        for (final id in finalOutSub) {
          perm.add(keptIds.indexOf(id));
        }
        res = res.transpose(perm);
      }

      final targetDType = res.dtype;
      if (out != null) {
        if (!listEquals(out.shape, res.shape) || out.dtype != targetDType) {
          throw ArgumentError(
            "Provided out buffer has incompatible shape or dtype (expected shape ${res.shape} and dtype $targetDType, got shape ${out.shape} and dtype ${out.dtype}).",
          );
        }
        res.copy(out: out);
        return out;
      }
      return _asTyped<R>(res.detachToParentScope());
    }

    if (operands.length == 2) {
      final subA = operandSubs[0];
      final subB = operandSubs[1];

      bool hasRepeated(List<int> s) {
        final seen = <int>{};
        for (final id in s) {
          if (seen.contains(id)) return true;
          seen.add(id);
        }
        return false;
      }

      if (!hasRepeated(subA) && !hasRepeated(subB)) {
        final shared = subA.where((id) => subB.contains(id)).toList();
        final contracted = shared
            .where((id) => !finalOutSub.contains(id))
            .toList();
        final batch = shared.where((id) => finalOutSub.contains(id)).toList();

        if (batch.isEmpty) {
          final axesA = contracted.map((id) => subA.indexOf(id)).toList();
          final axesB = contracted.map((id) => subB.indexOf(id)).toList();

          final freeA = subA.where((id) => !contracted.contains(id)).toList();
          final freeB = subB.where((id) => !contracted.contains(id)).toList();

          final allFreeInOut =
              freeA.every((id) => finalOutSub.contains(id)) &&
              freeB.every((id) => finalOutSub.contains(id));

          if (allFreeInOut) {
            final tdResIds = [...freeA, ...freeB];

            final tdRes = tensordot<dynamic, dynamic, R>(
              operands[0],
              operands[1],
              axes: TensordotAxes.explicit(axesA, axesB),
            );

            var finalRes = tdRes;
            if (!listEquals(tdResIds, finalOutSub)) {
              final perm = finalOutSub
                  .map((id) => tdResIds.indexOf(id))
                  .toList();
              finalRes = tdRes.transpose(perm);
            }

            final targetDType = finalRes.dtype;
            if (out != null) {
              if (!listEquals(out.shape, finalRes.shape) ||
                  out.dtype != targetDType) {
                throw ArgumentError(
                  "Provided out buffer has incompatible shape or dtype (expected shape ${finalRes.shape} and dtype $targetDType, got shape ${out.shape} and dtype ${out.dtype}).",
                );
              }
              finalRes.copy(out: out);
              return out;
            }
            return _asTyped<R>(finalRes.detachToParentScope());
          }
        }
      }
    }

    final allIdsSet = <int>{};
    for (final id in finalOutSub) {
      allIdsSet.add(id);
    }
    for (final sub in operandSubs) {
      for (final id in sub) {
        allIdsSet.add(id);
      }
    }
    final allIds = allIdsSet.toList();

    final expandedOperands = <NDArray>[];
    for (var i = 0; i < operands.length; i++) {
      var op = _asTyped<Object>(operands[i]);
      var sub = operandSubs[i];

      final seen = <int, int>{};
      for (var j = 0; j < sub.length; j++) {
        final id = sub[j];
        if (seen.containsKey(id)) {
          final firstAx = seen[id]!;
          op = _diagonalView(op, firstAx, j);
          final newSub = List<int>.from(sub)..removeAt(j);
          sub = newSub;
          j--;
        } else {
          seen[id] = j;
        }
      }

      final expShape = <int>[];
      final perm = <int>[];

      for (final id in allIds) {
        if (sub.contains(id)) {
          expShape.add(labelSizes[id]!);
          perm.add(sub.indexOf(id));
        } else {
          expShape.add(1);
        }
      }

      final opTransposed = op.transpose(perm);
      final opExpanded = opTransposed.reshape(expShape);
      expandedOperands.add(opExpanded);
    }

    NDArray combined = expandedOperands[0];
    for (var i = 1; i < expandedOperands.length; i++) {
      combined = multiply<Object, Object, Object>(
        combined as NDArray<Object>,
        expandedOperands[i] as NDArray<Object>,
      );
    }

    for (var j = allIds.length - 1; j >= 0; j--) {
      final id = allIds[j];
      if (!finalOutSub.contains(id)) {
        combined = sum<Object>(combined as NDArray<Object>, axis: j);
      }
    }

    final remainingIds = allIds
        .where((id) => finalOutSub.contains(id))
        .toList();
    if (!listEquals(remainingIds, finalOutSub)) {
      final perm = finalOutSub.map((id) => remainingIds.indexOf(id)).toList();
      combined = combined.transpose(perm);
    }

    final targetDType = combined.dtype;
    if (out != null) {
      if (!listEquals(out.shape, combined.shape) || out.dtype != targetDType) {
        throw ArgumentError(
          "Provided out buffer has incompatible shape or dtype (expected shape ${combined.shape} and dtype $targetDType, got shape ${out.shape} and dtype ${out.dtype}).",
        );
      }
      combined.copy(out: out);
      return out;
    }
    return _asTyped<R>(combined.detachToParentScope());
  });
}
