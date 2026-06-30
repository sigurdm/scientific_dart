// ignore_for_file: non_constant_identifier_names
import "dart:math" as math;
import "../ndarray.dart";
import "math.dart";
import "linalg.dart";
import "stats.dart";
import "helpers.dart";

bool _hasRepeatedLabels(String s) {
  final seen = <String>{};
  for (var i = 0; i < s.length; i++) {
    final char = s[i];
    if (seen.contains(char)) return true;
    seen.add(char);
  }
  return false;
}

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
/// - `axes` can be an [int] `N`: contracts the last `N` axes of [a] and first `N` axes of [b].
/// - `axes` can be a 2-element list `[axesA, axesB]` or record `(List<int>, List<int>)` specifying matching contracted axes.
///
/// It is an error if any input or output array is disposed, if the axes specification is invalid,
/// if axis dimensions mismatch, or if a specified axis index is out of bounds.
NDArray<R> tensordot<Ta, Tb, R>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  Object axes = 2,
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

  List<int> axesA = [];
  List<int> axesB = [];

  if (axes is int) {
    final n = axes;
    if (n < 0 || n > a.shape.length || n > b.shape.length) {
      throw ArgumentError(
        "Invalid number of axes $n for shapes ${a.shape} and ${b.shape}.",
      );
    }
    axesA = List.generate(n, (i) => a.shape.length - n + i);
    axesB = List.generate(n, (i) => i);
  } else if (axes is (List<int>, List<int>)) {
    axesA = List<int>.from(axes.$1);
    axesB = List<int>.from(axes.$2);
  } else if (axes is (List<dynamic>, List<dynamic>)) {
    axesA = axes.$1.map((e) => (e as num).toInt()).toList();
    axesB = axes.$2.map((e) => (e as num).toInt()).toList();
  } else if (axes is List) {
    if (axes.length == 2 &&
        axes[0] is int &&
        axes[1] is int &&
        a.shape.length == 1 &&
        b.shape.length == 1) {
      axesA = [axes[0] as int];
      axesB = [axes[1] as int];
    } else if (axes.length == 2 && axes[0] is List && axes[1] is List) {
      axesA = (axes[0] as List).map((e) => (e as num).toInt()).toList();
      axesB = (axes[1] as List).map((e) => (e as num).toInt()).toList();
    } else {
      throw ArgumentError("Unsupported axes format: $axes");
    }
  } else {
    throw ArgumentError("Unsupported axes format: $axes");
  }

  if (axesA.length != axesB.length) {
    throw ArgumentError(
      "Axes length mismatch: ${axesA.length} vs ${axesB.length}.",
    );
  }

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
/// Subscripts define how input tensor axes map to contracted or output dimensions.
///
/// Use one of the three primary constructors to create an [EinsumSubscripts] instance:
/// - [EinsumSubscripts.parse]: Parses a standard Einstein summation string (e.g. `'ij,jk->ik'`).
/// - [EinsumSubscripts.explicit]: Builds an explicit subscript from structured operand and output index lists.
/// - [EinsumSubscripts.implicit]: Builds an implicit subscript from structured operand index lists.
final class EinsumSubscripts {
  /// The list of index labels for each operand.
  final List<List<String>> operandSubscripts;

  /// The list of index labels for the output tensor, or `null` if implicit output.
  final List<String>? outputSubscript;

  /// Creates a raw [EinsumSubscripts] instance with operand and optional output index lists.
  const EinsumSubscripts(this.operandSubscripts, {this.outputSubscript});

  /// 1. Parses a subscript string in Einstein summation notation (e.g. `'ij,jk->ik'` or `'i,j->ij'`).
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

    final operandSubs = inStr.isEmpty
        ? <List<String>>[]
        : inStr.split(',').map((s) => _tokenizeSubscriptTerm(s)).toList();

    final outSubs = outStr != null ? _tokenizeSubscriptTerm(outStr) : null;

    return EinsumSubscripts(operandSubs, outputSubscript: outSubs);
  }

  /// 2. Creates an explicit [EinsumSubscripts] specification with defined operand and output index lists.
  factory EinsumSubscripts.explicit(
    List<List<Object>> operandSubscripts,
    List<Object> outputSubscript,
  ) {
    final opSubs = operandSubscripts
        .map((list) => list.map((e) => _normalizeLabel(e)).toList())
        .toList();
    final outSub = outputSubscript.map((e) => _normalizeLabel(e)).toList();
    return EinsumSubscripts(opSubs, outputSubscript: outSub);
  }

  /// 3. Creates an implicit [EinsumSubscripts] specification where output labels are automatically inferred.
  factory EinsumSubscripts.implicit(List<List<Object>> operandSubscripts) {
    final opSubs = operandSubscripts
        .map((list) => list.map((e) => _normalizeLabel(e)).toList())
        .toList();
    return EinsumSubscripts(opSubs, outputSubscript: null);
  }

  /// Creates an [EinsumSubscripts] from a string, a list of index lists, or an existing [EinsumSubscripts] instance.
  factory EinsumSubscripts.from(Object subscripts, {List<Object>? output}) {
    if (subscripts is EinsumSubscripts) {
      if (output != null) {
        return EinsumSubscripts(
          subscripts.operandSubscripts,
          outputSubscript: output.map((e) => e.toString()).toList(),
        );
      }
      return subscripts;
    }
    if (subscripts is String) {
      final parsed = EinsumSubscripts.parse(subscripts);
      if (output != null) {
        return EinsumSubscripts(
          parsed.operandSubscripts,
          outputSubscript: output.map((e) => e.toString()).toList(),
        );
      }
      return parsed;
    }
    if (subscripts is List) {
      final operandSubs = <List<String>>[];
      for (final item in subscripts) {
        if (item is List) {
          operandSubs.add(item.map((e) => _normalizeLabel(e)).toList());
        } else if (item is String) {
          operandSubs.add(_tokenizeSubscriptTerm(item));
        } else {
          throw ArgumentError('Invalid subscript operand element: $item');
        }
      }
      final outSub = output?.map((e) => _normalizeLabel(e)).toList();
      return EinsumSubscripts(operandSubs, outputSubscript: outSub);
    }
    throw ArgumentError('Unsupported subscripts format: $subscripts');
  }

  static List<String> _tokenizeSubscriptTerm(String term) {
    final labels = <String>[];
    for (var i = 0; i < term.length; i++) {
      if (i + 2 < term.length && term.substring(i, i + 3) == '...') {
        labels.add('...');
        i += 2;
      } else {
        labels.add(term[i]);
      }
    }
    return labels;
  }

  static String _normalizeLabel(Object label) {
    if (label is int) {
      if (label >= 0 && label < 26) {
        return String.fromCharCode(97 + label);
      }
      return 'idx_$label';
    }
    return label.toString();
  }

  /// Converts this subscript specification back into a standard Einstein summation string.
  String toSubscriptString() {
    final inStr = operandSubscripts.map((list) => list.join('')).join(',');
    if (outputSubscript != null) {
      return '$inStr->${outputSubscript!.join('')}';
    }
    return inStr;
  }

  @override
  String toString() => toSubscriptString();
}

/// Evaluates Einstein summation convention over multiple multi-dimensional array operands.
///
/// Einstein summation provides a concise notation for expressing complex tensor contractions,
/// matrix multiplications, transpositions, traces, and array operations by specifying index labels
/// for each input operand and the resulting output.
///
/// ### Einstein Summation Convention
/// In standard Einstein summation notation:
/// 1. Each dimension of an input tensor is assigned an index label (symbol).
/// 2. Labels that appear in multiple input operands but **not** in the output specification are **contracted**
///    (element-wise multiplied and summed over).
/// 3. Labels that appear in the output specification are preserved in the result tensor in the specified order.
/// 4. Repeated labels within a single input operand represent extracting diagonal elements along those axes.
///
/// Subscripts are specified using an [EinsumSubscripts] object created via one of its constructors:
/// - [EinsumSubscripts.parse]: Parses a standard string (e.g. `EinsumSubscripts.parse('ij,jk->ik')`).
/// - [EinsumSubscripts.explicit]: Explicit input and output index lists (e.g. `EinsumSubscripts.explicit([['i', 'j'], ['j', 'k']], ['i', 'k'])`).
/// - [EinsumSubscripts.implicit]: Implicit input index lists (e.g. `EinsumSubscripts.implicit([['i', 'j'], ['j', 'k']])`).
///
/// It is an error if operands is empty, if any operand or [out] is disposed, or if subscript syntax or shapes are invalid.
///
/// ### References & Further Reading
/// - [NumPy einsum Documentation](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
/// - [Wikipedia: Einstein Notation](https://en.wikipedia.org/wiki/Einstein_notation)
/// - [Einsum is All You Need](https://rockt.ai/2018/04/30/einsum)

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

  final cleanSub = subscripts.toSubscriptString();

  late final String inStr;
  late final String outStr;
  final isExplicit = cleanSub.contains("->");

  final parts = cleanSub.split("->");
  if (parts.length > 2) {
    throw ArgumentError(
      'Invalid einsum subscript: contains multiple "->" delimiters.',
    );
  }
  if (isExplicit) {
    inStr = parts[0];
    outStr = parts[1];
  } else {
    inStr = cleanSub;
    outStr = "";
  }

  final rawOperandSubs = inStr.split(",");
  if (rawOperandSubs.length != operands.length) {
    throw ArgumentError(
      "Number of subscript terms (${rawOperandSubs.length}) does not match number of operands (${operands.length}).",
    );
  }

  var hasEllipsis = false;
  for (final sub in rawOperandSubs) {
    if (sub.contains("...")) hasEllipsis = true;
  }
  if (outStr.contains("...")) hasEllipsis = true;

  final operandSubs = <String>[];
  String finalOutStr = outStr;

  if (hasEllipsis) {
    int maxEllipsisDims = 0;
    for (var i = 0; i < operands.length; i++) {
      final sub = rawOperandSubs[i];
      final explicitCount = sub.replaceAll("...", "").length;
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

    final usedChars = <String>{};
    for (final sub in rawOperandSubs) {
      usedChars.addAll(sub.replaceAll("...", "").split(""));
    }
    usedChars.addAll(outStr.replaceAll("...", "").split(""));

    final ellipsisChars = <String>[];
    var codePoint = 0x03B1; // Greek small letter alpha
    while (ellipsisChars.length < maxEllipsisDims) {
      final ch = String.fromCharCode(codePoint);
      if (!usedChars.contains(ch)) {
        ellipsisChars.add(ch);
      }
      codePoint++;
    }
    final ellipsisStr = ellipsisChars.join("");

    for (var i = 0; i < rawOperandSubs.length; i++) {
      final sub = rawOperandSubs[i];
      if (sub.contains("...")) {
        final explicitCount = sub.replaceAll("...", "").length;
        final count = operands[i].shape.length - explicitCount;
        final activeEllipsis = ellipsisChars.sublist(maxEllipsisDims - count);
        operandSubs.add(sub.replaceFirst("...", activeEllipsis.join("")));
      } else {
        operandSubs.add(sub);
      }
    }
    if (isExplicit) {
      finalOutStr = outStr.replaceFirst("...", ellipsisStr);
    } else {
      final labelCounts = <String, int>{};
      for (final sub in operandSubs) {
        for (final ch in sub.split("")) {
          labelCounts[ch] = (labelCounts[ch] ?? 0) + 1;
        }
      }
      final singleLabels =
          labelCounts.entries
              .where((e) => e.value == 1 && !ellipsisChars.contains(e.key))
              .map((e) => e.key)
              .toList()
            ..sort();
      finalOutStr = ellipsisStr + singleLabels.join("");
    }
  } else {
    operandSubs.addAll(rawOperandSubs);
    if (!isExplicit) {
      final labelCounts = <String, int>{};
      for (final sub in operandSubs) {
        for (final ch in sub.split("")) {
          labelCounts[ch] = (labelCounts[ch] ?? 0) + 1;
        }
      }
      final singleLabels =
          labelCounts.entries
              .where((e) => e.value == 1)
              .map((e) => e.key)
              .toList()
            ..sort();
      finalOutStr = singleLabels.join("");
    }
  }

  for (var i = 0; i < operands.length; i++) {
    if (operandSubs[i].length != operands[i].shape.length) {
      throw ArgumentError(
        "Operand $i shape ${operands[i].shape} rank does not match subscript ${operandSubs[i]} length (${operandSubs[i].length}).",
      );
    }
  }

  return NDArray.scope(() {
    final labelSizes = <String, int>{};
    for (var i = 0; i < operands.length; i++) {
      final sub = operandSubs[i];
      final shape = operands[i].shape;
      for (var j = 0; j < sub.length; j++) {
        final ch = sub[j];
        final size = shape[j];
        if (labelSizes.containsKey(ch)) {
          if (labelSizes[ch] != size) {
            throw ArgumentError(
              "Dimension mismatch for label $ch: ${labelSizes[ch]} vs $size",
            );
          }
        } else {
          labelSizes[ch] = size;
        }
      }
    }

    if (operands.length == 1) {
      var op = _asTyped<Object>(operands[0]);
      var sub = operandSubs[0];

      final seenInOp = <String, int>{};
      for (var j = 0; j < sub.length; j++) {
        final ch = sub[j];
        if (seenInOp.containsKey(ch)) {
          final firstAx = seenInOp[ch]!;
          final secondAx = j;
          op = _diagonalView(op, firstAx, secondAx);
          final newSubChars = sub.split("");
          newSubChars.removeAt(secondAx);
          sub = newSubChars.join("");
          j--;
        } else {
          seenInOp[ch] = j;
        }
      }

      final outChars = finalOutStr.split("");
      final remainingChars = sub.split("");
      final axesToSum = <int>[];
      final keptChars = <String>[];

      for (var j = 0; j < remainingChars.length; j++) {
        final ch = remainingChars[j];
        if (!outChars.contains(ch)) {
          axesToSum.add(j);
        } else {
          keptChars.add(ch);
        }
      }

      NDArray res = op;
      axesToSum.sort((a, b) => b.compareTo(a));
      for (final ax in axesToSum) {
        res = sum<Object>(res as NDArray<Object>, axis: ax);
      }

      if (keptChars.length > 1) {
        final perm = <int>[];
        for (final ch in outChars) {
          perm.add(keptChars.indexOf(ch));
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
      final charsA = subA.split("");
      final charsB = subB.split("");
      final outChars = finalOutStr.split("");

      if (!_hasRepeatedLabels(subA) && !_hasRepeatedLabels(subB)) {
        final shared = charsA.where((ch) => charsB.contains(ch)).toList();
        final contracted = shared
            .where((ch) => !outChars.contains(ch))
            .toList();
        final batch = shared.where((ch) => outChars.contains(ch)).toList();

        if (batch.isEmpty) {
          final axesA = contracted.map((ch) => charsA.indexOf(ch)).toList();
          final axesB = contracted.map((ch) => charsB.indexOf(ch)).toList();

          final freeA = charsA.where((ch) => !contracted.contains(ch)).toList();
          final freeB = charsB.where((ch) => !contracted.contains(ch)).toList();

          final allFreeInOut =
              freeA.every((ch) => outChars.contains(ch)) &&
              freeB.every((ch) => outChars.contains(ch));

          if (allFreeInOut) {
            final tdResChars = [...freeA, ...freeB];

            final tdRes = tensordot<dynamic, dynamic, R>(
              operands[0],
              operands[1],
              axes: (axesA, axesB),
            );

            var finalRes = tdRes;
            if (!listEquals(tdResChars, outChars)) {
              final perm = outChars
                  .map((ch) => tdResChars.indexOf(ch))
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

    final allLabelsSet = <String>{};
    for (final ch in finalOutStr.split("")) {
      if (ch.isNotEmpty) allLabelsSet.add(ch);
    }
    for (final sub in operandSubs) {
      for (final ch in sub.split("")) {
        if (ch.isNotEmpty) allLabelsSet.add(ch);
      }
    }
    final allLabels = allLabelsSet.toList();

    final expandedOperands = <NDArray>[];
    for (var i = 0; i < operands.length; i++) {
      var op = _asTyped<Object>(operands[i]);
      var sub = operandSubs[i];

      final seen = <String, int>{};
      for (var j = 0; j < sub.length; j++) {
        final ch = sub[j];
        if (seen.containsKey(ch)) {
          final firstAx = seen[ch]!;
          op = _diagonalView(op, firstAx, j);
          final chars = sub.split("");
          chars.removeAt(j);
          sub = chars.join("");
          j--;
        } else {
          seen[ch] = j;
        }
      }

      final opChars = sub.split("");
      final expShape = <int>[];
      final perm = <int>[];

      for (final lbl in allLabels) {
        if (opChars.contains(lbl)) {
          expShape.add(labelSizes[lbl]!);
          perm.add(opChars.indexOf(lbl));
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

    final outChars = finalOutStr.split("");
    for (var j = allLabels.length - 1; j >= 0; j--) {
      final lbl = allLabels[j];
      if (!outChars.contains(lbl)) {
        combined = sum<Object>(combined as NDArray<Object>, axis: j);
      }
    }

    final remainingLabels = allLabels
        .where((lbl) => outChars.contains(lbl))
        .toList();
    if (!listEquals(remainingLabels, outChars)) {
      final perm = outChars.map((lbl) => remainingLabels.indexOf(lbl)).toList();
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
