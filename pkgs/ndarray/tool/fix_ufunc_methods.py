import os

workspace_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
path = os.path.join(workspace_root, "lib/src/operations/math/ufunc_methods.dart")

with open(path, "r") as f:
    content = f.read()

# Add import
old_import = """import 'trigonometric.dart';"""
new_import = """import 'trigonometric.dart';
import '../sorting.dart' show where;"""
assert old_import in content, "old_import not found"
content = content.replace(old_import, new_import, 1)

# Finding 8: _elementwiseMin / _elementwiseMax
old_min_max = """NDArray<R> _elementwiseMin<T extends Object, R extends Object>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<R>? out,
}) {
  final result = (out as NDArray<T>?) ?? NDArray.create(a.shape, a.dtype);
  a.copy(out: result);
  final flatIndices = NDArray<int>.fromList(
    List<int>.generate(a.size, (i) => i),
    [a.size],
    DType.int64,
  );
  final resRavel = result.ravel();
  final bRavel = b.ravel();
  atUfunc(resRavel, flatIndices, bRavel, op: BinaryOp.minimum);
  resRavel.dispose();
  bRavel.dispose();
  flatIndices.dispose();
  return result as NDArray<R>;
}

NDArray<R> _elementwiseMax<T extends Object, R extends Object>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<R>? out,
}) {
  final result = (out as NDArray<T>?) ?? NDArray.create(a.shape, a.dtype);
  a.copy(out: result);
  final flatIndices = NDArray<int>.fromList(
    List<int>.generate(a.size, (i) => i),
    [a.size],
    DType.int64,
  );
  final resRavel = result.ravel();
  final bRavel = b.ravel();
  atUfunc(resRavel, flatIndices, bRavel, op: BinaryOp.maximum);
  resRavel.dispose();
  bRavel.dispose();
  flatIndices.dispose();
  return result as NDArray<R>;
}"""

new_min_max = """NDArray<R> _elementwiseMin<T extends Object, R extends Object>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<R>? out,
}) {
  final cond = less(a, b);
  final result = where(cond, a, b, out as NDArray<T>?);
  cond.dispose();
  return result as NDArray<R>;
}

NDArray<R> _elementwiseMax<T extends Object, R extends Object>(
  NDArray<T> a,
  NDArray<T> b, {
  NDArray<R>? out,
}) {
  final cond = greater(a, b);
  final result = where(cond, a, b, out as NDArray<T>?);
  cond.dispose();
  return result as NDArray<R>;
}"""
assert old_min_max in content, "old_min_max not found"
content = content.replace(old_min_max, new_min_max, 1)

# Finding 7: accumulateUfunc slice memory leak
old_accum = """    final sel0 = List<Selector>.generate(
      a.rank,
      (d) => d == normAxis ? Index(0) : Slice(),
    );
    final firstSlice = a.slice(sel0);
    final selRes0 = List<Selector>.generate(
      result.rank,
      (d) => d == normAxis ? Index(0) : Slice(),
    );
    firstSlice.copy(out: result.slice(selRes0));
    firstSlice.dispose();

    for (var i = 1; i < axisLen; i++) {
      final selPrev = List<Selector>.generate(
        result.rank,
        (d) => d == normAxis ? Index(i - 1) : Slice(),
      );
      final prev = result.slice(selPrev);
      final selCurr = List<Selector>.generate(
        a.rank,
        (d) => d == normAxis ? Index(i) : Slice(),
      );
      final curr = a.slice(selCurr);
      final stepRes = binaryUfunc(prev, curr, op: op);
      final selResI = List<Selector>.generate(
        result.rank,
        (d) => d == normAxis ? Index(i) : Slice(),
      );
      stepRes.copy(out: result.slice(selResI));
      prev.dispose();
      curr.dispose();
      stepRes.dispose();
    }"""

new_accum = """    final sel0 = List<Selector>.generate(
      a.rank,
      (d) => d == normAxis ? Index(0) : Slice(),
    );
    final firstSlice = a.slice(sel0);
    final selRes0 = List<Selector>.generate(
      result.rank,
      (d) => d == normAxis ? Index(0) : Slice(),
    );
    final resSlice0 = result.slice(selRes0);
    firstSlice.copy(out: resSlice0);
    resSlice0.dispose();
    firstSlice.dispose();

    for (var i = 1; i < axisLen; i++) {
      final selPrev = List<Selector>.generate(
        result.rank,
        (d) => d == normAxis ? Index(i - 1) : Slice(),
      );
      final prev = result.slice(selPrev);
      final selCurr = List<Selector>.generate(
        a.rank,
        (d) => d == normAxis ? Index(i) : Slice(),
      );
      final curr = a.slice(selCurr);
      final stepRes = binaryUfunc(prev, curr, op: op);
      final selResI = List<Selector>.generate(
        result.rank,
        (d) => d == normAxis ? Index(i) : Slice(),
      );
      final resSliceI = result.slice(selResI);
      stepRes.copy(out: resSliceI);
      resSliceI.dispose();
      prev.dispose();
      curr.dispose();
      stepRes.dispose();
    }"""
assert old_accum in content, "old_accum not found"
content = content.replace(old_accum, new_accum, 1)

# Finding 6: atUfunc 0-D scalar indices + ScratchArena wrap
old_at_start = """  final rankA = a.rank;
  final rankB = b.rank;
  final numIndices = indices.size;

  final cBuffer = ScratchArena.getStridedBuffer(rankA * 2 + rankB * 2);
  final cStridesA = cBuffer;
  final cShapeA = cBuffer + rankA;
  final cStridesB = cBuffer + (rankA * 2);
  final cShapeB = cBuffer + (rankA * 2) + rankB;

  for (var i = 0; i < rankA; i++) {
    cStridesA[i] = a.strides[i];
    cShapeA[i] = a.shape[i];
  }
  for (var i = 0; i < rankB; i++) {
    cStridesB[i] = b.strides[i];
    cShapeB[i] = b.shape[i];
  }"""

new_at_start = """  final rankA = a.rank;
  final rankB = b.rank;
  final numIndices = indices.size;
  final strideIdx = indices.strides.isEmpty ? 1 : indices.strides[0];

  final marker = ScratchArena.marker;
  try {
    final cBuffer = ScratchArena.getStridedBuffer(rankA * 2 + rankB * 2);
    final cStridesA = cBuffer;
    final cShapeA = cBuffer + rankA;
    final cStridesB = cBuffer + (rankA * 2);
    final cShapeB = cBuffer + (rankA * 2) + rankB;

    for (var i = 0; i < rankA; i++) {
      cStridesA[i] = a.strides[i];
      cShapeA[i] = a.shape[i];
    }
    for (var i = 0; i < rankB; i++) {
      cStridesB[i] = b.strides[i];
      cShapeB[i] = b.shape[i];
    }"""
assert old_at_start in content, "old_at_start not found"
content = content.replace(old_at_start, new_at_start, 1)

# Replace indices.strides[0] with strideIdx in atUfunc
content = content.replace("indices.strides[0],", "strideIdx,")

# Add finally { ScratchArena.reset(marker); } at the end of atUfunc
old_at_end = """    case DType.boolean:
      s_at_boolean(
        a.pointer.cast(),
        cStridesA,
        cShapeA,
        rankA,
        indices.pointer.cast(),
        numIndices,
        strideIdx,
        b.pointer.cast(),
        cStridesB,
        cShapeB,
        rankB,
        opCode,
      );
  }
}"""

new_at_end = """    case DType.boolean:
      s_at_boolean(
        a.pointer.cast(),
        cStridesA,
        cShapeA,
        rankA,
        indices.pointer.cast(),
        numIndices,
        strideIdx,
        b.pointer.cast(),
        cStridesB,
        cShapeB,
        rankB,
        opCode,
      );
  }
  } finally {
    ScratchArena.reset(marker);
  }
}"""
assert old_at_end in content, "old_at_end not found"
content = content.replace(old_at_end, new_at_end, 1)

with open(path, "w") as f:
    f.write(content)
print("Updated ufunc_methods.dart")
