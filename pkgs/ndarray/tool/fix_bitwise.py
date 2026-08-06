import os

workspace_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
path = os.path.join(workspace_root, "lib/src/operations/math/bitwise.dart")

with open(path, "r") as f:
    content = f.read()

# Finding 9: castNDArray
old_cast = """  // Upcast inputs if they do not match the resolved target integer type
  NDArray aCast = a;
  if (a.dtype != targetDType) {
    aCast = NDArray.fromList(a.toList(), a.shape, targetDType);
  }

  NDArray bCast = b;
  if (b.dtype != targetDType) {
    bCast = NDArray.fromList(b.toList(), b.shape, targetDType);
  }"""
new_cast = """  // Upcast inputs if they do not match the resolved target integer type
  final NDArray aCast = a.dtype != targetDType ? castNDArray(a, targetDType) : a;
  final NDArray bCast = b.dtype != targetDType ? castNDArray(b, targetDType) : b;"""
assert old_cast in content, "old_cast not found"
content = content.replace(old_cast, new_cast)

ops = [
    ("bitwise_and", "s_bitwise_and"),
    ("bitwise_or", "s_bitwise_or"),
    ("bitwise_xor", "s_bitwise_xor"),
    ("left_shift", "s_left_shift"),
    ("right_shift", "s_right_shift"),
]

for op_name, prefix in ops:
    old_block = f"""    }} else {{
      final rank = prep.commonShape.length;
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesB = cBuffer + (rank * 2);
      final cStridesRes = cBuffer + (rank * 3);

      for (var i = 0; i < rank; i++) {{
        cShape[i] = prep.commonShape[i];
        cStridesA[i] = prep.stridesA[i];
        cStridesB[i] = prep.stridesB[i];
        cStridesRes[i] = prep.result.strides[i];
      }}

      switch (result.dtype) {{
        case DType.int32:
          {prefix}_int32(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
        case DType.int64:
          {prefix}_int64(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
        case DType.uint8:
          {prefix}_uint8(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
        case DType.int16:
          {prefix}_int16(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${{result.dtype}}');
      }}
    }}"""

    new_block = f"""    }} else {{
      final rank = prep.commonShape.length;
      final marker = ScratchArena.marker;
      try {{
        final cBuffer = ScratchArena.getStridedBuffer(rank);
        final cShape = cBuffer;
        final cStridesA = cBuffer + rank;
        final cStridesB = cBuffer + (rank * 2);
        final cStridesRes = cBuffer + (rank * 3);

        for (var i = 0; i < rank; i++) {{
          cShape[i] = prep.commonShape[i];
          cStridesA[i] = prep.stridesA[i];
          cStridesB[i] = prep.stridesB[i];
          cStridesRes[i] = prep.result.strides[i];
        }}

        switch (result.dtype) {{
          case DType.int32:
            {prefix}_int32(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.int64:
            {prefix}_int64(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.uint8:
            {prefix}_uint8(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          case DType.int16:
            {prefix}_int16(
              aCast.pointer.cast(),
              cStridesA,
              bCast.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              maskHolder.pointer,
            );
          default:
            throw UnsupportedError('Unsupported integer DType: ${{result.dtype}}');
        }}
      }} finally {{
        ScratchArena.reset(marker);
      }}
    }}"""
    assert old_block in content, f"old_block for {op_name} not found"
    content = content.replace(old_block, new_block, 1)

# bitwise_not (invert)
old_not = """  } else {
    final rank = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesSrc = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);

    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesSrc[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }

    switch (a.dtype) {
      case DType.int32:
        s_invert_int32(
          a.pointer.cast(),
          cStridesSrc,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
      case DType.int64:
        s_invert_int64(
          a.pointer.cast(),
          cStridesSrc,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
      case DType.uint8:
        s_invert_uint8(
          a.pointer.cast(),
          cStridesSrc,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
      case DType.int16:
        s_invert_int16(
          a.pointer.cast(),
          cStridesSrc,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
      default:
        throw UnsupportedError('Unsupported integer DType: ${a.dtype}');
    }
  }"""

new_not = """  } else {
    final rank = a.shape.length;
    final marker = ScratchArena.marker;
    try {
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesSrc = cBuffer + rank;
      final cStridesRes = cBuffer + (rank * 2);

      for (var i = 0; i < rank; i++) {
        cShape[i] = a.shape[i];
        cStridesSrc[i] = a.strides[i];
        cStridesRes[i] = result.strides[i];
      }

      switch (a.dtype) {
        case DType.int32:
          s_invert_int32(
            a.pointer.cast(),
            cStridesSrc,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
        case DType.int64:
          s_invert_int64(
            a.pointer.cast(),
            cStridesSrc,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
        case DType.uint8:
          s_invert_uint8(
            a.pointer.cast(),
            cStridesSrc,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
        case DType.int16:
          s_invert_int16(
            a.pointer.cast(),
            cStridesSrc,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${a.dtype}');
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }"""
assert old_not in content, "old_not not found"
content = content.replace(old_not, new_not, 1)

with open(path, "w") as f:
    f.write(content)
print("Updated bitwise.dart successfully")
