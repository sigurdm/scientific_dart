import os

workspace_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
path = os.path.join(workspace_root, "lib/src/operations/math/exponential.dart")

with open(path, "r") as f:
    content = f.read()

# exp
old_exp = """  } else {
    final rank = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }

    switch (a.dtype) {
      case DType.float64:
        s_exp_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
        return result;
      case DType.float32:
        s_exp_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
        return result;
      case DType.complex128:
        s_exp_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
        return result;
      case DType.complex64:
        s_exp_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
        return result;
      default:
        break;
    }
  }"""

new_exp = """  } else {
    final rank = a.shape.length;
    final marker = ScratchArena.marker;
    try {
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesRes = cBuffer + (rank * 2);
      for (var i = 0; i < rank; i++) {
        cShape[i] = a.shape[i];
        cStridesA[i] = a.strides[i];
        cStridesRes[i] = result.strides[i];
      }

      switch (a.dtype) {
        case DType.float64:
          s_exp_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          s_exp_float(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          s_exp_complex128(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          s_exp_complex64(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        default:
          break;
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }"""
assert old_exp in content, "old_exp not found"
content = content.replace(old_exp, new_exp, 1)

# log
old_log = """  } else {
    final rank = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }
    switch (a.dtype) {
      case DType.float64:
        s_log_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
        return result;
      case DType.float32:
        s_log_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
        return result;
      case DType.complex128:
        s_log_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
        return result;
      case DType.complex64:
        s_log_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
        return result;
      default:
        break;
    }
  }"""

new_log = """  } else {
    final rank = a.shape.length;
    final marker = ScratchArena.marker;
    try {
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesRes = cBuffer + (rank * 2);
      for (var i = 0; i < rank; i++) {
        cShape[i] = a.shape[i];
        cStridesA[i] = a.strides[i];
        cStridesRes[i] = result.strides[i];
      }
      switch (a.dtype) {
        case DType.float64:
          s_log_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.float32:
          s_log_float(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.complex128:
          s_log_complex128(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        case DType.complex64:
          s_log_complex64(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            maskHolder.pointer,
          );
          return result;
        default:
          break;
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }"""
assert old_log in content, "old_log not found"
content = content.replace(old_log, new_log, 1)

# log2
old_log2 = """  } else {
    final rank = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }

    if (a.dtype == DType.float64) {
      s_log2_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        maskHolder.pointer,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      s_log2_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        maskHolder.pointer,
      );
      return result;
    } else if (a.dtype == DType.complex128) {
      s_log2_complex128(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        maskHolder.pointer,
      );
      return result;
    } else if (a.dtype == DType.complex64) {
      s_log2_complex64(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        maskHolder.pointer,
      );
      return result;
    }
  }"""

new_log2 = """  } else {
    final rank = a.shape.length;
    final marker = ScratchArena.marker;
    try {
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesRes = cBuffer + (rank * 2);
      for (var i = 0; i < rank; i++) {
        cShape[i] = a.shape[i];
        cStridesA[i] = a.strides[i];
        cStridesRes[i] = result.strides[i];
      }

      if (a.dtype == DType.float64) {
        s_log2_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
        return result;
      } else if (a.dtype == DType.float32) {
        s_log2_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
        return result;
      } else if (a.dtype == DType.complex128) {
        s_log2_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
        return result;
      } else if (a.dtype == DType.complex64) {
        s_log2_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
        return result;
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }"""
assert old_log2 in content, "old_log2 not found"
content = content.replace(old_log2, new_log2, 1)

# log10
old_log10 = """  } else {
    final rank = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesRes = cBuffer + (rank * 2);
    for (var i = 0; i < rank; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = a.strides[i];
      cStridesRes[i] = result.strides[i];
    }

    if (a.dtype == DType.float64) {
      s_log10_double(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        maskHolder.pointer,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      s_log10_float(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        maskHolder.pointer,
      );
      return result;
    } else if (a.dtype == DType.complex128) {
      s_log10_complex128(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        maskHolder.pointer,
      );
      return result;
    } else if (a.dtype == DType.complex64) {
      s_log10_complex64(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        maskHolder.pointer,
      );
      return result;
    }
  }"""

new_log10 = """  } else {
    final rank = a.shape.length;
    final marker = ScratchArena.marker;
    try {
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesRes = cBuffer + (rank * 2);
      for (var i = 0; i < rank; i++) {
        cShape[i] = a.shape[i];
        cStridesA[i] = a.strides[i];
        cStridesRes[i] = result.strides[i];
      }

      if (a.dtype == DType.float64) {
        s_log10_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
        return result;
      } else if (a.dtype == DType.float32) {
        s_log10_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
        return result;
      } else if (a.dtype == DType.complex128) {
        s_log10_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
        return result;
      } else if (a.dtype == DType.complex64) {
        s_log10_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
        return result;
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }"""
assert old_log10 in content, "old_log10 not found"
content = content.replace(old_log10, new_log10, 1)

with open(path, "w") as f:
    f.write(content)
print("Updated exponential.dart successfully")
