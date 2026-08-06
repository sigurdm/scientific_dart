import os

workspace_root = os.path.dirname(os.path.abspath(__file__)) + "/.."
path = os.path.join(workspace_root, "lib/src/operations/math/arithmetic.dart")

with open(path, "r") as f:
    content = f.read()

# 1. sqrt complex branch
old_sqrt = """  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
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
    if (a.dtype == DType.complex128) {
      s_sqrt_complex128(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        maskHolder.pointer,
      );
    } else {
      s_sqrt_complex64(
        a.pointer.cast(),
        cStridesA,
        result.pointer.cast(),
        cStridesRes,
        cShape,
        rank,
        maskHolder.pointer,
      );
    }
    return result;
  }"""

new_sqrt = """  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
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
      if (a.dtype == DType.complex128) {
        s_sqrt_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
      } else {
        s_sqrt_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          maskHolder.pointer,
        );
      }
      return result;
    } finally {
      ScratchArena.reset(marker);
    }
  }"""
assert old_sqrt in content, "old_sqrt not found"
content = content.replace(old_sqrt, new_sqrt, 1)

# 2. expm1
old_expm1 = """    final rank = a.shape.length;
    if (rank <= 8) {
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
          s_expm1_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.float32:
          s_expm1_float(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.complex128:
          s_expm1_complex128(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.complex64:
          s_expm1_complex64(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        default:
          break;
      }
    }"""

new_expm1 = """    final rank = a.shape.length;
    if (rank <= 8) {
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
            s_expm1_double(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          case DType.float32:
            s_expm1_float(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          case DType.complex128:
            s_expm1_complex128(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          case DType.complex64:
            s_expm1_complex64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          default:
            break;
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }"""
assert old_expm1 in content, "old_expm1 not found"
content = content.replace(old_expm1, new_expm1, 1)

# 3. log1p
old_log1p = """    final rank = a.shape.length;
    if (rank <= 8) {
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
          s_log1p_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.float32:
          s_log1p_float(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.complex128:
          s_log1p_complex128(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.complex64:
          s_log1p_complex64(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        default:
          break;
      }
    }"""

new_log1p = """    final rank = a.shape.length;
    if (rank <= 8) {
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
            s_log1p_double(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          case DType.float32:
            s_log1p_float(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          case DType.complex128:
            s_log1p_complex128(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          case DType.complex64:
            s_log1p_complex64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          default:
            break;
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }"""
assert old_log1p in content, "old_log1p not found"
content = content.replace(old_log1p, new_log1p, 1)

# 4. rint
old_rint = """    final rank = a.shape.length;
    if (rank <= 8) {
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
          s_rint_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.float32:
          s_rint_float(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        default:
          break;
      }
    }"""

new_rint = """    final rank = a.shape.length;
    if (rank <= 8) {
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
            s_rint_double(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          case DType.float32:
            s_rint_float(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          default:
            break;
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }"""
assert old_rint in content, "old_rint not found"
content = content.replace(old_rint, new_rint, 1)

# 5. trunc
old_trunc = """    final rank = a.shape.length;
    if (rank <= 8) {
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
          s_trunc_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.float32:
          s_trunc_float(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        default:
          break;
      }
    }"""

new_trunc = """    final rank = a.shape.length;
    if (rank <= 8) {
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
            s_trunc_double(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          case DType.float32:
            s_trunc_float(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          default:
            break;
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }"""
assert old_trunc in content, "old_trunc not found"
content = content.replace(old_trunc, new_trunc, 1)

# 6. square (Finding 2 validation + Finding 1 ScratchArena)
old_square = """NDArray<T> square<T>(NDArray<T> a, {NDArray<dynamic>? where, NDArray<T>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute square() on a disposed array.');
  }
  final result = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for square.',
      );
    }
  }"""

new_square = """NDArray<T> square<T>(NDArray<T> a, {NDArray<dynamic>? where, NDArray<T>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute square() on a disposed array.');
  }
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for square.',
      );
    }
  }
  final result = out ?? NDArray<T>.create(a.shape, a.dtype);"""
assert old_square in content, "old_square not found"
content = content.replace(old_square, new_square, 1)

old_square_strided = """  } else {
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
        s_square_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        return result;
      case DType.float32:
        s_square_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        return result;
      case DType.int64:
        s_square_int64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        return result;
      case DType.int32:
        s_square_int32(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        return result;
      case DType.complex128:
        s_square_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        return result;
      case DType.complex64:
        s_square_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        return result;
      case DType.boolean:
        a.copy(out: result);
        return result;
      case DType.uint8:
      case DType.int16:
        unaryOp<num, num>(
          result.data as List<num>,
          a.data as List<num>,
          a.shape,
          a.strides,
          result.strides,
          0,
          a.offsetElements,
          result.offsetElements,
          (x) => x * x,
        );
        return result;
    }
  }"""

new_square_strided = """  } else {
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
          s_square_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.float32:
          s_square_float(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.int64:
          s_square_int64(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.int32:
          s_square_int32(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.complex128:
          s_square_complex128(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.complex64:
          s_square_complex64(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.boolean:
          a.copy(out: result);
          return result;
        case DType.uint8:
        case DType.int16:
          unaryOp<num, num>(
            result.data as List<num>,
            a.data as List<num>,
            a.shape,
            a.strides,
            result.strides,
            0,
            a.offsetElements,
            result.offsetElements,
            (x) => x * x,
          );
          return result;
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }"""
assert old_square_strided in content, "old_square_strided not found"
content = content.replace(old_square_strided, new_square_strided, 1)

# 7. reciprocal
old_recip = """  } else {
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
        s_reciprocal_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        return result;
      case DType.float32:
        s_reciprocal_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        return result;
      case DType.complex128:
        s_reciprocal_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        return result;
      case DType.complex64:
        s_reciprocal_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        return result;
      case DType.int64:
        s_reciprocal_int64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        isInt = true;
        break;
      case DType.int32:
        s_reciprocal_int32(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        isInt = true;
        break;
      case DType.int16:
        s_reciprocal_int16(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        isInt = true;
        break;
      case DType.uint8:
        s_reciprocal_uint8(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        isInt = true;
        break;
      default:
        break;
    }
  }"""

new_recip = """  } else {
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
          s_reciprocal_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.float32:
          s_reciprocal_float(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.complex128:
          s_reciprocal_complex128(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.complex64:
          s_reciprocal_complex64(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.int64:
          s_reciprocal_int64(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          isInt = true;
          break;
        case DType.int32:
          s_reciprocal_int32(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          isInt = true;
          break;
        case DType.int16:
          s_reciprocal_int16(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          isInt = true;
          break;
        case DType.uint8:
          s_reciprocal_uint8(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          isInt = true;
          break;
        default:
          break;
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }"""
assert old_recip in content, "old_recip not found"
content = content.replace(old_recip, new_recip, 1)

# 8. positive
old_pos = """  } else {
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
        s_positive_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        return result;
      case DType.float32:
        s_positive_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        return result;
      case DType.complex128:
        s_positive_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        return result;
      case DType.complex64:
        s_positive_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        return result;
      case DType.int64:
        s_positive_int64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        return result;
      case DType.int32:
        s_positive_int32(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        return result;
      case DType.int16:
        s_positive_int16(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        return result;
      case DType.uint8:
        s_positive_uint8(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          ffi.nullptr,
        );
        return result;
      default:
        break;
    }
  }"""

new_pos = """  } else {
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
          s_positive_double(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.float32:
          s_positive_float(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.complex128:
          s_positive_complex128(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.complex64:
          s_positive_complex64(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.int64:
          s_positive_int64(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.int32:
          s_positive_int32(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.int16:
          s_positive_int16(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.uint8:
          s_positive_uint8(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        default:
          break;
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }"""
assert old_pos in content, "old_pos not found"
content = content.replace(old_pos, new_pos, 1)

# 9. floor_divide
old_fldiv = """  } else if (commonShape.length <= 8) {
    final rank = commonShape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesB = cBuffer + (rank * 2);
    final cStridesRes = cBuffer + (rank * 3);
    for (var i = 0; i < rank; i++) {
      cShape[i] = commonShape[i];
      cStridesA[i] = stridesA[i];
      cStridesB[i] = stridesB[i];
      cStridesRes[i] = result.strides[i];
    }
    switch (targetDType) {
      case DType.float64:
        if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
          s_floordiv_double(
            x1.pointer.cast(),
            cStridesA,
            x2.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        }
      case DType.float32:
        if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
          s_floordiv_float(
            x1.pointer.cast(),
            cStridesA,
            x2.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        }
      case DType.int64:
        if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
          s_floordiv_int64(
            x1.pointer.cast(),
            cStridesA,
            x2.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          final err = get_and_reset_division_error();
          if (err == 1) {
            throw UnsupportedError('Integer division by zero');
          }
          return result;
        }
      case DType.int32:
        if (x1.dtype == DType.int32 && x2.dtype == DType.int32) {
          s_floordiv_int32(
            x1.pointer.cast(),
            cStridesA,
            x2.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          final err = get_and_reset_division_error();
          if (err == 1) {
            throw UnsupportedError('Integer division by zero');
          }
          return result;
        }
      default:
        break;
    }
  }"""

new_fldiv = """  } else if (commonShape.length <= 8) {
    final rank = commonShape.length;
    final marker = ScratchArena.marker;
    try {
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesB = cBuffer + (rank * 2);
      final cStridesRes = cBuffer + (rank * 3);
      for (var i = 0; i < rank; i++) {
        cShape[i] = commonShape[i];
        cStridesA[i] = stridesA[i];
        cStridesB[i] = stridesB[i];
        cStridesRes[i] = result.strides[i];
      }
      switch (targetDType) {
        case DType.float64:
          if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
            s_floordiv_double(
              x1.pointer.cast(),
              cStridesA,
              x2.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          }
        case DType.float32:
          if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
            s_floordiv_float(
              x1.pointer.cast(),
              cStridesA,
              x2.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          }
        case DType.int64:
          if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
            s_floordiv_int64(
              x1.pointer.cast(),
              cStridesA,
              x2.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            final err = get_and_reset_division_error();
            if (err == 1) {
              throw UnsupportedError('Integer division by zero');
            }
            return result;
          }
        case DType.int32:
          if (x1.dtype == DType.int32 && x2.dtype == DType.int32) {
            s_floordiv_int32(
              x1.pointer.cast(),
              cStridesA,
              x2.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            final err = get_and_reset_division_error();
            if (err == 1) {
              throw UnsupportedError('Integer division by zero');
            }
            return result;
          }
        default:
          break;
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }"""
assert old_fldiv in content, "old_fldiv not found"
content = content.replace(old_fldiv, new_fldiv, 1)

# 10. remainder
old_rem = """  } else if (commonShape.length <= 8) {
    final rank = commonShape.length;
    final cBuffer = ScratchArena.getStridedBuffer(rank);
    final cShape = cBuffer;
    final cStridesA = cBuffer + rank;
    final cStridesB = cBuffer + (rank * 2);
    final cStridesRes = cBuffer + (rank * 3);
    for (var i = 0; i < rank; i++) {
      cShape[i] = commonShape[i];
      cStridesA[i] = stridesA[i];
      cStridesB[i] = stridesB[i];
      cStridesRes[i] = result.strides[i];
    }
    switch (targetDType) {
      case DType.float64:
        if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
          s_remainder_double(
            x1.pointer.cast(),
            cStridesA,
            x2.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        }
      case DType.float32:
        if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
          s_remainder_float(
            x1.pointer.cast(),
            cStridesA,
            x2.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        }
      case DType.int64:
        if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
          s_remainder_int64(
            x1.pointer.cast(),
            cStridesA,
            x2.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          final err = get_and_reset_division_error();
          if (err == 1) {
            throw UnsupportedError('Integer division by zero');
          }
          return result;
        }
      case DType.int32:
        if (x1.dtype == DType.int32 && x2.dtype == DType.int32) {
          s_remainder_int32(
            x1.pointer.cast(),
            cStridesA,
            x2.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          final err = get_and_reset_division_error();
          if (err == 1) {
            throw UnsupportedError('Integer division by zero');
          }
          return result;
        }
      default:
        break;
    }
  }"""

new_rem = """  } else if (commonShape.length <= 8) {
    final rank = commonShape.length;
    final marker = ScratchArena.marker;
    try {
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesB = cBuffer + (rank * 2);
      final cStridesRes = cBuffer + (rank * 3);
      for (var i = 0; i < rank; i++) {
        cShape[i] = commonShape[i];
        cStridesA[i] = stridesA[i];
        cStridesB[i] = stridesB[i];
        cStridesRes[i] = result.strides[i];
      }
      switch (targetDType) {
        case DType.float64:
          if (x1.dtype == DType.float64 && x2.dtype == DType.float64) {
            s_remainder_double(
              x1.pointer.cast(),
              cStridesA,
              x2.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          }
        case DType.float32:
          if (x1.dtype == DType.float32 && x2.dtype == DType.float32) {
            s_remainder_float(
              x1.pointer.cast(),
              cStridesA,
              x2.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          }
        case DType.int64:
          if (x1.dtype == DType.int64 && x2.dtype == DType.int64) {
            s_remainder_int64(
              x1.pointer.cast(),
              cStridesA,
              x2.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            final err = get_and_reset_division_error();
            if (err == 1) {
              throw UnsupportedError('Integer division by zero');
            }
            return result;
          }
        case DType.int32:
          if (x1.dtype == DType.int32 && x2.dtype == DType.int32) {
            s_remainder_int32(
              x1.pointer.cast(),
              cStridesA,
              x2.pointer.cast(),
              cStridesB,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            final err = get_and_reset_division_error();
            if (err == 1) {
              throw UnsupportedError('Integer division by zero');
            }
            return result;
          }
        default:
          break;
      }
    } finally {
      ScratchArena.reset(marker);
    }
  }"""
assert old_rem in content, "old_rem not found"
content = content.replace(old_rem, new_rem, 1)

# 11. abs
old_abs = """    final rank = a.shape.length;
    if (rank <= 8) {
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
        case DType.complex128:
          s_abs_complex128(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.complex64:
          s_abs_complex64(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.int64:
          s_abs_int64(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.int32:
          s_abs_int32(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.int16:
          s_abs_int16(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        case DType.uint8:
          s_abs_uint8(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            ffi.nullptr,
          );
          return result;
        default:
          break;
      }
    }"""

new_abs = """    final rank = a.shape.length;
    if (rank <= 8) {
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
          case DType.complex128:
            s_abs_complex128(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          case DType.complex64:
            s_abs_complex64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          case DType.int64:
            s_abs_int64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          case DType.int32:
            s_abs_int32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          case DType.int16:
            s_abs_int16(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          case DType.uint8:
            s_abs_uint8(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              ffi.nullptr,
            );
            return result;
          default:
            break;
        }
      } finally {
        ScratchArena.reset(marker);
      }
    }"""
assert old_abs in content, "old_abs not found"
content = content.replace(old_abs, new_abs, 1)

# 12. add, subtract, multiply, divide
old_add_start = """  final ndim = commonShape.length;
  final cBuffer = ScratchArena.getStridedBuffer(ndim);
  final cShape = cBuffer;
  final cStridesA = cBuffer + ndim;
  final cStridesB = cBuffer + (ndim * 2);
  final cStridesRes = cBuffer + (ndim * 3);

  for (var i = 0; i < commonShape.length; i++) {
    cShape[i] = commonShape[i];
    cStridesA[i] = stridesA[i];
    cStridesB[i] = stridesB[i];
    cStridesRes[i] = result.strides[i];
  }"""

new_add_start = """  final ndim = commonShape.length;
  final marker = ScratchArena.marker;
  try {
    final cBuffer = ScratchArena.getStridedBuffer(ndim);
    final cShape = cBuffer;
    final cStridesA = cBuffer + ndim;
    final cStridesB = cBuffer + (ndim * 2);
    final cStridesRes = cBuffer + (ndim * 3);

    for (var i = 0; i < commonShape.length; i++) {
      cShape[i] = commonShape[i];
      cStridesA[i] = stridesA[i];
      cStridesB[i] = stridesB[i];
      cStridesRes[i] = result.strides[i];
    }"""
assert old_add_start in content, "old_add_start not found"
content = content.replace(old_add_start, new_add_start, 1)

old_add_end = """        commonShape.length,
        maskHolder.pointer,
      );
      return result;
  }
  // ignore: dead_code
  throw UnsupportedError('Unsupported operand types');
}

/// Element-wise subtraction of two arrays."""

new_add_end = """        commonShape.length,
        maskHolder.pointer,
      );
      return result;
  }
  } finally {
    ScratchArena.reset(marker);
  }
  // ignore: dead_code
  throw UnsupportedError('Unsupported operand types');
}

/// Element-wise subtraction of two arrays."""
assert old_add_end in content, "old_add_end not found"
content = content.replace(old_add_end, new_add_end, 1)

# For subtract:
assert old_add_start in content, "old_sub_start not found"
content = content.replace(old_add_start, new_add_start, 1)

old_sub_end = """        commonShape.length,
        maskHolder.pointer,
      );
      return result;
  }
  // ignore: dead_code
  throw UnsupportedError('Unsupported operand types');
}

/// Element-wise multiplication of two arrays."""

new_sub_end = """        commonShape.length,
        maskHolder.pointer,
      );
      return result;
  }
  } finally {
    ScratchArena.reset(marker);
  }
  // ignore: dead_code
  throw UnsupportedError('Unsupported operand types');
}

/// Element-wise multiplication of two arrays."""
assert old_sub_end in content, "old_sub_end not found"
content = content.replace(old_sub_end, new_sub_end, 1)

# For multiply:
assert old_add_start in content, "old_mul_start not found"
content = content.replace(old_add_start, new_add_start, 1)

old_mul_end = """        commonShape.length,
        maskHolder.pointer,
      );
      return result;
  }
  // ignore: dead_code
  throw UnsupportedError('Unsupported operand types');
}

/// Element-wise true division of two arrays."""

new_mul_end = """        commonShape.length,
        maskHolder.pointer,
      );
      return result;
  }
  } finally {
    ScratchArena.reset(marker);
  }
  // ignore: dead_code
  throw UnsupportedError('Unsupported operand types');
}

/// Element-wise true division of two arrays."""
assert old_mul_end in content, "old_mul_end not found"
content = content.replace(old_mul_end, new_mul_end, 1)

# For divide:
assert old_add_start in content, "old_div_start not found"
content = content.replace(old_add_start, new_add_start, 1)

old_div_end = """        commonShape.length,
        maskHolder.pointer,
      );
      return result;
  }
  // ignore: dead_code
  throw UnsupportedError('Unsupported operand types');
}"""

new_div_end = """        commonShape.length,
        maskHolder.pointer,
      );
      return result;
  }
  } finally {
    ScratchArena.reset(marker);
  }
  // ignore: dead_code
  throw UnsupportedError('Unsupported operand types');
}"""
assert old_div_end in content, "old_div_end not found"
content = content.replace(old_div_end, new_div_end, 1)

with open(path, "w") as f:
    f.write(content)
print("Updated arithmetic.dart successfully")
