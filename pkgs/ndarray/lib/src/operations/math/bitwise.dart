// ignore_for_file: non_constant_identifier_names
import 'dart:ffi' as ffi;
import '../../ndarray.dart';
import '../../ndarray_bindings.dart';
import '../../scratch_arena.dart';
import '../broadcasting.dart';
import '../helpers.dart';

/// Computes the bitwise AND of two arrays, element-wise.
///
/// Calculates the bitwise AND of two integer arrays, element-wise.
///
/// **Preconditions:**
/// - [a] and [b] must be integer-typed arrays (`int32`, `int64`, `uint8`, `int16`).
/// - [a] and [b] must not be disposed.
/// - [a] and [b] must be broadcast-compatible.
/// - If provided, [out] must match the broadcasted shape and resolved integer dtype.
///
/// **Throws:**
/// - [StateError] if [a] or [b] is disposed.
/// - [ArgumentError] if [a] or [b] is not integer-typed.
/// - [ArgumentError] if shapes are incompatible for broadcasting.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For contiguous layouts, uses native C vector bitwise kernels.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy bitwise_and](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_and.html)
NDArray<Tr> bitwise_and<Ta, Tb, Tr>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<Tr>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute bitwise_and() on a disposed array.');
  }
  final prep = _prepareBinaryBitwise<Ta, Tb, Tr>(a, b, out, 'bitwise_and');
  final aCast = prep.aCast;
  final bCast = prep.bCast;
  final result = prep.result;

  try {
    if (prep.isContig) {
      final size = aCast.size;
      switch (result.dtype) {
        case DType.int32:
          v_bitwise_and_int32(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int64:
          v_bitwise_and_int64(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.uint8:
          v_bitwise_and_uint8(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int16:
          v_bitwise_and_int16(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    } else {
      final rank = prep.commonShape.length;
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesB = cBuffer + (rank * 2);
      final cStridesRes = cBuffer + (rank * 3);

      for (var i = 0; i < rank; i++) {
        cShape[i] = prep.commonShape[i];
        cStridesA[i] = prep.stridesA[i];
        cStridesB[i] = prep.stridesB[i];
        cStridesRes[i] = prep.result.strides[i];
      }

      switch (result.dtype) {
        case DType.int32:
          s_bitwise_and_int32(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int64:
          s_bitwise_and_int64(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.uint8:
          s_bitwise_and_uint8(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int16:
          s_bitwise_and_int16(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    }
  } finally {
    if (aCast != a) {
      aCast.dispose();
    }
    if (bCast != b) {
      bCast.dispose();
    }
  }

  return result;
}

/// Computes the bitwise OR of two arrays, element-wise.
///
/// Calculates the bitwise OR of two integer arrays, element-wise.
///
/// **Preconditions:**
/// - [a] and [b] must be integer-typed arrays (`int32`, `int64`, `uint8`, `int16`).
/// - [a] and [b] must not be disposed.
/// - [a] and [b] must be broadcast-compatible.
/// - If provided, [out] must match the broadcasted shape and resolved integer dtype.
///
/// **Throws:**
/// - [StateError] if [a] or [b] is disposed.
/// - [ArgumentError] if [a] or [b] is not integer-typed.
/// - [ArgumentError] if shapes are incompatible for broadcasting.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For contiguous layouts, uses native C vector bitwise kernels.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy bitwise_or](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_or.html)
NDArray<Tr> bitwise_or<Ta, Tb, Tr>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<Tr>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute bitwise_or() on a disposed array.');
  }
  final prep = _prepareBinaryBitwise<Ta, Tb, Tr>(a, b, out, 'bitwise_or');
  final aCast = prep.aCast;
  final bCast = prep.bCast;
  final result = prep.result;

  try {
    if (prep.isContig) {
      final size = aCast.size;
      switch (result.dtype) {
        case DType.int32:
          v_bitwise_or_int32(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int64:
          v_bitwise_or_int64(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.uint8:
          v_bitwise_or_uint8(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int16:
          v_bitwise_or_int16(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    } else {
      final rank = prep.commonShape.length;
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesB = cBuffer + (rank * 2);
      final cStridesRes = cBuffer + (rank * 3);

      for (var i = 0; i < rank; i++) {
        cShape[i] = prep.commonShape[i];
        cStridesA[i] = prep.stridesA[i];
        cStridesB[i] = prep.stridesB[i];
        cStridesRes[i] = prep.result.strides[i];
      }

      switch (result.dtype) {
        case DType.int32:
          s_bitwise_or_int32(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int64:
          s_bitwise_or_int64(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.uint8:
          s_bitwise_or_uint8(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int16:
          s_bitwise_or_int16(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    }
  } finally {
    if (aCast != a) {
      aCast.dispose();
    }
    if (bCast != b) {
      bCast.dispose();
    }
  }

  return result;
}

/// Computes the bitwise XOR of two arrays, element-wise.
///
/// Calculates the bitwise XOR of two integer arrays, element-wise.
///
/// **Preconditions:**
/// - [a] and [b] must be integer-typed arrays (`int32`, `int64`, `uint8`, `int16`).
/// - [a] and [b] must not be disposed.
/// - [a] and [b] must be broadcast-compatible.
/// - If provided, [out] must match the broadcasted shape and resolved integer dtype.
///
/// **Throws:**
/// - [StateError] if [a] or [b] is disposed.
/// - [ArgumentError] if [a] or [b] is not integer-typed.
/// - [ArgumentError] if shapes are incompatible for broadcasting.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For contiguous layouts, uses native C vector bitwise kernels.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy bitwise_xor](https://numpy.org/doc/stable/reference/generated/numpy.bitwise_xor.html)
NDArray<Tr> bitwise_xor<Ta, Tb, Tr>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<Tr>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute bitwise_xor() on a disposed array.');
  }
  final prep = _prepareBinaryBitwise<Ta, Tb, Tr>(a, b, out, 'bitwise_xor');
  final aCast = prep.aCast;
  final bCast = prep.bCast;
  final result = prep.result;

  try {
    if (prep.isContig) {
      final size = aCast.size;
      switch (result.dtype) {
        case DType.int32:
          v_bitwise_xor_int32(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int64:
          v_bitwise_xor_int64(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.uint8:
          v_bitwise_xor_uint8(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int16:
          v_bitwise_xor_int16(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    } else {
      final rank = prep.commonShape.length;
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesB = cBuffer + (rank * 2);
      final cStridesRes = cBuffer + (rank * 3);

      for (var i = 0; i < rank; i++) {
        cShape[i] = prep.commonShape[i];
        cStridesA[i] = prep.stridesA[i];
        cStridesB[i] = prep.stridesB[i];
        cStridesRes[i] = prep.result.strides[i];
      }

      switch (result.dtype) {
        case DType.int32:
          s_bitwise_xor_int32(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int64:
          s_bitwise_xor_int64(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.uint8:
          s_bitwise_xor_uint8(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int16:
          s_bitwise_xor_int16(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    }
  } finally {
    if (aCast != a) {
      aCast.dispose();
    }
    if (bCast != b) {
      bCast.dispose();
    }
  }

  return result;
}

/// Shift the bits of an integer to the left, element-wise.
///
/// Bits are shifted to the left by appending 0s at the right.
///
/// **Preconditions:**
/// - [a] and [b] must be integer-typed arrays (`int32`, `int64`, `uint8`, `int16`).
/// - [a] and [b] must not be disposed.
/// - [a] and [b] must be broadcast-compatible.
/// - If provided, [out] must match the broadcasted shape and resolved integer dtype.
///
/// **Throws:**
/// - [StateError] if [a] or [b] is disposed.
/// - [ArgumentError] if [a] or [b] is not integer-typed.
/// - [ArgumentError] if shapes are incompatible for broadcasting.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For contiguous layouts, uses native C vector bitwise kernels.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy left_shift](https://numpy.org/doc/stable/reference/generated/numpy.left_shift.html)
NDArray<Tr> left_shift<Ta, Tb, Tr>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<Tr>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute left_shift() on a disposed array.');
  }
  final prep = _prepareBinaryBitwise<Ta, Tb, Tr>(a, b, out, 'left_shift');
  final aCast = prep.aCast;
  final bCast = prep.bCast;
  final result = prep.result;

  try {
    if (prep.isContig) {
      final size = aCast.size;
      switch (result.dtype) {
        case DType.int32:
          v_left_shift_int32(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int64:
          v_left_shift_int64(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.uint8:
          v_left_shift_uint8(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int16:
          v_left_shift_int16(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    } else {
      final rank = prep.commonShape.length;
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesB = cBuffer + (rank * 2);
      final cStridesRes = cBuffer + (rank * 3);

      for (var i = 0; i < rank; i++) {
        cShape[i] = prep.commonShape[i];
        cStridesA[i] = prep.stridesA[i];
        cStridesB[i] = prep.stridesB[i];
        cStridesRes[i] = prep.result.strides[i];
      }

      switch (result.dtype) {
        case DType.int32:
          s_left_shift_int32(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int64:
          s_left_shift_int64(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.uint8:
          s_left_shift_uint8(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int16:
          s_left_shift_int16(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    }
  } finally {
    if (aCast != a) {
      aCast.dispose();
    }
    if (bCast != b) {
      bCast.dispose();
    }
  }

  return result;
}

/// Shift the bits of an integer to the right, element-wise.
///
/// Bits are shifted to the right.
///
/// **Preconditions:**
/// - [a] and [b] must be integer-typed arrays (`int32`, `int64`, `uint8`, `int16`).
/// - [a] and [b] must not be disposed.
/// - [a] and [b] must be broadcast-compatible.
/// - If provided, [out] must match the broadcasted shape and resolved integer dtype.
///
/// **Throws:**
/// - [StateError] if [a] or [b] is disposed.
/// - [ArgumentError] if [a] or [b] is not integer-typed.
/// - [ArgumentError] if shapes are incompatible for broadcasting.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For contiguous layouts, uses native C vector bitwise kernels.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy right_shift](https://numpy.org/doc/stable/reference/generated/numpy.right_shift.html)
NDArray<Tr> right_shift<Ta, Tb, Tr>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<Tr>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute right_shift() on a disposed array.');
  }
  final prep = _prepareBinaryBitwise<Ta, Tb, Tr>(a, b, out, 'right_shift');
  final aCast = prep.aCast;
  final bCast = prep.bCast;
  final result = prep.result;

  try {
    if (prep.isContig) {
      final size = aCast.size;
      switch (result.dtype) {
        case DType.int32:
          v_right_shift_int32(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int64:
          v_right_shift_int64(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.uint8:
          v_right_shift_uint8(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        case DType.int16:
          v_right_shift_int16(
            aCast.pointer.cast(),
            bCast.pointer.cast(),
            result.pointer.cast(),
            size,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    } else {
      final rank = prep.commonShape.length;
      final cBuffer = ScratchArena.getStridedBuffer(rank);
      final cShape = cBuffer;
      final cStridesA = cBuffer + rank;
      final cStridesB = cBuffer + (rank * 2);
      final cStridesRes = cBuffer + (rank * 3);

      for (var i = 0; i < rank; i++) {
        cShape[i] = prep.commonShape[i];
        cStridesA[i] = prep.stridesA[i];
        cStridesB[i] = prep.stridesB[i];
        cStridesRes[i] = prep.result.strides[i];
      }

      switch (result.dtype) {
        case DType.int32:
          s_right_shift_int32(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int64:
          s_right_shift_int64(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.uint8:
          s_right_shift_uint8(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        case DType.int16:
          s_right_shift_int16(
            aCast.pointer.cast(),
            cStridesA,
            bCast.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
        default:
          throw UnsupportedError('Unsupported integer DType: ${result.dtype}');
      }
    }
  } finally {
    if (aCast != a) {
      aCast.dispose();
    }
    if (bCast != b) {
      bCast.dispose();
    }
  }

  return result;
}

/// Computes bitwise inversion, or bitwise NOT, element-wise.
///
/// Calculates the bitwise NOT of an integer array, element-wise.
///
/// **Preconditions:**
/// - [a] must be an integer-typed array (`int32`, `int64`, `uint8`, `int16`).
/// - [a] must not be disposed.
/// - If provided, [out] must match the shape and dtype of [a].
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [a] is not integer-typed.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For contiguous layouts, uses native C vector bitwise kernels.
///
/// **Example:**
/// {@example /example/bitwise_example.dart lang=dart}
///
/// Reference: [NumPy invert](https://numpy.org/doc/stable/reference/generated/numpy.invert.html)
NDArray<Tr> invert<Ta, Tr>(NDArray<Ta> a, {NDArray<Tr>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute invert() on a disposed array.');
  }

  if (!a.dtype.isInteger) {
    throw ArgumentError(
      'Bitwise operations are only supported for integer data types.',
    );
  }

  final NDArray<Tr> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for invert.',
      );
    }
    result = out;
  } else {
    result = NDArray<Tr>.create(a.shape, a.dtype as DType<Tr>);
  }

  if (a.isContiguous && result.isContiguous) {
    final size = a.size;
    switch (a.dtype) {
      case DType.int32:
        v_invert_int32(a.pointer.cast(), result.pointer.cast(), size);
      case DType.int64:
        v_invert_int64(a.pointer.cast(), result.pointer.cast(), size);
      case DType.uint8:
        v_invert_uint8(a.pointer.cast(), result.pointer.cast(), size);
      case DType.int16:
        v_invert_int16(a.pointer.cast(), result.pointer.cast(), size);
      default:
        throw UnsupportedError('Unsupported integer DType: ${a.dtype}');
    }
  } else {
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
        );
      case DType.int64:
        s_invert_int64(
          a.pointer.cast(),
          cStridesSrc,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
      case DType.uint8:
        s_invert_uint8(
          a.pointer.cast(),
          cStridesSrc,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
      case DType.int16:
        s_invert_int16(
          a.pointer.cast(),
          cStridesSrc,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
      default:
        throw UnsupportedError('Unsupported integer DType: ${a.dtype}');
    }
  }

  return result;
}

({
  NDArray aCast,
  NDArray bCast,
  NDArray<Tr> result,
  List<int> commonShape,
  List<int> stridesA,
  List<int> stridesB,
  bool isContig,
})
_prepareBinaryBitwise<Ta, Tb, Tr>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  NDArray<Tr>? out,
  String opName,
) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot perform $opName on disposed arrays.');
  }

  if (!a.dtype.isInteger || !b.dtype.isInteger) {
    throw ArgumentError(
      'Bitwise operations are only supported for integer data types.',
    );
  }

  final DType targetDType = resolveDType(a.dtype, b.dtype);

  // Upcast inputs if they do not match the resolved target integer type
  NDArray aCast = a;
  if (a.dtype != targetDType) {
    aCast = NDArray.fromList(a.toList(), a.shape, targetDType);
  }

  NDArray bCast = b;
  if (b.dtype != targetDType) {
    bCast = NDArray.fromList(b.toList(), b.shape, targetDType);
  }

  final broadcastResult = broadcast(aCast, bCast);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final NDArray<Tr> result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for $opName.',
      );
    }
    result = out;
  } else {
    result = NDArray<Tr>.create(commonShape, targetDType as DType<Tr>);
  }

  final isContig =
      aCast.isContiguous &&
      bCast.isContiguous &&
      result.isContiguous &&
      listEquals(aCast.shape, bCast.shape);

  return (
    aCast: aCast,
    bCast: bCast,
    result: result,
    commonShape: commonShape,
    stridesA: stridesA,
    stridesB: stridesB,
    isContig: isContig,
  );
}
