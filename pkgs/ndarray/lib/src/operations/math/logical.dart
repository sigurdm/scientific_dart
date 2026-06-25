// ignore_for_file: non_constant_identifier_names
import 'dart:ffi' as ffi;
import '../../ndarray.dart';
import '../../ndarray_bindings.dart';
import '../../scratch_arena.dart';
import '../broadcasting.dart';


/// Returns a boolean array of the same shape as [a] containing the element-wise logical negation.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - If provided, the [out] recycler array must exactly match [a]'s shape and have [DType.boolean] dtype.
///
/// **Throws:**
/// - [StateError] if the input array [a] has been disposed.
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous layouts, uses native C vector logical kernels (`v_logical_not`).
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
///
/// Reference: [NumPy logical_not](https://numpy.org/doc/stable/reference/generated/numpy.logical_not.html)
NDArray<bool> logical_not<T>(NDArray<T> a, {NDArray<bool>? out}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute logical_not() on a disposed array.');
  }
  final NDArray<bool> result;
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for logical_not.',
      );
    }
    result = out;
  } else {
    result = NDArray<bool>.create(a.shape, DType.boolean);
  }
  final marker = ScratchArena.marker;

  final ffi.Pointer<ffi.Uint8> aBoolPtr;
  final List<int> aBoolStrides;
  if (a.dtype == DType.boolean) {
    aBoolPtr = a.pointer.cast();
    aBoolStrides = a.strides;
  } else {
    aBoolPtr = ScratchArena.allocate<ffi.Uint8>(a.size);
    aBoolStrides = NDArray.computeCStrides(a.shape);
    if (a.isContiguous) {
      switch (a.dtype) {
        case DType.float64:
          v_to_bool_double(a.pointer.cast(), aBoolPtr, a.size);
        case DType.float32:
          v_to_bool_float(a.pointer.cast(), aBoolPtr, a.size);
        case DType.int64:
          v_to_bool_int64(a.pointer.cast(), aBoolPtr, a.size);
        case DType.int32:
          v_to_bool_int32(a.pointer.cast(), aBoolPtr, a.size);
        case DType.uint8:
          v_to_bool_uint8(a.pointer.cast(), aBoolPtr, a.size);
        case DType.int16:
          v_to_bool_int16(a.pointer.cast(), aBoolPtr, a.size);
        case DType.complex128:
          v_to_bool_complex128(a.pointer.cast(), aBoolPtr, a.size);
        case DType.complex64:
          v_to_bool_complex64(a.pointer.cast(), aBoolPtr, a.size);
        case DType.boolean:
          break;
      }
    } else {
      final ndim = a.shape.length;
      final cBuffer = ScratchArena.getStridedBuffer(ndim);
      final cShape = cBuffer;
      final cStridesA = cBuffer + ndim;
      final cStridesTemp = cBuffer + (ndim * 2);
      for (var i = 0; i < ndim; i++) {
        cShape[i] = a.shape[i];
        cStridesA[i] = a.strides[i];
        cStridesTemp[i] = aBoolStrides[i];
      }
      switch (a.dtype) {
        case DType.float64:
          s_to_bool_double(
            a.pointer.cast(),
            cStridesA,
            aBoolPtr,
            cStridesTemp,
            cShape,
            ndim,
          );
        case DType.float32:
          s_to_bool_float(
            a.pointer.cast(),
            cStridesA,
            aBoolPtr,
            cStridesTemp,
            cShape,
            ndim,
          );
        case DType.int64:
          s_to_bool_int64(
            a.pointer.cast(),
            cStridesA,
            aBoolPtr,
            cStridesTemp,
            cShape,
            ndim,
          );
        case DType.int32:
          s_to_bool_int32(
            a.pointer.cast(),
            cStridesA,
            aBoolPtr,
            cStridesTemp,
            cShape,
            ndim,
          );
        case DType.uint8:
          s_to_bool_uint8(
            a.pointer.cast(),
            cStridesA,
            aBoolPtr,
            cStridesTemp,
            cShape,
            ndim,
          );
        case DType.int16:
          s_to_bool_int16(
            a.pointer.cast(),
            cStridesA,
            aBoolPtr,
            cStridesTemp,
            cShape,
            ndim,
          );
        case DType.complex128:
          s_to_bool_complex128(
            a.pointer.cast(),
            cStridesA,
            aBoolPtr,
            cStridesTemp,
            cShape,
            ndim,
          );
        case DType.complex64:
          s_to_bool_complex64(
            a.pointer.cast(),
            cStridesA,
            aBoolPtr,
            cStridesTemp,
            cShape,
            ndim,
          );
        case DType.boolean:
          break;
      }
    }
  }

  if (a.isContiguous && result.isContiguous) {
    v_logical_not(aBoolPtr, result.pointer.cast(), a.size);
  } else {
    final ndim = a.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(ndim);
    final cShape = cBuffer;
    final cStridesA = cBuffer + ndim;
    final cStridesRes = cBuffer + (ndim * 2);

    for (var i = 0; i < ndim; i++) {
      cShape[i] = a.shape[i];
      cStridesA[i] = aBoolStrides[i];
      cStridesRes[i] = result.strides[i];
    }

    s_logical_not(
      aBoolPtr,
      cStridesA,
      result.pointer.cast(),
      cStridesRes,
      cShape,
      ndim,
    );
  }

  ScratchArena.reset(marker);
  return result;
}

/// Element-wise comparison of [a] == [b] with broadcasting and recycling support.
NDArray<bool> equal(NDArray a, NDArray b, {NDArray<bool>? out}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError("Cannot execute equal() on a disposed array.");
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype.",
      );
    }
  }

  final result = out ?? NDArray<bool>.create(commonShape, DType.boolean);

  _compareHelper(
    a,
    b,
    result,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    CMP_OP_EQ,
  );
  return result;
}

NDArray<bool> notEqual(NDArray a, NDArray b, {NDArray<bool>? out}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError("Cannot execute notEqual() on a disposed array.");
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype.",
      );
    }
  }

  final result = out ?? NDArray<bool>.create(commonShape, DType.boolean);

  _compareHelper(
    a,
    b,
    result,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    CMP_OP_NE,
  );
  return result;
}

NDArray<bool> greater(NDArray a, NDArray b, {NDArray<bool>? out}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError("Cannot execute greater() on a disposed array.");
  }
  if (a.dtype.isComplex || b.dtype.isComplex) {
    throw UnsupportedError(
      "Complex numbers do not support inequality comparisons",
    );
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype.",
      );
    }
  }

  final result = out ?? NDArray<bool>.create(commonShape, DType.boolean);

  _compareHelper(
    a,
    b,
    result,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    CMP_OP_GT,
  );
  return result;
}

NDArray<bool> greaterEqual(NDArray a, NDArray b, {NDArray<bool>? out}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError("Cannot execute greaterEqual() on a disposed array.");
  }
  if (a.dtype.isComplex || b.dtype.isComplex) {
    throw UnsupportedError(
      "Complex numbers do not support inequality comparisons",
    );
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype.",
      );
    }
  }

  final result = out ?? NDArray<bool>.create(commonShape, DType.boolean);

  _compareHelper(
    a,
    b,
    result,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    CMP_OP_GE,
  );
  return result;
}

NDArray<bool> less(NDArray a, NDArray b, {NDArray<bool>? out}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError("Cannot execute less() on a disposed array.");
  }
  if (a.dtype.isComplex || b.dtype.isComplex) {
    throw UnsupportedError(
      "Complex numbers do not support inequality comparisons",
    );
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype.",
      );
    }
  }

  final result = out ?? NDArray<bool>.create(commonShape, DType.boolean);

  _compareHelper(
    a,
    b,
    result,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    CMP_OP_LT,
  );
  return result;
}

NDArray<bool> lessEqual(NDArray a, NDArray b, {NDArray<bool>? out}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError("Cannot execute lessEqual() on a disposed array.");
  }
  if (a.dtype.isComplex || b.dtype.isComplex) {
    throw UnsupportedError(
      "Complex numbers do not support inequality comparisons",
    );
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        "Provided out buffer has incompatible shape or dtype.",
      );
    }
  }

  final result = out ?? NDArray<bool>.create(commonShape, DType.boolean);

  _compareHelper(
    a,
    b,
    result,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    CMP_OP_LE,
  );
  return result;
}

void _compareHelper(
  NDArray a,
  NDArray b,
  NDArray<bool> result,
  List<int> stridesA,
  List<int> stridesB,
  int op,
) {
  final rank = result.shape.length;
  final marker = ScratchArena.marker;
  final cShape = result.shape.isEmpty
      ? ffi.nullptr
      : ScratchArena.copyInts(result.shape);
  final cStridesA = stridesA.isEmpty
      ? ffi.nullptr
      : ScratchArena.copyInts(stridesA);
  final cStridesB = stridesB.isEmpty
      ? ffi.nullptr
      : ScratchArena.copyInts(stridesB);
  final cStridesRes = result.strides.isEmpty
      ? ffi.nullptr
      : ScratchArena.copyInts(result.strides);

  try {
    ndarray_compare(
      op,
      a.dtype.index,
      b.dtype.index,
      a.pointer.cast(),
      cStridesA,
      b.pointer.cast(),
      cStridesB,
      result.pointer.cast(),
      cStridesRes,
      cShape,
      rank,
    );
  } finally {
    ScratchArena.reset(marker);
  }
}

/// Computes the element-wise truth value of [a] AND [b] with broadcasting support.
///
/// Returns a boolean array containing the element-wise logical AND results.
///
/// **Preconditions:**
/// - [a] and [b] must not be disposed.
/// - If provided, the [out] recycler array must exactly match the broadcasted shape of [a] and [b], and have [DType.boolean] dtype.
///
/// **Throws:**
/// - [StateError] if any input array has been disposed.
/// - [ArgumentError] if shapes of [a] and [b] are incompatible for broadcasting.
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous layouts, uses native C vector logical kernels (`v_logical_and`).
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
///
/// Reference: [NumPy logical_and](https://numpy.org/doc/stable/reference/generated/numpy.logical_and.html)
NDArray<bool> logical_and<Ta, Tb>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<bool>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute logical_and() on a disposed array.');
  }
  return _runBinaryLogical(
    a,
    b,
    out,
    v_logical_and,
    s_logical_and,
    'logical_and',
  );
}

/// Computes the element-wise truth value of [a] OR [b] with broadcasting support.
///
/// Returns a boolean array containing the element-wise logical OR results.
///
/// **Preconditions:**
/// - [a] and [b] must not be disposed.
/// - If provided, the [out] recycler array must exactly match the broadcasted shape of [a] and [b], and have [DType.boolean] dtype.
///
/// **Throws:**
/// - [StateError] if any input array has been disposed.
/// - [ArgumentError] if shapes of [a] and [b] are incompatible for broadcasting.
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous layouts, uses native C vector logical kernels (`v_logical_or`).
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
///
/// Reference: [NumPy logical_or](https://numpy.org/doc/stable/reference/generated/numpy.logical_or.html)
NDArray<bool> logical_or<Ta, Tb>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<bool>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute logical_or() on a disposed array.');
  }
  return _runBinaryLogical(a, b, out, v_logical_or, s_logical_or, 'logical_or');
}

/// Computes the element-wise truth value of [a] XOR [b] with broadcasting support.
///
/// Returns a boolean array containing the element-wise logical XOR results.
///
/// **Preconditions:**
/// - [a] and [b] must not be disposed.
/// - If provided, the [out] recycler array must exactly match the broadcasted shape of [a] and [b], and have [DType.boolean] dtype.
///
/// **Throws:**
/// - [StateError] if any input array has been disposed.
/// - [ArgumentError] if shapes of [a] and [b] are incompatible for broadcasting.
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous layouts, uses native C vector logical kernels (`v_logical_xor`).
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
///
/// Reference: [NumPy logical_xor](https://numpy.org/doc/stable/reference/generated/numpy.logical_xor.html)
NDArray<bool> logical_xor<Ta, Tb>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  NDArray<bool>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute logical_xor() on a disposed array.');
  }
  return _runBinaryLogical(
    a,
    b,
    out,
    v_logical_xor,
    s_logical_xor,
    'logical_xor',
  );
}

BroadcastResult _broadcastBinaryStrides(
  List<int> shapeA,
  List<int> stridesA,
  List<int> shapeB,
  List<int> stridesB,
) {
  if (listEquals(shapeA, shapeB)) {
    return BroadcastResult(shapeA, stridesA, stridesB);
  }
  final maxLen = shapeA.length > shapeB.length ? shapeA.length : shapeB.length;
  final commonShape = List<int>.filled(maxLen, 1);
  final newStridesA = List<int>.filled(maxLen, 0);
  final newStridesB = List<int>.filled(maxLen, 0);

  for (var i = 0; i < maxLen; i++) {
    final dimA = i < shapeA.length ? shapeA[shapeA.length - 1 - i] : 1;
    final dimB = i < shapeB.length ? shapeB[shapeB.length - 1 - i] : 1;

    if (dimA == dimB) {
      commonShape[maxLen - 1 - i] = dimA;
      if (i < shapeA.length) {
        newStridesA[maxLen - 1 - i] = stridesA[shapeA.length - 1 - i];
      }
      if (i < shapeB.length) {
        newStridesB[maxLen - 1 - i] = stridesB[shapeB.length - 1 - i];
      }
    } else if (dimA == 1) {
      commonShape[maxLen - 1 - i] = dimB;
      if (i < shapeB.length) {
        newStridesB[maxLen - 1 - i] = stridesB[shapeB.length - 1 - i];
      }
    } else if (dimB == 1) {
      commonShape[maxLen - 1 - i] = dimA;
      if (i < shapeA.length) {
        newStridesA[maxLen - 1 - i] = stridesA[shapeA.length - 1 - i];
      }
    } else {
      throw ArgumentError(
        'Shapes $shapeA and $shapeB are not compatible for broadcasting',
      );
    }
  }
  return BroadcastResult(commonShape, newStridesA, newStridesB);
}

ffi.Pointer<ffi.Uint8> _castToBoolean(
  NDArray x,
  ffi.Pointer<ffi.Uint8> destPtr,
  List<int> destStrides,
) {
  if (x.isContiguous) {
    switch (x.dtype) {
      case DType.float64:
        v_to_bool_double(x.pointer.cast(), destPtr, x.size);
      case DType.float32:
        v_to_bool_float(x.pointer.cast(), destPtr, x.size);
      case DType.int64:
        v_to_bool_int64(x.pointer.cast(), destPtr, x.size);
      case DType.int32:
        v_to_bool_int32(x.pointer.cast(), destPtr, x.size);
      case DType.uint8:
        v_to_bool_uint8(x.pointer.cast(), destPtr, x.size);
      case DType.int16:
        v_to_bool_int16(x.pointer.cast(), destPtr, x.size);
      case DType.complex128:
        v_to_bool_complex128(x.pointer.cast(), destPtr, x.size);
      case DType.complex64:
        v_to_bool_complex64(x.pointer.cast(), destPtr, x.size);
      case DType.boolean:
        break;
    }
  } else {
    final ndim = x.shape.length;
    final cBuffer = ScratchArena.getStridedBuffer(ndim);
    final cShape = cBuffer;
    final cStridesX = cBuffer + ndim;
    final cStridesTemp = cBuffer + (ndim * 2);
    for (var i = 0; i < ndim; i++) {
      cShape[i] = x.shape[i];
      cStridesX[i] = x.strides[i];
      cStridesTemp[i] = destStrides[i];
    }
    switch (x.dtype) {
      case DType.float64:
        s_to_bool_double(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
        );
      case DType.float32:
        s_to_bool_float(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
        );
      case DType.int64:
        s_to_bool_int64(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
        );
      case DType.int32:
        s_to_bool_int32(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
        );
      case DType.uint8:
        s_to_bool_uint8(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
        );
      case DType.int16:
        s_to_bool_int16(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
        );
      case DType.complex128:
        s_to_bool_complex128(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
        );
      case DType.complex64:
        s_to_bool_complex64(
          x.pointer.cast(),
          cStridesX,
          destPtr,
          cStridesTemp,
          cShape,
          ndim,
        );
      case DType.boolean:
        break;
    }
  }
  return destPtr;
}

NDArray<bool> _runBinaryLogical<Ta, Tb>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  NDArray<bool>? out,
  void Function(
    ffi.Pointer<ffi.Uint8>,
    ffi.Pointer<ffi.Uint8>,
    ffi.Pointer<ffi.Uint8>,
    int,
  )
  contiguousFn,
  void Function(
    ffi.Pointer<ffi.Uint8>,
    ffi.Pointer<ffi.Int>,
    ffi.Pointer<ffi.Uint8>,
    ffi.Pointer<ffi.Int>,
    ffi.Pointer<ffi.Uint8>,
    ffi.Pointer<ffi.Int>,
    ffi.Pointer<ffi.Int>,
    int,
  )
  stridedFn,
  String opName,
) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot perform $opName on disposed arrays.');
  }

  final marker = ScratchArena.marker;

  final ffi.Pointer<ffi.Uint8> aBoolPtr;
  final List<int> aBoolStrides;
  if (a.dtype == DType.boolean) {
    aBoolPtr = a.pointer.cast();
    aBoolStrides = a.strides;
  } else {
    aBoolPtr = ScratchArena.allocate<ffi.Uint8>(a.size);
    aBoolStrides = NDArray.computeCStrides(a.shape);
    _castToBoolean(a, aBoolPtr, aBoolStrides);
  }

  final ffi.Pointer<ffi.Uint8> bBoolPtr;
  final List<int> bBoolStrides;
  if (b.dtype == DType.boolean) {
    bBoolPtr = b.pointer.cast();
    bBoolStrides = b.strides;
  } else {
    bBoolPtr = ScratchArena.allocate<ffi.Uint8>(b.size);
    bBoolStrides = NDArray.computeCStrides(b.shape);
    _castToBoolean(b, bBoolPtr, bBoolStrides);
  }

  final broadcastResult = _broadcastBinaryStrides(
    a.shape,
    aBoolStrides,
    b.shape,
    bBoolStrides,
  );
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final NDArray<bool> result;
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for $opName.',
      );
    }
    result = out;
  } else {
    result = NDArray<bool>.create(commonShape, DType.boolean);
  }

  final isContig =
      a.isContiguous &&
      b.isContiguous &&
      result.isContiguous &&
      listEquals(a.shape, b.shape);

  if (isContig) {
    contiguousFn(aBoolPtr, bBoolPtr, result.pointer.cast(), result.size);
  } else {
    final ndim = commonShape.length;
    final cBuffer = ScratchArena.getStridedBuffer(ndim);
    final cShape = cBuffer;
    final cStridesA = cBuffer + ndim;
    final cStridesB = cBuffer + (ndim * 2);
    final cStridesRes = cBuffer + (ndim * 3);

    for (var i = 0; i < ndim; i++) {
      cShape[i] = commonShape[i];
      cStridesA[i] = stridesA[i];
      cStridesB[i] = stridesB[i];
      cStridesRes[i] = result.strides[i];
    }

    stridedFn(
      aBoolPtr,
      cStridesA,
      bBoolPtr,
      cStridesB,
      result.pointer.cast(),
      cStridesRes,
      cShape,
      ndim,
    );
  }

  ScratchArena.reset(marker);
  return result;
}
