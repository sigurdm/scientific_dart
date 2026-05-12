import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'ndarray.dart';
import 'operations.dart' as ops;
import 'ndarray_bindings.dart';
import 'broadcasting.dart';

/// Signature for C function strided binary operations.
typedef StridedBinaryOp =
    void Function(
      ffi.Pointer<ffi.Void> a,
      ffi.Pointer<ffi.Int> stridesA,
      ffi.Pointer<ffi.Void> b,
      ffi.Pointer<ffi.Int> stridesB,
      ffi.Pointer<ffi.Void> result,
      ffi.Pointer<ffi.Int> stridesResult,
      ffi.Pointer<ffi.Int> shape,
      int rank,
    );

/// Helper function for safe addition handling DType.complex64s.
dynamic _safeAdd(dynamic x, dynamic y) {
  if (y is Complex && x is! Complex) {
    return y + x;
  }
  return x + y;
}

/// Element-wise addition of two arrays.
///
/// Returns a new array with the promoted data type.
NDArray<R> add<Ta, Tb, R>(NDArray<Ta> a, NDArray<Tb> b, {NDArray<R>? out}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot add disposed arrays.');
  }
  final targetDType = ops.resolveDType(a.dtype, b.dtype);
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final NDArray<R> result;
  if (out != null) {
    if (!ops.listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    result = out;
  } else {
    result = NDArray<R>.create(commonShape, targetDType as DType<R>);
  }

  final resultStrides = NDArray.computeCStrides(commonShape);

  // Handle complex64 fallback as in original code
  if ((b.dtype as dynamic) == DType.complex64) {
    ops.ArithmeticNDArrayOperationsHelper(a).dynamicElementWiseOp(
      result,
      b,
      commonShape,
      stridesA,
      stridesB,
      resultStrides,
      _safeAdd,
    );
    return result;
  }

  // Specialized paths for Float64 (as in original extensions.dart)
  if (commonShape.length <= 8) {
    final isContig =
        a.isContiguous &&
        b.isContiguous &&
        result.isContiguous &&
        ops.listEquals(a.shape, b.shape);

    switch ((a.dtype, b.dtype)) {
      case (DType.float64, DType.float64) when isContig:
        v_add_double_double_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float64, DType.float64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_add_double_double_double);
      case (DType.float64, DType.float32) when isContig:
        v_add_double_float_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float64, DType.float32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_add_double_float_double);
      case (DType.float64, DType.int64) when isContig:
        v_add_double_int64_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float64, DType.int64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_add_double_int64_double);
      case (DType.float64, DType.int32) when isContig:
        v_add_double_int32_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float64, DType.int32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_add_double_int32_double);
      case (DType.float32, DType.float64) when isContig:
        v_add_float_double_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float32, DType.float64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_add_float_double_double);
      case (DType.float32, DType.float32) when isContig:
        v_add_float_float_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float32, DType.float32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_add_float_float_float);
      case (DType.float32, DType.int64) when isContig:
        v_add_float_int64_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float32, DType.int64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_add_float_int64_float);
      case (DType.float32, DType.int32) when isContig:
        v_add_float_int32_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float32, DType.int32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_add_float_int32_float);
      case (DType.int64, DType.float64) when isContig:
        v_add_int64_double_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int64, DType.float64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_add_int64_double_double);
      case (DType.int64, DType.float32) when isContig:
        v_add_int64_float_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int64, DType.float32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_add_int64_float_float);
      case (DType.int64, DType.int64) when isContig:
        v_add_int64_int64_int64(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int64, DType.int64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_add_int64_int64_int64);
      case (DType.int64, DType.int32) when isContig:
        v_add_int64_int32_int64(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int64, DType.int32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_add_int64_int32_int64);
      case (DType.int32, DType.float64) when isContig:
        v_add_int32_double_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int32, DType.float64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_add_int32_double_double);
      case (DType.int32, DType.float32) when isContig:
        v_add_int32_float_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int32, DType.float32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_add_int32_float_float);
      case (DType.int32, DType.int64) when isContig:
        v_add_int32_int64_int64(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int32, DType.int64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_add_int32_int64_int64);
      case (DType.int32, DType.int32) when isContig:
        v_add_int32_int32_int32(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int32, DType.int32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_add_int32_int32_int32);
      default:
        break;
    }
  }

  // Fallback to dynamic element-wise operation
  ops.ArithmeticNDArrayOperationsHelper(a).dynamicElementWiseOp(
    result,
    b,
    commonShape,
    stridesA,
    stridesB,
    resultStrides,
    _safeAdd,
  );
  return result;
}



/// Helper for strided operations to avoid code duplication.
NDArray<R> _stridedAdd<Ta, Tb, R>(
  NDArray<Ta> a,
  NDArray<Tb> b,
  NDArray<R> result,
  List<int> commonShape,
  List<int> stridesA,
  List<int> stridesB,
  List<int> resultStrides,
  Function sOp,
) {
  final cShape = malloc<ffi.Int>(commonShape.length);
  final cStridesA = malloc<ffi.Int>(stridesA.length);
  final cStridesB = malloc<ffi.Int>(stridesB.length);
  final cStridesRes = malloc<ffi.Int>(resultStrides.length);

  for (var i = 0; i < commonShape.length; i++) {
    cShape[i] = commonShape[i];
    cStridesA[i] = stridesA[i];
    cStridesB[i] = stridesB[i];
    cStridesRes[i] = resultStrides[i];
  }

  try {
    sOp(
      a.pointer.cast(),
      cStridesA,
      b.pointer.cast(),
      cStridesB,
      result.pointer.cast(),
      cStridesRes,
      cShape,
      commonShape.length,
    );
  } finally {
    malloc.free(cShape);
    malloc.free(cStridesA);
    malloc.free(cStridesB);
    malloc.free(cStridesRes);
  }
  return result;
}

dynamic _safeSub(dynamic x, dynamic y) {
  if (y is Complex && x is! Complex) {
    return Complex(x.toDouble() - y.real, -y.imag);
  }
  return x - y;
}

dynamic _safeMul(dynamic x, dynamic y) {
  if (y is Complex && x is! Complex) {
    return y * x;
  }
  return x * y;
}

dynamic _safeDiv(dynamic x, dynamic y) {
  if (y is Complex && x is! Complex) {
    return Complex(x.toDouble(), 0.0) / y;
  }
  return x / y;
}

/// Element-wise subtraction of two arrays.
NDArray<R> subtract<Ta, Tb, R>(NDArray<Ta> a, NDArray<Tb> b, {NDArray<R>? out}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot subtract disposed arrays.');
  }
  final targetDType = ops.resolveDType(a.dtype, b.dtype);
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final NDArray<R> result;
  if (out != null) {
    if (!ops.listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError('Provided out buffer has incompatible shape or dtype.');
    }
    result = out;
  } else {
    result = NDArray<R>.create(commonShape, targetDType as DType<R>);
  }

  final resultStrides = NDArray.computeCStrides(commonShape);

  if (b.dtype == DType.complex64) {
    ops.ArithmeticNDArrayOperationsHelper(a).dynamicElementWiseOp(
      result, b, commonShape, stridesA, stridesB, resultStrides, _safeSub, isSubtract: true);
    return result;
  }

  if (commonShape.length <= 8) {
    final isContig = a.isContiguous && b.isContiguous && result.isContiguous && ops.listEquals(a.shape, b.shape);

    switch ((a.dtype, b.dtype)) {
      case (DType.float64, DType.float64) when isContig:
        v_sub_double_double_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float64, DType.float64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_sub_double_double_double);
      case (DType.float64, DType.float32) when isContig:
        v_sub_double_float_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float64, DType.float32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_sub_double_float_double);
      case (DType.float64, DType.int64) when isContig:
        v_sub_double_int64_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float64, DType.int64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_sub_double_int64_double);
      case (DType.float64, DType.int32) when isContig:
        v_sub_double_int32_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float64, DType.int32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_sub_double_int32_double);
      case (DType.float32, DType.float64) when isContig:
        v_sub_float_double_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float32, DType.float64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_sub_float_double_double);
      case (DType.float32, DType.float32) when isContig:
        v_sub_float_float_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float32, DType.float32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_sub_float_float_float);
      case (DType.float32, DType.int64) when isContig:
        v_sub_float_int64_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float32, DType.int64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_sub_float_int64_float);
      case (DType.float32, DType.int32) when isContig:
        v_sub_float_int32_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float32, DType.int32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_sub_float_int32_float);
      case (DType.int64, DType.float64) when isContig:
        v_sub_int64_double_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int64, DType.float64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_sub_int64_double_double);
      case (DType.int64, DType.float32) when isContig:
        v_sub_int64_float_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int64, DType.float32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_sub_int64_float_float);
      case (DType.int64, DType.int64) when isContig:
        v_sub_int64_int64_int64(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int64, DType.int64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_sub_int64_int64_int64);
      case (DType.int64, DType.int32) when isContig:
        v_sub_int64_int32_int64(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int64, DType.int32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_sub_int64_int32_int64);
      case (DType.int32, DType.float64) when isContig:
        v_sub_int32_double_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int32, DType.float64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_sub_int32_double_double);
      case (DType.int32, DType.float32) when isContig:
        v_sub_int32_float_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int32, DType.float32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_sub_int32_float_float);
      case (DType.int32, DType.int64) when isContig:
        v_sub_int32_int64_int64(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int32, DType.int64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_sub_int32_int64_int64);
      case (DType.int32, DType.int32) when isContig:
        v_sub_int32_int32_int32(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int32, DType.int32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_sub_int32_int32_int32);
      default:
        break;
    }
  }

  ops.ArithmeticNDArrayOperationsHelper(a).dynamicElementWiseOp(
    result, b, commonShape, stridesA, stridesB, resultStrides, _safeSub, isSubtract: true);
  return result;
}

/// Element-wise multiplication of two arrays.
NDArray<R> multiply<Ta, Tb, R>(NDArray<Ta> a, NDArray<Tb> b, {NDArray<R>? out}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot multiply disposed arrays.');
  }
  final targetDType = ops.resolveDType(a.dtype, b.dtype);
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final NDArray<R> result;
  if (out != null) {
    if (!ops.listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError('Provided out buffer has incompatible shape or dtype.');
    }
    result = out;
  } else {
    result = NDArray<R>.create(commonShape, targetDType as DType<R>);
  }

  final resultStrides = NDArray.computeCStrides(commonShape);

  if (b.dtype == DType.complex64) {
    ops.ArithmeticNDArrayOperationsHelper(a).dynamicElementWiseOp(
      result, b, commonShape, stridesA, stridesB, resultStrides, _safeMul);
    return result;
  }

  if (commonShape.length <= 8) {
    final isContig = a.isContiguous && b.isContiguous && result.isContiguous && ops.listEquals(a.shape, b.shape);

    switch ((a.dtype, b.dtype)) {
      case (DType.float64, DType.float64) when isContig:
        v_mul_double_double_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float64, DType.float64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_mul_double_double_double);
      case (DType.float64, DType.float32) when isContig:
        v_mul_double_float_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float64, DType.float32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_mul_double_float_double);
      case (DType.float64, DType.int64) when isContig:
        v_mul_double_int64_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float64, DType.int64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_mul_double_int64_double);
      case (DType.float64, DType.int32) when isContig:
        v_mul_double_int32_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float64, DType.int32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_mul_double_int32_double);
      case (DType.float32, DType.float64) when isContig:
        v_mul_float_double_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float32, DType.float64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_mul_float_double_double);
      case (DType.float32, DType.float32) when isContig:
        v_mul_float_float_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float32, DType.float32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_mul_float_float_float);
      case (DType.float32, DType.int64) when isContig:
        v_mul_float_int64_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float32, DType.int64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_mul_float_int64_float);
      case (DType.float32, DType.int32) when isContig:
        v_mul_float_int32_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float32, DType.int32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_mul_float_int32_float);
      case (DType.int64, DType.float64) when isContig:
        v_mul_int64_double_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int64, DType.float64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_mul_int64_double_double);
      case (DType.int64, DType.float32) when isContig:
        v_mul_int64_float_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int64, DType.float32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_mul_int64_float_float);
      case (DType.int64, DType.int64) when isContig:
        v_mul_int64_int64_int64(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int64, DType.int64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_mul_int64_int64_int64);
      case (DType.int64, DType.int32) when isContig:
        v_mul_int64_int32_int64(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int64, DType.int32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_mul_int64_int32_int64);
      case (DType.int32, DType.float64) when isContig:
        v_mul_int32_double_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int32, DType.float64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_mul_int32_double_double);
      case (DType.int32, DType.float32) when isContig:
        v_mul_int32_float_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int32, DType.float32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_mul_int32_float_float);
      case (DType.int32, DType.int64) when isContig:
        v_mul_int32_int64_int64(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int32, DType.int64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_mul_int32_int64_int64);
      case (DType.int32, DType.int32) when isContig:
        v_mul_int32_int32_int32(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int32, DType.int32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_mul_int32_int32_int32);
      // Add more cases as needed
      default:
        break;
    }
  }

  ops.ArithmeticNDArrayOperationsHelper(a).dynamicElementWiseOp(
    result, b, commonShape, stridesA, stridesB, resultStrides, _safeMul);
  return result;
}

/// Element-wise division of two arrays.
NDArray<R> divide<Ta, Tb, R>(NDArray<Ta> a, NDArray<Tb> b, {NDArray<R>? out}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot divide disposed arrays.');
  }
  var targetDType = ops.resolveDType(a.dtype, b.dtype);
  if (targetDType.isInteger) {
    targetDType = DType.float64;
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final NDArray<R> result;
  if (out != null) {
    if (!ops.listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError('Provided out buffer has incompatible shape or dtype.');
    }
    result = out;
  } else {
    result = NDArray<R>.create(commonShape, targetDType as DType<R>);
  }

  final resultStrides = NDArray.computeCStrides(commonShape);

  if (b.dtype == DType.complex64) {
    ops.ArithmeticNDArrayOperationsHelper(a).dynamicElementWiseOp(
      result, b, commonShape, stridesA, stridesB, resultStrides, _safeDiv, isDivide: true);
    return result;
  }

  if (commonShape.length <= 8) {
    final isContig = a.isContiguous && b.isContiguous && result.isContiguous && ops.listEquals(a.shape, b.shape);

    switch ((a.dtype, b.dtype)) {
      case (DType.float64, DType.float64) when isContig:
        v_div_double_double_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float64, DType.float64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_div_double_double_double);
      case (DType.float64, DType.float32) when isContig:
        v_div_double_float_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float64, DType.float32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_div_double_float_double);
      case (DType.float64, DType.int64) when isContig:
        v_div_double_int64_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float64, DType.int64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_div_double_int64_double);
      case (DType.float64, DType.int32) when isContig:
        v_div_double_int32_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float64, DType.int32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_div_double_int32_double);
      case (DType.float32, DType.float64) when isContig:
        v_div_float_double_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float32, DType.float64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_div_float_double_double);
      case (DType.float32, DType.float32) when isContig:
        v_div_float_float_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float32, DType.float32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_div_float_float_float);
      case (DType.float32, DType.int64) when isContig:
        v_div_float_int64_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float32, DType.int64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_div_float_int64_float);
      case (DType.float32, DType.int32) when isContig:
        v_div_float_int32_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.float32, DType.int32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_div_float_int32_float);
      case (DType.int64, DType.float64) when isContig:
        v_div_int64_double_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int64, DType.float64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_div_int64_double_double);
      case (DType.int64, DType.float32) when isContig:
        v_div_int64_float_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int64, DType.float32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_div_int64_float_float);
      case (DType.int64, DType.int64) when isContig:
        v_div_int64_int64_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int64, DType.int64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_div_int64_int64_double);
      case (DType.int64, DType.int32) when isContig:
        v_div_int64_int32_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int64, DType.int32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_div_int64_int32_double);
      case (DType.int32, DType.float64) when isContig:
        v_div_int32_double_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int32, DType.float64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_div_int32_double_double);
      case (DType.int32, DType.float32) when isContig:
        v_div_int32_float_float(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int32, DType.float32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_div_int32_float_float);
      case (DType.int32, DType.int64) when isContig:
        v_div_int32_int64_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int32, DType.int64):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_div_int32_int64_double);
      case (DType.int32, DType.int32) when isContig:
        v_div_int32_int32_double(a.pointer.cast(), b.pointer.cast(), result.pointer.cast(), a.size);
        return result;
      case (DType.int32, DType.int32):
        return _stridedAdd(a, b, result, commonShape, stridesA, stridesB, resultStrides, s_div_int32_int32_double);
      // Add more cases as needed
      default:
        break;
    }
  }

  ops.ArithmeticNDArrayOperationsHelper(a).dynamicElementWiseOp(
    result, b, commonShape, stridesA, stridesB, resultStrides, _safeDiv, isDivide: true);
  return result;
}
