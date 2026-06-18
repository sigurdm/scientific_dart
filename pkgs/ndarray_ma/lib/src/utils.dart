part of 'masked_array.dart';

/// Resolves the common DType for binary operations.
DType _resolveDType(DType a, DType b) {
  if (a == DType.boolean && b == DType.boolean) return DType.uint8;
  if (a == DType.boolean) return b;
  if (b == DType.boolean) return a;
  if (a == b) return a;

  final isAIntLarge = a == DType.int64 || a == DType.int32;
  final isBIntLarge = b == DType.int64 || b == DType.int32;

  if (a == DType.complex128 || b == DType.complex128) return DType.complex128;
  if (a == DType.complex64 || b == DType.complex64) {
    if (a == DType.float64 || b == DType.float64) return DType.complex128;
    if (isAIntLarge || isBIntLarge) return DType.complex128;
    return DType.complex64;
  }
  if (a == DType.float64 || b == DType.float64) return DType.float64;
  if (a == DType.float32 || b == DType.float32) {
    if (isAIntLarge || isBIntLarge) return DType.float64;
    return DType.float32;
  }
  if (a == DType.int64 || b == DType.int64) return DType.int64;
  return DType.int32;
}

/// Default fill value mapping based on DType.
dynamic _defaultFillValue(DType dtype) {
  switch (dtype) {
    case DType.float64:
    case DType.float32:
      return 1e20;
    case DType.complex128:
    case DType.complex64:
      return Complex(1e20, 0.0);
    case DType.int64:
    case DType.int32:
    case DType.uint8:
    case DType.int16:
      return 999999;
    case DType.boolean:
      return true;
  }
}

/// Coerces a scalar value to the correct Dart type for a given DType.
S _coerceScalar<S>(dynamic value, DType<S> dtype) {
  if (dtype.isFloating) {
    return (value as num).toDouble() as S;
  } else if (dtype.isComplex) {
    if (value is Complex) return value as S;
    return Complex((value as num).toDouble(), 0.0) as S;
  } else if (dtype.isInteger) {
    return (value as num).toInt() as S;
  } else if (dtype == DType.boolean) {
    if (value is bool) return value as S;
    return ((value as num) != 0) as S;
  }
  throw ArgumentError("Unsupported dtype: $dtype");
}

/// Helper to coerce or get default fill value.
Object? _coerceOrGetDefault(Object? value, DType dtype) {
  if (value == null) return null;
  try {
    return _coerceScalar(value, dtype);
  } catch (_) {
    return _defaultFillValue(dtype);
  }
}

/// Wraps a scalar value into a 0-dimensional NDArray of the given DType.
NDArray<S> _wrapScalar<S extends Object>(dynamic value, DType<S> dtype) {
  if (value is NDArray<S>) return value;
  final coerced = _coerceScalar<S>(value, dtype);
  return NDArray<S>.fromList([coerced], [], dtype);
}

/// Helper to get max value for a DType, used in min reduction.
dynamic _maxValue(DType dtype) {
  switch (dtype) {
    case DType.float64:
    case DType.float32:
      return double.infinity;
    case DType.int64:
      return 9223372036854775807;
    case DType.int32:
      return 2147483647;
    case DType.int16:
      return 32767;
    case DType.uint8:
      return 255;
    default:
      throw ArgumentError("No max value for dtype $dtype");
  }
}

/// Helper to get min value for a DType, used in max reduction.
dynamic _minValue(DType dtype) {
  switch (dtype) {
    case DType.float64:
    case DType.float32:
      return double.negativeInfinity;
    case DType.int64:
      return -9223372036854775808;
    case DType.int32:
      return -2147483648;
    case DType.int16:
      return -32768;
    case DType.uint8:
      return 0;
    default:
      throw ArgumentError("No min value for dtype $dtype");
  }
}

/// Dispatches binary operations to the correct generic implementation.
NDArray<Object> _dispatchBinary(
  NDArray a,
  NDArray b,
  String opName,
  DType targetDType,
) {
  switch (opName) {
    case 'add':
      return _callGeneric2(a, b, ndops.add, targetDType);
    case 'sub':
      return _callGeneric2(a, b, ndops.subtract, targetDType);
    case 'mul':
      return _callGeneric2(a, b, ndops.multiply, targetDType);
    case 'div':
      final divDType = targetDType.isComplex ? DType.complex128 : DType.float64;
      return _callGeneric2(a, b, ndops.divide, divDType);
    default:
      throw ArgumentError("Unknown op: $opName");
  }
}

/// Helper to call generic ufunc with resolved type arguments.
NDArray<Object> _callGeneric2(
  NDArray a,
  NDArray b,
  Function ufunc,
  DType targetDType,
) {
  switch (targetDType) {
    case DType.float64:
      return ufunc<dynamic, dynamic, Float64>(a, b) as NDArray<Object>;
    case DType.float32:
      return ufunc<dynamic, dynamic, Float32>(a, b) as NDArray<Object>;
    case DType.complex128:
      return ufunc<dynamic, dynamic, Complex128>(a, b) as NDArray<Object>;
    case DType.complex64:
      return ufunc<dynamic, dynamic, Complex64>(a, b) as NDArray<Object>;
    case DType.int64:
      return ufunc<dynamic, dynamic, Int64>(a, b) as NDArray<Object>;
    case DType.int32:
      return ufunc<dynamic, dynamic, Int32>(a, b) as NDArray<Object>;
    case DType.int16:
      return ufunc<dynamic, dynamic, Int16>(a, b) as NDArray<Object>;
    case DType.uint8:
      return ufunc<dynamic, dynamic, Uint8>(a, b) as NDArray<Object>;
    case DType.boolean:
      return ufunc<dynamic, dynamic, bool>(a, b) as NDArray<Object>;
  }
}

/// Dispatches MaskedArray creation to preserve runtime type parameter.
MaskedArray<Object> dispatchCreateMaskedArray(
  NDArray<Object> data,
  NDArray<bool> mask, {
  Object? fillValue,
}) {
  switch (data.dtype) {
    case DType.float64:
      return MaskedArray<Float64>(
        data as NDArray<Float64>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.float64) as Float64?,
      );
    case DType.float32:
      return MaskedArray<Float32>(
        data as NDArray<Float32>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.float32) as Float32?,
      );
    case DType.complex128:
      return MaskedArray<Complex128>(
        data as NDArray<Complex128>,
        mask,
        fillValue:
            _coerceOrGetDefault(fillValue, DType.complex128) as Complex128?,
      );
    case DType.complex64:
      return MaskedArray<Complex64>(
        data as NDArray<Complex64>,
        mask,
        fillValue:
            _coerceOrGetDefault(fillValue, DType.complex64) as Complex64?,
      );
    case DType.int64:
      return MaskedArray<Int64>(
        data as NDArray<Int64>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.int64) as Int64?,
      );
    case DType.int32:
      return MaskedArray<Int32>(
        data as NDArray<Int32>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.int32) as Int32?,
      );
    case DType.int16:
      return MaskedArray<Int16>(
        data as NDArray<Int16>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.int16) as Int16?,
      );
    case DType.uint8:
      return MaskedArray<Uint8>(
        data as NDArray<Uint8>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.uint8) as Uint8?,
      );
    case DType.boolean:
      return MaskedArray<bool>(
        data as NDArray<bool>,
        mask,
        fillValue: _coerceOrGetDefault(fillValue, DType.boolean) as bool?,
      );
  }
}
