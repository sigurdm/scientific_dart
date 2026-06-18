part of '../masked_array.dart';

MaskedArray<T> _maSum<T extends Object>(MaskedArray<T> self, {int? axis}) =>
    _reduction<T>(
      self,
      (arr, {axis}) => ndops.sum<T>(arr, axis: axis),
      _zeroValue(self.dtype),
      axis,
    );

MaskedArray<T> _maProd<T extends Object>(MaskedArray<T> self, {int? axis}) =>
    _reduction<T>(
      self,
      (arr, {axis}) => ndops.prod<T>(arr, axis: axis),
      _oneValue(self.dtype),
      axis,
    );

MaskedArray<T> _maMin<T extends Object>(MaskedArray<T> self, {int? axis}) {
  final NDArray<T> Function(NDArray<T>, {int? axis}) redOp;
  switch (self.dtype) {
    case DType.float64:
    case DType.float32:
      redOp = (a, {axis}) =>
          ndops.min(a as NDArray<num>, axis: axis) as NDArray<T>;
    case DType.int64:
    case DType.int32:
    case DType.int16:
    case DType.uint8:
      redOp = (a, {axis}) =>
          ndops.min(a as NDArray<num>, axis: axis) as NDArray<T>;
    default:
      throw UnsupportedError('Unsupported dtype for min: ${self.dtype}');
  }
  return _reduction<T>(self, redOp, _maxValue(self.dtype), axis);
}

MaskedArray<T> _maMax<T extends Object>(MaskedArray<T> self, {int? axis}) {
  final NDArray<T> Function(NDArray<T>, {int? axis}) redOp;
  switch (self.dtype) {
    case DType.float64:
    case DType.float32:
      redOp = (a, {axis}) =>
          ndops.max(a as NDArray<num>, axis: axis) as NDArray<T>;
    case DType.int64:
    case DType.int32:
    case DType.int16:
    case DType.uint8:
      redOp = (a, {axis}) =>
          ndops.max(a as NDArray<num>, axis: axis) as NDArray<T>;
    default:
      throw UnsupportedError('Unsupported dtype for max: ${self.dtype}');
  }
  return _reduction<T>(self, redOp, _minValue(self.dtype), axis);
}

NDArray<Int32> _maCount(MaskedArray self, {int? axis}) {
  return NDArray.scope(() {
    final zeros = NDArray<Int32>.zeros(self.shape, DType.int32);
    final ones = NDArray<Int32>.ones(self.shape, DType.int32);
    final validMap = ndops.where(self.mask, zeros, ones) as NDArray<Int32>;
    final result = ndops.sum(validMap, axis: axis);
    return result.detachToParentScope();
  });
}

MaskedArray<dynamic> _maMean(MaskedArray self, {int? axis}) {
  return self.sum(axis: axis).divide(self.count(axis: axis));
}

MaskedArray<dynamic> _maVariance(MaskedArray self, {int? axis}) {
  return NDArray.scope(() {
    final m = self.mean(axis: axis);
    final MaskedArray<dynamic> mExpanded;
    if (axis != null) {
      mExpanded = m.expandDims(axis);
    } else {
      mExpanded = m;
    }
    final diff = self.subtract(mExpanded);
    final diffSq = diff.multiply(diff);
    final result = diffSq.mean(axis: axis);
    return result.detachToParentScope();
  });
}

MaskedArray<dynamic> _maStd(MaskedArray self, {int? axis}) {
  return self.variance(axis: axis).mapUnary((data) => ndops.sqrt(data));
}

MaskedArray<T> _reduction<T extends Object>(
  MaskedArray<T> self,
  NDArray<T> Function(NDArray<T>, {int? axis}) ndOp,
  dynamic fillValueForReduction,
  int? axis,
) {
  return NDArray.scope(() {
    final fillArray = _wrapScalar<T>(fillValueForReduction, self.dtype);
    final filledData =
        ndops.where(self.mask, fillArray, self.data) as NDArray<T>;
    final resultData = ndOp(filledData, axis: axis);
    final resultMask = ndops.all(self.mask, axis: axis);
    return dispatchCreateMaskedArray(
          resultData.detachToParentScope(),
          resultMask.detachToParentScope(),
          fillValue: self.fillValue,
        )
        as MaskedArray<T>;
  });
}

// Helpers for reduction default values (from Worker 1, but integrated)
dynamic _zeroValue(DType dtype) {
  if (dtype == DType.float64 || dtype == DType.float32) return 0.0;
  if (dtype == DType.complex128 || dtype == DType.complex64)
    return Complex(0, 0);
  if (dtype == DType.int64 ||
      dtype == DType.int32 ||
      dtype == DType.int16 ||
      dtype == DType.uint8)
    return 0;
  if (dtype == DType.boolean) return false;
  throw UnimplementedError('Unsupported dtype: $dtype');
}

dynamic _oneValue(DType dtype) {
  if (dtype == DType.float64 || dtype == DType.float32) return 1.0;
  if (dtype == DType.complex128 || dtype == DType.complex64)
    return Complex(1, 0);
  if (dtype == DType.int64 ||
      dtype == DType.int32 ||
      dtype == DType.int16 ||
      dtype == DType.uint8)
    return 1;
  if (dtype == DType.boolean) return true;
  throw UnimplementedError('Unsupported dtype: $dtype');
}
