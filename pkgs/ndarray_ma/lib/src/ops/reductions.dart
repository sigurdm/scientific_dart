part of '../masked_array.dart';

MaskedArray<T> _maSum<T extends DTypeTag>(MaskedArray<T> self, {int? axis}) =>
    _reduction<T>(
      self,
      (arr, {axis}) => ndops.sumAs<T, T>(
        arr,
        (self.dtype == DType.boolean ? DType.int64 : self.dtype) as DType<T>,
        axis: axis,
      ),
      _zeroValue(self.dtype),
      axis,
    );

MaskedArray<T> _maProd<T extends DTypeTag>(MaskedArray<T> self, {int? axis}) =>
    _reduction<T>(
      self,
      (arr, {axis}) => ndops.prodAs<T, T>(
        arr,
        (self.dtype == DType.boolean ? DType.int64 : self.dtype) as DType<T>,
        axis: axis,
      ),
      _oneValue(self.dtype),
      axis,
    );

MaskedArray<T> _maMin<T extends DTypeTag>(MaskedArray<T> self, {int? axis}) {
  if (self.dtype.isComplex || self.dtype == DType.boolean) {
    throw UnsupportedError('Unsupported dtype for min: ${self.dtype}');
  }
  return _reduction<T>(
    self,
    (a, {axis}) => ndops.min<T>(a, axis: axis),
    _maxValue(self.dtype),
    axis,
  );
}

MaskedArray<T> _maMax<T extends DTypeTag>(MaskedArray<T> self, {int? axis}) {
  if (self.dtype.isComplex || self.dtype == DType.boolean) {
    throw UnsupportedError('Unsupported dtype for max: ${self.dtype}');
  }
  return _reduction<T>(
    self,
    (a, {axis}) => ndops.max<T>(a, axis: axis),
    _minValue(self.dtype),
    axis,
  );
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

MaskedArray<DTypeTag> _maMean(MaskedArray self, {int? axis}) {
  return self.sum(axis: axis).divide(self.count(axis: axis));
}

MaskedArray<DTypeTag> _maVariance(MaskedArray self, {int? axis}) {
  return NDArray.scope(() {
    final m = self.mean(axis: axis);
    final MaskedArray<DTypeTag> mExpanded;
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

MaskedArray<DTypeTag> _maStd(MaskedArray self, {int? axis}) {
  return self
      .variance(axis: axis)
      .mapUnary((data) => ndops.sqrt(data as NDArray<AnySpec>));
}

MaskedArray<T> _reduction<T extends DTypeTag>(
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
