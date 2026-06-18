part of '../masked_array.dart';

MaskedArray<T> _maReshape<T extends Object>(
  MaskedArray<T> self,
  List<int> newShape,
) {
  return MaskedArray<T>(
    self.data.reshape(newShape),
    self.mask.reshape(newShape),
    fillValue: self.fillValue,
  );
}

MaskedArray<T> _maTranspose<T extends Object>(
  MaskedArray<T> self, [
  List<int>? axes,
]) {
  return MaskedArray<T>(
    self.data.transpose(axes),
    self.mask.transpose(axes),
    fillValue: self.fillValue,
  );
}

MaskedArray<T> _maExpandDims<T extends Object>(MaskedArray<T> self, int axis) {
  return MaskedArray<T>(
    self.data.expandDims(axis),
    self.mask.expandDims(axis),
    fillValue: self.fillValue,
  );
}

NDArray<T> _maCompressed<T extends Object>(MaskedArray<T> self) {
  return NDArray.scope(() {
    final flatData = self.data.reshape([self.size]);
    final flatMask = self.mask.reshape([self.size]);
    final invertedMask = ndops.logical_not(flatMask);
    final result = flatData.applyMask(invertedMask);
    return result.detachToParentScope();
  });
}

NDArray<T> _maFilled<T extends Object>(MaskedArray<T> self, {T? fillValue}) {
  return NDArray.scope(() {
    final val = fillValue ?? self.fillValue;
    final fillArray = _wrapScalar<T>(val, self.dtype);
    final result = ndops.where(self.mask, fillArray, self.data) as NDArray<T>;
    return result.detachToParentScope();
  });
}

MaskedArray<R> _maMapUnary<T extends Object, R extends Object>(
  MaskedArray<T> self,
  NDArray<R> Function(NDArray<T>) ufunc,
) {
  final mappedData = ufunc(self.data);
  final newFillValue = _coerceOrGetDefault(self.fillValue, mappedData.dtype);
  return dispatchCreateMaskedArray(
        mappedData,
        self.mask,
        fillValue: newFillValue,
      )
      as MaskedArray<R>;
}
