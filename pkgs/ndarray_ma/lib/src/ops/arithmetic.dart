part of '../masked_array.dart';

MaskedArray<dynamic> _maAdd(MaskedArray self, dynamic other) =>
    _binaryOp(self, other, 'add');
MaskedArray<dynamic> _maSubtract(MaskedArray self, dynamic other) =>
    _binaryOp(self, other, 'sub');
MaskedArray<dynamic> _maMultiply(MaskedArray self, dynamic other) =>
    _binaryOp(self, other, 'mul');
MaskedArray<dynamic> _maDivide(MaskedArray self, dynamic other) =>
    _binaryOp(self, other, 'div', isDivide: true);

MaskedArray<dynamic> _binaryOp(
  MaskedArray self,
  dynamic other,
  String opName, {
  bool isDivide = false,
}) {
  return NDArray.scope(() {
    final NDArray<Object> otherData;
    final NDArray<bool>? otherMask;

    if (other is MaskedArray) {
      otherData = other.data;
      otherMask = other.mask;
    } else if (other is NDArray) {
      otherData = other as NDArray<Object>;
      otherMask = null;
    } else {
      otherData = _wrapScalar(other, self.dtype);
      otherMask = null;
    }

    final targetDType = _resolveDType(self.dtype, otherData.dtype);

    // We need to determine the result shape to broadcast masks correctly.
    final broadcastResult = ndops.broadcast(self.data, otherData);
    final resultShape = broadcastResult.shape;

    final NDArray<Object> divisorData;
    if (isDivide) {
      final combinedMask = otherMask != null
          ? ndops.logical_or(self.mask, otherMask)
          : self.mask;
      final broadcastedCombinedMask = ndops.broadcastTo(
        combinedMask,
        resultShape,
      );
      final ones = _wrapScalar(1, otherData.dtype);
      divisorData =
          ndops.where(broadcastedCombinedMask, ones, otherData)
              as NDArray<Object>;
    } else {
      divisorData = otherData;
    }

    final resultData = _dispatchBinary(
      self.data,
      divisorData,
      opName,
      targetDType,
    );

    final broadcastedMaskA = ndops.broadcastTo(self.mask, resultData.shape);
    NDArray<bool> resultMask;

    if (otherMask != null) {
      final broadcastedMaskB = ndops.broadcastTo(otherMask, resultData.shape);
      resultMask = ndops.logical_or(broadcastedMaskA, broadcastedMaskB);
    } else {
      resultMask = broadcastedMaskA.copy();
    }

    if (isDivide) {
      final zeroArray = NDArray.zeros([], otherData.dtype);
      final isZero = ndops.equal(otherData, zeroArray);
      final broadcastedIsZero = ndops.broadcastTo(isZero, resultData.shape);
      final finalMask = ndops.logical_or(resultMask, broadcastedIsZero);
      return dispatchCreateMaskedArray(
        resultData.detachToParentScope(),
        finalMask.detachToParentScope(),
        fillValue: self.fillValue,
      );
    }

    return dispatchCreateMaskedArray(
      resultData.detachToParentScope(),
      resultMask.detachToParentScope(),
      fillValue: self.fillValue,
    );
  });
}
