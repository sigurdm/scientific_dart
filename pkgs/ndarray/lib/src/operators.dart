import 'ndarray.dart';
import 'operations.dart' as ops;

/// Extension defining operators on [NDArray] to avoid circular imports.
extension NDArrayOperators<T, M extends Marker> on NDArray<T, M> {
  /// Element-wise addition with full broadcasting support.
  NDArray operator +(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.add(this, otherArr);
  }

  /// Element-wise subtraction with full broadcasting support.
  NDArray operator -(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.subtract(this, otherArr);
  }

  /// Element-wise multiplication with full broadcasting support.
  NDArray operator *(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.multiply(this, otherArr);
  }

  /// Element-wise division with full broadcasting support.
  NDArray operator /(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.divide(this, otherArr);
  }

  /// Element-wise floor division with full broadcasting support.
  NDArray operator ~/(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.floor_divide(this, otherArr);
  }

  /// Element-wise remainder with full broadcasting support.
  NDArray operator %(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.remainder(this, otherArr);
  }

  /// Numerical negative, element-wise.
  NDArray operator -() {
    return ops.negative(this);
  }

  /// Element-wise bitwise AND with full broadcasting support.
  NDArray operator &(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.bitwise_and(this, otherArr);
  }

  /// Element-wise bitwise OR with full broadcasting support.
  NDArray operator |(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.bitwise_or(this, otherArr);
  }

  /// Element-wise bitwise XOR with full broadcasting support.
  NDArray operator ^(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.bitwise_xor(this, otherArr);
  }

  /// Element-wise bitwise NOT.
  NDArray operator ~() {
    return ops.invert(this);
  }

  /// Element-wise left shift with full broadcasting support.
  NDArray operator <<(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.left_shift(this, otherArr);
  }

  /// Element-wise right shift with full broadcasting support.
  NDArray operator >>(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.right_shift(this, otherArr);
  }

  /// Element-wise greater than comparison with full broadcasting support.
  NDArray<bool, BooleanMarker> operator >(dynamic other) {
    NDArray otherArr;
    if (other is NDArray) {
      otherArr = other;
    } else {
      otherArr = _wrapScalar(other, shape);
    }

    if (dtype == DType.complex128 ||
        dtype == DType.complex64 ||
        otherArr.dtype == DType.complex128 ||
        otherArr.dtype == DType.complex64) {
      throw UnsupportedError(
        'Complex numbers do not support inequality comparisons',
      );
    }

    final broadcastResult = ops.broadcast(this, otherArr);
    final commonShape = broadcastResult.shape;
    final result = NDArray<bool, BooleanMarker>.create(
      commonShape,
      DType.boolean,
    );
    final resultStrides = NDArray.computeCStrides(commonShape);

    dispatchCompare(
      result.data,
      this,
      otherArr,
      commonShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      resultStrides,
      (x, y) => (x as num) > (y as num),
    );
    return result;
  }

  /// Element-wise less than comparison with full broadcasting support.
  NDArray<bool, BooleanMarker> operator <(dynamic other) {
    NDArray otherArr;
    if (other is NDArray) {
      otherArr = other;
    } else {
      otherArr = _wrapScalar(other, shape);
    }

    if (dtype == DType.complex128 ||
        dtype == DType.complex64 ||
        otherArr.dtype == DType.complex128 ||
        otherArr.dtype == DType.complex64) {
      throw UnsupportedError(
        'Complex numbers do not support inequality comparisons',
      );
    }

    final broadcastResult = ops.broadcast(this, otherArr);
    final commonShape = broadcastResult.shape;
    final result = NDArray<bool, BooleanMarker>.create(
      commonShape,
      DType.boolean,
    );
    final resultStrides = NDArray.computeCStrides(commonShape);

    dispatchCompare(
      result.data,
      this,
      otherArr,
      commonShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      resultStrides,
      (x, y) => (x as num) < (y as num),
    );
    return result;
  }

  /// Element-wise greater-or-equal comparison with full broadcasting support.
  NDArray<bool, BooleanMarker> operator >=(dynamic other) {
    NDArray otherArr;
    if (other is NDArray) {
      otherArr = other;
    } else {
      otherArr = _wrapScalar(other, shape);
    }

    if (dtype == DType.complex128 ||
        dtype == DType.complex64 ||
        otherArr.dtype == DType.complex128 ||
        otherArr.dtype == DType.complex64) {
      throw UnsupportedError(
        'Complex numbers do not support inequality comparisons',
      );
    }

    final broadcastResult = ops.broadcast(this, otherArr);
    final commonShape = broadcastResult.shape;
    final result = NDArray<bool, BooleanMarker>.create(
      commonShape,
      DType.boolean,
    );
    final resultStrides = NDArray.computeCStrides(commonShape);

    dispatchCompare(
      result.data,
      this,
      otherArr,
      commonShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      resultStrides,
      (x, y) => (x as num) >= (y as num),
    );
    return result;
  }

  /// Element-wise less-or-equal comparison with full broadcasting support.
  NDArray<bool, BooleanMarker> operator <=(dynamic other) {
    NDArray otherArr;
    if (other is NDArray) {
      otherArr = other;
    } else {
      otherArr = _wrapScalar(other, shape);
    }

    if (dtype == DType.complex128 ||
        dtype == DType.complex64 ||
        otherArr.dtype == DType.complex128 ||
        otherArr.dtype == DType.complex64) {
      throw UnsupportedError(
        'Complex numbers do not support inequality comparisons',
      );
    }

    final broadcastResult = ops.broadcast(this, otherArr);
    final commonShape = broadcastResult.shape;
    final result = NDArray<bool, BooleanMarker>.create(
      commonShape,
      DType.boolean,
    );
    final resultStrides = NDArray.computeCStrides(commonShape);

    dispatchCompare(
      result.data,
      this,
      otherArr,
      commonShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      resultStrides,
      (x, y) => (x as num) <= (y as num),
    );
    return result;
  }

  /// Element-wise equality comparison with full broadcasting support.
  NDArray<bool, BooleanMarker> eq(dynamic other) {
    NDArray otherArr;
    if (other is NDArray) {
      otherArr = other;
    } else {
      otherArr = _wrapScalar(other, shape);
    }

    final broadcastResult = ops.broadcast(this, otherArr);
    final commonShape = broadcastResult.shape;
    final result = NDArray<bool, BooleanMarker>.create(
      commonShape,
      DType.boolean,
    );
    final resultStrides = NDArray.computeCStrides(commonShape);

    dispatchCompare(
      result.data,
      this,
      otherArr,
      commonShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      resultStrides,
      (x, y) => x == y,
    );
    return result;
  }
}

NDArray _wrapScalar(dynamic value, List<int> targetShape) {
  if (value is Complex) {
    return NDArray<Complex, Complex128Marker>.fromList(
      <Complex>[value],
      List.filled(targetShape.length, 1),
      DType.complex128,
    );
  } else if (value is int) {
    return NDArray<int, Int64Marker>.fromList(
      <int>[value],
      List.filled(targetShape.length, 1),
      DType.int64,
    );
  } else if (value is double) {
    return NDArray<double, Float64Marker>.fromList(
      <double>[value],
      List.filled(targetShape.length, 1),
      DType.float64,
    );
  } else if (value is bool) {
    return NDArray<bool, BooleanMarker>.fromList(
      <bool>[value],
      List.filled(targetShape.length, 1),
      DType.boolean,
    );
  } else {
    throw ArgumentError('Unsupported scalar type: ${value.runtimeType}');
  }
}
