import 'broadcasting.dart';

import 'ndarray.dart';
import 'numdart_bindings.dart';
import 'operations.dart' as ops;

// =============================================================================
// Float64NDArrayOperations (NDArray<Float64>)
// =============================================================================

extension Float64NDArrayOperations on NDArray<Float64> {
  /// Element-wise addition returning strongly-typed Float64 NDArray.
  NDArray<Float64> add(NDArray<Float64> other, {NDArray<Float64>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.float64) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    if (isContiguous && other.isContiguous && _listEquals(shape, other.shape)) {
      v_add_double(
        pointer.cast(),
        other.pointer.cast(),
        result.pointer.cast(),
        data.length,
      );
      return result;
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x + y,
    );
    return result;
  }

  /// Element-wise subtraction returning strongly-typed Float64 NDArray.
  NDArray<Float64> subtract(NDArray<Float64> other, {NDArray<Float64>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.float64) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    if (isContiguous && other.isContiguous && _listEquals(shape, other.shape)) {
      v_sub_double(
        pointer.cast(),
        other.pointer.cast(),
        result.pointer.cast(),
        data.length,
      );
      return result;
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x - y,
    );
    return result;
  }

  /// Element-wise multiplication returning strongly-typed Float64 NDArray.
  NDArray<Float64> multiply(NDArray<Float64> other, {NDArray<Float64>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.float64) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    if (isContiguous && other.isContiguous && _listEquals(shape, other.shape)) {
      v_mul_double(
        pointer.cast(),
        other.pointer.cast(),
        result.pointer.cast(),
        data.length,
      );
      return result;
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x * y,
    );
    return result;
  }

  /// Element-wise division returning strongly-typed Float64 NDArray.
  NDArray<Float64> divide(NDArray<Float64> other, {NDArray<Float64>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.float64) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    if (isContiguous && other.isContiguous && _listEquals(shape, other.shape)) {
      v_div_double(
        pointer.cast(),
        other.pointer.cast(),
        result.pointer.cast(),
        data.length,
      );
      return result;
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x / y,
    );
    return result;
  }

  /// Matrix multiplication returning Float64 NDArray.
  NDArray<Float64> matmul(NDArray<Float64> other, {NDArray<Float64>? into}) {
    if (into != null) {
      final res =
          ops.matmul(this as dynamic, other as dynamic) as NDArray<Float64>;
      into.data.setRange(0, res.data.length, res.data);
      res.dispose();
      return into;
    }
    return ops.matmul(this as dynamic, other as dynamic) as NDArray<Float64>;
  }

  // --- Mixed Float32 arguments ---

  NDArray<Float64> addFloat32(
    NDArray<Float32> other, {
    NDArray<Float64>? into,
  }) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x + y,
    );
    return result;
  }

  NDArray<Float64> subtractFloat32(
    NDArray<Float32> other, {
    NDArray<Float64>? into,
  }) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x - y,
    );
    return result;
  }

  NDArray<Float64> multiplyFloat32(
    NDArray<Float32> other, {
    NDArray<Float64>? into,
  }) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x * y,
    );
    return result;
  }

  NDArray<Float64> divideFloat32(
    NDArray<Float32> other, {
    NDArray<Float64>? into,
  }) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x / y,
    );
    return result;
  }

  // --- Mixed Int64 arguments ---

  NDArray<Float64> addInt64(NDArray<Int64> other, {NDArray<Float64>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleIntOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x + y,
    );
    return result;
  }

  NDArray<Float64> subtractInt64(
    NDArray<Int64> other, {
    NDArray<Float64>? into,
  }) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleIntOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x - y,
    );
    return result;
  }

  NDArray<Float64> multiplyInt64(
    NDArray<Int64> other, {
    NDArray<Float64>? into,
  }) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleIntOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x * y,
    );
    return result;
  }

  NDArray<Float64> divideInt64(NDArray<Int64> other, {NDArray<Float64>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleIntOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x / y,
    );
    return result;
  }

  // --- Mixed Scalar double/int arguments ---

  NDArray<Float64> addScalar(double scalar, {NDArray<Float64>? into}) {
    final scalarArr = _wrapScalar(scalar, shape) as NDArray<Float64>;
    return add(scalarArr, into: into);
  }

  NDArray<Float64> subtractScalar(double scalar, {NDArray<Float64>? into}) {
    final scalarArr = _wrapScalar(scalar, shape) as NDArray<Float64>;
    return subtract(scalarArr, into: into);
  }

  NDArray<Float64> multiplyScalar(double scalar, {NDArray<Float64>? into}) {
    final scalarArr = _wrapScalar(scalar, shape) as NDArray<Float64>;
    return multiply(scalarArr, into: into);
  }

  NDArray<Float64> divideScalar(double scalar, {NDArray<Float64>? into}) {
    final scalarArr = _wrapScalar(scalar, shape) as NDArray<Float64>;
    return divide(scalarArr, into: into);
  }
}

// =============================================================================
// Float32NDArrayOperations (NDArray<Float32>)
// =============================================================================

extension Float32NDArrayOperations on NDArray<Float32> {
  /// Element-wise addition returning strongly-typed Float32 NDArray.
  NDArray<Float32> add(NDArray<Float32> other, {NDArray<Float32>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float32>.create(expectedShape, DType.float32);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.float32) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    if (isContiguous && other.isContiguous && _listEquals(shape, other.shape)) {
      v_add_float(
        pointer.cast(),
        other.pointer.cast(),
        result.pointer.cast(),
        data.length,
      );
      return result;
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x + y,
    );
    return result;
  }

  /// Element-wise subtraction returning strongly-typed Float32 NDArray.
  NDArray<Float32> subtract(NDArray<Float32> other, {NDArray<Float32>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float32>.create(expectedShape, DType.float32);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.float32) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    if (isContiguous && other.isContiguous && _listEquals(shape, other.shape)) {
      v_sub_float(
        pointer.cast(),
        other.pointer.cast(),
        result.pointer.cast(),
        data.length,
      );
      return result;
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x - y,
    );
    return result;
  }

  /// Element-wise multiplication returning strongly-typed Float32 NDArray.
  NDArray<Float32> multiply(NDArray<Float32> other, {NDArray<Float32>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float32>.create(expectedShape, DType.float32);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.float32) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    if (isContiguous && other.isContiguous && _listEquals(shape, other.shape)) {
      v_mul_float(
        pointer.cast(),
        other.pointer.cast(),
        result.pointer.cast(),
        data.length,
      );
      return result;
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x * y,
    );
    return result;
  }

  /// Element-wise division returning strongly-typed Float32 NDArray.
  NDArray<Float32> divide(NDArray<Float32> other, {NDArray<Float32>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float32>.create(expectedShape, DType.float32);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.float32) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    if (isContiguous && other.isContiguous && _listEquals(shape, other.shape)) {
      v_div_float(
        pointer.cast(),
        other.pointer.cast(),
        result.pointer.cast(),
        data.length,
      );
      return result;
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x / y,
    );
    return result;
  }

  // --- Mixed Float64 arguments (Upcast promotion) ---

  NDArray<Float64> addFloat64(
    NDArray<Float64> other, {
    NDArray<Float64>? into,
  }) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x + y,
    );
    return result;
  }

  NDArray<Float64> subtractFloat64(
    NDArray<Float64> other, {
    NDArray<Float64>? into,
  }) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x - y,
    );
    return result;
  }

  NDArray<Float64> multiplyFloat64(
    NDArray<Float64> other, {
    NDArray<Float64>? into,
  }) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x * y,
    );
    return result;
  }

  NDArray<Float64> divideFloat64(
    NDArray<Float64> other, {
    NDArray<Float64>? into,
  }) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleOp(
      result.data as List<double>,
      data as List<double>,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x / y,
    );
    return result;
  }

  // --- Mixed Scalar double/int arguments ---

  NDArray<Float32> addScalar(double scalar, {NDArray<Float32>? into}) {
    final scalarArr = _wrapScalar(scalar, shape) as NDArray<Float32>;
    return add(scalarArr, into: into);
  }

  NDArray<Float32> subtractScalar(double scalar, {NDArray<Float32>? into}) {
    final scalarArr = _wrapScalar(scalar, shape) as NDArray<Float32>;
    return subtract(scalarArr, into: into);
  }

  NDArray<Float32> multiplyScalar(double scalar, {NDArray<Float32>? into}) {
    final scalarArr = _wrapScalar(scalar, shape) as NDArray<Float32>;
    return multiply(scalarArr, into: into);
  }

  NDArray<Float32> divideScalar(double scalar, {NDArray<Float32>? into}) {
    final scalarArr = _wrapScalar(scalar, shape) as NDArray<Float32>;
    return divide(scalarArr, into: into);
  }
}

// =============================================================================
// Int64NDArrayOperations (NDArray<Int64>)
// =============================================================================

extension Int64NDArrayOperations on NDArray<Int64> {
  /// Element-wise addition returning strongly-typed Int64 NDArray.
  NDArray<Int64> add(NDArray<Int64> other, {NDArray<Int64>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result = into ?? NDArray<Int64>.create(expectedShape, DType.int64);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.int64) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseIntOp(
      result.data as List<int>,
      data as List<int>,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x + y,
    );
    return result;
  }

  /// Element-wise subtraction returning strongly-typed Int64 NDArray.
  NDArray<Int64> subtract(NDArray<Int64> other, {NDArray<Int64>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result = into ?? NDArray<Int64>.create(expectedShape, DType.int64);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.int64) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseIntOp(
      result.data as List<int>,
      data as List<int>,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x - y,
    );
    return result;
  }

  /// Element-wise multiplication returning strongly-typed Int64 NDArray.
  NDArray<Int64> multiply(NDArray<Int64> other, {NDArray<Int64>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result = into ?? NDArray<Int64>.create(expectedShape, DType.int64);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.int64) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseIntOp(
      result.data as List<int>,
      data as List<int>,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x * y,
    );
    return result;
  }

  /// Element-wise division returning Float64 NDArray (due to division promotion).
  NDArray<Float64> divide(NDArray<Int64> other, {NDArray<Float64>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.float64) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseIntIntToDoubleOp(
      result.data as List<double>,
      data as List<int>,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x.toDouble() / y.toDouble(),
    );
    return result;
  }

  // --- Mixed Double/Float64 arguments (Upcast promotion) ---

  NDArray<Float64> addDouble(NDArray<Float64> other, {NDArray<Float64>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleIntOp(
      result.data as List<double>,
      other.data as List<double>,
      data as List<int>,
      expectedShape,
      broadcastResult.stridesB,
      broadcastResult.stridesA,
      NDArray.computeCStrides(expectedShape),
      0,
      other.offsetElements,
      offsetElements,
      result.offsetElements,
      (x, y) => y + x,
    );
    return result;
  }

  NDArray<Float64> subtractDouble(
    NDArray<Float64> other, {
    NDArray<Float64>? into,
  }) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleIntOp(
      result.data as List<double>,
      other.data as List<double>,
      data as List<int>,
      expectedShape,
      broadcastResult.stridesB,
      broadcastResult.stridesA,
      NDArray.computeCStrides(expectedShape),
      0,
      other.offsetElements,
      offsetElements,
      result.offsetElements,
      (x, y) => y - x,
    );
    return result;
  }

  NDArray<Float64> multiplyDouble(
    NDArray<Float64> other, {
    NDArray<Float64>? into,
  }) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    final broadcastResult = broadcast(this, other);
    _elementWiseDoubleIntOp(
      result.data as List<double>,
      other.data as List<double>,
      data as List<int>,
      expectedShape,
      broadcastResult.stridesB,
      broadcastResult.stridesA,
      NDArray.computeCStrides(expectedShape),
      0,
      other.offsetElements,
      offsetElements,
      result.offsetElements,
      (x, y) => y * x,
    );
    return result;
  }

  // --- Mixed Int32 arguments ---

  NDArray<Int64> addInt32(NDArray<Int32> other, {NDArray<Int64>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result = into ?? NDArray<Int64>.create(expectedShape, DType.int64);
    final broadcastResult = broadcast(this, other);
    _elementWiseIntOp(
      result.data as List<int>,
      data as List<int>,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x + y,
    );
    return result;
  }

  NDArray<Int64> subtractInt32(NDArray<Int32> other, {NDArray<Int64>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result = into ?? NDArray<Int64>.create(expectedShape, DType.int64);
    final broadcastResult = broadcast(this, other);
    _elementWiseIntOp(
      result.data as List<int>,
      data as List<int>,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x - y,
    );
    return result;
  }

  NDArray<Int64> multiplyInt32(NDArray<Int32> other, {NDArray<Int64>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result = into ?? NDArray<Int64>.create(expectedShape, DType.int64);
    final broadcastResult = broadcast(this, other);
    _elementWiseIntOp(
      result.data as List<int>,
      data as List<int>,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x * y,
    );
    return result;
  }

  // --- Mixed Scalar int arguments ---

  NDArray<Int64> addScalar(int scalar, {NDArray<Int64>? into}) {
    final scalarArr = _wrapScalar(scalar, shape) as NDArray<Int64>;
    return add(scalarArr, into: into);
  }

  NDArray<Int64> subtractScalar(int scalar, {NDArray<Int64>? into}) {
    final scalarArr = _wrapScalar(scalar, shape) as NDArray<Int64>;
    return subtract(scalarArr, into: into);
  }

  NDArray<Int64> multiplyScalar(int scalar, {NDArray<Int64>? into}) {
    final scalarArr = _wrapScalar(scalar, shape) as NDArray<Int64>;
    return multiply(scalarArr, into: into);
  }
}

// =============================================================================
// Int32NDArrayOperations (NDArray<Int32>)
// =============================================================================

extension Int32NDArrayOperations on NDArray<Int32> {
  /// Element-wise addition returning strongly-typed Int32 NDArray.
  NDArray<Int32> add(NDArray<Int32> other, {NDArray<Int32>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result = into ?? NDArray<Int32>.create(expectedShape, DType.int32);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.int32) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseIntOp(
      result.data as List<int>,
      data as List<int>,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x + y,
    );
    return result;
  }

  /// Element-wise subtraction returning strongly-typed Int32 NDArray.
  NDArray<Int32> subtract(NDArray<Int32> other, {NDArray<Int32>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result = into ?? NDArray<Int32>.create(expectedShape, DType.int32);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.int32) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseIntOp(
      result.data as List<int>,
      data as List<int>,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x - y,
    );
    return result;
  }

  /// Element-wise multiplication returning strongly-typed Int32 NDArray.
  NDArray<Int32> multiply(NDArray<Int32> other, {NDArray<Int32>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result = into ?? NDArray<Int32>.create(expectedShape, DType.int32);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.int32) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseIntOp(
      result.data as List<int>,
      data as List<int>,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x * y,
    );
    return result;
  }

  /// Element-wise division returning Float64 NDArray (due to division promotion).
  NDArray<Float64> divide(NDArray<Int32> other, {NDArray<Float64>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Float64>.create(expectedShape, DType.float64);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.float64) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseIntIntToDoubleOp(
      result.data as List<double>,
      data as List<int>,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x.toDouble() / y.toDouble(),
    );
    return result;
  }

  // --- Mixed Int64 arguments (Upcast promotion) ---

  NDArray<Int64> addInt64(NDArray<Int64> other, {NDArray<Int64>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result = into ?? NDArray<Int64>.create(expectedShape, DType.int64);
    final broadcastResult = broadcast(this, other);
    _elementWiseIntOp(
      result.data as List<int>,
      data as List<int>,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x + y,
    );
    return result;
  }

  NDArray<Int64> subtractInt64(NDArray<Int64> other, {NDArray<Int64>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result = into ?? NDArray<Int64>.create(expectedShape, DType.int64);
    final broadcastResult = broadcast(this, other);
    _elementWiseIntOp(
      result.data as List<int>,
      data as List<int>,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x - y,
    );
    return result;
  }

  NDArray<Int64> multiplyInt64(NDArray<Int64> other, {NDArray<Int64>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result = into ?? NDArray<Int64>.create(expectedShape, DType.int64);
    final broadcastResult = broadcast(this, other);
    _elementWiseIntOp(
      result.data as List<int>,
      data as List<int>,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x * y,
    );
    return result;
  }

  // --- Mixed Scalar int arguments ---

  NDArray<Int32> addScalar(int scalar, {NDArray<Int32>? into}) {
    final scalarArr = _wrapScalar(scalar, shape) as NDArray<Int32>;
    return add(scalarArr, into: into);
  }

  NDArray<Int32> subtractScalar(int scalar, {NDArray<Int32>? into}) {
    final scalarArr = _wrapScalar(scalar, shape) as NDArray<Int32>;
    return subtract(scalarArr, into: into);
  }

  NDArray<Int32> multiplyScalar(int scalar, {NDArray<Int32>? into}) {
    final scalarArr = _wrapScalar(scalar, shape) as NDArray<Int32>;
    return multiply(scalarArr, into: into);
  }
}

// =============================================================================
// ComplexNDArrayOperations (NDArray<Complex>)
// =============================================================================

extension ComplexNDArrayOperations on NDArray<Complex> {
  /// Element-wise addition with Complex NDArray.
  NDArray<Complex> add(NDArray<Complex> other, {NDArray<Complex>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Complex>.create(expectedShape, DType.complex128);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.complex128) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    if (isContiguous && other.isContiguous && _listEquals(shape, other.shape)) {
      v_add_complex(
        pointer.cast(),
        other.pointer.cast(),
        result.pointer.cast(),
        data.length,
      );
      return result;
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseComplexOp(
      result.data,
      data,
      other.data,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x + y,
    );
    return result;
  }

  /// Element-wise subtraction with Complex NDArray.
  NDArray<Complex> subtract(NDArray<Complex> other, {NDArray<Complex>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Complex>.create(expectedShape, DType.complex128);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.complex128) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    if (isContiguous && other.isContiguous && _listEquals(shape, other.shape)) {
      v_sub_complex(
        pointer.cast(),
        other.pointer.cast(),
        result.pointer.cast(),
        data.length,
      );
      return result;
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseComplexOp(
      result.data,
      data,
      other.data,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x - y,
    );
    return result;
  }

  /// Element-wise multiplication with Complex NDArray.
  NDArray<Complex> multiply(NDArray<Complex> other, {NDArray<Complex>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Complex>.create(expectedShape, DType.complex128);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.complex128) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    if (isContiguous && other.isContiguous && _listEquals(shape, other.shape)) {
      v_mul_complex(
        pointer.cast(),
        other.pointer.cast(),
        result.pointer.cast(),
        data.length,
      );
      return result;
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseComplexOp(
      result.data,
      data,
      other.data,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x * y,
    );
    return result;
  }

  /// Element-wise division with Complex NDArray.
  NDArray<Complex> divide(NDArray<Complex> other, {NDArray<Complex>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Complex>.create(expectedShape, DType.complex128);
    if (into != null) {
      if (!_listEquals(into.shape, expectedShape) ||
          into.dtype != DType.complex128) {
        throw ArgumentError('Incompatible into shape or dtype.');
      }
    }

    if (isContiguous && other.isContiguous && _listEquals(shape, other.shape)) {
      v_div_complex(
        pointer.cast(),
        other.pointer.cast(),
        result.pointer.cast(),
        data.length,
      );
      return result;
    }

    final broadcastResult = broadcast(this, other);
    _elementWiseComplexOp(
      result.data,
      data,
      other.data,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x / y,
    );
    return result;
  }

  // --- Mixed Float64 arguments ---

  NDArray<Complex> addFloat64(
    NDArray<Float64> other, {
    NDArray<Complex>? into,
  }) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Complex>.create(expectedShape, DType.complex128);
    final broadcastResult = broadcast(this, other);
    _elementWiseComplexDoubleOp(
      result.data,
      data,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x + y,
    );
    return result;
  }

  NDArray<Complex> subtractFloat64(
    NDArray<Float64> other, {
    NDArray<Complex>? into,
  }) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Complex>.create(expectedShape, DType.complex128);
    final broadcastResult = broadcast(this, other);
    _elementWiseComplexDoubleOp(
      result.data,
      data,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x - y,
    );
    return result;
  }

  NDArray<Complex> multiplyFloat64(
    NDArray<Float64> other, {
    NDArray<Complex>? into,
  }) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Complex>.create(expectedShape, DType.complex128);
    final broadcastResult = broadcast(this, other);
    _elementWiseComplexDoubleOp(
      result.data,
      data,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x * y,
    );
    return result;
  }

  NDArray<Complex> divideFloat64(
    NDArray<Float64> other, {
    NDArray<Complex>? into,
  }) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Complex>.create(expectedShape, DType.complex128);
    final broadcastResult = broadcast(this, other);
    _elementWiseComplexDoubleOp(
      result.data,
      data,
      other.data as List<double>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x / y,
    );
    return result;
  }

  // --- Mixed Int64 arguments ---

  NDArray<Complex> addInt64(NDArray<Int64> other, {NDArray<Complex>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Complex>.create(expectedShape, DType.complex128);
    final broadcastResult = broadcast(this, other);
    _elementWiseComplexIntOp(
      result.data,
      data,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x + y,
    );
    return result;
  }

  NDArray<Complex> subtractInt64(
    NDArray<Int64> other, {
    NDArray<Complex>? into,
  }) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Complex>.create(expectedShape, DType.complex128);
    final broadcastResult = broadcast(this, other);
    _elementWiseComplexIntOp(
      result.data,
      data,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x - y,
    );
    return result;
  }

  NDArray<Complex> multiplyInt64(
    NDArray<Int64> other, {
    NDArray<Complex>? into,
  }) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Complex>.create(expectedShape, DType.complex128);
    final broadcastResult = broadcast(this, other);
    _elementWiseComplexIntOp(
      result.data,
      data,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x * y,
    );
    return result;
  }

  NDArray<Complex> divideInt64(NDArray<Int64> other, {NDArray<Complex>? into}) {
    final expectedShape = ops.broadcastShapes(shape, other.shape);
    final result =
        into ?? NDArray<Complex>.create(expectedShape, DType.complex128);
    final broadcastResult = broadcast(this, other);
    _elementWiseComplexIntOp(
      result.data,
      data,
      other.data as List<int>,
      expectedShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      NDArray.computeCStrides(expectedShape),
      0,
      offsetElements,
      other.offsetElements,
      result.offsetElements,
      (x, y) => x / y,
    );
    return result;
  }

  // --- Mixed Scalar arguments ---

  NDArray<Complex> addScalar(dynamic scalar, {NDArray<Complex>? into}) {
    final scalarArr = _wrapScalar(scalar, shape) as NDArray<Complex>;
    return add(scalarArr, into: into);
  }

  NDArray<Complex> subtractScalar(dynamic scalar, {NDArray<Complex>? into}) {
    final scalarArr = _wrapScalar(scalar, shape) as NDArray<Complex>;
    return subtract(scalarArr, into: into);
  }

  NDArray<Complex> multiplyScalar(dynamic scalar, {NDArray<Complex>? into}) {
    final scalarArr = _wrapScalar(scalar, shape) as NDArray<Complex>;
    return multiply(scalarArr, into: into);
  }

  NDArray<Complex> divideScalar(dynamic scalar, {NDArray<Complex>? into}) {
    final scalarArr = _wrapScalar(scalar, shape) as NDArray<Complex>;
    return divide(scalarArr, into: into);
  }
}

// =============================================================================
// Private Recursive Strided Walkers
// =============================================================================

void _elementWiseDoubleOp(
  List<double> result,
  List<double> a,
  List<double> b,
  List<int> shape,
  List<int> stridesA,
  List<int> stridesB,
  List<int> stridesResult,
  int dim,
  int offsetA,
  int offsetB,
  int offsetResult,
  double Function(double, double) op,
) {
  if (dim == shape.length) {
    result[offsetResult] = op(a[offsetA], b[offsetB]);
    return;
  }
  for (var i = 0; i < shape[dim]; i++) {
    _elementWiseDoubleOp(
      result,
      a,
      b,
      shape,
      stridesA,
      stridesB,
      stridesResult,
      dim + 1,
      offsetA + i * stridesA[dim],
      offsetB + i * stridesB[dim],
      offsetResult + i * stridesResult[dim],
      op,
    );
  }
}

void _elementWiseIntOp(
  List<int> result,
  List<int> a,
  List<int> b,
  List<int> shape,
  List<int> stridesA,
  List<int> stridesB,
  List<int> stridesResult,
  int dim,
  int offsetA,
  int offsetB,
  int offsetResult,
  int Function(int, int) op,
) {
  if (dim == shape.length) {
    result[offsetResult] = op(a[offsetA], b[offsetB]);
    return;
  }
  for (var i = 0; i < shape[dim]; i++) {
    _elementWiseIntOp(
      result,
      a,
      b,
      shape,
      stridesA,
      stridesB,
      stridesResult,
      dim + 1,
      offsetA + i * stridesA[dim],
      offsetB + i * stridesB[dim],
      offsetResult + i * stridesResult[dim],
      op,
    );
  }
}

void _elementWiseDoubleIntOp(
  List<double> result,
  List<double> a,
  List<int> b,
  List<int> shape,
  List<int> stridesA,
  List<int> stridesB,
  List<int> stridesResult,
  int dim,
  int offsetA,
  int offsetB,
  int offsetResult,
  double Function(double, int) op,
) {
  if (dim == shape.length) {
    result[offsetResult] = op(a[offsetA], b[offsetB]);
    return;
  }
  for (var i = 0; i < shape[dim]; i++) {
    _elementWiseDoubleIntOp(
      result,
      a,
      b,
      shape,
      stridesA,
      stridesB,
      stridesResult,
      dim + 1,
      offsetA + i * stridesA[dim],
      offsetB + i * stridesB[dim],
      offsetResult + i * stridesResult[dim],
      op,
    );
  }
}

void _elementWiseComplexOp(
  List<Complex> result,
  List<Complex> a,
  List<Complex> b,
  List<int> shape,
  List<int> stridesA,
  List<int> stridesB,
  List<int> stridesResult,
  int dim,
  int offsetA,
  int offsetB,
  int offsetResult,
  Complex Function(Complex, Complex) op,
) {
  if (dim == shape.length) {
    result[offsetResult] = op(a[offsetA], b[offsetB]);
    return;
  }
  for (var i = 0; i < shape[dim]; i++) {
    _elementWiseComplexOp(
      result,
      a,
      b,
      shape,
      stridesA,
      stridesB,
      stridesResult,
      dim + 1,
      offsetA + i * stridesA[dim],
      offsetB + i * stridesB[dim],
      offsetResult + i * stridesResult[dim],
      op,
    );
  }
}

void _elementWiseComplexDoubleOp(
  List<Complex> result,
  List<Complex> a,
  List<double> b,
  List<int> shape,
  List<int> stridesA,
  List<int> stridesB,
  List<int> stridesResult,
  int dim,
  int offsetA,
  int offsetB,
  int offsetResult,
  Complex Function(Complex, double) op,
) {
  if (dim == shape.length) {
    result[offsetResult] = op(a[offsetA], b[offsetB]);
    return;
  }
  for (var i = 0; i < shape[dim]; i++) {
    _elementWiseComplexDoubleOp(
      result,
      a,
      b,
      shape,
      stridesA,
      stridesB,
      stridesResult,
      dim + 1,
      offsetA + i * stridesA[dim],
      offsetB + i * stridesB[dim],
      offsetResult + i * stridesResult[dim],
      op,
    );
  }
}

void _elementWiseComplexIntOp(
  List<Complex> result,
  List<Complex> a,
  List<int> b,
  List<int> shape,
  List<int> stridesA,
  List<int> stridesB,
  List<int> stridesResult,
  int dim,
  int offsetA,
  int offsetB,
  int offsetResult,
  Complex Function(Complex, int) op,
) {
  if (dim == shape.length) {
    result[offsetResult] = op(a[offsetA], b[offsetB]);
    return;
  }
  for (var i = 0; i < shape[dim]; i++) {
    _elementWiseComplexIntOp(
      result,
      a,
      b,
      shape,
      stridesA,
      stridesB,
      stridesResult,
      dim + 1,
      offsetA + i * stridesA[dim],
      offsetB + i * stridesB[dim],
      offsetResult + i * stridesResult[dim],
      op,
    );
  }
}

// =============================================================================
// Private Internal Helpers
// =============================================================================

bool _listEquals<E>(List<E> a, List<E> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

NDArray _wrapScalar(dynamic value, List<int> targetShape) {
  if (value is Complex) {
    return NDArray.fromList(
      <Complex>[value],
      List.filled(targetShape.length, 1),
      DType.complex128,
    );
  } else if (value is double) {
    return NDArray.fromList(
      <double>[value],
      List.filled(targetShape.length, 1),
      DType.float64,
    );
  } else if (value is int) {
    return NDArray.fromList(
      <int>[value],
      List.filled(targetShape.length, 1),
      DType.int64,
    );
  }
  throw ArgumentError('Unsupported scalar type: ${value.runtimeType}');
}

void _elementWiseIntIntToDoubleOp(
  List<double> result,
  List<int> a,
  List<int> b,
  List<int> shape,
  List<int> stridesA,
  List<int> stridesB,
  List<int> stridesResult,
  int dim,
  int offsetA,
  int offsetB,
  int offsetResult,
  double Function(int, int) op,
) {
  if (dim == shape.length) {
    result[offsetResult] = op(a[offsetA], b[offsetB]);
    return;
  }
  for (var i = 0; i < shape[dim]; i++) {
    _elementWiseIntIntToDoubleOp(
      result,
      a,
      b,
      shape,
      stridesA,
      stridesB,
      stridesResult,
      dim + 1,
      offsetA + i * stridesA[dim],
      offsetB + i * stridesB[dim],
      offsetResult + i * stridesResult[dim],
      op,
    );
  }
}
