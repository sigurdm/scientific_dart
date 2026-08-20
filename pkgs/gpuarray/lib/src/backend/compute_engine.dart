import 'dart:math' as math;
import 'dart:ffi' as ffi;
import 'package:ndarray/ndarray.dart' show Float16Utils;
import '../exceptions.dart';
import '../dtype.dart';
import '../buffer.dart';

/// Shape and stride calculation utilities for GPU tensors.
final class ShapeUtils {
  ShapeUtils._();

  /// Computes default C-contiguous strides (in element counts) for a given [shape].
  static List<int> computeCStrides(List<int> shape) {
    if (shape.isEmpty) return [];
    final strides = List<int>.filled(shape.length, 1);
    for (var i = shape.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  }

  /// Calculates total number of elements in a tensor of the given [shape].
  static int computeSize(List<int> shape) {
    if (shape.isEmpty) return 1;
    var size = 1;
    for (final dim in shape) {
      if (dim < 0) {
        throw ArgumentError('Shape dimensions cannot be negative: $shape');
      }
      size *= dim;
    }
    return size;
  }

  /// Checks if [shape] and [strides] represent a contiguous C-layout.
  static bool isContiguous(List<int> shape, List<int> strides) {
    if (shape.isEmpty) return true;
    final expectedStrides = computeCStrides(shape);
    for (var i = 0; i < shape.length; i++) {
      if (shape[i] > 1 && strides[i] != expectedStrides[i]) {
        return false;
      }
    }
    return true;
  }

  /// Broadcasts two tensor shapes according to NumPy-style broadcasting rules.
  static List<int> broadcastShapes(List<int> shapeA, List<int> shapeB) {
    final rankA = shapeA.length;
    final rankB = shapeB.length;
    final maxRank = math.max(rankA, rankB);
    final result = List<int>.filled(maxRank, 0);

    for (var i = 0; i < maxRank; i++) {
      final dimA = (i < maxRank - rankA) ? 1 : shapeA[i - (maxRank - rankA)];
      final dimB = (i < maxRank - rankB) ? 1 : shapeB[i - (maxRank - rankB)];

      if (dimA == dimB) {
        result[i] = dimA;
      } else if (dimA == 1) {
        result[i] = dimB;
      } else if (dimB == 1) {
        result[i] = dimA;
      } else {
        throw GpuShapeMismatchException('broadcast', shapeA, shapeB);
      }
    }
    return result;
  }

  /// Computes broadcasted strides for an array with [shape] and [strides]
  /// when expanded to [targetShape].
  static List<int> broadcastStrides(
    List<int> shape,
    List<int> strides,
    List<int> targetShape,
  ) {
    final rank = shape.length;
    final targetRank = targetShape.length;
    final result = List<int>.filled(targetRank, 0);

    final rankDiff = targetRank - rank;
    for (var i = 0; i < targetRank; i++) {
      if (i < rankDiff) {
        result[i] = 0;
      } else {
        final origDim = shape[i - rankDiff];
        final targetDim = targetShape[i];
        if (origDim == 1 && targetDim > 1) {
          result[i] = 0; // Broadcasted axis has stride 0
        } else {
          result[i] = strides[i - rankDiff];
        }
      }
    }
    return result;
  }
}

/// Dispatches high-performance compute kernels over GPU buffers.
final class ComputeEngine {
  ComputeEngine._();

  /// Reads a single typed value at [elementIndex] from [buffer].
  static dynamic readAny(
    GpuBuffer buffer,
    DType dtype,
    int elementIndex, {
    int offsetElements = 0,
  }) {
    final idx = offsetElements + elementIndex;
    final ptr = buffer.pointer;
    switch (dtype) {
      case DType.float64:
        return ptr.cast<ffi.Double>()[idx];
      case DType.float32:
        return ptr.cast<ffi.Float>()[idx];
      case DType.float16:
        return Float16(Float16Utils.decodeFloat16(ptr.cast<ffi.Uint16>()[idx]));
      case DType.bfloat16:
        return BFloat16(
          Float16Utils.decodeBFloat16(ptr.cast<ffi.Uint16>()[idx]),
        );
      case DType.int64:
        return Int64(ptr.cast<ffi.Int64>()[idx]);
      case DType.int32:
        return Int32(ptr.cast<ffi.Int32>()[idx]);
      case DType.int16:
        return Int16(ptr.cast<ffi.Int16>()[idx]);
      case DType.int8:
        return Int8(ptr.cast<ffi.Int8>()[idx]);
      case DType.uint64:
        return Uint64(ptr.cast<ffi.Uint64>()[idx]);
      case DType.uint32:
        return Uint32(ptr.cast<ffi.Uint32>()[idx]);
      case DType.uint16:
        return Uint16(ptr.cast<ffi.Uint16>()[idx]);
      case DType.uint8:
        return Uint8(ptr.cast<ffi.Uint8>()[idx]);
      case DType.boolean:
        return ptr.cast<ffi.Uint8>()[idx] != 0;
      case DType.complex64:
        final real = ptr.cast<ffi.Float>()[idx * 2];
        final imag = ptr.cast<ffi.Float>()[idx * 2 + 1];
        return Complex64(real, imag);
      case DType.complex128:
        final real = ptr.cast<ffi.Double>()[idx * 2];
        final imag = ptr.cast<ffi.Double>()[idx * 2 + 1];
        return Complex128(real, imag);
    }
  }

  /// Writes a single typed value at [elementIndex] to [buffer].
  static void writeAny(
    GpuBuffer buffer,
    DType dtype,
    int elementIndex,
    dynamic value, {
    int offsetElements = 0,
  }) {
    final idx = offsetElements + elementIndex;
    final ptr = buffer.pointer;
    double toDoubleVal(dynamic v) {
      if (v is bool) return v ? 1.0 : 0.0;
      if (v is num) return v.toDouble();
      if (v is Complex) return v.real;
      return 0.0;
    }

    int toIntVal(dynamic v) {
      if (v is bool) return v ? 1 : 0;
      if (v is num) return v.toInt();
      if (v is BigInt) return v.toSigned(64).toInt();
      return 0;
    }

    switch (dtype) {
      case DType.float64:
        ptr.cast<ffi.Double>()[idx] = toDoubleVal(value);
        break;
      case DType.float32:
        ptr.cast<ffi.Float>()[idx] = toDoubleVal(value);
        break;
      case DType.float16:
        ptr.cast<ffi.Uint16>()[idx] = Float16Utils.encodeFloat16(toDoubleVal(value));
        break;
      case DType.bfloat16:
        ptr.cast<ffi.Uint16>()[idx] = Float16Utils.encodeBFloat16(toDoubleVal(value));
        break;
      case DType.int64:
        ptr.cast<ffi.Int64>()[idx] = toIntVal(value);
        break;
      case DType.int32:
        ptr.cast<ffi.Int32>()[idx] = toIntVal(value);
        break;
      case DType.int16:
        ptr.cast<ffi.Int16>()[idx] = toIntVal(value);
        break;
      case DType.int8:
        ptr.cast<ffi.Int8>()[idx] = toIntVal(value);
        break;
      case DType.uint64:
        ptr.cast<ffi.Uint64>()[idx] = toIntVal(value);
        break;
      case DType.uint32:
        ptr.cast<ffi.Uint32>()[idx] = toIntVal(value);
        break;
      case DType.uint16:
        ptr.cast<ffi.Uint16>()[idx] = toIntVal(value);
        break;
      case DType.uint8:
        ptr.cast<ffi.Uint8>()[idx] = toIntVal(value);
        break;
      case DType.boolean:
        ptr.cast<ffi.Uint8>()[idx] = (value == true || (value is num && value != 0)) ? 1 : 0;
        break;
      case DType.complex64:
        if (value is Complex) {
          ptr.cast<ffi.Float>()[idx * 2] = value.real;
          ptr.cast<ffi.Float>()[idx * 2 + 1] = value.imag;
        } else {
          ptr.cast<ffi.Float>()[idx * 2] = toDoubleVal(value);
          ptr.cast<ffi.Float>()[idx * 2 + 1] = 0.0;
        }
        break;
      case DType.complex128:
        if (value is Complex) {
          ptr.cast<ffi.Double>()[idx * 2] = value.real;
          ptr.cast<ffi.Double>()[idx * 2 + 1] = value.imag;
        } else {
          ptr.cast<ffi.Double>()[idx * 2] = toDoubleVal(value);
          ptr.cast<ffi.Double>()[idx * 2 + 1] = 0.0;
        }
        break;
    }
  }

  /// Reads a single numerical value at [elementIndex] from [buffer].
  static double readValue(
    GpuBuffer buffer,
    DType dtype,
    int elementIndex, {
    int offsetElements = 0,
  }) {
    final idx = offsetElements + elementIndex;
    final ptr = buffer.pointer;
    switch (dtype) {
      case DType.float64:
        return ptr.cast<ffi.Double>()[idx];
      case DType.float32:
        return ptr.cast<ffi.Float>()[idx];
      case DType.float16:
        return Float16Utils.decodeFloat16(ptr.cast<ffi.Uint16>()[idx]);
      case DType.bfloat16:
        return Float16Utils.decodeBFloat16(ptr.cast<ffi.Uint16>()[idx]);
      case DType.int64:
        return ptr.cast<ffi.Int64>()[idx].toDouble();
      case DType.int32:
        return ptr.cast<ffi.Int32>()[idx].toDouble();
      case DType.int16:
        return ptr.cast<ffi.Int16>()[idx].toDouble();
      case DType.int8:
        return ptr.cast<ffi.Int8>()[idx].toDouble();
      case DType.uint64:
        final raw = ptr.cast<ffi.Uint64>()[idx];
        return raw >= 0
            ? raw.toDouble()
            : BigInt.from(raw).toUnsigned(64).toDouble();
      case DType.uint32:
        return ptr.cast<ffi.Uint32>()[idx].toDouble();
      case DType.uint16:
        return ptr.cast<ffi.Uint16>()[idx].toDouble();
      case DType.uint8:
        return ptr.cast<ffi.Uint8>()[idx].toDouble();
      case DType.boolean:
        return ptr.cast<ffi.Uint8>()[idx] != 0 ? 1.0 : 0.0;
      case DType.complex64:
        return ptr.cast<ffi.Float>()[idx * 2];
      case DType.complex128:
        return ptr.cast<ffi.Double>()[idx * 2];
    }
  }

  /// Writes a single numerical value at [elementIndex] to [buffer].
  static void writeValue(
    GpuBuffer buffer,
    DType dtype,
    int elementIndex,
    double value, {
    int offsetElements = 0,
  }) {
    writeAny(
      buffer,
      dtype,
      elementIndex,
      value,
      offsetElements: offsetElements,
    );
  }
}
