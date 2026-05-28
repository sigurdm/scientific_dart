// ignore_for_file: non_constant_identifier_names
import 'dart:typed_data';
import 'dart:math' as math;
import 'dart:math' show Random;
import 'dart:io';
import 'package:archive/archive.dart';
import 'package:pocketfft/pocketfft.dart';
import '../ndarray.dart';
import 'package:openblas/openblas.dart';
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import '../ndarray_bindings.dart';
import '../scratch_arena.dart';

// Standalone operational relative cross-imports
import 'math.dart';
import 'stats.dart';
import 'sorting.dart';
import 'linalg.dart';
import 'spacers.dart';
import 'manipulation.dart';
import 'broadcasting.dart';
import 'splitting.dart';
import 'shaping_meshes.dart';
import 'repeating_tiling.dart';
import 'io.dart';
import 'random.dart';
import 'fft.dart';
import 'calculus.dart';

int mapSortKind(SortKind kind) {
  switch (kind) {
    case SortKind.quicksort:
      return 0;
    case SortKind.mergesort:
    case SortKind.stable:
      return 1;
    case SortKind.heapsort:
      return 2;
  }
}

DType resolveDType(DType a, DType b) {
  if (a == DType.boolean && b == DType.boolean) return DType.uint8;
  if (a == DType.boolean) return b;
  if (b == DType.boolean) return a;
  if (a == b) return a;
  if (a == DType.complex128 || b == DType.complex128) return DType.complex128;
  if (a == DType.complex64 || b == DType.complex64) {
    if (a == DType.float64 || b == DType.float64) return DType.complex128;
    return DType.complex64;
  }
  if (a == DType.float64 || b == DType.float64) return DType.float64;
  if (a == DType.float32 || b == DType.float32) return DType.float32;
  if (a == DType.int64 || b == DType.int64) return DType.int64;
  return DType.int32;
}

DType<T> defaultDType<T>() {
  if (T == Complex) return DType.complex128 as DType<T>;
  if (T == int) return DType.int64 as DType<T>;
  return DType.float64 as DType<T>;
}

Object normalizeScalar(Object o, DType dtype) {
  switch (dtype) {
    case DType.complex64:
    case DType.complex128:
      if (o is Complex) return o;
      if (o is num) return Complex(o.toDouble(), 0.0);
      return Complex((o as num).toDouble(), 0.0);
    case DType.float32:
    case DType.float64:
      if (o is num) return o.toDouble();
      if (o is Complex) return o.real;
      return (o as num).toDouble();
    case DType.int32:
    case DType.int64:
    case DType.int16:
    case DType.uint8:
      if (o is num) return o.toInt();
      if (o is Complex) return o.real.toInt();
      return (o as num).toInt();
    case DType.boolean:
      if (o is bool) return o;
      if (o is num) return o != 0;
      return o as bool;
  }
}

NDArray<T> toNDArray<T>(Object o, DType<T> dtype) {
  if (o is NDArray) {
    if (o.dtype == dtype) return o as NDArray<T>;
    return castNDArray(o, dtype);
  }
  final normalized = normalizeScalar(o, dtype);
  return NDArray<T>.fromList([normalized], [], dtype);
}

(NDArray<T>, T) linspaceInternal<T>(
  T start,
  T stop,
  int numSamples, {
  bool endpoint = true,
  DType<T>? dtype,
}) {
  if (numSamples <= 0) throw ArgumentError('numSamples must be positive');

  final resolvedDType = dtype ?? defaultDType<T>();

  final div = endpoint ? (numSamples - 1) : numSamples;
  final arr = NDArray<T>.create([numSamples], resolvedDType);
  T step;

  switch (resolvedDType) {
    case DType.float64:
      final s = (start as num).toDouble();
      final e = (stop as num).toDouble();
      final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
      v_linspace_double(arr.pointer.cast(), s, stp, numSamples);
      step = normalizeScalar(stp, resolvedDType) as T;
    case DType.float32:
      final s = (start as num).toDouble();
      final e = (stop as num).toDouble();
      final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
      v_linspace_float(arr.pointer.cast(), s, stp, numSamples);
      step = normalizeScalar(stp, resolvedDType) as T;
    case DType.complex128:
      final s = normalizeScalar(start as Object, DType.complex128) as Complex;
      final e = normalizeScalar(stop as Object, DType.complex128) as Complex;
      final stp = numSamples <= 1 ? Complex(0, 0) : (e - s) / div;
      v_linspace_complex128(
        arr.pointer.cast(),
        s.real,
        s.imag,
        stp.real,
        stp.imag,
        numSamples,
      );
      step = normalizeScalar(stp, resolvedDType) as T;
    case DType.complex64:
      final s = normalizeScalar(start as Object, DType.complex128) as Complex;
      final e = normalizeScalar(stop as Object, DType.complex128) as Complex;
      final stp = numSamples <= 1 ? Complex(0, 0) : (e - s) / div;
      v_linspace_complex64(
        arr.pointer.cast(),
        s.real,
        s.imag,
        stp.real,
        stp.imag,
        numSamples,
      );
      step = normalizeScalar(stp, resolvedDType) as T;
    case DType.int64:
      final s = (start as num).toDouble();
      final e = (stop as num).toDouble();
      final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
      v_linspace_int64(arr.pointer.cast(), s, stp, numSamples);
      step = normalizeScalar(stp, resolvedDType) as T;
    case DType.int32:
      final s = (start as num).toDouble();
      final e = (stop as num).toDouble();
      final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
      v_linspace_int32(arr.pointer.cast(), s, stp, numSamples);
      step = normalizeScalar(stp, resolvedDType) as T;
    case DType.int16:
      final s = (start as num).toDouble();
      final e = (stop as num).toDouble();
      final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
      v_linspace_int16(arr.pointer.cast(), s, stp, numSamples);
      step = normalizeScalar(stp, resolvedDType) as T;
    case DType.uint8:
      final s = (start as num).toDouble();
      final e = (stop as num).toDouble();
      final stp = numSamples <= 1 ? 0.0 : (e - s) / div;
      v_linspace_uint8(arr.pointer.cast(), s, stp, numSamples);
      step = normalizeScalar(stp, resolvedDType) as T;
    case DType.boolean:
      throw UnsupportedError('linspace not supported for boolean arrays');
  }

  return (arr, step);
}

void elementWiseOp<Ta, Tb, Tr>(
  List<Tr> result,
  List<Ta> a,
  List<Tb> b,
  List<int> shape,
  List<int> stridesA,
  List<int> stridesB,
  List<int> stridesResult,
  int dim,
  int offsetA,
  int offsetB,
  int offsetResult,
  Tr Function(Ta, Tb) op,
) {
  if (shape.isEmpty) {
    result[offsetResult] = op(a[offsetA], b[offsetB]);
    return;
  }

  if (dim == shape.length - 1) {
    final limit = shape[dim];
    final strideA = stridesA[dim];
    final strideB = stridesB[dim];
    final strideResult = stridesResult[dim];

    for (var i = 0; i < limit; i++) {
      result[offsetResult + i * strideResult] = op(
        a[offsetA + i * strideA],
        b[offsetB + i * strideB],
      );
    }
    return;
  }

  final limit = shape[dim];
  final strideA = stridesA[dim];
  final strideB = stridesB[dim];
  final strideResult = stridesResult[dim];

  for (var i = 0; i < limit; i++) {
    elementWiseOp<Ta, Tb, Tr>(
      result,
      a,
      b,
      shape,
      stridesA,
      stridesB,
      stridesResult,
      dim + 1,
      offsetA + i * strideA,
      offsetB + i * strideB,
      offsetResult + i * strideResult,
      op,
    );
  }
}

/// Matrix multiplication for Float64 arrays using OpenBLAS.
List<int> broadcastStackShapes(List<int> sA, List<int> sB) {
  final lenA = sA.length;
  final lenB = sB.length;
  final maxLen = math.max(lenA, lenB);
  final result = List<int>.filled(maxLen, 0);

  for (var i = 0; i < maxLen; i++) {
    final dimA = (lenA - 1 - i >= 0) ? sA[lenA - 1 - i] : 1;
    final dimB = (lenB - 1 - i >= 0) ? sB[lenB - 1 - i] : 1;

    if (dimA == dimB) {
      result[maxLen - 1 - i] = dimA;
    } else if (dimA == 1) {
      result[maxLen - 1 - i] = dimB;
    } else if (dimB == 1) {
      result[maxLen - 1 - i] = dimA;
    } else {
      throw ArgumentError(
        'Incompatible stack shapes for broadcasting in matmul: $sA and $sB',
      );
    }
  }
  return result;
}

NDArray<double> promoteToDouble(NDArray a) {
  final res = NDArray<double>.create(a.shape, DType.float64);
  final data = res.data;
  final src = a.data;
  for (var i = 0; i < src.length; i++) {
    data[i] = (src[i] as num).toDouble();
  }
  return res;
}

NDArray<Complex> promoteToComplex(NDArray a) {
  final res = NDArray<Complex>.create(a.shape, DType.complex128);
  final data = res.data;
  final src = a.data;
  for (var i = 0; i < src.length; i++) {
    final val = src[i];
    if (val is Complex) {
      data[i] = val;
    } else {
      data[i] = Complex((val as num).toDouble(), 0.0);
    }
  }
  return res;
}

/// Recursive helper to accumulate sum and count of non-NaN elements along an axis.
void nanReduceRecursive<T>(
  NDArray<T> a,
  NDArray<T> result,
  NDArray<int> counts,
  List<int> coordA,
  List<int> coordRes,
  int axis,
  int dim,
) {
  if (dim == a.shape.length) {
    final val = a.getCell(coordA);
    if (val is double && val.isNaN) return;
    if (val is Complex && (val.real.isNaN || val.imag.isNaN)) return;
    final current = result.getCell(coordRes);
    result.setCell(coordRes, ((current as dynamic) + val) as T);
    counts.setCell(coordRes, counts.getCell(coordRes) + 1);
    return;
  }
  if (dim == axis) {
    for (var i = 0; i < a.shape[axis]; i++) {
      coordA[dim] = i;
      nanReduceRecursive<T>(a, result, counts, coordA, coordRes, axis, dim + 1);
    }
  } else {
    final resDim = dim < axis ? dim : dim - 1;
    for (var i = 0; i < a.shape[dim]; i++) {
      coordA[dim] = i;
      coordRes[resDim] = i;
      nanReduceRecursive<T>(a, result, counts, coordA, coordRes, axis, dim + 1);
    }
  }
}

void elementWiseMin(NDArray dest, NDArray src) {
  elementWiseMinRec(dest, src, dest.strides, src.strides, 0, 0, 0);
}

void elementWiseMinRec(
  NDArray dest,
  NDArray src,
  List<int> stridesDest,
  List<int> stridesSrc,
  int dim,
  int offsetDest,
  int offsetSrc,
) {
  final rank = dest.shape.length;
  if (dim == rank) {
    final dVal = dest.data[offsetDest];
    final sVal = src.data[offsetSrc];
    if (dVal is double && sVal is double) {
      dest.data[offsetDest] = math.min(dVal, sVal);
    } else if (dVal is int && sVal is int) {
      dest.data[offsetDest] = math.min(dVal, sVal);
    }
    return;
  }

  final limit = dest.shape[dim];
  for (var i = 0; i < limit; i++) {
    elementWiseMinRec(
      dest,
      src,
      stridesDest,
      stridesSrc,
      dim + 1,
      offsetDest + i * stridesDest[dim],
      offsetSrc + i * stridesSrc[dim],
    );
  }
}

void elementWiseMax(NDArray dest, NDArray src) {
  elementWiseMaxRec(dest, src, dest.strides, src.strides, 0, 0, 0);
}

void elementWiseMaxRec(
  NDArray dest,
  NDArray src,
  List<int> stridesDest,
  List<int> stridesSrc,
  int dim,
  int offsetDest,
  int offsetSrc,
) {
  final rank = dest.shape.length;
  if (dim == rank) {
    final dVal = dest.data[offsetDest];
    final sVal = src.data[offsetSrc];
    if (dVal is double && sVal is double) {
      dest.data[offsetDest] = math.max(dVal, sVal);
    } else if (dVal is int && sVal is int) {
      dest.data[offsetDest] = math.max(dVal, sVal);
    }
    return;
  }

  final limit = dest.shape[dim];
  for (var i = 0; i < limit; i++) {
    elementWiseMaxRec(
      dest,
      src,
      stridesDest,
      stridesSrc,
      dim + 1,
      offsetDest + i * stridesDest[dim],
      offsetSrc + i * stridesSrc[dim],
    );
  }
}

void elementWiseNanMin(NDArray dest, NDArray src) {
  elementWiseNanMinRec(dest, src, dest.strides, src.strides, 0, 0, 0);
}

void elementWiseNanMinRec(
  NDArray dest,
  NDArray src,
  List<int> stridesDest,
  List<int> stridesSrc,
  int dim,
  int offsetDest,
  int offsetSrc,
) {
  final rank = dest.shape.length;
  if (dim == rank) {
    final dVal = dest.data[offsetDest];
    final sVal = src.data[offsetSrc];
    if (dVal is double && sVal is double) {
      if (sVal.isNaN) return;
      if (dVal.isNaN || sVal < dVal) {
        dest.data[offsetDest] = sVal;
      }
    } else if (dVal is int && sVal is int) {
      if (sVal < dVal) {
        dest.data[offsetDest] = sVal;
      }
    }
    return;
  }

  final limit = dest.shape[dim];
  for (var i = 0; i < limit; i++) {
    elementWiseNanMinRec(
      dest,
      src,
      stridesDest,
      stridesSrc,
      dim + 1,
      offsetDest + i * stridesDest[dim],
      offsetSrc + i * stridesSrc[dim],
    );
  }
}

void elementWiseNanMax(NDArray dest, NDArray src) {
  elementWiseNanMaxRec(dest, src, dest.strides, src.strides, 0, 0, 0);
}

void elementWiseNanMaxRec(
  NDArray dest,
  NDArray src,
  List<int> stridesDest,
  List<int> stridesSrc,
  int dim,
  int offsetDest,
  int offsetSrc,
) {
  final rank = dest.shape.length;
  if (dim == rank) {
    final dVal = dest.data[offsetDest];
    final sVal = src.data[offsetSrc];
    if (dVal is double && sVal is double) {
      if (sVal.isNaN) return;
      if (dVal.isNaN || sVal > dVal) {
        dest.data[offsetDest] = sVal;
      }
    } else if (dVal is int && sVal is int) {
      if (sVal > dVal) {
        dest.data[offsetDest] = sVal;
      }
    }
    return;
  }

  final limit = dest.shape[dim];
  for (var i = 0; i < limit; i++) {
    elementWiseNanMaxRec(
      dest,
      src,
      stridesDest,
      stridesSrc,
      dim + 1,
      offsetDest + i * stridesDest[dim],
      offsetSrc + i * stridesSrc[dim],
    );
  }
}

/// Recursive helper to traverse the leading stack dimensions of a multi-dimensional array.
///
/// Generates multidimensional coordinates of the stack/batch dimensions.
void walkStackCoords(
  List<int> stackShape,
  List<int> currentCoords,
  int dim,
  void Function(List<int> coords) leafCallback,
) {
  if (dim == stackShape.length) {
    leafCallback(currentCoords);
    return;
  }
  final limit = stackShape[dim];
  for (var i = 0; i < limit; i++) {
    currentCoords[dim] = i;
    walkStackCoords(stackShape, currentCoords, dim + 1, leafCallback);
  }
}

/// Recursive helper to traverse and reduce an array along an axis.
void reduceRecursive<S extends Object, D extends Object>(
  NDArray<S> src,
  NDArray<D> dest,
  List<int> currentPos,
  List<int> destPos,
  int targetAxis,
  int currentDim,
  D Function(D acc, S val) op,
) {
  if (currentDim == src.shape.length) {
    // Calculate flat index for src
    var srcOffset = 0;
    for (var i = 0; i < src.shape.length; i++) {
      srcOffset += currentPos[i] * src.strides[i];
    }

    // Calculate flat index for dest
    var destOffset = 0;
    for (var i = 0; i < dest.shape.length; i++) {
      destOffset += destPos[i] * dest.strides[i];
    }

    dest.data[destOffset] = op(dest.data[destOffset], src.data[srcOffset]);
    return;
  }

  for (var i = 0; i < src.shape[currentDim]; i++) {
    currentPos[currentDim] = i;
    if (currentDim < targetAxis) {
      destPos[currentDim] = i;
    } else if (currentDim > targetAxis) {
      destPos[currentDim - 1] = i;
    }
    reduceRecursive<S, D>(
      src,
      dest,
      currentPos,
      destPos,
      targetAxis,
      currentDim + 1,
      op,
    );
  }
}

void copyContiguousFlat(NDArray src, NDArray dest, int destOffset, int size) {
  final width = src.dtype.byteWidth;
  final destPtr = dest.pointer.cast<ffi.Uint8>() + destOffset * width;
  final srcPtr = src.pointer.cast<ffi.Uint8>();
  custom_memcpy(destPtr.cast(), srcPtr.cast(), size * width);
}

void copyConcatenateRecursive<T>(
  NDArray<T> src,
  NDArray<T> dest,
  int axis,
  int axisOffset,
  List<int> currentIndices,
  int currentDim,
) {
  if (currentDim == src.shape.length) {
    final destIndices = List<int>.from(currentIndices);
    destIndices[axis] += axisOffset;
    dest[destIndices] = src[currentIndices];
    return;
  }

  for (var i = 0; i < src.shape[currentDim]; i++) {
    currentIndices[currentDim] = i;
    copyConcatenateRecursive(
      src,
      dest,
      axis,
      axisOffset,
      currentIndices,
      currentDim + 1,
    );
  }
}

void copyStackRecursive(
  NDArray src,
  NDArray dest,
  int targetAxis,
  int axisOffset,
  List<int> currentIndices,
  int currentDim,
) {
  if (currentDim == src.shape.length) {
    final destIndices = List<int>.from(currentIndices);
    destIndices.insert(targetAxis, axisOffset);
    dest[destIndices] = src[currentIndices];
    return;
  }

  for (var i = 0; i < src.shape[currentDim]; i++) {
    currentIndices[currentDim] = i;
    copyStackRecursive(
      src,
      dest,
      targetAxis,
      axisOffset,
      currentIndices,
      currentDim + 1,
    );
  }
}

void unaryOp<Ta, Tr>(
  List<Tr> result,
  List<Ta> a,
  List<int> shape,
  List<int> stridesA,
  List<int> stridesResult,
  int dim,
  int offsetA,
  int offsetResult,
  Tr Function(Ta) op,
) {
  if (dim == shape.length) {
    result[offsetResult] = op(a[offsetA]);
    return;
  }

  for (var i = 0; i < shape[dim]; i++) {
    unaryOp<Ta, Tr>(
      result,
      a,
      shape,
      stridesA,
      stridesResult,
      dim + 1,
      offsetA + i * stridesA[dim],
      offsetResult + i * stridesResult[dim],
      op,
    );
  }
}

void ternaryOp<Ta, Tb, Tc, Tr>(
  List<Tr> result,
  List<Ta> a,
  List<Tb> b,
  List<Tc> c,
  List<int> shape,
  List<int> stridesA,
  List<int> stridesB,
  List<int> stridesC,
  List<int> stridesResult,
  int dim,
  int offsetA,
  int offsetB,
  int offsetC,
  int offsetResult,
  Tr Function(Ta, Tb, Tc) op,
) {
  if (shape.isEmpty) {
    result[offsetResult] = op(a[offsetA], b[offsetB], c[offsetC]);
    return;
  }

  if (dim == shape.length - 1) {
    final limit = shape[dim];
    final strideA = stridesA[dim];
    final strideB = stridesB[dim];
    final strideC = stridesC[dim];
    final strideResult = stridesResult[dim];

    for (var i = 0; i < limit; i++) {
      result[offsetResult + i * strideResult] = op(
        a[offsetA + i * strideA],
        b[offsetB + i * strideB],
        c[offsetC + i * strideC],
      );
    }
    return;
  }

  final limit = shape[dim];
  final strideA = stridesA[dim];
  final strideB = stridesB[dim];
  final strideC = stridesC[dim];
  final strideResult = stridesResult[dim];

  for (var i = 0; i < limit; i++) {
    ternaryOp<Ta, Tb, Tc, Tr>(
      result,
      a,
      b,
      c,
      shape,
      stridesA,
      stridesB,
      stridesC,
      stridesResult,
      dim + 1,
      offsetA + i * strideA,
      offsetB + i * strideB,
      offsetC + i * strideC,
      offsetResult + i * strideResult,
      op,
    );
  }
}

bool isTrueHelper(dynamic x) {
  if (x is bool) {
    return x;
  } else if (x is Complex) {
    return x.real != 0.0 || x.imag != 0.0;
  } else if (x is num) {
    return x != 0;
  }
  return false;
}

void whereOpRec<Tc, Tx, Ty, Tr>(
  List<Tr> result,
  List<Tc> cond,
  List<Tx> x,
  List<Ty> y,
  List<int> shape,
  List<int> sCond,
  List<int> sX,
  List<int> sY,
  List<int> sResult,
  int dim,
  int oCond,
  int oX,
  int oY,
  int oResult,
) {
  if (dim == shape.length) {
    final cVal = cond[oCond];
    final isTrue = isTrueHelper(cVal);
    if (isTrue) {
      final xVal = x[oX];
      result[oResult] = (Tr == Complex && xVal is num)
          ? Complex(xVal.toDouble(), 0.0) as Tr
          : xVal as Tr;
    } else {
      final yVal = y[oY];
      result[oResult] = (Tr == Complex && yVal is num)
          ? Complex(yVal.toDouble(), 0.0) as Tr
          : yVal as Tr;
    }
    return;
  }

  for (var i = 0; i < shape[dim]; i++) {
    whereOpRec<Tc, Tx, Ty, Tr>(
      result,
      cond,
      x,
      y,
      shape,
      sCond,
      sX,
      sY,
      sResult,
      dim + 1,
      oCond + i * sCond[dim],
      oX + i * sX[dim],
      oY + i * sY[dim],
      oResult + i * sResult[dim],
    );
  }
}

void dispatchWhere(
  List rData,
  NDArray condition,
  NDArray x,
  NDArray y,
  List<int> shape,
  List<int> sCond,
  List<int> sX,
  List<int> sY,
  List<int> sResult,
) {
  final cData = condition.data;
  final xData = x.data;
  final yData = y.data;

  if (x.dtype == DType.complex128 ||
      x.dtype == DType.complex64 ||
      y.dtype == DType.complex128 ||
      y.dtype == DType.complex64) {
    final resList = rData as List<Complex>;
    whereOpRec<dynamic, dynamic, dynamic, Complex>(
      resList,
      cData,
      xData,
      yData,
      shape,
      sCond,
      sX,
      sY,
      sResult,
      0,
      0,
      0,
      0,
      0,
    );
  } else if (x.dtype == DType.float64 ||
      x.dtype == DType.float32 ||
      y.dtype == DType.float64 ||
      y.dtype == DType.float32) {
    final resList = rData as List<double>;
    whereOpRec<dynamic, dynamic, dynamic, double>(
      resList,
      cData,
      xData,
      yData,
      shape,
      sCond,
      sX,
      sY,
      sResult,
      0,
      0,
      0,
      0,
      0,
    );
  } else {
    final resList = rData as List<int>;
    whereOpRec<dynamic, dynamic, dynamic, int>(
      resList,
      cData,
      xData,
      yData,
      shape,
      sCond,
      sX,
      sY,
      sResult,
      0,
      0,
      0,
      0,
      0,
    );
  }
}

/// Return elements chosen from [x] or [y] depending on [condition].
///
/// This function corresponds to NumPy's `where` function.
///
/// **Ternary Select and SIMD Pipelines**:
/// - If [x] and [y] are omitted, this functions acts as a dynamic alias returning [nonzero(condition)].
/// - If all three arrays share identical shapes, are of `float32`/`int32` types, and are
///   C-contiguous, an optimized SIMD hardware vector pipeline can be wired.
/// - Non-contiguous layouts or mixed types fall back to portably generic upcasting loops.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
List<int> broadcastStrides(NDArray a, List<int> targetShape) {
  final strides = List<int>.filled(targetShape.length, 0);
  final offset = targetShape.length - a.shape.length;
  for (var i = 0; i < a.shape.length; i++) {
    final targetDim = targetShape[i + offset];
    final aDim = a.shape[i];
    if (aDim == targetDim) {
      strides[i + offset] = a.strides[i];
    } else if (aDim == 1) {
      strides[i + offset] = 0;
    } else {
      throw ArgumentError('Cannot broadcast shape ${a.shape} to $targetShape');
    }
  }
  return strides;
}

List<int> broadcast3Shapes(List<int> s1, List<int> s2, List<int> s3) {
  final len = math.max(s1.length, math.max(s2.length, s3.length));
  final common = List<int>.filled(len, 1);
  for (var i = 0; i < len; i++) {
    final dim1 = s1.length - 1 - i >= 0 ? s1[s1.length - 1 - i] : 1;
    final dim2 = s2.length - 1 - i >= 0 ? s2[s2.length - 1 - i] : 1;
    final dim3 = s3.length - 1 - i >= 0 ? s3[s3.length - 1 - i] : 1;

    final target = math.max(dim1, math.max(dim2, dim3));
    if (dim1 != target && dim1 != 1) {
      throw ArgumentError('Incompatible shapes for broadcasting');
    }
    if (dim2 != target && dim2 != 1) {
      throw ArgumentError('Incompatible shapes for broadcasting');
    }
    if (dim3 != target && dim3 != 1) {
      throw ArgumentError('Incompatible shapes for broadcasting');
    }
    common[len - 1 - i] = target;
  }
  return common;
}

void countNonzeroRecursive<T>(
  NDArray<T> src,
  NDArray<int> dest,
  List<int> srcPos,
  int targetAxis,
  int currentDim,
) {
  if (currentDim == src.shape.length) {
    if (isTrueHelper(src[srcPos])) {
      final destPos = List<int>.from(srcPos)..removeAt(targetAxis);
      var destOffset = 0;
      for (var i = 0; i < dest.shape.length; i++) {
        destOffset += destPos[i] * dest.strides[i];
      }
      dest.data[destOffset] += 1;
    }
    return;
  }

  for (var i = 0; i < src.shape[currentDim]; i++) {
    srcPos[currentDim] = i;
    countNonzeroRecursive<T>(src, dest, srcPos, targetAxis, currentDim + 1);
  }
}

void argMinMaxRecursive<T>(
  NDArray<T> src,
  NDArray<int> dest,
  List<int> srcPos,
  int targetAxis,
  int currentDim,
  bool isMax,
) {
  if (currentDim == src.shape.length) {
    final destPos = List<int>.from(srcPos)..removeAt(targetAxis);
    var destOffset = 0;
    for (var i = 0; i < dest.shape.length; i++) {
      destOffset += destPos[i] * dest.strides[i];
    }

    final currentBestIndex = dest.data[destOffset];
    final currentBestPos = List<int>.from(srcPos);
    currentBestPos[targetAxis] = currentBestIndex;

    final currentVal = src[srcPos] as num;
    final bestVal = src[currentBestPos] as num;

    if (isMax) {
      if (srcPos[targetAxis] > currentBestIndex && currentVal > bestVal) {
        dest.data[destOffset] = srcPos[targetAxis];
      }
    } else {
      if (srcPos[targetAxis] > currentBestIndex && currentVal < bestVal) {
        dest.data[destOffset] = srcPos[targetAxis];
      }
    }
    return;
  }

  for (var i = 0; i < src.shape[currentDim]; i++) {
    srcPos[currentDim] = i;
    argMinMaxRecursive<T>(src, dest, srcPos, targetAxis, currentDim + 1, isMax);
  }
}

void selectRecursive(
  NDArray result,
  List<NDArray<bool>> condlist,
  List<NDArray> choicelist,
  List<List<int>> stridesCond,
  List<List<int>> stridesChoice,
  List<int> resultStrides,
  List<int> currentPos,
  int dim,
  List<int> offsetsCond,
  List<int> offsetsChoice,
  int offsetRes,
  dynamic defaultValue,
) {
  final rank = result.shape.length;
  if (dim == rank) {
    var chosen = false;
    for (var j = 0; j < condlist.length; j++) {
      if (condlist[j].data[offsetsCond[j]]) {
        result.data[offsetRes] = castValue(
          choicelist[j].data[offsetsChoice[j]],
          result.dtype,
        );
        chosen = true;
        break;
      }
    }
    if (!chosen) {
      result.data[offsetRes] = castValue(defaultValue, result.dtype);
    }
    return;
  }

  final limit = result.shape[dim];
  for (var i = 0; i < limit; i++) {
    final nextOffsetsCond = List<int>.generate(
      condlist.length,
      (j) => offsetsCond[j] + i * stridesCond[j][dim],
    );
    final nextOffsetsChoice = List<int>.generate(
      choicelist.length,
      (j) => offsetsChoice[j] + i * stridesChoice[j][dim],
    );
    currentPos[dim] = i;
    selectRecursive(
      result,
      condlist,
      choicelist,
      stridesCond,
      stridesChoice,
      resultStrides,
      currentPos,
      dim + 1,
      nextOffsetsCond,
      nextOffsetsChoice,
      offsetRes + i * resultStrides[dim],
      defaultValue,
    );
  }
}

dynamic castValue(dynamic val, DType dtype) {
  if (dtype == DType.complex128 || dtype == DType.complex64) {
    if (val is Complex) return val;
    if (val is num) return Complex(val.toDouble(), 0.0);
    return Complex(0.0, 0.0);
  }
  if (dtype == DType.float64 || dtype == DType.float32) {
    if (val is num) return val.toDouble();
    if (val is Complex) return val.real;
    if (val is bool) return val ? 1.0 : 0.0;
    return 0.0;
  }
  if (dtype == DType.int64 ||
      dtype == DType.int32 ||
      dtype == DType.uint8 ||
      dtype == DType.int16) {
    if (val is num) return val.toInt();
    if (val is Complex) return val.real.toInt();
    if (val is bool) return val ? 1 : 0;
    return 0;
  }
  if (dtype == DType.boolean) {
    if (val is bool) return val;
    if (val is num) return val != 0;
    if (val is Complex) return val.real != 0.0 || val.imag != 0.0;
    return false;
  }
  return val;
}

enum CumOpType { sum, prod, min, max }

NDArray<T> cumOpFFI<T>(
  NDArray<T> a,
  int axis,
  NDArray<T> result,
  CumOpType opType,
) {
  final rank = a.shape.length;
  final cShape = malloc<ffi.Int>(rank);
  final cStridesA = malloc<ffi.Int>(rank);
  final cStridesRes = malloc<ffi.Int>(rank);

  for (var i = 0; i < rank; i++) {
    cShape[i] = a.shape[i];
    cStridesA[i] = a.strides[i];
    cStridesRes[i] = result.strides[i];
  }

  try {
    switch (opType) {
      case CumOpType.sum:
        final dtype = a.dtype;
        switch (dtype) {
          case DType.float64:
            s_cumsum_double(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.float32:
            s_cumsum_float(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.int64:
            s_cumsum_int64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.int32:
            s_cumsum_int32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.complex128:
            s_cumsum_complex128(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.complex64:
            s_cumsum_complex64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.uint8:
          case DType.int16:
          case DType.boolean:
            final doubleA = NDArray<double>.create(a.shape, DType.float64);
            for (var i = 0; i < a.data.length; i++) {
              doubleA.data[i] = (a.data[i] as num).toDouble();
            }
            final doubleRes = NDArray<double>.create(a.shape, DType.float64);
            final cStridesDoubleA = malloc<ffi.Int>(rank);
            final cStridesDoubleRes = malloc<ffi.Int>(rank);
            for (var i = 0; i < rank; i++) {
              cStridesDoubleA[i] = doubleA.strides[i];
              cStridesDoubleRes[i] = doubleRes.strides[i];
            }
            try {
              s_cumsum_double(
                doubleA.pointer.cast(),
                cStridesDoubleA,
                doubleRes.pointer.cast(),
                cStridesDoubleRes,
                cShape,
                rank,
                axis,
              );
            } finally {
              malloc.free(cStridesDoubleA);
              malloc.free(cStridesDoubleRes);
            }
            for (var i = 0; i < result.data.length; i++) {
              result.data[i] = castValue(doubleRes.data[i], a.dtype) as T;
            }
            doubleA.dispose();
            doubleRes.dispose();
        }

      case CumOpType.prod:
        final dtype = a.dtype;
        switch (dtype) {
          case DType.float64:
            s_cumprod_double(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.float32:
            s_cumprod_float(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.int64:
            s_cumprod_int64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.int32:
            s_cumprod_int32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.complex128:
            s_cumprod_complex128(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.complex64:
            s_cumprod_complex64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.uint8:
          case DType.int16:
          case DType.boolean:
            final doubleA = NDArray<double>.create(a.shape, DType.float64);
            for (var i = 0; i < a.data.length; i++) {
              doubleA.data[i] = (a.data[i] as num).toDouble();
            }
            final doubleRes = NDArray<double>.create(a.shape, DType.float64);
            final cStridesDoubleA = malloc<ffi.Int>(rank);
            final cStridesDoubleRes = malloc<ffi.Int>(rank);
            for (var i = 0; i < rank; i++) {
              cStridesDoubleA[i] = doubleA.strides[i];
              cStridesDoubleRes[i] = doubleRes.strides[i];
            }
            try {
              s_cumprod_double(
                doubleA.pointer.cast(),
                cStridesDoubleA,
                doubleRes.pointer.cast(),
                cStridesDoubleRes,
                cShape,
                rank,
                axis,
              );
            } finally {
              malloc.free(cStridesDoubleA);
              malloc.free(cStridesDoubleRes);
            }
            for (var i = 0; i < result.data.length; i++) {
              result.data[i] = castValue(doubleRes.data[i], a.dtype) as T;
            }
            doubleA.dispose();
            doubleRes.dispose();
        }

      case CumOpType.min:
        final dtype = a.dtype;
        switch (dtype) {
          case DType.float64:
            s_cummin_double(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.float32:
            s_cummin_float(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.int64:
            s_cummin_int64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.int32:
            s_cummin_int32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.uint8:
          case DType.int16:
          case DType.boolean:
            final doubleA = NDArray<double>.create(a.shape, DType.float64);
            for (var i = 0; i < a.data.length; i++) {
              doubleA.data[i] = (a.data[i] as num).toDouble();
            }
            final doubleRes = NDArray<double>.create(a.shape, DType.float64);
            final cStridesDoubleA = malloc<ffi.Int>(rank);
            final cStridesDoubleRes = malloc<ffi.Int>(rank);
            for (var i = 0; i < rank; i++) {
              cStridesDoubleA[i] = doubleA.strides[i];
              cStridesDoubleRes[i] = doubleRes.strides[i];
            }
            try {
              s_cummin_double(
                doubleA.pointer.cast(),
                cStridesDoubleA,
                doubleRes.pointer.cast(),
                cStridesDoubleRes,
                cShape,
                rank,
                axis,
              );
            } finally {
              malloc.free(cStridesDoubleA);
              malloc.free(cStridesDoubleRes);
            }
            for (var i = 0; i < result.data.length; i++) {
              result.data[i] = castValue(doubleRes.data[i], a.dtype) as T;
            }
            doubleA.dispose();
            doubleRes.dispose();
          case DType.complex128:
          case DType.complex64:
            throw ArgumentError(
              'Cumulative minimum is not defined for complex numbers.',
            );
        }

      case CumOpType.max:
        final dtype = a.dtype;
        switch (dtype) {
          case DType.float64:
            s_cummax_double(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.float32:
            s_cummax_float(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.int64:
            s_cummax_int64(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.int32:
            s_cummax_int32(
              a.pointer.cast(),
              cStridesA,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              axis,
            );
          case DType.uint8:
          case DType.int16:
          case DType.boolean:
            final doubleA = NDArray<double>.create(a.shape, DType.float64);
            for (var i = 0; i < a.data.length; i++) {
              doubleA.data[i] = (a.data[i] as num).toDouble();
            }
            final doubleRes = NDArray<double>.create(a.shape, DType.float64);
            final cStridesDoubleA = malloc<ffi.Int>(rank);
            final cStridesDoubleRes = malloc<ffi.Int>(rank);
            for (var i = 0; i < rank; i++) {
              cStridesDoubleA[i] = doubleA.strides[i];
              cStridesDoubleRes[i] = doubleRes.strides[i];
            }
            try {
              s_cummax_double(
                doubleA.pointer.cast(),
                cStridesDoubleA,
                doubleRes.pointer.cast(),
                cStridesDoubleRes,
                cShape,
                rank,
                axis,
              );
            } finally {
              malloc.free(cStridesDoubleA);
              malloc.free(cStridesDoubleRes);
            }
            for (var i = 0; i < result.data.length; i++) {
              result.data[i] = castValue(doubleRes.data[i], a.dtype) as T;
            }
            doubleA.dispose();
            doubleRes.dispose();
          case DType.complex128:
          case DType.complex64:
            throw ArgumentError(
              'Cumulative maximum is not defined for complex numbers.',
            );
        }
    }
  } finally {
    malloc.free(cShape);
    malloc.free(cStridesA);
    malloc.free(cStridesRes);
  }
  return result;
}

NDArray<R> castNDArray<R>(NDArray a, DType<R> targetDType) {
  if (a.dtype == targetDType) return a as NDArray<R>;
  final result = NDArray<R>.create(a.shape, targetDType);
  final aFlat = a.toList();
  for (var i = 0; i < aFlat.length; i++) {
    result.data[i] = castValue(aFlat[i], targetDType) as R;
  }
  return result;
}
