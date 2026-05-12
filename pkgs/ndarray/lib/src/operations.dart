// ignore_for_file: non_constant_identifier_names
@ffi.DefaultAsset('package:openblas/openblas')
library;

import 'dart:typed_data';
import 'dart:math' as math;
import 'ndarray.dart';
import 'broadcasting.dart';
import 'package:openblas/openblas.dart';
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'ndarray_bindings.dart';
import 'package:meta/meta.dart';

import 'binary_ops.dart';

// LAPACK Extension Bindings linking explicitly to package:openblas asset via DefaultAsset

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Uint8,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
    ffi.Int,
  )
>()
external int LAPACKE_dpotrf(
  int matrix_layout,
  int uplo,
  int n,
  ffi.Pointer<ffi.Double> a,
  int lda,
);

@ffi.Native<
  ffi.Int Function(ffi.Int, ffi.Uint8, ffi.Int, ffi.Pointer<ffi.Float>, ffi.Int)
>()
external int LAPACKE_spotrf(
  int matrix_layout,
  int uplo,
  int n,
  ffi.Pointer<ffi.Float> a,
  int lda,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
  )
>()
external int LAPACKE_dgeqrf(
  int matrix_layout,
  int m,
  int n,
  ffi.Pointer<ffi.Double> a,
  int lda,
  ffi.Pointer<ffi.Double> tau,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
  )
>()
external int LAPACKE_sgeqrf(
  int matrix_layout,
  int m,
  int n,
  ffi.Pointer<ffi.Float> a,
  int lda,
  ffi.Pointer<ffi.Float> tau,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
  )
>()
external int LAPACKE_dorgqr(
  int matrix_layout,
  int m,
  int n,
  int k,
  ffi.Pointer<ffi.Double> a,
  int lda,
  ffi.Pointer<ffi.Double> tau,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
  )
>()
external int LAPACKE_sorgqr(
  int matrix_layout,
  int m,
  int n,
  int k,
  ffi.Pointer<ffi.Float> a,
  int lda,
  ffi.Pointer<ffi.Float> tau,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Uint8,
    ffi.Uint8,
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Double>,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
  )
>()
external int LAPACKE_dgesvd(
  int matrix_layout,
  int jobu,
  int jobvt,
  int m,
  int n,
  ffi.Pointer<ffi.Double> a,
  int lda,
  ffi.Pointer<ffi.Double> s,
  ffi.Pointer<ffi.Double> u,
  int ldu,
  ffi.Pointer<ffi.Double> vt,
  int ldvt,
  ffi.Pointer<ffi.Double> superb,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Uint8,
    ffi.Uint8,
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
  )
>()
external int LAPACKE_sgesvd(
  int matrix_layout,
  int jobu,
  int jobvt,
  int m,
  int n,
  ffi.Pointer<ffi.Float> a,
  int lda,
  ffi.Pointer<ffi.Float> s,
  ffi.Pointer<ffi.Float> u,
  int ldu,
  ffi.Pointer<ffi.Float> vt,
  int ldvt,
  ffi.Pointer<ffi.Float> superb,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
    ffi.Pointer<ffi.Int>,
  )
>()
external int LAPACKE_sgetrf(
  int matrix_layout,
  int m,
  int n,
  ffi.Pointer<ffi.Float> a,
  int lda,
  ffi.Pointer<ffi.Int> ipiv,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
    ffi.Pointer<ffi.Int>,
  )
>()
external int LAPACKE_sgetri(
  int matrix_layout,
  int n,
  ffi.Pointer<ffi.Float> a,
  int lda,
  ffi.Pointer<ffi.Int> ipiv,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Uint8,
    ffi.Uint8,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Double>,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
    ffi.Int,
  )
>()
external int LAPACKE_dgeev(
  int matrix_layout,
  int jobvl,
  int jobvr,
  int n,
  ffi.Pointer<ffi.Double> a,
  int lda,
  ffi.Pointer<ffi.Double> wr,
  ffi.Pointer<ffi.Double> wi,
  ffi.Pointer<ffi.Double> vl,
  int ldvl,
  ffi.Pointer<ffi.Double> vr,
  int ldvr,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Uint8,
    ffi.Uint8,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
  )
>()
external int LAPACKE_sgeev(
  int matrix_layout,
  int jobvl,
  int jobvr,
  int n,
  ffi.Pointer<ffi.Float> a,
  int lda,
  ffi.Pointer<ffi.Float> wr,
  ffi.Pointer<ffi.Float> wi,
  ffi.Pointer<ffi.Float> vl,
  int ldvl,
  ffi.Pointer<ffi.Float> vr,
  int ldvr,
);

/// Operations that can call out to FFI.
enum Operation { add, matmul }

/// Global configuration for FFI breakover thresholds.
/// Keys are operation names, values are array size thresholds.
final Map<Operation, int> ffiThresholds = {Operation.matmul: 1};

/// Configure the number of parallel execution threads used by OpenBLAS at runtime.
///
/// **Preconditions:**
/// - [numThreads] must be greater than or equal to 1.
///
/// **Throws:**
/// - [ArgumentError] if [numThreads] is less than 1.
///
/// **Example:**
/// ```dart
/// setNumThreads(1); // Disable multi-threading to bypass overhead on small matrices
/// ```
void setNumThreads(int numThreads) {
  if (numThreads < 1) {
    throw ArgumentError(
      'Number of threads must be at least 1 (was $numThreads)',
    );
  }
  openblas_set_num_threads(numThreads);
}

bool listEquals(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

DType _resolveDType(DType a, DType b) {
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

void _elementWiseOp<Ta, Tb, Tr>(
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
    _elementWiseOp<Ta, Tb, Tr>(
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
List<int> _broadcastStackShapes(List<int> sA, List<int> sB) {
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

/// Matrix multiplication for Float64 and Float32 arrays using OpenBLAS, supporting high-dimensional stack broadcasting and 1D vector promotions.
NDArray matmul(NDArray a, NDArray b) {
  final DType<dynamic> targetDType =
      (a.dtype == DType.float32 || b.dtype == DType.float32)
      ? DType.float32
      : DType.float64;

  if (a.shape.length == 1 && b.shape.length == 1) {
    final n = a.shape[0];
    if (n != b.shape[0]) {
      throw ArgumentError(
        'Incompatible vector dimensions for 1D dot product in matmul: ${a.shape} and ${b.shape}',
      );
    }
    if (targetDType == DType.float64) {
      final scalarRes = cblas_ddot(
        n,
        a.pointer.cast<ffi.Double>(),
        1,
        b.pointer.cast<ffi.Double>(),
        1,
      );
      return NDArray.fromList([scalarRes], [], DType.float64);
    } else {
      final scalarRes = cblas_sdot(
        n,
        a.pointer.cast<ffi.Float>(),
        1,
        b.pointer.cast<ffi.Float>(),
        1,
      );
      return NDArray.fromList([scalarRes], [], DType.float32);
    }
  }

  // Copy upfront ONLY if neither inner strides is 1 (very rare custom sliced strides)
  if (a.shape.length >= 2) {
    final r = a.shape.length;
    if (a.strides[r - 1] != 1 && a.strides[r - 2] != 1) {
      a = a.copy();
    }
  }
  if (b.shape.length >= 2) {
    final r = b.shape.length;
    if (b.strides[r - 1] != 1 && b.strides[r - 2] != 1) {
      b = b.copy();
    }
  }

  var aPromoted = false;
  var bPromoted = false;

  NDArray aView = a;
  if (a.shape.length == 1) {
    aView = NDArray.view(
      a,
      shape: [1, a.shape[0]],
      strides: [0, a.strides[0]],
      offsetElements: 0,
    );
    aPromoted = true;
  }

  NDArray bView = b;
  if (b.shape.length == 1) {
    bView = NDArray.view(
      b,
      shape: [b.shape[0], 1],
      strides: [b.strides[0], 0],
      offsetElements: 0,
    );
    bPromoted = true;
  }

  final rankA = aView.shape.length;
  final rankB = bView.shape.length;

  final m = aView.shape[rankA - 2];
  final kA = aView.shape[rankA - 1];
  final kB = bView.shape[rankB - 2];
  final n = bView.shape[rankB - 1];

  if (kA != kB) {
    throw ArgumentError(
      'Incompatible inner matrix dimensions for matmul: kA($kA) != kB($kB). Shapes: ${a.shape} and ${b.shape}',
    );
  }

  final stackA = aView.shape.sublist(0, rankA - 2);
  final stackB = bView.shape.sublist(0, rankB - 2);
  final broadcastStack = _broadcastStackShapes(stackA, stackB);

  final resShape = [...broadcastStack, m, n];
  final result = NDArray.zeros(resShape, targetDType as dynamic);

  // Stride resolution logic for 100% copy-free BLAS matrix multiplication
  var transA = 111; // CblasNoTrans
  var lda = kA;
  if (!aPromoted) {
    if (aView.strides[rankA - 1] == 1) {
      transA = 111;
      lda = aView.strides[rankA - 2];
    } else if (aView.strides[rankA - 2] == 1) {
      transA = 112; // CblasTrans
      lda = aView.strides[rankA - 1];
    }
  }

  var transB = 111; // CblasNoTrans
  var ldb = n;
  if (!bPromoted) {
    if (bView.strides[rankB - 1] == 1) {
      transB = 111;
      ldb = bView.strides[rankB - 2];
    } else if (bView.strides[rankB - 2] == 1) {
      transB = 112; // CblasTrans
      ldb = bView.strides[rankB - 1];
    }
  }

  final lenA = stackA.length;
  final lenB = stackB.length;
  final lenResult = broadcastStack.length;

  final walkStridesA = List<int>.filled(lenResult, 0);
  final walkStridesB = List<int>.filled(lenResult, 0);

  for (var i = 0; i < lenResult; i++) {
    final resAxis = lenResult - 1 - i;
    final axisA = lenA - 1 - i;
    final axisB = lenB - 1 - i;

    if (axisA >= 0) {
      walkStridesA[resAxis] = (stackA[axisA] == broadcastStack[resAxis])
          ? aView.strides[axisA]
          : 0;
    } else {
      walkStridesA[resAxis] = 0;
    }

    if (axisB >= 0) {
      walkStridesB[resAxis] = (stackB[axisB] == broadcastStack[resAxis])
          ? bView.strides[axisB]
          : 0;
    } else {
      walkStridesB[resAxis] = 0;
    }
  }

  final walkStridesRes = List<int>.filled(lenResult, 0);
  var resStride = m * n;
  for (var i = lenResult - 1; i >= 0; i--) {
    walkStridesRes[i] = resStride;
    resStride *= broadcastStack[i];
  }

  void walk(int dim, int offsetA, int offsetB, int offsetRes) {
    if (dim == lenResult) {
      if (targetDType == DType.float64) {
        cblas_dgemm(
          101, // CblasRowMajor
          transA,
          transB,
          m,
          n,
          kA,
          1.0,
          aView.pointer.cast<ffi.Double>() + offsetA,
          lda,
          bView.pointer.cast<ffi.Double>() + offsetB,
          ldb,
          0.0,
          result.pointer.cast<ffi.Double>() + offsetRes,
          n, // ldc (result is always contiguous row-major)
        );
      } else {
        cblas_sgemm(
          101, // CblasRowMajor
          transA,
          transB,
          m,
          n,
          kA,
          1.0,
          aView.pointer.cast<ffi.Float>() + offsetA,
          lda,
          bView.pointer.cast<ffi.Float>() + offsetB,
          ldb,
          0.0,
          result.pointer.cast<ffi.Float>() + offsetRes,
          n, // ldc (result is always contiguous row-major)
        );
      }
      return;
    }

    final size = broadcastStack[dim];
    final strideA = walkStridesA[dim];
    final strideB = walkStridesB[dim];
    final strideRes = walkStridesRes[dim];

    for (var i = 0; i < size; i++) {
      walk(
        dim + 1,
        offsetA + i * strideA,
        offsetB + i * strideB,
        offsetRes + i * strideRes,
      );
    }
  }

  walk(0, 0, 0, 0);

  // Post-calculation 1D dummy dimensions demotions
  if (aPromoted && bPromoted) {
    return result.reshape([]); // 0D scalar array for pure vector dot products
  } else if (aPromoted) {
    final newShape = List<int>.from(result.shape)
      ..removeAt(result.shape.length - 2);
    return result.reshape(newShape);
  } else if (bPromoted) {
    final newShape = List<int>.from(result.shape)
      ..removeAt(result.shape.length - 1);
    return result.reshape(newShape);
  }

  return result;
}

/// Compute the sum of elements in the array.
///
/// If [axis] is provided, sums along that axis and returns a new array.
/// Otherwise, sums all elements and returns a scalar value of type [T].
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final s0 = sum(a, axis: 0); // Sum along rows
/// print(s0.data); // [4.0, 6.0]
/// ```
NDArray sum(NDArray a, {int? axis, NDArray? out}) {
  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    dynamic acc;
    if (a.isContiguous) {
      if (a.dtype == DType.float64) {
        acc = r_sum_double(a.pointer.cast(), size);
      } else if (a.dtype == DType.float32) {
        acc = r_sum_float(a.pointer.cast(), size);
      }
    }
    if (acc == null) {
      final List<dynamic> elements = size == a.data.length
          ? a.data
          : a.toList();
      acc = elements.first;
      for (var i = 1; i < elements.length; i++) {
        acc += elements[i];
      }
    }
    final result = out ?? NDArray.create([], a.dtype);
    result.data[0] = acc;
    return result;
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final result = out ?? NDArray.zeros(newShape, a.dtype as dynamic);
  if (out != null) {
    result.fill(0);
  }

  _reduceRecursive(
    a,
    result,
    List<int>.filled(a.shape.length, 0),
    List<int>.filled(newShape.length, 0),
    axis,
    0,
    (current, val) => ((current as dynamic) + val) as dynamic,
  );
  return result;
}

/// Compute the product of elements in the array.
///
/// If [axis] is provided, multiplies along that axis and returns a new array.
/// Otherwise, multiplies all elements and returns a scalar value of type [T].
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final p0 = prod(a, axis: 0); // Product along rows
/// print(p0.data); // [3.0, 8.0]
/// ```
NDArray prod(NDArray a, {int? axis, NDArray? out}) {
  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    dynamic acc;
    if (a.isContiguous) {
      if (a.dtype == DType.float64) {
        acc = r_prod_double(a.pointer.cast(), size);
      } else if (a.dtype == DType.float32) {
        acc = r_prod_float(a.pointer.cast(), size);
      }
    }
    if (acc == null) {
      final List<dynamic> elements = size == a.data.length
          ? a.data
          : a.toList();
      acc = elements.first;
      for (var i = 1; i < elements.length; i++) {
        acc *= elements[i];
      }
    }
    final result = out ?? NDArray.create([], a.dtype);
    result.data[0] = acc;
    return result;
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final result = out ?? NDArray.ones(newShape, a.dtype as dynamic);
  if (out != null) {
    result.fill(1);
  }

  _reduceRecursive(
    a,
    result,
    List<int>.filled(a.shape.length, 0),
    List<int>.filled(newShape.length, 0),
    axis,
    0,
    (current, val) => ((current as dynamic) * val) as dynamic,
  );
  return result;
}

/// Returns true if all elements along a given [axis] evaluate to True.
///
/// If [axis] is omitted/null, performs a global reduction and returns a single Dart [bool].
///
/// **Preconditions:**
/// - The array [a] must not be disposed.
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([true, true, false], [3], DType.boolean);
/// final res = all(a); // false
/// ```
NDArray<bool> all(NDArray a, {int? axis, NDArray<bool>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute all() on a disposed array.');
  }

  var targetAxis = axis;
  if (targetAxis != null && targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }

  final targetShape = targetAxis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(targetAxis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.boolean) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (targetAxis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final List elements = size == a.data.length ? a.data : a.toList();
    var allTrue = true;
    for (var i = 0; i < elements.length; i++) {
      if (!_isTrue(elements[i])) {
        allTrue = false;
        break;
      }
    }
    final result = out ?? NDArray<bool>.create([], DType.boolean);
    result.data[0] = allTrue;
    return result;
  }

  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(targetAxis);
  final result = out ?? NDArray<bool>.create(newShape, DType.boolean);
  result.fill(true); // Initialize to true everywhere

  _reduceRecursive(
    a as NDArray<Object>,
    result,
    List<int>.filled(a.shape.length, 0),
    List<int>.filled(newShape.length, 0),
    targetAxis,
    0,
    (current, val) => current && _isTrue(val),
  );

  return result;
}

/// Returns true if any element along a given [axis] evaluates to True.
///
/// If [axis] is omitted/null, performs a global reduction and returns a single Dart [bool].
///
/// **Preconditions:**
/// - The array [a] must not be disposed.
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([true, false, false], [3], DType.boolean);
/// final res = any(a); // true
/// ```
NDArray<bool> any(NDArray a, {int? axis, NDArray<bool>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute any() on a disposed array.');
  }

  var targetAxis = axis;
  if (targetAxis != null && targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }

  final targetShape = targetAxis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(targetAxis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.boolean) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (targetAxis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final List elements = size == a.data.length ? a.data : a.toList();
    var anyTrue = false;
    for (var i = 0; i < elements.length; i++) {
      if (_isTrue(elements[i])) {
        anyTrue = true;
        break;
      }
    }
    final result = out ?? NDArray<bool>.create([], DType.boolean);
    result.data[0] = anyTrue;
    return result;
  }

  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(targetAxis);
  final result =
      out ??
      NDArray<bool>.zeros(newShape, DType.boolean); // Pre-initialized to false
  if (out != null) {
    result.fill(false);
  }

  _reduceRecursive(
    a as NDArray<Object>,
    result,
    List<int>.filled(a.shape.length, 0),
    List<int>.filled(newShape.length, 0),
    targetAxis,
    0,
    (current, val) => current || _isTrue(val),
  );

  return result;
}

/// Compute the element-wise square root of the array.
///
/// Returns a new array with the results.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 4.0, 9.0], [3], DType.float64);
/// final b = sqrt(a);
/// print(b.data); // [1.0, 2.0, 3.0]
/// ```
///
/// **Gotchas:**
/// - Negative values will result in [double.nan].
NDArray sqrt(NDArray a, {NDArray? out}) {
  final DType<dynamic> targetDType = a.dtype == DType.float32
      ? DType.float32
      : DType.float64;
  final result = out ?? NDArray.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for sqrt.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_sqrt_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_sqrt_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  }

  final aNum = a as NDArray<num>;
  for (var i = 0; i < a.data.length; i++) {
    result.data[i] = math.sqrt(aNum.data[i].toDouble());
  }
  return result;
}

/// Compute the element-wise sine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous array layouts, offloads the loop directly to high-speed native C
///   vector math kernels (`v_sin_double`/`v_sin_float`), bypassing all Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/transcendental_example.dart lang=dart}
///
/// Reference: [Trigonometric Sine Function](https://en.wikipedia.org/wiki/Sine_and_cosine)
NDArray sin(NDArray a, {NDArray? out}) {
  final DType<dynamic> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  }
  final result = out ?? NDArray.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for sin.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_sin_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_sin_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.complex128) {
      v_sin_complex128(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.complex64) {
      v_sin_complex64(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  } else {
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
      if (a.dtype == DType.float64) {
        s_sin_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.float32) {
        s_sin_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.complex128) {
        s_sin_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.complex64) {
        s_sin_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesRes);
    }
  }

  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.sin(x.toDouble()),
    );
  } else {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.sin(x.toDouble()),
    );
  }
  return result;
}

/// Compute the element-wise cosine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous array layouts, offloads the loop directly to high-speed native C
///   vector math kernels (`v_cos_double`/`v_cos_float`), bypassing all Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/transcendental_example.dart lang=dart}
///
/// Reference: [Trigonometric Cosine Function](https://en.wikipedia.org/wiki/Sine_and_cosine)
NDArray cos(NDArray a, {NDArray? out}) {
  final DType<dynamic> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  }
  final result = out ?? NDArray.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for cos.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_cos_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_cos_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.complex128) {
      v_cos_complex128(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.complex64) {
      v_cos_complex64(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  } else {
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
      if (a.dtype == DType.float64) {
        s_cos_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.float32) {
        s_cos_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.complex128) {
        s_cos_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.complex64) {
        s_cos_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesRes);
    }
  }

  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.cos(x.toDouble()),
    );
  } else {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.cos(x.toDouble()),
    );
  }
  return result;
}

/// Compute the element-wise exponential of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous array layouts, offloads the loop directly to high-speed native C
///   vector math kernels (`v_exp_double`/`v_exp_float`), bypassing all Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/transcendental_example.dart lang=dart}
///
/// Reference: [Exponential Function](https://en.wikipedia.org/wiki/Exponential_function)
NDArray exp(NDArray a, {NDArray? out}) {
  final DType<dynamic> targetDType = a.dtype == DType.float32
      ? DType.float32
      : DType.float64;
  final result = out ?? NDArray.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for exp.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_exp_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_exp_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  } else {
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
      if (a.dtype == DType.float64) {
        s_exp_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.float32) {
        s_exp_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesRes);
    }
  }

  final aNum = a as NDArray<num>;
  for (var i = 0; i < a.data.length; i++) {
    result.data[i] = math.exp(aNum.data[i].toDouble());
  }
  return result;
}

/// Compute the element-wise natural logarithm of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous array layouts, offloads the loop directly to high-speed native C
///   vector math kernels (`v_log_double`/`v_log_float`), bypassing all Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/transcendental_example.dart lang=dart}
///
/// Reference: [Natural Logarithm](https://en.wikipedia.org/wiki/Natural_logarithm)
NDArray log(NDArray a, {NDArray? out}) {
  final DType<dynamic> targetDType = a.dtype == DType.float32
      ? DType.float32
      : DType.float64;
  final result = out ?? NDArray.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for log.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_log_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_log_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  } else {
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
      if (a.dtype == DType.float64) {
        s_log_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.float32) {
        s_log_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesRes);
    }
  }

  final aNum = a as NDArray<num>;
  for (var i = 0; i < a.data.length; i++) {
    result.data[i] = math.log(aNum.data[i].toDouble());
  }
  return result;
}

/// Compute the arithmetic mean of array elements along a specified axis.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num` or Complex).
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [RangeError] if [axis] is out of range.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final m = mean(a); // returns 2.5 scalar
/// final m0 = mean(a, axis: 0); // returns NDArray [2.0, 3.0]
/// ```
///
/// Reference: [Arithmetic Mean](https://en.wikipedia.org/wiki/Arithmetic_mean)
NDArray mean(NDArray a, {int? axis, NDArray? out}) {
  final DType targetDType = a.dtype.isComplex
      ? DType.complex128
      : DType.float64;

  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != targetDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  NDArray promotedA;
  if (a.dtype == DType.float64 ||
      a.dtype == DType.float32 ||
      a.dtype == DType.complex128 ||
      a.dtype == DType.complex64) {
    promotedA = a;
  } else {
    promotedA = NDArray<double>.create(a.shape, DType.float64);
    final doubleList = promotedA.data as List<double>;
    final aList = a.data;
    for (var i = 0; i < aList.length; i++) {
      doubleList[i] = (aList[i] as num).toDouble();
    }
  }

  if (axis == null) {
    final s = sum(promotedA, axis: axis);
    if (promotedA != a) {
      promotedA.dispose();
    }
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final meanVal = s.data[0] / size;
    final result = out ?? NDArray.create([], targetDType);
    result.data[0] = meanVal;
    s.dispose();
    return result;
  } else {
    final result = out ?? NDArray.create(targetShape, targetDType);
    sum(promotedA, axis: axis, out: result);
    if (promotedA != a) {
      promotedA.dispose();
    }
    final sizeAxis = a.shape[axis];
    for (var i = 0; i < result.data.length; i++) {
      result.data[i] = (result.data[i] / sizeAxis) as dynamic;
    }
    return result;
  }
}

/// Compute the variance of array elements along a specified axis.
///
/// Variance is a measure of the spread of a distribution. The variance is computed for
/// the flattened array by default, otherwise over the specified axis.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [RangeError] if [axis] is out of range.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
/// final v = variance(a); // returns 1.25 scalar
/// ```
///
/// Reference: [Variance](https://en.wikipedia.org/wiki/Variance)
NDArray<double> variance<T extends num>(
  NDArray<T> a, {
  int? axis,
  NDArray<double>? out,
}) {
  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  final m = mean(a, axis: axis);

  if (axis == null) {
    var sumSqDiff = 0.0;
    final meanVal = m.data[0] as double;
    m.dispose();

    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final List<num> elements = size == a.data.length
        ? a.data as List<num>
        : a.toList() as List<num>;

    for (var i = 0; i < elements.length; i++) {
      final diff = elements[i].toDouble() - meanVal;
      sumSqDiff += diff * diff;
    }
    final result = out ?? NDArray<double>.create([], DType.float64);
    result.data[0] = sumSqDiff / elements.length;
    return result;
  } else {
    // Reshape m to keep dimensions for broadcasting
    final targetShape = List<int>.from(a.shape);
    targetShape[axis] = 1;
    final reshapedM = m.reshape(targetShape);

    final diff = subtract(a, reshapedM);
    final sqDiff = multiply(diff, diff);

    final sqDiffDouble = NDArray<double>.create(sqDiff.shape, DType.float64);
    for (var i = 0; i < sqDiff.data.length; i++) {
      sqDiffDouble.data[i] = sqDiff.data[i].toDouble();
    }

    m.dispose();
    reshapedM.dispose();
    diff.dispose();
    sqDiff.dispose();

    final res = mean(sqDiffDouble, axis: axis, out: out);
    if (out != null) {
      sqDiffDouble.dispose();
      return out;
    }
    final resultVal = NDArray<double>.view(
      res,
      shape: res.shape,
      strides: res.strides,
    );
    sqDiffDouble.dispose();
    return resultVal;
  }
}

/// Compute the standard deviation of array elements along a specified axis.
///
/// Standard deviation is a measure of the spread of a distribution. The standard deviation
/// is computed for the flattened array by default, otherwise over the specified axis.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [RangeError] if [axis] is out of range.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
/// final s = std(a); // returns sqrt(1.25) scalar
/// ```
///
/// Reference: [Standard Deviation](https://en.wikipedia.org/wiki/Standard_deviation)
NDArray<double> std<T extends num>(
  NDArray<T> a, {
  int? axis,
  NDArray<double>? out,
}) {
  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  final v = variance(a, axis: axis);
  if (axis == null) {
    final stdVal = math.sqrt(v.data[0]);
    final result = out ?? NDArray<double>.create([], DType.float64);
    result.data[0] = stdVal;
    v.dispose();
    return result;
  } else {
    final res = sqrt(v, out: out);
    if (out != null) {
      v.dispose();
      return out;
    }
    final resultVal = NDArray<double>.view(
      res,
      shape: res.shape,
      strides: res.strides,
    );
    v.dispose();
    return resultVal;
  }
}

/// Compute the sum of array elements along a specified axis, treating NaNs as zeros.
///
/// **Preconditions:**
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total elements count.
///
/// **Example:**
/// ```dart
/// final a = `NDArray<double>`.fromList([1.0, double.nan, 3.0, double.nan], [2, 2], DType.float64);
/// final s = nansum(a); // returns 4.0
/// ```
NDArray nansum(NDArray a, {int? axis, NDArray? out}) {
  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final List<dynamic> elements = size == a.data.length ? a.data : a.toList();
    dynamic acc;
    if (a.dtype == DType.int32 || a.dtype == DType.int64) {
      var sumVal = 0;
      for (var i = 0; i < elements.length; i++) {
        sumVal += elements[i] as int;
      }
      acc = sumVal;
    } else if (a.dtype == DType.complex64 || a.dtype == DType.complex128) {
      var sumVal = Complex(0.0, 0.0);
      for (var i = 0; i < elements.length; i++) {
        final val = elements[i] as Complex;
        if (val.real.isNaN || val.imag.isNaN) continue;
        sumVal += val;
      }
      acc = sumVal;
    } else {
      var sumVal = 0.0;
      for (var i = 0; i < elements.length; i++) {
        final val = elements[i] as double;
        if (val.isNaN) continue;
        sumVal += val;
      }
      acc = sumVal;
    }
    final result = out ?? NDArray.create([], a.dtype);
    result.data[0] = acc;
    return result;
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final result = out ?? NDArray.zeros(newShape, a.dtype as dynamic);
  if (out != null) {
    result.fill(0);
  }

  _reduceRecursive(
    a,
    result,
    List<int>.filled(a.shape.length, 0),
    List<int>.filled(newShape.length, 0),
    axis,
    0,
    (current, val) {
      if (val is double && val.isNaN) return current;
      return ((current as dynamic) + val) as dynamic;
    },
  );
  return result;
}

/// Compute the arithmetic mean along a specified axis, ignoring NaNs.
///
/// **Preconditions:**
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total elements count, walking
///   coordinate strides and tracking counts dynamically.
///
/// **Example:**
/// ```dart
/// final a = `NDArray<double>`.fromList([1.0, double.nan, 3.0, 4.0], [2, 2], DType.float64);
/// final m = nanmean(a); // returns 2.6666666666666665
/// ```
NDArray nanmean(NDArray a, {int? axis, NDArray? out}) {
  final DType targetDType = a.dtype.isComplex
      ? DType.complex128
      : DType.float64;

  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != targetDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  NDArray promotedA;
  if (a.dtype == DType.float64 ||
      a.dtype == DType.float32 ||
      a.dtype == DType.complex128 ||
      a.dtype == DType.complex64) {
    promotedA = a;
  } else {
    promotedA = NDArray<double>.create(a.shape, DType.float64);
    final doubleList = promotedA.data as List<double>;
    final aList = a.data;
    for (var i = 0; i < aList.length; i++) {
      doubleList[i] = (aList[i] as num).toDouble();
    }
  }

  if (axis == null) {
    final size = promotedA.shape.isEmpty
        ? 1
        : promotedA.shape.reduce((x, y) => x * y);
    final List<dynamic> elements = size == promotedA.data.length
        ? promotedA.data
        : promotedA.toList();
    var sumVal = 0.0;
    var count = 0;
    for (var i = 0; i < elements.length; i++) {
      final val = elements[i];
      if (val is double && val.isNaN) continue;
      sumVal += (val as num).toDouble();
      count++;
    }
    if (promotedA != a) {
      promotedA.dispose();
    }
    final result = out ?? NDArray<double>.create([], DType.float64);
    if (count == 0) {
      result.data[0] = double.nan;
    } else {
      result.data[0] = sumVal / count;
    }
    return result;
  }

  if (axis < 0 || axis >= promotedA.shape.length) {
    if (promotedA != a) {
      promotedA.dispose();
    }
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(promotedA.shape)..removeAt(axis);
  final result = out ?? NDArray.zeros(newShape, targetDType as dynamic);
  if (out != null) {
    result.fill(0);
  }
  final counts = NDArray<int>.zeros(newShape, DType.int32);

  _nanReduceRecursive(
    promotedA,
    result,
    counts,
    List<int>.filled(promotedA.shape.length, 0),
    List<int>.filled(newShape.length, 0),
    axis,
    0,
  );

  for (var i = 0; i < result.data.length; i++) {
    final c = counts.data[i];
    if (c == 0) {
      result.data[i] = double.nan as dynamic;
    } else {
      result.data[i] = ((result.data[i] as dynamic) / c) as dynamic;
    }
  }
  counts.dispose();
  if (promotedA != a) {
    promotedA.dispose();
  }
  return result;
}

/// Recursive helper to accumulate sum and count of non-NaN elements along an axis.
void _nanReduceRecursive(
  NDArray a,
  NDArray result,
  NDArray<int> counts,
  List<int> coordA,
  List<int> coordRes,
  int axis,
  int dim,
) {
  if (dim == a.shape.length) {
    final val = a.getCell(coordA);
    if (val is double && val.isNaN) return;
    final current = result.getCell(coordRes);
    result.setCell(coordRes, ((current as dynamic) + val) as dynamic);
    counts.setCell(coordRes, counts.getCell(coordRes) + 1);
    return;
  }
  if (dim == axis) {
    for (var i = 0; i < a.shape[axis]; i++) {
      coordA[dim] = i;
      _nanReduceRecursive(a, result, counts, coordA, coordRes, axis, dim + 1);
    }
  } else {
    final resDim = dim < axis ? dim : dim - 1;
    for (var i = 0; i < a.shape[dim]; i++) {
      coordA[dim] = i;
      coordRes[resDim] = i;
      _nanReduceRecursive(a, result, counts, coordA, coordRes, axis, dim + 1);
    }
  }
}

/// Compute the variance along the specified axis, ignoring NaNs.
///
/// **Preconditions:**
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total elements count.
///
/// **Example:**
/// ```dart
/// final a = `NDArray<double>`.fromList([1.0, double.nan, 2.0, 3.0], [2, 2], DType.float64);
/// final v = nanvar(a); // returns 0.6666666666666666
/// ```
NDArray<double> nanvar<T extends num>(
  NDArray<T> a, {
  int? axis,
  NDArray<double>? out,
}) {
  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  final m = nanmean(a, axis: axis);

  if (axis == null) {
    var sumSqDiff = 0.0;
    final meanVal = m.data[0];
    m.dispose();
    if (meanVal.isNaN) {
      final result = out ?? NDArray<double>.create([], DType.float64);
      result.data[0] = double.nan;
      return result;
    }

    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final List<num> elements = size == a.data.length
        ? a.data as List<num>
        : a.toList() as List<num>;

    var count = 0;
    for (var i = 0; i < elements.length; i++) {
      final val = elements[i].toDouble();
      if (val.isNaN) continue;
      final diff = val - meanVal;
      sumSqDiff += diff * diff;
      count++;
    }
    final result = out ?? NDArray<double>.create([], DType.float64);
    if (count == 0) {
      result.data[0] = double.nan;
    } else {
      result.data[0] = sumSqDiff / count;
    }
    return result;
  } else {
    // Reshape m to keep dimensions for broadcasting
    final targetShape = List<int>.from(a.shape);
    targetShape[axis] = 1;
    final reshapedM = m.reshape(targetShape);

    final diff = subtract(a, reshapedM);
    final sqDiff = multiply(diff, diff);

    // Convert to `NDArray<double>` to avoid truncation in nanmean
    final sqDiffDouble = NDArray<double>.create(sqDiff.shape, DType.float64);
    for (var i = 0; i < sqDiff.data.length; i++) {
      sqDiffDouble.data[i] = sqDiff.data[i].toDouble();
    }

    m.dispose();
    reshapedM.dispose();
    diff.dispose();
    sqDiff.dispose();

    final res = nanmean(sqDiffDouble, axis: axis, out: out);
    if (out != null) {
      sqDiffDouble.dispose();
      return out;
    }
    final resultVal = NDArray<double>.view(
      res,
      shape: res.shape,
      strides: res.strides,
    );
    sqDiffDouble.dispose();
    return resultVal;
  }
}

/// Compute the standard deviation along the specified axis, ignoring NaNs.
///
/// **Preconditions:**
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total elements count.
///
/// **Example:**
/// ```dart
/// final a = `NDArray<double>`.fromList([1.0, double.nan, 2.0, 3.0], [2, 2], DType.float64);
/// final s = nanstd(a); // returns sqrt(0.6666666666666666)
/// ```
NDArray<double> nanstd<T extends num>(
  NDArray<T> a, {
  int? axis,
  NDArray<double>? out,
}) {
  final targetShape = axis == null
      ? <int>[]
      : (List<int>.from(a.shape)..removeAt(axis));
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  final v = nanvar(a, axis: axis);
  if (axis == null) {
    final stdVal = math.sqrt(v.data[0]);
    final result = out ?? NDArray<double>.create([], DType.float64);
    result.data[0] = stdVal;
    v.dispose();
    return result;
  } else {
    final res = sqrt(v, out: out);
    if (out != null) {
      v.dispose();
      return out;
    }
    final resultVal = NDArray<double>.view(
      res,
      shape: res.shape,
      strides: res.strides,
    );
    v.dispose();
    return resultVal;
  }
}

/// Compute the minimum of elements in the array.
///
/// **Gotchas:**
/// - Returns a 0-dimensional [NDArray] if [axis] is null, or a new [NDArray] if [axis] is provided.
/// - Preserves the original data type (DType) of the input array along the reduction axis.
NDArray<T> min<T extends num>(NDArray<T> a, {int? axis}) {
  if (axis == null) {
    final minVal = a.data.reduce((value, element) => math.min(value, element));
    final result = NDArray<T>.create([], a.dtype);
    result.data[0] = minVal;
    return result;
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final selectors = List<Selector>.generate(a.shape.length, (j) {
    if (j == axis) return Index(0);
    return Slice();
  });
  final firstSlice = a.slice(selectors);

  final result = NDArray<T>.fromList(firstSlice.toList(), newShape, a.dtype);

  for (var i = 1; i < a.shape[axis]; i++) {
    final currentSelectors = List<Selector>.generate(a.shape.length, (j) {
      if (j == axis) return Index(i);
      return Slice();
    });
    final currentSlice = a.slice(currentSelectors);
    _elementWiseMin(result, currentSlice);
  }

  return result;
}

/// Compute the minimum of elements along a specified axis, ignoring NaNs.
///
/// This corresponds to NumPy's `nanmin` function.
///
/// **Preconditions:**
/// - [axis], if provided, must be a valid axis index within `[0, rank - 1]`.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds.
/// - [UnsupportedError] if the array contains Complex numbers.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.nan, 3.0], [3], DType.float64);
/// print(nanmin(a)); // 1.0
/// ```
NDArray<T> nanmin<T extends Object>(NDArray<T> a, {int? axis}) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for nanmin');
  }

  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final List<dynamic> elements = size == a.data.length ? a.data : a.toList();

    var minVal = double.infinity;
    var hasValid = false;
    var hasNan = false;

    for (var i = 0; i < elements.length; i++) {
      final val = elements[i];
      if (val is double) {
        if (val.isNaN) {
          hasNan = true;
          continue;
        }
        if (val < minVal) {
          minVal = val;
          hasValid = true;
        }
      } else if (val is num) {
        final dVal = val.toDouble();
        if (dVal < minVal) {
          minVal = dVal;
          hasValid = true;
        }
      }
    }

    final result = NDArray<T>.create([], a.dtype);
    if (!hasValid) {
      result.data[0] = (hasNan ? double.nan : double.infinity) as dynamic;
    } else {
      result.data[0] =
          ((a.dtype == DType.float64 || a.dtype == DType.float32)
                  ? minVal
                  : minVal.toInt())
              as dynamic;
    }
    return result;
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final selectors = List<Selector>.generate(a.shape.length, (j) {
    if (j == axis) return Index(0);
    return Slice();
  });
  final firstSlice = a.slice(selectors);

  final result = NDArray<T>.fromList(firstSlice.toList(), newShape, a.dtype);

  for (var i = 1; i < a.shape[axis]; i++) {
    final currentSelectors = List<Selector>.generate(a.shape.length, (j) {
      if (j == axis) return Index(i);
      return Slice();
    });
    final currentSlice = a.slice(currentSelectors);
    _elementWiseNanMin(result, currentSlice);
  }

  return result;
}

/// Compute the maximum of elements in the array.
///
/// **Gotchas:**
/// - Returns a 0-dimensional [NDArray] if [axis] is null, or a new [NDArray] if [axis] is provided.
/// - Preserves the original data type (DType) of the input array along the reduction axis.
NDArray<T> max<T extends num>(NDArray<T> a, {int? axis}) {
  if (axis == null) {
    final maxVal = a.data.reduce((value, element) => math.max(value, element));
    final result = NDArray<T>.create([], a.dtype);
    result.data[0] = maxVal;
    return result;
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final selectors = List<Selector>.generate(a.shape.length, (j) {
    if (j == axis) return Index(0);
    return Slice();
  });
  final firstSlice = a.slice(selectors);

  final result = NDArray<T>.fromList(firstSlice.toList(), newShape, a.dtype);

  for (var i = 1; i < a.shape[axis]; i++) {
    final currentSelectors = List<Selector>.generate(a.shape.length, (j) {
      if (j == axis) return Index(i);
      return Slice();
    });
    final currentSlice = a.slice(currentSelectors);
    _elementWiseMax(result, currentSlice);
  }

  return result;
}

/// Compute the maximum of elements along a specified axis, ignoring NaNs.
///
/// This corresponds to NumPy's `nanmax` function.
///
/// **Preconditions:**
/// - [axis], if provided, must be a valid axis index within `[0, rank - 1]`.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds.
/// - [UnsupportedError] if the array contains Complex numbers.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.nan, 3.0], [3], DType.float64);
/// print(nanmax(a)); // 3.0
/// ```
NDArray<T> nanmax<T extends Object>(NDArray<T> a, {int? axis}) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for nanmax');
  }

  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final List<dynamic> elements = size == a.data.length ? a.data : a.toList();

    var maxVal = -double.infinity;
    var hasValid = false;
    var hasNan = false;

    for (var i = 0; i < elements.length; i++) {
      final val = elements[i];
      if (val is double) {
        if (val.isNaN) {
          hasNan = true;
          continue;
        }
        if (val > maxVal) {
          maxVal = val;
          hasValid = true;
        }
      } else if (val is num) {
        final dVal = val.toDouble();
        if (dVal > maxVal) {
          maxVal = dVal;
          hasValid = true;
        }
      }
    }

    final result = NDArray<T>.create([], a.dtype);
    if (!hasValid) {
      result.data[0] = (hasNan ? double.nan : -double.infinity) as dynamic;
    } else {
      result.data[0] =
          ((a.dtype == DType.float64 || a.dtype == DType.float32)
                  ? maxVal
                  : maxVal.toInt())
              as dynamic;
    }
    return result;
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final selectors = List<Selector>.generate(a.shape.length, (j) {
    if (j == axis) return Index(0);
    return Slice();
  });
  final firstSlice = a.slice(selectors);

  final result = NDArray<T>.fromList(firstSlice.toList(), newShape, a.dtype);

  for (var i = 1; i < a.shape[axis]; i++) {
    final currentSelectors = List<Selector>.generate(a.shape.length, (j) {
      if (j == axis) return Index(i);
      return Slice();
    });
    final currentSlice = a.slice(currentSelectors);
    _elementWiseNanMax(result, currentSlice);
  }

  return result;
}

void _elementWiseMin(NDArray dest, NDArray src) {
  _elementWiseMinRec(dest, src, dest.strides, src.strides, 0, 0, 0);
}

void _elementWiseMinRec(
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
    _elementWiseMinRec(
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

void _elementWiseMax(NDArray dest, NDArray src) {
  _elementWiseMaxRec(dest, src, dest.strides, src.strides, 0, 0, 0);
}

void _elementWiseMaxRec(
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
    _elementWiseMaxRec(
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

void _elementWiseNanMin(NDArray dest, NDArray src) {
  _elementWiseNanMinRec(dest, src, dest.strides, src.strides, 0, 0, 0);
}

void _elementWiseNanMinRec(
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
    _elementWiseNanMinRec(
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

void _elementWiseNanMax(NDArray dest, NDArray src) {
  _elementWiseNanMaxRec(dest, src, dest.strides, src.strides, 0, 0, 0);
}

void _elementWiseNanMaxRec(
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
    _elementWiseNanMaxRec(
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

/// Compute the multiplicative inverse of a square 2D matrix.
///
/// Natively offloads computation to high-speed OpenBLAS LAPACK LU decomposition routines
/// (`LAPACKE_dgetrf`/`LAPACKE_dgetri` for Float64, and `LAPACKE_sgetrf`/`LAPACKE_sgetri` for Float32),
/// yielding maximum sequential execution throughput.
///
/// **Preconditions:**
/// - Input array [a] must be a square 2D matrix (`shape.length == 2` and `shape[0] == shape[1]`).
/// - If provided, the [out] recycler array must exactly match the shape and target float dtype of [a].
/// - The matrix must be non-singular (invertible).
///
/// **Throws:**
/// - [ArgumentError] if [a] is not square or not 2D.
/// - [ArgumentError] if [out] is provided but has incompatible dimensions or dtype.
/// - [ArgumentError] if the matrix is singular (non-invertible) during LU pivoting.
/// - [StateError] if FFI memory allocations fail.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N^3)$ where $N$ is the matrix dimension length.
/// - For non-contiguous views, automatically flattens the matrix first, recycling allocation views
///   where safe to minimize heap churn.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([4.0, 7.0, 2.0, 6.0], [2, 2], DType.float64);
/// final b = inv(a);
/// print(b.toList()); // [0.6, -0.7, -0.2, 0.4]
/// ```
///
/// Reference: [Matrix Inversion](https://en.wikipedia.org/wiki/Invertible_matrix)
NDArray inv(NDArray a, {NDArray? out}) {
  if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
    throw ArgumentError('Matrix must be square and 2D (was ${a.shape})');
  }
  final n = a.shape[0];
  final DType<dynamic> targetDType = a.dtype == DType.float32
      ? DType.float32
      : DType.float64;

  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for matrix inversion.',
      );
    }
  }

  final NDArray src;
  final bool wasCopied;
  if (!a.isContiguous) {
    src = NDArray.fromList(a.toList(), a.shape, a.dtype);
    wasCopied = true;
  } else {
    src = a;
    wasCopied = false;
  }

  final ipiv = malloc<ffi.Int>(n);

  try {
    if (targetDType == DType.float32) {
      final NDArray result;
      if (out != null) {
        result = out;
        if (src.dtype == DType.float32) {
          (result.data as Float32List).setRange(
            0,
            src.data.length,
            src.data as Float32List,
          );
        } else {
          for (var i = 0; i < src.data.length; i++) {
            result.data[i] = (src.data[i] as num).toDouble();
          }
        }
      } else if (wasCopied) {
        result = src;
      } else {
        result = NDArray.create(src.shape, DType.float32);
        if (src.dtype == DType.float32) {
          (result.data as Float32List).setRange(
            0,
            src.data.length,
            src.data as Float32List,
          );
        } else {
          for (var i = 0; i < src.data.length; i++) {
            result.data[i] = (src.data[i] as num).toDouble();
          }
        }
      }

      final info = LAPACKE_sgetrf(
        101,
        n,
        n,
        result.pointer.cast<ffi.Float>(),
        n,
        ipiv,
      );
      if (info < 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_sgetrf: $info');
      }
      if (info > 0) {
        throw ArgumentError('Matrix is singular and cannot be inverted');
      }

      final infoTri = LAPACKE_sgetri(
        101,
        n,
        result.pointer.cast<ffi.Float>(),
        n,
        ipiv,
      );
      if (infoTri < 0) {
        throw ArgumentError(
          'Illegal value in call to LAPACKE_sgetri: $infoTri',
        );
      }
      return result;
    } else {
      final NDArray result;
      if (out != null) {
        result = out;
        if (src.dtype == DType.float64) {
          (result.data as Float64List).setRange(
            0,
            src.data.length,
            src.data as Float64List,
          );
        } else {
          for (var i = 0; i < src.data.length; i++) {
            result.data[i] = (src.data[i] as num).toDouble();
          }
        }
      } else if (wasCopied) {
        result = src;
      } else {
        result = NDArray.create(src.shape, DType.float64);
        if (src.dtype == DType.float64) {
          (result.data as Float64List).setRange(
            0,
            src.data.length,
            src.data as Float64List,
          );
        } else {
          for (var i = 0; i < src.data.length; i++) {
            result.data[i] = (src.data[i] as num).toDouble();
          }
        }
      }

      final info = LAPACKE_dgetrf(
        101,
        n,
        n,
        result.pointer.cast<ffi.Double>(),
        n,
        ipiv,
      );
      if (info < 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_dgetrf: $info');
      }
      if (info > 0) {
        throw ArgumentError('Matrix is singular and cannot be inverted');
      }

      final infoTri = LAPACKE_dgetri(
        101,
        n,
        result.pointer.cast<ffi.Double>(),
        n,
        ipiv,
      );
      if (infoTri < 0) {
        throw ArgumentError(
          'Illegal value in call to LAPACKE_dgetri: $infoTri',
        );
      }
      return result;
    }
  } finally {
    malloc.free(ipiv);
    if (wasCopied && out != null) {
      src.dispose();
    }
  }
}

/// Compute the determinant of a square 2D matrix using OpenBLAS.
///
/// Transforms the matrix and calculates its determinant natively via LAPACK LU decomposition.
/// Returns the determinant as a double.
///
/// **Preconditions:**
/// - Matrix [a] must be square (size $N \times N$) and 2-dimensional.
/// - Data type [a.dtype] must be float32 or float64.
///
/// **Throws:**
/// - [ArgumentError] if [a] is not square or not 2D.
/// - [ArgumentError] if [a.dtype] is not a supported floating point data type.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N^3)$ leveraging optimized native LAPACK linear algebra solvers.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final d = det(a);
/// print(d); // -2.0
/// ```
///
/// Refer to the [Determinant Reference](https://en.wikipedia.org/wiki/Determinant)
/// and [LAPACK LU solver](https://en.wikipedia.org/wiki/LU_decomposition) for additional details.
double det(NDArray a) {
  if (a.dtype != DType.float64 && a.dtype != DType.float32) {
    throw ArgumentError('det only supports Float64 and Float32 dtypes');
  }
  if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
    throw ArgumentError('Matrix must be square and 2D (was ${a.shape})');
  }
  final n = a.shape[0];

  if (a.dtype == DType.float64) {
    // Create a copy of the matrix because dgetrf overwrites it
    final aCopy = NDArray<double>.create(a.shape, DType.float64);
    if (a.isContiguous) {
      (aCopy.data as Float64List).setRange(
        0,
        a.data.length,
        a.data as Float64List,
      );
    } else {
      aCopy.data.setRange(
        0,
        a.data.length,
        a.toList().cast<double>() as dynamic,
      );
    }

    final ipiv = malloc<ffi.Int>(n);

    try {
      final info = LAPACKE_dgetrf(
        101, // LAPACK_ROW_MAJOR
        n,
        n,
        aCopy.pointer.cast<ffi.Double>(),
        n,
        ipiv,
      );

      if (info < 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_dgetrf: $info');
      }

      // If info > 0, U(i,i) is exactly zero. The factorization has been completed,
      // but the factor U is exactly singular.
      // In this case, the determinant is 0.
      if (info > 0) {
        return 0.0;
      }

      var detValue = 1.0;
      var swaps = 0;

      final aCopyData = aCopy.data;
      for (var i = 0; i < n; i++) {
        detValue *= aCopyData[i * n + i];
        if (ipiv[i] != i + 1) {
          swaps++;
        }
      }

      if (swaps % 2 != 0) {
        detValue = -detValue;
      }

      return detValue;
    } finally {
      malloc.free(ipiv);
      aCopy.dispose();
    }
  } else {
    // Create a copy of the matrix because sgetrf overwrites it
    final aCopy = NDArray<double>.create(a.shape, DType.float32);
    if (a.isContiguous) {
      (aCopy.data as Float32List).setRange(
        0,
        a.data.length,
        a.data as Float32List,
      );
    } else {
      aCopy.data.setRange(
        0,
        a.data.length,
        a.toList().cast<double>() as dynamic,
      );
    }

    final ipiv = malloc<ffi.Int>(n);

    try {
      final info = LAPACKE_sgetrf(
        101, // LAPACK_ROW_MAJOR
        n,
        n,
        aCopy.pointer.cast<ffi.Float>(),
        n,
        ipiv,
      );

      if (info < 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_sgetrf: $info');
      }

      if (info > 0) {
        return 0.0;
      }

      var detValue = 1.0;
      var swaps = 0;

      final aCopyData = aCopy.data;
      for (var i = 0; i < n; i++) {
        detValue *= aCopyData[i * n + i];
        if (ipiv[i] != i + 1) {
          swaps++;
        }
      }

      if (swaps % 2 != 0) {
        detValue = -detValue;
      }

      return detValue;
    } finally {
      malloc.free(ipiv);
      aCopy.dispose();
    }
  }
}

/// Solve a linear matrix equation, or system of linear scalar equations.
///
/// Computes the "exact" solution, `x`, of the linear equation `a * x = b`.
/// Natively offloads to LAPACK solvers (`dgesv`, `sgesv`, `zgesv`, `cgesv`) depending on precision.
///
/// **Preconditions:**
/// - Matrix [a] must be square (size $N \times N$) and 2-dimensional.
/// - Array [b] first dimension must exactly equal the first dimension of [a] ($N$).
/// - Matrix [a] must be non-singular (invertible).
///
/// **Throws:**
/// - [ArgumentError] if [a] is not square or not 2D.
/// - [ArgumentError] if [b] dimensions do not match [a].
/// - [ArgumentError] if [a] is singular and cannot be solved.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N^3)$ executed in native C-compiled space.
///
/// **Example:**
/// ```dart
/// final a = `NDArray<double>`.fromList([3.0, 1.0, 1.0, 2.0], [2, 2], DType.float64);
/// final b = `NDArray<double>`.fromList([9.0, 8.0], [2], DType.float64);
/// final x = solve(a, b);
/// print(x.toList()); // [2.0, 3.0]
/// ```
NDArray solve(NDArray a, NDArray b) {
  if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
    throw ArgumentError('Matrix a must be square and 2D (was ${a.shape})');
  }
  final n = a.shape[0];

  if (b.shape.isEmpty || b.shape[0] != n) {
    throw ArgumentError(
      'Dimensions of b must match dimensions of a (expected first dimension $n, was ${b.shape.isEmpty ? 0 : b.shape[0]})',
    );
  }

  final nrhs = b.shape.length > 1 ? b.shape[1] : 1;
  final ipiv = malloc<ffi.Int>(n);

  try {
    if (a.dtype == DType.float64 && b.dtype == DType.float64) {
      final aCopy = NDArray<double>.create(a.shape, DType.float64);
      if (a.isContiguous) {
        (aCopy.data as Float64List).setRange(
          0,
          a.data.length,
          a.data as Float64List,
        );
      } else {
        aCopy.data.setRange(0, a.data.length, a.toList() as List<double>);
      }

      final bCopy = NDArray<double>.create(b.shape, DType.float64);
      if (b.isContiguous) {
        (bCopy.data as Float64List).setRange(
          0,
          b.data.length,
          b.data as Float64List,
        );
      } else {
        bCopy.data.setRange(0, b.data.length, b.toList() as List<double>);
      }

      final info = LAPACKE_dgesv(
        101,
        n,
        nrhs,
        aCopy.pointer.cast<ffi.Double>(),
        n,
        ipiv,
        bCopy.pointer.cast<ffi.Double>(),
        nrhs,
      );
      if (info < 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_dgesv: $info');
      }
      if (info > 0) {
        throw ArgumentError('Matrix is singular and cannot be solved');
      }
      aCopy.dispose();
      return bCopy;
    } else if (a.dtype == DType.float32 && b.dtype == DType.float32) {
      final aCopy = NDArray<double>.create(a.shape, DType.float32);
      if (a.isContiguous) {
        (aCopy.data as Float32List).setRange(
          0,
          a.data.length,
          a.data as Float32List,
        );
      } else {
        aCopy.data.setRange(0, a.data.length, a.toList() as List<double>);
      }

      final bCopy = NDArray<double>.create(b.shape, DType.float32);
      if (b.isContiguous) {
        (bCopy.data as Float32List).setRange(
          0,
          b.data.length,
          b.data as Float32List,
        );
      } else {
        bCopy.data.setRange(0, b.data.length, b.toList() as List<double>);
      }

      final info = LAPACKE_sgesv(
        101,
        n,
        nrhs,
        aCopy.pointer.cast<ffi.Float>(),
        n,
        ipiv,
        bCopy.pointer.cast<ffi.Float>(),
        nrhs,
      );
      if (info < 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_sgesv: $info');
      }
      if (info > 0) {
        throw ArgumentError('Matrix is singular and cannot be solved');
      }
      aCopy.dispose();
      return bCopy;
    } else if (a.dtype == DType.complex128 && b.dtype == DType.complex128) {
      final aList = (a.data as ComplexList).backingList;
      final bList = (b.data as ComplexList).backingList;

      final aCopy = NDArray<Complex>.create(a.shape, DType.complex128);
      (aCopy.data as ComplexList).backingList.setRange(0, aList.length, aList);

      final bCopy = NDArray<Complex>.create(b.shape, DType.complex128);
      (bCopy.data as ComplexList).backingList.setRange(0, bList.length, bList);

      final info = LAPACKE_zgesv(
        101,
        n,
        nrhs,
        aCopy.pointer.cast<ffi.Double>(),
        n,
        ipiv,
        bCopy.pointer.cast<ffi.Double>(),
        nrhs,
      );
      if (info < 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_zgesv: $info');
      }
      if (info > 0) {
        throw ArgumentError('Matrix is singular and cannot be solved');
      }
      aCopy.dispose();
      return bCopy;
    } else if (a.dtype == DType.complex64 && b.dtype == DType.complex64) {
      final aList = (a.data as ComplexList).backingList;
      final bList = (b.data as ComplexList).backingList;

      final aCopy = NDArray<Complex>.create(a.shape, DType.complex64);
      (aCopy.data as ComplexList).backingList.setRange(0, aList.length, aList);

      final bCopy = NDArray<Complex>.create(b.shape, DType.complex64);
      (bCopy.data as ComplexList).backingList.setRange(0, bList.length, bList);

      final info = LAPACKE_cgesv(
        101,
        n,
        nrhs,
        aCopy.pointer.cast<ffi.Float>(),
        n,
        ipiv,
        bCopy.pointer.cast<ffi.Float>(),
        nrhs,
      );
      if (info < 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_cgesv: $info');
      }
      if (info > 0) {
        throw ArgumentError('Matrix is singular and cannot be solved');
      }
      aCopy.dispose();
      return bCopy;
    } else {
      // Fallback: convert to Float64
      final aDouble = NDArray<double>.create(a.shape, DType.float64);
      for (var i = 0; i < a.data.length; i++) {
        aDouble.data[i] = (a.data[i] as num).toDouble();
      }

      final bDouble = NDArray<double>.create(b.shape, DType.float64);
      for (var i = 0; i < b.data.length; i++) {
        bDouble.data[i] = (b.data[i] as num).toDouble();
      }

      final info = LAPACKE_dgesv(
        101,
        n,
        nrhs,
        aDouble.pointer.cast<ffi.Double>(),
        n,
        ipiv,
        bDouble.pointer.cast<ffi.Double>(),
        nrhs,
      );
      if (info < 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_dgesv: $info');
      }
      if (info > 0) {
        throw ArgumentError('Matrix is singular and cannot be solved');
      }

      aDouble.dispose();
      return bDouble;
    }
  } finally {
    malloc.free(ipiv);
  }
}

/// Compute the eigenvalues and right eigenvectors of a square array.
///
/// Returns a Map with keys 'eigenvalues' and 'eigenvectors'.
/// Both are returned as `NDArray<Complex>` because they can be complex
/// even for real matrices.
Map<String, NDArray<Complex>> eig(NDArray a) {
  if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
    throw ArgumentError('Matrix must be square and 2D (was ${a.shape})');
  }
  final n = a.shape[0];

  final jobvl = 'N'.codeUnitAt(0);
  final jobvr = 'V'.codeUnitAt(0);

  if (a.dtype == DType.complex128) {
    final aComplex = NDArray<Complex>.create(a.shape, DType.complex128);
    final backing = (a.data as ComplexList).backingList;
    (aComplex.data as ComplexList).backingList.setRange(
      0,
      backing.length,
      backing,
    );

    final w = NDArray<Complex>.create([n], DType.complex128);
    final vr = NDArray<Complex>.create([n, n], DType.complex128);

    final info = LAPACKE_zgeev(
      101, // LAPACK_ROW_MAJOR
      jobvl,
      jobvr,
      n,
      aComplex.pointer.cast<ffi.Double>(),
      n,
      w.pointer.cast<ffi.Double>(),
      ffi.nullptr.cast<ffi.Double>(),
      n,
      vr.pointer.cast<ffi.Double>(),
      n,
    );

    if (info < 0) {
      throw ArgumentError('Illegal value in call to LAPACKE_zgeev: $info');
    }
    if (info > 0) {
      throw ArgumentError(
        'The QR algorithm failed to compute all eigenvalues.',
      );
    }

    aComplex.dispose();
    return {'eigenvalues': w, 'eigenvectors': vr};
  } else if (a.dtype == DType.complex64) {
    final aComplex = NDArray<Complex>.create(a.shape, DType.complex64);
    final backing = (a.data as ComplexList).backingList;
    (aComplex.data as ComplexList).backingList.setRange(
      0,
      backing.length,
      backing,
    );

    final w = NDArray<Complex>.create([n], DType.complex64);
    final vr = NDArray<Complex>.create([n, n], DType.complex64);

    final info = LAPACKE_cgeev(
      101, // LAPACK_ROW_MAJOR
      jobvl,
      jobvr,
      n,
      aComplex.pointer.cast<ffi.Float>(),
      n,
      w.pointer.cast<ffi.Float>(),
      ffi.nullptr.cast<ffi.Float>(),
      n,
      vr.pointer.cast<ffi.Float>(),
      n,
    );

    if (info < 0) {
      throw ArgumentError('Illegal value in call to LAPACKE_cgeev: $info');
    }
    if (info > 0) {
      throw ArgumentError(
        'The QR algorithm failed to compute all eigenvalues.',
      );
    }

    aComplex.dispose();
    return {'eigenvalues': w, 'eigenvectors': vr};
  } else if (a.dtype == DType.float64 ||
      a.dtype == DType.int32 ||
      a.dtype == DType.int64) {
    final aReal = NDArray<double>.create(a.shape, DType.float64);
    for (var i = 0; i < a.data.length; i++) {
      aReal.data[i] = (a.data[i] as num).toDouble();
    }

    final wr = NDArray<double>.zeros([n], DType.float64);
    final wi = NDArray<double>.zeros([n], DType.float64);
    final vrReal = NDArray<double>.create([n, n], DType.float64);

    final info = LAPACKE_dgeev(
      101,
      jobvl,
      jobvr,
      n,
      aReal.pointer.cast<ffi.Double>(),
      n,
      wr.pointer.cast<ffi.Double>(),
      wi.pointer.cast<ffi.Double>(),
      ffi.nullptr.cast<ffi.Double>(),
      n,
      vrReal.pointer.cast<ffi.Double>(),
      n,
    );

    if (info < 0) {
      throw ArgumentError('Illegal value in call to LAPACKE_dgeev: $info');
    }
    if (info > 0) {
      throw ArgumentError(
        'The QR algorithm failed to compute all eigenvalues.',
      );
    }

    final w = NDArray<Complex>.create([n], DType.complex128);
    final vr = NDArray<Complex>.create([n, n], DType.complex128);

    for (var j = 0; j < n; j++) {
      w.data[j] = Complex(wr.data[j], wi.data[j]);
    }

    var j = 0;
    while (j < n) {
      if (wi.data[j] == 0.0) {
        for (var r = 0; r < n; r++) {
          vr.data[r * n + j] = Complex(vrReal.data[r * n + j], 0.0);
        }
        j++;
      } else {
        for (var r = 0; r < n; r++) {
          final realPart = vrReal.data[r * n + j];
          final imagPart = vrReal.data[r * n + j + 1];
          vr.data[r * n + j] = Complex(realPart, imagPart);
          vr.data[r * n + j + 1] = Complex(realPart, -imagPart);
        }
        j += 2;
      }
    }

    aReal.dispose();
    wr.dispose();
    wi.dispose();
    vrReal.dispose();
    return {'eigenvalues': w, 'eigenvectors': vr};
  } else if (a.dtype == DType.float32) {
    final aReal = NDArray<double>.create(a.shape, DType.float32);
    for (var i = 0; i < a.data.length; i++) {
      aReal.data[i] = (a.data[i] as num).toDouble();
    }

    final wr = NDArray<double>.zeros([n], DType.float32);
    final wi = NDArray<double>.zeros([n], DType.float32);
    final vrReal = NDArray<double>.create([n, n], DType.float32);

    final info = LAPACKE_sgeev(
      101,
      jobvl,
      jobvr,
      n,
      aReal.pointer.cast<ffi.Float>(),
      n,
      wr.pointer.cast<ffi.Float>(),
      wi.pointer.cast<ffi.Float>(),
      ffi.nullptr.cast<ffi.Float>(),
      n,
      vrReal.pointer.cast<ffi.Float>(),
      n,
    );

    if (info < 0) {
      throw ArgumentError('Illegal value in call to LAPACKE_sgeev: $info');
    }
    if (info > 0) {
      throw ArgumentError(
        'The QR algorithm failed to compute all eigenvalues.',
      );
    }

    final w = NDArray<Complex>.create([n], DType.complex64);
    final vr = NDArray<Complex>.create([n, n], DType.complex64);

    for (var j = 0; j < n; j++) {
      w.data[j] = Complex(wr.data[j], wi.data[j]);
    }

    var j = 0;
    while (j < n) {
      if (wi.data[j] == 0.0) {
        for (var r = 0; r < n; r++) {
          vr.data[r * n + j] = Complex(vrReal.data[r * n + j], 0.0);
        }
        j++;
      } else {
        for (var r = 0; r < n; r++) {
          final realPart = vrReal.data[r * n + j];
          final imagPart = vrReal.data[r * n + j + 1];
          vr.data[r * n + j] = Complex(realPart, imagPart);
          vr.data[r * n + j + 1] = Complex(realPart, -imagPart);
        }
        j += 2;
      }
    }

    aReal.dispose();
    wr.dispose();
    wi.dispose();
    vrReal.dispose();
    return {'eigenvalues': w, 'eigenvectors': vr};
  } else {
    throw UnimplementedError('Type ${a.dtype} not supported for eig');
  }
}

/// Recursive helper to traverse and reduce an array along an axis.
void _reduceRecursive(
  NDArray src,
  NDArray dest,
  List<int> currentPos,
  List<int> destPos,
  int targetAxis,
  int currentDim,
  Function op,
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
    _reduceRecursive(
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

/// Concatenates a list of arrays along a specified axis.
NDArray<T> concatenate<T extends Object>(
  List<NDArray<T>> arrays, {
  int axis = 0,
}) {
  if (arrays.isEmpty) {
    throw ArgumentError('List of arrays must not be empty');
  }

  final first = arrays.first;
  final rank = first.shape.length;
  final dtype = first.dtype;

  if (axis < 0 || axis >= rank) {
    throw RangeError.index(axis, first.shape, 'axis out of range');
  }

  for (var i = 1; i < arrays.length; i++) {
    final arr = arrays[i];
    if (arr.dtype != dtype) {
      throw ArgumentError('All arrays must have the same DType');
    }
    if (arr.shape.length != rank) {
      throw ArgumentError('All arrays must have the same rank');
    }
    for (var j = 0; j < rank; j++) {
      if (j != axis && arr.shape[j] != first.shape[j]) {
        throw ArgumentError('Shapes must match except in dimension $axis');
      }
    }
  }

  final targetShape = List<int>.from(first.shape);
  var totalAxisSize = 0;
  for (final arr in arrays) {
    totalAxisSize += arr.shape[axis];
  }
  targetShape[axis] = totalAxisSize;

  final result = NDArray<T>.create(targetShape, dtype);

  var allContiguous = true;
  for (final arr in arrays) {
    if (!arr.isContiguous) {
      allContiguous = false;
      break;
    }
  }

  if (allContiguous && axis == 0) {
    var destOffset = 0;
    for (final arr in arrays) {
      final size = arr.shape.isEmpty ? 1 : arr.shape.reduce((a, b) => a * b);
      _copyContiguousFlat(arr, result, destOffset, size);
      destOffset += size;
    }
    return result;
  }

  var axisOffset = 0;
  for (final arr in arrays) {
    _copyConcatenateRecursive(
      arr,
      result,
      axis,
      axisOffset,
      List<int>.filled(rank, 0),
      0,
    );
    axisOffset += arr.shape[axis];
  }

  return result;
}

void _copyContiguousFlat(NDArray src, NDArray dest, int destOffset, int size) {
  final width = src.dtype.byteWidth;
  final destPtr = dest.pointer.cast<ffi.Uint8>() + destOffset * width;
  final srcPtr = src.pointer.cast<ffi.Uint8>();
  custom_memcpy(destPtr.cast(), srcPtr.cast(), size * width);
}

void _copyConcatenateRecursive<T extends Object>(
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
    _copyConcatenateRecursive(
      src,
      dest,
      axis,
      axisOffset,
      currentIndices,
      currentDim + 1,
    );
  }
}

/// Stacks arrays in sequence vertically (row wise).
NDArray<T> vstack<T extends Object>(List<NDArray<T>> arrays) {
  return concatenate(arrays, axis: 0);
}

/// Stacks arrays in sequence horizontally (column wise).
NDArray<T> hstack<T extends Object>(List<NDArray<T>> arrays) {
  return concatenate(arrays, axis: 1);
}

/// Return an array copy of the given object.
///
/// This function corresponds to NumPy's `copy` function. It returns a deep copy
/// of [a] that respects shape, strides, and `DType.`
///
/// **Throws:**
/// - [StateError] if the array [a] is already disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1, 2], [2], DType.int32);
/// final b = copy(a);
/// b.data[0] = 99;
/// print(a.data[0]); // 1 (decoupled memory!)
/// ```
NDArray<T> copy<T extends Object>(NDArray<T> a) {
  return a.copy();
}

/// Join a sequence of arrays along a new axis.
///
/// Stacks the input [arrays] along a new dimension at [axis]. All arrays in the
/// list must have the exact same shape and `DType.`
///
/// **Preconditions:**
/// - Input list [arrays] must be non-empty.
/// - All arrays in [arrays] must not be disposed.
/// - All arrays in [arrays] must share identical shapes and DTypes.
///
/// **Throws:**
/// - [ArgumentError] if [arrays] is empty, or shape/DType mismatches occur.
/// - [RangeError] if [axis] is out of bounds.
/// - [StateError] if any array is disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1, 2], [2], DType.int32);
/// final b = NDArray.fromList([3, 4], [2], DType.int32);
/// final s = stack([a, b], axis: 0); // shape [2, 2], values [[1, 2], [3, 4]]
/// ```
NDArray stack(List<NDArray> arrays, {int axis = 0}) {
  if (arrays.isEmpty) {
    throw ArgumentError('List of arrays to stack must not be empty.');
  }

  for (final arr in arrays) {
    if (arr.isDisposed) {
      throw StateError('Cannot execute stack() on a disposed array.');
    }
  }

  final first = arrays.first;
  final rank = first.shape.length;
  final dtype = first.dtype;

  // Validate identical shapes and DTypes
  for (var i = 1; i < arrays.length; i++) {
    final arr = arrays[i];
    if (arr.dtype != dtype) {
      throw ArgumentError('All arrays in stack must have identical DTypes.');
    }
    if (!listEquals(arr.shape, first.shape)) {
      throw ArgumentError('All arrays in stack must have identical shapes.');
    }
  }

  // Target axis can range from -(rank + 1) to rank
  final targetAxis = axis < 0 ? rank + 1 + axis : axis;
  if (targetAxis < 0 || targetAxis > rank) {
    throw RangeError.range(targetAxis, 0, rank, 'axis');
  }

  // Build stacked output shape by inserting arrays.length at targetAxis index
  final stackedShape = List<int>.from(first.shape);
  stackedShape.insert(targetAxis, arrays.length);

  final result = NDArray.zeros(stackedShape, dtype);

  for (var i = 0; i < arrays.length; i++) {
    final currentIndices = List<int>.filled(first.shape.length, 0);
    _copyStackRecursive(arrays[i], result, targetAxis, i, currentIndices, 0);
  }

  return result;
}

void _copyStackRecursive(
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
    _copyStackRecursive(
      src,
      dest,
      targetAxis,
      axisOffset,
      currentIndices,
      currentDim + 1,
    );
  }
}

void _unaryOp<Ta, Tr>(
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
    _unaryOp<Ta, Tr>(
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

/// Compute the element-wise tangent of the array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray tan(NDArray a, {NDArray? out}) {
  final DType<dynamic> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  }
  final result = out ?? NDArray.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for tan.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_tan_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_tan_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.complex128) {
      v_tan_complex128(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.complex64) {
      v_tan_complex64(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  } else {
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
      if (a.dtype == DType.float64) {
        s_tan_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.float32) {
        s_tan_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.complex128) {
        s_tan_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.complex64) {
        s_tan_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesRes);
    }
  }
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.tan(x.toDouble()),
    );
  } else {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.tan(x),
    );
  }
  return result;
}

/// Compute the element-wise arc sine (inverse sine) of the array.
///
/// **Preconditions:**
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([0.0, 1.0], [2], DType.float64);
/// final b = asin(a); // [0.0, 1.570796...]
/// ```
NDArray asin(NDArray a, {NDArray? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute asin() on a disposed array.');
  }
  final DType<dynamic> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  }
  final result = out ?? NDArray.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for asin.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_asin_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_asin_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.complex128) {
      v_asin_complex128(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.complex64) {
      v_asin_complex64(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  } else {
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
      if (a.dtype == DType.float64) {
        s_asin_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.float32) {
        s_asin_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.complex128) {
        s_asin_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.complex64) {
        s_asin_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesRes);
    }
  }

  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.asin(x.toDouble()),
    );
  } else {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.asin(x),
    );
  }
  return result;
}

/// Compute the element-wise arc cosine (inverse cosine) of the array.
///
/// **Preconditions:**
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 0.0], [2], DType.float64);
/// final b = acos(a); // [0.0, 1.570796...]
/// ```
NDArray acos(NDArray a, {NDArray? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute acos() on a disposed array.');
  }
  final DType<dynamic> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  }
  final result = out ?? NDArray.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for acos.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_acos_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_acos_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.complex128) {
      v_acos_complex128(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.complex64) {
      v_acos_complex64(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  } else {
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
      if (a.dtype == DType.float64) {
        s_acos_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.float32) {
        s_acos_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.complex128) {
        s_acos_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.complex64) {
        s_acos_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesRes);
    }
  }

  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.acos(x.toDouble()),
    );
  } else {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.acos(x),
    );
  }
  return result;
}

/// Compute the element-wise arc tangent (inverse tangent) of the array.
///
/// **Preconditions:**
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([0.0, 1.0], [2], DType.float64);
/// final b = atan(a); // [0.0, 0.785398...]
/// ```
NDArray atan(NDArray a, {NDArray? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute atan() on a disposed array.');
  }
  final DType<dynamic> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  }
  final result = out ?? NDArray.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for atan.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_atan_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_atan_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.complex128) {
      v_atan_complex128(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.complex64) {
      v_atan_complex64(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  } else {
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
      if (a.dtype == DType.float64) {
        s_atan_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.float32) {
        s_atan_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.complex128) {
        s_atan_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.complex64) {
        s_atan_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesRes);
    }
  }

  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.atan(x.toDouble()),
    );
  } else {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.atan(x),
    );
  }
  return result;
}

/// Compute the element-wise hyperbolic sine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/hyperbolic_example.dart lang=dart}
NDArray<double> sinh<T extends num>(NDArray<T> a, {NDArray<double>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute sinh() on a disposed array.');
  }
  final DType<double> targetDType = a.dtype == DType.float32
      ? DType.float32 as DType<double>
      : DType.float64 as DType<double>;
  final result = out ?? NDArray<double>.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for sinh.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_sinh_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_sinh_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  } else {
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
      if (a.dtype == DType.float64) {
        s_sinh_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.float32) {
        s_sinh_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesRes);
    }
  }

  for (var i = 0; i < a.data.length; i++) {
    final x = (a.data[i] as num).toDouble();
    result.data[i] = (math.exp(x) - math.exp(-x)) / 2.0;
  }
  return result;
}

/// Compute the element-wise hyperbolic cosine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/hyperbolic_example.dart lang=dart}
NDArray<double> cosh<T extends num>(NDArray<T> a, {NDArray<double>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cosh() on a disposed array.');
  }
  final DType<double> targetDType = a.dtype == DType.float32
      ? DType.float32 as DType<double>
      : DType.float64 as DType<double>;
  final result = out ?? NDArray<double>.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for cosh.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_cosh_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_cosh_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  } else {
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
      if (a.dtype == DType.float64) {
        s_cosh_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.float32) {
        s_cosh_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesRes);
    }
  }

  for (var i = 0; i < a.data.length; i++) {
    final x = (a.data[i] as num).toDouble();
    result.data[i] = (math.exp(x) + math.exp(-x)) / 2.0;
  }
  return result;
}

/// Compute the element-wise hyperbolic tangent of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/hyperbolic_example.dart lang=dart}
NDArray<double> tanh<T extends num>(NDArray<T> a, {NDArray<double>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute tanh() on a disposed array.');
  }
  final DType<double> targetDType = a.dtype == DType.float32
      ? DType.float32 as DType<double>
      : DType.float64 as DType<double>;
  final result = out ?? NDArray<double>.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for tanh.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_tanh_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_tanh_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  } else {
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
      if (a.dtype == DType.float64) {
        s_tanh_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.float32) {
        s_tanh_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesRes);
    }
  }

  for (var i = 0; i < a.data.length; i++) {
    final x = (a.data[i] as num).toDouble();
    result.data[i] = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1);
  }
  return result;
}

/// Compute the element-wise inverse hyperbolic sine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/hyperbolic_example.dart lang=dart}
NDArray<double> asinh<T extends num>(NDArray<T> a, {NDArray<double>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute asinh() on a disposed array.');
  }
  final DType<double> targetDType = a.dtype == DType.float32
      ? DType.float32 as DType<double>
      : DType.float64 as DType<double>;
  final result = out ?? NDArray<double>.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for asinh.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_asinh_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_asinh_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  } else {
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
      if (a.dtype == DType.float64) {
        s_asinh_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.float32) {
        s_asinh_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesRes);
    }
  }

  for (var i = 0; i < a.data.length; i++) {
    final x = (a.data[i] as num).toDouble();
    result.data[i] = math.log(x + math.sqrt(x * x + 1));
  }
  return result;
}

/// Compute the element-wise inverse hyperbolic cosine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/hyperbolic_example.dart lang=dart}
NDArray<double> acosh<T extends num>(NDArray<T> a, {NDArray<double>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute acosh() on a disposed array.');
  }
  final DType<double> targetDType = a.dtype == DType.float32
      ? DType.float32 as DType<double>
      : DType.float64 as DType<double>;
  final result = out ?? NDArray<double>.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for acosh.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_acosh_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_acosh_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  } else {
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
      if (a.dtype == DType.float64) {
        s_acosh_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.float32) {
        s_acosh_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesRes);
    }
  }

  for (var i = 0; i < a.data.length; i++) {
    final x = (a.data[i] as num).toDouble();
    result.data[i] = math.log(x + math.sqrt(x * x - 1));
  }
  return result;
}

/// Compute the element-wise inverse hyperbolic tangent of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/hyperbolic_example.dart lang=dart}
NDArray atanh(NDArray a, {NDArray? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute atanh() on a disposed array.');
  }
  final DType<dynamic> targetDType;
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    targetDType = a.dtype;
  } else {
    targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  }
  final result = out ?? NDArray.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for atanh.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_atanh_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_atanh_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.complex128) {
      v_atanh_complex128(
        a.pointer.cast(),
        result.pointer.cast(),
        a.data.length,
      );
      return result;
    } else if (a.dtype == DType.complex64) {
      v_atanh_complex64(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  } else {
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
      if (a.dtype == DType.float64) {
        s_atanh_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.float32) {
        s_atanh_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.complex128) {
        s_atanh_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      } else if (a.dtype == DType.complex64) {
        s_atanh_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
        );
        return result;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesRes);
    }
  }

  final aNum = a as NDArray<num>;
  for (var i = 0; i < a.data.length; i++) {
    final x = aNum.data[i].toDouble();
    result.data[i] = 0.5 * math.log((1 + x) / (1 - x));
  }
  return result;
}

/// Compute the element-wise arc tangent of [y] / [x] with full broadcasting support.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray atan2(NDArray y, NDArray x) {
  if (y.dtype == DType.complex128 ||
      y.dtype == DType.complex64 ||
      x.dtype == DType.complex128 ||
      x.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for atan2');
  }
  final broadcastResult = broadcast(y, x);
  final shape = broadcastResult.shape;
  final DType<dynamic> targetDType =
      (y.dtype == DType.float32 && x.dtype == DType.float32)
      ? DType.float32
      : DType.float64;

  final result = NDArray.create(shape, targetDType);

  // 0. Native C Vector Extension Fast-Path Gate for Contiguous Same-Shape arrays
  if (y.isContiguous && x.isContiguous && listEquals(y.shape, x.shape)) {
    if (y.dtype == DType.float64 && x.dtype == DType.float64) {
      v_atan2_double(
        y.pointer.cast(),
        x.pointer.cast(),
        result.pointer.cast(),
        y.data.length,
      );
      return result;
    } else if (y.dtype == DType.float32 && x.dtype == DType.float32) {
      v_atan2_float(
        y.pointer.cast(),
        x.pointer.cast(),
        result.pointer.cast(),
        y.data.length,
      );
      return result;
    }
  }

  final resultStrides = NDArray.computeCStrides(shape);
  final stridesY = broadcastResult.stridesA;
  final stridesX = broadcastResult.stridesB;

  // 0C. General Multidimensional Strided Broadcasting Engine in C (Rank <= 8)
  if (shape.length <= 8) {
    final cShape = malloc<ffi.Int>(shape.length);
    final cStridesY = malloc<ffi.Int>(stridesY.length);
    final cStridesX = malloc<ffi.Int>(stridesX.length);
    final cStridesRes = malloc<ffi.Int>(resultStrides.length);

    for (var i = 0; i < shape.length; i++) {
      cShape[i] = shape[i];
      cStridesY[i] = stridesY[i];
      cStridesX[i] = stridesX[i];
      cStridesRes[i] = resultStrides[i];
    }

    try {
      if (targetDType == DType.float64 &&
          y.dtype == DType.float64 &&
          x.dtype == DType.float64) {
        s_atan2_double(
          y.pointer.cast(),
          cStridesY,
          x.pointer.cast(),
          cStridesX,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          shape.length,
        );
        return result;
      } else if (targetDType == DType.float32 &&
          y.dtype == DType.float32 &&
          x.dtype == DType.float32) {
        s_atan2_float(
          y.pointer.cast(),
          cStridesY,
          x.pointer.cast(),
          cStridesX,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          shape.length,
        );
        return result;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesY);
      malloc.free(cStridesX);
      malloc.free(cStridesRes);
    }
  }

  final rData = result.data as List<double>;

  if (y.dtype == DType.float64 || y.dtype == DType.float32) {
    final yData = y.data as List<double>;
    if (x.dtype == DType.float64 || x.dtype == DType.float32) {
      _elementWiseOp<double, double, double>(
        rData,
        yData,
        x.data as List<double>,
        shape,
        broadcastResult.stridesA,
        broadcastResult.stridesB,
        resultStrides,
        0,
        0,
        0,
        0,
        (a, b) => math.atan2(a, b),
      );
    } else {
      _elementWiseOp<double, int, double>(
        rData,
        yData,
        x.data as List<int>,
        shape,
        broadcastResult.stridesA,
        broadcastResult.stridesB,
        resultStrides,
        0,
        0,
        0,
        0,
        (a, b) => math.atan2(a, b.toDouble()),
      );
    }
  } else {
    final yData = y.data as List<int>;
    if (x.dtype == DType.float64 || x.dtype == DType.float32) {
      _elementWiseOp<int, double, double>(
        rData,
        yData,
        x.data as List<double>,
        shape,
        broadcastResult.stridesA,
        broadcastResult.stridesB,
        resultStrides,
        0,
        0,
        0,
        0,
        (a, b) => math.atan2(a.toDouble(), b),
      );
    } else {
      _elementWiseOp<int, int, double>(
        rData,
        yData,
        x.data as List<int>,
        shape,
        broadcastResult.stridesA,
        broadcastResult.stridesB,
        resultStrides,
        0,
        0,
        0,
        0,
        (a, b) => math.atan2(a.toDouble(), b.toDouble()),
      );
    }
  }
  return result;
}

/// Compute the element-wise hypotenuse `sqrt(x1**2 + x2**2)` with broadcasting support.
///
/// **Example:**
/// ```dart
/// final h = hypot(a, b);
/// ```
NDArray<double> hypot(NDArray x1, NDArray x2) {
  final broadcastResult = broadcast(x1, x2);
  final shape = broadcastResult.shape;
  final DType<double> targetDType =
      (x1.dtype == DType.complex64 || x2.dtype == DType.complex64)
      ? DType.float32 as DType<double>
      : DType.float64 as DType<double>;
  final result = NDArray<double>.create(shape, targetDType);
  final resultStrides = NDArray.computeCStrides(shape);

  if (x1.dtype == DType.complex128 ||
      x2.dtype == DType.complex128 ||
      x1.dtype == DType.complex64 ||
      x2.dtype == DType.complex64) {
    final aCpx = (x1.dtype == DType.complex128 || x1.dtype == DType.complex64)
        ? x1
        : NDArray<Complex>.fromList(
            x1.data.map((e) => Complex((e as num).toDouble(), 0.0)).toList(),
            x1.shape,
            DType.complex128,
          );
    final bCpx = (x2.dtype == DType.complex128 || x2.dtype == DType.complex64)
        ? x2
        : NDArray<Complex>.fromList(
            x2.data.map((e) => Complex((e as num).toDouble(), 0.0)).toList(),
            x2.shape,
            DType.complex128,
          );
    if (listEquals(x1.shape, x2.shape) &&
        x1.isContiguous &&
        x2.isContiguous &&
        result.isContiguous) {
      if (aCpx.dtype == DType.complex128) {
        v_hypot_complex128(
          aCpx.pointer.cast(),
          bCpx.pointer.cast(),
          result.pointer.cast(),
          aCpx.data.length,
        );
        return result;
      } else {
        v_hypot_complex64(
          aCpx.pointer.cast(),
          bCpx.pointer.cast(),
          result.pointer.cast(),
          aCpx.data.length,
        );
        return result;
      }
    } else {
      final rank = shape.length;
      final cShape = malloc<ffi.Int>(rank);
      final cStridesA = malloc<ffi.Int>(rank);
      final cStridesB = malloc<ffi.Int>(rank);
      final cStridesRes = malloc<ffi.Int>(rank);
      for (var i = 0; i < rank; i++) {
        cShape[i] = shape[i];
        cStridesA[i] = broadcastResult.stridesA[i];
        cStridesB[i] = broadcastResult.stridesB[i];
        cStridesRes[i] = resultStrides[i];
      }
      try {
        if (aCpx.dtype == DType.complex128) {
          s_hypot_complex128(
            aCpx.pointer.cast(),
            cStridesA,
            bCpx.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        } else {
          s_hypot_complex64(
            aCpx.pointer.cast(),
            cStridesA,
            bCpx.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        }
      } finally {
        malloc.free(cShape);
        malloc.free(cStridesA);
        malloc.free(cStridesB);
        malloc.free(cStridesRes);
      }
    }
  }

  final rData = result.data;

  double hypotOp(double x, double y) {
    x = x.abs();
    y = y.abs();
    if (x < y) {
      final temp = x;
      x = y;
      y = temp;
    }
    if (x == 0) return 0.0;
    final t = y / x;
    return x * math.sqrt(1.0 + t * t);
  }

  _elementWiseOp<num, num, double>(
    rData,
    x1.data as List<num>,
    x2.data as List<num>,
    shape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    0,
    0,
    0,
    0,
    (valA, valB) => hypotOp(valA.toDouble(), valB.toDouble()),
  );

  return result;
}

/// Compute the element-wise power `x1**x2` with broadcasting support.
///
/// **Example:**
/// ```dart
/// final p = power(a, b);
/// ```
NDArray power(NDArray x1, NDArray x2) {
  final broadcastResult = broadcast(x1, x2);
  final shape = broadcastResult.shape;
  final DType<dynamic> targetDType;
  if (x1.dtype == DType.complex128 ||
      x2.dtype == DType.complex128 ||
      x1.dtype == DType.complex64 ||
      x2.dtype == DType.complex64) {
    targetDType = (x1.dtype == DType.complex64 || x2.dtype == DType.complex64)
        ? DType.complex64
        : DType.complex128;
  } else {
    targetDType = (x1.dtype == DType.float32 || x2.dtype == DType.float32)
        ? DType.float32
        : DType.float64;
  }
  final result = NDArray.create(shape, targetDType);
  final resultStrides = NDArray.computeCStrides(shape);

  if (targetDType == DType.complex128 || targetDType == DType.complex64) {
    final aCpx = (x1.dtype == DType.complex128 || x1.dtype == DType.complex64)
        ? x1
        : NDArray<Complex>.fromList(
            x1.data.map((e) => Complex((e as num).toDouble(), 0.0)).toList(),
            x1.shape,
            DType.complex128,
          );
    final bCpx = (x2.dtype == DType.complex128 || x2.dtype == DType.complex64)
        ? x2
        : NDArray<Complex>.fromList(
            x2.data.map((e) => Complex((e as num).toDouble(), 0.0)).toList(),
            x2.shape,
            DType.complex128,
          );
    if (listEquals(x1.shape, x2.shape) &&
        x1.isContiguous &&
        x2.isContiguous &&
        result.isContiguous) {
      if (aCpx.dtype == DType.complex128) {
        v_pow_complex128(
          aCpx.pointer.cast(),
          bCpx.pointer.cast(),
          result.pointer.cast(),
          aCpx.data.length,
        );
        return result;
      } else {
        v_pow_complex64(
          aCpx.pointer.cast(),
          bCpx.pointer.cast(),
          result.pointer.cast(),
          aCpx.data.length,
        );
        return result;
      }
    } else {
      final rank = shape.length;
      final cShape = malloc<ffi.Int>(rank);
      final cStridesA = malloc<ffi.Int>(rank);
      final cStridesB = malloc<ffi.Int>(rank);
      final cStridesRes = malloc<ffi.Int>(rank);
      for (var i = 0; i < rank; i++) {
        cShape[i] = shape[i];
        cStridesA[i] = broadcastResult.stridesA[i];
        cStridesB[i] = broadcastResult.stridesB[i];
        cStridesRes[i] = resultStrides[i];
      }
      try {
        if (aCpx.dtype == DType.complex128) {
          s_pow_complex128(
            aCpx.pointer.cast(),
            cStridesA,
            bCpx.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        } else {
          s_pow_complex64(
            aCpx.pointer.cast(),
            cStridesA,
            bCpx.pointer.cast(),
            cStridesB,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        }
      } finally {
        malloc.free(cShape);
        malloc.free(cStridesA);
        malloc.free(cStridesB);
        malloc.free(cStridesRes);
      }
    }
  }

  _elementWiseOp<num, num, double>(
    result.data as List<double>,
    x1.data as List<num>,
    x2.data as List<num>,
    shape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    0,
    0,
    0,
    0,
    (valA, valB) => math.pow(valA.toDouble(), valB.toDouble()).toDouble(),
  );

  return result;
}

/// Numerical negative, element-wise.
///
/// **Example:**
/// ```dart
/// final b = negative(a);
/// ```
NDArray negative(NDArray a) {
  final result = NDArray.create(a.shape, a.dtype);
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    _unaryOp<Complex, Complex>(
      result.data as List<Complex>,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => -x,
    );
  } else if (a.dtype == DType.float64 || a.dtype == DType.float32) {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => -x,
    );
  } else if (a.dtype == DType.int64 || a.dtype == DType.int32) {
    _unaryOp<int, int>(
      result.data as List<int>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => -x,
    );
  } else if (a.dtype == DType.boolean) {
    throw UnsupportedError('Boolean arrays do not support negative operator');
  }
  return result;
}

/// Element-wise floor division with broadcasting and dtype upcasting support.
///
/// Corresponds to Dart's `~/` operator.
///
/// **Example:**
/// ```dart
/// final c = floor_divide(a, b);
/// ```
NDArray floor_divide(NDArray a, NDArray b) {
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final DType<dynamic> targetDType = _resolveDType(a.dtype, b.dtype);
  if (targetDType.isComplex) {
    throw UnsupportedError('Complex numbers do not support floor division');
  }
  final result = NDArray.create(commonShape, targetDType);
  final resultStrides = NDArray.computeCStrides(commonShape);

  if (targetDType == DType.float64 || targetDType == DType.float32) {
    _elementWiseOp<double, double, double>(
      result.data as List<double>,
      a.data as List<double>,
      b.data as List<double>,
      commonShape,
      stridesA,
      stridesB,
      resultStrides,
      0,
      0,
      0,
      0,
      (x, y) => (x ~/ y).toDouble(),
    );
  } else {
    _elementWiseOp<int, int, int>(
      result.data as List<int>,
      a.data as List<int>,
      b.data as List<int>,
      commonShape,
      stridesA,
      stridesB,
      resultStrides,
      0,
      0,
      0,
      0,
      (x, y) => x ~/ y,
    );
  }
  return result;
}

/// Element-wise remainder of division with broadcasting and dtype upcasting support.
///
/// Corresponds to Dart's `%` operator.
///
/// **Example:**
/// ```dart
/// final c = remainder(a, b);
/// ```
NDArray remainder(NDArray a, NDArray b) {
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final DType<dynamic> targetDType = _resolveDType(a.dtype, b.dtype);
  if (targetDType.isComplex) {
    throw UnsupportedError('Complex numbers do not support remainder');
  }
  final result = NDArray.create(commonShape, targetDType);
  final resultStrides = NDArray.computeCStrides(commonShape);

  if (targetDType == DType.float64 || targetDType == DType.float32) {
    _elementWiseOp<double, double, double>(
      result.data as List<double>,
      a.data as List<double>,
      b.data as List<double>,
      commonShape,
      stridesA,
      stridesB,
      resultStrides,
      0,
      0,
      0,
      0,
      (x, y) => x % y,
    );
  } else {
    _elementWiseOp<int, int, int>(
      result.data as List<int>,
      a.data as List<int>,
      b.data as List<int>,
      commonShape,
      stridesA,
      stridesB,
      resultStrides,
      0,
      0,
      0,
      0,
      (x, y) => x % y,
    );
  }
  return result;
}

/// Element-wise bitwise AND with broadcasting support.
///
/// **Example:**
/// ```dart
/// final c = bitwise_and(a, b);
/// ```
NDArray bitwise_and(NDArray a, NDArray b) {
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  if (!a.dtype.isInteger || !b.dtype.isInteger) {
    throw UnsupportedError(
      'Bitwise operations only supported for integer dtypes',
    );
  }
  final DType<dynamic> targetDType = _resolveDType(a.dtype, b.dtype);
  final result = NDArray.create(commonShape, targetDType);
  final resultStrides = NDArray.computeCStrides(commonShape);

  _elementWiseOp<int, int, int>(
    result.data as List<int>,
    a.data as List<int>,
    b.data as List<int>,
    commonShape,
    stridesA,
    stridesB,
    resultStrides,
    0,
    0,
    0,
    0,
    (x, y) => x & y,
  );
  return result;
}

/// Element-wise bitwise OR with broadcasting support.
///
/// **Example:**
/// ```dart
/// final c = bitwise_or(a, b);
/// ```
NDArray bitwise_or(NDArray a, NDArray b) {
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  if (!a.dtype.isInteger || !b.dtype.isInteger) {
    throw UnsupportedError(
      'Bitwise operations only supported for integer dtypes',
    );
  }
  final DType<dynamic> targetDType = _resolveDType(a.dtype, b.dtype);
  final result = NDArray.create(commonShape, targetDType);
  final resultStrides = NDArray.computeCStrides(commonShape);

  _elementWiseOp<int, int, int>(
    result.data as List<int>,
    a.data as List<int>,
    b.data as List<int>,
    commonShape,
    stridesA,
    stridesB,
    resultStrides,
    0,
    0,
    0,
    0,
    (x, y) => x | y,
  );
  return result;
}

/// Element-wise bitwise XOR with broadcasting support.
///
/// **Example:**
/// ```dart
/// final c = bitwise_xor(a, b);
/// ```
NDArray bitwise_xor(NDArray a, NDArray b) {
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  if (!a.dtype.isInteger || !b.dtype.isInteger) {
    throw UnsupportedError(
      'Bitwise operations only supported for integer dtypes',
    );
  }
  final DType<dynamic> targetDType = _resolveDType(a.dtype, b.dtype);
  final result = NDArray.create(commonShape, targetDType);
  final resultStrides = NDArray.computeCStrides(commonShape);

  _elementWiseOp<int, int, int>(
    result.data as List<int>,
    a.data as List<int>,
    b.data as List<int>,
    commonShape,
    stridesA,
    stridesB,
    resultStrides,
    0,
    0,
    0,
    0,
    (x, y) => x ^ y,
  );
  return result;
}

/// Element-wise bitwise NOT.
///
/// **Example:**
/// ```dart
/// final b = bitwise_not(a);
/// ```
NDArray bitwise_not(NDArray a) {
  if (!a.dtype.isInteger) {
    throw UnsupportedError(
      'Bitwise operations only supported for integer dtypes',
    );
  }
  final result = NDArray.create(a.shape, a.dtype);
  final resultStrides = NDArray.computeCStrides(a.shape);

  _unaryOp<int, int>(
    result.data as List<int>,
    a.data as List<int>,
    a.shape,
    a.strides,
    resultStrides,
    0,
    0,
    0,
    (x) => ~x,
  );
  return result;
}

/// Element-wise left shift with broadcasting support.
///
/// **Example:**
/// ```dart
/// final c = left_shift(a, b);
/// ```
NDArray left_shift(NDArray a, NDArray b) {
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  if (!a.dtype.isInteger || !b.dtype.isInteger) {
    throw UnsupportedError(
      'Shift operations only supported for integer dtypes',
    );
  }
  final DType<dynamic> targetDType = _resolveDType(a.dtype, b.dtype);
  final result = NDArray.create(commonShape, targetDType);
  final resultStrides = NDArray.computeCStrides(commonShape);

  _elementWiseOp<int, int, int>(
    result.data as List<int>,
    a.data as List<int>,
    b.data as List<int>,
    commonShape,
    stridesA,
    stridesB,
    resultStrides,
    0,
    0,
    0,
    0,
    (x, y) => x << y,
  );
  return result;
}

/// Element-wise right shift with broadcasting support.
///
/// **Example:**
/// ```dart
/// final c = right_shift(a, b);
/// ```
NDArray right_shift(NDArray a, NDArray b) {
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  if (!a.dtype.isInteger || !b.dtype.isInteger) {
    throw UnsupportedError(
      'Shift operations only supported for integer dtypes',
    );
  }
  final DType<dynamic> targetDType = _resolveDType(a.dtype, b.dtype);
  final result = NDArray.create(commonShape, targetDType);
  final resultStrides = NDArray.computeCStrides(commonShape);

  _elementWiseOp<int, int, int>(
    result.data as List<int>,
    a.data as List<int>,
    b.data as List<int>,
    commonShape,
    stridesA,
    stridesB,
    resultStrides,
    0,
    0,
    0,
    0,
    (x, y) => x >> y,
  );
  return result;
}

/// Compute the element-wise absolute value of the array.
///
/// For complex numbers, returns the magnitude as a real array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray abs(NDArray a) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    final DType<dynamic> targetDType = a.dtype == DType.complex64
        ? DType.float32
        : DType.float64;
    final result = NDArray.create(a.shape, targetDType);
    final resultStrides = NDArray.computeCStrides(a.shape);

    _unaryOp<Complex, double>(
      result.data as List<double>,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (c) => math.sqrt(c.real * c.real + c.imag * c.imag),
    );
    return result;
  }

  final result = NDArray.create(a.shape, a.dtype);
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, int>(
      result.data as List<int>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.abs(),
    );
  } else {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.abs(),
    );
  }
  return result;
}

/// Compute the element-wise sign of the array.
///
/// For real numbers, returns:
/// - -1 if x < 0
/// - 0 if x == 0
/// - 1 if x > 0
/// - nan if x is nan
///
/// For complex numbers, returns `x / |x|` (or 0 if x is 0).
///
/// **Example:**
/// ```dart
/// final s = sign(a);
/// ```
NDArray sign(NDArray a) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    final result = NDArray.create(a.shape, a.dtype);
    final resultStrides = NDArray.computeCStrides(a.shape);

    _unaryOp<Complex, Complex>(
      result.data as List<Complex>,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (c) {
        if (c.real == 0 && c.imag == 0) return Complex(0, 0);
        final mag = math.sqrt(c.real * c.real + c.imag * c.imag);
        return Complex(c.real / mag, c.imag / mag);
      },
    );
    return result;
  }

  final result = NDArray.create(a.shape, a.dtype);
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, int>(
      result.data as List<int>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.sign,
    );
  } else {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.sign,
    );
  }
  return result;
}

/// Compute the element-wise ceiling of the array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray ceil(NDArray a) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for ceil');
  }
  final result = NDArray.create(a.shape, a.dtype);

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_ceil_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_ceil_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  }
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, int>(
      result.data as List<int>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x,
    );
  } else {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.ceilToDouble(),
    );
  }
  return result;
}

/// Compute the element-wise floor of the array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray floor(NDArray a) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for floor');
  }
  final result = NDArray.create(a.shape, a.dtype);

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_floor_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_floor_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  }
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, int>(
      result.data as List<int>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x,
    );
  } else {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.floorToDouble(),
    );
  }
  return result;
}

/// Compute the element-wise rounding of the array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray round(NDArray a) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for round');
  }
  final result = NDArray.create(a.shape, a.dtype);

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_round_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_round_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  }
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, int>(
      result.data as List<int>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x,
    );
  } else {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.roundToDouble(),
    );
  }
  return result;
}

/// Enumerates elements of a multidimensional array yielding coordinates and values.
///
/// Yields records containing the coordinate list and the element value at that coordinate
/// in standard C-contiguous order.
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
///
/// **Throws:**
/// - [StateError] if [a] has been disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.int32);
/// for (final entry in ndenumerate(a)) {
///   print('coord: ${entry.$1}, value: ${entry.$2}');
/// }
/// // Yields:
/// // ([0, 0], 10)
/// // ([0, 1], 20)
/// // ([1, 0], 30)
/// // ([1, 1], 40)
/// ```
Iterable<(List<int> coordinate, T value)> ndenumerate<T>(NDArray<T> a) sync* {
  if (a.isDisposed) {
    throw StateError('Cannot execute ndenumerate() on a disposed array.');
  }

  final shape = a.shape;
  final strides = a.strides;
  final totalSize = shape.isEmpty ? 1 : shape.reduce((x, y) => x * y);

  if (shape.isEmpty) {
    yield ([], a.data[0]);
    return;
  }

  final coord = List<int>.filled(shape.length, 0);
  int offset = 0;

  for (int el = 0; el < totalSize; el++) {
    // Yield a copy of the coordinate list so that users don't receive the same mutated buffer!
    yield (List<int>.from(coord), a.data[offset]);

    // Advance odometer multidimensional coordinate odometer walk!
    for (int d = shape.length - 1; d >= 0; d--) {
      coord[d]++;
      if (coord[d] < shape[d]) {
        offset += strides[d];
        break;
      }
      coord[d] = 0;
      offset -= (shape[d] - 1) * strides[d];
    }
  }
}

/// Returns the real part of a complex array element-wise.
///
/// If the input array [a] is already real (integer or float), returns a zero-copy
/// view of the array [a].
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
/// - If provided, the output recycler [out] must match the expected target shape and float `DType.`
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [ArgumentError] if [out] is provided but has a shape or DType mismatch.
///
/// **Example:**
/// ```dart
/// final a = `NDArray<Complex>`.create([2], `DType.complex128);`
/// a.data[0] = Complex(3.0, 4.0);
/// a.data[1] = Complex(-1.0, 0.0);
/// final r = real(a); // [3.0, -1.0] (`DType.float64)`
/// ```
NDArray real(NDArray a, {NDArray? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute real() on a disposed array.');
  }

  final DType<double> targetDType = a.dtype == DType.complex64
      ? DType.float32
      : DType.float64;

  if (a.dtype != DType.complex128 && a.dtype != DType.complex64) {
    if (out != null) {
      if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
        throw ArgumentError(
          'Recycler out must match shape ${a.shape} and DType ${a.dtype}',
        );
      }
      // Copy elements out out
      final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
      out.data.setRange(0, size, a.toList());
      return out;
    }
    return NDArray.view(
      a,
      shape: a.shape,
      strides: a.strides,
    ); // Zero-copy view for already real arrays!
  }

  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Recycler out must match shape ${a.shape} and DType $targetDType',
      );
    }
  }

  final result = out ?? NDArray<double>.zeros(a.shape, targetDType);
  final resultStrides = NDArray.computeCStrides(a.shape);

  _unaryOp<Complex, double>(
    result.data as List<double>,
    a.data as List<Complex>,
    a.shape,
    a.strides,
    resultStrides,
    0,
    0,
    0,
    (x) => x.real,
  );

  return result;
}

/// Returns the imaginary part of a complex array element-wise.
///
/// If the input array [a] is already real, returns a zero-filled array of matching shape
/// and target float `DType.`
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
/// - If provided, the output recycler [out] must match the expected target shape and float `DType.`
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [ArgumentError] if [out] is provided but has a shape or DType mismatch.
///
/// **Example:**
/// ```dart
/// final a = `NDArray<Complex>`.create([2], `DType.complex128);`
/// a.data[0] = Complex(3.0, 4.0);
/// a.data[1] = Complex(-1.0, 0.0);
/// final im = imag(a); // [4.0, 0.0] (`DType.float64)`
/// ```
NDArray imag(NDArray a, {NDArray? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute imag() on a disposed array.');
  }

  final DType<double> targetDType = a.dtype == DType.complex64
      ? DType.float32
      : DType.float64;

  if (a.dtype != DType.complex128 && a.dtype != DType.complex64) {
    if (out != null) {
      if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
        throw ArgumentError(
          'Recycler out must match shape ${a.shape} and DType $targetDType',
        );
      }
      out.data.fillRange(0, out.data.length, 0.0);
      return out;
    }
    return NDArray<double>.zeros(a.shape, targetDType);
  }

  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Recycler out must match shape ${a.shape} and DType $targetDType',
      );
    }
  }

  final result = out ?? NDArray<double>.zeros(a.shape, targetDType);
  final resultStrides = NDArray.computeCStrides(a.shape);

  _unaryOp<Complex, double>(
    result.data as List<double>,
    a.data as List<Complex>,
    a.shape,
    a.strides,
    resultStrides,
    0,
    0,
    0,
    (x) => x.imag,
  );

  return result;
}

/// Converts angles from degrees to radians element-wise.
///
/// **Preconditions:**
/// - Input array [a] must not be disposed.
/// - Input array [a] must not contain complex numbers.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [UnsupportedError] if the array has a complex data type.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([180.0, 90.0, 45.0], [3], DType.float64);
/// final r = deg2rad(a); // [pi, pi / 2.0, pi / 4.0]
/// ```
NDArray deg2rad(NDArray a, {NDArray? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute deg2rad on a disposed array.');
  }
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for deg2rad');
  }
  // pi / 180.0 = 0.017453292519943295
  final DType<dynamic> factorDType = a.dtype == DType.float32
      ? DType.float32
      : DType.float64;
  final factor = NDArray.fromList([0.017453292519943295], [1], factorDType);
  return multiply(a, factor, out: out);
}

/// Converts angles from radians to degrees element-wise.
///
/// **Preconditions:**
/// - Input array [a] must not be disposed.
/// - Input array [a] must not contain complex numbers.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [UnsupportedError] if the array has a complex data type.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([math.pi, math.pi / 2.0], [2], DType.float64);
/// final d = rad2deg(a); // [180.0, 90.0]
/// ```
NDArray rad2deg(NDArray a, {NDArray? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute rad2deg on a disposed array.');
  }
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for rad2deg');
  }
  // 180.0 / pi = 57.29577951308232
  final DType<dynamic> factorDType = a.dtype == DType.float32
      ? DType.float32
      : DType.float64;
  final factor = NDArray.fromList([57.29577951308232], [1], factorDType);
  return multiply(a, factor, out: out);
}

/// Returns an element-wise boolean mask indicating which elements of the array are NaN (Not-a-Number).
///
/// **Preconditions:**
/// - The array must not be disposed.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.nan, 3.0], [3], DType.float64);
/// final mask = isnan(a); // [false, true, false]
/// ```
NDArray<bool> isnan(NDArray a) {
  if (a.isDisposed) {
    throw StateError('Cannot execute isnan on a disposed array.');
  }
  final result = NDArray<bool>.create(a.shape, DType.boolean);
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    _unaryOp<Complex, bool>(
      result.data,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.real.isNaN || x.imag.isNaN,
    );
  } else if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, bool>(
      result.data,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => false,
    );
  } else {
    _unaryOp<double, bool>(
      result.data,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.isNaN,
    );
  }
  return result;
}

/// Returns an element-wise boolean mask indicating which elements of the array are positive or negative infinity.
///
/// **Preconditions:**
/// - The array must not be disposed.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.infinity, 3.0], [3], DType.float64);
/// final mask = isinf(a); // [false, true, false]
/// ```
NDArray<bool> isinf(NDArray a) {
  if (a.isDisposed) {
    throw StateError('Cannot execute isinf on a disposed array.');
  }
  final result = NDArray<bool>.create(a.shape, DType.boolean);
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    _unaryOp<Complex, bool>(
      result.data,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.real.isInfinite || x.imag.isInfinite,
    );
  } else if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, bool>(
      result.data,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => false,
    );
  } else {
    _unaryOp<double, bool>(
      result.data,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.isInfinite,
    );
  }
  return result;
}

/// Returns an element-wise boolean mask indicating which elements of the array are finite (neither NaN nor infinite).
///
/// **Preconditions:**
/// - The array must not be disposed.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.nan, double.infinity], [3], DType.float64);
/// final mask = isfinite(a); // [true, false, false]
/// ```
NDArray<bool> isfinite(NDArray a) {
  if (a.isDisposed) {
    throw StateError('Cannot execute isfinite on a disposed array.');
  }
  final result = NDArray<bool>.create(a.shape, DType.boolean);
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    _unaryOp<Complex, bool>(
      result.data,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.real.isFinite && x.imag.isFinite,
    );
  } else if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, bool>(
      result.data,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => true,
    );
  } else {
    _unaryOp<double, bool>(
      result.data,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.isFinite,
    );
  }
  return result;
}

/// Clip (limit) the values in an array.
///
/// Given an interval `[min, max]`, values outside the interval are clipped
/// to the interval edges.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray clip(NDArray a, {required num min, required num max, NDArray? out}) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for clip');
  }
  final result = out ?? NDArray.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for clip.',
      );
    }
    if (out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible DType for clip.',
      );
    }
  }

  final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_clip_double(
        a.pointer.cast(),
        result.pointer.cast(),
        min.toDouble(),
        max.toDouble(),
        size,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      v_clip_float(
        a.pointer.cast(),
        result.pointer.cast(),
        min.toDouble(),
        max.toDouble(),
        size,
      );
      return result;
    }
  }
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    final mn = min.toInt();
    final mx = max.toInt();
    _unaryOp<int, int>(
      result.data as List<int>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.clamp(mn, mx),
    );
  } else {
    final mn = min.toDouble();
    final mx = max.toDouble();
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.clamp(mn, mx),
    );
  }
  return result;
}

bool _isTrue(dynamic x) {
  if (x is bool) {
    return x;
  } else if (x is Complex) {
    return x.real != 0.0 || x.imag != 0.0;
  } else if (x is num) {
    return x != 0;
  }
  return false;
}

/// Compute the element-wise truth value of NOT [a].
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray logical_not(NDArray a) {
  final result = NDArray.create(a.shape, DType.int32);
  final resultStrides = NDArray.computeCStrides(a.shape);
  final rData = result.data as List<int>;

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    _unaryOp<Complex, int>(
      rData,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => _isTrue(x) ? 0 : 1,
    );
  } else if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, int>(
      rData,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => _isTrue(x) ? 0 : 1,
    );
  } else {
    _unaryOp<double, int>(
      rData,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => _isTrue(x) ? 0 : 1,
    );
  }
  return result;
}

/// Element-wise comparison of [a] == [b] with broadcasting and recycling support.
NDArray<bool> equal(NDArray a, NDArray b, {NDArray<bool>? out}) {
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final result = out ?? NDArray<bool>.create(commonShape, DType.boolean);
  final resultStrides = NDArray.computeCStrides(commonShape);

  a.dispatchCompare(
    result.data,
    a,
    b,
    commonShape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    (x, y) => x == y,
  );
  return result;
}

/// Element-wise comparison of [a] != [b] with broadcasting and recycling support.
NDArray<bool> not_equal(NDArray a, NDArray b, {NDArray<bool>? out}) {
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final result = out ?? NDArray<bool>.create(commonShape, DType.boolean);
  final resultStrides = NDArray.computeCStrides(commonShape);

  a.dispatchCompare(
    result.data,
    a,
    b,
    commonShape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    (x, y) => x != y,
  );
  return result;
}

/// Element-wise comparison of [a] > [b] with broadcasting and recycling support.
NDArray<bool> greater(NDArray a, NDArray b, {NDArray<bool>? out}) {
  if (a.dtype.isComplex || b.dtype.isComplex) {
    throw UnsupportedError(
      'Complex numbers do not support inequality comparisons',
    );
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final result = out ?? NDArray<bool>.create(commonShape, DType.boolean);
  final resultStrides = NDArray.computeCStrides(commonShape);

  a.dispatchCompare(
    result.data,
    a,
    b,
    commonShape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    (x, y) => (x as num) > (y as num),
  );
  return result;
}

/// Element-wise comparison of [a] >= [b] with broadcasting and recycling support.
NDArray<bool> greater_equal(NDArray a, NDArray b, {NDArray<bool>? out}) {
  if (a.dtype.isComplex || b.dtype.isComplex) {
    throw UnsupportedError(
      'Complex numbers do not support inequality comparisons',
    );
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final result = out ?? NDArray<bool>.create(commonShape, DType.boolean);
  final resultStrides = NDArray.computeCStrides(commonShape);

  a.dispatchCompare(
    result.data,
    a,
    b,
    commonShape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    (x, y) => (x as num) >= (y as num),
  );
  return result;
}

/// Element-wise comparison of [a] < [b] with broadcasting and recycling support.
NDArray<bool> less(NDArray a, NDArray b, {NDArray<bool>? out}) {
  if (a.dtype.isComplex || b.dtype.isComplex) {
    throw UnsupportedError(
      'Complex numbers do not support inequality comparisons',
    );
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final result = out ?? NDArray<bool>.create(commonShape, DType.boolean);
  final resultStrides = NDArray.computeCStrides(commonShape);

  a.dispatchCompare(
    result.data,
    a,
    b,
    commonShape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    (x, y) => (x as num) < (y as num),
  );
  return result;
}

/// Element-wise comparison of [a] <= [b] with broadcasting and recycling support.
NDArray<bool> less_equal(NDArray a, NDArray b, {NDArray<bool>? out}) {
  if (a.dtype.isComplex || b.dtype.isComplex) {
    throw UnsupportedError(
      'Complex numbers do not support inequality comparisons',
    );
  }
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != DType.boolean) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final result = out ?? NDArray<bool>.create(commonShape, DType.boolean);
  final resultStrides = NDArray.computeCStrides(commonShape);

  a.dispatchCompare(
    result.data,
    a,
    b,
    commonShape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    (x, y) => (x as num) <= (y as num),
  );
  return result;
}

/// Compute the element-wise truth value of [a] AND [b] with broadcasting support.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray logical_and(NDArray a, NDArray b) {
  final broadcastResult = broadcast(a, b);
  final shape = broadcastResult.shape;
  final result = NDArray.create(shape, DType.int32);
  final resultStrides = NDArray.computeCStrides(shape);
  final rData = result.data as List<int>;

  _dispatchBinaryLogical(
    rData,
    a,
    b,
    shape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    (x, y) => (_isTrue(x) && _isTrue(y)) ? 1 : 0,
  );
  return result;
}

/// Compute the element-wise truth value of [a] OR [b] with broadcasting support.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray logical_or(NDArray a, NDArray b) {
  final broadcastResult = broadcast(a, b);
  final shape = broadcastResult.shape;
  final result = NDArray.create(shape, DType.int32);
  final resultStrides = NDArray.computeCStrides(shape);
  final rData = result.data as List<int>;

  _dispatchBinaryLogical(
    rData,
    a,
    b,
    shape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    (x, y) => (_isTrue(x) || _isTrue(y)) ? 1 : 0,
  );
  return result;
}

/// Compute the element-wise truth value of [a] XOR [b] with broadcasting support.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray logical_xor(NDArray a, NDArray b) {
  final broadcastResult = broadcast(a, b);
  final shape = broadcastResult.shape;
  final result = NDArray.create(shape, DType.int32);
  final resultStrides = NDArray.computeCStrides(shape);
  final rData = result.data as List<int>;

  _dispatchBinaryLogical(
    rData,
    a,
    b,
    shape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    (x, y) => (_isTrue(x) != _isTrue(y)) ? 1 : 0,
  );
  return result;
}

void _dispatchBinaryLogical(
  List<int> rData,
  NDArray a,
  NDArray b,
  List<int> shape,
  List<int> sA,
  List<int> sB,
  List<int> sR,
  int Function(dynamic, dynamic) op,
) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    final aData = a.data as List<Complex>;
    if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
      _elementWiseOp<Complex, Complex, int>(
        rData,
        aData,
        b.data as List<Complex>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else if (b.dtype == DType.float64 || b.dtype == DType.float32) {
      _elementWiseOp<Complex, double, int>(
        rData,
        aData,
        b.data as List<double>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else if (b.dtype == DType.boolean) {
      _elementWiseOp<Complex, bool, int>(
        rData,
        aData,
        b.data as List<bool>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else {
      _elementWiseOp<Complex, int, int>(
        rData,
        aData,
        b.data as List<int>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    }
  } else if (a.dtype == DType.float64 || a.dtype == DType.float32) {
    final aData = a.data as List<double>;
    if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
      _elementWiseOp<double, Complex, int>(
        rData,
        aData,
        b.data as List<Complex>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else if (b.dtype == DType.float64 || b.dtype == DType.float32) {
      _elementWiseOp<double, double, int>(
        rData,
        aData,
        b.data as List<double>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else if (b.dtype == DType.boolean) {
      _elementWiseOp<double, bool, int>(
        rData,
        aData,
        b.data as List<bool>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else {
      _elementWiseOp<double, int, int>(
        rData,
        aData,
        b.data as List<int>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    }
  } else if (a.dtype == DType.boolean) {
    final aData = a.data as List<bool>;
    if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
      _elementWiseOp<bool, Complex, int>(
        rData,
        aData,
        b.data as List<Complex>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else if (b.dtype == DType.float64 || b.dtype == DType.float32) {
      _elementWiseOp<bool, double, int>(
        rData,
        aData,
        b.data as List<double>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else if (b.dtype == DType.boolean) {
      _elementWiseOp<bool, bool, int>(
        rData,
        aData,
        b.data as List<bool>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else {
      _elementWiseOp<bool, int, int>(
        rData,
        aData,
        b.data as List<int>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    }
  } else {
    final aData = a.data as List<int>;
    if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
      _elementWiseOp<int, Complex, int>(
        rData,
        aData,
        b.data as List<Complex>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else if (b.dtype == DType.float64 || b.dtype == DType.float32) {
      _elementWiseOp<int, double, int>(
        rData,
        aData,
        b.data as List<double>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else if (b.dtype == DType.boolean) {
      _elementWiseOp<int, bool, int>(
        rData,
        aData,
        b.data as List<bool>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else {
      _elementWiseOp<int, int, int>(
        rData,
        aData,
        b.data as List<int>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    }
  }
}

/// Returns a sorted copy of an array along a specified [axis].
///
/// This function corresponds to NumPy's `sort` function.
///
/// It uses native ANSI C `qsort` via FFI to perform zero-copy, high-speed
/// in-place sorting straight on the C heap for contiguous last-axis rows, completely
/// bypassing Dart memory marshalling.
///
/// Complex numbers are sorted lexicographically: by their real parts first,
/// and by their imaginary parts if the real parts are equal.
///
/// **Preconditions:**
/// - [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [RangeError] if [axis] is out of bounds.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
NDArray sort(NDArray a, {int axis = -1}) {
  final rank = a.shape.length;
  if (rank == 0) {
    return NDArray.fromList(List.from(a.data), [], a.dtype);
  }

  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(targetAxis, 0, rank - 1, 'axis');
  }

  if (targetAxis != rank - 1) {
    final swappedView = a.swapaxes(targetAxis, rank - 1);
    final sortedView = sort(swappedView, axis: rank - 1);
    return sortedView.swapaxes(targetAxis, rank - 1);
  }

  NDArray src = a;
  if (!a.isContiguous) {
    src = NDArray.fromList(a.toList(), a.shape, a.dtype);
  }

  final result = NDArray.create(src.shape, src.dtype);
  if (src.dtype == DType.float64) {
    (result.data as Float64List).setRange(
      0,
      src.data.length,
      src.data as Float64List,
    );
  } else if (src.dtype == DType.float32) {
    (result.data as Float32List).setRange(
      0,
      src.data.length,
      src.data as Float32List,
    );
  } else if (src.dtype == DType.int64) {
    (result.data as Int64List).setRange(
      0,
      src.data.length,
      src.data as Int64List,
    );
  } else if (src.dtype == DType.int32) {
    (result.data as Int32List).setRange(
      0,
      src.data.length,
      src.data as Int32List,
    );
  } else if (src.dtype == DType.complex128 || src.dtype == DType.complex64) {
    final srcBacking = (src.data as ComplexList).backingList;
    final resBacking = (result.data as ComplexList).backingList;
    if (srcBacking is Float64List && resBacking is Float64List) {
      resBacking.setRange(0, srcBacking.length, srcBacking);
    } else if (srcBacking is Float32List && resBacking is Float32List) {
      resBacking.setRange(0, srcBacking.length, srcBacking);
    }
  } else if (src.dtype == DType.boolean) {
    final srcBacking = (src.data as BoolList).backingList;
    final resBacking = (result.data as BoolList).backingList;
    resBacking.setRange(0, srcBacking.length, srcBacking);
  }

  final n = src.shape.last;
  final totalSize = src.shape.isEmpty ? 1 : src.shape.reduce((x, y) => x * y);
  final numRows = totalSize ~/ n;

  int elementSizeInBytes;
  if (src.dtype == DType.float64 || src.dtype == DType.int64) {
    elementSizeInBytes = 8;
  } else if (src.dtype == DType.float32 || src.dtype == DType.int32) {
    elementSizeInBytes = 4;
  } else if (src.dtype == DType.complex64) {
    elementSizeInBytes = 8;
  } else if (src.dtype == DType.complex128) {
    elementSizeInBytes = 16;
  } else {
    throw UnimplementedError('Unsupported dtype for sort: ${src.dtype}');
  }

  final baseCast = result.pointer.cast<ffi.Uint8>();
  final rowSizeInBytes = n * elementSizeInBytes;

  for (var r = 0; r < numRows; r++) {
    final rowPtr = baseCast + (r * rowSizeInBytes);

    // High-speed direct C sorters bypassing FFI context switches
    if (src.dtype == DType.float64) {
      native_sort_double(rowPtr.cast<ffi.Double>(), n);
    } else if (src.dtype == DType.float32) {
      native_sort_float(rowPtr.cast<ffi.Float>(), n);
    } else if (src.dtype == DType.int64) {
      native_sort_int64(rowPtr.cast<ffi.LongLong>(), n);
    } else if (src.dtype == DType.int32) {
      native_sort_int32(rowPtr.cast<ffi.Int>(), n);
    } else if (src.dtype == DType.complex128) {
      native_sort_complex128(rowPtr.cast<ffi.Double>(), n);
    } else if (src.dtype == DType.complex64) {
      native_sort_complex64(rowPtr.cast<ffi.Float>(), n);
    }
  }

  return result;
}

/// Returns the indices that would sort an array along a specified [axis].
///
/// This function corresponds to NumPy's `argsort` function.
///
/// It performs indirect index sorting, fully supporting complex numbers
/// lexicographical logic and internal axes reorientation via the axis-swapping pipeline.
///
/// **Preconditions:**
/// - [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [RangeError] if [axis] is out of bounds.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
NDArray<int> argsort(NDArray a, {int axis = -1}) {
  final rank = a.shape.length;
  if (rank == 0) {
    return NDArray.fromList(<int>[0], [], DType.int32);
  }

  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(targetAxis, 0, rank - 1, 'axis');
  }

  if (targetAxis != rank - 1) {
    final swappedView = a.swapaxes(targetAxis, rank - 1);
    final sortedIndicesView = argsort(swappedView, axis: rank - 1);
    return sortedIndicesView.swapaxes(targetAxis, rank - 1);
  }

  NDArray src = a;
  bool needsDispose = false;
  if (!a.isContiguous) {
    src = NDArray.create(a.shape, a.dtype);
    src.data.setRange(0, src.data.length, a.toList());
    needsDispose = true;
  }

  try {
    final n = src.shape.last;
    final totalSize = src.shape.isEmpty ? 1 : src.shape.reduce((x, y) => x * y);
    final numRows = totalSize ~/ n;

    final result = NDArray<int>.create(src.shape, DType.int32);

    if (src.dtype == DType.float64) {
      final dataPtr = src.pointer.cast<ffi.Double>();
      final resPtr = result.pointer.cast<ffi.Int>();
      for (var r = 0; r < numRows; r++) {
        native_argsort_double(dataPtr + r * n, resPtr + r * n, n);
      }
      return result;
    } else if (src.dtype == DType.float32) {
      final dataPtr = src.pointer.cast<ffi.Float>();
      final resPtr = result.pointer.cast<ffi.Int>();
      for (var r = 0; r < numRows; r++) {
        native_argsort_float(dataPtr + r * n, resPtr + r * n, n);
      }
      return result;
    } else if (src.dtype == DType.int64) {
      final dataPtr = src.pointer.cast<ffi.LongLong>();
      final resPtr = result.pointer.cast<ffi.Int>();
      for (var r = 0; r < numRows; r++) {
        native_argsort_int64(dataPtr + r * n, resPtr + r * n, n);
      }
      return result;
    } else if (src.dtype == DType.int32) {
      final dataPtr = src.pointer.cast<ffi.Int>();
      final resPtr = result.pointer.cast<ffi.Int>();
      for (var r = 0; r < numRows; r++) {
        native_argsort_int32(dataPtr + r * n, resPtr + r * n, n);
      }
      return result;
    }

    for (var r = 0; r < numRows; r++) {
      final rowStart = r * n;
      final indices = List<int>.generate(n, (i) => i);

      if (src.dtype == DType.complex128 || src.dtype == DType.complex64) {
        final dataList = src.data as List<Complex>;
        indices.sort((i, j) {
          final cA = dataList[rowStart + i];
          final cB = dataList[rowStart + j];
          if (cA.real != cB.real) return cA.real.compareTo(cB.real);
          return cA.imag.compareTo(cB.imag);
        });
      } else {
        throw UnimplementedError('Unsupported dtype for argsort: ${src.dtype}');
      }

      for (var i = 0; i < n; i++) {
        result.data[rowStart + i] = indices[i];
      }
    }

    return result;
  } finally {
    if (needsDispose) {
      src.dispose();
    }
  }
}

void _whereOpRec<Tc, Tx, Ty, Tr>(
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
    final isTrue = _isTrue(cVal);
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
    _whereOpRec<Tc, Tx, Ty, Tr>(
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

void _dispatchWhere(
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
    _whereOpRec<dynamic, dynamic, dynamic, Complex>(
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
    _whereOpRec<dynamic, dynamic, dynamic, double>(
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
    _whereOpRec<dynamic, dynamic, dynamic, int>(
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
List<int> _broadcastStrides(NDArray a, List<int> targetShape) {
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

List<int> _broadcast3Shapes(List<int> s1, List<int> s2, List<int> s3) {
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

dynamic where(NDArray condition, [NDArray? x, NDArray? y, NDArray? out]) {
  if (x == null && y == null) {
    if (out != null) {
      throw ArgumentError(
        'out buffer cannot be provided when x and y are omitted.',
      );
    }
    return nonzero(condition);
  }

  if ((x == null && y != null) || (x != null && y == null)) {
    throw ArgumentError('Either both or neither of x and y must be given');
  }

  // Calculate target common shape via high-speed 3-way broadcast matching
  final commonShape = _broadcast3Shapes(condition.shape, x!.shape, y!.shape);

  final DType<dynamic> targetDType = _resolveDType(x.dtype, y.dtype);

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for where() result.',
      );
    }
  }

  // Compute precise broadcasted strides for each operand independently to commonShape
  final stridesCond = _broadcastStrides(condition, commonShape);
  final stridesX = _broadcastStrides(x, commonShape);
  final stridesY = _broadcastStrides(y, commonShape);

  final result = out ?? NDArray.create(commonShape, targetDType);
  final resultStrides = NDArray.computeCStrides(commonShape);

  // 0. Advanced ND Odometer Ternary Broadcasting Engine in C (Rank <= 8)
  if (commonShape.length <= 8 && condition.dtype == DType.boolean) {
    final cShape = malloc<ffi.Int>(commonShape.length);
    final cStridesCond = malloc<ffi.Int>(stridesCond.length);
    final cStridesX = malloc<ffi.Int>(stridesX.length);
    final cStridesY = malloc<ffi.Int>(stridesY.length);
    final cStridesRes = malloc<ffi.Int>(resultStrides.length);

    for (var i = 0; i < commonShape.length; i++) {
      cShape[i] = commonShape[i];
      cStridesCond[i] = stridesCond[i];
      cStridesX[i] = stridesX[i];
      cStridesY[i] = stridesY[i];
      cStridesRes[i] = resultStrides[i];
    }

    try {
      if (targetDType == DType.float64 &&
          x.dtype == DType.float64 &&
          y.dtype == DType.float64) {
        s_where_double(
          condition.pointer.cast(),
          cStridesCond,
          x.pointer.cast(),
          cStridesX,
          y.pointer.cast(),
          cStridesY,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
        );
        return result;
      } else if (targetDType == DType.float32 &&
          x.dtype == DType.float32 &&
          y.dtype == DType.float32) {
        s_where_float(
          condition.pointer.cast(),
          cStridesCond,
          x.pointer.cast(),
          cStridesX,
          y.pointer.cast(),
          cStridesY,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
        );
        return result;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesCond);
      malloc.free(cStridesX);
      malloc.free(cStridesY);
      malloc.free(cStridesRes);
    }
  }

  _dispatchWhere(
    result.data,
    condition,
    x,
    y,
    commonShape,
    stridesCond,
    stridesX,
    stridesY,
    resultStrides,
  );

  return result;
}

/// Returns the indices of the elements that are non-zero.
///
/// Returns a `List<NDArray<int>>` containing 1D integer arrays, one for each dimension
/// of [a], which give the coordinates of the non-zero elements along that dimension.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
List<NDArray<int>> nonzero(NDArray a) {
  final rank = a.shape.length;
  final count = count_nonzero(a) as int;

  final results = List.generate(
    rank,
    (_) => NDArray<int>.create([count], DType.int32, zeroInit: true),
  );

  if (count == 0) {
    return results;
  }

  final shape = a.shape;
  final strides = a.strides;
  final totalSize = shape.isEmpty ? 1 : shape.reduce((x, y) => x * y);

  final coord = List<int>.filled(rank, 0);
  int offset = 0;
  int writeIdx = 0;

  for (int el = 0; el < totalSize; el++) {
    final val = a.data[offset];
    if (_isTrue(val)) {
      for (var d = 0; d < rank; d++) {
        results[d].data[writeIdx] = coord[d];
      }
      writeIdx++;
    }

    // Advance odometer multidimensional coordinate odometer walk!
    for (int d = rank - 1; d >= 0; d--) {
      coord[d]++;
      if (coord[d] < shape[d]) {
        offset += strides[d];
        break;
      }
      coord[d] = 0;
      offset -= (shape[d] - 1) * strides[d];
    }
  }

  return results;
}

/// Count the number of non-zero elements in the array [a].
///
/// If [axis] is provided, counts along that axis and returns a new array.
/// Otherwise, counts all elements globally and returns a single scalar integer.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
dynamic count_nonzero(NDArray a, {int? axis}) {
  if (axis == null) {
    var count = 0;
    if (a.isContiguous) {
      for (var i = 0; i < a.data.length; i++) {
        if (_isTrue(a.data[i])) count++;
      }
      return count;
    }
    final rank = a.shape.length;
    final pos = List<int>.filled(rank, 0);
    int countWalk(int dim) {
      if (dim == rank) return _isTrue(a[pos]) ? 1 : 0;
      var subCount = 0;
      for (var i = 0; i < a.shape[dim]; i++) {
        pos[dim] = i;
        subCount += countWalk(dim + 1);
      }
      return subCount;
    }

    return countWalk(0);
  }

  final rank = a.shape.length;
  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(targetAxis, 0, rank - 1, 'axis');
  }

  final targetShape = List<int>.from(a.shape)..removeAt(targetAxis);
  final result = NDArray<int>.zeros(targetShape, DType.int32);

  _countNonzeroRecursive(a, result, List<int>.filled(rank, 0), targetAxis, 0);

  return result;
}

void _countNonzeroRecursive(
  NDArray src,
  NDArray<int> dest,
  List<int> srcPos,
  int targetAxis,
  int currentDim,
) {
  if (currentDim == src.shape.length) {
    if (_isTrue(src[srcPos])) {
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
    _countNonzeroRecursive(src, dest, srcPos, targetAxis, currentDim + 1);
  }
}

/// Returns the indices of the maximum values along an [axis].
///
/// If [axis] is null, flattens the array and returns a flat scalar integer index.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
dynamic argmax(NDArray a, {int? axis}) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for argmax');
  }

  if (axis == null) {
    NDArray src = a;
    if (!a.isContiguous) {
      src = NDArray.fromList(a.toList(), a.shape, a.dtype);
    }

    var maxIdx = 0;
    if (src.dtype == DType.int32 || src.dtype == DType.int64) {
      final dataList = src.data as List<int>;
      var maxVal = dataList[0];
      for (var i = 1; i < dataList.length; i++) {
        if (dataList[i] > maxVal) {
          maxVal = dataList[i];
          maxIdx = i;
        }
      }
    } else {
      final dataList = src.data as List<double>;
      var maxVal = dataList[0];
      for (var i = 1; i < dataList.length; i++) {
        if (dataList[i] > maxVal) {
          maxVal = dataList[i];
          maxIdx = i;
        }
      }
    }
    return maxIdx;
  }

  final rank = a.shape.length;
  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(targetAxis, 0, rank - 1, 'axis');
  }

  final targetShape = List<int>.from(a.shape)..removeAt(targetAxis);
  final result = NDArray<int>.create(targetShape, DType.int32);
  result.data.fillRange(0, result.data.length, 0);

  _argMinMaxRecursive(
    a,
    result,
    List<int>.filled(rank, 0),
    targetAxis,
    0,
    true,
  );
  return result;
}

/// Returns the indices of the minimum values along an [axis].
///
/// If [axis] is null, flattens the array and returns a flat scalar integer index.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
dynamic argmin(NDArray a, {int? axis}) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for argmin');
  }

  if (axis == null) {
    NDArray src = a;
    if (!a.isContiguous) {
      src = NDArray.fromList(a.toList(), a.shape, a.dtype);
    }

    var minIdx = 0;
    if (src.dtype == DType.int32 || src.dtype == DType.int64) {
      final dataList = src.data as List<int>;
      var minVal = dataList[0];
      for (var i = 1; i < dataList.length; i++) {
        if (dataList[i] < minVal) {
          minVal = dataList[i];
          minIdx = i;
        }
      }
    } else {
      final dataList = src.data as List<double>;
      var minVal = dataList[0];
      for (var i = 1; i < dataList.length; i++) {
        if (dataList[i] < minVal) {
          minVal = dataList[i];
          minIdx = i;
        }
      }
    }
    return minIdx;
  }

  final rank = a.shape.length;
  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(targetAxis, 0, rank - 1, 'axis');
  }

  final targetShape = List<int>.from(a.shape)..removeAt(targetAxis);
  final result = NDArray<int>.create(targetShape, DType.int32);
  result.data.fillRange(0, result.data.length, 0);

  _argMinMaxRecursive(
    a,
    result,
    List<int>.filled(rank, 0),
    targetAxis,
    0,
    false,
  );
  return result;
}

void _argMinMaxRecursive(
  NDArray src,
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
    _argMinMaxRecursive(src, dest, srcPos, targetAxis, currentDim + 1, isMax);
  }
}

/// Compute the Cholesky decomposition of a square 2D matrix.
///
/// Factorizes a symmetric, positive-definite matrix [a] out `A = L * L^T`, where `L`
/// is a lower triangular matrix factor.
///
/// Returns a Map `{'L': lMatrix}` containing the lower factor.
///
/// **Example:**
/// {@example /example/linalg_example.dart lang=dart}
Map<String, NDArray> cholesky(NDArray a) {
  if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
    throw ArgumentError('Matrix must be square and 2D (was ${a.shape})');
  }
  final n = a.shape[0];
  final DType<dynamic> targetDType = a.dtype == DType.float32
      ? DType.float32
      : DType.float64;

  final NDArray src;
  final bool wasCopied;
  if (!a.isContiguous) {
    src = NDArray.fromList(a.toList(), a.shape, a.dtype);
    wasCopied = true;
  } else {
    src = a;
    wasCopied = false;
  }

  // Create result matrix L and copy elements from src
  final lMat = targetDType == DType.float32
      ? NDArray<Float32>.zeros([n, n], DType.float32)
      : NDArray<Float64>.zeros([n, n], DType.float64);

  try {
    if (src.dtype == DType.float64) {
      (lMat.data as Float64List).setRange(0, n * n, src.data as Float64List);
    } else if (src.dtype == DType.float32) {
      (lMat.data as Float32List).setRange(0, n * n, src.data as Float32List);
    } else {
      for (var i = 0; i < n * n; i++) {
        lMat.data[i] = (src.data[i] as num).toDouble();
      }
    }

    // Char 'L' in ASCII is 76
    const uploL = 76;

    if (targetDType == DType.float64) {
      final info = LAPACKE_dpotrf(
        101, // ROW_MAJOR
        uploL,
        n,
        lMat.pointer.cast<ffi.Double>(),
        n,
      );
      if (info < 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_dpotrf: $info');
      }
      if (info > 0) {
        throw ArgumentError(
          'Matrix must be symmetric positive-definite for Cholesky decomposition',
        );
      }

      // Zero-out strictly upper triangular part
      final lData = lMat.data as Float64List;
      for (var i = 0; i < n; i++) {
        for (var j = i + 1; j < n; j++) {
          lData[i * n + j] = 0.0;
        }
      }
    } else {
      final info = LAPACKE_spotrf(
        101, // ROW_MAJOR
        uploL,
        n,
        lMat.pointer.cast<ffi.Float>(),
        n,
      );
      if (info < 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_spotrf: $info');
      }
      if (info > 0) {
        throw ArgumentError(
          'Matrix must be symmetric positive-definite for Cholesky decomposition',
        );
      }

      // Zero-out strictly upper triangular part
      final lData = lMat.data as Float32List;
      for (var i = 0; i < n; i++) {
        for (var j = i + 1; j < n; j++) {
          lData[i * n + j] = 0.0;
        }
      }
    }
  } finally {
    if (wasCopied) {
      src.dispose();
    }
  }

  return {'L': lMat};
}

/// Computes the QR decomposition of a matrix $A = Q R$.
///
/// Decomposes a matrix [a] out an orthogonal matrix `Q` and an upper triangular matrix `R`
/// such that `a = Q * R`.
/// Natively offloads to LAPACK solvers (`dgeqrf` / `sgeqrf` and `dorgqr` / `sorgqr`) depending on precision.
///
/// **Preconditions:**
/// - Input matrix [a] must be 2-dimensional.
///
/// **Throws:**
/// - [ArgumentError] if [a] is not 2D.
/// - [StateError] if native FFI memory allocation or LAPACK solver initialization fails.
///
/// **Performance considerations:**
/// - Executes at high-speed natively in unmanaged C space.
///
/// **Example:**
/// ```dart
/// final a = `NDArray<double>`.fromList([12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0], [3, 3], DType.float64);
/// final res = qr(a);
/// final q = res['Q']!;
/// final r = res['R']!;
/// ```
Map<String, NDArray> qr(NDArray a) {
  if (a.shape.length != 2) {
    throw ArgumentError('Matrix must be 2D (was ${a.shape})');
  }
  final m = a.shape[0];
  final n = a.shape[1];
  final k = m < n ? m : n;
  final DType<dynamic> targetDType = a.dtype == DType.float32
      ? DType.float32
      : DType.float64;

  final aCopy = NDArray.create([m, n], targetDType);
  if (a.isContiguous && a.dtype == targetDType) {
    if (targetDType == DType.float64) {
      aCopy.pointer
          .cast<ffi.Double>()
          .asTypedList(m * n)
          .setRange(0, m * n, a.data as List<double>);
    } else {
      aCopy.pointer
          .cast<ffi.Float>()
          .asTypedList(m * n)
          .setRange(0, m * n, a.data as List<double>);
    }
  } else {
    final flat = a.toList();
    if (targetDType == DType.float64) {
      aCopy.pointer
          .cast<ffi.Double>()
          .asTypedList(m * n)
          .setRange(0, m * n, flat.cast<double>());
    } else {
      aCopy.pointer
          .cast<ffi.Float>()
          .asTypedList(m * n)
          .setRange(0, m * n, flat.cast<double>());
    }
  }

  final rMat = targetDType == DType.float32
      ? NDArray<Float32>.zeros([k, n], DType.float32)
      : NDArray<Float64>.zeros([k, n], DType.float64);
  final qMat = targetDType == DType.float32
      ? NDArray<Float32>.zeros([m, k], DType.float32)
      : NDArray<Float64>.zeros([m, k], DType.float64);

  if (targetDType == DType.float64) {
    final tau = malloc<ffi.Double>(k);
    try {
      final info = LAPACKE_dgeqrf(
        101, // ROW_MAJOR
        m,
        n,
        aCopy.pointer.cast<ffi.Double>(),
        n,
        tau,
      );
      if (info != 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_dgeqrf: $info');
      }

      // Extract upper triangular matrix R from aCopy
      final rData = rMat.data;
      final aCopyData = aCopy.data as List<double>;
      for (var i = 0; i < k; i++) {
        for (var j = i; j < n; j++) {
          rData[i * n + j] = aCopyData[i * n + j];
        }
      }

      // Copy reflectors to qMat (first k columns of aCopy)
      final qData = qMat.data;
      for (var i = 0; i < m; i++) {
        for (var j = 0; j < k; j++) {
          qData[i * k + j] = aCopyData[i * n + j];
        }
      }

      // Call dorgqr to reconstruct the orthonormal columns of Q in-place in qMat
      final infoOrg = LAPACKE_dorgqr(
        101, // ROW_MAJOR
        m,
        k,
        k,
        qMat.pointer.cast<ffi.Double>(),
        k,
        tau,
      );
      if (infoOrg != 0) {
        throw ArgumentError(
          'Illegal value in call to LAPACKE_dorgqr: $infoOrg',
        );
      }
    } finally {
      malloc.free(tau);
      aCopy.dispose();
    }
  } else {
    final tau = malloc<ffi.Float>(k);
    try {
      final info = LAPACKE_sgeqrf(
        101, // ROW_MAJOR
        m,
        n,
        aCopy.pointer.cast<ffi.Float>(),
        n,
        tau,
      );
      if (info != 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_sgeqrf: $info');
      }

      // Extract upper triangular matrix R from aCopy
      final rData = rMat.data;
      final aCopyData = aCopy.data as List<double>;
      for (var i = 0; i < k; i++) {
        for (var j = i; j < n; j++) {
          rData[i * n + j] = aCopyData[i * n + j];
        }
      }

      // Copy reflectors to qMat (first k columns of aCopy)
      final qData = qMat.data;
      for (var i = 0; i < m; i++) {
        for (var j = 0; j < k; j++) {
          qData[i * k + j] = aCopyData[i * n + j];
        }
      }

      // Call sorgqr to reconstruct the orthonormal columns of Q in-place in qMat
      final infoOrg = LAPACKE_sorgqr(
        101, // ROW_MAJOR
        m,
        k,
        k,
        qMat.pointer.cast<ffi.Float>(),
        k,
        tau,
      );
      if (infoOrg != 0) {
        throw ArgumentError(
          'Illegal value in call to LAPACKE_sorgqr: $infoOrg',
        );
      }
    } finally {
      malloc.free(tau);
      aCopy.dispose();
    }
  }

  return {'Q': qMat, 'R': rMat};
}

/// Computes the Singular Value Decomposition (SVD) of a matrix $A = U S V^h$.
///
/// Decomposes a matrix [a] out left singular vectors `U`, singular values `S`,
/// and right singular vectors Vh such that `a = U * diag(S) * Vh`.
/// Natively offloads to LAPACK solvers (`dgesdd` / `sgesdd`) depending on precision.
///
/// **Preconditions:**
/// - Input matrix [a] must be 2-dimensional.
///
/// **Throws:**
/// - [ArgumentError] if [a] is not 2D.
/// - [StateError] if native FFI memory allocation or LAPACK solver initialization fails.
///
/// **Performance considerations:**
/// - Executes at high-speed natively in unmanaged C space.
///
/// **Example:**
/// ```dart
/// final a = `NDArray<double>`.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], DType.float64);
/// final res = svd(a);
/// final u = res['U']!;
/// final s = res['S']!;
/// final vh = res['Vh']!;
/// ```
Map<String, NDArray> svd(NDArray a) {
  if (a.shape.length != 2) {
    throw ArgumentError('Matrix must be 2D (was ${a.shape})');
  }
  final m = a.shape[0];
  final n = a.shape[1];
  final DType<dynamic> targetDType = a.dtype == DType.float32
      ? DType.float32
      : DType.float64;

  if (m < n) {
    final aT = a.transpose();
    final resT = svd(aT);
    final uNew = resT['U']!;
    final sNew = resT['S']!;
    final vhNew = resT['Vh']!;

    final uResult = vhNew.transpose();
    final vhResult = uNew.transpose();

    return {'U': uResult, 'S': sNew, 'Vh': vhResult};
  }

  final aCopy = NDArray.create([m, n], targetDType);
  if (a.isContiguous && a.dtype == targetDType) {
    if (targetDType == DType.float64) {
      aCopy.pointer
          .cast<ffi.Double>()
          .asTypedList(m * n)
          .setRange(0, m * n, a.data as List<double>);
    } else {
      aCopy.pointer
          .cast<ffi.Float>()
          .asTypedList(m * n)
          .setRange(0, m * n, a.data as List<double>);
    }
  } else {
    final flat = a.toList();
    if (targetDType == DType.float64) {
      aCopy.pointer
          .cast<ffi.Double>()
          .asTypedList(m * n)
          .setRange(0, m * n, flat.cast<double>());
    } else {
      aCopy.pointer
          .cast<ffi.Float>()
          .asTypedList(m * n)
          .setRange(0, m * n, flat.cast<double>());
    }
  }

  final sMat = targetDType == DType.float32
      ? NDArray<Float32>.zeros([n], DType.float32)
      : NDArray<Float64>.zeros([n], DType.float64);
  final uMat = targetDType == DType.float32
      ? NDArray<Float32>.zeros([m, m], DType.float32)
      : NDArray<Float64>.zeros([m, m], DType.float64);
  final vtMat = targetDType == DType.float32
      ? NDArray<Float32>.zeros([n, n], DType.float32)
      : NDArray<Float64>.zeros([n, n], DType.float64);

  if (targetDType == DType.float64) {
    final superb = malloc<ffi.Double>(math.max(1, n - 1));
    try {
      final info = LAPACKE_dgesvd(
        101, // ROW_MAJOR
        65, // 'A'
        65, // 'A'
        m,
        n,
        aCopy.pointer.cast<ffi.Double>(),
        n,
        sMat.pointer.cast<ffi.Double>(),
        uMat.pointer.cast<ffi.Double>(),
        m,
        vtMat.pointer.cast<ffi.Double>(),
        n,
        superb,
      );
      if (info != 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_dgesvd: $info');
      }
    } finally {
      malloc.free(superb);
      aCopy.dispose();
    }
  } else {
    final superb = malloc<ffi.Float>(math.max(1, n - 1));
    try {
      final info = LAPACKE_sgesvd(
        101, // ROW_MAJOR
        65, // 'A'
        65, // 'A'
        m,
        n,
        aCopy.pointer.cast<ffi.Float>(),
        n,
        sMat.pointer.cast<ffi.Float>(),
        uMat.pointer.cast<ffi.Float>(),
        m,
        vtMat.pointer.cast<ffi.Float>(),
        n,
        superb,
      );
      if (info != 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_sgesvd: $info');
      }
    } finally {
      malloc.free(superb);
      aCopy.dispose();
    }
  }

  return {'U': uMat, 'S': sMat, 'Vh': vtMat};
}

/// Extract a diagonal or construct a diagonal array.
///
/// If [v] is a 2D matrix, extracts the k-th diagonal elements vector as a zero-copy 1D view.
/// If [v] is a 1D vector, constructs a 2D square matrix with [v] as the k-th diagonal and zeros elsewhere.
///
/// **Preconditions:**
/// - Input [v] must be a 1D or 2D array.
///
/// **Throws:**
/// - [ArgumentError] if [v] rank is not 1 or 2.
///
/// **Example:**
/// {@example /example/diag_example.dart lang=dart}
///
/// Reference: [Diagonal Matrix](https://en.wikipedia.org/wiki/Diagonal_matrix)
NDArray<T> diag<T>(NDArray<T> v, {int k = 0, NDArray<T>? out}) {
  if (v.shape.length == 2) {
    final m = v.shape[0];
    final n = v.shape[1];

    int startRow;
    int startCol;
    int len;

    if (k >= 0) {
      startRow = 0;
      startCol = k;
      if (startCol >= n) {
        return NDArray<T>.create([0], v.dtype);
      }
      len = math.min(m, n - k);
    } else {
      startRow = -k;
      startCol = 0;
      if (startRow >= m) {
        return NDArray<T>.create([0], v.dtype);
      }
      len = math.min(m + k, n);
    }

    if (len <= 0) {
      return NDArray<T>.create([0], v.dtype);
    }

    final offsetElements = startRow * v.strides[0] + startCol * v.strides[1];
    final diagStride = v.strides[0] + v.strides[1];

    return NDArray<T>.view(
      v,
      shape: [len],
      strides: [diagStride],
      offsetElements: offsetElements,
    );
  } else if (v.shape.length == 1) {
    final n = v.shape[0];
    final size = n + k.abs();
    final targetShape = [size, size];

    final result = out ?? NDArray<T>.zeros(targetShape, v.dtype);
    if (out != null) {
      if (!listEquals(out.shape, targetShape) || out.dtype != v.dtype) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
      for (var i = 0; i < result.data.length; i++) {
        result.data[i] = _castValue(0, v.dtype) as T;
      }
    }

    int startRow;
    int startCol;

    if (k >= 0) {
      startRow = 0;
      startCol = k;
    } else {
      startRow = -k;
      startCol = 0;
    }

    final vList = v.toList();
    final resData = result.data;
    final resStrides = result.strides;

    for (var i = 0; i < n; i++) {
      final targetIdx =
          (startRow + i) * resStrides[0] + (startCol + i) * resStrides[1];
      resData[targetIdx] = vList[i];
    }

    return result;
  } else {
    throw ArgumentError('Input array must be 1- or 2-dimensional.');
  }
}

/// Returns a boolean [NDArray] where two arrays are element-wise equal within a tolerance.
///
/// The tolerance relation is defined as:
/// `abs(a - b) <= (atol + rtol * abs(b))`
///
/// **Preconditions:**
/// - Input [a] and [b] must be numeric arrays.
/// - [a] and [b] must have compatible broadcast shapes.
///
/// **Example:**
/// {@example /example/isclose_example.dart lang=dart}
///
/// Reference: [Approximate Equality](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)
NDArray<bool> isclose(
  NDArray a,
  NDArray b, {
  double rtol = 1e-05,
  double atol = 1e-08,
  bool equalNan = false,
}) {
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  final size = commonShape.isEmpty ? 1 : commonShape.reduce((x, y) => x * y);
  final result = NDArray<bool>.zeros(commonShape, DType.boolean);

  final aList = a.toList().cast<num>();
  final bList = b.toList().cast<num>();
  final resData = result.data;

  final broadcastResultRes = broadcast(a, b);
  final stridesA = broadcastResultRes.stridesA;
  final stridesB = broadcastResultRes.stridesB;
  final stridesRes = NDArray.computeCStrides(commonShape);

  final coord = List<int>.filled(commonShape.length, 0);

  for (var el = 0; el < size; el++) {
    var offsetA = 0;
    var offsetB = 0;
    var offsetRes = 0;
    for (var d = 0; d < commonShape.length; d++) {
      offsetA += coord[d] * stridesA[d];
      offsetB += coord[d] * stridesB[d];
      offsetRes += coord[d] * stridesRes[d];
    }

    final valA = aList[offsetA].toDouble();
    final valB = bList[offsetB].toDouble();

    var match = false;
    if (equalNan && valA.isNaN && valB.isNaN) {
      match = true;
    } else if (valA.isInfinite || valB.isInfinite) {
      match = valA == valB;
    } else {
      final diff = (valA - valB).abs();
      final limit = atol + rtol * valB.abs();
      match = diff <= limit;
    }

    resData[offsetRes] = match;

    // Advance coord odometer
    for (var d = commonShape.length - 1; d >= 0; d--) {
      coord[d]++;
      if (coord[d] < commonShape[d]) break;
      coord[d] = 0;
    }
  }

  return result;
}

/// Returns true if two arrays are element-wise equal within a tolerance.
///
/// The tolerance relation is defined as:
/// `abs(a - b) <= (atol + rtol * abs(b))`
///
/// **Preconditions:**
/// - Input [a] and [b] must be numeric arrays.
/// - [a] and [b] must have compatible broadcast shapes.
///
/// **Example:**
/// {@example /example/isclose_example.dart lang=dart}
///
/// Reference: [Approximate Equality](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html)
bool allclose(
  NDArray a,
  NDArray b, {
  double rtol = 1e-05,
  double atol = 1e-08,
  bool equalNan = false,
}) {
  final closeMask = isclose(a, b, rtol: rtol, atol: atol, equalNan: equalNan);
  final maskList = closeMask.toList();
  closeMask.dispose();
  for (final val in maskList) {
    if (!val) return false;
  }
  return true;
}

/// Replace NaN with zero and infinity with large finite numbers.
///
/// By default, maps NaN to [nan] (which defaults to 0.0), maps positive infinity
/// to [posinf] (or the maximum finite float value if null), and maps negative infinity
/// to [neginf] (or the minimum finite float value if null).
///
/// **Preconditions:**
/// - Input [a] must be a numeric array.
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/nan_to_num_example.dart lang=dart}
///
/// Reference: [Replace NaN and Infinities](https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
NDArray nan_to_num(
  NDArray a, {
  double nan = 0.0,
  double? posinf,
  double? neginf,
  NDArray? out,
}) {
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for nan_to_num.',
      );
    }
  }

  final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
  final aList = a.toList();
  final resultCopy = out ?? NDArray.create(a.shape, a.dtype);

  final maxLimit = a.dtype == DType.float32
      ? 3.4028234663852886e+38
      : double.maxFinite;
  final minLimit = -maxLimit;

  final targetPosInf = posinf ?? maxLimit;
  final targetNegInf = neginf ?? minLimit;

  final cleanList = <dynamic>[];

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    final complexList = aList.cast<Complex>();
    for (var i = 0; i < size; i++) {
      var r = complexList[i].real;
      var img = complexList[i].imag;

      if (r.isNaN) r = nan;
      if (r == double.infinity) r = targetPosInf;
      if (r == double.negativeInfinity) r = targetNegInf;

      if (img.isNaN) img = nan;
      if (img == double.infinity) img = targetPosInf;
      if (img == double.negativeInfinity) img = targetNegInf;

      cleanList.add(Complex(r, img));
    }
  } else {
    final numList = aList.cast<num>();
    for (var i = 0; i < size; i++) {
      var val = numList[i].toDouble();

      if (val.isNaN) {
        val = nan;
      } else if (val == double.infinity) {
        val = targetPosInf;
      } else if (val == double.negativeInfinity) {
        val = targetNegInf;
      }

      cleanList.add(val);
    }
  }

  // View-Safe Strided Odometer Write Back!
  final resData = resultCopy.data;
  final resStrides = resultCopy.strides;
  final coord = List<int>.filled(a.shape.length, 0);

  for (var i = 0; i < size; i++) {
    var offsetRes = 0;
    for (var d = 0; d < a.shape.length; d++) {
      offsetRes += coord[d] * resStrides[d];
    }

    resData[offsetRes] = cleanList[i];

    // Advance odometer
    for (var d = a.shape.length - 1; d >= 0; d--) {
      coord[d]++;
      if (coord[d] < a.shape[d]) break;
      coord[d] = 0;
    }
  }

  return resultCopy;
}

/// Expand the shape of an array by inserting a new axis of size 1.
///
/// Inserts a new dimension of size 1 at the specified [axis] position.
/// If [axis] is negative, it is normalized relative to the rank of [a] plus 1.
///
/// **Preconditions:**
/// - [axis] normalized value must be between `0` and the rank of [a] inclusive.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Performance considerations:**
/// - This is a zero-copy view manipulation executing in absolute $O(1)$ time complexity.
///
/// **Example:**
/// {@example /example/shape_view_example.dart lang=dart}
///
/// Reference: [Expand Dimensions](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html)
NDArray expand_dims(NDArray a, int axis) {
  final rank = a.shape.length;
  var targetAxis = axis < 0 ? rank + 1 + axis : axis;

  if (targetAxis < 0 || targetAxis > rank) {
    throw ArgumentError(
      'Axis $axis is out of bounds for array of rank $rank (valid bounds: [${-rank - 1}, $rank])',
    );
  }

  final newShape = List<int>.from(a.shape);
  final newStrides = List<int>.from(a.strides);

  newShape.insert(targetAxis, 1);
  // Insert stride mapping: copy sibling stride or default to 1 if rank is 0
  final siblingStride = targetAxis < rank ? a.strides[targetAxis] : 1;
  newStrides.insert(targetAxis, siblingStride);

  return NDArray.view(
    a,
    shape: newShape,
    strides: newStrides,
    offsetElements: 0,
  );
}

/// Remove axes of size 1 from the shape of an array.
///
/// If [axis] is provided, only removes the specified dimensions (which must have size 1).
/// If [axis] is null, removes all dimensions of size 1.
///
/// **Preconditions:**
/// - Specified [axis] entries must indeed correspond to dimensions of size 1.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds or targets a dimension whose size is not 1.
///
/// **Performance considerations:**
/// - This is a zero-copy view manipulation executing in absolute $O(1)$ time complexity.
///
/// **Example:**
/// {@example /example/shape_view_example.dart lang=dart}
///
/// Reference: [Squeeze Dimensions](https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html)
NDArray squeeze(NDArray a, {List<int>? axis}) {
  final rank = a.shape.length;
  final shape = a.shape;
  final strides = a.strides;

  final newShape = <int>[];
  final newStrides = <int>[];

  final squeezeAxes = <int>{};
  if (axis != null) {
    for (final ax in axis) {
      final targetAx = ax < 0 ? rank + ax : ax;
      if (targetAx < 0 || targetAx >= rank) {
        throw ArgumentError('Axis $ax is out of bounds for rank $rank');
      }
      if (shape[targetAx] != 1) {
        throw ArgumentError(
          'Cannot squeeze axis $ax because its dimension size is ${shape[targetAx]} (must be 1)',
        );
      }
      squeezeAxes.add(targetAx);
    }
  } else {
    for (var i = 0; i < rank; i++) {
      if (shape[i] == 1) {
        squeezeAxes.add(i);
      }
    }
  }

  for (var i = 0; i < rank; i++) {
    if (!squeezeAxes.contains(i)) {
      newShape.add(shape[i]);
      newStrides.add(strides[i]);
    }
  }

  // Squeezing all dimensions of a 1D unit tensor (e.g. shape [1]) yields a 0D scalar shape []
  return NDArray.view(
    a,
    shape: newShape,
    strides: newStrides,
    offsetElements: 0,
  );
}

/// Create a sliding window view over several dimensions of an array.
///
/// This corresponds to NumPy's `lib.stride_tricks.sliding_window_view` function.
///
/// **Mathematical Mechanics**:
/// By manipulating strides, a sliding window of shape `windowShape` can be created
/// without copying any element data (completely zero-copy, copy-free, and zero-allocation).
///
/// For an input array `a` with shape $S = (s_0, \dots, s_{D-1})$ and strides $V = (v_0, \dots, v_{D-1})$:
/// - The axes to apply sliding windows are specified by [axis] (defaults to all axes).
/// - The output shape has the original dimensions reduced by `windowShape - 1`, with the window
///   dimensions appended at the end:
///   $$S_{\text{out}} = (s_0 - w_0 + 1, \dots, s_{k} - w_k + 1, \dots, w_0, \dots, w_k)$$
/// - The output strides has the original strides, with the original strides of the window axes appended at the end:
///   $$V_{\text{out}} = (v_0, \dots, v_{D-1}, v_{\text{axis}_0}, \dots, v_{\text{axis}_k})$$
///
/// **Preconditions:**
/// - [windowShape] length must match [axis] length.
/// - Each window dimension must be strictly positive and less than or equal to the corresponding axis size.
///
/// **Throws:**
/// - [ArgumentError] if axes are out of range, shapes mismatch, or window dimensions exceed axis sizes.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0, 5.0], [5], DType.float64);
/// final view = slidingWindowView(a, [3]);
/// print(view.shape); // [3, 3]
/// print(view.toList()); // [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]
/// ```
NDArray slidingWindowView(NDArray a, List<int> windowShape, {List<int>? axis}) {
  final rank = a.shape.length;

  // 1. Resolve and validate target axes
  final targetAxes = <int>[];
  if (axis != null) {
    for (var ax in axis) {
      final resolved = ax < 0 ? rank + ax : ax;
      if (resolved < 0 || resolved >= rank) {
        throw RangeError.range(resolved, 0, rank - 1, 'axis');
      }
      if (targetAxes.contains(resolved)) {
        throw ArgumentError(
          'Duplicate axis specified in sliding window: $resolved',
        );
      }
      targetAxes.add(resolved);
    }
  } else {
    // Default is all axes
    for (var i = 0; i < rank; i++) {
      targetAxes.add(i);
    }
  }

  if (windowShape.length != targetAxes.length) {
    throw ArgumentError(
      'windowShape length (${windowShape.length}) must match axes length (${targetAxes.length})',
    );
  }

  // 2. Calculate output shape and strides
  final outShape = List<int>.from(a.shape);
  final outStrides = List<int>.from(a.strides);

  for (var i = 0; i < targetAxes.length; i++) {
    final ax = targetAxes[i];
    final wSize = windowShape[i];
    final aSize = a.shape[ax];

    if (wSize <= 0) {
      throw ArgumentError(
        'windowShape dimensions must be strictly positive (was $wSize)',
      );
    }
    if (wSize > aSize) {
      throw ArgumentError(
        'windowShape dimension ($wSize) cannot exceed axis size ($aSize) for axis $ax',
      );
    }

    outShape[ax] = aSize - wSize + 1;
  }

  // Append window dimensions and their corresponding strides at the end
  for (var i = 0; i < targetAxes.length; i++) {
    final ax = targetAxes[i];
    outShape.add(windowShape[i]);
    outStrides.add(a.strides[ax]);
  }

  // 3. Return the zero-copy NDArray view sharing backing unmanaged C memory
  return NDArray.view(a, shape: outShape, strides: outStrides);
}

/// Compute the broadcasted shape list of two shapes.
List<int> broadcastShapes(List<int> s1, List<int> s2) {
  final len = math.max(s1.length, s2.length);
  final common = List<int>.filled(len, 1);
  for (var i = 0; i < len; i++) {
    final dim1 = s1.length - 1 - i >= 0 ? s1[s1.length - 1 - i] : 1;
    final dim2 = s2.length - 1 - i >= 0 ? s2[s2.length - 1 - i] : 1;

    final target = math.max(dim1, dim2);
    if (dim1 != target && dim1 != 1) {
      throw ArgumentError('Incompatible shapes for broadcasting');
    }
    if (dim2 != target && dim2 != 1) {
      throw ArgumentError('Incompatible shapes for broadcasting');
    }
    common[len - 1 - i] = target;
  }
  return common;
}

/// Return an array drawn from elements in [choicelist], depending on conditions in [condlist].
///
/// This corresponds to NumPy's `select` function.
///
/// **Mathematical Mechanics**:
/// - Evaluates a list of boolean conditions in [condlist] sequentially per cell.
/// - Draws corresponding values from the same-indexed array in [choicelist].
/// - If no condition is met, falls back to [defaultValue].
/// - Leverages zero-copy, zero-allocation $N$-dimensional strides recursive walk in a single pass!
///
/// **Preconditions:**
/// - [condlist] and [choicelist] must have the same length.
/// - All condition and choice arrays must broadcast perfectly to a common shape.
///
/// **Throws:**
/// - [ArgumentError] if [condlist] and [choicelist] lengths mismatch, or if any shape is incompatible.
///
/// **Example:**
/// ```dart
/// final cond1 = NDArray.fromList([true, false], [2], DType.boolean);
/// final cond2 = NDArray.fromList([false, true], [2], DType.boolean);
/// final choice1 = NDArray.fromList([10, 20], [2], DType.int32);
/// final choice2 = NDArray.fromList([100, 200], [2], DType.int32);
/// final res = select([cond1, cond2], [choice1, choice2], defaultValue: 999);
/// print(res.toList()); // [10, 200]
/// ```
NDArray select(
  List<NDArray<bool>> condlist,
  List<NDArray> choicelist, {
  dynamic defaultValue = 0,
}) {
  if (condlist.isEmpty || choicelist.isEmpty) {
    throw ArgumentError('condlist and choicelist must not be empty');
  }
  if (condlist.length != choicelist.length) {
    throw ArgumentError(
      'condlist length (${condlist.length}) must match choicelist length (${choicelist.length})',
    );
  }

  // 1. Calculate common broadcasted shape
  final allShapes = <List<int>>[];
  for (final c in condlist) {
    allShapes.add(c.shape);
  }
  for (final c in choicelist) {
    allShapes.add(c.shape);
  }

  var commonShape = allShapes[0];
  for (var i = 1; i < allShapes.length; i++) {
    commonShape = broadcastShapes(commonShape, allShapes[i]);
  }

  // 2. Determine target upcasted DType
  var targetDType = choicelist[0].dtype;
  for (var i = 1; i < choicelist.length; i++) {
    targetDType = _resolveDType(targetDType, choicelist[i].dtype);
  }
  if (defaultValue is double &&
      !targetDType.isFloating &&
      !targetDType.isComplex) {
    targetDType = DType.float64;
  }

  final result = NDArray.create(commonShape, targetDType);

  // 3. Compute strides for all condition and choice operands independently to commonShape
  final stridesCond = condlist
      .map((c) => _broadcastStrides(c, commonShape))
      .toList();
  final stridesChoice = choicelist
      .map((c) => _broadcastStrides(c, commonShape))
      .toList();
  final resultStrides = NDArray.computeCStrides(commonShape);

  // 4. Execute recursive multi-operand strided walk
  final currentPos = List<int>.filled(commonShape.length, 0);
  final initialOffsetsCond = List<int>.filled(condlist.length, 0);
  final initialOffsetsChoice = List<int>.filled(choicelist.length, 0);

  _selectRecursive(
    result,
    condlist,
    choicelist,
    stridesCond,
    stridesChoice,
    resultStrides,
    currentPos,
    0,
    initialOffsetsCond,
    initialOffsetsChoice,
    0,
    defaultValue,
  );

  return result;
}

void _selectRecursive(
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
        result.data[offsetRes] = _castValue(
          choicelist[j].data[offsetsChoice[j]],
          result.dtype,
        );
        chosen = true;
        break;
      }
    }
    if (!chosen) {
      result.data[offsetRes] = _castValue(defaultValue, result.dtype);
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
    _selectRecursive(
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

dynamic _castValue(dynamic val, DType dtype) {
  if (dtype == DType.complex128 || dtype == DType.complex64) {
    if (val is Complex) return val;
    if (val is num) return Complex(val.toDouble(), 0.0);
    return Complex(0.0, 0.0);
  }
  if (dtype == DType.float64 || dtype == DType.float32) {
    if (val is num) return val.toDouble();
    if (val is Complex) return val.real;
    return 0.0;
  }
  if (dtype == DType.int64 || dtype == DType.int32) {
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

/// Return the Hanning window.
///
/// The Hanning window is a taper formed by using a weighted cosine.
///
/// **Example:**
/// ```dart
/// final window = hanning(512);
/// ```
NDArray<double> hanning(int M, {DType dtype = DType.float64}) {
  if (M < 1) return NDArray.create([0], dtype as dynamic) as NDArray<double>;
  if (M == 1) {
    return NDArray.fromList([1.0], [1], dtype as dynamic) as NDArray<double>;
  }

  final res = NDArray.create([M], dtype as dynamic);
  for (var n = 0; n < M; n++) {
    res.data[n] = 0.5 - 0.5 * math.cos(2.0 * math.pi * n / (M - 1));
  }
  return res as NDArray<double>;
}

/// Return the Hamming window.
///
/// The Hamming window is a taper formed by using a weighted cosine.
///
/// **Example:**
/// ```dart
/// final window = hamming(512);
/// ```
NDArray<double> hamming(int M, {DType dtype = DType.float64}) {
  if (M < 1) return NDArray.create([0], dtype as dynamic) as NDArray<double>;
  if (M == 1) {
    return NDArray.fromList([1.0], [1], dtype as dynamic) as NDArray<double>;
  }

  final res = NDArray.create([M], dtype as dynamic);
  for (var n = 0; n < M; n++) {
    res.data[n] = 0.54 - 0.46 * math.cos(2.0 * math.pi * n / (M - 1));
  }
  return res as NDArray<double>;
}

/// Extract a lower triangular matrix (on and below the k-th diagonal) element-wise.
///
/// **Preconditions:**
/// - Input [a] must be an array with rank >= 2.
/// - If provided, the [out] recycler must have matching shape and dtype.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [a] has rank < 2.
/// - [ArgumentError] if [out] has mismatched shape or dtype.
///
/// **Example:**
/// {@example /example/triangular_example.dart lang=dart}
NDArray<T> tril<T>(NDArray<T> a, {int k = 0, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute tril() on a disposed array.');
  }
  if (a.shape.length < 2) {
    throw ArgumentError('Input array must have rank >= 2.');
  }
  final result = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final rank = a.shape.length;
  final rows = a.shape[rank - 2];
  final cols = a.shape[rank - 1];

  final batchCount = a.shape.isEmpty
      ? 1
      : a.shape.sublist(0, rank - 2).isEmpty
      ? 1
      : a.shape.sublist(0, rank - 2).reduce((x, y) => x * y);

  if (a.isContiguous && result.isContiguous) {
    if (a.dtype == DType.float64) {
      v_tril_double(
        a.pointer.cast(),
        result.pointer.cast(),
        batchCount,
        rows,
        cols,
        k,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      v_tril_float(
        a.pointer.cast(),
        result.pointer.cast(),
        batchCount,
        rows,
        cols,
        k,
      );
      return result;
    }
  }

  final aList = a.isContiguous ? a.data : a.toList();
  final resData = result.data;
  final matrixSize = rows * cols;

  for (var b = 0; b < batchCount; b++) {
    final offset = b * matrixSize;
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        final idx = offset + r * cols + c;
        resData[idx] = (c <= r + k) ? aList[idx] : _castValue(0, a.dtype) as T;
      }
    }
  }
  return result;
}

/// Extract an upper triangular matrix (on and above the k-th diagonal) element-wise.
///
/// **Preconditions:**
/// - Input [a] must be an array with rank >= 2.
/// - If provided, the [out] recycler must have matching shape and dtype.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [a] has rank < 2.
/// - [ArgumentError] if [out] has mismatched shape or dtype.
///
/// **Example:**
/// {@example /example/triangular_example.dart lang=dart}
NDArray<T> triu<T>(NDArray<T> a, {int k = 0, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute triu() on a disposed array.');
  }
  if (a.shape.length < 2) {
    throw ArgumentError('Input array must have rank >= 2.');
  }
  final result = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final rank = a.shape.length;
  final rows = a.shape[rank - 2];
  final cols = a.shape[rank - 1];

  final batchCount = a.shape.isEmpty
      ? 1
      : a.shape.sublist(0, rank - 2).isEmpty
      ? 1
      : a.shape.sublist(0, rank - 2).reduce((x, y) => x * y);

  if (a.isContiguous && result.isContiguous) {
    if (a.dtype == DType.float64) {
      v_triu_double(
        a.pointer.cast(),
        result.pointer.cast(),
        batchCount,
        rows,
        cols,
        k,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      v_triu_float(
        a.pointer.cast(),
        result.pointer.cast(),
        batchCount,
        rows,
        cols,
        k,
      );
      return result;
    }
  }

  final aList = a.isContiguous ? a.data : a.toList();
  final resData = result.data;
  final matrixSize = rows * cols;

  for (var b = 0; b < batchCount; b++) {
    final offset = b * matrixSize;
    for (var r = 0; r < rows; r++) {
      for (var c = 0; c < cols; c++) {
        final idx = offset + r * cols + c;
        resData[idx] = (c >= r + k) ? aList[idx] : _castValue(0, a.dtype) as T;
      }
    }
  }
  return result;
}

/// Compute the Moore-Penrose pseudo-inverse of a 2D matrix.
///
/// Uses Singular Value Decomposition (SVD) to resolve the pseudo-inverse.
/// Singular values smaller than [rcond] * max(singular_value) are treated as zero.
///
/// **Preconditions:**
/// - Input [a] must be a 2D matrix.
/// - If provided, [out] must have shape `[cols, rows]` and matching dtype.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [a] has rank != 2.
/// - [ArgumentError] if [out] has mismatched shape or dtype.
///
/// **Example:**
/// {@example /example/linalg_premium_example.dart lang=dart}
NDArray pinv(NDArray a, {double? rcond, NDArray? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute pinv() on a disposed array.');
  }
  if (a.shape.length != 2) {
    throw ArgumentError(
      'Moore-Penrose pseudo-inverse is only defined for 2D matrices (was shape ${a.shape}).',
    );
  }
  final m = a.shape[0];
  final n = a.shape[1];

  final targetShape = [n, m];
  final result = out ?? NDArray.create(targetShape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final svdResult = svd(a);
  final u = svdResult['U']!;
  final s = svdResult['S']!;
  final vt = svdResult['Vh']!;

  final maxSingularVal = s.data[0] as double;
  final epsilon = 2.220446049250313e-16;
  final maxDim = m > n ? m : n;
  final resolvedRcond = rcond ?? (maxDim * epsilon);
  final threshold = resolvedRcond * maxSingularVal;

  final sPlus = NDArray.zeros([n, m], a.dtype as dynamic);
  for (var i = 0; i < s.data.length; i++) {
    final sVal = s.data[i] as double;
    if (sVal > threshold) {
      sPlus.setCell([i, i], _castValue(1.0 / sVal, a.dtype));
    }
  }

  final v = vt.transpose();
  final ut = u.transpose();

  final temp = matmul(v, sPlus);
  final temp2 = matmul(temp, ut);
  for (var i = 0; i < result.data.length; i++) {
    result.data[i] = temp2.data[i];
  }

  u.dispose();
  s.dispose();
  vt.dispose();
  sPlus.dispose();
  v.dispose();
  ut.dispose();
  temp.dispose();
  temp2.dispose();

  return result;
}

/// Raise a square 2D matrix to the integer power [n].
///
/// Computes $A^n$ using highly optimized binary exponentiation (square-and-multiply)
/// in $O(\log n)$ matrix multiplications.
///
/// **Preconditions:**
/// - Input [a] must be a square 2D matrix.
/// - If provided, [out] must have matching shape and dtype.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [a] has rank != 2 or is not square.
/// - [ArgumentError] if [out] has mismatched shape or dtype.
///
/// **Example:**
/// {@example /example/linalg_premium_example.dart lang=dart}
NDArray matrix_power(NDArray a, int n, {NDArray? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute matrix_power() on a disposed array.');
  }
  if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
    throw ArgumentError(
      'matrix_power is only defined for 2D square matrices (was shape ${a.shape}).',
    );
  }

  final size = a.shape[0];
  final result = out ?? NDArray.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  if (n == 0) {
    final eye = NDArray.eye(size, a.dtype as dynamic);
    result.fill(0);
    for (var i = 0; i < size; i++) {
      result.setCell([i, i], eye.getCell([i, i]));
    }
    eye.dispose();
    return result;
  }

  NDArray base;
  bool wasBaseAllocated = false;

  if (n < 0) {
    base = inv(a);
    wasBaseAllocated = true;
    n = -n;
  } else {
    base = a;
  }

  var res = NDArray.eye(size, a.dtype as dynamic);
  var current = base.copy();

  var exponent = n;
  while (exponent > 0) {
    if ((exponent & 1) == 1) {
      final nextRes = matmul(res, current);
      res.dispose();
      res = nextRes;
    }
    final nextCurrent = matmul(current, current);
    current.dispose();
    current = nextCurrent;
    exponent >>= 1;
  }

  current.dispose();
  if (wasBaseAllocated) {
    base.dispose();
  }

  for (var i = 0; i < result.data.length; i++) {
    result.data[i] = res.data[i];
  }
  res.dispose();

  return result;
}

/// Compute the cumulative sum of array elements along a specified axis.
///
/// **Preconditions:**
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
/// - If provided, the [out] recycler must have compatible shape and dtype.
///
/// **Throws:**
/// - [StateError] if the array is disposed.
/// - [ArgumentError] if [axis] is out of bounds.
/// - [ArgumentError] if [out] recycler shape or dtype is incompatible.
///
/// **Example:**
/// {@example /example/cumulative_example.dart lang=dart}
/// Calculate the n-th discrete difference along the given axis.
///
/// The first difference is given by `out[i] = a[i+1] - a[i]` along the given axis.
/// Higher differences are calculated recursively.
///
/// **Preconditions:**
/// - Input [a] must not be disposed.
/// - [n] must be >= 0.
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
/// - If provided, [out] must have compatible shape and dtype.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [n] is negative.
/// - [ArgumentError] if [axis] is out of bounds.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1, 2, 4, 7, 0], [5], DType.int64);
/// final res = diff(a); // [1, 2, 3, -7]
/// ```
NDArray<T> diff<T>(NDArray<T> a, {int n = 1, int axis = -1, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute diff() on a disposed array.');
  }
  if (n < 0) {
    throw ArgumentError('Order of difference n must be >= 0 (was $n).');
  }
  if (n == 0) {
    final result = out ?? a.copy();
    if (out != null) {
      for (var i = 0; i < result.data.length; i++) {
        result.data[i] = a.data[i];
      }
    }
    return result;
  }

  var targetAxis = axis;
  if (targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }
  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  if (n >= a.shape[targetAxis]) {
    final emptyShape = List<int>.from(a.shape);
    emptyShape[targetAxis] = 0;
    return out ?? NDArray<T>.create(emptyShape, a.dtype);
  }

  if (n > 1) {
    final step = diff(a, n: n - 1, axis: targetAxis);
    final result = diff(step, n: 1, axis: targetAxis, out: out);
    step.dispose();
    return result;
  }

  final targetShape = List<int>.from(a.shape);
  targetShape[targetAxis] = a.shape[targetAxis] - 1;

  final result = out ?? NDArray<T>.create(targetShape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, targetShape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

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
    final dtype = a.dtype;
    switch (dtype) {
      case DType.float64:
        s_diff_double(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.float32:
        s_diff_float(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.int64:
        s_diff_int64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.int32:
        s_diff_int32(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.complex128:
        s_diff_complex128(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.complex64:
        s_diff_complex64(
          a.pointer.cast(),
          cStridesA,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          rank,
          targetAxis,
        );
      case DType.uint8:
      case DType.int16:
      case DType.boolean:
        final doubleA = NDArray<double>.create(a.shape, DType.float64);
        for (var i = 0; i < a.data.length; i++) {
          doubleA.data[i] = (a.data[i] as num).toDouble();
        }
        final doubleRes = NDArray<double>.create(targetShape, DType.float64);
        final cStridesDoubleA = malloc<ffi.Int>(rank);
        final cStridesDoubleRes = malloc<ffi.Int>(rank);

        for (var i = 0; i < rank; i++) {
          cStridesDoubleA[i] = doubleA.strides[i];
          cStridesDoubleRes[i] = doubleRes.strides[i];
        }

        try {
          s_diff_double(
            doubleA.pointer.cast(),
            cStridesDoubleA,
            doubleRes.pointer.cast(),
            cStridesDoubleRes,
            cShape,
            rank,
            targetAxis,
          );
        } finally {
          malloc.free(cStridesDoubleA);
          malloc.free(cStridesDoubleRes);
        }

        for (var i = 0; i < result.data.length; i++) {
          result.data[i] = _castValue(doubleRes.data[i], a.dtype) as T;
        }
        doubleA.dispose();
        doubleRes.dispose();
    }
  } finally {
    malloc.free(cShape);
    malloc.free(cStridesA);
    malloc.free(cStridesRes);
  }

  return result;
}

NDArray<T> cumsum<T>(NDArray<T> a, {int? axis, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cumsum() on a disposed array.');
  }

  final NDArray<T> result;
  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    result = out ?? NDArray<T>.create([size], a.dtype);
    if (out != null) {
      if (!listEquals(out.shape, [size]) || out.dtype != a.dtype) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
    }

    final List elements = size == a.data.length ? a.data : a.toList();
    dynamic acc;
    for (var i = 0; i < elements.length; i++) {
      acc = (i == 0)
          ? elements[i]
          : ((acc as dynamic) + elements[i]) as dynamic;
      result.data[i] = acc as T;
    }
    return result;
  }

  var targetAxis = axis;
  if (targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }
  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  result = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  return _cumOpFFI(a, targetAxis, result, _CumOpType.sum);
}

/// Compute the cumulative product of array elements along a specified axis.
///
/// **Preconditions:**
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
/// - If provided, the [out] recycler must have compatible shape and dtype.
///
/// **Throws:**
/// - [StateError] if the array is disposed.
/// - [ArgumentError] if [axis] is out of bounds.
/// - [ArgumentError] if [out] recycler shape or dtype is incompatible.
///
/// **Example:**
/// {@example /example/cumulative_example.dart lang=dart}
NDArray<T> cumprod<T>(NDArray<T> a, {int? axis, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cumprod() on a disposed array.');
  }

  final NDArray<T> result;
  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    result = out ?? NDArray<T>.create([size], a.dtype);
    if (out != null) {
      if (!listEquals(out.shape, [size]) || out.dtype != a.dtype) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
    }

    final List elements = size == a.data.length ? a.data : a.toList();
    dynamic acc;
    for (var i = 0; i < elements.length; i++) {
      acc = (i == 0)
          ? elements[i]
          : ((acc as dynamic) * elements[i]) as dynamic;
      result.data[i] = acc as T;
    }
    return result;
  }

  var targetAxis = axis;
  if (targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }
  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  result = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  return _cumOpFFI(a, targetAxis, result, _CumOpType.prod);
}

/// Compute the cumulative minimum of array elements along a specified axis.
///
/// **Preconditions:**
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
/// - If provided, the [out] recycler must have compatible shape and dtype.
///
/// **Throws:**
/// - [StateError] if the array is disposed.
/// - [ArgumentError] if [axis] is out of bounds.
/// - [ArgumentError] if [out] recycler shape or dtype is incompatible.
///
/// **Example:**
/// {@example /example/cumulative_example.dart lang=dart}
NDArray<T> cummin<T>(NDArray<T> a, {int? axis, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cummin() on a disposed array.');
  }

  final NDArray<T> result;
  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    result = out ?? NDArray<T>.create([size], a.dtype);
    if (out != null) {
      if (!listEquals(out.shape, [size]) || out.dtype != a.dtype) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
    }

    final List elements = size == a.data.length ? a.data : a.toList();
    dynamic acc;
    for (var i = 0; i < elements.length; i++) {
      acc = (i == 0)
          ? elements[i]
          : (((acc as Comparable).compareTo(elements[i]) < 0)
                ? acc
                : elements[i]);
      result.data[i] = acc as T;
    }
    return result;
  }

  var targetAxis = axis;
  if (targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }
  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  result = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  return _cumOpFFI(a, targetAxis, result, _CumOpType.min);
}

/// Compute the cumulative maximum of array elements along a specified axis.
///
/// **Preconditions:**
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
/// - If provided, the [out] recycler must have compatible shape and dtype.
///
/// **Throws:**
/// - [StateError] if the array is disposed.
/// - [ArgumentError] if [axis] is out of bounds.
/// - [ArgumentError] if [out] recycler shape or dtype is incompatible.
///
/// **Example:**
/// {@example /example/cumulative_example.dart lang=dart}
NDArray<T> cummax<T>(NDArray<T> a, {int? axis, NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cummax() on a disposed array.');
  }

  final NDArray<T> result;
  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    result = out ?? NDArray<T>.create([size], a.dtype);
    if (out != null) {
      if (!listEquals(out.shape, [size]) || out.dtype != a.dtype) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
    }

    final List elements = size == a.data.length ? a.data : a.toList();
    dynamic acc;
    for (var i = 0; i < elements.length; i++) {
      acc = (i == 0)
          ? elements[i]
          : (((acc as Comparable).compareTo(elements[i]) > 0)
                ? acc
                : elements[i]);
      result.data[i] = acc as T;
    }
    return result;
  }

  var targetAxis = axis;
  if (targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }
  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  result = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  return _cumOpFFI(a, targetAxis, result, _CumOpType.max);
}

enum _CumOpType { sum, prod, min, max }

NDArray<T> _cumOpFFI<T>(
  NDArray<T> a,
  int axis,
  NDArray<T> result,
  _CumOpType opType,
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
      case _CumOpType.sum:
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
              result.data[i] = _castValue(doubleRes.data[i], a.dtype) as T;
            }
            doubleA.dispose();
            doubleRes.dispose();
        }

      case _CumOpType.prod:
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
              result.data[i] = _castValue(doubleRes.data[i], a.dtype) as T;
            }
            doubleA.dispose();
            doubleRes.dispose();
        }

      case _CumOpType.min:
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
              result.data[i] = _castValue(doubleRes.data[i], a.dtype) as T;
            }
            doubleA.dispose();
            doubleRes.dispose();
          case DType.complex128:
          case DType.complex64:
            throw ArgumentError(
              'Cumulative minimum is not defined for complex numbers.',
            );
        }

      case _CumOpType.max:
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
              result.data[i] = _castValue(doubleRes.data[i], a.dtype) as T;
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

/// Compute the element-wise complex conjugate of the array elements.
///
/// **Preconditions:**
/// - The array must not be disposed.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([Complex(1.0, 2.0)], [1], DType.complex128);
/// final c = conj(a); // [Complex(1.0, -2.0)]
/// ```
NDArray conj(NDArray a, {NDArray? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute conj() on a disposed array.');
  }
  final targetDType = a.dtype;
  final result = out ?? NDArray.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for conj.',
      );
    }
  }

  switch (targetDType) {
    case DType.complex128:
      if (a.isContiguous && result.isContiguous) {
        v_conj_complex128(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      } else {
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
          s_conj_complex128(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        } finally {
          malloc.free(cShape);
          malloc.free(cStridesA);
          malloc.free(cStridesRes);
        }
      }
    case DType.complex64:
      if (a.isContiguous && result.isContiguous) {
        v_conj_complex64(
          a.pointer.cast(),
          result.pointer.cast(),
          a.data.length,
        );
        return result;
      } else {
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
          s_conj_complex64(
            a.pointer.cast(),
            cStridesA,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
          );
          return result;
        } finally {
          malloc.free(cShape);
          malloc.free(cStridesA);
          malloc.free(cStridesRes);
        }
      }
    case DType.float64:
    case DType.float32:
    case DType.int64:
    case DType.int32:
    case DType.uint8:
    case DType.int16:
    case DType.boolean:
      // Real/boolean numbers are their own complex conjugates!
      if (a.isContiguous && result.isContiguous) {
        final totalSize = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
        result.data.setRange(0, totalSize, a.data);
      } else {
        result.fill(0); // initialize
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
          switch (targetDType) {
            case DType.float64:
              s_flatten_double(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cShape,
                rank,
              );
            case DType.float32:
              s_flatten_float(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cShape,
                rank,
              );
            case DType.int64:
              s_flatten_int64(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cShape,
                rank,
              );
            case DType.int32:
              s_flatten_int32(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cShape,
                rank,
              );
            case DType.boolean:
              s_flatten_boolean(
                a.pointer.cast(),
                cStridesA,
                result.pointer.cast(),
                cShape,
                rank,
              );
            default:
              // Fallback recursive copy for other strided types
              for (var i = 0; i < a.data.length; i++) {
                result.data[i] = a.data[i];
              }
          }
        } finally {
          malloc.free(cShape);
          malloc.free(cStridesA);
          malloc.free(cStridesRes);
        }
      }
      return result;
  }
}

/// Alias for [conj].
NDArray conjugate(NDArray a, {NDArray? out}) => conj(a, out: out);

/// Rolls array elements along a given axis.
///
/// Elements that roll beyond the last position are re-introduced at the first.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - If [axis] is a list, [shift] must be an integer or a list of the same length.
/// - Each axis must be a valid axis index for [a] (within `[-rank, rank - 1]`).
/// - If [axis] is `null`, [shift] must be an integer.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [shift] and [axis] configurations are mismatched or invalid.
/// - [RangeError] if any axis is out of bounds.
///
/// **Performance considerations:**
/// - Algorithmic Time Complexity is $O(N)$ where $N$ is the total number of elements,
///   as it creates a new array and copies elements.
/// - Space Complexity is $O(N)$ for the newly allocated output array.
///
/// **Reference:**
/// Refer to [NumPy roll documentation](https://numpy.org/doc/stable/reference/generated/numpy.roll.html).
///
/// {@example /example/rearranging_example.dart lang=dart}
NDArray<T> roll<T extends Object>(NDArray<T> a, dynamic shift, {dynamic axis}) {
  if (a.isDisposed) {
    throw StateError('Cannot roll a disposed array.');
  }

  // Parse shifts and axes
  final List<int> shifts;
  final List<int>? axes;

  if (axis == null) {
    if (shift is int) {
      shifts = [shift];
      axes = null;
    } else if (shift is List<int>) {
      if (shift.length != 1) {
        throw ArgumentError('shift must be an integer when axis is null');
      }
      shifts = shift;
      axes = null;
    } else {
      throw ArgumentError('shift must be an integer or a list of integers');
    }
  } else if (axis is int) {
    if (shift is int) {
      shifts = [shift];
      axes = [axis];
    } else if (shift is List<int>) {
      if (shift.length != 1) {
        throw ArgumentError(
          'shift and axis must have the same number of elements',
        );
      }
      shifts = shift;
      axes = [axis];
    } else {
      throw ArgumentError('shift must be an integer or a list of integers');
    }
  } else if (axis is List<int>) {
    if (shift is int) {
      shifts = List<int>.filled(axis.length, shift);
      axes = axis;
    } else if (shift is List<int>) {
      if (shift.length != axis.length) {
        throw ArgumentError(
          'shift and axis must have the same number of elements',
        );
      }
      shifts = shift;
      axes = axis;
    } else {
      throw ArgumentError('shift must be an integer or a list of integers');
    }
  } else {
    throw ArgumentError('axis must be null, an integer, or a list of integers');
  }

  if (a.rank == 0) {
    return a.copy();
  }

  return NDArray.scope(() {
    NDArray<T> current = a;

    if (axes == null) {
      final flat = current.ravel();
      final s = shifts[0];
      final rolledFlat = _rollSingle1D(flat, s);
      final result = rolledFlat.reshape(a.shape);
      return result.detachToParentScope();
    } else {
      for (var i = 0; i < axes.length; i++) {
        current = _rollSingle(current, shifts[i], axes[i]);
      }
      return current.copy().detachToParentScope();
    }
  });
}

NDArray<T> _rollSingle1D<T extends Object>(NDArray<T> a, int shift) {
  final size = a.size;
  if (size == 0) return a.copy();
  final s = shift % size;
  if (s == 0) return a.copy();

  final realShift = s < 0 ? size + s : s;

  final part1 = a.slice([Slice(start: size - realShift, stop: size)]);
  final part2 = a.slice([Slice(start: 0, stop: size - realShift)]);
  return concatenate([part1, part2], axis: 0);
}

NDArray<T> _rollSingle<T extends Object>(NDArray<T> a, int shift, int axis) {
  final rank = a.rank;
  final normAx = axis < 0 ? rank + axis : axis;
  if (normAx < 0 || normAx >= rank) {
    throw RangeError.range(normAx, 0, rank - 1, 'axis');
  }

  final dimSize = a.shape[normAx];
  if (dimSize == 0) return a.copy();

  final s = shift % dimSize;
  if (s == 0) return a.copy();

  final realShift = s < 0 ? dimSize + s : s;

  final selectors1 = List<Selector>.generate(
    rank,
    (i) => i == normAx
        ? Slice(start: dimSize - realShift, stop: dimSize)
        : const Slice.all(),
  );
  final selectors2 = List<Selector>.generate(
    rank,
    (i) => i == normAx
        ? Slice(start: 0, stop: dimSize - realShift)
        : const Slice.all(),
  );

  final part1 = a.slice(selectors1);
  final part2 = a.slice(selectors2);

  return concatenate([part1, part2], axis: normAx);
}

/// Reverses the order of elements along the given axis/axes.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - If [axis] is a list, all elements must be unique.
/// - Each axis must be a valid axis index for [a] (within `[-rank, rank - 1]`).
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [RangeError] if any axis is out of bounds.
/// - [ArgumentError] if [axis] contains duplicate indices.
///
/// **Performance considerations:**
/// - Algorithmic Time Complexity is $O(1)$ because it returns a zero-copy strided view.
/// - Space Complexity is $O(1)$ as no new array data is allocated.
///
/// **Reference:**
/// Refer to [NumPy flip documentation](https://numpy.org/doc/stable/reference/generated/numpy.flip.html).
///
/// {@example /example/rearranging_example.dart lang=dart}
NDArray<T> flip<T extends Object>(NDArray<T> a, {dynamic axis}) {
  if (a.isDisposed) {
    throw StateError('Cannot flip a disposed array.');
  }

  final rank = a.rank;
  if (rank == 0) {
    return NDArray.view(
      a,
      shape: [],
      strides: [],
      offsetElements: a.offsetElements,
    );
  }

  final List<int> axesToFlip;
  if (axis == null) {
    axesToFlip = List.generate(rank, (i) => i);
  } else if (axis is int) {
    final normAx = axis < 0 ? rank + axis : axis;
    if (normAx < 0 || normAx >= rank) {
      throw RangeError.range(normAx, 0, rank - 1, 'axis');
    }
    axesToFlip = [normAx];
  } else if (axis is List<int>) {
    final uniqueAxes = <int>{};
    for (final ax in axis) {
      final normAx = ax < 0 ? rank + ax : ax;
      if (normAx < 0 || normAx >= rank) {
        throw RangeError.range(normAx, 0, rank - 1, 'axis');
      }
      if (!uniqueAxes.add(normAx)) {
        throw ArgumentError('axes must be unique');
      }
    }
    axesToFlip = uniqueAxes.toList();
  } else {
    throw ArgumentError('axis must be null, an integer, or a list of integers');
  }

  final newStrides = List<int>.from(a.strides);
  var offset = a.offsetElements;

  for (final ax in axesToFlip) {
    newStrides[ax] = a.strides[ax] * -1;
    offset += (a.shape[ax] - 1) * a.strides[ax];
  }

  return NDArray.view(
    a,
    shape: a.shape,
    strides: newStrides,
    offsetElements: offset,
  );
}

/// Flips the array in the left/right direction (column-wise).
///
/// Equivalent to `flip(a, axis: 1)`.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [a] must have a rank of at least 2.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [a] rank is less than 2.
///
/// **Performance considerations:**
/// - Algorithmic Time Complexity is $O(1)$ because it returns a zero-copy strided view.
/// - Space Complexity is $O(1)$ as no new array data is allocated.
///
/// **Reference:**
/// Refer to [NumPy fliplr documentation](https://numpy.org/doc/stable/reference/generated/numpy.fliplr.html).
///
/// {@example /example/rearranging_example.dart lang=dart}
NDArray<T> fliplr<T extends Object>(NDArray<T> a) {
  if (a.isDisposed) {
    throw StateError('Cannot fliplr a disposed array.');
  }
  if (a.rank < 2) {
    throw ArgumentError('Input must be >= 2-D.');
  }
  return flip(a, axis: 1);
}

/// Flips the array in the up/down direction (row-wise).
///
/// Equivalent to `flip(a, axis: 0)`.
///
/// **Preconditions:**
/// - [a] must not be disposed.
/// - [a] must have a rank of at least 1.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [a] rank is less than 1.
///
/// **Performance considerations:**
/// - Algorithmic Time Complexity is $O(1)$ because it returns a zero-copy strided view.
/// - Space Complexity is $O(1)$ as no new array data is allocated.
///
/// **Reference:**
/// Refer to [NumPy flipud documentation](https://numpy.org/doc/stable/reference/generated/numpy.flipud.html).
///
/// {@example /example/rearranging_example.dart lang=dart}
NDArray<T> flipud<T extends Object>(NDArray<T> a) {
  if (a.isDisposed) {
    throw StateError('Cannot flipud a disposed array.');
  }
  if (a.rank < 1) {
    throw ArgumentError('Input must be >= 1-D.');
  }
  return flip(a, axis: 0);
}



@internal
DType resolveDType(DType a, DType b) => _resolveDType(a, b);

@internal
extension ArithmeticNDArrayOperationsHelper<T> on NDArray<T> {
  @internal
  void dynamicElementWiseOp<R>(
    NDArray<R> result,
    NDArray other,
    List<int> commonShape,
    List<int> stridesA,
    List<int> stridesB,
    List<int> resultStrides,
    dynamic Function(dynamic, dynamic) op, {
    bool isSubtract = false,
    bool isDivide = false,
  }) {
    _dynamicElementWiseOp<R>(
      result,
      other,
      commonShape,
      stridesA,
      stridesB,
      resultStrides,
      op,
      isSubtract: isSubtract,
      isDivide: isDivide,
    );
  }

  // Helper method to execute dynamic element-wise ops cleanly on the fallback branches
  void _dynamicElementWiseOp<R>(
    NDArray<R> result,
    NDArray other,
    List<int> commonShape,
    List<int> stridesA,
    List<int> stridesB,
    List<int> resultStrides,
    dynamic Function(dynamic, dynamic) op, {
    bool isSubtract = false,
    bool isDivide = false,
  }) {
    final targetDType = result.dtype;
    switch (targetDType) {
      case DType.complex128:
      case DType.complex64:
        final rData = result.data as List<Complex>;
        switch (dtype) {
          case DType.complex128:
          case DType.complex64:
            final aData = data as List<Complex>;
            switch (other.dtype) {
              case DType.complex128:
              case DType.complex64:
                _elementWiseOp<Complex, Complex, Complex>(
                  rData,
                  aData,
                  other.data as List<Complex>,
                  commonShape,
                  stridesA,
                  stridesB,
                  resultStrides,
                  0,
                  offsetElements,
                  other.offsetElements,
                  result.offsetElements,
                  (x, y) => op(x, y) as Complex,
                );
                break;
              case DType.float64:
              case DType.float32:
                _elementWiseOp<Complex, double, Complex>(
                  rData,
                  aData,
                  other.data as List<double>,
                  commonShape,
                  stridesA,
                  stridesB,
                  resultStrides,
                  0,
                  offsetElements,
                  other.offsetElements,
                  result.offsetElements,
                  (x, y) => op(x, y) as Complex,
                );
                break;
              default:
                _elementWiseOp<Complex, int, Complex>(
                  rData,
                  aData,
                  other.data as List<int>,
                  commonShape,
                  stridesA,
                  stridesB,
                  resultStrides,
                  0,
                  offsetElements,
                  other.offsetElements,
                  result.offsetElements,
                  (x, y) => op(x, y) as Complex,
                );
                break;
            }
            break;
          case DType.float64:
          case DType.float32:
            final aData = data as List<double>;
            if (other.dtype.isComplex) {
              if (isSubtract) {
                _elementWiseOp<double, Complex, Complex>(
                  rData,
                  aData,
                  other.data as List<Complex>,
                  commonShape,
                  stridesA,
                  stridesB,
                  resultStrides,
                  0,
                  offsetElements,
                  other.offsetElements,
                  result.offsetElements,
                  (x, y) => Complex(x - y.real, -y.imag),
                );
              } else {
                _elementWiseOp<double, Complex, Complex>(
                  rData,
                  aData,
                  other.data as List<Complex>,
                  commonShape,
                  stridesA,
                  stridesB,
                  resultStrides,
                  0,
                  offsetElements,
                  other.offsetElements,
                  result.offsetElements,
                  (x, y) => op(x, y) as Complex,
                );
              }
            }
            break;
          default:
            final aData = data as List<int>;
            if (other.dtype.isComplex) {
              if (isSubtract) {
                _elementWiseOp<int, Complex, Complex>(
                  rData,
                  aData,
                  other.data as List<Complex>,
                  commonShape,
                  stridesA,
                  stridesB,
                  resultStrides,
                  0,
                  offsetElements,
                  other.offsetElements,
                  result.offsetElements,
                  (x, y) => Complex(x - y.real, -y.imag),
                );
              } else {
                _elementWiseOp<int, Complex, Complex>(
                  rData,
                  aData,
                  other.data as List<Complex>,
                  commonShape,
                  stridesA,
                  stridesB,
                  resultStrides,
                  0,
                  offsetElements,
                  other.offsetElements,
                  result.offsetElements,
                  (x, y) => op(x, y) as Complex,
                );
              }
            }
            break;
        }
        break;

      case DType.float64:
      case DType.float32:
        final rData = result.data as List<double>;
        if (dtype.isFloating) {
          final aData = data as List<double>;
          if (other.dtype.isFloating) {
            _elementWiseOp<double, double, double>(
              rData,
              aData,
              other.data as List<double>,
              commonShape,
              stridesA,
              stridesB,
              resultStrides,
              0,
              offsetElements,
              other.offsetElements,
              result.offsetElements,
              (x, y) => (op(x, y) as num).toDouble(),
            );
          } else {
            _elementWiseOp<double, int, double>(
              rData,
              aData,
              other.data as List<int>,
              commonShape,
              stridesA,
              stridesB,
              resultStrides,
              0,
              offsetElements,
              other.offsetElements,
              result.offsetElements,
              (x, y) => (op(x, y) as num).toDouble(),
            );
          }
        } else {
          final aData = data as List<int>;
          if (other.dtype.isFloating) {
            _elementWiseOp<int, double, double>(
              rData,
              aData,
              other.data as List<double>,
              commonShape,
              stridesA,
              stridesB,
              resultStrides,
              0,
              offsetElements,
              other.offsetElements,
              result.offsetElements,
              (x, y) => (op(x, y) as num).toDouble(),
            );
          } else {
            _elementWiseOp<int, int, double>(
              rData,
              aData,
              other.data as List<int>,
              commonShape,
              stridesA,
              stridesB,
              resultStrides,
              0,
              offsetElements,
              other.offsetElements,
              result.offsetElements,
              (x, y) => (op(x, y) as num).toDouble(),
            );
          }
        }
        break;

      default:
        if (isDivide) {
          _elementWiseOp<int, int, double>(
            result.data as List<double>,
            data as List<int>,
            other.data as List<int>,
            commonShape,
            stridesA,
            stridesB,
            resultStrides,
            0,
            offsetElements,
            other.offsetElements,
            result.offsetElements,
            (x, y) => x.toDouble() / y.toDouble(),
          );
        } else {
          _elementWiseOp<int, int, int>(
            result.data as List<int>,
            data as List<int>,
            other.data as List<int>,
            commonShape,
            stridesA,
            stridesB,
            resultStrides,
            0,
            offsetElements,
            other.offsetElements,
            result.offsetElements,
            (x, y) => (op(x, y) as num).toInt(),
          );
        }
        break;
    }
  }
}
