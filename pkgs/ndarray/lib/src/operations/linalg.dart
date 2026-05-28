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
import 'helpers.dart';

/// Matrix multiplication using OpenBLAS, supporting high-dimensional stack broadcasting and 1D vector promotions.
NDArray<R> matmul<Ta, Tb, R>(NDArray<Ta> a, NDArray<Tb> b, {NDArray<R>? out}) {
  final targetDType = resolveDType(a.dtype, b.dtype);

  if (a.shape.length == 1 && b.shape.length == 1) {
    final n = a.shape[0];
    if (n != b.shape[0]) {
      throw ArgumentError(
        'Incompatible vector dimensions for 1D dot product in matmul: ${a.shape} and ${b.shape}',
      );
    }
    if (out != null) {
      if (!listEquals(out.shape, []) || out.dtype != targetDType) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype (expected shape [] and dtype $targetDType, got shape ${out.shape} and dtype ${out.dtype}).',
        );
      }
    }
    if (targetDType == DType.float64) {
      final scalarRes = cblas_ddot(
        n,
        a.pointer.cast<ffi.Double>(),
        1,
        b.pointer.cast<ffi.Double>(),
        1,
      );
      final result =
          out ??
          (NDArray.fromList([scalarRes], [], DType.float64) as NDArray<R>);
      if (out != null) {
        out.data[0] = scalarRes as R;
      }
      return result;
    } else if (targetDType == DType.float32) {
      final scalarRes = cblas_sdot(
        n,
        a.pointer.cast<ffi.Float>(),
        1,
        b.pointer.cast<ffi.Float>(),
        1,
      );
      final result =
          out ??
          (NDArray.fromList([scalarRes], [], DType.float32) as NDArray<R>);
      if (out != null) {
        out.data[0] = scalarRes as R;
      }
      return result;
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
  final broadcastStack = broadcastStackShapes(stackA, stackB);

  final expectedFinalShape = <int>[];
  if (aPromoted && bPromoted) {
    // empty []
  } else if (aPromoted) {
    expectedFinalShape.addAll([...broadcastStack, n]);
  } else if (bPromoted) {
    expectedFinalShape.addAll([...broadcastStack, m]);
  } else {
    expectedFinalShape.addAll([...broadcastStack, m, n]);
  }

  if (out != null) {
    if (!listEquals(out.shape, expectedFinalShape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype (expected shape $expectedFinalShape and dtype $targetDType, got shape ${out.shape} and dtype ${out.dtype}).',
      );
    }
  }

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

  ffi.Pointer<ffi.Double> alphaZ = ffi.nullptr.cast();
  ffi.Pointer<ffi.Double> betaZ = ffi.nullptr.cast();
  ffi.Pointer<ffi.Float> alphaC = ffi.nullptr.cast();
  ffi.Pointer<ffi.Float> betaC = ffi.nullptr.cast();

  if (targetDType == DType.complex128) {
    alphaZ = malloc<ffi.Double>(2);
    alphaZ[0] = 1.0;
    alphaZ[1] = 0.0;
    betaZ = malloc<ffi.Double>(2);
    betaZ[0] = 0.0;
    betaZ[1] = 0.0;
  } else if (targetDType == DType.complex64) {
    alphaC = malloc<ffi.Float>(2);
    alphaC[0] = 1.0;
    alphaC[1] = 0.0;
    betaC = malloc<ffi.Float>(2);
    betaC[0] = 0.0;
    betaC[1] = 0.0;
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
      } else if (targetDType == DType.float32) {
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
      } else if (targetDType == DType.complex128) {
        cblas_zgemm(
          101,
          transA,
          transB,
          m,
          n,
          kA,
          alphaZ,
          aView.pointer.cast<ffi.Double>() + (offsetA * 2),
          lda,
          bView.pointer.cast<ffi.Double>() + (offsetB * 2),
          ldb,
          betaZ,
          result.pointer.cast<ffi.Double>() + (offsetRes * 2),
          n,
        );
      } else if (targetDType == DType.complex64) {
        cblas_cgemm(
          101,
          transA,
          transB,
          m,
          n,
          kA,
          alphaC,
          aView.pointer.cast<ffi.Float>() + (offsetA * 2),
          lda,
          bView.pointer.cast<ffi.Float>() + (offsetB * 2),
          ldb,
          betaC,
          result.pointer.cast<ffi.Float>() + (offsetRes * 2),
          n,
        );
      } else if (targetDType.isInteger) {
        final resData = result.data;
        final aData = aView.data;
        final bData = bView.data;

        final strideARow = aView.strides[rankA - 2];
        final strideACol = aView.strides[rankA - 1];

        final strideBRow = bView.strides[rankB - 2];
        final strideBCol = bView.strides[rankB - 1];

        final strideResRow = result.strides[resShape.length - 2];
        final strideResCol = result.strides[resShape.length - 1];

        for (var r = 0; r < m; r++) {
          for (var c = 0; c < n; c++) {
            var sum = 0;
            for (var i = 0; i < kA; i++) {
              final idxA = offsetA + r * strideARow + i * strideACol;
              final idxB = offsetB + i * strideBRow + c * strideBCol;
              sum += (aData[idxA] as int) * (bData[idxB] as int);
            }
            final idxRes = offsetRes + r * strideResRow + c * strideResCol;
            resData[idxRes] = sum;
          }
        }
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

  if (alphaZ.address != 0) malloc.free(alphaZ);
  if (betaZ.address != 0) malloc.free(betaZ);
  if (alphaC.address != 0) malloc.free(alphaC);
  if (betaC.address != 0) malloc.free(betaC);

  if (out != null) {
    if (out.isContiguous && result.isContiguous) {
      final byteCount = result.data.length * targetDType.byteWidth;
      ffi.Pointer.fromAddress(out.pointer.address)
          .cast<ffi.Uint8>()
          .asTypedList(byteCount)
          .setAll(
            0,
            ffi.Pointer.fromAddress(
              result.pointer.address,
            ).cast<ffi.Uint8>().asTypedList(byteCount),
          );
    } else {
      final resFlat = result.toList();
      for (var i = 0; i < resFlat.length; i++) {
        out.data[i] = resFlat[i] as R;
      }
    }
    result.dispose();
    return out;
  }

  // Post-calculation 1D dummy dimensions demotions
  if (aPromoted && bPromoted) {
    return result.reshape([])
        as NDArray<R>; // 0D scalar array for pure vector dot products
  } else if (aPromoted) {
    final newShape = List<int>.from(result.shape)
      ..removeAt(result.shape.length - 2);
    return result.reshape(newShape) as NDArray<R>;
  } else if (bPromoted) {
    final newShape = List<int>.from(result.shape)
      ..removeAt(result.shape.length - 1);
    return result.reshape(newShape) as NDArray<R>;
  }

  return result as NDArray<R>;
}

/// Computes the product of two or more arrays in a single function call,
/// while automatically selecting the fastest evaluation order.
///
/// Solves the matrix chain multiplication problem using standard dynamic programming in $O(N^3)$ time.
///
/// **Preconditions:**
/// - [arrays] must contain at least 2 arrays.
/// - The first array may be 1-dimensional (treated as a row vector) or 2-dimensional.
/// - The last array may be 1-dimensional (treated as a column vector) or 2-dimensional.
/// - All intermediate arrays must be 2-dimensional.
/// - Adjacent arrays must have compatible inner dimensions for matrix multiplication.
/// - If provided, the recycler [out] must have matching shape and dtype.
///
/// **Throws:**
/// - [ArgumentError] if [arrays] has fewer than 2 elements.
/// - [ArgumentError] if any intermediate array is not 2-dimensional.
/// - [ArgumentError] if first or last array has rank > 2 or rank < 1.
/// - [ArgumentError] if inner dimensions of adjacent matrices are incompatible.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
///
/// **Performance considerations:**
/// - Automatically optimizes the order of operations to minimize total scalar multiplications.
/// - All intermediate transient arrays are automatically disposed of to guarantee zero memory leaks.
///
/// **Example:**
/// {@example /example/linalg_multi_dot_example.dart lang=dart}
///
/// Reference: [NumPy linalg.multi_dot](https://numpy.org/doc/stable/reference/generated/numpy.linalg.multi_dot.html)
NDArray<T> multi_dot<T>(List<NDArray<Object>> arrays, {NDArray<T>? out}) {
  if (arrays.length < 2) {
    throw ArgumentError(
      'multi_dot requires at least 2 arrays (got ${arrays.length}).',
    );
  }

  final n = arrays.length;

  // Check dimensions & validate rank preconditions
  for (var i = 0; i < n; i++) {
    final rank = arrays[i].shape.length;
    if (i == 0 || i == n - 1) {
      if (rank != 1 && rank != 2) {
        throw ArgumentError(
          'First and last arrays in multi_dot must be 1D or 2D (array $i was shape ${arrays[i].shape}).',
        );
      }
    } else {
      if (rank != 2) {
        throw ArgumentError(
          'All intermediate arrays in multi_dot must be 2D (array $i was shape ${arrays[i].shape}).',
        );
      }
    }
  }

  // Build dimensions list p
  final p = List<int>.filled(n + 1, 0);
  if (arrays[0].shape.length == 1) {
    p[0] = 1;
    p[1] = arrays[0].shape[0];
  } else {
    p[0] = arrays[0].shape[0];
    p[1] = arrays[0].shape[1];
  }

  for (var i = 1; i < n - 1; i++) {
    final shape = arrays[i].shape;
    if (shape[0] != p[i]) {
      throw ArgumentError(
        'Incompatible matrix dimensions in multi_dot: array $i first dimension (${shape[0]}) must match previous dimension (${p[i]}).',
      );
    }
    p[i + 1] = shape[1];
  }

  // Last array
  final lastIdx = n - 1;
  final lastShape = arrays[lastIdx].shape;
  if (lastShape[0] != p[lastIdx]) {
    throw ArgumentError(
      'Incompatible matrix dimensions in multi_dot: last array first dimension (${lastShape[0]}) must match previous dimension (${p[lastIdx]}).',
    );
  }
  if (lastShape.length == 1) {
    p[n] = 1;
  } else {
    p[n] = lastShape[1];
  }

  // Resolve target DType and upcasted type
  DType<dynamic> targetDType = arrays[0].dtype;
  for (var i = 1; i < n; i++) {
    targetDType = resolveDType(targetDType, arrays[i].dtype);
  }
  if (!targetDType.isFloating && !targetDType.isComplex) {
    targetDType = DType.float64;
  }

  // If out is provided, validate it
  final expectedFinalShape = <int>[];
  final first1D = arrays[0].shape.length == 1;
  final last1D = arrays[lastIdx].shape.length == 1;
  if (first1D && last1D) {
    // Result is 0D scalar shape []
  } else if (first1D) {
    expectedFinalShape.add(p[n]);
  } else if (last1D) {
    expectedFinalShape.add(p[0]);
  } else {
    expectedFinalShape.addAll([p[0], p[n]]);
  }

  if (out != null) {
    if (!listEquals(out.shape, expectedFinalShape) ||
        out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out recycler has incompatible shape or dtype (expected shape $expectedFinalShape and dtype $targetDType, got shape ${out.shape} and dtype ${out.dtype}).',
      );
    }
  }

  // Dynamic programming to find the optimal parenthesization
  final m = List.generate(n + 1, (_) => List<int>.filled(n + 1, 0));
  final s = List.generate(n + 1, (_) => List<int>.filled(n + 1, 0));

  for (var l = 2; l <= n; l++) {
    for (var i = 1; i <= n - l + 1; i++) {
      final j = i + l - 1;
      m[i][j] = 99999999999999; // large number as infinity
      for (var k = i; k < j; k++) {
        final cost = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j];
        if (cost < m[i][j]) {
          m[i][j] = cost;
          s[i][j] = k;
        }
      }
    }
  }

  // Helper function to recursively evaluate matrix multiplication chain
  NDArray eval(int i, int j) {
    if (i == j) {
      // Return a contiguous copy of arrays[i-1] casted to the correct targetDType
      final src = arrays[i - 1];
      final copy = NDArray.create(src.shape, targetDType);
      if (src.isContiguous && src.dtype == targetDType) {
        if (targetDType.isComplex) {
          copy.data.setRange(0, src.data.length, src.data as List<Complex>);
        } else {
          copy.data.setRange(0, src.data.length, src.data as List<num>);
        }
      } else {
        final flat = src.toList();
        if (targetDType.isComplex) {
          copy.data.setRange(0, flat.length, flat.cast<Complex>());
        } else {
          copy.data.setRange(0, flat.length, flat.cast<num>());
        }
      }
      return copy;
    }

    final k = s[i][j];
    final left = eval(i, k);
    final right = eval(k + 1, j);

    // Perform matrix multiplication
    final res = matmul(left, right);
    left.dispose();
    right.dispose();
    return res;
  }

  // Top-level split point evaluation
  final k = s[1][n];
  final left = eval(1, k);
  final right = eval(k + 1, n);

  final finalResult = matmul(left, right, out: out);
  left.dispose();
  right.dispose();

  finalResult.detachToParentScope();
  return finalResult;
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

/// Compute the determinant of a square matrix or a stack of square matrices using OpenBLAS.
///
/// Transforms the matrix and calculates its determinant natively via LAPACK LU decomposition.
/// Returns the determinant as a double or a stack of determinants.
///
/// **Preconditions:**
/// - Matrix [a] must be square in its last two dimensions (size $N \times N$) and at least 2-dimensional.
/// - Data type [a.dtype] must be float32 or float64.
///
/// **Throws:**
/// - [ArgumentError] if [a] is not square or less than 2D.
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
NDArray<double> det<T>(NDArray<T> a) {
  if (a.dtype != DType.float64 && a.dtype != DType.float32) {
    throw ArgumentError('det only supports Float64 and Float32 dtypes');
  }
  final rank = a.shape.length;
  if (rank < 2 || a.shape[rank - 1] != a.shape[rank - 2]) {
    throw ArgumentError(
      'Matrix must be square and at least 2D (was ${a.shape})',
    );
  }
  final n = a.shape[rank - 1];
  final stackShape = a.shape.sublist(0, rank - 2);
  final result = NDArray.zeros(stackShape, DType.float64);

  final aCopy = NDArray.create([n, n], a.dtype);
  final marker = ScratchArena.marker;
  final ipiv = ScratchArena.allocate<ffi.Int>(n * ffi.sizeOf<ffi.Int>());

  try {
    walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
      coords,
    ) {
      var offsetA = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetA += coords[i] * a.strides[i];
      }
      var offsetRes = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetRes += coords[i] * result.strides[i];
      }

      if (a.dtype == DType.float64) {
        final aCopyData = aCopy.data as Float64List;
        if (a.isContiguous) {
          final aData = a.data as Float64List;
          aCopyData.setRange(0, n * n, aData, offsetA);
        } else {
          final sliceView = NDArray.view(
            a,
            shape: [n, n],
            strides: a.strides.sublist(rank - 2),
            offsetElements: offsetA,
          );
          aCopyData.setRange(0, n * n, sliceView.toList().cast<double>());
        }

        double detValue = 1.0;
        final info = LAPACKE_dgetrf(
          101, // ROW_MAJOR
          n,
          n,
          aCopy.pointer.cast<ffi.Double>(),
          n,
          ipiv,
        );
        if (info < 0) throw ArgumentError('Illegal value in dgetrf: $info');
        if (info > 0) {
          detValue = 0.0;
        } else {
          final data = aCopy.data as Float64List;
          for (var i = 0; i < n; i++) {
            detValue *= data[i * n + i];
          }
          var swaps = 0;
          for (var i = 0; i < n; i++) {
            if (ipiv[i] != i + 1) swaps++;
          }
          if (swaps % 2 != 0) detValue = -detValue;
        }
        (result.data as Float64List)[offsetRes] = detValue;
      } else {
        final aCopyData = aCopy.data as Float32List;
        if (a.isContiguous) {
          final aData = a.data as Float32List;
          aCopyData.setRange(0, n * n, aData, offsetA);
        } else {
          final sliceView = NDArray.view(
            a,
            shape: [n, n],
            strides: a.strides.sublist(rank - 2),
            offsetElements: offsetA,
          );
          aCopyData.setRange(0, n * n, sliceView.toList().cast<double>());
        }

        double detValue = 1.0;
        final info = LAPACKE_sgetrf(
          101, // ROW_MAJOR
          n,
          n,
          aCopy.pointer.cast<ffi.Float>(),
          n,
          ipiv,
        );
        if (info < 0) throw ArgumentError('Illegal value in sgetrf: $info');
        if (info > 0) {
          detValue = 0.0;
        } else {
          final data = aCopy.data as Float32List;
          for (var i = 0; i < n; i++) {
            detValue *= data[i * n + i];
          }
          var swaps = 0;
          for (var i = 0; i < n; i++) {
            if (ipiv[i] != i + 1) swaps++;
          }
          if (swaps % 2 != 0) detValue = -detValue;
        }
        (result.data as Float64List)[offsetRes] = detValue;
      }
    });
  } finally {
    ScratchArena.reset(marker);
    aCopy.dispose();
  }

  return result;
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
NDArray<R> solve<Ta, Tb, R>(NDArray<Ta> a, NDArray<Tb> b) {
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
      return bCopy as NDArray<R>;
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
      return bCopy as NDArray<R>;
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
      return bCopy as NDArray<R>;
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
      return bCopy as NDArray<R>;
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
      return bDouble as NDArray<R>;
    }
  } finally {
    malloc.free(ipiv);
  }
}

/// Compute the eigenvalues and right eigenvectors of a square array or stack of square arrays.
///
/// Returns a Map with keys 'eigenvalues' and 'eigenvectors'.
/// Both are returned as `NDArray<Complex>` because they can be complex
/// even for real matrices.
///
/// **Preconditions:**
/// - Matrix [a] must be square in its last two dimensions (size $N \times N$) and at least 2-dimensional.
///
/// **Throws:**
/// - [ArgumentError] if [a] is not square or less than 2D.
/// - [UnimplementedError] if the DType of [a] is not supported.
Map<String, NDArray<Complex>> eig<T>(NDArray<T> a) {
  final rank = a.shape.length;
  if (rank < 2 || a.shape[rank - 1] != a.shape[rank - 2]) {
    throw ArgumentError(
      'Matrix must be square and at least 2D (was ${a.shape})',
    );
  }
  final n = a.shape[rank - 1];
  final stackShape = a.shape.sublist(0, rank - 2);

  final DType<Complex> compDType =
      (a.dtype == DType.float32 || a.dtype == DType.complex64)
      ? DType.complex64
      : DType.complex128;

  final wShape = [...stackShape, n];
  final vrShape = [...stackShape, n, n];

  final w = NDArray<Complex>.create(wShape, compDType);
  final vr = NDArray<Complex>.create(vrShape, compDType);

  final jobvl = 'N'.codeUnitAt(0);
  final jobvr = 'V'.codeUnitAt(0);

  if (a.dtype != DType.complex128 &&
      a.dtype != DType.complex64 &&
      a.dtype != DType.float64 &&
      a.dtype != DType.int32 &&
      a.dtype != DType.int64 &&
      a.dtype != DType.float32) {
    throw UnimplementedError('Type ${a.dtype} not supported for eig');
  }

  final DType<dynamic> sliceCopyDType = (a.dtype == DType.float32)
      ? DType.float32
      : ((a.dtype == DType.complex64)
            ? DType.complex64
            : ((a.dtype == DType.complex128)
                  ? DType.complex128
                  : DType.float64));

  final sliceCopy = NDArray.create([n, n], sliceCopyDType);

  try {
    walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
      coords,
    ) {
      var offsetA = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetA += coords[i] * a.strides[i];
      }

      if (a.dtype == DType.complex128) {
        final sliceCopyData = (sliceCopy.data as ComplexList).backingList;
        if (a.isContiguous) {
          final aData = (a.data as ComplexList).backingList;
          sliceCopyData.setRange(0, n * n * 2, aData, offsetA * 2);
        } else {
          final sliceView = NDArray.view(
            a,
            shape: [n, n],
            strides: a.strides.sublist(rank - 2),
            offsetElements: offsetA,
          );
          final backing = (sliceView.data as ComplexList).backingList;
          sliceCopyData.setRange(0, n * n * 2, backing);
        }
      } else if (a.dtype == DType.complex64) {
        final sliceCopyData = (sliceCopy.data as ComplexList).backingList;
        if (a.isContiguous) {
          final aData = (a.data as ComplexList).backingList;
          sliceCopyData.setRange(0, n * n * 2, aData, offsetA * 2);
        } else {
          final sliceView = NDArray.view(
            a,
            shape: [n, n],
            strides: a.strides.sublist(rank - 2),
            offsetElements: offsetA,
          );
          final backing = (sliceView.data as ComplexList).backingList;
          sliceCopyData.setRange(0, n * n * 2, backing);
        }
      } else if (a.dtype == DType.float64 ||
          a.dtype == DType.int32 ||
          a.dtype == DType.int64) {
        final sliceCopyData = sliceCopy.data as Float64List;
        if (a.isContiguous && a.dtype == DType.float64) {
          final aData = a.data as Float64List;
          sliceCopyData.setRange(0, n * n, aData, offsetA);
        } else {
          final sliceView = NDArray.view(
            a,
            shape: [n, n],
            strides: a.strides.sublist(rank - 2),
            offsetElements: offsetA,
          );
          final list = sliceView.toList();
          for (var i = 0; i < n * n; i++) {
            sliceCopyData[i] = (list[i] as num).toDouble();
          }
        }
      } else if (a.dtype == DType.float32) {
        final sliceCopyData = sliceCopy.data as Float32List;
        if (a.isContiguous && a.dtype == DType.float32) {
          final aData = a.data as Float32List;
          sliceCopyData.setRange(0, n * n, aData, offsetA);
        } else {
          final sliceView = NDArray.view(
            a,
            shape: [n, n],
            strides: a.strides.sublist(rank - 2),
            offsetElements: offsetA,
          );
          final list = sliceView.toList();
          for (var i = 0; i < n * n; i++) {
            sliceCopyData[i] = (list[i] as num).toDouble();
          }
        }
      }

      var offsetW = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetW += coords[i] * w.strides[i];
      }
      var offsetVR = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetVR += coords[i] * vr.strides[i];
      }

      if (a.dtype == DType.complex128) {
        final w2D = NDArray<Complex>.create([n], DType.complex128);
        final vr2D = NDArray<Complex>.create([n, n], DType.complex128);

        final info = LAPACKE_zgeev(
          101, // ROW_MAJOR
          jobvl,
          jobvr,
          n,
          sliceCopy.pointer.cast<ffi.Double>(),
          n,
          w2D.pointer.cast<ffi.Double>(),
          ffi.nullptr.cast<ffi.Double>(),
          n,
          vr2D.pointer.cast<ffi.Double>(),
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

        final w2DData = w2D.data;
        final strideWLast = w.strides.isEmpty ? 1 : w.strides.last;
        for (var j = 0; j < n; j++) {
          w.data[offsetW + j * strideWLast] = w2DData[j];
        }

        final vr2DData = vr2D.data;
        final strideVR1 = vr.strides[rank - 2];
        final strideVR2 = vr.strides[rank - 1];
        for (var r = 0; r < n; r++) {
          for (var c = 0; c < n; c++) {
            vr.data[offsetVR + r * strideVR1 + c * strideVR2] =
                vr2DData[r * n + c];
          }
        }
        w2D.dispose();
        vr2D.dispose();
      } else if (a.dtype == DType.complex64) {
        final w2D = NDArray<Complex>.create([n], DType.complex64);
        final vr2D = NDArray<Complex>.create([n, n], DType.complex64);

        final info = LAPACKE_cgeev(
          101, // ROW_MAJOR
          jobvl,
          jobvr,
          n,
          sliceCopy.pointer.cast<ffi.Float>(),
          n,
          w2D.pointer.cast<ffi.Float>(),
          ffi.nullptr.cast<ffi.Float>(),
          n,
          vr2D.pointer.cast<ffi.Float>(),
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

        final w2DData = w2D.data;
        final strideWLast = w.strides.isEmpty ? 1 : w.strides.last;
        for (var j = 0; j < n; j++) {
          w.data[offsetW + j * strideWLast] = w2DData[j];
        }

        final vr2DData = vr2D.data;
        final strideVR1 = vr.strides[rank - 2];
        final strideVR2 = vr.strides[rank - 1];
        for (var r = 0; r < n; r++) {
          for (var c = 0; c < n; c++) {
            vr.data[offsetVR + r * strideVR1 + c * strideVR2] =
                vr2DData[r * n + c];
          }
        }
        w2D.dispose();
        vr2D.dispose();
      } else if (a.dtype == DType.float64 ||
          a.dtype == DType.int32 ||
          a.dtype == DType.int64) {
        final wr = NDArray<double>.zeros([n], DType.float64);
        final wi = NDArray<double>.zeros([n], DType.float64);
        final vrReal = NDArray<double>.create([n, n], DType.float64);

        final info = LAPACKE_dgeev(
          101,
          jobvl,
          jobvr,
          n,
          sliceCopy.pointer.cast<ffi.Double>(),
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

        final strideWLast = w.strides.isEmpty ? 1 : w.strides.last;
        for (var j = 0; j < n; j++) {
          w.data[offsetW + j * strideWLast] = Complex(wr.data[j], wi.data[j]);
        }

        final strideVR1 = vr.strides[rank - 2];
        final strideVR2 = vr.strides[rank - 1];
        var j = 0;
        while (j < n) {
          if (wi.data[j] == 0.0) {
            for (var r = 0; r < n; r++) {
              vr.data[offsetVR + r * strideVR1 + j * strideVR2] = Complex(
                vrReal.data[r * n + j],
                0.0,
              );
            }
            j++;
          } else {
            for (var r = 0; r < n; r++) {
              final realPart = vrReal.data[r * n + j];
              final imagPart = vrReal.data[r * n + j + 1];
              vr.data[offsetVR + r * strideVR1 + j * strideVR2] = Complex(
                realPart,
                imagPart,
              );
              vr.data[offsetVR + r * strideVR1 + (j + 1) * strideVR2] = Complex(
                realPart,
                -imagPart,
              );
            }
            j += 2;
          }
        }

        wr.dispose();
        wi.dispose();
        vrReal.dispose();
      } else if (a.dtype == DType.float32) {
        final wr = NDArray<double>.zeros([n], DType.float32);
        final wi = NDArray<double>.zeros([n], DType.float32);
        final vrReal = NDArray<double>.create([n, n], DType.float32);

        final info = LAPACKE_sgeev(
          101,
          jobvl,
          jobvr,
          n,
          sliceCopy.pointer.cast<ffi.Float>(),
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

        final strideWLast = w.strides.isEmpty ? 1 : w.strides.last;
        for (var j = 0; j < n; j++) {
          w.data[offsetW + j * strideWLast] = Complex(wr.data[j], wi.data[j]);
        }

        final strideVR1 = vr.strides[rank - 2];
        final strideVR2 = vr.strides[rank - 1];
        var j = 0;
        while (j < n) {
          if (wi.data[j] == 0.0) {
            for (var r = 0; r < n; r++) {
              vr.data[offsetVR + r * strideVR1 + j * strideVR2] = Complex(
                vrReal.data[r * n + j],
                0.0,
              );
            }
            j++;
          } else {
            for (var r = 0; r < n; r++) {
              final realPart = vrReal.data[r * n + j];
              final imagPart = vrReal.data[r * n + j + 1];
              vr.data[offsetVR + r * strideVR1 + j * strideVR2] = Complex(
                realPart,
                imagPart,
              );
              vr.data[offsetVR + r * strideVR1 + (j + 1) * strideVR2] = Complex(
                realPart,
                -imagPart,
              );
            }
            j += 2;
          }
        }

        wr.dispose();
        wi.dispose();
        vrReal.dispose();
      }
    });
  } finally {
    sliceCopy.dispose();
  }

  return {'eigenvalues': w, 'eigenvectors': vr};
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
NDArray<R> pinv<T, R>(NDArray<T> a, {double? rcond, NDArray<R>? out}) {
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
  final result = out ?? (NDArray.create(targetShape, a.dtype) as NDArray<R>);
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
      sPlus.setCell([i, i], castValue(1.0 / sVal, a.dtype));
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
NDArray<R> matrix_power<T, R>(NDArray<T> a, int n, {NDArray<R>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute matrix_power() on a disposed array.');
  }
  if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
    throw ArgumentError(
      'matrix_power is only defined for 2D square matrices (was shape ${a.shape}).',
    );
  }

  final size = a.shape[0];
  final result = out ?? (NDArray.create(a.shape, a.dtype) as NDArray<R>);
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
