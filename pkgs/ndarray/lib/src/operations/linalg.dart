// ignore_for_file: non_constant_identifier_names
import 'dart:typed_data';
import 'dart:math' as math;
import '../ndarray.dart';
import 'package:openblas/openblas.dart';
import 'dart:ffi' as ffi;
import '../scratch_arena.dart';
import '../exceptions.dart';
import '../ndarray_extensions_bindings.dart';
import '../ndarray_bindings.dart'
    hide
        s_det_double,
        s_det_float,
        s_det_complex_double,
        s_det_complex_float,
        s_slogdet_double,
        s_slogdet_float,
        s_slogdet_complex_double,
        s_slogdet_complex_float;

// Standalone operational relative cross-imports
import 'math.dart';
import 'helpers.dart';

/// Matrix multiplication using OpenBLAS, supporting high-dimensional stack broadcasting and 1D vector promotions.
NDArray<R> matmul<Ta, Tb, R>(NDArray<Ta> a, NDArray<Tb> b, {NDArray<R>? out}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute matmul() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write matmul result to a disposed output array.');
  }
  final targetDType = resolveDType(a.dtype, b.dtype);

  NDArray? aCast;
  NDArray? bCast;
  NDArray? aCopy;
  NDArray? bCopy;
  NDArray<R>? result;
  var success = false;

  try {
    aCast = a.dtype == targetDType ? a : castNDArray(a, targetDType);
    bCast = b.dtype == targetDType ? b : castNDArray(b, targetDType);

    if (aCast.shape.length == 1 && bCast.shape.length == 1) {
      final n = aCast.shape[0];
      if (n != bCast.shape[0]) {
        throw ArgumentError(
          'Incompatible vector dimensions for 1D dot product in matmul: ${aCast.shape} and ${bCast.shape}',
        );
      }
      if (out != null) {
        if (!listEquals(out.shape, []) || out.dtype != targetDType) {
          throw ArgumentError(
            'Provided out buffer has incompatible shape or dtype (expected shape [] and dtype $targetDType, got shape ${out.shape} and dtype ${out.dtype}).',
          );
        }
      }
      switch (targetDType) {
        case DType.float64:
          final scalarRes = cblas_ddot(
            n,
            aCast.pointer.cast<ffi.Double>(),
            1,
            bCast.pointer.cast<ffi.Double>(),
            1,
          );
          result =
              out ?? (NDArray.scalar(scalarRes, DType.float64) as NDArray<R>);
          if (out != null) {
            out.data[0] = scalarRes as R;
          }
          success = true;
          return result;
        case DType.float32:
          final scalarRes = cblas_sdot(
            n,
            aCast.pointer.cast<ffi.Float>(),
            1,
            bCast.pointer.cast<ffi.Float>(),
            1,
          );
          result =
              out ?? (NDArray.scalar(scalarRes, DType.float32) as NDArray<R>);
          if (out != null) {
            out.data[0] = scalarRes as R;
          }
          success = true;
          return result;
        default:
          break;
      }
    }

    // Copy upfront ONLY if neither inner strides is 1 (very rare custom sliced strides)
    if (aCast.shape.length >= 2) {
      final r = aCast.shape.length;
      if (aCast.strides[r - 1] != 1 && aCast.strides[r - 2] != 1) {
        aCopy = aCast.copy();
      }
    }
    if (bCast.shape.length >= 2) {
      final r = bCast.shape.length;
      if (bCast.strides[r - 1] != 1 && bCast.strides[r - 2] != 1) {
        bCopy = bCast.copy();
      }
    }

    final aToUse = aCopy ?? aCast;
    final bToUse = bCopy ?? bCast;

    var aPromoted = false;
    var bPromoted = false;

    NDArray aView = aToUse;
    if (aToUse.shape.length == 1) {
      aView = NDArray.view(
        aToUse,
        shape: [1, aToUse.shape[0]],
        strides: [0, aToUse.strides[0]],
        offsetElements: 0,
      );
      aPromoted = true;
    }

    NDArray bView = bToUse;
    if (bToUse.shape.length == 1) {
      bView = NDArray.view(
        bToUse,
        shape: [bToUse.shape[0], 1],
        strides: [bToUse.strides[0], 0],
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
        'Incompatible inner matrix dimensions for matmul: kA($kA) != kB($kB). Shapes: ${aCast.shape} and ${bCast.shape}',
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
      if (!out.isContiguous) {
        throw ArgumentError('out buffer must be contiguous.');
      }
      if (!listEquals(out.shape, expectedFinalShape) ||
          out.dtype != targetDType) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype (expected shape $expectedFinalShape and dtype $targetDType, got shape ${out.shape} and dtype ${out.dtype}).',
        );
      }
    }

    final resShape = [...broadcastStack, m, n];
    result = NDArray.zeros(resShape, targetDType as DType<R>);

    // Stride resolution logic for 100% copy-free BLAS matrix multiplication
    var transA = 111; // CblasNoTrans
    var lda = kA;
    if (!aPromoted) {
      if (aView.strides[rankA - 1] == 1) {
        transA = 111;
        lda = math.max(aView.strides[rankA - 2], kA);
      } else if (aView.strides[rankA - 2] == 1) {
        transA = 112; // CblasTrans
        lda = math.max(aView.strides[rankA - 1], m);
      }
    }

    var transB = 111; // CblasNoTrans
    var ldb = n;
    if (!bPromoted) {
      if (bView.strides[rankB - 1] == 1) {
        transB = 111;
        ldb = math.max(bView.strides[rankB - 2], n);
      } else if (bView.strides[rankB - 2] == 1) {
        transB = 112; // CblasTrans
        ldb = math.max(bView.strides[rankB - 1], kB);
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

    final marker = ScratchArena.marker;
    try {
      ffi.Pointer<ffi.Double> alphaZ = ffi.nullptr.cast();
      ffi.Pointer<ffi.Double> betaZ = ffi.nullptr.cast();
      ffi.Pointer<ffi.Float> alphaC = ffi.nullptr.cast();
      ffi.Pointer<ffi.Float> betaC = ffi.nullptr.cast();

      switch (targetDType) {
        case DType.complex128:
          alphaZ = ScratchArena.allocate<ffi.Double>(
            2 * ffi.sizeOf<ffi.Double>(),
          );
          alphaZ[0] = 1.0;
          alphaZ[1] = 0.0;
          betaZ = ScratchArena.allocate<ffi.Double>(
            2 * ffi.sizeOf<ffi.Double>(),
          );
          betaZ[0] = 0.0;
          betaZ[1] = 0.0;
        case DType.complex64:
          alphaC = ScratchArena.allocate<ffi.Float>(
            2 * ffi.sizeOf<ffi.Float>(),
          );
          alphaC[0] = 1.0;
          alphaC[1] = 0.0;
          betaC = ScratchArena.allocate<ffi.Float>(2 * ffi.sizeOf<ffi.Float>());
          betaC[0] = 0.0;
          betaC[1] = 0.0;
        default:
          break;
      }

      void walk(int dim, int offsetA, int offsetB, int offsetRes) {
        if (dim == lenResult) {
          switch (targetDType) {
            case DType.float64:
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
                result!.pointer.cast<ffi.Double>() + offsetRes,
                n, // ldc (result is always contiguous row-major)
              );
            case DType.float32:
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
                result!.pointer.cast<ffi.Float>() + offsetRes,
                n, // ldc (result is always contiguous row-major)
              );
            case DType.complex128:
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
                result!.pointer.cast<ffi.Double>() + (offsetRes * 2),
                n,
              );
            case DType.complex64:
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
                result!.pointer.cast<ffi.Float>() + (offsetRes * 2),
                n,
              );
            case DType.int64:
            case DType.int32:
            case DType.int16:
            case DType.uint8:
              final strideARow = aView.strides[rankA - 2];
              final strideACol = aView.strides[rankA - 1];

              final strideBRow = bView.strides[rankB - 2];
              final strideBCol = bView.strides[rankB - 1];

              final strideResRow = result!.strides[resShape.length - 2];
              final strideResCol = result.strides[resShape.length - 1];

              switch (targetDType) {
                case DType.int64:
                  matmul_int64(
                    result.pointer.cast<ffi.Int64>() + offsetRes,
                    strideResRow,
                    strideResCol,
                    aView.pointer.cast<ffi.Int64>() + offsetA,
                    strideARow,
                    strideACol,
                    bView.pointer.cast<ffi.Int64>() + offsetB,
                    strideBRow,
                    strideBCol,
                    m,
                    n,
                    kA,
                  );
                case DType.int32:
                  matmul_int32(
                    result.pointer.cast<ffi.Int32>() + offsetRes,
                    strideResRow,
                    strideResCol,
                    aView.pointer.cast<ffi.Int32>() + offsetA,
                    strideARow,
                    strideACol,
                    bView.pointer.cast<ffi.Int32>() + offsetB,
                    strideBRow,
                    strideBCol,
                    m,
                    n,
                    kA,
                  );
                case DType.int16:
                  matmul_int16(
                    result.pointer.cast<ffi.Int16>() + offsetRes,
                    strideResRow,
                    strideResCol,
                    aView.pointer.cast<ffi.Int16>() + offsetA,
                    strideARow,
                    strideACol,
                    bView.pointer.cast<ffi.Int16>() + offsetB,
                    strideBRow,
                    strideBCol,
                    m,
                    n,
                    kA,
                  );
                case DType.uint8:
                  matmul_uint8(
                    result.pointer.cast<ffi.Uint8>() + offsetRes,
                    strideResRow,
                    strideResCol,
                    aView.pointer.cast<ffi.Uint8>() + offsetA,
                    strideARow,
                    strideACol,
                    bView.pointer.cast<ffi.Uint8>() + offsetB,
                    strideBRow,
                    strideBCol,
                    m,
                    n,
                    kA,
                  );
                default:
                  throw UnsupportedError(
                    'Unsupported integer type: $targetDType',
                  );
              }
            default:
              throw UnsupportedError('Unsupported type: $targetDType');
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
    } finally {
      ScratchArena.reset(marker);
    }

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
          out.data[i] = resFlat[i];
        }
      }
      result.dispose();
      success = true;
      return out;
    }

    // Post-calculation 1D dummy dimensions demotions
    if (aPromoted && bPromoted) {
      final finalRes = result.reshape([]);
      success = true;
      return finalRes; // 0D scalar array for pure vector dot products
    } else if (aPromoted) {
      final newShape = List<int>.from(result.shape)
        ..removeAt(result.shape.length - 2);
      final finalRes = result.reshape(newShape);
      success = true;
      return finalRes;
    } else if (bPromoted) {
      final newShape = List<int>.from(result.shape)
        ..removeAt(result.shape.length - 1);
      final finalRes = result.reshape(newShape);
      success = true;
      return finalRes;
    }

    success = true;
    return result;
  } finally {
    if (aCast != null && aCast != a) aCast.dispose();
    if (bCast != null && bCast != b) bCast.dispose();
    aCopy?.dispose();
    bCopy?.dispose();
    if (!success) {
      result?.dispose();
    }
  }
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
  for (final a in arrays) {
    if (a.isDisposed) {
      throw StateError(
        'Cannot execute multi_dot() with a disposed array in the list.',
      );
    }
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write multi_dot result to a disposed output array.',
    );
  }
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

  return NDArray.scope(() {
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

    return finalResult.detachToParentScope();
  });
}

/// Computes the multiplicative inverse of a square 2D matrix.
///
/// Uses OpenBLAS LAPACK LU decomposition routines
/// (`LAPACKE_dgetrf`/`LAPACKE_dgetri` for Float64, and `LAPACKE_sgetrf`/`LAPACKE_sgetri` for Float32).
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
NDArray<T> inv<T>(NDArray<T> a, {NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute inverse of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write inverse to a disposed output array.');
  }
  if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
    throw ArgumentError('Matrix must be square and 2D (was ${a.shape})');
  }

  if (a.dtype != DType.float32 &&
      a.dtype != DType.float64 &&
      a.dtype != DType.complex64 &&
      a.dtype != DType.complex128) {
    throw ArgumentError(
      'Matrix inversion only supports float or complex dtypes (got ${a.dtype}).',
    );
  }
  final n = a.shape[0];
  final DType<T> targetDType = a.dtype;

  if (out != null) {
    if (!out.isContiguous) {
      throw ArgumentError('out buffer must be contiguous.');
    }
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for matrix inversion.',
      );
    }
  }

  return NDArray.scope(() {
    final NDArray<T> result;
    if (out != null) {
      result = out;
      a.copy(out: result);
    } else {
      result = a.copy();
    }

    final marker = ScratchArena.marker;
    final ipiv = ScratchArena.allocate<ffi.Int>(n * ffi.sizeOf<ffi.Int>());

    try {
      switch (targetDType) {
        case DType.float32:
          final info = LAPACKE_sgetrf(
            101,
            n,
            n,
            result.pointer.cast<ffi.Float>(),
            n,
            ipiv,
          );
          if (info < 0) {
            throw ArgumentError(
              'Illegal value in call to LAPACKE_sgetrf: $info',
            );
          }
          if (info > 0) {
            throw SingularMatrixException(
              'Matrix is singular and cannot be inverted',
            );
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
        case DType.float64:
          final info = LAPACKE_dgetrf(
            101,
            n,
            n,
            result.pointer.cast<ffi.Double>(),
            n,
            ipiv,
          );
          if (info < 0) {
            throw ArgumentError(
              'Illegal value in call to LAPACKE_dgetrf: $info',
            );
          }
          if (info > 0) {
            throw SingularMatrixException(
              'Matrix is singular and cannot be inverted',
            );
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
        case DType.complex64:
          final info = LAPACKE_cgetrf(
            101,
            n,
            n,
            result.pointer.cast<ffi.Float>(),
            n,
            ipiv,
          );
          if (info < 0) {
            throw ArgumentError(
              'Illegal value in call to LAPACKE_cgetrf: $info',
            );
          }
          if (info > 0) {
            throw SingularMatrixException(
              'Matrix is singular and cannot be inverted',
            );
          }
          final infoTri = LAPACKE_cgetri(
            101,
            n,
            result.pointer.cast<ffi.Float>(),
            n,
            ipiv,
          );
          if (infoTri < 0) {
            throw ArgumentError(
              'Illegal value in call to LAPACKE_cgetri: $infoTri',
            );
          }
        case DType.complex128:
          final info = LAPACKE_zgetrf(
            101,
            n,
            n,
            result.pointer.cast<ffi.Double>(),
            n,
            ipiv,
          );
          if (info < 0) {
            throw ArgumentError(
              'Illegal value in call to LAPACKE_zgetrf: $info',
            );
          }
          if (info > 0) {
            throw SingularMatrixException(
              'Matrix is singular and cannot be inverted',
            );
          }
          final infoTri = LAPACKE_zgetri(
            101,
            n,
            result.pointer.cast<ffi.Double>(),
            n,
            ipiv,
          );
          if (infoTri < 0) {
            throw ArgumentError(
              'Illegal value in call to LAPACKE_zgetri: $infoTri',
            );
          }
        default:
          throw UnsupportedError(
            'Unsupported type for matrix inversion: $targetDType',
          );
      }

      if (out == null) {
        result.detachToParentScope();
      }
      return result;
    } finally {
      ScratchArena.reset(marker);
    }
  });
}

/// Computes the determinant of a square matrix or a stack of square matrices using OpenBLAS.
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
/// - Algorithmic complexity is $O(N^3)$ using LAPACK linear algebra solvers.
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
/// Computes the determinant of a square matrix or a stack of square matrices using OpenBLAS/LAPACK.
///
/// Transforms the matrix and calculates its determinant natively via LAPACK LU decomposition.
/// Supports both real (float32, float64) and complex (complex64, complex128) data types.
/// Returns the determinant stack as an array of corresponding types (float64 for real inputs,
/// and complex64/complex128 for complex inputs).
///
/// **Preconditions:**
/// - Matrix [a] must be square in its last two dimensions (size $N \times N$) and at least 2-dimensional.
/// - Data type [a.dtype] must be float32, float64, complex64, or complex128.
///
/// **Throws:**
/// - [ArgumentError] if [a] is not square or less than 2D.
/// - [ArgumentError] if [a.dtype] is not a supported data type.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N^3)$ using LAPACK linear algebra solvers.
/// - Fully vectorized and batched in native C for float64, complex64, and complex128, minimizing FFI transitions.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final d = det(a);
/// print(d); // -2.0
/// ```
/// Computes the determinant of a square matrix or a stack of square matrices using OpenBLAS/LAPACK.
///
/// Transforms the matrix and calculates its determinant natively via LAPACK LU decomposition.
/// Supports both real (float32, float64) and complex (complex64, complex128) data types.
/// Returns the determinant stack as an array of corresponding types (float64 for real inputs,
/// and complex64/complex128 for complex inputs).
///
/// **Preconditions:**
/// - Matrix [a] must be square in its last two dimensions (size $N \times N$) and at least 2-dimensional.
/// - Data type [a.dtype] must be float32, float64, complex64, or complex128.
///
/// **Throws:**
/// - [ArgumentError] if [a] is not square or less than 2D.
/// - [ArgumentError] if [a.dtype] is not a supported data type.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N^3)$ using LAPACK linear algebra solvers.
/// - Fully vectorized and batched in native C for float64, complex64, and complex128, minimizing FFI transitions.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final d = det(a);
/// print(d.data); // [-2.0] (0-D array)
/// ```
///
/// Refer to the [determinant](https://en.wikipedia.org/wiki/Determinant)
/// and [LAPACK LU solver](https://en.wikipedia.org/wiki/LU_decomposition) for additional details.
///
/// Returns a 0-dimensional [NDArray] if [a] is a 2D matrix, or a new [NDArray] with stack dimensions if [a] is a stack of matrices.
NDArray<T> det<T>(NDArray<T> a, {NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute determinant of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write determinant to a disposed output array.');
  }
  if (a.dtype != DType.float64 &&
      a.dtype != DType.float32 &&
      a.dtype != DType.complex128 &&
      a.dtype != DType.complex64) {
    throw ArgumentError('det only supports float and complex dtypes');
  }
  final rank = a.shape.length;
  if (rank < 2 || a.shape[rank - 1] != a.shape[rank - 2]) {
    throw ArgumentError(
      'Matrix must be square and at least 2D (was ${a.shape})',
    );
  }
  final stackShape = a.shape.sublist(0, rank - 2);
  final expectedDType = a.dtype;

  if (out != null) {
    if (!listEquals(out.shape, stackShape) || out.dtype != expectedDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
    if (!out.isContiguous) {
      throw ArgumentError('Provided out buffer must be contiguous.');
    }
  }

  return NDArray.scope(() {
    switch (a.dtype) {
      case DType.float64:
        final result =
            out ?? (NDArray.zeros(stackShape, DType.float64) as NDArray<T>);
        final marker = ScratchArena.marker;
        try {
          final cStridesA = ScratchArena.copyInts(a.strides);
          final cStridesRes = ScratchArena.copyInts(result.strides);
          final cShape = ScratchArena.copyInts(a.shape);

          final n = a.shape[rank - 1];
          final cCopy = ScratchArena.allocate<ffi.Double>(
            n * n * ffi.sizeOf<ffi.Double>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );

          s_det_double(
            a.pointer.cast<ffi.Double>(),
            cStridesA,
            result.pointer.cast<ffi.Double>(),
            cStridesRes,
            cShape,
            rank,
            cCopy,
            cIpiv,
            get_dgetrf_ptr(),
          );
        } finally {
          ScratchArena.reset(marker);
        }
        if (out == null) {
          result.detachToParentScope();
        }
        return result;
      case DType.complex128:
        final result =
            out ?? (NDArray.zeros(stackShape, DType.complex128) as NDArray<T>);
        final marker = ScratchArena.marker;
        try {
          final cStridesA = ScratchArena.copyInts(a.strides);
          final cStridesRes = ScratchArena.copyInts(result.strides);
          final cShape = ScratchArena.copyInts(a.shape);

          final n = a.shape[rank - 1];
          final cCopy = ScratchArena.allocate<ffi.Double>(
            2 * n * n * ffi.sizeOf<ffi.Double>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );

          s_det_complex_double(
            a.pointer.cast<ffi.Double>(),
            cStridesA,
            result.pointer.cast<ffi.Double>(),
            cStridesRes,
            cShape,
            rank,
            cCopy,
            cIpiv,
            get_zgetrf_ptr(),
          );
        } finally {
          ScratchArena.reset(marker);
        }
        if (out == null) {
          result.detachToParentScope();
        }
        return result;
      case DType.complex64:
        final result =
            out ?? (NDArray.zeros(stackShape, DType.complex64) as NDArray<T>);
        final marker = ScratchArena.marker;
        try {
          final cStridesA = ScratchArena.copyInts(a.strides);
          final cStridesRes = ScratchArena.copyInts(result.strides);
          final cShape = ScratchArena.copyInts(a.shape);

          final n = a.shape[rank - 1];
          final cCopy = ScratchArena.allocate<ffi.Float>(
            2 * n * n * ffi.sizeOf<ffi.Float>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );

          s_det_complex_float(
            a.pointer.cast<ffi.Float>(),
            cStridesA,
            result.pointer.cast<ffi.Float>(),
            cStridesRes,
            cShape,
            rank,
            cCopy,
            cIpiv,
            get_cgetrf_ptr(),
          );
        } finally {
          ScratchArena.reset(marker);
        }
        if (out == null) {
          result.detachToParentScope();
        }
        return result;
      case DType.float32:
        final result =
            out ?? (NDArray.zeros(stackShape, DType.float32) as NDArray<T>);
        final marker = ScratchArena.marker;
        try {
          final cStridesA = ScratchArena.copyInts(a.strides);
          final cStridesRes = ScratchArena.copyInts(result.strides);
          final cShape = ScratchArena.copyInts(a.shape);

          final n = a.shape[rank - 1];
          final cCopy = ScratchArena.allocate<ffi.Float>(
            n * n * ffi.sizeOf<ffi.Float>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );

          s_det_float(
            a.pointer.cast<ffi.Float>(),
            cStridesA,
            result.pointer.cast<ffi.Float>(),
            cStridesRes,
            cShape,
            rank,
            cCopy,
            cIpiv,
            get_sgetrf_ptr(),
          );
        } finally {
          ScratchArena.reset(marker);
        }
        if (out == null) {
          result.detachToParentScope();
        }
        return result;
      default:
        throw ArgumentError('Unsupported dtype for determinant');
    }
  });
}

/// Computes the sign and natural logarithm of the absolute value of the determinant of a square 2D matrix or stack of matrices.
///
/// **Preconditions:**
/// - Input array [a] must have rank >= 2 and the last two dimensions must be square (`a.shape[rank - 1] == a.shape[rank - 2]`).
/// - The dtype of [a] must be [DType.float64], [DType.float32], [DType.complex128], or [DType.complex64].
/// - If provided, recycler buffers [outSign] and [outLogdet] must be contiguous, match the shape of the stack (`a.shape.sublist(0, rank - 2)`), and have correct dtypes:
///   - [outSign] must have the same dtype as [a].
///   - [outLogdet] must have the corresponding real dtype (e.g., [DType.float64] for [DType.float64]/[DType.complex128] inputs, or [DType.float32] for [DType.float32]/[DType.complex64] inputs).
///
/// **Throws:**
/// - [ArgumentError] if [a] rank < 2, or the last two dimensions are not square.
/// - [ArgumentError] if [a] dtype is unsupported.
/// - [ArgumentError] if [outSign] or [outLogdet] shape/dtype are incompatible.
///
/// **Returns:**
/// - A record `(sign, logdet)` of two NDArrays, representing the sign (or phase) and log of the absolute determinant.
///
/// Reference: [NumPy linalg.slogdet](https://numpy.org/doc/stable/reference/generated/numpy.linalg.slogdet.html)
(NDArray<T> sign, NDArray<R> logdet) slogdet<T, R extends num>(
  NDArray<T> a, {
  NDArray<T>? outSign,
  NDArray<R>? outLogdet,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute slogdet of a disposed array.');
  }
  if (outSign != null && outSign.isDisposed) {
    throw StateError('Cannot write slogdet sign to a disposed output array.');
  }
  if (outLogdet != null && outLogdet.isDisposed) {
    throw StateError('Cannot write slogdet logdet to a disposed output array.');
  }
  if (a.dtype != DType.float64 &&
      a.dtype != DType.float32 &&
      a.dtype != DType.complex128 &&
      a.dtype != DType.complex64) {
    throw ArgumentError('slogdet only supports float and complex dtypes');
  }
  final rank = a.shape.length;
  if (rank < 2 || a.shape[rank - 1] != a.shape[rank - 2]) {
    throw ArgumentError(
      'Matrix must be square and at least 2D (was ${a.shape})',
    );
  }
  final stackShape = a.shape.sublist(0, rank - 2);

  final DType<R> logdetDType =
      (a.dtype == DType.float32 || a.dtype == DType.complex64)
      ? DType.float32 as DType<R>
      : DType.float64 as DType<R>;

  if (outSign != null) {
    if (!listEquals(outSign.shape, stackShape) || outSign.dtype != a.dtype) {
      throw ArgumentError(
        'Provided outSign buffer has incompatible shape or dtype.',
      );
    }
    if (!outSign.isContiguous) {
      throw ArgumentError('Provided outSign buffer must be contiguous.');
    }
  }

  if (outLogdet != null) {
    if (!listEquals(outLogdet.shape, stackShape) ||
        outLogdet.dtype != logdetDType) {
      throw ArgumentError(
        'Provided outLogdet buffer has incompatible shape or dtype.',
      );
    }
    if (!outLogdet.isContiguous) {
      throw ArgumentError('Provided outLogdet buffer must be contiguous.');
    }
  }

  return NDArray.scope(() {
    final signResult = outSign ?? NDArray<T>.zeros(stackShape, a.dtype);
    final logdetResult = outLogdet ?? NDArray<R>.zeros(stackShape, logdetDType);

    final marker = ScratchArena.marker;
    try {
      final cStridesA = ScratchArena.copyInts(a.strides);
      final cStridesSign = ScratchArena.copyInts(signResult.strides);
      final cStridesLogdet = ScratchArena.copyInts(logdetResult.strides);
      final cShape = ScratchArena.copyInts(a.shape);

      final n = a.shape[rank - 1];

      switch (a.dtype) {
        case DType.float64:
          final cCopy = ScratchArena.allocate<ffi.Double>(
            n * n * ffi.sizeOf<ffi.Double>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );
          s_slogdet_double(
            a.pointer.cast<ffi.Double>(),
            cStridesA,
            signResult.pointer.cast<ffi.Double>(),
            cStridesSign,
            logdetResult.pointer.cast<ffi.Double>(),
            cStridesLogdet,
            cShape,
            rank,
            cCopy,
            cIpiv,
            get_dgetrf_ptr(),
          );
        case DType.float32:
          final cCopy = ScratchArena.allocate<ffi.Float>(
            n * n * ffi.sizeOf<ffi.Float>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );
          s_slogdet_float(
            a.pointer.cast<ffi.Float>(),
            cStridesA,
            signResult.pointer.cast<ffi.Float>(),
            cStridesSign,
            logdetResult.pointer.cast<ffi.Float>(),
            cStridesLogdet,
            cShape,
            rank,
            cCopy,
            cIpiv,
            get_sgetrf_ptr(),
          );
        case DType.complex128:
          final cCopy = ScratchArena.allocate<ffi.Double>(
            2 * n * n * ffi.sizeOf<ffi.Double>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );
          s_slogdet_complex_double(
            a.pointer.cast<ffi.Double>(),
            cStridesA,
            signResult.pointer.cast<ffi.Double>(),
            cStridesSign,
            logdetResult.pointer.cast<ffi.Double>(),
            cStridesLogdet,
            cShape,
            rank,
            cCopy,
            cIpiv,
            get_zgetrf_ptr(),
          );
        case DType.complex64:
          final cCopy = ScratchArena.allocate<ffi.Float>(
            2 * n * n * ffi.sizeOf<ffi.Float>(),
          );
          final cIpiv = ScratchArena.allocate<ffi.Int>(
            n * ffi.sizeOf<ffi.Int>(),
          );
          s_slogdet_complex_float(
            a.pointer.cast<ffi.Float>(),
            cStridesA,
            signResult.pointer.cast<ffi.Float>(),
            cStridesSign,
            logdetResult.pointer.cast<ffi.Float>(),
            cStridesLogdet,
            cShape,
            rank,
            cCopy,
            cIpiv,
            get_cgetrf_ptr(),
          );
        default:
          throw UnsupportedError('Unsupported dtype ${a.dtype}');
      }
    } finally {
      ScratchArena.reset(marker);
    }

    if (outSign == null) {
      signResult.detachToParentScope();
    }
    if (outLogdet == null) {
      logdetResult.detachToParentScope();
    }

    return (signResult, logdetResult);
  });
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
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N^3)$ executed natively.
///
/// **Example:**
/// ```dart
/// final a = NDArray<double>.fromList([3.0, 1.0, 1.0, 2.0], [2, 2], DType.float64);
/// final b = NDArray<double>.fromList([9.0, 8.0], [2], DType.float64);
/// final x = solve(a, b);
/// print(x.toList()); // [2.0, 3.0]
/// ```
NDArray<T> solve<T>(NDArray<T> a, NDArray<T> b, {NDArray<T>? out}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute solve() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write solve result to a disposed output array.');
  }
  if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
    throw ArgumentError('Matrix a must be square and 2D (was ${a.shape})');
  }
  final n = a.shape[0];

  if (b.shape.isEmpty || b.shape[0] != n) {
    throw ArgumentError(
      'Dimensions of b must match dimensions of a (expected first dimension $n, was ${b.shape.isEmpty ? 0 : b.shape[0]})',
    );
  }

  if (a.dtype != b.dtype) {
    throw ArgumentError(
      'Mismatched dtypes for solve: a has dtype ${a.dtype}, b has dtype ${b.dtype}.',
    );
  }

  if (a.dtype != DType.float64 &&
      a.dtype != DType.float32 &&
      a.dtype != DType.complex128 &&
      a.dtype != DType.complex64) {
    throw ArgumentError(
      'solve only supports float64, float32, complex128, or complex64 dtypes (got ${a.dtype}).',
    );
  }

  if (out != null) {
    if (!listEquals(out.shape, b.shape) || out.dtype != b.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype (expected shape ${b.shape} and dtype ${b.dtype}, got shape ${out.shape} and dtype ${out.dtype}).',
      );
    }
    if (!out.isContiguous) {
      throw ArgumentError('Provided out buffer must be contiguous.');
    }
  }

  return NDArray.scope(() {
    final nrhs = b.shape.length > 1 ? b.shape[1] : 1;
    final marker = ScratchArena.marker;
    final ipiv = ScratchArena.allocate<ffi.Int>(n * ffi.sizeOf<ffi.Int>());

    try {
      // Prepare a copy of a because LAPACKE_*gesv mutates a
      final aCopy = a.copy();

      // Prepare bCopy: if out is provided, copy b into it and use it as bCopy;
      // otherwise create a new copy.
      final NDArray<T> bCopy;
      if (out != null) {
        bCopy = out;
        b.copy(out: bCopy);
      } else {
        bCopy = b.copy();
      }

      switch (a.dtype) {
        case DType.float64:
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
            throw ArgumentError(
              'Illegal value in call to LAPACKE_dgesv: $info',
            );
          }
          if (info > 0) {
            throw SingularMatrixException(
              'Matrix is singular and cannot be solved',
            );
          }
        case DType.float32:
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
            throw ArgumentError(
              'Illegal value in call to LAPACKE_sgesv: $info',
            );
          }
          if (info > 0) {
            throw SingularMatrixException(
              'Matrix is singular and cannot be solved',
            );
          }
        case DType.complex128:
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
            throw ArgumentError(
              'Illegal value in call to LAPACKE_zgesv: $info',
            );
          }
          if (info > 0) {
            throw SingularMatrixException(
              'Matrix is singular and cannot be solved',
            );
          }
        case DType.complex64:
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
            throw ArgumentError(
              'Illegal value in call to LAPACKE_cgesv: $info',
            );
          }
          if (info > 0) {
            throw SingularMatrixException(
              'Matrix is singular and cannot be solved',
            );
          }
        default:
          throw UnimplementedError('Type ${a.dtype} not supported for solve');
      }

      if (out == null) {
        bCopy.detachToParentScope();
      }
      return bCopy;
    } finally {
      ScratchArena.reset(marker);
    }
  });
}

/// Computes the eigenvalues and right eigenvectors of a square array or stack of square arrays.
///
/// Returns a record `(eigenvalues, eigenvectors)` containing:
/// - **eigenvalues**: An `NDArray<Complex>` of shape `[..., N]` containing the eigenvalues.
/// - **eigenvectors**: An `NDArray<Complex>` of shape `[..., N, N]` containing the corresponding right eigenvectors as columns.
///
/// Both are returned with `Complex` elements because eigenvalues and eigenvectors can be complex
/// even for real matrices.
///
/// **Preconditions:**
/// - Matrix [a] must be square in its last two dimensions (size $N \times N$) and at least 2-dimensional.
/// - If provided, [out] must contain two pre-allocated contiguous `NDArray<Complex>` buffers with shapes `[..., N]` and `[..., N, N]` respectively.
///
/// **Throws:**
/// - [ArgumentError] if [a] is not square or less than 2D.
/// - [UnimplementedError] if the DType of [a] is not supported.
({NDArray<Complex> eigenvalues, NDArray<Complex> eigenvectors}) eig<T>(
  NDArray<T> a, {
  ({NDArray<Complex> eigenvalues, NDArray<Complex> eigenvectors})? out,
}) {
  final rank = a.shape.length;
  if (rank < 2 || a.shape[rank - 1] != a.shape[rank - 2]) {
    throw ArgumentError(
      'Matrix must be square and at least 2D (was ${a.shape})',
    );
  }
  final n = a.shape[rank - 1];
  final stackShape = a.shape.sublist(0, rank - 2);

  final compDType = (a.dtype == DType.float32 || a.dtype == DType.complex64)
      ? DType.complex64
      : DType.complex128;

  final wShape = [...stackShape, n];
  final vrShape = [...stackShape, n, n];

  return NDArray.scope(() {
    final NDArray<Complex> w;
    final NDArray<Complex> vr;

    if (out != null) {
      w = out.eigenvalues;
      vr = out.eigenvectors;
      if (!listEquals(w.shape, wShape) || w.dtype != compDType) {
        throw ArgumentError(
          'Provided out eigenvalues buffer has incompatible shape or dtype (expected shape $wShape and dtype $compDType, got shape ${w.shape} and dtype ${w.dtype}).',
        );
      }
      if (!w.isContiguous) {
        throw ArgumentError(
          'Provided out eigenvalues buffer must be contiguous.',
        );
      }
      if (!listEquals(vr.shape, vrShape) || vr.dtype != compDType) {
        throw ArgumentError(
          'Provided out eigenvectors buffer has incompatible shape or dtype (expected shape $vrShape and dtype $compDType, got shape ${vr.shape} and dtype ${vr.dtype}).',
        );
      }
      if (!vr.isContiguous) {
        throw ArgumentError(
          'Provided out eigenvectors buffer must be contiguous.',
        );
      }
    } else {
      w = NDArray<Complex>.create(wShape, compDType);
      vr = NDArray<Complex>.create(vrShape, compDType);
    }

    final jobvl = 'N'.codeUnitAt(0);
    final jobvr = 'V'.codeUnitAt(0);

    if (a.dtype.isInteger) {
      throw ArgumentError(
        'Integer arrays are not supported directly for eigenvalue decomposition. '
        'Please convert the array to float64 or float32 manually.',
      );
    }

    if (a.dtype != DType.complex128 &&
        a.dtype != DType.complex64 &&
        a.dtype != DType.float64 &&
        a.dtype != DType.float32) {
      throw UnimplementedError('Type ${a.dtype} not supported for eig');
    }

    final NDArray src = a;
    walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
      coords,
    ) {
      var offsetA = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetA += coords[i] * src.strides[i];
      }

      final sliceView = NDArray.view(
        src,
        shape: [n, n],
        strides: src.strides.sublist(rank - 2),
        offsetElements: offsetA,
      );
      final sliceCopy = sliceView.copy();

      var offsetW = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetW += coords[i] * w.strides[i];
      }
      var offsetVR = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetVR += coords[i] * vr.strides[i];
      }

      switch (src.dtype) {
        case DType.complex128:
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
            throw ArgumentError(
              'Illegal value in call to LAPACKE_zgeev: $info',
            );
          }
          if (info > 0) {
            throw ArgumentError(
              'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
            );
          }

          final wView = NDArray<Complex>.view(
            w,
            shape: [n],
            strides: w.strides.isEmpty ? [1] : [w.strides.last],
            offsetElements: offsetW,
          );
          w2D.copy(out: wView);

          final vrView = NDArray<Complex>.view(
            vr,
            shape: [n, n],
            strides: vr.strides.sublist(rank - 2),
            offsetElements: offsetVR,
          );
          vr2D.copy(out: vrView);

          w2D.dispose();
          vr2D.dispose();
        case DType.complex64:
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
            throw ArgumentError(
              'Illegal value in call to LAPACKE_cgeev: $info',
            );
          }
          if (info > 0) {
            throw ArgumentError(
              'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
            );
          }

          final wView = NDArray<Complex>.view(
            w,
            shape: [n],
            strides: w.strides.isEmpty ? [1] : [w.strides.last],
            offsetElements: offsetW,
          );
          w2D.copy(out: wView);

          final vrView = NDArray<Complex>.view(
            vr,
            shape: [n, n],
            strides: vr.strides.sublist(rank - 2),
            offsetElements: offsetVR,
          );
          vr2D.copy(out: vrView);

          w2D.dispose();
          vr2D.dispose();
        case DType.float64:
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
            throw ArgumentError(
              'Illegal value in call to LAPACKE_dgeev: $info',
            );
          }
          if (info > 0) {
            throw ArgumentError(
              'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
            );
          }

          final strideWLast = w.strides.isEmpty ? 1 : w.strides.last;
          final strideVR1 = vr.strides[rank - 2];
          final strideVR2 = vr.strides[rank - 1];
          assemble_eigenvectors_double(
            w.pointer.cast<cpx_t>() + offsetW,
            strideWLast,
            vr.pointer.cast<cpx_t>() + offsetVR,
            strideVR1,
            strideVR2,
            wr.pointer.cast<ffi.Double>(),
            wi.pointer.cast<ffi.Double>(),
            vrReal.pointer.cast<ffi.Double>(),
            n,
          );

          wr.dispose();
          wi.dispose();
          vrReal.dispose();
        case DType.float32:
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
            throw ArgumentError(
              'Illegal value in call to LAPACKE_sgeev: $info',
            );
          }
          if (info > 0) {
            throw ArgumentError(
              'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
            );
          }

          final strideWLast = w.strides.isEmpty ? 1 : w.strides.last;
          final strideVR1 = vr.strides[rank - 2];
          final strideVR2 = vr.strides[rank - 1];
          assemble_eigenvectors_float(
            w.pointer.cast<cpx_f_t>() + offsetW,
            strideWLast,
            vr.pointer.cast<cpx_f_t>() + offsetVR,
            strideVR1,
            strideVR2,
            wr.pointer.cast<ffi.Float>(),
            wi.pointer.cast<ffi.Float>(),
            vrReal.pointer.cast<ffi.Float>(),
            n,
          );

          wr.dispose();
          wi.dispose();
          vrReal.dispose();
        default:
          throw UnimplementedError('Type ${src.dtype} not supported for eig');
      }
      sliceCopy.dispose();
    });

    if (out == null) {
      w.detachToParentScope();
      vr.detachToParentScope();
    }
    return (eigenvalues: w, eigenvectors: vr);
  });
}

/// Extension on eigenvalue decomposition result record type to support easy disposal of both arrays.
extension EigRecordDispose
    on ({NDArray<Complex> eigenvalues, NDArray<Complex> eigenvectors}) {
  /// Disposes both [eigenvalues] and [eigenvectors] simultaneously,
  /// freeing their underlying unmanaged C memory.
  ///
  /// Call this method when both matrices are no longer needed to avoid native memory leaks.
  void dispose() {
    eigenvalues.dispose();
    eigenvectors.dispose();
  }
}

/// Computes only the eigenvalues of a general square 2D matrix or stack of matrices.
///
/// Unlike [eig], this function does not compute eigenvectors, making it much faster.
///
/// **Preconditions:**
/// - Input matrix [a] must be square and at least 2-dimensional (was shape stack x N x N).
/// - Matrix [a] cannot have integer data type (throws [ArgumentError]; cast to float manually).
/// - If provided, the [out] buffer must be contiguous, match the expected shape of the eigenvalues stack,
///   and have the correct complex dtype ([DType.complex64] for Float32/Complex64 inputs, or [DType.complex128] for Float64/Complex128 inputs).
///
/// **Throws:**
/// - [ArgumentError] if [a] is not square or rank < 2.
/// - [ArgumentError] if [a] has integer dtype.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
/// - [UnimplementedError] if [a] dtype is unsupported.
///
/// **Returns:**
/// - A contiguous `NDArray<Complex>` containing the computed eigenvalues.
///
/// Reference: [NumPy linalg.eigvals](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvals.html)
NDArray<Complex> eigvals<T>(NDArray<T> a, {NDArray<Complex>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot compute eigvals of a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write eigvals result to a disposed output array.');
  }
  final rank = a.shape.length;
  if (rank < 2 || a.shape[rank - 1] != a.shape[rank - 2]) {
    throw ArgumentError(
      'Matrix must be square and at least 2D (was ${a.shape})',
    );
  }
  final n = a.shape[rank - 1];
  final stackShape = a.shape.sublist(0, rank - 2);

  final compDType = (a.dtype == DType.float32 || a.dtype == DType.complex64)
      ? DType.complex64
      : DType.complex128;

  final wShape = [...stackShape, n];

  return NDArray.scope(() {
    final NDArray<Complex> w;

    if (out != null) {
      w = out;
      if (!listEquals(w.shape, wShape) || w.dtype != compDType) {
        throw ArgumentError(
          'Provided out eigenvalues buffer has incompatible shape or dtype (expected shape $wShape and dtype $compDType, got shape ${w.shape} and dtype ${w.dtype}).',
        );
      }
      if (!w.isContiguous) {
        throw ArgumentError(
          'Provided out eigenvalues buffer must be contiguous.',
        );
      }
    } else {
      w = NDArray<Complex>.create(wShape, compDType);
    }

    final jobvl = 'N'.codeUnitAt(0);
    final jobvr = 'N'.codeUnitAt(0);

    if (a.dtype.isInteger) {
      throw ArgumentError(
        'Integer arrays are not supported directly for eigenvalue decomposition. '
        'Please convert the array to float64 or float32 manually.',
      );
    }

    if (a.dtype != DType.complex128 &&
        a.dtype != DType.complex64 &&
        a.dtype != DType.float64 &&
        a.dtype != DType.float32) {
      throw UnimplementedError('Type ${a.dtype} not supported for eigvals');
    }

    final NDArray src = a;
    walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
      coords,
    ) {
      var offsetA = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetA += coords[i] * src.strides[i];
      }

      final sliceView = NDArray.view(
        src,
        shape: [n, n],
        strides: src.strides.sublist(rank - 2),
        offsetElements: offsetA,
      );
      final sliceCopy = sliceView.copy();

      var offsetW = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetW += coords[i] * w.strides[i];
      }

      switch (src.dtype) {
        case DType.complex128:
          final w2D = NDArray<Complex>.create([n], DType.complex128);

          final info = LAPACKE_zgeev(
            101, // ROW_MAJOR
            jobvl,
            jobvr,
            n,
            sliceCopy.pointer.cast<ffi.Double>(),
            n,
            w2D.pointer.cast<ffi.Double>(),
            ffi.nullptr.cast<ffi.Double>(),
            1, // ldvl
            ffi.nullptr.cast<ffi.Double>(),
            1, // ldvr
          );

          if (info < 0) {
            throw ArgumentError(
              'Illegal value in call to LAPACKE_zgeev: $info',
            );
          }
          if (info > 0) {
            throw ArgumentError(
              'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
            );
          }

          final wView = NDArray<Complex>.view(
            w,
            shape: [n],
            strides: w.strides.isEmpty ? [1] : [w.strides.last],
            offsetElements: offsetW,
          );
          w2D.copy(out: wView);
          w2D.dispose();

        case DType.complex64:
          final w2D = NDArray<Complex>.create([n], DType.complex64);

          final info = LAPACKE_cgeev(
            101, // ROW_MAJOR
            jobvl,
            jobvr,
            n,
            sliceCopy.pointer.cast<ffi.Float>(),
            n,
            w2D.pointer.cast<ffi.Float>(),
            ffi.nullptr.cast<ffi.Float>(),
            1, // ldvl
            ffi.nullptr.cast<ffi.Float>(),
            1, // ldvr
          );

          if (info < 0) {
            throw ArgumentError(
              'Illegal value in call to LAPACKE_cgeev: $info',
            );
          }
          if (info > 0) {
            throw ArgumentError(
              'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
            );
          }

          final wView = NDArray<Complex>.view(
            w,
            shape: [n],
            strides: w.strides.isEmpty ? [1] : [w.strides.last],
            offsetElements: offsetW,
          );
          w2D.copy(out: wView);
          w2D.dispose();

        case DType.float64:
          final wr = NDArray<double>.zeros([n], DType.float64);
          final wi = NDArray<double>.zeros([n], DType.float64);

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
            1, // ldvl
            ffi.nullptr.cast<ffi.Double>(),
            1, // ldvr
          );

          if (info < 0) {
            throw ArgumentError(
              'Illegal value in call to LAPACKE_dgeev: $info',
            );
          }
          if (info > 0) {
            throw ArgumentError(
              'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
            );
          }

          final strideWLast = w.strides.isEmpty ? 1 : w.strides.last;
          assemble_eigenvalues_double(
            w.pointer.cast<cpx_t>() + offsetW,
            strideWLast,
            wr.pointer.cast<ffi.Double>(),
            wi.pointer.cast<ffi.Double>(),
            n,
          );

          wr.dispose();
          wi.dispose();

        case DType.float32:
          final wr = NDArray<double>.zeros([n], DType.float32);
          final wi = NDArray<double>.zeros([n], DType.float32);

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
            1, // ldvl
            ffi.nullptr.cast<ffi.Float>(),
            1, // ldvr
          );

          if (info < 0) {
            throw ArgumentError(
              'Illegal value in call to LAPACKE_sgeev: $info',
            );
          }
          if (info > 0) {
            throw ArgumentError(
              'The LAPACK QR algorithm failed to converge; only eigenvalues from 1-based index ${info + 1} to $n successfully converged.',
            );
          }

          final strideWLast = w.strides.isEmpty ? 1 : w.strides.last;
          assemble_eigenvalues_float(
            w.pointer.cast<cpx_f_t>() + offsetW,
            strideWLast,
            wr.pointer.cast<ffi.Float>(),
            wi.pointer.cast<ffi.Float>(),
            n,
          );

          wr.dispose();
          wi.dispose();
        default:
          throw UnimplementedError(
            'Type ${src.dtype} not supported for eigvals',
          );
      }
      sliceCopy.dispose();
    });

    if (out == null) {
      w.detachToParentScope();
    }
    return w;
  });
}

/// Computes the Moore-Penrose pseudo-inverse of a 2D matrix.
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
  if (out != null && out.isDisposed) {
    throw StateError('Cannot write pinv result to a disposed output array.');
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

  return NDArray.scope(() {
    final svdResult = svd(a);
    final u = svdResult.U;
    final s = svdResult.S;
    final vt = svdResult.Vh;

    final maxSingularVal = s.data[0];
    final epsilon = 2.220446049250313e-16;
    final maxDim = m > n ? m : n;
    final resolvedRcond = rcond ?? (maxDim * epsilon);
    final threshold = resolvedRcond * maxSingularVal;

    final sPlus = NDArray.zeros([n, m], a.dtype);
    for (var i = 0; i < s.data.length; i++) {
      final sVal = s.data[i];
      if (sVal > threshold) {
        sPlus.setCell([i, i], castValue(1.0 / sVal, a.dtype));
      }
    }

    final v = conjugate(vt.transpose());
    final ut = conjugate(u.transpose());

    final temp = matmul(v, sPlus);
    matmul(temp, ut, out: result);

    if (out == null) {
      result.detachToParentScope();
    }
    return result;
  });
}

/// Raise a square 2D matrix to the integer power [n].
///
/// Computes $A^n$ using binary exponentiation (square-and-multiply)
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
NDArray<T> matrix_power<T>(NDArray<T> a, int n, {NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute matrix_power() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write matrix_power result to a disposed output array.',
    );
  }
  if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
    throw ArgumentError(
      'matrix_power is only defined for 2D square matrices (was shape ${a.shape}).',
    );
  }
  if (n < 0 && a.dtype.isInteger) {
    throw ArgumentError(
      'Integer matrices cannot be raised to negative powers because matrix inversion '
      'requires floating point types. Please convert the matrix to float64 or float32 first.',
    );
  }

  final size = a.shape[0];
  final result = out ?? NDArray<T>.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  return NDArray.scope(() {
    if (n == 0) {
      final eye = NDArray.eye(size, a.dtype);
      result.fill(normalizeScalar(0, a.dtype) as T);
      for (var i = 0; i < size; i++) {
        result.setCell([i, i], eye.getCell([i, i]));
      }
      if (out == null) {
        result.detachToParentScope();
      }
      return result;
    }

    NDArray base;
    if (n < 0) {
      base = inv(a);
      n = -n;
    } else {
      base = a;
    }

    if (n == 1) {
      base.copy(out: result);
      if (out == null) {
        result.detachToParentScope();
      }
      return result;
    }

    var res = NDArray<T>.eye(size, a.dtype);
    var tempRes = NDArray<T>.zeros(a.shape, a.dtype);

    var current = base.copy() as NDArray<T>;
    var tempCurrent = NDArray<T>.zeros(a.shape, a.dtype);

    var exponent = n;
    while (exponent > 0) {
      if ((exponent & 1) == 1) {
        matmul(res, current, out: tempRes);
        final tmp = res;
        res = tempRes;
        tempRes = tmp;
      }
      if (exponent > 1) {
        matmul(current, current, out: tempCurrent);
        final tmp = current;
        current = tempCurrent;
        tempCurrent = tmp;
      }
      exponent >>= 1;
    }

    res.copy(out: result);

    if (out == null) {
      result.detachToParentScope();
    }
    return result;
  });
}

/// Computes the Cholesky decomposition of a square, positive-definite 2D matrix.
///
/// Factorizes a symmetric (or Hermitian for complex), positive-definite matrix [a] into
/// $A = L L^*$ (or $A = L L^T$ for real matrices), where $L$ is a lower triangular matrix
/// factor and $L^*$ is the conjugate transpose of $L$.
///
/// Natively offloads to LAPACK solvers (`dpotrf`, `spotrf`, `cpotrf`, `zpotrf`) depending on precision and complexity.
///
/// **Preconditions:**
/// - The input matrix [a] must not be disposed.
/// - The input matrix [a] must be 2D (shape length exactly 2).
/// - The input matrix [a] must be square (`shape[0] == shape[1]`).
/// - The input matrix [a] must have a floating-point or complex data type (`float32`, `float64`, `complex64`, or `complex128`).
/// - The input matrix [a] must be symmetric/Hermitian positive-definite.
/// - If provided, the [out] destination matrix must have the same shape and dtype as [a], and must be contiguous.
///
/// **Throws:**
/// - [StateError] if the input matrix [a] is disposed.
/// - [ArgumentError] if [a] is not square or not 2D.
/// - [ArgumentError] if [a] has an unsupported dtype (e.g. integer or boolean).
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape, dtype, or is not contiguous.
/// - [ArgumentError] if the matrix is not positive-definite, or if LAPACK returns an error code.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(n^3)$ flops for an $n \times n$ matrix.
/// - Uses LAPACK solvers.
/// - Performs zero memory allocations if a pre-allocated [out] buffer is provided and the input [a] is contiguous.
///
/// **Example:**
/// {@example /example/linalg_example.dart lang=dart}
///
/// Reference: [NumPy linalg.cholesky](https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html)
NDArray<T> cholesky<T>(NDArray<T> a, {NDArray<T>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute cholesky() on a disposed array.');
  }
  if (out != null && out.isDisposed) {
    throw StateError(
      'Cannot write cholesky result to a disposed output array.',
    );
  }
  if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
    throw ArgumentError('Matrix must be square and 2D (was ${a.shape})');
  }
  if (!a.dtype.isFloating && !a.dtype.isComplex) {
    throw ArgumentError(
      'Cholesky decomposition is only supported for float and complex dtypes (was ${a.dtype})',
    );
  }
  final n = a.shape[0];
  final targetDType = a.dtype;

  final NDArray<T> src;
  final bool wasCopied;
  if (!a.isContiguous) {
    src = a.copy();
    wasCopied = true;
  } else {
    src = a;
    wasCopied = false;
  }

  final NDArray<T> lMat;
  if (out != null) {
    lMat = out;
    if (!listEquals(lMat.shape, a.shape) || lMat.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out L buffer has incompatible shape or dtype.',
      );
    }
    if (!lMat.isContiguous) {
      throw ArgumentError('Provided out L buffer must be contiguous.');
    }
    src.copy(out: lMat);
  } else {
    lMat = src.copy();
  }

  try {
    // Char 'L' in ASCII is 76
    const uploL = 76;

    final int info;
    switch (targetDType) {
      case DType.float64:
        info = LAPACKE_dpotrf(
          101, // ROW_MAJOR
          uploL,
          n,
          lMat.pointer.cast<ffi.Double>(),
          n,
        );
      case DType.float32:
        info = LAPACKE_spotrf(
          101, // ROW_MAJOR
          uploL,
          n,
          lMat.pointer.cast<ffi.Float>(),
          n,
        );
      case DType.complex128:
        info = LAPACKE_zpotrf(
          101, // ROW_MAJOR
          uploL,
          n,
          lMat.pointer.cast<ffi.Double>(),
          n,
        );
      case DType.complex64:
        info = LAPACKE_cpotrf(
          101, // ROW_MAJOR
          uploL,
          n,
          lMat.pointer.cast<ffi.Float>(),
          n,
        );
      default:
        throw UnimplementedError(
          'Unsupported dtype for Cholesky: $targetDType',
        );
    }

    if (info < 0) {
      throw ArgumentError(
        'Illegal value in call to LAPACKE Cholesky solver: $info',
      );
    }
    if (info > 0) {
      throw ArgumentError(
        'Matrix must be positive-definite for Cholesky decomposition',
      );
    }

    v_zero_upper_triangular(
      lMat.pointer.cast<ffi.Void>(),
      n,
      encodeDType(targetDType),
    );
  } finally {
    if (wasCopied) {
      src.dispose();
    }
  }

  return lMat;
}

/// Computes the QR decomposition of a matrix or a stack of matrices $A = Q R$.
///
/// Decomposes a matrix [a] out an orthogonal matrix `Q` and an upper triangular matrix `R`
/// such that `a = Q * R`.
/// Uses LAPACK solvers (`dgeqrf` / `sgeqrf` and `dorgqr` / `sorgqr`) depending on precision.
///
/// **Preconditions:**
/// - Input matrix [a] must be at least 2-dimensional.
///
/// **Throws:**
/// - [ArgumentError] if [a] rank is less than 2.
/// - [StateError] if native FFI memory allocation or LAPACK solver initialization fails.
///
/// **Example:**
/// ```dart
/// final a = NDArray<double>.fromList([12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0], [3, 3], DType.float64);
/// final res = qr(a);
/// final q = res.Q;
/// final r = res.R;
/// ```
({NDArray<T> Q, NDArray<T> R}) qr<T>(
  NDArray<T> a, {
  ({NDArray<T> Q, NDArray<T> R})? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute qr() on a disposed array.');
  }
  final rank = a.shape.length;
  if (rank < 2) {
    throw ArgumentError('Matrix must be at least 2D (was ${a.shape})');
  }
  if (!a.dtype.isFloating && !a.dtype.isComplex) {
    throw ArgumentError(
      'QR decomposition is only supported for float and complex dtypes (was ${a.dtype})',
    );
  }
  final m = a.shape[rank - 2];
  final n = a.shape[rank - 1];
  final k = m < n ? m : n;
  final stackShape = a.shape.sublist(0, rank - 2);

  final DType<double> targetDType = a.dtype == DType.float32
      ? DType.float32 as DType<double>
      : DType.float64 as DType<double>;

  final qShape = [...stackShape, m, k];
  final rShape = [...stackShape, k, n];

  final NDArray<double> qMat;
  final NDArray<double> rMat;
  if (out != null) {
    qMat = out.Q as NDArray<double>;
    rMat = out.R as NDArray<double>;
    if (!listEquals(qMat.shape, qShape) || qMat.dtype != targetDType) {
      throw ArgumentError(
        'Provided out Q buffer has incompatible shape or dtype.',
      );
    }
    if (!qMat.isContiguous) {
      throw ArgumentError('Provided out Q buffer must be contiguous.');
    }
    if (!listEquals(rMat.shape, rShape) || rMat.dtype != targetDType) {
      throw ArgumentError(
        'Provided out R buffer has incompatible shape or dtype.',
      );
    }
    if (!rMat.isContiguous) {
      throw ArgumentError('Provided out R buffer must be contiguous.');
    }
  } else {
    qMat = NDArray<double>.zeros(qShape, targetDType);
    rMat = NDArray<double>.zeros(rShape, targetDType);
  }

  final NDArray<T> aCast =
      (a.dtype == targetDType ? a : castNDArray(a, targetDType)) as NDArray<T>;
  final bool wasCast = a.dtype != targetDType;

  final aCopy = NDArray.create([m, n], targetDType);
  final marker = ScratchArena.marker;

  try {
    final ffi.Pointer<ffi.Void> tau;
    if (targetDType == DType.float64) {
      tau = ScratchArena.allocate<ffi.Double>(
        k * ffi.sizeOf<ffi.Double>(),
      ).cast<ffi.Void>();
    } else {
      tau = ScratchArena.allocate<ffi.Float>(
        k * ffi.sizeOf<ffi.Float>(),
      ).cast<ffi.Void>();
    }

    walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
      coords,
    ) {
      var offsetA = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetA += coords[i] * aCast.strides[i];
      }

      final sliceView = NDArray.view(
        aCast,
        shape: [m, n],
        strides: aCast.strides.sublist(rank - 2),
        offsetElements: offsetA,
      );
      sliceView.copy(out: aCopy as NDArray<T>);
      sliceView.dispose();

      final r2D = targetDType == DType.float32
          ? NDArray<Float32>.zeros([k, n], DType.float32)
          : NDArray<Float64>.zeros([k, n], DType.float64);
      final q2D = targetDType == DType.float32
          ? NDArray<Float32>.zeros([m, k], DType.float32)
          : NDArray<Float64>.zeros([m, k], DType.float64);

      if (targetDType == DType.float64) {
        final info = LAPACKE_dgeqrf(
          101, // ROW_MAJOR
          m,
          n,
          aCopy.pointer.cast<ffi.Double>(),
          n,
          tau.cast<ffi.Double>(),
        );
        if (info != 0) {
          throw ArgumentError('Illegal value in call to LAPACKE_dgeqrf: $info');
        }

        final r2DData = r2D.data as Float64List;
        final aCopyData = aCopy.data as Float64List;
        for (var i = 0; i < k; i++) {
          for (var j = i; j < n; j++) {
            r2DData[i * n + j] = aCopyData[i * n + j];
          }
        }

        final q2DData = q2D.data as Float64List;
        for (var i = 0; i < m; i++) {
          for (var j = 0; j < k; j++) {
            q2DData[i * k + j] = aCopyData[i * n + j];
          }
        }

        final infoOrg = LAPACKE_dorgqr(
          101, // ROW_MAJOR
          m,
          k,
          k,
          q2D.pointer.cast<ffi.Double>(),
          k,
          tau.cast<ffi.Double>(),
        );
        if (infoOrg != 0) {
          throw ArgumentError(
            'Illegal value in call to LAPACKE_dorgqr: $infoOrg',
          );
        }
      } else {
        final info = LAPACKE_sgeqrf(
          101, // ROW_MAJOR
          m,
          n,
          aCopy.pointer.cast<ffi.Float>(),
          n,
          tau.cast<ffi.Float>(),
        );
        if (info != 0) {
          throw ArgumentError('Illegal value in call to LAPACKE_sgeqrf: $info');
        }

        final r2DData = r2D.data as Float32List;
        final aCopyData = aCopy.data as Float32List;
        for (var i = 0; i < k; i++) {
          for (var j = i; j < n; j++) {
            r2DData[i * n + j] = aCopyData[i * n + j];
          }
        }

        final q2DData = q2D.data as Float32List;
        for (var i = 0; i < m; i++) {
          for (var j = 0; j < k; j++) {
            q2DData[i * k + j] = aCopyData[i * n + j];
          }
        }

        final infoOrg = LAPACKE_sorgqr(
          101, // ROW_MAJOR
          m,
          k,
          k,
          q2D.pointer.cast<ffi.Float>(),
          k,
          tau.cast<ffi.Float>(),
        );
        if (infoOrg != 0) {
          throw ArgumentError(
            'Illegal value in call to LAPACKE_sorgqr: $infoOrg',
          );
        }
      }

      var offsetQ = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetQ += coords[i] * qMat.strides[i];
      }
      var offsetR = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetR += coords[i] * rMat.strides[i];
      }

      final qSlice = NDArray.view(
        qMat,
        shape: [m, k],
        strides: qMat.strides.sublist(rank - 2),
        offsetElements: offsetQ,
      );
      q2D.copy(out: qSlice);
      qSlice.dispose();

      final rSlice = NDArray.view(
        rMat,
        shape: [k, n],
        strides: rMat.strides.sublist(rank - 2),
        offsetElements: offsetR,
      );
      r2D.copy(out: rSlice);
      rSlice.dispose();

      q2D.dispose();
      r2D.dispose();
    });
  } finally {
    ScratchArena.reset(marker);
    aCopy.dispose();
    if (wasCast) {
      aCast.dispose();
    }
  }

  return (Q: qMat as NDArray<T>, R: rMat as NDArray<T>);
}

/// Computes the Singular Value Decomposition (SVD) of a matrix or a stack of matrices $A = U S V^h$.
///
/// Decomposes a matrix [a] out left singular vectors `U`, singular values `S`,
/// and right singular vectors Vh such that `a = U * diag(S) * Vh`.
/// Uses LAPACK solvers (`dgesdd` / `sgesdd`) depending on precision.
///
/// **Preconditions:**
/// - Input matrix [a] must be at least 2-dimensional.
///
/// **Throws:**
/// - [ArgumentError] if [a] rank is less than 2.
/// - [StateError] if native FFI memory allocation or LAPACK solver initialization fails.
///
/// **Example:**
/// ```dart
/// final a = NDArray<double>.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], DType.float64);
/// final res = svd(a);
/// final u = res.U;
/// final s = res.S;
/// final vh = res.Vh;
/// ```
({NDArray<T> U, NDArray<double> S, NDArray<T> Vh}) svd<T>(
  NDArray<T> a, {
  ({NDArray<T> U, NDArray<double> S, NDArray<T> Vh})? out,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute svd() on a disposed array.');
  }
  if (out != null) {
    if (out.U.isDisposed || out.S.isDisposed || out.Vh.isDisposed) {
      throw StateError('Cannot write SVD result to a disposed output array.');
    }
  }
  if (!a.dtype.isFloating && !a.dtype.isComplex) {
    throw ArgumentError(
      'SVD decomposition is only supported for float and complex dtypes (was ${a.dtype})',
    );
  }
  final rank = a.shape.length;
  if (rank < 2) {
    throw ArgumentError('Matrix must be at least 2D (was ${a.shape})');
  }
  final m = a.shape[rank - 2];
  final n = a.shape[rank - 1];
  final stackShape = a.shape.sublist(0, rank - 2);

  final dtypeS = a.dtype.isComplex
      ? (a.dtype == DType.complex128 ? DType.float64 : DType.float32)
      : a.dtype;

  final uShape = [...stackShape, m, m];
  final sShape = m < n ? [...stackShape, m] : [...stackShape, n];
  final vtShape = [...stackShape, n, n];

  if (out != null) {
    if (!out.U.isContiguous || !out.S.isContiguous || !out.Vh.isContiguous) {
      throw ArgumentError('Provided out buffers must be contiguous.');
    }
    if (!listEquals(out.U.shape, uShape) || out.U.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out U buffer has incompatible shape or dtype.',
      );
    }
    if (!listEquals(out.S.shape, sShape) || out.S.dtype != dtypeS) {
      throw ArgumentError(
        'Provided out S buffer has incompatible shape or dtype.',
      );
    }
    if (!listEquals(out.Vh.shape, vtShape) || out.Vh.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out Vh buffer has incompatible shape or dtype.',
      );
    }
  }

  return _svd(a, out: out);
}

({NDArray<T> U, NDArray<double> S, NDArray<T> Vh}) _svd<T>(
  NDArray<T> a, {
  ({NDArray<T> U, NDArray<double> S, NDArray<T> Vh})? out,
}) {
  final rank = a.shape.length;
  final m = a.shape[rank - 2];
  final n = a.shape[rank - 1];
  final stackShape = a.shape.sublist(0, rank - 2);

  if (m < n) {
    final axes = List<int>.generate(rank, (i) => i);
    axes[rank - 2] = rank - 1;
    axes[rank - 1] = rank - 2;

    final aT = a.transpose(axes);

    // Do NOT pass out to recursive call, let it allocate contiguous buffers.
    final resT = _svd(aT);
    final uNew = resT.U;
    final sNew = resT.S;
    final vhNew = resT.Vh;

    final uResult = vhNew.transpose(axes);
    final vhResult = uNew.transpose(axes);

    if (out != null) {
      uResult.copy(out: out.U);
      sNew.copy(out: out.S);
      vhResult.copy(out: out.Vh);

      uNew.dispose();
      sNew.dispose();
      vhNew.dispose();
      uResult.dispose();
      vhResult.dispose();

      return out;
    } else {
      return (U: uResult, S: sNew, Vh: vhResult);
    }
  }

  final dtypeS = a.dtype.isComplex
      ? (a.dtype == DType.complex128 ? DType.float64 : DType.float32)
      : a.dtype;

  final uShape = [...stackShape, m, m];
  final sShape = [...stackShape, n];
  final vtShape = [...stackShape, n, n];

  final NDArray<T> uMat = out?.U ?? NDArray<T>.zeros(uShape, a.dtype);
  final NDArray<double> sMat =
      out?.S ?? NDArray<double>.zeros(sShape, dtypeS as DType<double>);
  final NDArray<T> vtMat = out?.Vh ?? NDArray<T>.zeros(vtShape, a.dtype);

  final aCopy = NDArray<T>.create([m, n], a.dtype);
  final marker = ScratchArena.marker;

  try {
    final ffi.Pointer<ffi.Void> superb;
    final superbLen = math.max(1, n - 1);
    if (a.dtype == DType.float64 || a.dtype == DType.complex128) {
      superb = ScratchArena.allocate<ffi.Double>(
        superbLen * ffi.sizeOf<ffi.Double>(),
      ).cast<ffi.Void>();
    } else {
      superb = ScratchArena.allocate<ffi.Float>(
        superbLen * ffi.sizeOf<ffi.Float>(),
      ).cast<ffi.Void>();
    }

    walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
      coords,
    ) {
      var offsetA = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetA += coords[i] * a.strides[i];
      }

      final sliceView = NDArray.view(
        a,
        shape: [m, n],
        strides: a.strides.sublist(rank - 2),
        offsetElements: offsetA,
      );
      sliceView.copy(out: aCopy);
      sliceView.dispose();

      final NDArray<double> s2D =
          (a.dtype == DType.float32 || a.dtype == DType.complex64)
          ? NDArray<Float32>.zeros([n], DType.float32)
          : NDArray<Float64>.zeros([n], DType.float64);

      final NDArray<T> u2D = NDArray<T>.zeros([m, m], a.dtype);
      final NDArray<T> vt2D = NDArray<T>.zeros([n, n], a.dtype);

      switch (a.dtype) {
        case DType.float64:
          final info = LAPACKE_dgesvd(
            101,
            65,
            65,
            m,
            n,
            aCopy.pointer.cast<ffi.Double>(),
            n,
            s2D.pointer.cast<ffi.Double>(),
            u2D.pointer.cast<ffi.Double>(),
            m,
            vt2D.pointer.cast<ffi.Double>(),
            n,
            superb.cast<ffi.Double>(),
          );
          if (info != 0) throw ArgumentError('LAPACKE_dgesvd failed: $info');

        case DType.float32:
          final info = LAPACKE_sgesvd(
            101,
            65,
            65,
            m,
            n,
            aCopy.pointer.cast<ffi.Float>(),
            n,
            s2D.pointer.cast<ffi.Float>(),
            u2D.pointer.cast<ffi.Float>(),
            m,
            vt2D.pointer.cast<ffi.Float>(),
            n,
            superb.cast<ffi.Float>(),
          );
          if (info != 0) throw ArgumentError('LAPACKE_sgesvd failed: $info');

        case DType.complex128:
          final info = LAPACKE_zgesvd(
            101,
            65,
            65,
            m,
            n,
            aCopy.pointer.cast<ffi.Double>(),
            n,
            s2D.pointer.cast<ffi.Double>(),
            u2D.pointer.cast<ffi.Double>(),
            m,
            vt2D.pointer.cast<ffi.Double>(),
            n,
            superb.cast<ffi.Double>(),
          );
          if (info != 0) throw ArgumentError('LAPACKE_zgesvd failed: $info');

        case DType.complex64:
          final info = LAPACKE_cgesvd(
            101,
            65,
            65,
            m,
            n,
            aCopy.pointer.cast<ffi.Float>(),
            n,
            s2D.pointer.cast<ffi.Float>(),
            u2D.pointer.cast<ffi.Float>(),
            m,
            vt2D.pointer.cast<ffi.Float>(),
            n,
            superb.cast<ffi.Float>(),
          );
          if (info != 0) throw ArgumentError('LAPACKE_cgesvd failed: $info');
        default:
          throw UnimplementedError();
      }

      var offsetU = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetU += coords[i] * uMat.strides[i];
      }
      var offsetS = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetS += coords[i] * sMat.strides[i];
      }
      var offsetVt = 0;
      for (var i = 0; i < coords.length; i++) {
        offsetVt += coords[i] * vtMat.strides[i];
      }

      final uSlice = NDArray<T>.view(
        uMat,
        shape: [m, m],
        strides: uMat.strides.sublist(rank - 2),
        offsetElements: offsetU,
      );
      u2D.copy(out: uSlice);
      uSlice.dispose();

      final sSlice = NDArray<double>.view(
        sMat,
        shape: [n],
        strides: sMat.strides.isEmpty ? [1] : [sMat.strides.last],
        offsetElements: offsetS,
      );
      s2D.copy(out: sSlice);
      sSlice.dispose();

      final vtSlice = NDArray<T>.view(
        vtMat,
        shape: [n, n],
        strides: vtMat.strides.sublist(rank - 2),
        offsetElements: offsetVt,
      );
      vt2D.copy(out: vtSlice);
      vtSlice.dispose();

      s2D.dispose();
      u2D.dispose();
      vt2D.dispose();
    });
  } finally {
    ScratchArena.reset(marker);
    aCopy.dispose();
  }

  return (U: uMat, S: sMat, Vh: vtMat);
}

/// Computes the eigenvalues and eigenvectors of a complex Hermitian (conjugate symmetric) or a real symmetric matrix.
///
/// Returns a record containing:
/// - [eigenvalues]: A 1D array containing the eigenvalues in ascending order.
/// - [eigenvectors]: A 2D matrix whose columns are the normalized eigenvectors.
///
/// **Preconditions:**
/// - [a] must be a square 2D matrix, or a stack of square 2D matrices.
/// - [a] must have a floating-point or complex dtype (`Float32`, `Float64`, `Complex64`, `Complex128`).
///   Integer types are promoted to `Float64`.
/// - If provided, [outEigenvalues] and [outEigenvectors] must have compatible shapes and dtypes.
///
/// **Throws:**
/// - [ArgumentError] if [a] is not square or has rank < 2.
/// - [ArgumentError] if [a] has unsupported dtype.
/// - [ArgumentError] if [outEigenvalues] or [outEigenvectors] are incompatible.
/// - [StateError] if the LAPACK call fails.
({NDArray<num> eigenvalues, NDArray eigenvectors}) eigh<T>(
  NDArray<T> a, {
  String uplo = 'L',
  NDArray<num>? outEigenvalues,
  NDArray? outEigenvectors,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot calculate eigh on a disposed array.');
  }
  if (a.rank < 2) {
    throw ArgumentError('Array must be at least 2-dimensional.');
  }
  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m != n) {
    throw ArgumentError('Last two dimensions must be square (got $m x $n).');
  }

  final bool promoted = a.dtype.isInteger;
  DType targetDType = a.dtype;
  if (promoted) {
    targetDType = DType.float64;
  }

  if (targetDType != DType.float64 &&
      targetDType != DType.float32 &&
      targetDType != DType.complex128 &&
      targetDType != DType.complex64) {
    throw ArgumentError('Unsupported dtype: ${a.dtype}');
  }

  final DType<num> eigenvalueDType =
      (targetDType.isComplex
              ? (targetDType == DType.complex128
                    ? DType.float64
                    : DType.float32)
              : targetDType)
          as DType<num>;

  final stackShape = a.shape.sublist(0, a.rank - 2);

  final eigenvaluesShape = [...stackShape, n];
  final eigenvectorsShape = [...stackShape, n, n];

  if (outEigenvalues != null) {
    if (outEigenvalues.isDisposed) {
      throw StateError('outEigenvalues is disposed.');
    }
    if (!outEigenvalues.isContiguous) {
      throw ArgumentError('outEigenvalues must be contiguous.');
    }
    if (!listEquals(outEigenvalues.shape, eigenvaluesShape) ||
        outEigenvalues.dtype != eigenvalueDType) {
      throw ArgumentError(
        'Incompatible outEigenvalues (expected shape $eigenvaluesShape and dtype $eigenvalueDType, got shape ${outEigenvalues.shape} and dtype ${outEigenvalues.dtype}).',
      );
    }
  }

  if (outEigenvectors != null) {
    if (outEigenvectors.isDisposed) {
      throw StateError('outEigenvectors is disposed.');
    }
    if (!outEigenvectors.isContiguous) {
      throw ArgumentError('outEigenvectors must be contiguous.');
    }
    if (!listEquals(outEigenvectors.shape, eigenvectorsShape) ||
        outEigenvectors.dtype != targetDType) {
      throw ArgumentError(
        'Incompatible outEigenvectors (expected shape $eigenvectorsShape and dtype $targetDType, got shape ${outEigenvectors.shape} and dtype ${outEigenvectors.dtype}).',
      );
    }
  }

  final NDArray<num> wMat;
  if (outEigenvalues != null) {
    wMat = outEigenvalues;
  } else {
    wMat = _zerosTyped(eigenvaluesShape, eigenvalueDType) as NDArray<num>;
  }

  final NDArray vMat;
  if (outEigenvectors != null) {
    vMat = outEigenvectors;
  } else {
    vMat = _zerosTyped(eigenvectorsShape, targetDType);
  }

  final uploChar = uplo.toUpperCase();
  if (uploChar != 'L' && uploChar != 'U') {
    throw ArgumentError("uplo must be 'L' or 'U'.");
  }
  final uploVal = uploChar.codeUnitAt(0);
  final jobzVal = 86; // 'V'

  final aCopy2D = _createTyped2D(n, n, targetDType);
  final w2D = _zerosTyped([n], eigenvalueDType) as NDArray<num>;

  final marker = ScratchArena.marker;
  try {
    walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
      coords,
    ) {
      final sliceView = a.slice([
        ...coords.map((c) => Index(c)),
        Slice.all(),
        Slice.all(),
      ]);
      if (sliceView.dtype == targetDType) {
        sliceView.copy(out: aCopy2D as NDArray<T>);
      } else {
        final casted = castNDArray(sliceView, targetDType);
        casted.copy(out: aCopy2D);
        casted.dispose();
      }
      sliceView.dispose();

      int info = 0;
      switch (targetDType) {
        case DType.float64:
          info = LAPACKE_dsyevd(
            101,
            jobzVal,
            uploVal,
            n,
            aCopy2D.pointer.cast<ffi.Double>(),
            n,
            w2D.pointer.cast<ffi.Double>(),
          );
          if (info != 0) throw StateError('LAPACKE_dsyevd failed: $info');
        case DType.float32:
          info = LAPACKE_ssyevd(
            101,
            jobzVal,
            uploVal,
            n,
            aCopy2D.pointer.cast<ffi.Float>(),
            n,
            w2D.pointer.cast<ffi.Float>(),
          );
          if (info != 0) throw StateError('LAPACKE_ssyevd failed: $info');
        case DType.complex128:
          info = LAPACKE_zheevd(
            101,
            jobzVal,
            uploVal,
            n,
            aCopy2D.pointer.cast<ffi.Double>(),
            n,
            w2D.pointer.cast<ffi.Double>(),
          );
          if (info != 0) throw StateError('LAPACKE_zheevd failed: $info');
        case DType.complex64:
          info = LAPACKE_cheevd(
            101,
            jobzVal,
            uploVal,
            n,
            aCopy2D.pointer.cast<ffi.Float>(),
            n,
            w2D.pointer.cast<ffi.Float>(),
          );
          if (info != 0) throw StateError('LAPACKE_cheevd failed: $info');
        default:
          throw UnimplementedError();
      }

      final wSlice = wMat.slice([...coords.map((c) => Index(c)), Slice.all()]);
      w2D.copyToContiguous(wSlice);
      wSlice.dispose();

      final vSlice = vMat.slice([
        ...coords.map((c) => Index(c)),
        Slice.all(),
        Slice.all(),
      ]);
      aCopy2D.copyToContiguous(vSlice);
      vSlice.dispose();
    });
  } finally {
    ScratchArena.reset(marker);
    aCopy2D.dispose();
    w2D.dispose();
  }

  return (eigenvalues: wMat, eigenvectors: vMat);
}

/// Computes the eigenvalues of a complex Hermitian or real symmetric matrix.
///
/// Returns a 1D array containing the eigenvalues in ascending order.
///
/// **Preconditions:**
/// - [a] must be a square 2D matrix, or a stack of square 2D matrices.
/// - [a] must have a floating-point or complex dtype (`Float32`, `Float64`, `Complex64`, `Complex128`).
///   Integer types are promoted to `Float64`.
/// - If provided, [out] must have compatible shape and dtype.
///
/// **Throws:**
/// - [ArgumentError] if [a] is not square or has rank < 2.
/// - [ArgumentError] if [a] has unsupported dtype.
/// - [ArgumentError] if [out] is incompatible.
/// - [StateError] if the LAPACK call fails.
NDArray<num> eigvalsh<T>(NDArray<T> a, {String uplo = 'L', NDArray<num>? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot calculate eigvalsh on a disposed array.');
  }
  if (a.rank < 2) {
    throw ArgumentError('Array must be at least 2-dimensional.');
  }
  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m != n) {
    throw ArgumentError('Last two dimensions must be square (got $m x $n).');
  }

  final bool promoted = a.dtype.isInteger;
  DType targetDType = a.dtype;
  if (promoted) {
    targetDType = DType.float64;
  }

  if (targetDType != DType.float64 &&
      targetDType != DType.float32 &&
      targetDType != DType.complex128 &&
      targetDType != DType.complex64) {
    throw ArgumentError('Unsupported dtype: ${a.dtype}');
  }

  final DType<num> eigenvalueDType =
      (targetDType.isComplex
              ? (targetDType == DType.complex128
                    ? DType.float64
                    : DType.float32)
              : targetDType)
          as DType<num>;

  final stackShape = a.shape.sublist(0, a.rank - 2);
  final eigenvaluesShape = [...stackShape, n];

  if (out != null) {
    if (out.isDisposed) {
      throw StateError('out is disposed.');
    }
    if (!out.isContiguous) {
      throw ArgumentError('out must be contiguous.');
    }
    if (!listEquals(out.shape, eigenvaluesShape) ||
        out.dtype != eigenvalueDType) {
      throw ArgumentError(
        'Incompatible out (expected shape $eigenvaluesShape and dtype $eigenvalueDType, got shape ${out.shape} and dtype ${out.dtype}).',
      );
    }
  }

  final NDArray<num> wMat;
  if (out != null) {
    wMat = out;
  } else {
    wMat = _zerosTyped(eigenvaluesShape, eigenvalueDType) as NDArray<num>;
  }

  final uploChar = uplo.toUpperCase();
  if (uploChar != 'L' && uploChar != 'U') {
    throw ArgumentError("uplo must be 'L' or 'U'.");
  }
  final uploVal = uploChar.codeUnitAt(0);
  final jobzVal = 78; // 'N'

  final aCopy2D = _createTyped2D(n, n, targetDType);
  final w2D = _zerosTyped([n], eigenvalueDType) as NDArray<num>;

  final marker = ScratchArena.marker;
  try {
    walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
      coords,
    ) {
      final sliceView = a.slice([
        ...coords.map((c) => Index(c)),
        Slice.all(),
        Slice.all(),
      ]);
      if (sliceView.dtype == targetDType) {
        sliceView.copy(out: aCopy2D as NDArray<T>);
      } else {
        final casted = castNDArray(sliceView, targetDType);
        casted.copy(out: aCopy2D);
        casted.dispose();
      }
      sliceView.dispose();

      int info = 0;
      switch (targetDType) {
        case DType.float64:
          info = LAPACKE_dsyevd(
            101,
            jobzVal,
            uploVal,
            n,
            aCopy2D.pointer.cast<ffi.Double>(),
            n,
            w2D.pointer.cast<ffi.Double>(),
          );
          if (info != 0) throw StateError('LAPACKE_dsyevd failed: $info');
        case DType.float32:
          info = LAPACKE_ssyevd(
            101,
            jobzVal,
            uploVal,
            n,
            aCopy2D.pointer.cast<ffi.Float>(),
            n,
            w2D.pointer.cast<ffi.Float>(),
          );
          if (info != 0) throw StateError('LAPACKE_ssyevd failed: $info');
        case DType.complex128:
          info = LAPACKE_zheevd(
            101,
            jobzVal,
            uploVal,
            n,
            aCopy2D.pointer.cast<ffi.Double>(),
            n,
            w2D.pointer.cast<ffi.Double>(),
          );
          if (info != 0) throw StateError('LAPACKE_zheevd failed: $info');
        case DType.complex64:
          info = LAPACKE_cheevd(
            101,
            jobzVal,
            uploVal,
            n,
            aCopy2D.pointer.cast<ffi.Float>(),
            n,
            w2D.pointer.cast<ffi.Float>(),
          );
          if (info != 0) throw StateError('LAPACKE_cheevd failed: $info');
        default:
          throw UnimplementedError();
      }

      final wSlice = wMat.slice([...coords.map((c) => Index(c)), Slice.all()]);
      w2D.copyToContiguous(wSlice);
      wSlice.dispose();
    });
  } finally {
    ScratchArena.reset(marker);
    aCopy2D.dispose();
    w2D.dispose();
  }

  return wMat;
}

/// Computes the Schur decomposition of a matrix.
///
/// A = Z * T * Z^H
///
/// Returns a record containing:
/// - [T]: The Schur form. For real input and `output = 'real'`, it is quasi-upper triangular.
///   For `output = 'complex'`, it is upper triangular.
/// - [Z]: The unitary matrix of Schur vectors.
///
/// **Preconditions:**
/// - [a] must be a square 2D matrix, or a stack of square 2D matrices.
/// - [a] must have a floating-point or complex dtype. Integer types are promoted to `Float64`.
/// - [output] must be 'real' or 'complex'.
/// - If provided, [outT] and [outZ] must have compatible shapes and dtypes.
///
/// **Throws:**
/// - [ArgumentError] if [a] is not square or has rank < 2.
/// - [ArgumentError] if [output] is invalid.
/// - [ArgumentError] if [outT] or [outZ] are incompatible.
/// - [StateError] if the LAPACK call fails.
({NDArray T, NDArray Z}) schur<T>(
  NDArray<T> a, {
  String output = 'real',
  NDArray? outT,
  NDArray? outZ,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot calculate schur on a disposed array.');
  }
  if (a.rank < 2) {
    throw ArgumentError('Array must be at least 2-dimensional.');
  }
  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m != n) {
    throw ArgumentError('Last two dimensions must be square (got $m x $n).');
  }

  final outputLower = output.toLowerCase();
  if (outputLower != 'real' && outputLower != 'complex') {
    throw ArgumentError("output must be 'real' or 'complex'.");
  }

  final bool promoted = a.dtype.isInteger;
  DType targetDType = a.dtype;
  if (promoted) {
    targetDType = DType.float64;
  }

  if (targetDType != DType.float64 &&
      targetDType != DType.float32 &&
      targetDType != DType.complex128 &&
      targetDType != DType.complex64) {
    throw ArgumentError('Unsupported dtype: ${a.dtype}');
  }

  if (outputLower == 'complex' && !targetDType.isComplex) {
    if (targetDType == DType.float64) {
      targetDType = DType.complex128;
    } else {
      targetDType = DType.complex64;
    }
  }

  final stackShape = a.shape.sublist(0, a.rank - 2);
  final schurShape = [...stackShape, n, n];

  if (outT != null) {
    if (outT.isDisposed) throw StateError('outT is disposed.');
    if (!outT.isContiguous) throw ArgumentError('outT must be contiguous.');
    if (!listEquals(outT.shape, schurShape) || outT.dtype != targetDType) {
      throw ArgumentError('Incompatible outT.');
    }
  }

  if (outZ != null) {
    if (outZ.isDisposed) throw StateError('outZ is disposed.');
    if (!outZ.isContiguous) throw ArgumentError('outZ must be contiguous.');
    if (!listEquals(outZ.shape, schurShape) || outZ.dtype != targetDType) {
      throw ArgumentError('Incompatible outZ.');
    }
  }

  final NDArray tMat = outT ?? _zerosTyped(schurShape, targetDType);
  final NDArray zMat = outZ ?? _zerosTyped(schurShape, targetDType);

  final jobvsVal = 86; // 'V'
  final sortVal = 78; // 'N'

  final aCopy2D = _createTyped2D(n, n, targetDType);
  final z2D = _zerosTyped([n, n], targetDType);

  final marker = ScratchArena.marker;
  try {
    final ffi.Pointer<ffi.Void> wr;
    final ffi.Pointer<ffi.Void> wi;
    final ffi.Pointer<ffi.Void> w;

    if (targetDType == DType.float64) {
      wr = ScratchArena.allocate<ffi.Double>(
        n * ffi.sizeOf<ffi.Double>(),
      ).cast<ffi.Void>();
      wi = ScratchArena.allocate<ffi.Double>(
        n * ffi.sizeOf<ffi.Double>(),
      ).cast<ffi.Void>();
      w = ffi.nullptr.cast<ffi.Void>();
    } else if (targetDType == DType.float32) {
      wr = ScratchArena.allocate<ffi.Float>(
        n * ffi.sizeOf<ffi.Float>(),
      ).cast<ffi.Void>();
      wi = ScratchArena.allocate<ffi.Float>(
        n * ffi.sizeOf<ffi.Float>(),
      ).cast<ffi.Void>();
      w = ffi.nullptr.cast<ffi.Void>();
    } else if (targetDType == DType.complex128) {
      wr = ffi.nullptr.cast<ffi.Void>();
      wi = ffi.nullptr.cast<ffi.Void>();
      w = ScratchArena.allocate<ffi.Double>(
        2 * n * ffi.sizeOf<ffi.Double>(),
      ).cast<ffi.Void>();
    } else {
      // complex64
      wr = ffi.nullptr.cast<ffi.Void>();
      wi = ffi.nullptr.cast<ffi.Void>();
      w = ScratchArena.allocate<ffi.Float>(
        2 * n * ffi.sizeOf<ffi.Float>(),
      ).cast<ffi.Void>();
    }

    final sdimPtr = ScratchArena.allocate<lapack_int>(ffi.sizeOf<lapack_int>());

    walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
      coords,
    ) {
      final sliceView = a.slice([
        ...coords.map((c) => Index(c)),
        Slice.all(),
        Slice.all(),
      ]);

      if (sliceView.dtype == targetDType) {
        sliceView.copy(out: aCopy2D as NDArray<T>);
      } else {
        final casted = castNDArray(sliceView, targetDType);
        casted.copy(out: aCopy2D);
        casted.dispose();
      }
      sliceView.dispose();

      int info = 0;
      switch (targetDType) {
        case DType.float64:
          info = LAPACKE_dgees(
            101,
            jobvsVal,
            sortVal,
            ffi.nullptr.cast(),
            n,
            aCopy2D.pointer.cast<ffi.Double>(),
            n,
            sdimPtr,
            wr.cast<ffi.Double>(),
            wi.cast<ffi.Double>(),
            z2D.pointer.cast<ffi.Double>(),
            n,
          );
          if (info != 0) throw StateError('LAPACKE_dgees failed: $info');
        case DType.float32:
          info = LAPACKE_sgees(
            101,
            jobvsVal,
            sortVal,
            ffi.nullptr.cast(),
            n,
            aCopy2D.pointer.cast<ffi.Float>(),
            n,
            sdimPtr,
            wr.cast<ffi.Float>(),
            wi.cast<ffi.Float>(),
            z2D.pointer.cast<ffi.Float>(),
            n,
          );
          if (info != 0) throw StateError('LAPACKE_sgees failed: $info');
        case DType.complex128:
          info = LAPACKE_zgees(
            101,
            jobvsVal,
            sortVal,
            ffi.nullptr.cast(),
            n,
            aCopy2D.pointer.cast<ffi.Double>(),
            n,
            sdimPtr,
            w.cast<ffi.Double>(),
            z2D.pointer.cast<ffi.Double>(),
            n,
          );
          if (info != 0) throw StateError('LAPACKE_zgees failed: $info');
        case DType.complex64:
          info = LAPACKE_cgees(
            101,
            jobvsVal,
            sortVal,
            ffi.nullptr.cast(),
            n,
            aCopy2D.pointer.cast<ffi.Float>(),
            n,
            sdimPtr,
            w.cast<ffi.Float>(),
            z2D.pointer.cast<ffi.Float>(),
            n,
          );
          if (info != 0) throw StateError('LAPACKE_cgees failed: $info');
        default:
          throw UnimplementedError();
      }

      final tSlice = tMat.slice([
        ...coords.map((c) => Index(c)),
        Slice.all(),
        Slice.all(),
      ]);
      aCopy2D.copyToContiguous(tSlice);
      tSlice.dispose();

      final zSlice = zMat.slice([
        ...coords.map((c) => Index(c)),
        Slice.all(),
        Slice.all(),
      ]);
      z2D.copyToContiguous(zSlice);
      zSlice.dispose();
    });
  } finally {
    ScratchArena.reset(marker);
    aCopy2D.dispose();
    z2D.dispose();
  }

  return (T: tMat, Z: zMat);
}

/// Computes the Hessenberg decomposition of a matrix.
///
/// A = Q * H * Q^H
///
/// Returns a record containing:
/// - [H]: The Hessenberg matrix (zero below the first subdiagonal).
/// - [Q]: The unitary matrix.
///
/// **Preconditions:**
/// - [a] must be a square 2D matrix, or a stack of square 2D matrices.
/// - [a] must have a floating-point or complex dtype. Integer types are promoted to `Float64`.
/// - If provided, [outH] and [outQ] must have compatible shapes and dtypes.
///
/// **Throws:**
/// - [ArgumentError] if [a] is not square or has rank < 2.
/// - [ArgumentError] if [outH] or [outQ] are incompatible.
/// - [StateError] if the LAPACK call fails.
({NDArray H, NDArray Q}) hessenberg<T>(
  NDArray<T> a, {
  NDArray? outH,
  NDArray? outQ,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot calculate hessenberg on a disposed array.');
  }
  if (a.rank < 2) {
    throw ArgumentError('Array must be at least 2-dimensional.');
  }
  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m != n) {
    throw ArgumentError('Last two dimensions must be square (got $m x $n).');
  }

  final bool promoted = a.dtype.isInteger;
  DType targetDType = a.dtype;
  if (promoted) {
    targetDType = DType.float64;
  }

  if (targetDType != DType.float64 &&
      targetDType != DType.float32 &&
      targetDType != DType.complex128 &&
      targetDType != DType.complex64) {
    throw ArgumentError('Unsupported dtype: ${a.dtype}');
  }

  final stackShape = a.shape.sublist(0, a.rank - 2);
  final hessenbergShape = [...stackShape, n, n];

  if (outH != null) {
    if (outH.isDisposed) throw StateError('outH is disposed.');
    if (!outH.isContiguous) throw ArgumentError('outH must be contiguous.');
    if (!listEquals(outH.shape, hessenbergShape) || outH.dtype != targetDType) {
      throw ArgumentError('Incompatible outH.');
    }
  }

  if (outQ != null) {
    if (outQ.isDisposed) throw StateError('outQ is disposed.');
    if (!outQ.isContiguous) throw ArgumentError('outQ must be contiguous.');
    if (!listEquals(outQ.shape, hessenbergShape) || outQ.dtype != targetDType) {
      throw ArgumentError('Incompatible outQ.');
    }
  }

  final NDArray hMat = outH ?? _zerosTyped(hessenbergShape, targetDType);
  final NDArray qMat = outQ ?? _zerosTyped(hessenbergShape, targetDType);

  final aCopy2D = _createTyped2D(n, n, targetDType);
  final q2D = _zerosTyped([n, n], targetDType);

  final marker = ScratchArena.marker;
  try {
    final ffi.Pointer<ffi.Void> tau;
    final int elements = (n - 1) * (targetDType.isComplex ? 2 : 1);
    if (targetDType == DType.float64 || targetDType == DType.complex128) {
      tau = ScratchArena.allocate<ffi.Double>(
        elements * ffi.sizeOf<ffi.Double>(),
      ).cast<ffi.Void>();
    } else {
      tau = ScratchArena.allocate<ffi.Float>(
        elements * ffi.sizeOf<ffi.Float>(),
      ).cast<ffi.Void>();
    }

    walkStackCoords(stackShape, List<int>.filled(stackShape.length, 0), 0, (
      coords,
    ) {
      final sliceView = a.slice([
        ...coords.map((c) => Index(c)),
        Slice.all(),
        Slice.all(),
      ]);
      if (sliceView.dtype == targetDType) {
        sliceView.copy(out: aCopy2D as NDArray<T>);
      } else {
        final casted = castNDArray(sliceView, targetDType);
        casted.copy(out: aCopy2D);
        casted.dispose();
      }
      sliceView.dispose();

      int info = 0;
      final ilo = 1;
      final ihi = n;

      switch (targetDType) {
        case DType.float64:
          info = LAPACKE_dgehrd(
            101,
            n,
            ilo,
            ihi,
            aCopy2D.pointer.cast<ffi.Double>(),
            n,
            tau.cast<ffi.Double>(),
          );
          if (info != 0) throw StateError('LAPACKE_dgehrd failed: $info');
        case DType.float32:
          info = LAPACKE_sgehrd(
            101,
            n,
            ilo,
            ihi,
            aCopy2D.pointer.cast<ffi.Float>(),
            n,
            tau.cast<ffi.Float>(),
          );
          if (info != 0) throw StateError('LAPACKE_sgehrd failed: $info');
        case DType.complex128:
          info = LAPACKE_zgehrd(
            101,
            n,
            ilo,
            ihi,
            aCopy2D.pointer.cast<ffi.Double>(),
            n,
            tau.cast<ffi.Double>(),
          );
          if (info != 0) throw StateError('LAPACKE_zgehrd failed: $info');
        case DType.complex64:
          info = LAPACKE_cgehrd(
            101,
            n,
            ilo,
            ihi,
            aCopy2D.pointer.cast<ffi.Float>(),
            n,
            tau.cast<ffi.Float>(),
          );
          if (info != 0) throw StateError('LAPACKE_cgehrd failed: $info');
        default:
          throw UnimplementedError();
      }

      aCopy2D.copy(out: q2D);

      final hSlice = hMat.slice([
        ...coords.map((c) => Index(c)),
        Slice.all(),
        Slice.all(),
      ]);
      aCopy2D.copyToContiguous(hSlice);

      // Zero out elements below the first subdiagonal in H using direct pointer access.
      switch (targetDType) {
        case DType.float64:
          final ptr = hSlice.pointer.cast<ffi.Double>();
          for (var i = 2; i < n; i++) {
            for (var j = 0; j < i - 1; j++) {
              ptr[i * n + j] = 0.0;
            }
          }
        case DType.float32:
          final ptr = hSlice.pointer.cast<ffi.Float>();
          for (var i = 2; i < n; i++) {
            for (var j = 0; j < i - 1; j++) {
              ptr[i * n + j] = 0.0;
            }
          }
        case DType.complex128:
          final ptr = hSlice.pointer.cast<ffi.Double>();
          for (var i = 2; i < n; i++) {
            for (var j = 0; j < i - 1; j++) {
              ptr[2 * (i * n + j)] = 0.0;
              ptr[2 * (i * n + j) + 1] = 0.0;
            }
          }
        case DType.complex64:
          final ptr = hSlice.pointer.cast<ffi.Float>();
          for (var i = 2; i < n; i++) {
            for (var j = 0; j < i - 1; j++) {
              ptr[2 * (i * n + j)] = 0.0;
              ptr[2 * (i * n + j) + 1] = 0.0;
            }
          }
        default:
          break;
      }

      switch (targetDType) {
        case DType.float64:
          info = LAPACKE_dorghr(
            101,
            n,
            ilo,
            ihi,
            q2D.pointer.cast<ffi.Double>(),
            n,
            tau.cast<ffi.Double>(),
          );
          if (info != 0) throw StateError('LAPACKE_dorghr failed: $info');
        case DType.float32:
          info = LAPACKE_sorghr(
            101,
            n,
            ilo,
            ihi,
            q2D.pointer.cast<ffi.Float>(),
            n,
            tau.cast<ffi.Float>(),
          );
          if (info != 0) throw StateError('LAPACKE_sorghr failed: $info');
        case DType.complex128:
          info = LAPACKE_zunghr(
            101,
            n,
            ilo,
            ihi,
            q2D.pointer.cast<ffi.Double>(),
            n,
            tau.cast<ffi.Double>(),
          );
          if (info != 0) throw StateError('LAPACKE_zunghr failed: $info');
        case DType.complex64:
          info = LAPACKE_cunghr(
            101,
            n,
            ilo,
            ihi,
            q2D.pointer.cast<ffi.Float>(),
            n,
            tau.cast<ffi.Float>(),
          );
          if (info != 0) throw StateError('LAPACKE_cunghr failed: $info');
        default:
          throw UnimplementedError();
      }

      hSlice.dispose();

      final qSlice = qMat.slice([
        ...coords.map((c) => Index(c)),
        Slice.all(),
        Slice.all(),
      ]);
      q2D.copyToContiguous(qSlice);
      qSlice.dispose();
    });
  } finally {
    ScratchArena.reset(marker);
    aCopy2D.dispose();
    q2D.dispose();
  }

  return (H: hMat, Q: qMat);
}

/// Alias for [hessenberg] matching alternative spelling.
({NDArray H, NDArray Q}) heessenberg<T>(
  NDArray<T> a, {
  NDArray? outH,
  NDArray? outQ,
}) => hessenberg(a, outH: outH, outQ: outQ);

NDArray _createTyped2D(int rows, int cols, DType dtype) {
  switch (dtype) {
    case DType.float64:
      return NDArray<Float64>.create([rows, cols], DType.float64);
    case DType.float32:
      return NDArray<Float32>.create([rows, cols], DType.float32);
    case DType.complex128:
      return NDArray<Complex128>.create([rows, cols], DType.complex128);
    case DType.complex64:
      return NDArray<Complex64>.create([rows, cols], DType.complex64);
    default:
      throw UnimplementedError('Unsupported dtype: $dtype');
  }
}

NDArray _zerosTyped(List<int> shape, DType dtype) {
  switch (dtype) {
    case DType.float64:
      return NDArray<Float64>.zeros(shape, DType.float64);
    case DType.float32:
      return NDArray<Float32>.zeros(shape, DType.float32);
    case DType.complex128:
      return NDArray<Complex128>.zeros(shape, DType.complex128);
    case DType.complex64:
      return NDArray<Complex64>.zeros(shape, DType.complex64);
    default:
      throw UnimplementedError('Unsupported dtype: $dtype');
  }
}

/// Computes the Kronecker product of two arrays.
///
/// **Example:**
/// {@example /example/linalg_advanced_example.dart lang=dart}
///
/// Reference: [NumPy kron](https://numpy.org/doc/stable/reference/generated/numpy.kron.html)
NDArray<R> kron<Ta, Tb, R>(NDArray<Ta> a, NDArray<Tb> b, {NDArray<R>? out}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute kron() on a disposed array.');
  }

  final rankA = a.rank;
  final rankB = b.rank;
  final maxRank = math.max(rankA, rankB);

  final paddedShapeA = List<int>.filled(maxRank, 1);
  final paddedStridesA = List<int>.filled(maxRank, 0);
  for (var i = 0; i < rankA; i++) {
    paddedShapeA[maxRank - rankA + i] = a.shape[i];
    paddedStridesA[maxRank - rankA + i] = a.strides[i];
  }

  final paddedShapeB = List<int>.filled(maxRank, 1);
  final paddedStridesB = List<int>.filled(maxRank, 0);
  for (var i = 0; i < rankB; i++) {
    paddedShapeB[maxRank - rankB + i] = b.shape[i];
    paddedStridesB[maxRank - rankB + i] = b.strides[i];
  }

  final expectedShape = List<int>.filled(maxRank, 0);
  for (var i = 0; i < maxRank; i++) {
    expectedShape[i] = paddedShapeA[i] * paddedShapeB[i];
  }

  final targetDType = resolveDType(a.dtype, b.dtype);
  if (out != null) {
    if (!listEquals(out.shape, expectedShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out recycler has incompatible shape or dtype (expected shape $expectedShape and dtype $targetDType).',
      );
    }
  }

  final result =
      out ?? NDArray<R>.create(expectedShape, targetDType as DType<R>);

  final aCast = castNDArray(a, targetDType);
  final bCast = castNDArray(b, targetDType);

  final marker = ScratchArena.marker;
  final cStridesA = ScratchArena.allocate<ffi.Int>(
    maxRank * ffi.sizeOf<ffi.Int>(),
  );
  final cShapeA = ScratchArena.allocate<ffi.Int>(
    maxRank * ffi.sizeOf<ffi.Int>(),
  );
  final cStridesB = ScratchArena.allocate<ffi.Int>(
    maxRank * ffi.sizeOf<ffi.Int>(),
  );
  final cShapeB = ScratchArena.allocate<ffi.Int>(
    maxRank * ffi.sizeOf<ffi.Int>(),
  );
  final cStridesRes = ScratchArena.allocate<ffi.Int>(
    maxRank * ffi.sizeOf<ffi.Int>(),
  );
  final cShapeRes = ScratchArena.allocate<ffi.Int>(
    maxRank * ffi.sizeOf<ffi.Int>(),
  );

  for (var i = 0; i < maxRank; i++) {
    cStridesA[i] = paddedStridesA[i];
    cShapeA[i] = paddedShapeA[i];
    cStridesB[i] = paddedStridesB[i];
    cShapeB[i] = paddedShapeB[i];
    cStridesRes[i] = result.strides[i];
    cShapeRes[i] = result.shape[i];
  }

  try {
    switch (targetDType) {
      case DType.float64:
        s_kron_double(
          aCast.pointer.cast(),
          cStridesA,
          cShapeA,
          bCast.pointer.cast(),
          cStridesB,
          cShapeB,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          maxRank,
        );
      case DType.float32:
        s_kron_float(
          aCast.pointer.cast(),
          cStridesA,
          cShapeA,
          bCast.pointer.cast(),
          cStridesB,
          cShapeB,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          maxRank,
        );
      case DType.int64:
        s_kron_int64(
          aCast.pointer.cast(),
          cStridesA,
          cShapeA,
          bCast.pointer.cast(),
          cStridesB,
          cShapeB,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          maxRank,
        );
      case DType.int32:
        s_kron_int32(
          aCast.pointer.cast(),
          cStridesA,
          cShapeA,
          bCast.pointer.cast(),
          cStridesB,
          cShapeB,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          maxRank,
        );
      case DType.uint8:
        s_kron_uint8(
          aCast.pointer.cast(),
          cStridesA,
          cShapeA,
          bCast.pointer.cast(),
          cStridesB,
          cShapeB,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          maxRank,
        );
      case DType.int16:
        s_kron_int16(
          aCast.pointer.cast(),
          cStridesA,
          cShapeA,
          bCast.pointer.cast(),
          cStridesB,
          cShapeB,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          maxRank,
        );
      case DType.complex128:
        s_kron_complex128(
          aCast.pointer.cast(),
          cStridesA,
          cShapeA,
          bCast.pointer.cast(),
          cStridesB,
          cShapeB,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          maxRank,
        );
      case DType.complex64:
        s_kron_complex64(
          aCast.pointer.cast(),
          cStridesA,
          cShapeA,
          bCast.pointer.cast(),
          cStridesB,
          cShapeB,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          maxRank,
        );
      case DType.boolean:
        s_kron_boolean(
          aCast.pointer.cast(),
          cStridesA,
          cShapeA,
          bCast.pointer.cast(),
          cStridesB,
          cShapeB,
          result.pointer.cast(),
          cStridesRes,
          cShapeRes,
          maxRank,
        );
    }
  } finally {
    ScratchArena.reset(marker);
    if (aCast != a) aCast.dispose();
    if (bCast != b) bCast.dispose();
  }

  if (out == null) {
    result.detachToParentScope();
  }
  return result;
}

/// Computes the outer product of two vectors.
///
/// Given two input vectors [a] and [b], computes the outer product matrix:
/// `res[i, j] = a[i] * b[j]`.
/// If the input arrays are not 1-dimensional, they are flattened first.
///
/// **Preconditions:**
/// - Both arrays [a] and [b] must not be disposed.
/// - If provided, the [out] recycler must have shape `[size(a), size(b)]` and the correct dtype.
///
/// **Throws:**
/// - [StateError] if [a] or [b] is disposed.
/// - [ArgumentError] if [out] has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N_a \times N_b)$ using highly optimized native strided loops.
///
/// **Example:**
/// {@example /example/linalg_advanced_example.dart lang=dart}
///
/// Reference: [NumPy outer](https://numpy.org/doc/stable/reference/generated/numpy.outer.html)
NDArray<R> outer<Ta, Tb, R>(NDArray<Ta> a, NDArray<Tb> b, {NDArray<R>? out}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute outer() on a disposed array.');
  }

  final sizeA = a.size;
  final sizeB = b.size;
  final expectedShape = [sizeA, sizeB];
  final targetDType = resolveDType(a.dtype, b.dtype);

  if (out != null) {
    if (!listEquals(out.shape, expectedShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out recycler has incompatible shape or dtype (expected shape $expectedShape and dtype $targetDType).',
      );
    }
  }

  final result =
      out ?? NDArray<R>.create(expectedShape, targetDType as DType<R>);

  final flatA = a.rank == 1 ? a : a.ravel();
  final flatB = b.rank == 1 ? b : b.ravel();

  final aCast = castNDArray(flatA, targetDType);
  final bCast = castNDArray(flatB, targetDType);

  try {
    switch (targetDType) {
      case DType.float64:
        s_outer_double(
          aCast.pointer.cast(),
          aCast.strides.isEmpty ? 1 : aCast.strides[0],
          sizeA,
          bCast.pointer.cast(),
          bCast.strides.isEmpty ? 1 : bCast.strides[0],
          sizeB,
          result.pointer.cast(),
          result.strides[0],
          result.strides[1],
        );
      case DType.float32:
        s_outer_float(
          aCast.pointer.cast(),
          aCast.strides.isEmpty ? 1 : aCast.strides[0],
          sizeA,
          bCast.pointer.cast(),
          bCast.strides.isEmpty ? 1 : bCast.strides[0],
          sizeB,
          result.pointer.cast(),
          result.strides[0],
          result.strides[1],
        );
      case DType.int64:
        s_outer_int64(
          aCast.pointer.cast(),
          aCast.strides.isEmpty ? 1 : aCast.strides[0],
          sizeA,
          bCast.pointer.cast(),
          bCast.strides.isEmpty ? 1 : bCast.strides[0],
          sizeB,
          result.pointer.cast(),
          result.strides[0],
          result.strides[1],
        );
      case DType.int32:
        s_outer_int32(
          aCast.pointer.cast(),
          aCast.strides.isEmpty ? 1 : aCast.strides[0],
          sizeA,
          bCast.pointer.cast(),
          bCast.strides.isEmpty ? 1 : bCast.strides[0],
          sizeB,
          result.pointer.cast(),
          result.strides[0],
          result.strides[1],
        );
      case DType.uint8:
        s_outer_uint8(
          aCast.pointer.cast(),
          aCast.strides.isEmpty ? 1 : aCast.strides[0],
          sizeA,
          bCast.pointer.cast(),
          bCast.strides.isEmpty ? 1 : bCast.strides[0],
          sizeB,
          result.pointer.cast(),
          result.strides[0],
          result.strides[1],
        );
      case DType.int16:
        s_outer_int16(
          aCast.pointer.cast(),
          aCast.strides.isEmpty ? 1 : aCast.strides[0],
          sizeA,
          bCast.pointer.cast(),
          bCast.strides.isEmpty ? 1 : bCast.strides[0],
          sizeB,
          result.pointer.cast(),
          result.strides[0],
          result.strides[1],
        );
      case DType.complex128:
        s_outer_complex128(
          aCast.pointer.cast(),
          aCast.strides.isEmpty ? 1 : aCast.strides[0],
          sizeA,
          bCast.pointer.cast(),
          bCast.strides.isEmpty ? 1 : bCast.strides[0],
          sizeB,
          result.pointer.cast(),
          result.strides[0],
          result.strides[1],
        );
      case DType.complex64:
        s_outer_complex64(
          aCast.pointer.cast(),
          aCast.strides.isEmpty ? 1 : aCast.strides[0],
          sizeA,
          bCast.pointer.cast(),
          bCast.strides.isEmpty ? 1 : bCast.strides[0],
          sizeB,
          result.pointer.cast(),
          result.strides[0],
          result.strides[1],
        );
      case DType.boolean:
        s_outer_boolean(
          aCast.pointer.cast(),
          aCast.strides.isEmpty ? 1 : aCast.strides[0],
          sizeA,
          bCast.pointer.cast(),
          bCast.strides.isEmpty ? 1 : bCast.strides[0],
          sizeB,
          result.pointer.cast(),
          result.strides[0],
          result.strides[1],
        );
    }
  } finally {
    if (flatA != a) flatA.dispose();
    if (flatB != b) flatB.dispose();
    if (aCast != flatA) aCast.dispose();
    if (bCast != flatB) bCast.dispose();
  }

  if (out == null) {
    result.detachToParentScope();
  }
  return result;
}

/// Computes the cross product of two (arrays of) vectors.
///
/// The cross product of two vectors is defined in 3D (and 2D, where it returns the z-component as a scalar).
/// If the inputs are multidimensional, the cross product is computed along the specified axes.
///
/// **Preconditions:**
/// - Both arrays [a] and [b] must not be disposed.
/// - The size of the cross product axes must be 2 or 3.
/// - If provided, the recycler [out] must have the correct shape and dtype.
///
/// **Throws:**
/// - [StateError] if [a] or [b] is disposed.
/// - [ArgumentError] if axes sizes are not 2 or 3, or are mismatched.
/// - [ArgumentError] if [out] has incompatible shape or dtype.
///
/// **Performance considerations:**
/// - Uses native C vector cross loops.
///
/// **Example:**
/// {@example /example/linalg_advanced_example.dart lang=dart}
///
/// Reference: [NumPy cross](https://numpy.org/doc/stable/reference/generated/numpy.cross.html)
NDArray<R> cross<Ta, Tb, R>(
  NDArray<Ta> a,
  NDArray<Tb> b, {
  int? axisa,
  int? axisb,
  int? axisc,
  int? axis,
  NDArray<R>? out,
}) {
  if (a.isDisposed || b.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute cross() on a disposed array.');
  }

  var axisA = axis ?? axisa ?? -1;
  var axisB = axis ?? axisb ?? -1;
  var axisC = axis ?? axisc ?? -1;

  if (axisA < 0) axisA = a.rank + axisA;
  if (axisB < 0) axisB = b.rank + axisB;

  if (axisA < 0 || axisA >= a.rank) {
    throw ArgumentError('axisa $axisA out of bounds for shape ${a.shape}');
  }
  if (axisB < 0 || axisB >= b.rank) {
    throw ArgumentError('axisb $axisB out of bounds for shape ${b.shape}');
  }

  final lenA = a.shape[axisA];
  final lenB = b.shape[axisB];

  if ((lenA != 2 && lenA != 3) || (lenB != 2 && lenB != 3)) {
    throw ArgumentError(
      'Cross product axes sizes must be 2 or 3 (got axisa size $lenA and axisb size $lenB).',
    );
  }
  if (lenA != lenB) {
    throw ArgumentError(
      'Mismatched cross product axes sizes: axisa size $lenA != axisb size $lenB.',
    );
  }

  final is3D = lenA == 3;

  final stackA = List<int>.from(a.shape)..removeAt(axisA);
  final stackB = List<int>.from(b.shape)..removeAt(axisB);
  final broadcastStack = broadcastStackShapes(stackA, stackB);

  final expectedShape = List<int>.from(broadcastStack);
  if (is3D) {
    var finalAxisC = axisC;
    if (finalAxisC < 0) finalAxisC = expectedShape.length + 1 + finalAxisC;
    if (finalAxisC < 0 || finalAxisC > expectedShape.length) {
      finalAxisC = expectedShape.length;
    }
    expectedShape.insert(finalAxisC, 3);
    axisC = finalAxisC;
  }

  final targetDType = resolveDType(a.dtype, b.dtype);
  if (out != null) {
    if (!listEquals(out.shape, expectedShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out recycler has incompatible shape or dtype (expected shape $expectedShape and dtype $targetDType).',
      );
    }
  }

  final result =
      out ?? NDArray<R>.create(expectedShape, targetDType as DType<R>);

  final aCast = castNDArray(a, targetDType);
  final bCast = castNDArray(b, targetDType);

  final lenResult = broadcastStack.length;
  final walkStridesA = List<int>.filled(lenResult, 0);
  final walkStridesB = List<int>.filled(lenResult, 0);
  final walkStridesRes = List<int>.filled(lenResult, 0);

  for (var i = 0; i < lenResult; i++) {
    final resAxis = lenResult - 1 - i;
    final axisIdxA = stackA.length - 1 - i;
    final axisIdxB = stackB.length - 1 - i;

    var resAxisIdx = resAxis;
    if (is3D && resAxis >= axisC) {
      resAxisIdx = resAxis + 1;
    }

    if (axisIdxA >= 0) {
      final origAxisA = axisIdxA < axisA ? axisIdxA : axisIdxA + 1;
      walkStridesA[resAxis] = (stackA[axisIdxA] == broadcastStack[resAxis])
          ? aCast.strides[origAxisA]
          : 0;
    }
    if (axisIdxB >= 0) {
      final origAxisB = axisIdxB < axisB ? axisIdxB : axisIdxB + 1;
      walkStridesB[resAxis] = (stackB[axisIdxB] == broadcastStack[resAxis])
          ? bCast.strides[origAxisB]
          : 0;
    }
    walkStridesRes[resAxis] = result.strides[resAxisIdx];
  }

  final strideVecA = aCast.strides[axisA];
  final strideVecB = bCast.strides[axisB];
  final strideVecRes = is3D ? result.strides[axisC] : 0;

  void walk(int dim, int offsetA, int offsetB, int offsetRes) {
    if (dim == lenResult) {
      switch (targetDType) {
        case DType.float64:
          if (is3D) {
            s_cross_3d_double(
              aCast.pointer.cast<ffi.Double>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Double>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Double>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_double(
              aCast.pointer.cast<ffi.Double>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Double>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Double>() + offsetRes,
            );
          }
        case DType.float32:
          if (is3D) {
            s_cross_3d_float(
              aCast.pointer.cast<ffi.Float>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Float>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Float>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_float(
              aCast.pointer.cast<ffi.Float>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Float>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Float>() + offsetRes,
            );
          }
        case DType.int64:
          if (is3D) {
            s_cross_3d_int64(
              aCast.pointer.cast<ffi.Int64>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Int64>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Int64>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_int64(
              aCast.pointer.cast<ffi.Int64>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Int64>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Int64>() + offsetRes,
            );
          }
        case DType.int32:
          if (is3D) {
            s_cross_3d_int32(
              aCast.pointer.cast<ffi.Int32>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Int32>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Int32>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_int32(
              aCast.pointer.cast<ffi.Int32>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Int32>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Int32>() + offsetRes,
            );
          }
        case DType.uint8:
          if (is3D) {
            s_cross_3d_uint8(
              aCast.pointer.cast<ffi.Uint8>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Uint8>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Uint8>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_uint8(
              aCast.pointer.cast<ffi.Uint8>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Uint8>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Uint8>() + offsetRes,
            );
          }
        case DType.int16:
          if (is3D) {
            s_cross_3d_int16(
              aCast.pointer.cast<ffi.Int16>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Int16>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Int16>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_int16(
              aCast.pointer.cast<ffi.Int16>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Int16>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Int16>() + offsetRes,
            );
          }
        case DType.complex128:
          if (is3D) {
            s_cross_3d_complex128(
              aCast.pointer.cast<cpx_t>() + offsetA,
              strideVecA,
              bCast.pointer.cast<cpx_t>() + offsetB,
              strideVecB,
              result.pointer.cast<cpx_t>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_complex128(
              aCast.pointer.cast<cpx_t>() + offsetA,
              strideVecA,
              bCast.pointer.cast<cpx_t>() + offsetB,
              strideVecB,
              result.pointer.cast<cpx_t>() + offsetRes,
            );
          }
        case DType.complex64:
          if (is3D) {
            s_cross_3d_complex64(
              aCast.pointer.cast<cpx_f_t>() + offsetA,
              strideVecA,
              bCast.pointer.cast<cpx_f_t>() + offsetB,
              strideVecB,
              result.pointer.cast<cpx_f_t>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_complex64(
              aCast.pointer.cast<cpx_f_t>() + offsetA,
              strideVecA,
              bCast.pointer.cast<cpx_f_t>() + offsetB,
              strideVecB,
              result.pointer.cast<cpx_f_t>() + offsetRes,
            );
          }
        case DType.boolean:
          if (is3D) {
            s_cross_3d_boolean(
              aCast.pointer.cast<ffi.Uint8>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Uint8>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Uint8>() + offsetRes,
              strideVecRes,
            );
          } else {
            s_cross_2d_boolean(
              aCast.pointer.cast<ffi.Uint8>() + offsetA,
              strideVecA,
              bCast.pointer.cast<ffi.Uint8>() + offsetB,
              strideVecB,
              result.pointer.cast<ffi.Uint8>() + offsetRes,
            );
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

  if (aCast != a) aCast.dispose();
  if (bCast != b) bCast.dispose();

  if (out == null) {
    result.detachToParentScope();
  }
  return result;
}

/// Computes a vector or matrix norm.
///
/// Computes one of the standard vector or matrix norms (magnitude) along the specified axis/axes.
/// The result is always a real-valued floating-point array.
///
/// **Preconditions:**
/// - Input [a] must not be disposed.
/// - If provided, the recycler [out] must have the correct shape and dtype.
/// - If provided, [axis] must be within bounds.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
/// - [ArgumentError] if [axis] or [ord] combinations are invalid.
///
/// **Performance considerations:**
/// - Uses native vector reductions for Chebyshev, L1, and L2 vector calculations.
///
/// **Example:**
/// {@example /example/linalg_advanced_example.dart lang=dart}
///
/// Reference: [NumPy linalg.norm](https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html)
NDArray<double> norm<T>(
  NDArray<T> a, {
  dynamic ord,
  dynamic axis,
  bool keepdims = false,
  NDArray<double>? out,
}) {
  if (a.isDisposed || (out != null && out.isDisposed)) {
    throw StateError('Cannot execute norm() on a disposed array.');
  }

  final rank = a.shape.length;
  List<int> targetAxes;
  if (axis == null) {
    if (rank > 2) {
      throw ArgumentError(
        'Improper axis specification: If axis is null, input must be 1D or 2D.',
      );
    }
    targetAxes = List<int>.generate(rank, (i) => i);
  } else if (axis is int) {
    var normAx = axis;
    if (normAx < 0) normAx = rank + normAx;
    if (normAx < 0 || normAx >= rank) {
      throw ArgumentError('axis $axis is out of bounds.');
    }
    targetAxes = [normAx];
  } else if (axis is List<int>) {
    if (axis.length != 2) {
      throw ArgumentError('axis list must contain exactly 1 or 2 elements.');
    }
    final normAxes = List<int>.from(axis);
    for (var i = 0; i < 2; i++) {
      if (normAxes[i] < 0) normAxes[i] = rank + normAxes[i];
      if (normAxes[i] < 0 || normAxes[i] >= rank) {
        throw ArgumentError('axis ${axis[i]} is out of bounds.');
      }
    }
    if (normAxes[0] == normAxes[1]) {
      throw ArgumentError('axes must be distinct.');
    }
    targetAxes = normAxes;
  } else {
    throw ArgumentError('axis must be null, int, or List<int>.');
  }

  final isVecNorm = targetAxes.length == 1;
  final DType targetDType =
      (a.dtype == DType.float32 || a.dtype == DType.complex64)
      ? DType.float32
      : DType.float64;

  final List<int> expectedShape;
  if (keepdims) {
    expectedShape = List<int>.from(a.shape);
    for (final ax in targetAxes) {
      expectedShape[ax] = 1;
    }
  } else {
    expectedShape = List<int>.from(a.shape);
    final sortedAxes = List<int>.from(targetAxes)
      ..sort((x, y) => y.compareTo(x));
    for (final ax in sortedAxes) {
      expectedShape.removeAt(ax);
    }
  }

  if (out != null) {
    if (!listEquals(out.shape, expectedShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype.',
      );
    }
  }

  final result =
      out ??
      NDArray<double>.create(expectedShape, targetDType as DType<double>);

  if (targetAxes.length == rank && !keepdims) {
    // Global norm
    if (isVecNorm) {
      final val = _vectorNorm<T>(a, ord, targetDType);
      result.data[0] = val;
    } else {
      final val = _matrixNorm<T>(a, ord, targetDType);
      result.data[0] = val;
    }
    if (out == null) {
      result.detachToParentScope();
    }
    return result;
  }

  // Reduction along specific axes
  final List<int> currentCoords = List<int>.filled(a.shape.length, 0);

  final List<int> stackShape = List<int>.from(a.shape);
  final sortedAxes = List<int>.from(targetAxes)..sort((x, y) => y.compareTo(x));
  for (final ax in sortedAxes) {
    stackShape.removeAt(ax);
  }

  void walkStack(int dim, List<int> coords) {
    if (dim == stackShape.length) {
      // Reconstruct original coordinates for slicing
      var stackIdx = 0;
      for (var i = 0; i < a.shape.length; i++) {
        if (!targetAxes.contains(i)) {
          currentCoords[i] = coords[stackIdx++];
        }
      }

      final NDArray<T> slice;
      if (isVecNorm) {
        final ax = targetAxes[0];
        final len = a.shape[ax];
        var offset = a.offsetElements;
        for (var i = 0; i < a.shape.length; i++) {
          if (i != ax) {
            offset += currentCoords[i] * a.strides[i];
          }
        }
        slice = NDArray.view(
          a,
          shape: [len],
          strides: [a.strides[ax]],
          offsetElements: offset,
        );
      } else {
        final ax0 = targetAxes[0];
        final ax1 = targetAxes[1];
        final len0 = a.shape[ax0];
        final len1 = a.shape[ax1];
        var offset = a.offsetElements;
        for (var i = 0; i < a.shape.length; i++) {
          if (i != ax0 && i != ax1) {
            offset += currentCoords[i] * a.strides[i];
          }
        }
        slice = NDArray.view(
          a,
          shape: [len0, len1],
          strides: [a.strides[ax0], a.strides[ax1]],
          offsetElements: offset,
        );
      }

      final double val;
      if (isVecNorm) {
        val = _vectorNorm<T>(slice, ord, targetDType);
      } else {
        val = _matrixNorm<T>(slice, ord, targetDType);
      }
      slice.dispose();

      // Calculate dest flat index
      var destOffset = result.offsetElements;
      if (keepdims) {
        for (var i = 0; i < result.shape.length; i++) {
          if (!targetAxes.contains(i)) {
            destOffset += currentCoords[i] * result.strides[i];
          }
        }
      } else {
        for (var i = 0; i < result.shape.length; i++) {
          destOffset += coords[i] * result.strides[i];
        }
      }
      result.data[destOffset] = val;
      return;
    }

    final limit = stackShape[dim];
    for (var i = 0; i < limit; i++) {
      coords[dim] = i;
      walkStack(dim + 1, coords);
    }
  }

  walkStack(0, List<int>.filled(stackShape.length, 0));

  if (out == null) {
    result.detachToParentScope();
  }
  return result;
}

double _vectorNorm<T>(NDArray<T> a, dynamic ord, DType targetDType) {
  final needsCast = a.dtype != targetDType;
  final castedA = needsCast ? castNDArray(a, targetDType) : a;

  final size = castedA.size;
  final stride = castedA.strides.isEmpty ? 1 : castedA.strides[0];

  try {
    if (ord == null || ord == 2) {
      double sum;
      if (targetDType == DType.float32) {
        if (castedA.dtype.isComplex) {
          sum = r_norm_l2_complex64(castedA.pointer.cast(), stride, size);
        } else {
          sum = r_norm_l2_float(castedA.pointer.cast(), stride, size);
        }
      } else {
        if (castedA.dtype.isComplex) {
          sum = r_norm_l2_complex128(castedA.pointer.cast(), stride, size);
        } else {
          sum = r_norm_l2_double(castedA.pointer.cast(), stride, size);
        }
      }
      return math.sqrt(sum);
    } else if (ord == 1) {
      if (targetDType == DType.float32) {
        if (castedA.dtype.isComplex) {
          return r_norm_l1_complex64(castedA.pointer.cast(), stride, size);
        } else {
          return r_norm_l1_float(castedA.pointer.cast(), stride, size);
        }
      } else {
        if (castedA.dtype.isComplex) {
          return r_norm_l1_complex128(castedA.pointer.cast(), stride, size);
        } else {
          return r_norm_l1_double(castedA.pointer.cast(), stride, size);
        }
      }
    } else if (ord == double.infinity) {
      if (targetDType == DType.float32) {
        if (castedA.dtype.isComplex) {
          return r_norm_inf_complex64(castedA.pointer.cast(), stride, size);
        } else {
          return r_norm_inf_float(castedA.pointer.cast(), stride, size);
        }
      } else {
        if (castedA.dtype.isComplex) {
          return r_norm_inf_complex128(castedA.pointer.cast(), stride, size);
        } else {
          return r_norm_inf_double(castedA.pointer.cast(), stride, size);
        }
      }
    } else if (ord == double.negativeInfinity) {
      if (targetDType == DType.float32) {
        if (castedA.dtype.isComplex) {
          return r_norm_neg_inf_complex64(castedA.pointer.cast(), stride, size);
        } else {
          return r_norm_neg_inf_float(castedA.pointer.cast(), stride, size);
        }
      } else {
        if (castedA.dtype.isComplex) {
          return r_norm_neg_inf_complex128(
            castedA.pointer.cast(),
            stride,
            size,
          );
        } else {
          return r_norm_neg_inf_double(castedA.pointer.cast(), stride, size);
        }
      }
    } else if (ord is num) {
      double sum;
      final p = ord.toDouble();
      if (targetDType == DType.float32) {
        if (castedA.dtype.isComplex) {
          sum = r_norm_lp_complex64(castedA.pointer.cast(), stride, size, p);
        } else {
          sum = r_norm_lp_float(castedA.pointer.cast(), stride, size, p);
        }
      } else {
        if (castedA.dtype.isComplex) {
          sum = r_norm_lp_complex128(castedA.pointer.cast(), stride, size, p);
        } else {
          sum = r_norm_lp_double(castedA.pointer.cast(), stride, size, p);
        }
      }
      return math.pow(sum, 1.0 / p).toDouble();
    } else {
      throw ArgumentError('Invalid vector norm order: $ord');
    }
  } finally {
    if (needsCast) castedA.dispose();
  }
}

double _matrixNorm<T>(NDArray<T> a, dynamic ord, DType targetDType) {
  final rows = a.shape[0];
  final cols = a.shape[1];

  if (ord == null || ord == 'fro') {
    final flat = a.ravel();
    final res = _vectorNorm(flat, 2, targetDType);
    flat.dispose();
    return res;
  } else if (ord == 1) {
    var maxColSum = 0.0;
    for (var c = 0; c < cols; c++) {
      final colSlice = NDArray.view(
        a,
        shape: [rows],
        strides: [a.strides[0]],
        offsetElements: c * a.strides[1],
      );
      final sum = _vectorNorm(colSlice, 1, targetDType);
      if (sum > maxColSum) maxColSum = sum;
      colSlice.dispose();
    }
    return maxColSum;
  } else if (ord == -1) {
    double? minColSum;
    for (var c = 0; c < cols; c++) {
      final colSlice = NDArray.view(
        a,
        shape: [rows],
        strides: [a.strides[0]],
        offsetElements: c * a.strides[1],
      );
      final sum = _vectorNorm(colSlice, 1, targetDType);
      if (minColSum == null || sum < minColSum) minColSum = sum;
      colSlice.dispose();
    }
    return minColSum ?? 0.0;
  } else if (ord == double.infinity) {
    var maxRowSum = 0.0;
    for (var r = 0; r < rows; r++) {
      final rowSlice = NDArray.view(
        a,
        shape: [cols],
        strides: [a.strides[1]],
        offsetElements: r * a.strides[0],
      );
      final sum = _vectorNorm(rowSlice, 1, targetDType);
      if (sum > maxRowSum) maxRowSum = sum;
      rowSlice.dispose();
    }
    return maxRowSum;
  } else if (ord == double.negativeInfinity) {
    double? minRowSum;
    for (var r = 0; r < rows; r++) {
      final rowSlice = NDArray.view(
        a,
        shape: [cols],
        strides: [a.strides[1]],
        offsetElements: r * a.strides[0],
      );
      final sum = _vectorNorm(rowSlice, 1, targetDType);
      if (minRowSum == null || sum < minRowSum) minRowSum = sum;
      rowSlice.dispose();
    }
    return minRowSum ?? 0.0;
  } else if (ord == 2 || ord == -2) {
    final castedA =
        (a.dtype == DType.float64 ||
            a.dtype == DType.float32 ||
            a.dtype.isComplex)
        ? a
        : castNDArray(a, DType.float64);
    try {
      switch (castedA.dtype) {
        case DType.complex128:
        case DType.complex64:
          final svdRes = svd(castedA as NDArray<Complex>);
          final s = svdRes.S;
          final val = ord == 2 ? s.data[0] : s.data[s.data.length - 1];
          svdRes.dispose();
          return val;
        case DType.float64:
        case DType.float32:
          final svdRes = svd(castedA as NDArray<double>);
          final s = svdRes.S;
          final val = ord == 2 ? s.data[0] : s.data[s.data.length - 1];
          svdRes.dispose();
          return (val as num).toDouble();
        default:
          throw UnimplementedError();
      }
    } finally {
      if (castedA != a) castedA.dispose();
    }
  } else {
    throw ArgumentError('Invalid matrix norm order: $ord');
  }
}

extension QRRecordDispose on ({NDArray Q, NDArray R}) {
  void dispose() {
    Q.dispose();
    R.dispose();
  }
}

extension SVDRecordDispose on ({NDArray U, NDArray S, NDArray Vh}) {
  void dispose() {
    U.dispose();
    S.dispose();
    Vh.dispose();
  }
}

/// Result of a least-squares solver [lstsq].
///
/// Reference: [NumPy linalg.lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
final class LstsqResult<T> {
  /// Least-squares solution.
  ///
  /// If the input [b] is 1-dimensional, [x] has shape `[N]`.
  /// If [b] is 2-dimensional, [x] has shape `[N, K]`.
  final NDArray<T> x;

  /// Sums of squared residuals.
  ///
  /// Squared Euclidean 2-norm for each column in $b - a x$.
  /// If the input [b] is 1-dimensional, [residuals] has shape `[1]`.
  /// If [b] is 2-dimensional, [residuals] has shape `[K]`.
  ///
  /// **Note:** Residuals are only computed if the first dimension of the input matrix $a$
  /// is strictly greater than its second dimension ($M > N$) and the effective rank is $N$.
  /// Otherwise, it is returned as an empty array of shape `[0]`.
  final NDArray<double> residuals;

  /// Effective rank of the input matrix $a$.
  final int rank;

  /// Singular values of the input matrix $a$.
  ///
  /// Stored in descending order of magnitude.
  /// Shape is `[min(M, N)]`.
  final NDArray<double> s;

  /// Creates a new [LstsqResult] instance.
  LstsqResult({
    required this.x,
    required this.residuals,
    required this.rank,
    required this.s,
  });
}

/// Computes the least-squares solution to a linear matrix equation $a x = b$.
///
/// Solves the equation $a x = b$ by computing a vector/matrix $x$ that minimizes the
/// Euclidean 2-norm $\|b - a x\|_2^2$.
///
/// Natively offloads to LAPACK divide-and-conquer SVD-based least-squares solvers
/// (`dgelsd`, `sgelsd`, `zgelsd`, `cgelsd`) depending on precision.
///
/// The optional parameter [rcond] acts as the cut-off ratio for small singular values.
/// Singular values smaller than `rcond * largest_singular_value` are treated as zero.
/// If [rcond] is omitted or null, a negative value is passed to the LAPACK solver,
/// which falls back to using the machine precision to determine the effective rank.
///
/// The optional recycler parameter [out] allows reusing an existing array for the output,
/// avoiding new memory allocation.
///
/// **Preconditions:**
/// - Input matrix [a] must be 2-dimensional of shape `[M, N]`.
/// - Input array [b] must be 1-dimensional of shape `[M]` or 2-dimensional of shape `[M, K]`.
/// - The first dimension of [b] must exactly match the first dimension of [a] ($M$).
/// - Input arrays [a] and [b] must have the matching floating-point or complex [DType]. Integers or boolean
///   arrays are not supported.
/// - If provided, the recycler [out] must have the shape `[N]` (if [b] is 1D) or `[N, K]` (if [b] is 2D),
///   and its dtype must exactly match the dtype of [a] and [b].
///
/// **Throws:**
/// - [StateError] if [a] or [b] is disposed.
/// - [ArgumentError] if [a] or [b] does not have a floating-point or complex DType.
/// - [ArgumentError] if [b]'s DType does not match [a]'s DType.
/// - [ArgumentError] if [a] is not 2D, or [b] is not 1D or 2D.
/// - [ArgumentError] if [b]'s first dimension does not match [a]'s first dimension.
/// - [StateError] if [out] is provided but disposed.
/// - [ArgumentError] if [out] has mismatched shape or dtype.
/// - [StateError] if native FFI memory allocation fails or the SVD solver fails to converge.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(M N \min(M, N))$ operations executed natively.
///
/// **Example:**
/// {@example /example/linalg_lstsq_example.dart lang=dart}
///
/// Reference: [NumPy linalg.lstsq](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
LstsqResult<T> lstsq<T>(
  NDArray<T> a,
  NDArray<T> b, {
  double? rcond,
  NDArray<T>? out,
}) {
  if (a.isDisposed || b.isDisposed) {
    throw StateError('Cannot execute lstsq() on a disposed array.');
  }
  if (!a.dtype.isFloating && !a.dtype.isComplex) {
    throw ArgumentError(
      'Input array a must have a floating-point or complex DType (was ${a.dtype}).',
    );
  }
  if (a.dtype != b.dtype) {
    throw ArgumentError(
      'Input array b must have the matching DType as a (expected ${a.dtype}, was ${b.dtype}).',
    );
  }
  if (a.shape.length != 2) {
    throw ArgumentError(
      'Input matrix a must be 2-dimensional (was shape ${a.shape}).',
    );
  }
  if (b.shape.length != 1 && b.shape.length != 2) {
    throw ArgumentError(
      'Input right-hand side b must be 1D or 2D (was shape ${b.shape}).',
    );
  }
  final m = a.shape[0];
  final n = a.shape[1];
  if (b.shape[0] != m) {
    throw ArgumentError(
      'First dimension of b (${b.shape[0]}) must match first dimension of a ($m).',
    );
  }

  final nrhs = b.shape.length > 1 ? b.shape[1] : 1;

  if (out != null) {
    if (out.isDisposed) {
      throw StateError('Cannot write to a disposed out buffer.');
    }
    final expectedXShape = b.shape.length > 1 ? [n, nrhs] : [n];
    if (!listEquals(out.shape, expectedXShape) || out.dtype != a.dtype) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  // Create a contiguous copy of `a` (overwrite-safe)
  final aCopy = a.copy();

  // Row-major LAPACKE_gelsd requires b array size to be max(m, n) * nrhs
  final maxMN = m > n ? m : n;
  final bCopyShape = b.shape.length > 1 ? [maxMN, nrhs] : [maxMN];
  final bCopy = NDArray<T>.zeros(bCopyShape, a.dtype);

  // Copy b into bCopy
  final byteCount = b.data.length * a.dtype.byteWidth;
  if (b.isContiguous) {
    ffi.Pointer.fromAddress(bCopy.pointer.address)
        .cast<ffi.Uint8>()
        .asTypedList(byteCount)
        .setAll(
          0,
          ffi.Pointer.fromAddress(
            b.pointer.address,
          ).cast<ffi.Uint8>().asTypedList(byteCount),
        );
  } else {
    final bContig = b.copy();
    ffi.Pointer.fromAddress(bCopy.pointer.address)
        .cast<ffi.Uint8>()
        .asTypedList(byteCount)
        .setAll(
          0,
          ffi.Pointer.fromAddress(
            bContig.pointer.address,
          ).cast<ffi.Uint8>().asTypedList(byteCount),
        );
    bContig.dispose();
  }

  final minMN = m < n ? m : n;
  // Singular values s is always real
  final sDType = (a.dtype == DType.complex64 || a.dtype == DType.float32)
      ? DType.float32
      : DType.float64;
  final s = NDArray<double>.zeros([minMN], sDType as dynamic);

  final marker = ScratchArena.marker;
  final rankPtr = ScratchArena.allocate<ffi.Int>(4);
  final resolvedRcond = rcond ?? -1.0; // negative rcond uses machine precision

  try {
    int info;
    switch (a.dtype) {
      case DType.float64:
        info = LAPACKE_dgelsd(
          101, // ROW_MAJOR
          m,
          n,
          nrhs,
          aCopy.pointer.cast<ffi.Double>(),
          n,
          bCopy.pointer.cast<ffi.Double>(),
          nrhs,
          s.pointer.cast<ffi.Double>(),
          resolvedRcond,
          rankPtr,
        );
      case DType.float32:
        info = LAPACKE_sgelsd(
          101,
          m,
          n,
          nrhs,
          aCopy.pointer.cast<ffi.Float>(),
          n,
          bCopy.pointer.cast<ffi.Float>(),
          nrhs,
          s.pointer.cast<ffi.Float>(),
          resolvedRcond,
          rankPtr,
        );
      case DType.complex128:
        info = LAPACKE_zgelsd(
          101,
          m,
          n,
          nrhs,
          aCopy.pointer.cast<ffi.Double>(),
          n,
          bCopy.pointer.cast<ffi.Double>(),
          nrhs,
          s.pointer.cast<ffi.Double>(),
          resolvedRcond,
          rankPtr,
        );
      case DType.complex64:
        info = LAPACKE_cgelsd(
          101,
          m,
          n,
          nrhs,
          aCopy.pointer.cast<ffi.Float>(),
          n,
          bCopy.pointer.cast<ffi.Float>(),
          nrhs,
          s.pointer.cast<ffi.Float>(),
          resolvedRcond,
          rankPtr,
        );
      default:
        throw UnimplementedError(
          'Unsupported target DType for lstsq: ${a.dtype}',
        );
    }

    if (info < 0) {
      throw ArgumentError('Illegal value in call to LAPACKE gelsd: $info');
    }
    if (info > 0) {
      throw StateError(
        'The SVD algorithm in LAPACKE gelsd failed to converge ($info).',
      );
    }

    final rank = rankPtr.value;

    // Extract solution x: first n rows of bCopy
    final xShape = b.shape.length > 1 ? [n, nrhs] : [n];
    final x = out ?? NDArray<T>.zeros(xShape, a.dtype);
    final elementsToCopy = n * nrhs;
    x.data.setRange(0, elementsToCopy, bCopy.data.sublist(0, elementsToCopy));

    // Extract residuals: sum of squares of elements from row n to m-1 for each column
    final NDArray<double> residuals;
    if (m > n && rank == n) {
      final resShape = b.shape.length > 1 ? [nrhs] : [1];
      residuals = NDArray<double>.zeros(resShape, sDType as dynamic);
      if (a.dtype.isComplex) {
        for (var j = 0; j < nrhs; j++) {
          var sum = 0.0;
          for (var i = n; i < m; i++) {
            final complexVal = bCopy.data[i * nrhs + j] as Complex;
            sum +=
                complexVal.real * complexVal.real +
                complexVal.imag * complexVal.imag;
          }
          residuals.data[j] = sum;
        }
      } else {
        for (var j = 0; j < nrhs; j++) {
          var sum = 0.0;
          for (var i = n; i < m; i++) {
            final val = bCopy.data[i * nrhs + j] as num;
            sum += val * val;
          }
          residuals.data[j] = sum;
        }
      }
    } else {
      residuals = NDArray<double>.zeros([0], sDType as dynamic);
    }

    // Attach to scope or return
    if (out == null) {
      x.detachToParentScope();
    }
    residuals.detachToParentScope();
    s.detachToParentScope();

    return LstsqResult<T>(x: x, residuals: residuals, rank: rank, s: s);
  } finally {
    ScratchArena.reset(marker);
    aCopy.dispose();
    bCopy.dispose();
  }
}
