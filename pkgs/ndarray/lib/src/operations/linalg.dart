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
    hide s_det_double, s_det_float, s_det_complex_double, s_det_complex_float;

// Standalone operational relative cross-imports
import 'math.dart';
import 'helpers.dart';

/// Matrix multiplication using OpenBLAS, supporting high-dimensional stack broadcasting and 1D vector promotions.
NDArray<R> matmul<Ta, Tb, R>(NDArray<Ta> a, NDArray<Tb> b, {NDArray<R>? out}) {
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
              out ??
              (NDArray.fromList([scalarRes], [], DType.float64) as NDArray<R>);
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
              out ??
              (NDArray.fromList([scalarRes], [], DType.float32) as NDArray<R>);
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
NDArray<T> inv<T>(NDArray<T> a, {NDArray<T>? out}) {
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
            throw SingularMatrixException('Matrix is singular and cannot be inverted');
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
            throw SingularMatrixException('Matrix is singular and cannot be inverted');
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
            throw SingularMatrixException('Matrix is singular and cannot be inverted');
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
            throw SingularMatrixException('Matrix is singular and cannot be inverted');
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
/// Compute the determinant of a square matrix or a stack of square matrices using OpenBLAS/LAPACK.
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
/// - Algorithmic complexity is $O(N^3)$ leveraging optimized native LAPACK linear algebra solvers.
/// - Fully vectorized and batched in native C for float64, complex64, and complex128, minimizing FFI transitions.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final d = det(a);
/// print(d); // -2.0
/// ```
/// Compute the determinant of a square matrix or a stack of square matrices using OpenBLAS/LAPACK.
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
/// - Algorithmic complexity is $O(N^3)$ leveraging optimized native LAPACK linear algebra solvers.
/// - Fully vectorized and batched in native C for float64, complex64, and complex128, minimizing FFI transitions.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final d = det(a);
/// print(d); // -2.0
/// ```
///
/// Refer to the [determinant](https://en.wikipedia.org/wiki/Determinant)
/// and [LAPACK LU solver](https://en.wikipedia.org/wiki/LU_decomposition) for additional details.
NDArray<T> det<T>(NDArray<T> a, {NDArray<T>? out}) {
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
        final result = out ??
            (NDArray.zeros(stackShape, DType.float64) as NDArray<T>);
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
        final result = out ??
            (NDArray.zeros(stackShape, DType.complex128) as NDArray<T>);
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
        final result = out ??
            (NDArray.zeros(stackShape, DType.complex64) as NDArray<T>);
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
        final result = out ??
            (NDArray.zeros(stackShape, DType.float32) as NDArray<T>);
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
/// - Algorithmic complexity is $O(N^3)$ executed in native C-compiled space.
///
/// **Example:**
/// ```dart
/// final a = `NDArray<double>`.fromList([3.0, 1.0, 1.0, 2.0], [2, 2], DType.float64);
/// final b = `NDArray<double>`.fromList([9.0, 8.0], [2], DType.float64);
/// final x = solve(a, b);
/// print(x.toList()); // [2.0, 3.0]
/// ```
NDArray<T> solve<T>(NDArray<T> a, NDArray<T> b, {NDArray<T>? out}) {
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
            throw SingularMatrixException('Matrix is singular and cannot be solved');
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
            throw SingularMatrixException('Matrix is singular and cannot be solved');
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
            throw SingularMatrixException('Matrix is singular and cannot be solved');
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
            throw SingularMatrixException('Matrix is singular and cannot be solved');
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

/// Compute the eigenvalues and right eigenvectors of a square array or stack of square arrays.
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

/// Compute the Cholesky decomposition of a square, positive-definite 2D matrix.
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
/// Natively offloads to LAPACK solvers (`dgeqrf` / `sgeqrf` and `dorgqr` / `sorgqr`) depending on precision.
///
/// **Preconditions:**
/// - Input matrix [a] must be at least 2-dimensional.
///
/// **Throws:**
/// - [ArgumentError] if [a] rank is less than 2.
/// - [StateError] if native FFI memory allocation or LAPACK solver initialization fails.
///
/// **Performance considerations:**
/// - Executes at high-speed natively in unmanaged C space.
///
/// **Example:**
/// ```dart
/// final a = `NDArray<double>`.fromList([12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0], [3, 3], DType.float64);
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
/// Natively offloads to LAPACK solvers (`dgesdd` / `sgesdd`) depending on precision.
///
/// **Preconditions:**
/// - Input matrix [a] must be at least 2-dimensional.
///
/// **Throws:**
/// - [ArgumentError] if [a] rank is less than 2.
/// - [StateError] if native FFI memory allocation or LAPACK solver initialization fails.
///
/// **Performance considerations:**
/// - Executes at high-speed natively in unmanaged C space.
///
/// **Example:**
/// ```dart
/// final a = `NDArray<double>`.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], DType.float64);
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
  final NDArray<double> sMat = out?.S ?? NDArray<double>.zeros(
    sShape,
    dtypeS as DType<double>,
  );
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

  return (
    U: uMat,
    S: sMat,
    Vh: vtMat,
  );
}




