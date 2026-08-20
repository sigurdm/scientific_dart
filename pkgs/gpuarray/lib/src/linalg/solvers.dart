// ignore_for_file: non_constant_identifier_names
import 'dart:math' as math;
import '../dtype.dart';
import '../gpu_array.dart';
import '../exceptions.dart';
import '../backend/compute_engine.dart';
import 'decompositions.dart';

/// Result of sign and natural logarithm of determinant (slogdet).
final class SlogdetResult<T> {
  /// Sign of determinant (-1, 0, or 1).
  final GpuArray<T> sign;

  /// Natural logarithm of the absolute value of determinant.
  final GpuArray<T> logabsdet;

  const SlogdetResult({required this.sign, required this.logabsdet});

  @override
  String toString() => 'SlogdetResult(sign: $sign, logabsdet: $logabsdet)';
}

/// Solves a linear matrix equation $A x = b$.
GpuArray<T> solve<T>(GpuArray<T> a, GpuArray<T> b) {
  if (a.rank < 2) {
    throw ArgumentError('solve() requires matrix A of at least 2 dimensions.');
  }
  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m != n) {
    throw GpuShapeMismatchException('solve', a.shape, a.shape);
  }

  final luRes = lu_factor(a);
  return lu_solve(luRes.lu, luRes.piv, b);
}

/// Computes the multiplicative inverse of a square matrix [a].
GpuArray<T> inv<T>(GpuArray<T> a) {
  if (a.rank < 2) {
    throw ArgumentError('inv() requires matrix of at least 2 dimensions.');
  }
  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m != n) {
    throw GpuShapeMismatchException('inv', a.shape, a.shape);
  }

  final batchShape = a.shape.sublist(0, a.rank - 2);
  final eyeShape = [...batchShape, n, n];
  final eye = GpuArray<T>.zeros(eyeShape, a.dtype, device: a.device);

  final batchSize = ShapeUtils.computeSize(eyeShape) ~/ (n * n);
  for (var b = 0; b < batchSize; b++) {
    for (var i = 0; i < n; i++) {
      ComputeEngine.writeValue(eye.buffer, a.dtype, b * n * n + i * n + i, 1.0);
    }
  }

  return solve(a, eye);
}

/// Computes the Moore-Penrose pseudo-inverse of a matrix [a].
GpuArray<T> pinv<T>(GpuArray<T> a, {double rcond = 1e-15}) {
  if (a.rank < 2) {
    throw ArgumentError('pinv() requires matrix of at least 2 dimensions.');
  }

  final svdRes = svd(a, fullMatrices: false);
  final u = svdRes.u;
  final s = svdRes.s;
  final vt = svdRes.vt;

  final k = s.shape[s.rank - 1];
  final sData = s.toList().cast<num>().map((e) => e.toDouble()).toList();
  final batchSize = ShapeUtils.computeSize(s.shape) ~/ k;

  final sInvData = List<double>.filled(batchSize * k, 0.0);
  for (var b = 0; b < batchSize; b++) {
    final maxS = sData[b * k]; // already sorted descending
    final cutoff = rcond * maxS;
    for (var i = 0; i < k; i++) {
      final val = sData[b * k + i];
      sInvData[b * k + i] = (val > cutoff) ? 1.0 / val : 0.0;
    }
  }

  // V * S_inv * U^T = (V * S_inv) @ U^T
  // vt is (k, N) -> v is vt.transpose(-1, -2) which is (N, k)
  final v = vt.swapaxes(-1, -2);
  final sInvDiag = GpuArray<T>.zeros(
    [...s.shape.sublist(0, s.rank - 1), k, k],
    a.dtype,
    device: a.device,
  );

  for (var b = 0; b < batchSize; b++) {
    for (var i = 0; i < k; i++) {
      ComputeEngine.writeValue(
        sInvDiag.buffer,
        a.dtype,
        b * k * k + i * k + i,
        sInvData[b * k + i],
      );
    }
  }

  final vSinv = v.matmul<T>(sInvDiag);
  final uT = u.swapaxes(-1, -2);
  return vSinv.matmul<T>(uT);
}

/// Computes the determinant of a square matrix [a].
dynamic det<T>(GpuArray<T> a) {
  if (a.rank < 2) {
    throw ArgumentError('det() requires matrix of at least 2 dimensions.');
  }
  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m != n) {
    throw GpuShapeMismatchException('det', a.shape, a.shape);
  }

  final luFact = lu_factor(a);
  final lu = luFact.lu;
  final piv = luFact.piv;

  final batchShape = a.shape.sublist(0, a.rank - 2);
  final outShape = batchShape;
  final result = GpuArray<T>.empty(outShape, a.dtype, device: a.device);

  final batchSize = ShapeUtils.computeSize(a.shape) ~/ (n * n);
  final luFlat = lu.toNDArray();
  final luData = luFlat.toList().cast<num>().map((e) => e.toDouble()).toList();
  final pivData = piv.toList().cast<int>();

  for (var b = 0; b < batchSize; b++) {
    var d = 1.0;
    for (var i = 0; i < n; i++) {
      d *= luData[b * n * n + i * n + i];
    }

    var swaps = 0;
    for (var i = 0; i < n; i++) {
      if (pivData[b * n + i] != i) {
        swaps++;
      }
    }
    final permSign = (swaps % 2 == 1) ? -1.0 : 1.0;

    ComputeEngine.writeValue(result.buffer, a.dtype, b, permSign * d);
  }

  luFlat.dispose();

  return result.shape.isEmpty ? result.item() : result;
}

/// Computes the sign and natural logarithm of the determinant of matrix [a].
SlogdetResult<T> slogdet<T>(GpuArray<T> a) {
  if (a.rank < 2) {
    throw ArgumentError('slogdet() requires matrix of at least 2 dimensions.');
  }
  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  if (m != n) {
    throw GpuShapeMismatchException('slogdet', a.shape, a.shape);
  }

  final luFact = lu_factor(a);
  final lu = luFact.lu;
  final piv = luFact.piv;

  final outShape = a.shape.sublist(0, a.rank - 2);
  final signArr = GpuArray<T>.empty(outShape, a.dtype, device: a.device);
  final logabsdetArr = GpuArray<T>.empty(outShape, a.dtype, device: a.device);

  final batchSize = ShapeUtils.computeSize(a.shape) ~/ (n * n);
  final luFlat = lu.toNDArray();
  final luData = luFlat.toList().cast<num>().map((e) => e.toDouble()).toList();
  final pivData = piv.toList().cast<int>();

  for (var b = 0; b < batchSize; b++) {
    var logSum = 0.0;
    var sign = 1.0;

    for (var i = 0; i < n; i++) {
      final diag = luData[b * n * n + i * n + i];
      if (diag == 0.0) {
        sign = 0.0;
        logSum = double.negativeInfinity;
        break;
      }
      if (diag < 0.0) {
        sign = -sign;
      }
      logSum += math.log(diag.abs());
    }

    if (sign != 0.0) {
      var swaps = 0;
      for (var i = 0; i < n; i++) {
        if (pivData[b * n + i] != i) {
          swaps++;
        }
      }
      if (swaps % 2 == 1) {
        sign = -sign;
      }
    }

    ComputeEngine.writeValue(signArr.buffer, a.dtype, b, sign);
    ComputeEngine.writeValue(logabsdetArr.buffer, a.dtype, b, logSum);
  }

  luFlat.dispose();

  return SlogdetResult<T>(sign: signArr, logabsdet: logabsdetArr);
}

/// Raises a square matrix to the integer power [n].
GpuArray<T> matrix_power<T>(GpuArray<T> a, int n) {
  if (a.rank < 2) {
    throw ArgumentError(
      'matrix_power() requires matrix of at least 2 dimensions.',
    );
  }
  final m = a.shape[a.rank - 2];
  final cols = a.shape[a.rank - 1];
  if (m != cols) {
    throw GpuShapeMismatchException('matrix_power', a.shape, a.shape);
  }

  if (n == 0) {
    final batchShape = a.shape.sublist(0, a.rank - 2);
    final eyeShape = [...batchShape, m, m];
    final eye = GpuArray<T>.zeros(eyeShape, a.dtype, device: a.device);
    final batchSize = ShapeUtils.computeSize(eyeShape) ~/ (m * m);
    for (var b = 0; b < batchSize; b++) {
      for (var i = 0; i < m; i++) {
        ComputeEngine.writeValue(
          eye.buffer,
          a.dtype,
          b * m * m + i * m + i,
          1.0,
        );
      }
    }
    return eye;
  }

  var base = (n < 0) ? inv(a) : a;
  var exp = n.abs();

  var result = matrix_power(base, 0);
  while (exp > 0) {
    if (exp % 2 == 1) {
      result = result.matmul<T>(base);
    }
    base = base.matmul<T>(base);
    exp ~/= 2;
  }

  return result;
}

/// Computes the numerical rank of a matrix [a] using SVD.
dynamic matrix_rank<T>(GpuArray<T> a, {double? tol}) {
  if (a.rank < 2) {
    throw ArgumentError(
      'matrix_rank() requires matrix of at least 2 dimensions.',
    );
  }
  final m = a.shape[a.rank - 2];
  final n = a.shape[a.rank - 1];
  final k = math.min(m, n);

  final s = svd(a, fullMatrices: false, computeUv: false).s;
  final sData = s.toList().cast<num>().map((e) => e.toDouble()).toList();
  final batchSize = ShapeUtils.computeSize(s.shape) ~/ k;

  final outShape = a.shape.sublist(0, a.rank - 2);
  final result = GpuArray<Int32>.empty(outShape, DType.int32, device: a.device);

  for (var b = 0; b < batchSize; b++) {
    final maxS = sData[b * k];
    final threshold = tol ?? (maxS * math.max(m, n) * 1e-15);
    var count = 0;
    for (var i = 0; i < k; i++) {
      if (sData[b * k + i] > threshold) count++;
    }
    ComputeEngine.writeAny(result.buffer, DType.int32, b, count);
  }

  return result.shape.isEmpty ? result.item() : result;
}

/// Matrix or vector norm.
dynamic norm<T>(
  GpuArray<T> a, {
  dynamic ord,
  dynamic axis,
  bool keepdims = false,
}) {
  if (axis == null) {
    // 2-norm / Frobenius norm over all elements
    final flat = a.flatten();
    final sumSq = flat.multiply(flat).sum();
    final val = math.sqrt(sumSq.item() as num);
    if (keepdims) {
      final shape = List<int>.filled(a.rank, 1);
      return GpuArray.fromList([val], shape, a.dtype, device: a.device);
    }
    return val;
  }

  if (axis is int) {
    final ax = axis < 0 ? axis + a.rank : axis;
    final sq = a.multiply(a);
    final sumAx = sq.sum(axis: ax, keepDims: keepdims);
    return sumAx.sqrt();
  }

  throw ArgumentError('Unsupported norm axis: $axis');
}

/// Computes the condition number of a matrix [a].
dynamic cond<T>(GpuArray<T> a, {dynamic p}) {
  final s = svd(a, fullMatrices: false, computeUv: false).s;
  final k = s.shape[s.rank - 1];
  final sData = s.toList().cast<num>().map((e) => e.toDouble()).toList();
  final batchSize = ShapeUtils.computeSize(s.shape) ~/ k;

  final outShape = a.shape.sublist(0, a.rank - 2);
  final result = GpuArray<T>.empty(outShape, a.dtype, device: a.device);

  for (var b = 0; b < batchSize; b++) {
    final maxS = sData[b * k];
    final minS = sData[b * k + k - 1];
    final c = (minS > 1e-15) ? (maxS / minS) : double.infinity;
    ComputeEngine.writeValue(result.buffer, a.dtype, b, c);
  }

  return result.shape.isEmpty ? result.item() : result;
}
