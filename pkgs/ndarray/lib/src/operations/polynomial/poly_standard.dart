// Standard polynomial operations (polyval, polyfit, roots).
library;

import "../../ndarray.dart";
import "../helpers.dart";
import "../math.dart";
import "../linalg.dart";

Object _divScalar(Object a, Object b) {
  if (a is Complex || b is Complex) {
    final ca = a is Complex ? a : Complex((a as num).toDouble(), 0.0);
    final cb = b is Complex ? b : Complex((b as num).toDouble(), 0.0);
    return ca / cb;
  }
  return (a as num).toDouble() / (b as num).toDouble();
}

Object _mulScalar(Object a, Object b) {
  if (a is Complex || b is Complex) {
    final ca = a is Complex ? a : Complex((a as num).toDouble(), 0.0);
    final cb = b is Complex ? b : Complex((b as num).toDouble(), 0.0);
    return ca * cb;
  }
  return (a as num).toDouble() * (b as num).toDouble();
}

Object _negScalar(Object a) {
  if (a is Complex) {
    return -a;
  }
  return -(a as num).toDouble();
}

bool _isZeroScalar(Object a) {
  if (a is Complex) {
    return a.real == 0.0 && a.imag == 0.0;
  }
  return (a as num) == 0;
}

NDArray<R> _ensureDType<T, R>(NDArray<T> a, DType<R> targetDType) {
  if (a.dtype == targetDType) {
    return a as NDArray<R>;
  }
  return castNDArray(a, targetDType);
}

NDArray<R> _makeScalar<R>(Object val, DType<R> dtype) {
  final norm = normalizeScalar(val, dtype) as R;
  return NDArray<R>.scalar(norm, dtype: dtype);
}

NDArray<R> _filledArray<R>(List<int> shape, Object scalarVal, DType<R> dtype) {
  final ones = NDArray<R>.ones(shape, dtype);
  return multiply(ones, _makeScalar<R>(scalarVal, dtype));
}

void _copyInto<R>(NDArray src, NDArray<R> out) {
  src.copy(out: out);
}

/// Evaluates a polynomial with coefficients [c] at points [x].
///
/// If [c] has length N, this function evaluates:
/// p(x) = c[0] x^(N-1) + c[1] x^(N-2) + ... + c[N-1]
///
/// **Preconditions:**
/// - [c] and [x] must not be disposed.
/// - [c] must be a 1-dimensional array.
/// - [c] must not be empty.
///
/// **Throws:**
/// - [StateError] if any input array or [out] buffer is disposed.
/// - [ArgumentError] if [c] is not 1-dimensional, or if [c] is empty.
/// - [ArgumentError] if [out] shape or dtype is incompatible with [x].
///
/// Reference: [NumPy polyval](https://numpy.org/doc/stable/reference/generated/numpy.polyval.html)
NDArray<R> polyval<Tc, Tx, R>(NDArray<Tc> c, NDArray<Tx> x, {NDArray<R>? out}) {
  if (c.isDisposed || x.isDisposed || (out != null && out.isDisposed)) {
    throw StateError("Cannot execute polyval() on a disposed array.");
  }
  if (c.shape.length != 1) {
    throw ArgumentError("Coefficient array c must be 1-dimensional.");
  }
  if (c.shape[0] == 0) {
    throw ArgumentError("Coefficient array c must not be empty.");
  }

  final targetDType = resolveDType(c.dtype, x.dtype) as DType<R>;
  if (out != null) {
    if (!listEquals(out.shape, x.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        "Incompatible out buffer shape or dtype for polyval.",
      );
    }
  }

  return NDArray.scope(() {
    final n = c.shape[0];

    if (n == 1) {
      final c0 = c.getCell([0]) as Object;
      final res = _filledArray(x.shape, c0, targetDType);
      if (out != null) {
        _copyInto(res, out);
        return out;
      }
      return res.detachToParentScope();
    }

    final xCast = _ensureDType(x, targetDType);
    var val = _filledArray(x.shape, c.getCell([0]) as Object, targetDType);
    for (var i = 1; i < n; i++) {
      val = add(
        multiply(val, xCast),
        _makeScalar<R>(c.getCell([i]) as Object, targetDType),
      );
    }

    if (out != null) {
      _copyInto(val, out);
      return out;
    }
    return val.detachToParentScope();
  });
}

/// Least-squares fit of a polynomial to data points ([x], [y]).
///
/// Fits a polynomial of degree [deg] to points (x_i, y_i) by minimizing squared error.
/// Optionally weighted by [w].
/// Returns an array of coefficients of length deg + 1, ordered highest degree first.
///
/// **Preconditions:**
/// - [x], [y], and optional [w] must not be disposed.
/// - [x] and [y] must be 1-dimensional arrays of identical length.
/// - Length of [x] must be greater than [deg].
/// - Degree [deg] must be non-negative.
///
/// **Throws:**
/// - [StateError] if any input array or [out] buffer is disposed.
/// - [ArgumentError] if input arrays are not 1D, mismatch lengths, or if [deg] is invalid.
/// - [ArgumentError] if [out] shape or dtype is incompatible.
///
/// Reference: [NumPy polyfit](https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html)
NDArray<R> polyfit<Tx, Ty, Tw, R>(
  NDArray<Tx> x,
  NDArray<Ty> y,
  int deg, {
  NDArray<Tw>? w,
  double? rcond,
  NDArray<R>? out,
}) {
  if (x.isDisposed ||
      y.isDisposed ||
      (w != null && w.isDisposed) ||
      (out != null && out.isDisposed)) {
    throw StateError("Cannot execute polyfit() on a disposed array.");
  }
  if (x.shape.length != 1 || y.shape.length != 1) {
    throw ArgumentError("Input arrays x and y must be 1-dimensional.");
  }
  if (x.shape[0] != y.shape[0]) {
    throw ArgumentError("Input arrays x and y must have equal length.");
  }
  if (deg < 0) {
    throw ArgumentError("Polynomial degree deg must be non-negative.");
  }
  final m = x.shape[0];
  if (m <= deg) {
    throw ArgumentError(
      "Number of data points ($m) must be greater than deg ($deg).",
    );
  }
  if (w != null && (w.shape.length != 1 || w.shape[0] != m)) {
    throw ArgumentError("Weights w must be a 1D array of same length as x.");
  }

  var resolvedType = resolveDType(x.dtype, y.dtype);
  if (w != null) {
    resolvedType = resolveDType(resolvedType, w.dtype);
  }
  final targetDType = resolvedType as DType<R>;

  if (out != null) {
    if (!listEquals(out.shape, [deg + 1]) || out.dtype != targetDType) {
      throw ArgumentError(
        "Incompatible out buffer shape or dtype for polyfit.",
      );
    }
  }

  return NDArray.scope(() {
    final nCols = deg + 1;
    final vMat = NDArray<R>.zeros([m, nCols], targetDType);
    final xCast = _ensureDType(x, targetDType);

    for (var j = 0; j < nCols; j++) {
      final p = deg - j;
      NDArray col;
      if (p == 0) {
        col = NDArray<R>.ones([m], targetDType);
      } else if (p == 1) {
        col = xCast;
      } else {
        col = power(xCast, _makeScalar<R>(p, targetDType));
      }

      for (var i = 0; i < m; i++) {
        vMat.setCell([i, j], col.getCell([i]) as R);
      }
    }

    NDArray<R> rhs = _ensureDType(y, targetDType);
    NDArray<R> lhs = vMat;

    if (w != null) {
      final wCast = _ensureDType(w, targetDType);
      final lhsW = NDArray<R>.zeros([m, nCols], targetDType);
      final rhsW = NDArray<R>.zeros([m], targetDType);

      for (var i = 0; i < m; i++) {
        final wi = wCast.getCell([i]) as Object;
        final yi = rhs.getCell([i]) as Object;
        rhsW.setCell([i], castValue(_mulScalar(wi, yi), targetDType) as R);

        for (var j = 0; j < nCols; j++) {
          final vij = vMat.getCell([i, j]) as Object;
          lhsW.setCell([
            i,
            j,
          ], castValue(_mulScalar(wi, vij), targetDType) as R);
        }
      }
      lhs = lhsW;
      rhs = rhsW;
    }

    final lstsqRes = lstsq(lhs, rhs, rcond: rcond);
    final coeffs = lstsqRes.x;

    if (out != null) {
      _copyInto(coeffs, out);
      return out;
    }
    return coeffs.detachToParentScope();
  });
}

/// Computes the roots of a polynomial with coefficients [p].
///
/// The coefficient array [p] is ordered from highest degree to constant term.
/// Returns an `NDArray<Complex>` containing the roots.
///
/// **Preconditions:**
/// - [p] and optional [out] must not be disposed.
/// - [p] must be a 1-dimensional array.
///
/// **Throws:**
/// - [StateError] if [p] or [out] buffer is disposed.
/// - [ArgumentError] if [p] is not 1-dimensional.
///
/// Reference: [NumPy roots](https://numpy.org/doc/stable/reference/generated/numpy.roots.html)
NDArray<Complex> roots<T>(NDArray<T> p, {NDArray<Complex>? out}) {
  if (p.isDisposed || (out != null && out.isDisposed)) {
    throw StateError("Cannot execute roots() on a disposed array.");
  }
  if (p.shape.length != 1) {
    throw ArgumentError("Coefficient array p must be 1-dimensional.");
  }

  return NDArray.scope(() {
    final size = p.shape[0];
    var firstNonZero = -1;
    for (var i = 0; i < size; i++) {
      if (!_isZeroScalar(p.getCell([i]) as Object)) {
        firstNonZero = i;
        break;
      }
    }

    if (firstNonZero == -1 || (size - firstNonZero) <= 1) {
      final res = NDArray<Complex>.zeros([0], DType.complex128);
      if (out != null) {
        if (!listEquals(out.shape, [0]) || out.dtype != DType.complex128) {
          throw ArgumentError(
            "Incompatible out buffer for empty roots result.",
          );
        }
        _copyInto(res, out);
        return out;
      }
      return res.detachToParentScope();
    }

    final nCoeffs = size - firstNonZero;
    final deg = nCoeffs - 1;

    if (deg == 1) {
      final c0 = p.getCell([firstNonZero]) as Object;
      final c1 = p.getCell([firstNonZero + 1]) as Object;
      final rootVal = _divScalar(_negScalar(c1), c0);
      final complexRoot = rootVal is Complex
          ? rootVal
          : Complex((rootVal as num).toDouble(), 0.0);
      final res = NDArray<Complex>.fromList(
        [complexRoot],
        [1],
        DType.complex128,
      );
      if (out != null) {
        if (!listEquals(out.shape, [1]) || out.dtype != DType.complex128) {
          throw ArgumentError("Incompatible out buffer for roots result.");
        }
        _copyInto(res, out);
        return out;
      }
      return res.detachToParentScope();
    }

    final bool isComp =
        p.dtype == DType.complex64 || p.dtype == DType.complex128;
    final NDArray aMat;
    switch (p.dtype) {
      case DType.complex64:
      case DType.complex128:
        aMat = NDArray<Complex>.zeros([deg, deg], p.dtype as DType<Complex>);
        break;
      default:
        aMat = NDArray<double>.zeros([deg, deg], DType.float64);
        break;
    }

    final c0 = p.getCell([firstNonZero]) as Object;
    final targetMatDType = aMat.dtype;

    for (var j = 0; j < deg; j++) {
      final cj = p.getCell([firstNonZero + j + 1]) as Object;
      final val = _divScalar(_negScalar(cj), c0);
      aMat.setCell([0, j], castValue(val, targetMatDType));
    }
    for (var i = 1; i < deg; i++) {
      final one = isComp ? Complex(1.0, 0.0) : 1.0;
      aMat.setCell([i, i - 1], castValue(one, targetMatDType));
    }

    final res = eigvals(aMat, out: out);
    if (out != null) return out;
    return res.detachToParentScope();
  });
}
