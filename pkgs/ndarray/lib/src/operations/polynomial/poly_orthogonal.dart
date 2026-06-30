// Orthogonal polynomial series (Chebyshev, Legendre, Hermite).
library;

import "../../ndarray.dart";
import "../helpers.dart";
import "../math.dart";
import "../linalg.dart";

enum _OrthoKind { chebyshev, legendre, hermite }

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

Object _subScalar(Object a, Object b) {
  if (a is Complex || b is Complex) {
    final ca = a is Complex ? a : Complex((a as num).toDouble(), 0.0);
    final cb = b is Complex ? b : Complex((b as num).toDouble(), 0.0);
    return ca - cb;
  }
  return (a as num).toDouble() - (b as num).toDouble();
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

/// Evaluates a Chebyshev series at points [x] with coefficients [c].
///
/// Uses backward Clenshaw recurrence to evaluate p(x) = sum(c[i] * T_i(x)).
/// Supports flexible argument order (c, x) or (x, c).
///
/// **Preconditions:**
/// - Input arrays must not be disposed.
/// - Coefficient array must be 1-dimensional and non-empty.
///
/// **Throws:**
/// - [StateError] if any input or [out] buffer is disposed.
/// - [ArgumentError] if coefficient array is invalid or [out] buffer mismatches.
///
/// Reference: [NumPy chebval](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.chebyshev.chebval.html)
NDArray<R> chebval<T1, T2, R>(
  NDArray<T1> arg1,
  NDArray<T2> arg2, {
  NDArray<R>? out,
}) {
  NDArray cArr;
  NDArray xArr;
  if (arg1.shape.length != 1 && arg2.shape.length == 1) {
    xArr = arg1;
    cArr = arg2;
  } else {
    cArr = arg1;
    xArr = arg2;
  }
  return _evalClenshaw(cArr, xArr, _OrthoKind.chebyshev, out: out);
}

/// Evaluates a Legendre series at points [x] with coefficients [c].
///
/// Uses backward Clenshaw recurrence to evaluate p(x) = sum(c[i] * P_i(x)).
/// Supports flexible argument order (c, x) or (x, c).
///
/// Reference: [NumPy legval](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.legendre.legval.html)
NDArray<R> legval<T1, T2, R>(
  NDArray<T1> arg1,
  NDArray<T2> arg2, {
  NDArray<R>? out,
}) {
  NDArray cArr;
  NDArray xArr;
  if (arg1.shape.length != 1 && arg2.shape.length == 1) {
    xArr = arg1;
    cArr = arg2;
  } else {
    cArr = arg1;
    xArr = arg2;
  }
  return _evalClenshaw(cArr, xArr, _OrthoKind.legendre, out: out);
}

/// Evaluates a Hermite series at points [x] with coefficients [c].
///
/// Uses backward Clenshaw recurrence to evaluate p(x) = sum(c[i] * H_i(x)).
/// Supports flexible argument order (c, x) or (x, c).
///
/// Reference: [NumPy hermval](https://numpy.org/doc/stable/reference/generated/numpy.polynomial.hermite.hermval.html)
NDArray<R> hermval<T1, T2, R>(
  NDArray<T1> arg1,
  NDArray<T2> arg2, {
  NDArray<R>? out,
}) {
  NDArray cArr;
  NDArray xArr;
  if (arg1.shape.length != 1 && arg2.shape.length == 1) {
    xArr = arg1;
    cArr = arg2;
  } else {
    cArr = arg1;
    xArr = arg2;
  }
  return _evalClenshaw(cArr, xArr, _OrthoKind.hermite, out: out);
}

NDArray<R> _evalClenshaw<Tc, Tx, R>(
  NDArray<Tc> c,
  NDArray<Tx> x,
  _OrthoKind kind, {
  NDArray<R>? out,
}) {
  if (c.isDisposed || x.isDisposed || (out != null && out.isDisposed)) {
    throw StateError("Cannot execute series evaluation on a disposed array.");
  }
  if (c.shape.length != 1) {
    throw ArgumentError("Coefficient array c must be 1-dimensional.");
  }
  final m = c.shape[0] - 1;
  if (m < 0) {
    throw ArgumentError("Coefficient array c must not be empty.");
  }

  final targetDType = resolveDType(c.dtype, x.dtype) as DType<R>;
  if (out != null) {
    if (!listEquals(out.shape, x.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        "Incompatible out buffer shape or dtype for series evaluation.",
      );
    }
  }

  return NDArray.scope(() {
    final xCast = _ensureDType(x, targetDType);

    if (m == 0) {
      final c0 = c.getCell([0]) as Object;
      final res = _filledArray(x.shape, c0, targetDType);
      if (out != null) {
        _copyInto(res, out);
        return out;
      }
      return res.detachToParentScope();
    }

    NDArray b1 = NDArray<R>.zeros(x.shape, targetDType);
    NDArray b2 = NDArray<R>.zeros(x.shape, targetDType);

    switch (kind) {
      case _OrthoKind.chebyshev:
        final x2 = multiply(xCast, _makeScalar<R>(2.0, targetDType));
        for (var i = m; i >= 1; i--) {
          final tmp = b1;
          final ci = _makeScalar<R>(c.getCell([i]) as Object, targetDType);
          b1 = add(subtract(multiply(x2, b1), b2), ci);
          b2 = tmp;
        }
        final c0 = _makeScalar<R>(c.getCell([0]) as Object, targetDType);
        final res = add(subtract(multiply(xCast, b1), b2), c0);
        if (out != null) {
          _copyInto(res, out);
          return out;
        }
        return (res as NDArray<R>).detachToParentScope();

      case _OrthoKind.legendre:
        for (var k = m; k >= 1; k--) {
          final tmp = b1;
          final f1 = (2 * k + 1) / (k + 1);
          final f2 = (k + 1) / (k + 2);
          final ci = _makeScalar<R>(c.getCell([k]) as Object, targetDType);
          final term1 = multiply(
            multiply(xCast, b1),
            _makeScalar<R>(f1, targetDType),
          );
          final term2 = multiply(b2, _makeScalar<R>(f2, targetDType));
          b1 = add(subtract(term1, term2), ci);
          b2 = tmp;
        }
        final c0 = _makeScalar<R>(c.getCell([0]) as Object, targetDType);
        final term1 = multiply(xCast, b1);
        final term2 = multiply(b2, _makeScalar<R>(0.5, targetDType));
        final res = add(subtract(term1, term2), c0);
        if (out != null) {
          _copyInto(res, out);
          return out;
        }
        return (res as NDArray<R>).detachToParentScope();

      case _OrthoKind.hermite:
        final x2 = multiply(xCast, _makeScalar<R>(2.0, targetDType));
        for (var k = m; k >= 1; k--) {
          final tmp = b1;
          final f2 = 2.0 * k + 2.0;
          final ci = _makeScalar<R>(c.getCell([k]) as Object, targetDType);
          final term1 = multiply(x2, b1);
          final term2 = multiply(b2, _makeScalar<R>(f2, targetDType));
          b1 = add(subtract(term1, term2), ci);
          b2 = tmp;
        }
        final c0 = _makeScalar<R>(c.getCell([0]) as Object, targetDType);
        final term1 = multiply(x2, b1);
        final term2 = multiply(b2, _makeScalar<R>(2.0, targetDType));
        final res = add(subtract(term1, term2), c0);
        if (out != null) {
          _copyInto(res, out);
          return out;
        }
        return (res as NDArray<R>).detachToParentScope();
    }
  });
}

/// Finds roots of a Chebyshev series.
NDArray<Complex> chebroots<T>(NDArray<T> c, {NDArray<Complex>? out}) {
  return _orthoRoots(c, _OrthoKind.chebyshev, out: out);
}

/// Finds roots of a Legendre series.
NDArray<Complex> legroots<T>(NDArray<T> c, {NDArray<Complex>? out}) {
  return _orthoRoots(c, _OrthoKind.legendre, out: out);
}

/// Finds roots of a Hermite series.
NDArray<Complex> hermroots<T>(NDArray<T> c, {NDArray<Complex>? out}) {
  return _orthoRoots(c, _OrthoKind.hermite, out: out);
}

NDArray<Complex> _orthoRoots<T>(
  NDArray<T> c,
  _OrthoKind kind, {
  NDArray<Complex>? out,
}) {
  if (c.isDisposed || (out != null && out.isDisposed)) {
    throw StateError("Cannot execute root finding on a disposed array.");
  }
  if (c.shape.length != 1) {
    throw ArgumentError("Coefficient array c must be 1-dimensional.");
  }

  return NDArray.scope(() {
    var n = c.shape[0] - 1;
    while (n > 0) {
      if (!_isZeroScalar(c.getCell([n]) as Object)) break;
      n--;
    }
    if (n <= 0) {
      final res = NDArray<Complex>.zeros([0], DType.complex128);
      if (out != null) {
        _copyInto(res, out);
        return out;
      }
      return res.detachToParentScope();
    }

    final cn = c.getCell([n]) as Object;
    if (n == 1) {
      final c0 = c.getCell([0]) as Object;
      Object rootVal;
      switch (kind) {
        case _OrthoKind.chebyshev:
        case _OrthoKind.legendre:
          rootVal = _divScalar(_negScalar(c0), cn);
          break;
        case _OrthoKind.hermite:
          rootVal = _divScalar(_negScalar(c0), _mulScalar(cn, 2.0));
          break;
      }
      final complexRoot = rootVal is Complex
          ? rootVal
          : Complex((rootVal as num).toDouble(), 0.0);
      final res = NDArray<Complex>.fromList(
        [complexRoot],
        [1],
        DType.complex128,
      );
      if (out != null) {
        _copyInto(res, out);
        return out;
      }
      return res.detachToParentScope();
    }

    final bool isComp =
        c.dtype == DType.complex64 || c.dtype == DType.complex128;
    final NDArray cMat;
    switch (c.dtype) {
      case DType.complex64:
      case DType.complex128:
        cMat = NDArray<Complex>.zeros([n, n], c.dtype as DType<Complex>);
        break;
      default:
        cMat = NDArray<double>.zeros([n, n], DType.float64);
        break;
    }
    final targetMatDType = cMat.dtype;

    switch (kind) {
      case _OrthoKind.chebyshev:
        cMat.setCell([
          1,
          0,
        ], castValue(isComp ? Complex(1.0, 0.0) : 1.0, targetMatDType));
        for (var i = 1; i < n - 1; i++) {
          cMat.setCell([
            i + 1,
            i,
          ], castValue(isComp ? Complex(0.5, 0.0) : 0.5, targetMatDType));
        }
        for (var i = 0; i < n - 1; i++) {
          cMat.setCell([
            i,
            i + 1,
          ], castValue(isComp ? Complex(0.5, 0.0) : 0.5, targetMatDType));
        }
        for (var i = 0; i < n; i++) {
          final ci = c.getCell([i]) as Object;
          final factor = (i == n - 1) ? 1.0 : 2.0;
          final denom = _mulScalar(cn, factor);
          final norm = _divScalar(ci, denom);
          final cur = cMat.getCell([i, n - 1]) as Object;
          final updated = _subScalar(cur, norm);
          cMat.setCell([i, n - 1], castValue(updated, targetMatDType));
        }
        break;

      case _OrthoKind.legendre:
        for (var i = 0; i < n - 1; i++) {
          final sub = (i + 1) / (2 * i + 3);
          final sup = (i + 1) / (2 * i + 1);
          cMat.setCell([
            i + 1,
            i,
          ], castValue(isComp ? Complex(sub, 0.0) : sub, targetMatDType));
          cMat.setCell([
            i,
            i + 1,
          ], castValue(isComp ? Complex(sup, 0.0) : sup, targetMatDType));
        }
        final factor = (2 * n + 1) / n;
        for (var i = 0; i < n; i++) {
          final ci = c.getCell([i]) as Object;
          final denom = _mulScalar(cn, factor);
          final norm = _divScalar(ci, denom);
          final cur = cMat.getCell([i, n - 1]) as Object;
          final updated = _subScalar(cur, norm);
          cMat.setCell([i, n - 1], castValue(updated, targetMatDType));
        }
        break;

      case _OrthoKind.hermite:
        for (var i = 0; i < n - 1; i++) {
          cMat.setCell([
            i + 1,
            i,
          ], castValue(isComp ? Complex(0.5, 0.0) : 0.5, targetMatDType));
          cMat.setCell(
            [i, i + 1],
            castValue(
              isComp ? Complex((i + 1).toDouble(), 0.0) : (i + 1).toDouble(),
              targetMatDType,
            ),
          );
        }
        for (var i = 0; i < n; i++) {
          final ci = c.getCell([i]) as Object;
          final denom = _mulScalar(cn, 2.0);
          final norm = _divScalar(ci, denom);
          final cur = cMat.getCell([i, n - 1]) as Object;
          final updated = _subScalar(cur, norm);
          cMat.setCell([i, n - 1], castValue(updated, targetMatDType));
        }
        break;
    }

    final res = eigvals(cMat, out: out);
    if (out != null) return out;
    return res.detachToParentScope();
  });
}
