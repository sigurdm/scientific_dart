import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'ffi/flint_bindings.dart' as fl;
import 'package:resource_scope/resource_scope.dart';
import 'expr.dart';

final _fmpqPolyFinalizer = ffi.NativeFinalizer(
  ffi.Native.addressOf<
        ffi.NativeFunction<ffi.Void Function(ffi.Pointer<fl.fmpq_poly_struct>)>
      >(fl.fmpq_poly_clear)
      .cast<ffi.NativeFinalizerFunction>(),
);

/// Represents an exact univariate polynomial over the rational numbers Q[x]
/// backed by FLINT's `fmpq_poly` exact arithmetic engine.
///
/// Supports exact GCD, quotient/remainder division, derivative, integral, and
/// exact polynomial factorization over Q[x].
final class FlintRationalPoly implements ffi.Finalizable, ScopedResource {
  final ffi.Pointer<fl.fmpq_poly_struct> _ptr;
  bool _disposed = false;

  FlintRationalPoly._(this._ptr) {
    if (_ptr == ffi.nullptr) {
      throw StateError('Cannot wrap a nullptr fmpq_poly_struct');
    }
    _fmpqPolyFinalizer.attach(this, _ptr.cast(), detach: this);
    ResourceScope.track(this);
  }

  /// Allocates a new empty FLINT rational polynomial.
  static ffi.Pointer<fl.fmpq_poly_struct> _alloc() {
    final p = calloc<fl.fmpq_poly_struct>();
    fl.fmpq_poly_init(p);
    return p;
  }

  @override
  bool get isDisposed => _disposed;

  /// Explicitly releases the underlying native C++/C memory immediately.
  @override
  void dispose() {
    if (_disposed) return;
    _fmpqPolyFinalizer.detach(this);
    fl.fmpq_poly_clear(_ptr);
    calloc.free(_ptr);
    _disposed = true;
    ResourceScope.untrack(this);
  }

  @override
  FlintRationalPoly detachFromScope() {
    ResourceScope.untrack(this);
    return this;
  }

  @override
  FlintRationalPoly detachToParentScope() {
    ResourceScope.promoteToParent(this);
    return this;
  }

  void _checkDisposed() {
    if (_disposed) {
      throw StateError('This FlintRationalPoly has already been disposed.');
    }
  }

  ffi.Pointer<fl.fmpq_poly_struct> get pointer {
    _checkDisposed();
    return _ptr;
  }

  /// Creates a polynomial from integer coefficients in ascending order of degree:
  /// `coeffs[0] + coeffs[1]*x + ... + coeffs[d]*x^d`.
  factory FlintRationalPoly.fromIntCoefficients(List<int> coeffs) {
    final ptr = _alloc();
    for (var i = 0; i < coeffs.length; i++) {
      fl.fmpq_poly_set_coeff_si(ptr, i, coeffs[i]);
    }
    return FlintRationalPoly._(ptr);
  }

  /// Creates a polynomial from rational coefficients `({numerator, denominator})`
  /// in ascending order of degree.
  factory FlintRationalPoly.fromRationalCoefficients(
    List<({int numerator, int denominator})> coeffs,
  ) {
    final ptr = _alloc();
    final q = calloc<fl.fmpq>();
    try {
      for (var i = 0; i < coeffs.length; i++) {
        final c = coeffs[i];
        if (c.denominator == 0) {
          throw ArgumentError('Polynomial coefficient denominator cannot be 0');
        }
        fl.fmpq_set_si(q, c.numerator, c.denominator);
        fl.fmpq_poly_set_coeff_fmpq(ptr, i, q);
      }
    } finally {
      calloc.free(q);
    }
    return FlintRationalPoly._(ptr);
  }

  /// The zero polynomial `0`.
  factory FlintRationalPoly.zero() =>
      FlintRationalPoly.fromIntCoefficients([0]);

  /// The constant polynomial `1`.
  factory FlintRationalPoly.one() => FlintRationalPoly.fromIntCoefficients([1]);

  /// The monomial polynomial `a * x^d`.
  factory FlintRationalPoly.monomial(int degree, {int coefficient = 1}) {
    if (degree < 0) {
      throw ArgumentError('Monomial degree must be non-negative.');
    }
    final ptr = _alloc();
    fl.fmpq_poly_set_coeff_si(ptr, degree, coefficient);
    return FlintRationalPoly._(ptr);
  }

  /// Returns the degree of this polynomial (`-1` for the zero polynomial).
  int get degree => pointer.ref.length - 1;

  /// Returns the number of terms (degree + 1, or 0 if zero polynomial).
  int get length => pointer.ref.length;

  /// Returns the numeric value of the coefficient at [degreeIndex] as a double.
  double getCoefficientAsDouble(int degreeIndex) {
    if (degreeIndex < 0 || degreeIndex >= length) return 0.0;
    final q = calloc<fl.fmpq>();
    final numPtr = calloc<ffi.Int64>().cast<fl.fmpz>();
    final denPtr = calloc<ffi.Int64>().cast<fl.fmpz>();
    try {
      fl.fmpq_poly_get_coeff_fmpq(q, pointer, degreeIndex);
      fl.fmpq_numerator(numPtr, q);
      fl.fmpq_denominator(denPtr, q);
      final n = fl.fmpz_get_si(numPtr);
      final d = fl.fmpz_get_si(denPtr);
      return d == 0 ? 0.0 : n / d;
    } finally {
      calloc.free(q);
      calloc.free(numPtr);
      calloc.free(denPtr);
    }
  }

  /// Exact polynomial addition.
  FlintRationalPoly operator +(FlintRationalPoly other) {
    final res = _alloc();
    fl.fmpq_poly_add(res, pointer, other.pointer);
    return FlintRationalPoly._(res);
  }

  /// Exact polynomial subtraction.
  FlintRationalPoly operator -(FlintRationalPoly other) {
    final res = _alloc();
    fl.fmpq_poly_sub(res, pointer, other.pointer);
    return FlintRationalPoly._(res);
  }

  /// Exact polynomial multiplication.
  FlintRationalPoly operator *(FlintRationalPoly other) {
    final res = _alloc();
    fl.fmpq_poly_mul(res, pointer, other.pointer);
    return FlintRationalPoly._(res);
  }

  /// Exact polynomial quotient (`this // other`).
  FlintRationalPoly operator /(FlintRationalPoly other) {
    if (other.length == 0) {
      throw StateError('Polynomial division by zero.');
    }
    final q = _alloc();
    final r = _alloc();
    fl.fmpq_poly_divrem(q, r, pointer, other.pointer);
    fl.fmpq_poly_clear(r);
    calloc.free(r);
    return FlintRationalPoly._(q);
  }

  /// Exact polynomial remainder (`this % other`).
  FlintRationalPoly operator %(FlintRationalPoly other) {
    if (other.length == 0) {
      throw StateError('Polynomial division by zero.');
    }
    final q = _alloc();
    final r = _alloc();
    fl.fmpq_poly_divrem(q, r, pointer, other.pointer);
    fl.fmpq_poly_clear(q);
    calloc.free(q);
    return FlintRationalPoly._(r);
  }

  /// Returns both quotient and remainder `(quotient, remainder)`.
  ({FlintRationalPoly quotient, FlintRationalPoly remainder}) divmod(
    FlintRationalPoly other,
  ) {
    if (other.length == 0) {
      throw StateError('Polynomial division by zero.');
    }
    final q = _alloc();
    final r = _alloc();
    fl.fmpq_poly_divrem(q, r, pointer, other.pointer);
    return (
      quotient: FlintRationalPoly._(q),
      remainder: FlintRationalPoly._(r),
    );
  }

  /// Computes the exact polynomial derivative `dP(x)/dx`.
  FlintRationalPoly derivative() {
    final res = _alloc();
    fl.fmpq_poly_derivative(res, pointer);
    return FlintRationalPoly._(res);
  }

  /// Computes the exact polynomial integral `\int P(x) dx` with zero constant.
  FlintRationalPoly integral() {
    final res = _alloc();
    fl.fmpq_poly_integral(res, pointer);
    return FlintRationalPoly._(res);
  }

  /// Computes the exact monic Greatest Common Divisor (GCD) over Q[x].
  FlintRationalPoly gcd(FlintRationalPoly other) {
    final res = _alloc();
    fl.fmpq_poly_gcd(res, pointer, other.pointer);
    return FlintRationalPoly._(res);
  }

  /// Exact polynomial exponentiation `this^exponent` over Q[x].
  FlintRationalPoly pow(int exponent) {
    if (exponent < 0) {
      throw ArgumentError(
        'Polynomial exponent must be non-negative, got $exponent',
      );
    }
    final res = _alloc();
    fl.fmpq_poly_pow(res, pointer, exponent);
    return FlintRationalPoly._(res);
  }

  /// Shorthand operator for exact polynomial exponentiation `this^exponent`.
  FlintRationalPoly operator ^(int exponent) => pow(exponent);

  /// Exact polynomial composition `P(Q(x))` over Q[x].
  FlintRationalPoly compose(FlintRationalPoly other) {
    final res = _alloc();
    fl.fmpq_poly_compose(res, pointer, other.pointer);
    return FlintRationalPoly._(res);
  }

  /// Evaluates this polynomial numerically at a real number [x].
  double evaluate(double x) {
    final len = length;
    if (len == 0) return 0.0;
    var res = 0.0;
    for (var i = len - 1; i >= 0; i--) {
      res = res * x + getCoefficientAsDouble(i);
    }
    return res;
  }

  /// Evaluates this polynomial exactly at a rational number `num / den`,
  /// returning the exact rational result `(numerator, denominator)`.
  ({BigInt numerator, BigInt denominator}) evaluateRational(
    int num, [
    int den = 1,
  ]) {
    if (den == 0) throw ArgumentError('Denominator cannot be zero.');
    final qVal = calloc<fl.fmpq>();
    final qRes = calloc<fl.fmpq>();
    final nPtr = calloc<ffi.Int64>().cast<fl.fmpz>();
    final dPtr = calloc<ffi.Int64>().cast<fl.fmpz>();
    try {
      fl.fmpq_set_si(qVal, num, den);
      fl.fmpq_poly_evaluate_fmpq(qRes, pointer, qVal);
      fl.fmpq_numerator(nPtr, qRes);
      fl.fmpq_denominator(dPtr, qRes);
      final nSi = fl.fmpz_get_si(nPtr);
      final dSi = fl.fmpz_get_si(dPtr);
      return (numerator: BigInt.from(nSi), denominator: BigInt.from(dSi));
    } finally {
      calloc.free(qVal);
      calloc.free(qRes);
      calloc.free(nPtr);
      calloc.free(dPtr);
    }
  }

  /// Converts this FLINT exact polynomial into a SymEngine [Expr] using
  /// the symbolic variable [x].
  Expr toExpr(Expr x) {
    var sum = Expr.zero;
    final q = calloc<fl.fmpq>();
    final numPtr = calloc<ffi.Int64>().cast<fl.fmpz>();
    final denPtr = calloc<ffi.Int64>().cast<fl.fmpz>();
    try {
      final len = length;
      for (var i = 0; i < len; i++) {
        fl.fmpq_poly_get_coeff_fmpq(q, pointer, i);
        fl.fmpq_numerator(numPtr, q);
        fl.fmpq_denominator(denPtr, q);
        final n = fl.fmpz_get_si(numPtr);
        final d = fl.fmpz_get_si(denPtr);
        if (n != 0) {
          final termCoeff = Expr.rational(n, d);
          if (i == 0) {
            sum = sum + termCoeff;
          } else if (i == 1) {
            sum = sum + (termCoeff * x);
          } else {
            sum = sum + (termCoeff * (x ^ i));
          }
        }
      }
      return sum;
    } finally {
      calloc.free(numPtr);
      calloc.free(denPtr);
      calloc.free(q);
    }
  }

  /// Performs exact polynomial factorization over Q[x] into monic/primitive
  /// irreducible factors and rational content:
  /// `P(x) = content * \prod_i factor_i^{exponent_i}`.
  ({
    ({int numerator, int denominator}) content,
    List<({FlintRationalPoly factor, int exponent})> factors,
  })
  factor() {
    var D = BigInt.one;
    final q = calloc<fl.fmpq>();
    final numPtr = calloc<ffi.Int64>().cast<fl.fmpz>();
    final denPtr = calloc<ffi.Int64>().cast<fl.fmpz>();
    try {
      final len = length;
      for (var i = 0; i < len; i++) {
        fl.fmpq_poly_get_coeff_fmpq(q, pointer, i);
        fl.fmpq_denominator(denPtr, q);
        final dVal = BigInt.from(fl.fmpz_get_si(denPtr));
        if (dVal != BigInt.zero) {
          D = (D * dVal) ~/ D.gcd(dVal);
        }
      }

      final zPoly = calloc<fl.fmpz_poly_struct>();
      fl.fmpz_poly_init(zPoly);
      try {
        for (var i = 0; i < len; i++) {
          fl.fmpq_poly_get_coeff_fmpq(q, pointer, i);
          fl.fmpq_numerator(numPtr, q);
          fl.fmpq_denominator(denPtr, q);
          final nVal = BigInt.from(fl.fmpz_get_si(numPtr));
          final dVal = BigInt.from(fl.fmpz_get_si(denPtr));
          final scaledCoeff = (nVal * D) ~/ dVal;
          fl.fmpz_poly_set_coeff_si(zPoly, i, scaledCoeff.toInt());
        }

        final fac = calloc<fl.fmpz_poly_factor_struct>();
        fl.fmpz_poly_factor_init(fac);
        try {
          fl.fmpz_poly_factor(fac, zPoly);

          fl.fmpz_poly_factor_get_fmpz(numPtr, fac);
          final intContent = fl.fmpz_get_si(numPtr);

          final cGcd = BigInt.from(intContent).abs().gcd(D);
          final contentNumerator = (intContent ~/ cGcd.toInt());
          final contentDenominator = (D ~/ cGcd).toInt();

          final numFactors = fac.ref.num;
          final factorList = <({FlintRationalPoly factor, int exponent})>[];

          final factorZPoly = calloc<fl.fmpz_poly_struct>();
          fl.fmpz_poly_init(factorZPoly);
          try {
            for (var i = 0; i < numFactors; i++) {
              fl.fmpz_poly_factor_get_fmpz_poly(factorZPoly, fac, i);
              final exp = fac.ref.exp[i];

              final factorQPoly = FlintRationalPoly._(
                FlintRationalPoly._alloc(),
              );
              final facLen = factorZPoly.ref.length;
              for (var j = 0; j < facLen; j++) {
                final coeffInt = fl.fmpz_poly_get_coeff_si(factorZPoly, j);
                fl.fmpq_set_si(q, coeffInt, 1);
                fl.fmpq_poly_set_coeff_fmpq(factorQPoly.pointer, j, q);
              }
              factorList.add((factor: factorQPoly, exponent: exp));
            }
          } finally {
            fl.fmpz_poly_clear(factorZPoly);
            calloc.free(factorZPoly);
          }

          return (
            content: (
              numerator: contentNumerator,
              denominator: contentDenominator,
            ),
            factors: factorList,
          );
        } finally {
          fl.fmpz_poly_factor_clear(fac);
          calloc.free(fac);
        }
      } finally {
        fl.fmpz_poly_clear(zPoly);
        calloc.free(zPoly);
      }
    } finally {
      calloc.free(numPtr);
      calloc.free(denPtr);
      calloc.free(q);
    }
  }

  @override
  String toString() {
    _checkDisposed();
    final cStr = fl.fmpq_poly_get_str(pointer);
    if (cStr == ffi.nullptr) return '0';
    return cStr.cast<Utf8>().toDartString();
  }
}
