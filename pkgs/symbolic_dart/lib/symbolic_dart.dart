/// High-performance symbolic computer algebra system (CAS) for Dart,
/// powered by native SymEngine (C++) and FLINT (exact number theory & algebra).
///
/// Features:
/// - Expression trees ([Expr], [Symbol], [Integer], [Real], [Rational])
/// - Symbolic calculus ([diff], [expand], [subs])
/// - Vectorized numerical evaluation over `NDArray` via [lambdify]
/// - Exact univariate polynomial algebra and factorization over Q[x] via [FlintRationalPoly]
library;

export 'src/expr.dart';
export 'src/lambdify.dart';
export 'src/matrix.dart';
export 'src/ndarray_integration.dart';
export 'src/optimizer.dart';
export 'src/poly.dart';
