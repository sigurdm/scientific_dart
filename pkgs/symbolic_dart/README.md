# symbolic_dart

A high-performance symbolic computer algebra system (CAS) for Dart, built on top of native **SymEngine** (C++) and **FLINT** (exact number theory & computer algebra).

## Features

- **Symbolic Expressions (`Expr`)**:
  - Expression trees (`Symbol`, `Integer`, `Real`, `Rational`)
  - Arithmetic operators (`+`, `-`, `*`, `/`, `-`, `^`)
  - Elementary functions (`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `exp`, `log`, `sqrt`, `abs`)
  - Symbolic differentiation (`diff`), algebraic expansion (`expand`), and substitution (`subs`)
  - Formatting (`toString`, `toLatex`, `toCCode`)
- **Vectorized Array Evaluation (`SymbolicLambda`)**:
  - Compiles symbolic formulas via `.lambdify([x, y])` into functions that evaluate vectorized over multi-dimensional `NDArray<Float64>` arrays from `package:ndarray` with automatic shape broadcasting.
- **Exact Polynomial Ring over $\mathbb{Q}[x]$ (`FlintRationalPoly`)**:
  - Exact addition, subtraction, multiplication, and quotient/remainder division (`divmod`)
  - Exact derivative, integral, and polynomial GCD (`gcd`)
  - **Exact Polynomial Factorization (`factor`)** into monic/primitive irreducible factors and rational scalar content.

## Quickstart

```dart
import 'package:ndarray/ndarray.dart';
import 'package:symbolic_dart/symbolic_dart.dart';

void main() {
  final x = Symbol('x');
  final y = Symbol('y');

  // Symbolic calculus
  final f = sin(x ^ 2) + (Integer(2) * y);
  print('f(x, y) = $f');
  print('df/dx   = ${f.diff(x)}');

  // Vectorized evaluation over NDArray
  final wave = sin(x) * exp(Real(-0.1) * y);
  final lambda = wave.lambdify([x, y]);

  final xArr = NDArray.fromList([0.0, 1.57, 3.14], [3], DType.float64);
  final yArr = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
  final zArr = lambda.callArray([xArr, yArr]);
  print('Vectorized output shape: ${zArr.shape}');

  // Exact polynomial factorization over Q[x]
  final p1 = FlintRationalPoly.fromIntCoefficients([-4, 0, 1]); // x^2 - 4
  final p2 = FlintRationalPoly.fromIntCoefficients([3, 2]);     // 2x + 3
  final poly = p1 * p2 * p2;

  final fac = poly.factor();
  print('Content = ${fac.content.numerator}/${fac.content.denominator}');
  for (final item in fac.factors) {
    print('  Factor (${item.factor}) ^ ${item.exponent}');
  }
}
```
