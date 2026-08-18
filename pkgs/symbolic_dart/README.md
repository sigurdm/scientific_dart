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
- **Symbolic Matrices & Analytical Jacobians (`SymbolicMatrix`)**:
  - Exact matrix addition, subtraction, multiplication (`A * B`), and transpose (`transpose`).
  - Exact symbolic determinant (`det`) and matrix inverse (`inv`).
  - Exact linear equation solving (`A.solve(b)`) via LU decomposition.
  - Analytical Jacobian matrix generation (`vecF.jacobian(variables)`) for vector systems $\vec{f}(\vec{x})$.
  - Bidirectional conversion to/from 2D numeric arrays: `SymbolicMatrix.fromNDArray(arr)` and `.toNDArray()`.
- **Seamless `NDArray` Integration (`NDArraySymbolicExtension` & `evaluateSymbolic`)**:
  - Univariate element-wise mapping: `arr.mapSymbolic(sin(x) + x^2, x)`
  - Named broadcasting evaluation: `evaluateSymbolic(f, inputs: {x: xArr, y: yArr})`
- **Auto-Diff Root Solvers & Optimizers (`SymbolicOptimizer`)**:
  - `SymbolicOptimizer.solveNewtonRaphson`: Solves non-linear equations $\vec{f}(\vec{x}) = \vec{0}$ over `NDArray<Float64>` state vectors by automatically deriving and inverting exact analytical Jacobian matrices $J(\vec{x})$.
  - `SymbolicOptimizer.minimizeGradientDescent`: Minimizes objective functions $L(\vec{x})$ automatically using exact symbolic gradients $\nabla L(\vec{x})$.
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

  // Exact equation & linear system solving
  final roots = Expr.solvePoly((x ^ 2) - Integer(9), x);
  print('Roots of x^2 - 9 = 0: $roots');

  final sol = Expr.solveLinearSystem([
    (Integer(2) * x) + y - Integer(5),
    x + (Integer(3) * y) - Integer(5),
  ], [x, y]);
  print('Linear system solution (x, y): $sol');

  // Code Generation
  print('C code: ${wave.toCCode()}');
  print('JS code: ${wave.toJSCode()}');
}
```

## Scoped Memory Management

`package:symbolic_dart` integrates with **`package:resource_scope`** (`ScopedResource`) and `package:ndarray`'s zone-based scopes (`NDArray.scope`).

All symbolic AST nodes (`Expr`), matrices (`SymbolicMatrix`), exact polynomials (`FlintRationalPoly`), and numerical buffers (`NDArray`) created inside a scope block are automatically and deterministically disposed (`.dispose()`) when the block exits:

```dart
import 'package:ndarray/ndarray.dart';
import 'package:symbolic_dart/symbolic_dart.dart';

void main() {
  // Free all intermediate C++ SymEngine AST nodes and C buffers immediately at scope exit
  final result = NDArray.returning(() {
    final x = Symbol('x');
    final mat = SymbolicMatrix.fromList([
      [x ^ 2, sin(x)],
      [cos(x), Integer(5)],
    ]);
    final jacobian = mat.jacobian([x]);
    return jacobian.subs({x: 2.0}).toNDArray(); // Promoted out of scope
  });

  print(result);
  result.dispose();
}
```

