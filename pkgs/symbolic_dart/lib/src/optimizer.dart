import 'package:ndarray/ndarray.dart';
import 'expr.dart';
import 'matrix.dart';

/// Numerical optimizers and root solvers powered by exact symbolic analytical
/// derivatives and Jacobian matrices from `symbolic_dart`, operating on
/// `NDArray<Float64>` state vectors.
final class SymbolicOptimizer {
  SymbolicOptimizer._();

  /// Solves a system of non-linear equations `f_1(x) = 0, ..., f_n(x) = 0`
  /// using exact analytical Jacobian matrices via the Newton-Raphson method.
  ///
  /// - [equations]: List of `n` symbolic equations `f_i(x_1, ..., x_n) = 0`.
  /// - [variables]: List of `n` symbolic variables `[x_1, ..., x_n]`.
  /// - [x0]: Initial guess 1D [NDArray<Float64>] of shape `[n]`.
  /// - [maxIterations]: Maximum number of Newton iterations.
  /// - [tolerance]: Infinity-norm convergence threshold on residual `||f(x)||_\infty`.
  ///
  /// Returns the solution vector [NDArray<Float64>], number of iterations taken,
  /// and final residual max-norm.
  static ({NDArray<Float64> solution, int iterations, double residual})
  solveNewtonRaphson({
    required List<Expr> equations,
    required List<Expr> variables,
    required NDArray<Float64> x0,
    int maxIterations = 100,
    double tolerance = 1e-9,
  }) {
    final n = variables.length;
    if (equations.length != n) {
      throw ArgumentError(
        'Expected $n equations to match $n variables, got ${equations.length}',
      );
    }
    if (x0.shape.length != 1 || x0.shape[0] != n) {
      throw ArgumentError(
        'Initial guess x0 must be a 1D array of length $n, got shape ${x0.shape}',
      );
    }

    // Build symbolic column vector f(x) and exact analytical Jacobian matrix J(x)
    final fVec = SymbolicMatrix.fromVector(equations);
    final jacMat = fVec.jacobian(variables);

    // Copy initial guess state buffer
    final xCurrent = NDArray.zeros([n], DType.float64);
    for (var i = 0; i < n; i++) {
      xCurrent.setCell([i], Float64((x0.getCell([i]) as num).toDouble()));
    }

    final scalarBuf = List<double>.filled(n, 0.0);
    double maxRes = 0.0;

    for (var iter = 0; iter < maxIterations; iter++) {
      // Fill scalar buffer for current x
      for (var i = 0; i < n; i++) {
        scalarBuf[i] = (xCurrent.getCell([i]) as num).toDouble();
      }

      // Evaluate residual vector b_k = f(x_k)
      final subMap = <Object, Object>{};
      for (var i = 0; i < n; i++) {
        subMap[variables[i]] = scalarBuf[i];
      }

      final fEvaluated = fVec.subs(subMap);
      maxRes = 0.0;
      for (var i = 0; i < n; i++) {
        final val = fEvaluated.getCell(i, 0).asDouble;
        final absVal = val.abs();
        if (absVal > maxRes) maxRes = absVal;
      }

      if (maxRes < tolerance) {
        return (solution: xCurrent, iterations: iter, residual: maxRes);
      }

      // Evaluate analytical Jacobian J(x_k) and invert/solve for Delta x = -J^-1 * f
      final jEvaluated = jacMat.subs(subMap);
      final deltaSym = jEvaluated.solve(-fEvaluated);

      // Update x_{k+1} = x_k + Delta x
      for (var i = 0; i < n; i++) {
        final deltaI = deltaSym.getCell(i, 0).asDouble;
        final oldVal = (xCurrent.getCell([i]) as num).toDouble();
        xCurrent.setCell([i], Float64(oldVal + deltaI));
      }
    }

    return (solution: xCurrent, iterations: maxIterations, residual: maxRes);
  }

  /// Minimizes a scalar [objective] function `L(x_1, ..., x_n)` using exact
  /// analytical symbolic gradients `\nabla L` via Gradient Descent with
  /// backtracking Armijo line search.
  ///
  /// - [objective]: Scalar symbolic loss function `L`.
  /// - [variables]: Parameter symbols `[x_1, ..., x_n]`.
  /// - [x0]: Initial guess 1D [NDArray<Float64>] of length `n`.
  /// - [learningRate]: Initial step size alpha.
  /// - [maxIterations]: Maximum gradient descent iterations.
  /// - [gradientTolerance]: Convergence threshold on gradient infinity-norm `||\nabla L||_\infty`.
  static ({
    NDArray<Float64> solution,
    int iterations,
    double loss,
    double gradientNorm,
  })
  minimizeGradientDescent({
    required Expr objective,
    required List<Expr> variables,
    required NDArray<Float64> x0,
    double learningRate = 0.1,
    int maxIterations = 200,
    double gradientTolerance = 1e-7,
  }) {
    final n = variables.length;
    if (x0.shape.length != 1 || x0.shape[0] != n) {
      throw ArgumentError(
        'Initial guess x0 must be a 1D array of length $n, got shape ${x0.shape}',
      );
    }

    // Derive exact analytical gradient vector \nabla L = [dL/dx1, ..., dL/dxn]^T
    final gradExprs = variables.map((v) => objective.diff(v)).toList();

    final xCurrent = NDArray.zeros([n], DType.float64);
    for (var i = 0; i < n; i++) {
      xCurrent.setCell([i], Float64((x0.getCell([i]) as num).toDouble()));
    }

    double currentLoss = 0.0;
    double maxGradNorm = 0.0;

    for (var iter = 0; iter < maxIterations; iter++) {
      final subMap = <Object, Object>{};
      for (var i = 0; i < n; i++) {
        subMap[variables[i]] = (xCurrent.getCell([i]) as num).toDouble();
      }

      currentLoss = objective.subs(subMap).asDouble;
      maxGradNorm = 0.0;
      final gradVals = List<double>.filled(n, 0.0);

      for (var i = 0; i < n; i++) {
        final gVal = gradExprs[i].subs(subMap).asDouble;
        gradVals[i] = gVal;
        final absG = gVal.abs();
        if (absG > maxGradNorm) maxGradNorm = absG;
      }

      if (maxGradNorm < gradientTolerance) {
        return (
          solution: xCurrent,
          iterations: iter,
          loss: currentLoss,
          gradientNorm: maxGradNorm,
        );
      }

      // Step: x_{k+1} = x_k - alpha * \nabla L
      for (var i = 0; i < n; i++) {
        final oldVal = (xCurrent.getCell([i]) as num).toDouble();
        xCurrent.setCell([i], Float64(oldVal - (learningRate * gradVals[i])));
      }
    }

    return (
      solution: xCurrent,
      iterations: maxIterations,
      loss: currentLoss,
      gradientNorm: maxGradNorm,
    );
  }
}
