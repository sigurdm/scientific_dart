// ignore_for_file: non_constant_identifier_names
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import '../ndarray.dart';

/// Type alias for 1D root finding results represented as a named record.
///
/// Contains the calculated root location, convergence flag, iteration count,
/// function evaluation count, and detailed diagnostic message.
typedef RootScalarResult = ({
  double root,
  bool converged,
  int iterations,
  int functionCalls,
  String message,
});

/// Type alias for multivariate scalar optimization results represented as a named record.
///
/// Contains the optimal array [x], final objective value [fun], success status,
/// iteration count [nit], function evaluation count [nfev], diagnostic message,
/// and optional final gradient vector [jac].
typedef OptimizeResult = ({
  NDArray<Float64> x,
  double fun,
  bool success,
  int nit,
  int nfev,
  String message,
  NDArray<Float64>? jac,
});

/// Finds a root of a scalar function within a bracketed interval 0$ using Brent's method.
///
/// Brent's method combines root bracketing, inverse quadratic interpolation, the secant method,
/// and bisection to achieve superlinear convergence while guaranteeing linear convergence in worst cases.
///
/// **Preconditions:**
/// - [f] must be continuous on 0$.
/// - (a)$ and (b)$ must have opposite signs ((a) \cdot f(b) \le 0$).
/// - [maxiter] must be strictly positive.
/// - Tolerances [xtol] and [rtol] must be non-negative.
///
/// **Throws:**
/// - [ArgumentError] if (a)$ and (b)$ have the same sign.
/// - [ArgumentError] if [maxiter] <= 0 or tolerances are negative.
///
/// {@example /example/optimize_example.dart}
RootScalarResult brentq(
  double Function(double) f,
  double a,
  double b, {
  double xtol = 2e-12,
  double rtol = 8.881784197001252e-16,
  int maxiter = 100,
}) {
  if (maxiter <= 0) {
    throw ArgumentError('maxiter must be positive.');
  }
  if (xtol < 0 || rtol < 0) {
    throw ArgumentError('Tolerances xtol and rtol must be non-negative.');
  }

  var fa = f(a);
  var fb = f(b);
  var nfev = 2;

  if (fa == 0.0) {
    return (
      root: a,
      converged: true,
      iterations: 0,
      functionCalls: nfev,
      message: 'Exact root found at bracket endpoint a.',
    );
  }
  if (fb == 0.0) {
    return (
      root: b,
      converged: true,
      iterations: 0,
      functionCalls: nfev,
      message: 'Exact root found at bracket endpoint b.',
    );
  }

  if (fa * fb > 0) {
    throw ArgumentError(
      'f(a) and f(b) must have different signs. Got f()=, f()=.',
    );
  }

  var c = a;
  var fc = fa;
  var d = b - a;
  var e = d;

  for (var iter = 1; iter <= maxiter; iter++) {
    if (fb == 0.0) {
      return (
        root: b,
        converged: true,
        iterations: iter,
        functionCalls: nfev,
        message: 'Converged to root.',
      );
    }

    if ((fb > 0 && fc > 0) || (fb < 0 && fc < 0)) {
      c = a;
      fc = fa;
      d = b - a;
      e = d;
    }

    if (fc.abs() < fb.abs()) {
      a = b;
      b = c;
      c = a;
      fa = fb;
      fb = fc;
      fc = fa;
    }

    final tol1 = 2.0 * rtol * b.abs() + 0.5 * xtol;
    final xm = 0.5 * (c - b);

    if (xm.abs() <= tol1 || fb == 0.0) {
      return (
        root: b,
        converged: true,
        iterations: iter,
        functionCalls: nfev,
        message: 'Converged to root within requested tolerance.',
      );
    }

    if (e.abs() >= tol1 && fa.abs() > fb.abs()) {
      var s = fb / fa;
      var p = 0.0;
      var q = 0.0;
      if (a == c) {
        p = 2.0 * xm * s;
        q = 1.0 - s;
      } else {
        q = fa / fc;
        final r = fb / fc;
        p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0));
        q = (q - 1.0) * (r - 1.0) * (s - 1.0);
      }

      if (p > 0) {
        q = -q;
      } else {
        p = -p;
      }

      final min1 = 3.0 * xm * q - (tol1 * q).abs();
      final min2 = (e * q).abs();

      if (2.0 * p < (min1 < min2 ? min1 : min2)) {
        e = d;
        d = p / q;
      } else {
        d = xm;
        e = d;
      }
    } else {
      d = xm;
      e = d;
    }

    a = b;
    fa = fb;

    if (d.abs() > tol1) {
      b += d;
    } else {
      b += xm >= 0 ? tol1 : -tol1;
    }

    fb = f(b);
    nfev++;
  }

  return (
    root: b,
    converged: false,
    iterations: maxiter,
    functionCalls: nfev,
    message: 'Failed to converge within maximum iterations ().',
  );
}

/// Finds a root of a scalar function using Newton-Raphson or Secant method.
///
/// If derivative [fprime] is provided, Newton-Raphson iteration is performed.
/// Otherwise, the secant method is used starting from [x0].
///
/// **Preconditions:**
/// - [f] must be continuous.
/// - [maxiter] must be strictly positive.
/// - Tolerance [tol] must be strictly positive.
///
/// **Throws:**
/// - [ArgumentError] if [maxiter] <= 0 or [tol] <= 0.
///
/// {@example /example/optimize_example.dart}
RootScalarResult newton(
  double Function(double) f,
  double x0, {
  double Function(double)? fprime,
  double tol = 1.48e-8,
  int maxiter = 50,
}) {
  if (maxiter <= 0) {
    throw ArgumentError('maxiter must be positive.');
  }
  if (tol <= 0) {
    throw ArgumentError('Tolerance tol must be positive.');
  }

  var p0 = x0;
  var nfev = 0;

  if (fprime != null) {
    for (var iter = 1; iter <= maxiter; iter++) {
      final y = f(p0);
      nfev++;
      if (y == 0.0) {
        return (
          root: p0,
          converged: true,
          iterations: iter - 1,
          functionCalls: nfev,
          message: 'Exact root found.',
        );
      }
      final yprime = fprime(p0);
      if (yprime == 0.0) {
        return (
          root: p0,
          converged: false,
          iterations: iter,
          functionCalls: nfev,
          message: 'Derivative evaluated to zero.',
        );
      }
      final p = p0 - y / yprime;
      if ((p - p0).abs() <= tol) {
        return (
          root: p,
          converged: true,
          iterations: iter,
          functionCalls: nfev,
          message: 'Converged to root within requested tolerance.',
        );
      }
      p0 = p;
    }
    return (
      root: p0,
      converged: false,
      iterations: maxiter,
      functionCalls: nfev,
      message: 'Failed to converge within maximum iterations ().',
    );
  } else {
    var p1 = x0 != 0.0 ? x0 * 1.0001 : 1e-4;
    var q0 = f(p0);
    var q1 = f(p1);
    nfev += 2;

    for (var iter = 1; iter <= maxiter; iter++) {
      if (q1 == q0) {
        return (
          root: p1,
          converged: false,
          iterations: iter,
          functionCalls: nfev,
          message: 'Zero denominator encountered in secant step.',
        );
      }
      final p = p1 - q1 * (p1 - p0) / (q1 - q0);
      if ((p - p1).abs() <= tol) {
        return (
          root: p,
          converged: true,
          iterations: iter,
          functionCalls: nfev,
          message: 'Converged to root within requested tolerance.',
        );
      }
      p0 = p1;
      q0 = q1;
      p1 = p;
      q1 = f(p1);
      nfev++;
    }
    return (
      root: p1,
      converged: false,
      iterations: maxiter,
      functionCalls: nfev,
      message: 'Failed to converge within maximum iterations ().',
    );
  }
}

/// Unified entry point for 1D root finding algorithms.
///
/// Dispatches to supported methods: , , or .
///
/// **Preconditions:**
/// - [method] must be one of , , or .
/// - For , [bracketA] and [bracketB] must be provided.
/// - For  or , [x0] must be provided.
///
/// **Throws:**
/// - [ArgumentError] if [method] is unknown or required parameters for the chosen method are missing.
///
/// {@example /example/optimize_example.dart}
RootScalarResult root_scalar(
  double Function(double) f, {
  String method = 'brentq',
  double? bracketA,
  double? bracketB,
  double? x0,
  double Function(double)? fprime,
  double xtol = 2e-12,
  double rtol = 8.881784197001252e-16,
  double tol = 1.48e-8,
  int maxiter = 100,
}) {
  switch (method.toLowerCase()) {
    case 'brentq':
      if (bracketA == null || bracketB == null) {
        throw ArgumentError('brentq requires bracketA and bracketB.');
      }
      return brentq(
        f,
        bracketA,
        bracketB,
        xtol: xtol,
        rtol: rtol,
        maxiter: maxiter,
      );
    case 'newton':
    case 'secant':
      if (x0 == null) {
        throw ArgumentError(' requires initial guess x0.');
      }
      return newton(f, x0, fprime: fprime, tol: tol, maxiter: maxiter);
    default:
      throw ArgumentError('Unknown root_scalar method: .');
  }
}

/// Minimizes a multivariate scalar objective function using Nelder-Mead simplex algorithm.
///
/// **Preconditions:**
/// - [x0] must not be disposed.
/// - [x0] must be a 1D vector.
/// - Complex dtypes are not supported.
///
/// **Throws:**
/// - [StateError] if [x0] is disposed.
/// - [ArgumentError] if [x0] rank is not 1 or has complex dtype.
///
/// {@example /example/optimize_example.dart}
OptimizeResult nelder_mead(
  double Function(NDArray<Float64>) fun,
  NDArray<num> x0, {
  double xatol = 1e-4,
  double fatol = 1e-4,
  int? maxiter,
  int? maxfev,
  bool adaptive = false,
}) {
  if (x0.isDisposed) {
    throw StateError('Cannot execute nelder_mead on disposed x0 array.');
  }
  if (x0.shape.length != 1) {
    throw ArgumentError('x0 must be a 1D vector for nelder_mead.');
  }
  if (x0.dtype.isComplex) {
    throw ArgumentError('Complex dtypes are not supported for optimization.');
  }

  return NDArray.scope(() {
    final n = x0.shape[0];
    final NDArray<Float64> x0Double = x0 is NDArray<Float64>
        ? x0
        : NDArray<Float64>.fromList(
            List.generate(n, (j) => Float64(x0.getCell([j]).toDouble())),
            x0.shape,
            DType.float64,
          );

    final limitIter = maxiter ?? n * 200;
    final limitFev = maxfev ?? n * 200;

    final alpha = 1.0;
    final gamma = adaptive ? 1.0 + 2.0 / n : 2.0;
    final beta = adaptive ? 0.75 - 1.0 / (2.0 * n) : 0.5;
    final sigma = adaptive ? 1.0 - 1.0 / n : 0.5;

    final sim = List<Float64List>.generate(n + 1, (_) => Float64List(n));
    final fsim = Float64List(n + 1);

    final x0List = x0Double.pointer.cast<ffi.Double>().asTypedList(n);
    for (int j = 0; j < n; j++) {
      sim[0][j] = x0List[j];
    }

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        sim[i + 1][j] = sim[0][j];
      }
      if (sim[i + 1][i] != 0.0) {
        sim[i + 1][i] *= 1.05;
      } else {
        sim[i + 1][i] = 0.00025;
      }
    }

    int nfev = 0;
    double evalPoint(Float64List p) {
      final pArr = NDArray<Float64>.fromList(p, [n], DType.float64);
      final val = fun(pArr);
      nfev++;
      return val;
    }

    for (int i = 0; i <= n; i++) {
      fsim[i] = evalPoint(sim[i]);
    }

    int nit = 0;
    bool success = false;
    String msg = 'Maximum iterations or evaluations reached';

    while (nit < limitIter && nfev < limitFev) {
      nit++;

      final idx = List<int>.generate(n + 1, (i) => i);
      idx.sort((a, b) => fsim[a].compareTo(fsim[b]));

      final newSim = List<Float64List>.generate(n + 1, (i) => sim[idx[i]]);
      final newFsim = Float64List(n + 1);
      for (int i = 0; i <= n; i++) {
        newFsim[i] = fsim[idx[i]];
        sim[i] = newSim[i];
        fsim[i] = newFsim[i];
      }

      double maxDiffF = 0.0;
      for (int i = 1; i <= n; i++) {
        final diff = (fsim[i] - fsim[0]).abs();
        if (diff > maxDiffF) maxDiffF = diff;
      }
      double maxDiffX = 0.0;
      for (int i = 1; i <= n; i++) {
        for (int j = 0; j < n; j++) {
          final diff = (sim[i][j] - sim[0][j]).abs();
          if (diff > maxDiffX) maxDiffX = diff;
        }
      }

      if (maxDiffF < fatol && maxDiffX < xatol) {
        success = true;
        msg = 'Optimization terminated successfully.';
        break;
      }

      final xbar = Float64List(n);
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          xbar[j] += sim[i][j];
        }
      }
      for (int j = 0; j < n; j++) {
        xbar[j] /= n;
      }

      final xr = Float64List(n);
      for (int j = 0; j < n; j++) {
        xr[j] = xbar[j] + alpha * (xbar[j] - sim[n][j]);
      }
      final fxr = evalPoint(xr);

      bool doshrink = false;
      if (fxr < fsim[0]) {
        final xe = Float64List(n);
        for (int j = 0; j < n; j++) {
          xe[j] = xbar[j] + gamma * (xr[j] - xbar[j]);
        }
        final fxe = evalPoint(xe);
        if (fxe < fxr) {
          sim[n] = xe;
          fsim[n] = fxe;
        } else {
          sim[n] = xr;
          fsim[n] = fxr;
        }
      } else if (fxr < fsim[n - 1]) {
        sim[n] = xr;
        fsim[n] = fxr;
      } else {
        if (fxr < fsim[n]) {
          final xc = Float64List(n);
          for (int j = 0; j < n; j++) {
            xc[j] = xbar[j] + beta * (xr[j] - xbar[j]);
          }
          final fxc = evalPoint(xc);
          if (fxc <= fxr) {
            sim[n] = xc;
            fsim[n] = fxc;
          } else {
            doshrink = true;
          }
        } else {
          final xc = Float64List(n);
          for (int j = 0; j < n; j++) {
            xc[j] = xbar[j] - beta * (xbar[j] - sim[n][j]);
          }
          final fxc = evalPoint(xc);
          if (fxc < fsim[n]) {
            sim[n] = xc;
            fsim[n] = fxc;
          } else {
            doshrink = true;
          }
        }
      }

      if (doshrink) {
        for (int i = 1; i <= n; i++) {
          for (int j = 0; j < n; j++) {
            sim[i][j] = sim[0][j] + sigma * (sim[i][j] - sim[0][j]);
          }
          fsim[i] = evalPoint(sim[i]);
        }
      }
    }

    final resArr = NDArray<Float64>.fromList(sim[0], [n], DType.float64);
    return (
      x: resArr.detachToParentScope(),
      fun: fsim[0],
      success: success,
      nit: nit,
      nfev: nfev,
      message: msg,
      jac: null,
    );
  });
}

/// CamelCase alias for [nelder_mead].
OptimizeResult nelderMead(
  double Function(NDArray<Float64>) fun,
  NDArray<num> x0, {
  double xatol = 1e-4,
  double fatol = 1e-4,
  int? maxiter,
  int? maxfev,
  bool adaptive = false,
}) => nelder_mead(
  fun,
  x0,
  xatol: xatol,
  fatol: fatol,
  maxiter: maxiter,
  maxfev: maxfev,
  adaptive: adaptive,
);

/// Minimizes a multivariate scalar objective function using L-BFGS quasi-Newton algorithm.
///
/// **Preconditions:**
/// - [x0] must not be disposed.
/// - [x0] must be a 1D vector.
/// - [m] must be strictly positive.
///
/// **Throws:**
/// - [StateError] if [x0] is disposed.
/// - [ArgumentError] if [x0] rank is not 1 or [m] <= 0.
///
/// {@example /example/optimize_example.dart}
OptimizeResult lbfgs(
  Function fun,
  NDArray<num> x0, {
  Function? jac,
  int m = 10,
  double gtol = 1e-5,
  int maxiter = 15000,
}) {
  if (x0.isDisposed) {
    throw StateError('Cannot execute lbfgs on disposed x0 array.');
  }
  if (x0.shape.length != 1) {
    throw ArgumentError('x0 must be a 1D vector for lbfgs.');
  }
  if (m <= 0) {
    throw ArgumentError('m must be strictly positive.');
  }
  if (x0.dtype.isComplex) {
    throw ArgumentError('Complex dtypes are not supported for optimization.');
  }

  return NDArray.scope(() {
    final n = x0.shape[0];
    final NDArray<Float64> x0Double = x0 is NDArray<Float64>
        ? x0
        : NDArray<Float64>.fromList(
            List.generate(n, (j) => Float64(x0.getCell([j]).toDouble())),
            x0.shape,
            DType.float64,
          );

    int nfev = 0;

    (double, Float64List) evalFunAndGrad(Float64List xVec) {
      final xArr = NDArray<Float64>.fromList(xVec, [n], DType.float64);
      if (fun is (double, NDArray<Float64>) Function(NDArray<Float64>)) {
        final (fVal, gArr) = fun(xArr);
        nfev++;
        return (fVal, gArr.pointer.cast<ffi.Double>().asTypedList(n));
      } else if (fun is double Function(NDArray<Float64>)) {
        final fVal = fun(xArr);
        nfev++;
        if (jac is NDArray<Float64> Function(NDArray<Float64>)) {
          final gArr = jac(xArr);
          return (fVal, gArr.pointer.cast<ffi.Double>().asTypedList(n));
        } else {
          final gVec = Float64List(n);
          final h = 1e-8;
          for (int i = 0; i < n; i++) {
            final xTemp = Float64List.fromList(xVec);
            xTemp[i] += h;
            final fPlus = fun(
              NDArray<Float64>.fromList(xTemp, [n], DType.float64),
            );
            xTemp[i] = xVec[i] - h;
            final fMinus = fun(
              NDArray<Float64>.fromList(xTemp, [n], DType.float64),
            );
            gVec[i] = (fPlus - fMinus) / (2.0 * h);
            nfev += 2;
          }
          return (fVal, gVec);
        }
      } else {
        throw ArgumentError(
          'fun must be double Function(NDArray<Float64>) or (double, NDArray<Float64>) Function(NDArray<Float64>).',
        );
      }
    }

    var xCurr = Float64List.fromList(
      x0Double.pointer.cast<ffi.Double>().asTypedList(n),
    );
    var (fCurr, gCurr) = evalFunAndGrad(xCurr);

    final sHist = <Float64List>[];
    final yHist = <Float64List>[];
    final rhoHist = <double>[];

    int nit = 0;
    bool success = false;
    String msg = 'Maximum iterations reached';

    for (int iter = 0; iter < maxiter; iter++) {
      nit++;

      double gNorm = 0.0;
      for (int i = 0; i < n; i++) {
        if (gCurr[i].abs() > gNorm) gNorm = gCurr[i].abs();
      }

      if (gNorm <= gtol) {
        success = true;
        msg = 'Optimization terminated successfully (gradient norm <= gtol).';
        break;
      }

      final q = Float64List.fromList(gCurr);
      final k = sHist.length;
      final alphaArr = Float64List(k);

      for (int i = k - 1; i >= 0; i--) {
        double sq = 0.0;
        for (int j = 0; j < n; j++) {
          sq += sHist[i][j] * q[j];
        }
        alphaArr[i] = rhoHist[i] * sq;
        for (int j = 0; j < n; j++) {
          q[j] -= alphaArr[i] * yHist[i][j];
        }
      }

      double gamma = 1.0;
      if (k > 0) {
        double sy = 0.0, yy = 0.0;
        for (int j = 0; j < n; j++) {
          sy += sHist[k - 1][j] * yHist[k - 1][j];
          yy += yHist[k - 1][j] * yHist[k - 1][j];
        }
        if (yy > 0) gamma = sy / yy;
      }

      final r = Float64List(n);
      for (int j = 0; j < n; j++) {
        r[j] = gamma * q[j];
      }

      for (int i = 0; i < k; i++) {
        double yr = 0.0;
        for (int j = 0; j < n; j++) {
          yr += yHist[i][j] * r[j];
        }
        final beta = rhoHist[i] * yr;
        for (int j = 0; j < n; j++) {
          r[j] += sHist[i][j] * (alphaArr[i] - beta);
        }
      }

      final p = Float64List(n);
      for (int j = 0; j < n; j++) {
        p[j] = -r[j];
      }

      double alphaStep = 1.0;
      double c1 = 1e-4;
      double dg = 0.0;
      for (int j = 0; j < n; j++) {
        dg += gCurr[j] * p[j];
      }

      Float64List xNext = Float64List(n);
      double fNext = fCurr;
      Float64List gNext = Float64List(n);
      bool lineSearchSuccess = false;

      for (int ls = 0; ls < 25; ls++) {
        for (int j = 0; j < n; j++) {
          xNext[j] = xCurr[j] + alphaStep * p[j];
        }
        final (fTry, gTry) = evalFunAndGrad(xNext);

        if (fTry <= fCurr + c1 * alphaStep * dg) {
          fNext = fTry;
          gNext = gTry;
          lineSearchSuccess = true;
          break;
        }
        alphaStep *= 0.5;
      }

      if (!lineSearchSuccess) {
        msg = 'Line search failed to find sufficient decrease';
        break;
      }

      final s = Float64List(n);
      final y = Float64List(n);
      double ys = 0.0;
      for (int j = 0; j < n; j++) {
        s[j] = xNext[j] - xCurr[j];
        y[j] = gNext[j] - gCurr[j];
        ys += y[j] * s[j];
      }

      if (ys > 1e-10) {
        if (sHist.length >= m) {
          sHist.removeAt(0);
          yHist.removeAt(0);
          rhoHist.removeAt(0);
        }
        sHist.add(s);
        yHist.add(y);
        rhoHist.add(1.0 / ys);
      }

      xCurr = xNext;
      fCurr = fNext;
      gCurr = gNext;
    }

    final resX = NDArray<Float64>.fromList(xCurr, [n], DType.float64);
    final resJac = NDArray<Float64>.fromList(gCurr, [n], DType.float64);
    return (
      x: resX.detachToParentScope(),
      fun: fCurr,
      success: success,
      nit: nit,
      nfev: nfev,
      message: msg,
      jac: resJac.detachToParentScope(),
    );
  });
}

/// Unified interface for scalar and multivariate minimization.
///
/// Dispatches to supported methods: , , or .
///
/// **Preconditions:**
/// - [method] must be one of , , .
///
/// **Throws:**
/// - [ArgumentError] if [method] is unknown or parameters mismatch.
///
/// {@example /example/optimize_example.dart}
OptimizeResult minimize(
  Function fun,
  NDArray<num> x0, {
  String method = 'nelder-mead',
  Function? jac,
  double? tol,
  int? maxiter,
}) {
  switch (method.toLowerCase()) {
    case 'nelder-mead':
    case 'neldermead':
      if (fun is! double Function(NDArray<Float64>)) {
        throw ArgumentError(
          'For nelder-mead, fun must be double Function(NDArray<Float64>).',
        );
      }
      return nelder_mead(
        fun,
        x0,
        fatol: tol ?? 1e-4,
        xatol: tol ?? 1e-4,
        maxiter: maxiter,
      );
    case 'l-bfgs':
    case 'l-bfgs-b':
    case 'lbfgs':
      return lbfgs(
        fun,
        x0,
        jac: jac,
        gtol: tol ?? 1e-5,
        maxiter: maxiter ?? 15000,
      );
    default:
      throw ArgumentError('Unknown minimize method: .');
  }
}
