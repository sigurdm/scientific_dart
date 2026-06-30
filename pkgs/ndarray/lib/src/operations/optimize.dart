// ignore_for_file: non_constant_identifier_names
import 'dart:ffi' as ffi;
import 'package:openblas/openblas.dart';
import '../ndarray.dart';
import '../scratch_arena.dart';

/// Method selection for 1D scalar root finding ([root_scalar]).
enum RootMethod {
  /// Brent's algorithm combining bisection, secant, and inverse quadratic interpolation.
  brentq,

  /// Newton-Raphson method (requires derivative [fprime]).
  newton,

  /// Secant method (derivative-free secant iteration).
  secant,
}

/// Method selection for multivariate scalar function minimization ([minimize]).
enum MinimizeMethod {
  /// Nelder-Mead simplex algorithm (derivative-free).
  nelderMead,

  /// Limited-memory Broyden-Fletcher-Goldfarb-Shanno quasi-Newton algorithm.
  lbfgs,
}

/// Type alias for 1D root finding results represented as a named record.
///
/// Contains the calculated root location [root], convergence flag [converged],
/// iteration count [iterations], function evaluation count [functionCalls],
/// and diagnostic message [message].
typedef RootScalarResult = ({
  double root,
  bool converged,
  int iterations,
  int functionCalls,
  String message,
});

/// Type alias for multivariate scalar optimization results represented as a named record.
///
/// Contains the optimal vector [x], final objective value [fun], success status [success],
/// iteration count [nit], function evaluation count [nfev], diagnostic message [message],
/// and optional final gradient vector [jac].
///
/// ### Diagnostic Message Format ([message])
/// The [message] field provides detailed human-readable feedback on algorithm termination states:
/// - `'Optimization terminated successfully.'`: Standard convergence when parameters meet `xatol`/`fatol` tolerances.
/// - `'Optimization terminated successfully (gradient norm <= gtol).'`: Gradient norm reached tolerance in L-BFGS.
/// - `'Exact root found at bracket endpoint a.'` / `'Exact root found at bracket endpoint b.'`: Exact zero found at bracket endpoints.
/// - `'Converged to root within requested tolerance.'`: Scalar root iteration satisfied absolute/relative tolerances.
/// - `'Maximum iterations or evaluations reached'`: Iteration limit reached prior to convergence.
/// - `'Line search failed to find sufficient decrease'`: Line search step length could not satisfy the Armijo condition.
/// - `'Derivative evaluated to zero.'`: Newton iteration step failed due to zero derivative.
/// - `'Zero denominator encountered in secant step.'`: Secant step failed due to flat function value.
typedef OptimizeResult = ({
  NDArray<Float64> x,
  double fun,
  bool success,
  int nit,
  int nfev,
  String message,
  NDArray<Float64>? jac,
});

/// Finds a root of a scalar function within a bracketed interval $[a, b]$ using Brent's method.
///
/// Brent's method combines root bracketing, inverse quadratic interpolation, the secant method,
/// and bisection to achieve superlinear convergence while guaranteeing linear convergence in worst cases.
///
/// It is an error if $f(a)$ and $f(b)$ have the same sign ($f(a) \cdot f(b) > 0$).
/// It is an error if [maxiter] is less than or equal to zero.
/// It is an error if [xtol] or [rtol] is negative.
///
/// ### References & Further Reading
/// - [NumPy / SciPy brentq Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html)
/// - [Wikipedia: Brent's method](https://en.wikipedia.org/wiki/Brent%27s_method)
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
      'f(a) and f(b) must have different signs. Got f(a)=$fa, f(b)=$fb.',
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
    message: 'Failed to converge within maximum iterations ($maxiter).',
  );
}

/// Finds a root of a scalar function using Newton-Raphson or Secant iteration.
///
/// If derivative function [fprime] is provided, Newton-Raphson iteration is performed:
/// $$x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}$$
/// If [fprime] is `null`, secant iteration is used starting from initial guess [x0].
///
/// It is an error if [maxiter] is less than or equal to zero.
/// It is an error if [tol] is less than or equal to zero.
///
/// ### References & Further Reading
/// - [SciPy newton Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html)
/// - [Wikipedia: Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method)
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
      message: 'Failed to converge within maximum iterations ($maxiter).',
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
      message: 'Failed to converge within maximum iterations ($maxiter).',
    );
  }
}

/// Unified entry point for 1D scalar root finding algorithms.
///
/// Dispatches to supported root-finding methods defined in [RootMethod]:
/// - [RootMethod.brentq]: Requires bracket endpoints [bracketA] and [bracketB].
/// - [RootMethod.newton]: Requires initial guess [x0] and optional derivative [fprime].
/// - [RootMethod.secant]: Requires initial guess [x0].
///
/// It is an error if [method] is [RootMethod.brentq] and [bracketA] or [bracketB] is omitted.
/// It is an error if [method] is [RootMethod.newton] or [RootMethod.secant] and [x0] is omitted.
///
/// ### References & Further Reading
/// - [SciPy root_scalar Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html)
///
/// {@example /example/optimize_example.dart}
RootScalarResult root_scalar(
  double Function(double) f, {
  RootMethod method = RootMethod.brentq,
  double? bracketA,
  double? bracketB,
  double? x0,
  double Function(double)? fprime,
  double xtol = 2e-12,
  double rtol = 8.881784197001252e-16,
  double tol = 1.48e-8,
  int maxiter = 100,
}) {
  switch (method) {
    case RootMethod.brentq:
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
    case RootMethod.newton:
    case RootMethod.secant:
      if (x0 == null) {
        throw ArgumentError('$method requires initial guess x0.');
      }
      return newton(f, x0, fprime: fprime, tol: tol, maxiter: maxiter);
  }
}

/// Minimizes a multivariate scalar objective function using the Nelder-Mead simplex algorithm.
///
/// All heavy vector updates and distance calculations are offloaded to C OpenBLAS intrinsics.
///
/// It is an error if [x0] is disposed.
/// It is an error if [x0] is not a 1-dimensional array.
/// It is an error if [xatol] or [fatol] is negative.
///
/// ### References & Further Reading
/// - [SciPy minimize(method='Nelder-Mead') Documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html)
/// - [Wikipedia: Nelder-Mead method](https://en.wikipedia.org/wiki/Nelder%E2%80%93Mead_method)
///
/// {@example /example/optimize_example.dart}
OptimizeResult nelder_mead(
  double Function(NDArray<Float64>) fun,
  NDArray<Float64> x0, {
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
  if (xatol < 0 || fatol < 0) {
    throw ArgumentError('Tolerances xatol and fatol must be non-negative.');
  }

  return NDArray.scope(() {
    final n = x0.shape[0];
    final limitIter = maxiter ?? n * 200;
    final limitFev = maxfev ?? n * 200;

    final alpha = 1.0;
    final gamma = adaptive ? 1.0 + 2.0 / n : 2.0;
    final beta = adaptive ? 0.75 - 1.0 / (2.0 * n) : 0.5;
    final sigma = adaptive ? 1.0 - 1.0 / n : 0.5;

    final arenaMarker = ScratchArena.marker;
    try {
      final doubleBytes = ffi.sizeOf<ffi.Double>();
      final pSim = List<ffi.Pointer<ffi.Double>>.generate(
        n + 1,
        (_) => ScratchArena.allocate<ffi.Double>(n * doubleBytes),
      );
      final pFSim = ScratchArena.allocate<ffi.Double>((n + 1) * doubleBytes);
      final pXBar = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pXR = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pXE = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pXC = ScratchArena.allocate<ffi.Double>(n * doubleBytes);

      final x0Ptr = x0.pointer.cast<ffi.Double>();
      cblas_dcopy(n, x0Ptr, 1, pSim[0], 1);

      for (int i = 0; i < n; i++) {
        cblas_dcopy(n, pSim[0], 1, pSim[i + 1], 1);
        final val = pSim[i + 1][i];
        pSim[i + 1][i] = val != 0.0 ? val * 1.05 : 0.00025;
      }

      int nfev = 0;
      double evalPoint(ffi.Pointer<ffi.Double> ptr) {
        final pArr = NDArray<Float64>.create([n], DType.float64);
        cblas_dcopy(n, ptr, 1, pArr.pointer.cast<ffi.Double>(), 1);
        final val = fun(pArr);
        nfev++;
        return val;
      }

      for (int i = 0; i <= n; i++) {
        pFSim[i] = evalPoint(pSim[i]);
      }

      int nit = 0;
      bool success = false;
      String msg = 'Maximum iterations or evaluations reached';

      while (nit < limitIter && nfev < limitFev) {
        nit++;

        final idx = List<int>.generate(n + 1, (i) => i);
        idx.sort((a, b) => pFSim[a].compareTo(pFSim[b]));

        final tempSim = List<ffi.Pointer<ffi.Double>>.generate(
          n + 1,
          (i) => pSim[idx[i]],
        );
        final tempFSim = List<double>.generate(n + 1, (i) => pFSim[idx[i]]);
        for (int i = 0; i <= n; i++) {
          pSim[i] = tempSim[i];
          pFSim[i] = tempFSim[i];
        }

        double maxDiffF = 0.0;
        for (int i = 1; i <= n; i++) {
          final diff = (pFSim[i] - pFSim[0]).abs();
          if (diff > maxDiffF) maxDiffF = diff;
        }

        double maxDiffX = 0.0;
        for (int i = 1; i <= n; i++) {
          cblas_dcopy(n, pSim[i], 1, pXR, 1);
          cblas_daxpy(n, -1.0, pSim[0], 1, pXR, 1);
          final norm = cblas_dnrm2(n, pXR, 1);
          if (norm > maxDiffX) maxDiffX = norm;
        }

        if (maxDiffF < fatol && maxDiffX < xatol) {
          success = true;
          msg = 'Optimization terminated successfully.';
          break;
        }

        final u8Ptr = pXBar.cast<ffi.Uint8>();
        for (int b = 0; b < n * doubleBytes; b++) {
          u8Ptr[b] = 0;
        }
        for (int i = 0; i < n; i++) {
          cblas_daxpy(n, 1.0, pSim[i], 1, pXBar, 1);
        }
        cblas_dscal(n, 1.0 / n, pXBar, 1);

        cblas_dcopy(n, pXBar, 1, pXR, 1);
        cblas_dscal(n, 1.0 + alpha, pXR, 1);
        cblas_daxpy(n, -alpha, pSim[n], 1, pXR, 1);
        final fxr = evalPoint(pXR);

        bool doshrink = false;
        if (fxr < pFSim[0]) {
          cblas_dcopy(n, pXBar, 1, pXE, 1);
          cblas_dscal(n, 1.0 - gamma, pXE, 1);
          cblas_daxpy(n, gamma, pXR, 1, pXE, 1);
          final fxe = evalPoint(pXE);
          if (fxe < fxr) {
            cblas_dcopy(n, pXE, 1, pSim[n], 1);
            pFSim[n] = fxe;
          } else {
            cblas_dcopy(n, pXR, 1, pSim[n], 1);
            pFSim[n] = fxr;
          }
        } else if (fxr < pFSim[n - 1]) {
          cblas_dcopy(n, pXR, 1, pSim[n], 1);
          pFSim[n] = fxr;
        } else {
          if (fxr < pFSim[n]) {
            cblas_dcopy(n, pXBar, 1, pXC, 1);
            cblas_dscal(n, 1.0 - beta, pXC, 1);
            cblas_daxpy(n, beta, pXR, 1, pXC, 1);
            final fxc = evalPoint(pXC);
            if (fxc <= fxr) {
              cblas_dcopy(n, pXC, 1, pSim[n], 1);
              pFSim[n] = fxc;
            } else {
              doshrink = true;
            }
          } else {
            cblas_dcopy(n, pXBar, 1, pXC, 1);
            cblas_dscal(n, 1.0 - beta, pXC, 1);
            cblas_daxpy(n, beta, pSim[n], 1, pXC, 1);
            final fxc = evalPoint(pXC);
            if (fxc < pFSim[n]) {
              cblas_dcopy(n, pXC, 1, pSim[n], 1);
              pFSim[n] = fxc;
            } else {
              doshrink = true;
            }
          }
        }

        if (doshrink) {
          for (int i = 1; i <= n; i++) {
            cblas_dscal(n, sigma, pSim[i], 1);
            cblas_daxpy(n, 1.0 - sigma, pSim[0], 1, pSim[i], 1);
            pFSim[i] = evalPoint(pSim[i]);
          }
        }
      }

      final resArr = NDArray<Float64>.create([n], DType.float64);
      cblas_dcopy(n, pSim[0], 1, resArr.pointer.cast<ffi.Double>(), 1);

      return (
        x: resArr.detachToParentScope(),
        fun: pFSim[0],
        success: success,
        nit: nit,
        nfev: nfev,
        message: msg,
        jac: null,
      );
    } finally {
      ScratchArena.reset(arenaMarker);
    }
  });
}

/// CamelCase alias for [nelder_mead].
OptimizeResult nelderMead(
  double Function(NDArray<Float64>) fun,
  NDArray<Float64> x0, {
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

/// Minimizes a multivariate scalar objective function using the L-BFGS quasi-Newton algorithm.
///
/// All heavy matrix-free two-loop recursion step calculations and vector updates are offloaded to C OpenBLAS intrinsics.
///
/// It is an error if [x0] is disposed.
/// It is an error if [x0] is not a 1-dimensional array.
/// It is an error if [m] is less than or equal to zero.
/// It is an error if [gtol] is less than or equal to zero or [maxiter] is less than or equal to zero.
///
/// ### References & Further Reading
/// - [SciPy minimize(method='L-BFGS-B') Documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)
/// - [Wikipedia: Limited-memory BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)
///
/// {@example /example/optimize_example.dart}
OptimizeResult lbfgs(
  double Function(NDArray<Float64>) fun,
  NDArray<Float64> x0, {
  NDArray<Float64> Function(NDArray<Float64>)? jac,
  (double, NDArray<Float64>) Function(NDArray<Float64>)? funAndGrad,
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
  if (gtol <= 0 || maxiter <= 0) {
    throw ArgumentError('gtol and maxiter must be positive.');
  }

  return NDArray.scope(() {
    final n = x0.shape[0];
    final doubleBytes = ffi.sizeOf<ffi.Double>();
    final arenaMarker = ScratchArena.marker;

    try {
      final pXCurr = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pGCurr = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pQ = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pR = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pP = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pXNext = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pGNext = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pS = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pY = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
      final pXTemp = ScratchArena.allocate<ffi.Double>(n * doubleBytes);

      int nfev = 0;

      (double, ffi.Pointer<ffi.Double>) evalFunAndGrad(
        ffi.Pointer<ffi.Double> pX,
        ffi.Pointer<ffi.Double> pGOut,
      ) {
        final xArr = NDArray<Float64>.create([n], DType.float64);
        cblas_dcopy(n, pX, 1, xArr.pointer.cast<ffi.Double>(), 1);

        if (funAndGrad != null) {
          final (fVal, gArr) = funAndGrad(xArr);
          nfev++;
          cblas_dcopy(n, gArr.pointer.cast<ffi.Double>(), 1, pGOut, 1);
          return (fVal, pGOut);
        } else if (jac != null) {
          final fVal = fun(xArr);
          nfev++;
          final gArr = jac(xArr);
          cblas_dcopy(n, gArr.pointer.cast<ffi.Double>(), 1, pGOut, 1);
          return (fVal, pGOut);
        } else {
          final fVal = fun(xArr);
          nfev++;
          final h = 1e-8;
          for (int i = 0; i < n; i++) {
            cblas_dcopy(n, pX, 1, pXTemp, 1);
            pXTemp[i] += h;
            final xPlus = NDArray<Float64>.create([n], DType.float64);
            cblas_dcopy(n, pXTemp, 1, xPlus.pointer.cast<ffi.Double>(), 1);
            final fPlus = fun(xPlus);

            pXTemp[i] = pX[i] - h;
            final xMinus = NDArray<Float64>.create([n], DType.float64);
            cblas_dcopy(n, pXTemp, 1, xMinus.pointer.cast<ffi.Double>(), 1);
            final fMinus = fun(xMinus);

            pGOut[i] = (fPlus - fMinus) / (2.0 * h);
            nfev += 2;
          }
          return (fVal, pGOut);
        }
      }

      cblas_dcopy(n, x0.pointer.cast<ffi.Double>(), 1, pXCurr, 1);
      var (fCurr, _) = evalFunAndGrad(pXCurr, pGCurr);

      final pSHist = <ffi.Pointer<ffi.Double>>[];
      final pYHist = <ffi.Pointer<ffi.Double>>[];
      final rhoHist = <double>[];

      int nit = 0;
      bool success = false;
      String msg = 'Maximum iterations reached';

      for (int iter = 0; iter < maxiter; iter++) {
        nit++;

        double gNorm = 0.0;
        for (int i = 0; i < n; i++) {
          if (pGCurr[i].abs() > gNorm) gNorm = pGCurr[i].abs();
        }

        if (gNorm <= gtol) {
          success = true;
          msg = 'Optimization terminated successfully (gradient norm <= gtol).';
          break;
        }

        cblas_dcopy(n, pGCurr, 1, pQ, 1);
        final k = pSHist.length;
        final alphaArr = List<double>.filled(k, 0.0);

        for (int i = k - 1; i >= 0; i--) {
          final sq = cblas_ddot(n, pSHist[i], 1, pQ, 1);
          alphaArr[i] = rhoHist[i] * sq;
          cblas_daxpy(n, -alphaArr[i], pYHist[i], 1, pQ, 1);
        }

        double gamma = 1.0;
        if (k > 0) {
          final sy = cblas_ddot(n, pSHist[k - 1], 1, pYHist[k - 1], 1);
          final yy = cblas_ddot(n, pYHist[k - 1], 1, pYHist[k - 1], 1);
          if (yy > 0) gamma = sy / yy;
        }

        cblas_dcopy(n, pQ, 1, pR, 1);
        cblas_dscal(n, gamma, pR, 1);

        for (int i = 0; i < k; i++) {
          final yr = cblas_ddot(n, pYHist[i], 1, pR, 1);
          final beta = rhoHist[i] * yr;
          cblas_daxpy(n, alphaArr[i] - beta, pSHist[i], 1, pR, 1);
        }

        cblas_dcopy(n, pR, 1, pP, 1);
        cblas_dscal(n, -1.0, pP, 1);

        double alphaStep = 1.0;
        double c1 = 1e-4;
        final dg = cblas_ddot(n, pGCurr, 1, pP, 1);

        double fNext = fCurr;
        bool lineSearchSuccess = false;

        for (int ls = 0; ls < 25; ls++) {
          cblas_dcopy(n, pXCurr, 1, pXNext, 1);
          cblas_daxpy(n, alphaStep, pP, 1, pXNext, 1);

          final (fTry, _) = evalFunAndGrad(pXNext, pGNext);

          if (fTry <= fCurr + c1 * alphaStep * dg) {
            fNext = fTry;
            lineSearchSuccess = true;
            break;
          }
          alphaStep *= 0.5;
        }

        if (!lineSearchSuccess) {
          msg = 'Line search failed to find sufficient decrease';
          break;
        }

        cblas_dcopy(n, pXNext, 1, pS, 1);
        cblas_daxpy(n, -1.0, pXCurr, 1, pS, 1);

        cblas_dcopy(n, pGNext, 1, pY, 1);
        cblas_daxpy(n, -1.0, pGCurr, 1, pY, 1);

        final ys = cblas_ddot(n, pY, 1, pS, 1);

        if (ys > 1e-10) {
          if (pSHist.length >= m) {
            pSHist.removeAt(0);
            pYHist.removeAt(0);
            rhoHist.removeAt(0);
          }
          final newS = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
          final newY = ScratchArena.allocate<ffi.Double>(n * doubleBytes);
          cblas_dcopy(n, pS, 1, newS, 1);
          cblas_dcopy(n, pY, 1, newY, 1);
          pSHist.add(newS);
          pYHist.add(newY);
          rhoHist.add(1.0 / ys);
        }

        cblas_dcopy(n, pXNext, 1, pXCurr, 1);
        cblas_dcopy(n, pGNext, 1, pGCurr, 1);
        fCurr = fNext;
      }

      final resX = NDArray<Float64>.create([n], DType.float64);
      final resJac = NDArray<Float64>.create([n], DType.float64);
      cblas_dcopy(n, pXCurr, 1, resX.pointer.cast<ffi.Double>(), 1);
      cblas_dcopy(n, pGCurr, 1, resJac.pointer.cast<ffi.Double>(), 1);

      return (
        x: resX.detachToParentScope(),
        fun: fCurr,
        success: success,
        nit: nit,
        nfev: nfev,
        message: msg,
        jac: resJac.detachToParentScope(),
      );
    } finally {
      ScratchArena.reset(arenaMarker);
    }
  });
}

/// Unified entry point for multivariate scalar function minimization.
///
/// Dispatches to optimization algorithms defined in [MinimizeMethod]:
/// - [MinimizeMethod.nelderMead]: Derivative-free simplex method via [nelder_mead].
/// - [MinimizeMethod.lbfgs]: Quasi-Newton gradient method via [lbfgs].
///
/// It is an error if [x0] is disposed or if [x0] is not a 1-dimensional vector.
///
/// ### References & Further Reading
/// - [SciPy minimize Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
///
/// {@example /example/optimize_example.dart}
OptimizeResult minimize(
  double Function(NDArray<Float64>) fun,
  NDArray<Float64> x0, {
  MinimizeMethod method = MinimizeMethod.nelderMead,
  NDArray<Float64> Function(NDArray<Float64>)? jac,
  (double, NDArray<Float64>) Function(NDArray<Float64>)? funAndGrad,
  double? tol,
  int? maxiter,
}) {
  switch (method) {
    case MinimizeMethod.nelderMead:
      return nelder_mead(
        fun,
        x0,
        fatol: tol ?? 1e-4,
        xatol: tol ?? 1e-4,
        maxiter: maxiter,
      );
    case MinimizeMethod.lbfgs:
      return lbfgs(
        fun,
        x0,
        jac: jac,
        funAndGrad: funAndGrad,
        gtol: tol ?? 1e-5,
        maxiter: maxiter ?? 15000,
      );
  }
}
