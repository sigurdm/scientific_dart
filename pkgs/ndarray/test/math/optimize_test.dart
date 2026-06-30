import 'dart:math' as math;
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('1D Root Finder: brentq()', () {
    test('Finds root of polynomial x^2 - 4 on [0, 3]', () {
      final res = brentq((x) => x * x - 4.0, 0.0, 3.0);
      expect(res.converged, isTrue);
      expect(res.root, closeTo(2.0, 1e-9));
    });

    test('Finds root of transcendental cos(x) - x on [0, 1]', () {
      final res = brentq((x) => math.cos(x) - x, 0.0, 1.0);
      expect(res.converged, isTrue);
      expect(res.root, closeTo(0.7390851332, 1e-7));
    });

    test('Exact root at endpoint', () {
      final res = brentq((x) => x - 2.0, 2.0, 5.0);
      expect(res.converged, isTrue);
      expect(res.root, 2.0);
    });

    test('Throws ArgumentError if interval does not bracket a root', () {
      expect(() => brentq((x) => x * x + 1.0, 0.0, 3.0), throwsArgumentError);
    });
  });

  group('1D Root Finder: newton()', () {
    test('Newton-Raphson with analytical derivative for sqrt(2)', () {
      final res = newton((x) => x * x - 2.0, 1.0, fprime: (x) => 2.0 * x);
      expect(res.converged, isTrue);
      expect(res.root, closeTo(1.414213562373095, 1e-9));
    });

    test('Secant method without derivative for x^3 - x - 2', () {
      final res = newton((x) => x * x * x - x - 2.0, 1.0);
      expect(res.converged, isTrue);
      expect(res.root, closeTo(1.5213797068, 1e-6));
    });
  });

  group('1D Root Finder: root_scalar()', () {
    test('Unified interface dispatches to brentq', () {
      final res = root_scalar(
        (x) => x * x - 9.0,
        method: 'brentq',
        bracketA: 0.0,
        bracketB: 5.0,
      );
      expect(res.converged, isTrue);
      expect(res.root, closeTo(3.0, 1e-9));
    });

    test('Unified interface dispatches to newton', () {
      final res = root_scalar(
        (x) => x * x - 9.0,
        method: 'newton',
        x0: 1.0,
        fprime: (x) => 2.0 * x,
      );
      expect(res.converged, isTrue);
      expect(res.root, closeTo(3.0, 1e-9));
    });

    test('Unified interface dispatches to secant', () {
      double f(double x) => x * x * x - x - 2.0;
      final result = root_scalar(f, method: 'secant', x0: 1.5);
      expect(result.converged, isTrue);
      expect(result.root, closeTo(1.5213797068, 1e-6));
    });
    test('Throws ArgumentError for unknown method', () {
      expect(
        () => root_scalar((x) => x, method: 'unknown'),
        throwsArgumentError,
      );
    });
  });

  group('Multivariate Minimization: nelder_mead()', () {
    test('Minimizes quadratic 2D bowl f(x,y) = (x-3)^2 + (y+2)^2', () {
      NDArray.scope(() {
        final x0 = NDArray<double>.fromList([0.0, 0.0], [2], DType.float64);
        final res = nelder_mead((x) {
          final px = (x.getCell([0]) as num).toDouble();
          final py = (x.getCell([1]) as num).toDouble();
          return (px - 3.0) * (px - 3.0) + (py + 2.0) * (py + 2.0);
        }, x0);

        expect(res.success, isTrue);
        expect((res.x.getCell([0]) as num).toDouble(), closeTo(3.0, 1e-3));
        expect((res.x.getCell([1]) as num).toDouble(), closeTo(-2.0, 1e-3));
        expect(res.fun, closeTo(0.0, 1e-5));
      });
    });

    test('Minimizes Rosenbrock function using camelCase alias nelderMead', () {
      double rosenbrock(NDArray<Float64> x) {
        final x0 = x.getCell([0]).toDouble();
        final x1 = x.getCell([1]).toDouble();
        return 100.0 * math.pow(x1 - x0 * x0, 2).toDouble() +
            math.pow(1.0 - x0, 2).toDouble();
      }

      final x0 = NDArray<Float64>.fromList([-1.2, 1.0], [2], DType.float64);
      final result = nelderMead(rosenbrock, x0, maxiter: 2000);

      expect(result.success, isTrue);
      expect(result.x.getCell([0]).toDouble(), closeTo(1.0, 1e-2));
      expect(result.x.getCell([1]).toDouble(), closeTo(1.0, 1e-2));
    });
    test('Minimizes Rosenbrock function', () {
      NDArray.scope(() {
        final x0 = NDArray<double>.fromList([-1.2, 1.0], [2], DType.float64);
        final res = nelder_mead(
          (x) {
            final px = (x.getCell([0]) as num).toDouble();
            final py = (x.getCell([1]) as num).toDouble();
            final term1 = 1.0 - px;
            final term2 = py - px * px;
            return term1 * term1 + 100.0 * term2 * term2;
          },
          x0,
          maxiter: 2000,
        );

        expect(res.success, isTrue);
        expect((res.x.getCell([0]) as num).toDouble(), closeTo(1.0, 1e-2));
        expect((res.x.getCell([1]) as num).toDouble(), closeTo(1.0, 1e-2));
      });
    });

    test('Handles int32 input with promotion', () {
      NDArray.scope(() {
        final x0 = NDArray<int>.fromList([0, 0], [2], DType.int32);
        final res = nelder_mead((x) {
          final px = (x.getCell([0]) as num).toDouble();
          final py = (x.getCell([1]) as num).toDouble();
          return (px - 1.0) * (px - 1.0) + (py - 1.0) * (py - 1.0);
        }, x0);

        expect(res.success, isTrue);
        expect((res.x.getCell([0]) as num).toDouble(), closeTo(1.0, 1e-3));
      });
    });
  });

  group('Multivariate Minimization: lbfgs()', () {
    test('Minimizes quadratic bowl with analytical gradient tuple', () {
      NDArray.scope(() {
        final x0 = NDArray<double>.fromList([0.0, 0.0], [2], DType.float64);
        final res = lbfgs((x) {
          final px = (x.getCell([0]) as num).toDouble();
          final py = (x.getCell([1]) as num).toDouble();
          final fVal = (px - 3.0) * (px - 3.0) + (py + 2.0) * (py + 2.0);
          final g = NDArray<Float64>.fromList(
            [Float64(2.0 * (px - 3.0)), Float64(2.0 * (py + 2.0))],
            [2],
            DType.float64,
          );
          return (fVal, g);
        }, x0);

        expect(res.success, isTrue);
        expect((res.x.getCell([0]) as num).toDouble(), closeTo(3.0, 1e-4));
        expect((res.x.getCell([1]) as num).toDouble(), closeTo(-2.0, 1e-4));
        expect(res.fun, closeTo(0.0, 1e-7));
      });
    });

    test('Minimizes quadratic bowl with separate jac function', () {
      NDArray.scope(() {
        final x0 = NDArray<double>.fromList([0.0, 0.0], [2], DType.float64);
        final res = lbfgs(
          (x) {
            final px = (x.getCell([0]) as num).toDouble();
            final py = (x.getCell([1]) as num).toDouble();
            return (px - 3.0) * (px - 3.0) + (py + 2.0) * (py + 2.0);
          },
          x0,
          jac: (x) {
            final px = (x.getCell([0]) as num).toDouble();
            final py = (x.getCell([1]) as num).toDouble();
            return NDArray<Float64>.fromList(
              [Float64(2.0 * (px - 3.0)), Float64(2.0 * (py + 2.0))],
              [2],
              DType.float64,
            );
          },
        );

        expect(res.success, isTrue);
        expect((res.x.getCell([0]) as num).toDouble(), closeTo(3.0, 1e-4));
        expect((res.x.getCell([1]) as num).toDouble(), closeTo(-2.0, 1e-4));
      });
    });

    test(
      'Minimizes quadratic bowl using numerical finite-differences gradient',
      () {
        NDArray.scope(() {
          final x0 = NDArray<double>.fromList([5.0, 5.0], [2], DType.float64);
          final res = lbfgs((x) {
            final px = (x.getCell([0]) as num).toDouble();
            final py = (x.getCell([1]) as num).toDouble();
            return (px - 1.0) * (px - 1.0) + (py - 2.0) * (py - 2.0);
          }, x0);

          expect(res.success, isTrue);
          expect(res.fun, closeTo(0.0, 1e-6));
          expect((res.x.getCell([0]) as num).toDouble(), closeTo(1.0, 1e-4));
          expect((res.x.getCell([1]) as num).toDouble(), closeTo(2.0, 1e-4));
        });
      },
    );
  });

  group('Unified Minimization: minimize()', () {
    test('Dispatches to nelder-mead', () {
      NDArray.scope(() {
        final x0 = NDArray<double>.fromList([0.0, 0.0], [2], DType.float64);
        final res = minimize(
          (NDArray<Float64> x) {
            final px = (x.getCell([0]) as num).toDouble();
            final py = (x.getCell([1]) as num).toDouble();
            return (px - 2.0) * (px - 2.0) + (py - 4.0) * (py - 4.0);
          },
          x0,
          method: 'nelder-mead',
        );

        expect(res.success, isTrue);
        expect((res.x.getCell([0]) as num).toDouble(), closeTo(2.0, 1e-3));
        expect((res.x.getCell([1]) as num).toDouble(), closeTo(4.0, 1e-3));
      });
    });

    test('Dispatches to l-bfgs with fun and jac', () {
      NDArray.scope(() {
        final x0 = NDArray<double>.fromList([0.0, 0.0], [2], DType.float64);
        final res = minimize(
          (NDArray<Float64> x) {
            final px = (x.getCell([0]) as num).toDouble();
            final py = (x.getCell([1]) as num).toDouble();
            return (px - 2.0) * (px - 2.0) + (py - 4.0) * (py - 4.0);
          },
          x0,
          method: 'l-bfgs',
          jac: (NDArray<Float64> x) {
            final px = (x.getCell([0]) as num).toDouble();
            final py = (x.getCell([1]) as num).toDouble();
            return NDArray<Float64>.fromList(
              [Float64(2.0 * (px - 2.0)), Float64(2.0 * (py - 4.0))],
              [2],
              DType.float64,
            );
          },
        );

        expect(res.success, isTrue);
        expect((res.x.getCell([0]) as num).toDouble(), closeTo(2.0, 1e-4));
        expect((res.x.getCell([1]) as num).toDouble(), closeTo(4.0, 1e-4));
      });
    });
  });
}
