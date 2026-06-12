import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Quantitative Financial ufuncs Tests', () {
    group('Future Value (fv) Tests', () {
      test('Scalar-like 0D arrays', () {
        NDArray.scope(() {
          final rate = NDArray.fromList([0.05 / 12], [], DType.float64);
          final nper = NDArray.fromList([10.0 * 12], [], DType.float64);
          final pmt = NDArray.fromList([-100.0], [], DType.float64);
          final pvVal = NDArray.fromList([-100.0], [], DType.float64);

          final res = fv(rate, nper, pmt, pvVal);
          expect(res.shape, <int>[]);
          expect(res.dtype, DType.float64);
          expect(res.scalar, closeTo(15692.92889433575, 1e-5));
        });
      });

      test('Rate = 0 case', () {
        NDArray.scope(() {
          final rate = NDArray.fromList([0.0], [], DType.float64);
          final nper = NDArray.fromList([120.0], [], DType.float64);
          final pmt = NDArray.fromList([-100.0], [], DType.float64);
          final pvVal = NDArray.fromList([-100.0], [], DType.float64);

          final res = fv(rate, nper, pmt, pvVal);
          // fv = - (pv + pmt * nper) = - (-100 + -100 * 120) = - (-12100) = 12100
          expect(res.scalar, closeTo(12100.0, 1e-5));
        });
      });

      test('Broadcasting rate array', () {
        NDArray.scope(() {
          final rate = NDArray.fromList(
            [0.05 / 12, 0.06 / 12, 0.07 / 12],
            [3],
            DType.float64,
          );
          final nper = NDArray.fromList([10.0 * 12], [], DType.float64);
          final pmt = NDArray.fromList([-100.0], [], DType.float64);
          final pvVal = NDArray.fromList([-100.0], [], DType.float64);

          final res = fv(rate, nper, pmt, pvVal);
          expect(res.shape, [3]);
          expect(res.getCell([0]), closeTo(15692.92889434, 1e-5));
          expect(res.getCell([1]), closeTo(16569.87435405, 1e-5));
          expect(res.getCell([2]), closeTo(17509.44688102, 1e-5));
        });
      });

      test('when = "begin" (1) vs "end" (0)', () {
        NDArray.scope(() {
          final rate = NDArray.fromList([0.05 / 12], [], DType.float64);
          final nper = NDArray.fromList([10.0 * 12], [], DType.float64);
          final pmt = NDArray.fromList([-100.0], [], DType.float64);
          final pvVal = NDArray.fromList([-100.0], [], DType.float64);

          final resEnd = fv(rate, nper, pmt, pvVal, when: 'end');
          final resBegin = fv(rate, nper, pmt, pvVal, when: 'begin');

          expect(resEnd.scalar, closeTo(15692.92889433575, 1e-5));
          // with when=1, payments are at beginning, so more interest
          expect(resBegin.scalar, greaterThan(resEnd.scalar));
          expect(resBegin.scalar, closeTo(15757.629844104778, 1e-5));
        });
      });

      test('out parameter recycling', () {
        NDArray.scope(() {
          final rate = NDArray.fromList([0.05 / 12], [], DType.float64);
          final nper = NDArray.fromList([10.0 * 12], [], DType.float64);
          final pmt = NDArray.fromList([-100.0], [], DType.float64);
          final pvVal = NDArray.fromList([-100.0], [], DType.float64);
          final out = NDArray<Float64>.zeros([], DType.float64);

          final res = fv(rate, nper, pmt, pvVal, out: out);
          expect(identical(res, out), true);
          expect(out.scalar, closeTo(15692.92889433575, 1e-5));
        });
      });
    });

    group('Present Value (pv) Tests', () {
      test('Scalar-like 0D arrays', () {
        NDArray.scope(() {
          final rate = NDArray.fromList([0.05 / 12], [], DType.float64);
          final nper = NDArray.fromList([10.0 * 12], [], DType.float64);
          final pmt = NDArray.fromList([-100.0], [], DType.float64);
          final fvVal = NDArray.fromList(
            [15692.92889433575],
            [],
            DType.float64,
          );

          final res = pv(rate, nper, pmt, fvVal);
          expect(res.shape, <int>[]);
          expect(res.scalar, closeTo(-100.0, 1e-5));
        });
      });

      test('Rate = 0 case', () {
        NDArray.scope(() {
          final rate = NDArray.fromList([0.0], [], DType.float64);
          final nper = NDArray.fromList([120.0], [], DType.float64);
          final pmt = NDArray.fromList([-100.0], [], DType.float64);
          final fvVal = NDArray.fromList([12100.0], [], DType.float64);

          final res = pv(rate, nper, pmt, fvVal);
          // pv = - (fv + pmt * nper) = - (12100 + -100 * 120) = - (12100 - 12000) = -100
          expect(res.scalar, closeTo(-100.0, 1e-5));
        });
      });

      test('Broadcasting rate array', () {
        NDArray.scope(() {
          final rate = NDArray.fromList(
            [0.05 / 12, 0.04 / 12, 0.03 / 12],
            [3],
            DType.float64,
          );
          final nper = NDArray.fromList([10.0 * 12], [], DType.float64);
          final pmt = NDArray.fromList([-100.0], [], DType.float64);
          final fvVal = NDArray.fromList([15692.93], [], DType.float64);

          final res = pv(rate, nper, pmt, fvVal);
          expect(res.shape, [3]);
          expect(res.getCell([0]), closeTo(-100.00067132, 1e-5));
          expect(res.getCell([1]), closeTo(-649.26771385, 1e-5));
          expect(res.getCell([2]), closeTo(-1273.78633713, 1e-5));
        });
      });
    });

    group('Net Present Value (npv) Tests', () {
      test('1D cash flows, scalar rate', () {
        NDArray.scope(() {
          final rate = NDArray.fromList([0.08], [], DType.float64);
          final cashflows = NDArray.fromList(
            [-40000.0, 5000.0, 8000.0, 12000.0, 30000.0],
            [5],
            DType.float64,
          );

          final res = npv(rate, cashflows);
          expect(res.shape, <int>[]);
          expect(res.scalar, closeTo(3065.22267, 1e-5));
        });
      });

      test('Broadcasting multiple rates and multiple cash flows', () {
        NDArray.scope(() {
          final rates = NDArray.fromList(
            [0.00, 0.05, 0.10],
            [3],
            DType.float64,
          );
          final cashflows = NDArray.fromList(
            [-4000.0, 500.0, 800.0, -5000.0, 600.0, 900.0],
            [2, 3],
            DType.float64,
          );

          final res = npv(rates, cashflows);
          expect(res.shape, [3, 2]);

          expect(res.getCell([0, 0]), closeTo(-2700.0, 1e-2));
          expect(res.getCell([0, 1]), closeTo(-3500.0, 1e-2));
          expect(res.getCell([1, 0]), closeTo(-2798.19, 1e-2));
          expect(res.getCell([1, 1]), closeTo(-3612.24, 1e-2));
          expect(res.getCell([2, 0]), closeTo(-2884.30, 1e-2));
          expect(res.getCell([2, 1]), closeTo(-3710.74, 1e-2));
        });
      });
    });

    group('Internal Rate of Return (irr) Tests', () {
      test('Simple cash flows 1D', () {
        NDArray.scope(() {
          final cashflows = NDArray.fromList(
            [-100.0, 39.0, 59.0, 55.0, 20.0],
            [5],
            DType.float64,
          );
          final res = irr(cashflows);
          expect(res.shape, <int>[]);
          expect(res.scalar, closeTo(0.28095, 1e-5));
        });
      });

      test('Cash flows with negative rate result', () {
        NDArray.scope(() {
          final cashflows = NDArray.fromList(
            [-100.0, 0.0, 0.0, 74.0],
            [4],
            DType.float64,
          );
          final res = irr(cashflows);
          expect(res.scalar, closeTo(-0.0955, 1e-4));
        });
      });

      test('Same sign cash flows returns NaN', () {
        NDArray.scope(() {
          final cashflows = NDArray.fromList(
            [-100.0, -50.0, -20.0],
            [3],
            DType.float64,
          );
          final res = irr(cashflows);
          expect(res.scalar.isNaN, true);
        });
      });

      test('Same sign cash flows throws exception when requested', () {
        NDArray.scope(() {
          final cashflows = NDArray.fromList(
            [-100.0, -50.0, -20.0],
            [3],
            DType.float64,
          );
          expect(
            () => irr(cashflows, raiseExceptions: true),
            throwsA(isA<NoRealSolutionException>()),
          );
        });
      });
    });
  });
}
