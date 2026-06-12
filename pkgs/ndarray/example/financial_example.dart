import 'package:ndarray/ndarray.dart';

void main() {
  NDArray.scope(() {
    // 1. Future Value (fv)
    // What is the future value after 10 years of saving $100 now, with
    // an additional monthly savings of $100. Assume the interest rate is
    // 5% (annually) compounded monthly?
    final NDArray<Float64> rate = NDArray.fromList(
      [0.05 / 12],
      [],
      DType.float64,
    );
    final NDArray<Float64> nper = NDArray.fromList(
      [10.0 * 12],
      [],
      DType.float64,
    );
    final NDArray<Float64> pmt = NDArray.fromList([-100.0], [], DType.float64);
    final NDArray<Float64> pvVal = NDArray.fromList(
      [-100.0],
      [],
      DType.float64,
    );

    final NDArray<Float64> fvResult = fv(rate, nper, pmt, pvVal);
    print(
      'FV Result: \$${fvResult.scalar.toStringAsFixed(2)}',
    ); // Expected: $15692.93

    // 2. Present Value (pv)
    // What is the present value (investment) needed to get $15,692.93 after
    // 10 years of saving $100/month at 5% annual interest?
    final NDArray<Float64> fvVal = NDArray.fromList(
      [15692.92889433575],
      [],
      DType.float64,
    );
    final NDArray<Float64> pvResult = pv(rate, nper, pmt, fvVal);
    print(
      'PV Result: \$${pvResult.scalar.toStringAsFixed(2)}',
    ); // Expected: -$100.00

    // 3. Net Present Value (npv)
    // Cash flows: invest 100, then withdraw 39, 59, 55, 20
    final NDArray<Float64> cashFlows = NDArray.fromList(
      [-100.0, 39.0, 59.0, 55.0, 20.0],
      [5],
      DType.float64,
    );
    final NDArray<Float64> discountRate = NDArray.fromList(
      [0.08],
      [],
      DType.float64,
    );
    final NDArray<Float64> npvResult = npv(discountRate, cashFlows);
    print('NPV Result: \$${npvResult.scalar.toStringAsFixed(2)}');

    // 4. Internal Rate of Return (irr)
    // Solves for rate where NPV is 0.
    final NDArray<Float64> irrResult = irr(cashFlows);
    print(
      'IRR Result: ${(irrResult.scalar * 100).toStringAsFixed(2)}%',
    ); // Expected: 28.10%
  });
}
