import 'package:criterion/criterion.dart';
import 'package:ndarray/ndarray.dart';

void main() async {
  setNumThreads(1);
  const nSims = 10000;

  await criterion(
    'NDArray Quantitative Financial Operations Benchmark Suite',
    (c) {
      final rate =
          linspace<double>(0.01, 0.15, nSims, dtype: DType.float64)
              as NDArray<Float64>;
      final nper =
          linspace<double>(1.0, 30.0, nSims, dtype: DType.float64)
              as NDArray<Float64>;
      final pmt =
          linspace<double>(-1000.0, -100.0, nSims, dtype: DType.float64)
              as NDArray<Float64>;
      final pvVal =
          linspace<double>(10000.0, 100000.0, nSims, dtype: DType.float64)
              as NDArray<Float64>;
      final fvVal =
          linspace<double>(0.0, 50000.0, nSims, dtype: DType.float64)
              as NDArray<Float64>;

      c.group('1. Time Value of Money (10k parameter simulations)', () {
        c.bench('fv(rate, nper, pmt, pv) [10k]', () {
          final res = fv(rate, nper, pmt, pvVal);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(nSims));

        c.bench('pv(rate, nper, pmt, fv) [10k]', () {
          final res = pv(rate, nper, pmt, fvVal);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(nSims));
      });

      c.group('2. Cash Flow Discounting & Returns', () {
        const nPeriods = 10000;
        final singleRate = NDArray<Float64>.scalar(
          Float64(0.05),
          dtype: DType.float64,
        );
        final cashFlows =
            linspace<double>(-1000.0, 500.0, nPeriods, dtype: DType.float64)
                as NDArray<Float64>;

        c.bench(
          'npv(rate=0.05, cashflows=[10k])',
          () {
            final res = npv(singleRate, cashFlows);
            blackhole(res);
            res.dispose();
          },
          throughput: Throughput.elements(nPeriods),
        );

        final irrFlows = NDArray<Float64>.fromList(
          [-10000.0, ...List.filled(49, 350.0)],
          [50],
          DType.float64,
        );

        c.bench('irr(cashflows=[50 periods])', () {
          final res = irr(irrFlows);
          blackhole(res);
          res.dispose();
        });
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/financial',
    ),
  );
}
