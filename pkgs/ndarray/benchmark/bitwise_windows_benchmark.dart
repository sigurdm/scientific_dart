import 'package:criterion/criterion.dart';
import 'package:ndarray/ndarray.dart';

void main() async {
  setNumThreads(1);
  const size = 100000;

  await criterion(
    'NDArray Bitwise, Windows & Special Functions Benchmark Suite',
    (c) {
      c.group('1. DSP Windowing Functions (100k points)', () {
        c.bench('hanning(100k)', () {
          final res = hanning<double>(size);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('hamming(100k)', () {
          final res = hamming<double>(size);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));
      });

      c.group('2. Special Mathematical Functions (100k elements)', () {
        final floatVec = linspace<double>(
          0.0,
          10.0,
          size,
          dtype: DType.float64,
        );

        c.bench('i0(x) (Bessel I0) [100k]', () {
          final res = i0<double, double>(floatVec);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench(
          'sinc(x) (Normalized Sinc) [100k]',
          () {
            final res = sinc<double, double>(floatVec);
            blackhole(res);
            res.dispose();
          },
          throughput: Throughput.elements(size),
        );
      });

      c.group('3. Bitwise Integer Operations (100k elements)', () {
        final intA = NDArray<int>.fromList(List.generate(size, (i) => i * 13), [
          size,
        ], DType.int32);
        final intB = NDArray<int>.fromList(
          List.generate(size, (i) => i * 7 + 1),
          [size],
          DType.int32,
        );

        c.bench('bitwise_and(a, b) [100k Int32]', () {
          final res = bitwise_and(intA, intB);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('bitwise_or(a, b) [100k Int32]', () {
          final res = bitwise_or(intA, intB);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('bitwise_xor(a, b) [100k Int32]', () {
          final res = bitwise_xor(intA, intB);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        c.bench('invert(a) [100k Int32]', () {
          final res = invert(intA);
          blackhole(res);
          res.dispose();
        }, throughput: Throughput.elements(size));

        final shiftAmt = NDArray<int>.fromList(
          List.generate(size, (i) => (i % 8)),
          [size],
          DType.int32,
        );

        c.bench(
          'left_shift(a, shift) [100k Int32]',
          () {
            final res = left_shift(intA, shiftAmt);
            blackhole(res);
            res.dispose();
          },
          throughput: Throughput.elements(size),
        );

        c.bench(
          'right_shift(a, shift) [100k Int32]',
          () {
            final res = right_shift(intA, shiftAmt);
            blackhole(res);
            res.dispose();
          },
          throughput: Throughput.elements(size),
        );
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/bitwise_windows',
    ),
  );
}
