import 'dart:math' as math;
import 'package:ndarray/ndarray.dart';

void benchmark(
  String name,
  Function runFn, {
  int iterations = 200,
  int warmup = 20,
}) {
  for (var i = 0; i < warmup; i++) {
    runFn();
  }
  final sw = Stopwatch()..start();
  for (var i = 0; i < iterations; i++) {
    runFn();
  }
  sw.stop();
  final usPerOp = sw.elapsedMicroseconds / iterations;
  print(
    "Dart  ${name.padRight(45)}: ${usPerOp.toStringAsFixed(2).padLeft(8)} us/op",
  );
}

void main() {
  print(
    "============================================================================",
  );
  print(
    "         Dart Scientific NDArray Benchmark Suite                            ",
  );
  print(
    "============================================================================",
  );

  NDArray.scope(() {
    // 1. einsum matrix multiplication
    final aList = List.generate(10000, (i) => i.toDouble());
    final bList = List.generate(10000, (i) => i.toDouble());
    final aMat = NDArray<Float64>.fromList(aList, [100, 100], DType.float64);
    final bMat = NDArray<Float64>.fromList(bList, [100, 100], DType.float64);
    final subs2d = EinsumSubscripts.parse('ij,jk->ik');
    benchmark("einsum matrix mult ('ij,jk->ik') [100x100]", () {
      einsum(subs2d, [aMat, bMat]);
    }, iterations: 500);

    final outMat = NDArray<Float64>.create([100, 100], DType.float64);
    benchmark("einsum matrix mult with out: [100x100]", () {
      einsum(subs2d, [aMat, bMat], out: outMat);
    }, iterations: 500);

    // 2. einsum ellipsis broadcasting
    final a3dList = List.generate(4000, (i) => i.toDouble());
    final b3dList = List.generate(4000, (i) => i.toDouble());
    final a3d = NDArray<Float64>.fromList(a3dList, [10, 20, 20], DType.float64);
    final b3d = NDArray<Float64>.fromList(b3dList, [10, 20, 20], DType.float64);
    final subs3d = EinsumSubscripts.parse('...ij,...jk->...ik');

    benchmark("einsum batch matmul ('...ij,...jk->...ik') [10x20x20]", () {
      einsum(subs3d, [a3d, b3d]);
    }, iterations: 500);

    final out3d = NDArray<Float64>.create([10, 20, 20], DType.float64);
    benchmark("einsum batch matmul with out: [10x20x20]", () {
      einsum(subs3d, [a3d, b3d], out: out3d);
    }, iterations: 500);

    final subs3Op = EinsumSubscripts.parse('ij,jk,kl->il');
    final cMat = NDArray<Float64>.fromList(bList, [100, 100], DType.float64);
    benchmark("einsum 3-operand ('ij,jk,kl->il') [100x100]", () {
      einsum(subs3Op, [aMat, bMat, cMat]);
    }, iterations: 100);

    // 3. tensordot count=2
    final axesCount2 = const TensordotAxes.count(2);
    benchmark("tensordot axes=2 [100x100]", () {
      tensordot(aMat, bMat, axes: axesCount2);
    }, iterations: 500);

    // 4. tensordot explicit ([1],[0])
    final axesExplicit = const TensordotAxes.explicit([1], [0]);
    benchmark("tensordot axes=([1],[0]) [100x100]", () {
      tensordot(aMat, bMat, axes: axesExplicit);
    }, iterations: 500);

    // 5. correlate 1D full mode
    final a1dList = List.generate(10000, (i) => i.toDouble());
    final v1dList = List.generate(100, (i) => i.toDouble());
    final a1d = NDArray<Float64>.fromList(a1dList, [10000], DType.float64);
    final v1d = NDArray<Float64>.fromList(v1dList, [100], DType.float64);

    benchmark("correlate mode='full' [N=10,000, K=100]", () {
      correlate(a1d, v1d, mode: ConvMode.full);
    }, iterations: 200);

    // 6. convolve 1D full mode
    benchmark("convolve mode='full' [N=10,000, K=100]", () {
      convolve(a1d, v1d, mode: ConvMode.full);
    }, iterations: 200);

    // 7. brentq 1D root finding
    benchmark("brentq root scalar [x^2 - 4]", () {
      brentq((x) => x * x - 4.0, 0.0, 3.0);
    }, iterations: 1000);

    // 8. newton 1D root finding
    benchmark("newton root scalar [x^2 - 2]", () {
      newton((x) => x * x - 2.0, 1.0, fprime: (x) => 2.0 * x);
    }, iterations: 1000);

    // 9. nelder_mead 2D Rosenbrock minimization
    final x0Rosen = NDArray<Float64>.fromList([-1.2, 1.0], [2], DType.float64);
    benchmark("nelder_mead minimization [Rosenbrock 2D]", () {
      nelder_mead(
        (x) {
          final px = x.getCell([0]).toDouble();
          final py = x.getCell([1]).toDouble();
          return 100.0 * math.pow(py - px * px, 2).toDouble() +
              math.pow(1.0 - px, 2).toDouble();
        },
        x0Rosen,
        maxiter: 1000,
      );
    }, iterations: 100);

    // 10. lbfgs 2D quadratic bowl minimization
    final x0Bowl = NDArray<Float64>.fromList([5.0, 5.0], [2], DType.float64);
    benchmark("lbfgs minimization [Quadratic 2D]", () {
      lbfgs((x) {
        final px = x.getCell([0]).toDouble();
        final py = x.getCell([1]).toDouble();
        return (px - 1.0) * (px - 1.0) + (py - 2.0) * (py - 2.0);
      }, x0Bowl);
    }, iterations: 100);
  });
}
