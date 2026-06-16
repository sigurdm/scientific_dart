import 'dart:typed_data';
import 'dart:math' as math;
import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ndarray/ndarray.dart';

/// Emitter that divides the raw benchmark_harness score by 10.
/// By default, benchmark_harness's default exercise() method runs the benchmark's
/// run() method 10 times, meaning the raw output represents the duration of 10 runs.
/// Dividing by 10 gives the average duration of a single execution of run().
class DividedScoreEmitter implements ScoreEmitter {
  const DividedScoreEmitter();

  @override
  void emit(String testName, double value) {
    final singleRunScore = value / 10.0;
    print('$testName(RunTime): $singleRunScore us.');
  }
}

/// Custom benchmark base that enforces the DividedScoreEmitter.
abstract class NdarrayBenchmarkBase extends BenchmarkBase {
  const NdarrayBenchmarkBase(super.name)
    : super(emitter: const DividedScoreEmitter());
}

// ============================================================================
// 1. PROBABILITY DISTRIBUTIONS & RNG TRACK (Section 2)
// ============================================================================

class NormalDistributionBenchmark extends NdarrayBenchmarkBase {
  const NormalDistributionBenchmark()
    : super(
        'RNG Track  | Seeded normal() (Gaussian samples)       [size=50,000]',
      );

  @override
  void run() {
    final arr = normal([50000], seed: 42);
    arr.dispose();
  }
}

class PoissonDistributionBenchmark extends NdarrayBenchmarkBase {
  const PoissonDistributionBenchmark()
    : super(
        'RNG Track  | Seeded poisson() (Knuth vs Gaussian)      [size=20,000]',
      );

  @override
  void run() {
    final arr = poisson([20000], lam: 35.0, seed: 42);
    arr.dispose();
  }
}

class BinomialDistributionBenchmark extends NdarrayBenchmarkBase {
  const BinomialDistributionBenchmark()
    : super(
        'RNG Track  | Seeded binomial() (Bernoulli vs Normal)   [size=20,000]',
      );

  @override
  void run() {
    // n >= 50 triggers Normal distribution approximation track natively
    final arr = binomial([20000], n: 60, p: 0.4, seed: 42);
    arr.dispose();
  }
}

// ============================================================================
// 2. SORTING & SEARCHING TRACK (Section 7)
// ============================================================================

class NativeQSortContiguousBenchmark extends NdarrayBenchmarkBase {
  late Float64List templateData;
  late NDArray<double> target;

  NativeQSortContiguousBenchmark()
    : super(
        'SORT Track | Native C Heap sort() (Contiguous vector)   [size=30,000]',
      );

  @override
  void setup() {
    templateData = Float64List(30000);
    for (var i = 0; i < 30000; i++) {
      templateData[i] = (30000 - i).toDouble();
    }
    // Allocate target buffer once on setup
    target = NDArray.zeros([30000], DType.float64);
  }

  @override
  void run() {
    // High-speed bit-level block copy, bypassing any object allocations inside the timed run loop!
    target.data.setRange(0, 30000, templateData);
    final res = sort(target);
    res.dispose();
  }

  @override
  void teardown() {
    target.dispose();
  }
}

class NativeQSortRandomBenchmark extends NdarrayBenchmarkBase {
  late Float64List templateData;
  late NDArray<double> target;

  NativeQSortRandomBenchmark()
    : super(
        'SORT Track | Native C Heap sort() (Random vector)       [size=30,000]',
      );

  @override
  void setup() {
    final rand = math.Random(42);
    templateData = Float64List.fromList(
      List.generate(30000, (_) => rand.nextDouble()),
    );
    target = NDArray.zeros([30000], DType.float64);
  }

  @override
  void run() {
    target.data.setRange(0, 30000, templateData);
    final res = sort(target);
    res.dispose();
  }

  @override
  void teardown() {
    target.dispose();
  }
}

class BooleanMaskUnpackingBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> target;
  late NDArray<bool> mask;

  BooleanMaskUnpackingBenchmark()
    : super(
        'INDEX Track | Boolean Mask Advanced Indexing           [size=100,000]',
      );

  @override
  void setup() {
    final rand = math.Random(42);
    target = NDArray.zeros([100000], DType.float64);
    final maskData = List.generate(100000, (_) => rand.nextBool());
    mask = NDArray.fromList(maskData, [100000], DType.boolean);
  }

  @override
  void run() {
    final res = target[mask];
    res.dispose();
  }

  @override
  void teardown() {
    target.dispose();
    mask.dispose();
  }
}

class ArgsortBenchmark extends NdarrayBenchmarkBase {
  late Float64List templateData;
  late NDArray<double> target;

  ArgsortBenchmark()
    : super('SORT Track | Argsort (argsort)                     [size=30,000]');

  @override
  void setup() {
    templateData = Float64List(30000);
    for (var i = 0; i < 30000; i++) {
      templateData[i] = (30000 - i).toDouble();
    }
    target = NDArray.zeros([30000], DType.float64);
  }

  @override
  void run() {
    target.data.setRange(0, 30000, templateData);
    final indices = argsort(target);
    indices.dispose();
  }

  @override
  void teardown() {
    target.dispose();
  }
}

class TernaryWhereBroadcastingBenchmark extends NdarrayBenchmarkBase {
  late NDArray<bool> cond;
  late NDArray<double> x;
  late NDArray<double> y;
  late NDArray<double> outBuffer;

  TernaryWhereBroadcastingBenchmark()
    : super(
        'SORT Track | Ternary where() 3-Way Broadcasting       [shape=100x100]',
      );

  @override
  void setup() {
    cond = NDArray.zeros([100, 100], DType.boolean);
    x = NDArray.ones([100], DType.float64); // 1D vector stretching
    y = NDArray.ones([100, 100], DType.float64);
    outBuffer = NDArray.zeros([100, 100], DType.float64);
  }

  @override
  void run() {
    where(cond, x, y, outBuffer);
  }

  @override
  void teardown() {
    cond.dispose();
    x.dispose();
    y.dispose();
    outBuffer.dispose();
  }
}

// ============================================================================
// 3. ADVANCED LINEAR ALGEBRA & SIGNALS TRACK (Section 6)
// ============================================================================

class LapackMatrixInversionBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> a;

  LapackMatrixInversionBenchmark()
    : super(
        'LINALG Track| OpenBLAS LU Matrix Inversion (inv)       [shape=100x100]',
      );

  @override
  void setup() {
    a = NDArray.eye(100, DType.float64);
  }

  @override
  void run() {
    final res = inv(a);
    res.dispose();
  }

  @override
  void teardown() {
    a.dispose();
  }
}

class QrDecompositionBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> a;

  QrDecompositionBenchmark()
    : super(
        'LINALG Track| QR Decomposition (qr)                  [shape=30x30]',
      );

  @override
  void setup() {
    a = NDArray.zeros([30, 30], DType.float64);
    for (var i = 0; i < 30; i++) {
      for (var j = 0; j < 30; j++) {
        a.data[i * 30 + j] = (i + j + 1.0) / 10.0;
        if (i == j) a.data[i * 30 + j] += 1.0;
      }
    }
  }

  @override
  void run() {
    final res = qr(a);
    res.Q.dispose();
    res.R.dispose();
  }

  @override
  void teardown() {
    a.dispose();
  }
}

class SvdDecompositionBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> a;

  SvdDecompositionBenchmark()
    : super(
        'LINALG Track| SVD Decomposition (svd)                [shape=30x30]',
      );

  @override
  void setup() {
    a = NDArray.zeros([30, 30], DType.float64);
    for (var i = 0; i < 30; i++) {
      for (var j = 0; j < 30; j++) {
        a.data[i * 30 + j] = (i + j + 1.0) / 10.0;
        if (i == j) a.data[i * 30 + j] += 1.0;
      }
    }
  }

  @override
  void run() {
    final res = svd(a);
    res.dispose();
  }

  @override
  void teardown() {
    a.dispose();
  }
}

class NativeFftTransformBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> signal;

  NativeFftTransformBenchmark()
    : super(
        'LINALG Track| Native Mixed-Radix C FFI pocketfft (fft) [length=2048]',
      );

  @override
  void setup() {
    signal = NDArray.zeros([2048], DType.float64);
  }

  @override
  void run() {
    final res = fft(signal);
    res.dispose();
  }

  @override
  void teardown() {
    signal.dispose();
  }
}

class CholeskyDecompositionBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> a;

  CholeskyDecompositionBenchmark()
    : super(
        'LINALG Track| Cholesky Decomposition (cholesky)       [shape=30x30]',
      );

  @override
  void setup() {
    a = NDArray.zeros([30, 30], DType.float64);
    for (var i = 0; i < 30; i++) {
      for (var j = 0; j < 30; j++) {
        a.data[i * 30 + j] = (i + j + 1.0) / 10.0;
        if (i == j) {
          a.data[i * 30 + j] += 30.0; // Make it diagonally dominant
        }
      }
    }
  }

  @override
  void run() {
    final res = cholesky(a);
    res.dispose();
  }

  @override
  void teardown() {
    a.dispose();
  }
}

class MatmulBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> a;
  late NDArray<double> b;

  MatmulBenchmark()
    : super(
        'LINALG Track| Matrix Multiplication (matmul)          [shape=100x100]',
      );

  @override
  void setup() {
    a = NDArray.ones([100, 100], DType.float64);
    b = NDArray.ones([100, 100], DType.float64);
  }

  @override
  void run() {
    final res = matmul(a, b);
    res.dispose();
  }

  @override
  void teardown() {
    a.dispose();
    b.dispose();
  }
}

// ============================================================================
// 4. UNIVERSAL UFUNCS, REDUCTIONS & MEMORY TRACK (Section 9 Target)
// ============================================================================

class ElementwiseAddBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> x;
  late NDArray<double> y;
  late NDArray<double> outBuffer;

  ElementwiseAddBenchmark()
    : super(
        'MEMORY Track| Element-wise Same-Shape add(x, y)       [size=300,000]',
      );

  @override
  void setup() {
    x = NDArray.ones([300000], DType.float64);
    y = NDArray.ones([300000], DType.float64);
    outBuffer = NDArray.create([300000], DType.float64);
  }

  @override
  void run() {
    add(x, y, out: outBuffer);
  }

  @override
  void teardown() {
    x.dispose();
    y.dispose();
    outBuffer.dispose();
  }
}

class ScalarAdditionBroadcastBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> x;
  late NDArray<double> scalarArr;
  late NDArray<double> outBuffer;

  ScalarAdditionBroadcastBenchmark()
    : super(
        'MEMORY Track| Scalar Array Broadcast add(x, scalar)   [size=300,000]',
      );

  @override
  void setup() {
    x = NDArray.ones([300000], DType.float64);
    scalarArr = NDArray.fromList(Float64List.fromList([5.0]), [
      1,
    ], DType.float64);
    outBuffer = NDArray.create([300000], DType.float64);
  }

  @override
  void run() {
    add(x, scalarArr, out: outBuffer);
  }

  @override
  void teardown() {
    x.dispose();
    scalarArr.dispose();
    outBuffer.dispose();
  }
}

class SinUfuncBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> x;
  late NDArray<double> outBuffer;

  SinUfuncBenchmark()
    : super(
        'MEMORY Track| Universal math function sin(x)          [size=100,000]',
      );

  @override
  void setup() {
    x = NDArray.ones([100000], DType.float64);
    outBuffer = NDArray.create([100000], DType.float64);
  }

  @override
  void run() {
    sin(x, out: outBuffer);
  }

  @override
  void teardown() {
    x.dispose();
    outBuffer.dispose();
  }
}

class CosUfuncBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> x;
  late NDArray<double> outBuffer;

  CosUfuncBenchmark()
    : super(
        'MEMORY Track| Universal math function cos(x)          [size=100,000]',
      );

  @override
  void setup() {
    x = NDArray.ones([100000], DType.float64);
    outBuffer = NDArray.create([100000], DType.float64);
  }

  @override
  void run() {
    cos(x, out: outBuffer);
  }

  @override
  void teardown() {
    x.dispose();
    outBuffer.dispose();
  }
}

class ExpUfuncBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> x;
  late NDArray<double> outBuffer;

  ExpUfuncBenchmark()
    : super(
        'MEMORY Track| Universal math function exp(x)          [size=100,000]',
      );

  @override
  void setup() {
    x = NDArray.ones([100000], DType.float64);
    outBuffer = NDArray.create([100000], DType.float64);
  }

  @override
  void run() {
    exp(x, out: outBuffer);
  }

  @override
  void teardown() {
    x.dispose();
    outBuffer.dispose();
  }
}

class ClipUfuncBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> x;
  late NDArray<double> outBuffer;

  ClipUfuncBenchmark()
    : super(
        'MEMORY Track| Universal math function clip(x)          [size=300,000]',
      );

  @override
  void setup() {
    x = NDArray.ones([300000], DType.float64);
    outBuffer = NDArray.create([300000], DType.float64);
  }

  @override
  void run() {
    clip(x, min: 0.0, max: 0.5, out: outBuffer);
  }

  @override
  void teardown() {
    x.dispose();
    outBuffer.dispose();
  }
}

class SumReductionBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> x;

  SumReductionBenchmark()
    : super(
        'MEMORY Track| Flat Memory Reduction walk sum(x)       [size=300,000]',
      );

  @override
  void setup() {
    x = NDArray.ones([300000], DType.float64);
  }

  @override
  void run() {
    sum(x);
  }

  @override
  void teardown() {
    x.dispose();
  }
}

class ZerosBenchmark extends NdarrayBenchmarkBase {
  late List<int> shape;

  ZerosBenchmark()
    : super(
        'MEMORY Track| Zeros Array Creation (zeros)         [size=1,000,000]',
      );

  @override
  void setup() {
    shape = [1000, 1000];
  }

  @override
  void run() {
    final arr = NDArray<double>.zeros(shape, DType.float64);
    arr.dispose();
  }
}

class ConcatenateBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> a;
  late NDArray<double> b;

  ConcatenateBenchmark()
    : super(
        'MEMORY Track| Flat Array Concatenation (concatenate) [size=1,000,000]',
      );

  @override
  void setup() {
    a = NDArray.ones([500000], DType.float64);
    b = NDArray.ones([500000], DType.float64);
  }

  @override
  void run() {
    final res = concatenate([a, b], axis: 0);
    res.dispose();
  }

  @override
  void teardown() {
    a.dispose();
    b.dispose();
  }
}

class ContiguousViewFlattenBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> parent;
  late NDArray<double> view;

  ContiguousViewFlattenBenchmark()
    : super(
        'MEMORY Track| Contiguous View Flatten (flatten)       [size=300,000]',
      );

  @override
  void setup() {
    parent = NDArray.ones([600000], DType.float64);
    // Take a slice representing first 300,000 elements. It is contiguous, but totalSize < data.length!
    view = parent.slice([Slice(start: 0, stop: 300000)]);
  }

  @override
  void run() {
    final res = view.flatten();
    res.dispose();
  }

  @override
  void teardown() {
    parent.dispose();
    view.dispose();
  }
}

class ContiguousViewSumBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> parent;
  late NDArray<double> view;

  ContiguousViewSumBenchmark()
    : super(
        'MEMORY Track| Contiguous View Sum Reduction (sum)     [size=300,000]',
      );

  @override
  void setup() {
    parent = NDArray.ones([600000], DType.float64);
    view = parent.slice([Slice(start: 0, stop: 300000)]);
  }

  @override
  void run() {
    sum(view);
  }

  @override
  void teardown() {
    parent.dispose();
    view.dispose();
  }
}

class StridedElementwiseAddBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> parentX;
  late NDArray<double> parentY;
  late NDArray<double> xView;
  late NDArray<double> yView;
  late NDArray<double> outBuffer;

  StridedElementwiseAddBenchmark()
    : super(
        'MEMORY Track| Strided non-contiguous add(x, y)         [shape=500x500]',
      );

  @override
  void setup() {
    parentX = NDArray.ones([500, 500], DType.float64);
    parentY = NDArray.ones([500, 500], DType.float64);
    xView = parentX.transposed;
    yView = parentY.transposed;
    outBuffer = NDArray.create([500, 500], DType.float64);
  }

  @override
  void run() {
    add(xView, yView, out: outBuffer);
  }

  @override
  void teardown() {
    parentX.dispose();
    parentY.dispose();
    xView.dispose();
    yView.dispose();
    outBuffer.dispose();
  }
}

// ============================================================================
// 5. DISTANCE METRICS TRACK (pdist & cdist)
// ============================================================================

class PdistEuclideanBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> x;

  PdistEuclideanBenchmark()
    : super(
        'DISTANCE Track| pdist Euclidean                                 [shape=500x100]',
      );

  @override
  void setup() {
    x = normal([500, 100], seed: 42);
  }

  @override
  void run() {
    final res = pdist(x, metric: DistanceMetric.euclidean);
    res.dispose();
  }

  @override
  void teardown() {
    x.dispose();
  }
}

class PdistCosineBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> x;

  PdistCosineBenchmark()
    : super(
        'DISTANCE Track| pdist Cosine                                    [shape=500x100]',
      );

  @override
  void setup() {
    x = normal([500, 100], seed: 42);
  }

  @override
  void run() {
    final res = pdist(x, metric: DistanceMetric.cosine);
    res.dispose();
  }

  @override
  void teardown() {
    x.dispose();
  }
}

class PdistHammingBenchmark extends NdarrayBenchmarkBase {
  late NDArray<Int32> x;

  PdistHammingBenchmark()
    : super(
        'DISTANCE Track| pdist Hamming                                   [shape=500x100]',
      );

  @override
  void setup() {
    x = randint([500, 100], low: 0, high: 2, dtype: DType.int32, seed: 42);
  }

  @override
  void run() {
    final res = pdist(x, metric: DistanceMetric.hamming);
    res.dispose();
  }

  @override
  void teardown() {
    x.dispose();
  }
}

class CdistEuclideanBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> xa;
  late NDArray<double> xb;

  CdistEuclideanBenchmark()
    : super(
        'DISTANCE Track| cdist Euclidean                                 [shape=500x100 vs 500x100]',
      );

  @override
  void setup() {
    xa = normal([500, 100], seed: 42);
    xb = normal([500, 100], seed: 43);
  }

  @override
  void run() {
    final res = cdist(xa, xb, metric: DistanceMetric.euclidean);
    res.dispose();
  }

  @override
  void teardown() {
    xa.dispose();
    xb.dispose();
  }
}

class CdistCosineBenchmark extends NdarrayBenchmarkBase {
  late NDArray<double> xa;
  late NDArray<double> xb;

  CdistCosineBenchmark()
    : super(
        'DISTANCE Track| cdist Cosine                                    [shape=500x100 vs 500x100]',
      );

  @override
  void setup() {
    xa = normal([500, 100], seed: 42);
    xb = normal([500, 100], seed: 43);
  }

  @override
  void run() {
    final res = cdist(xa, xb, metric: DistanceMetric.cosine);
    res.dispose();
  }

  @override
  void teardown() {
    xa.dispose();
    xb.dispose();
  }
}

class CdistHammingBenchmark extends NdarrayBenchmarkBase {
  late NDArray<Int32> xa;
  late NDArray<Int32> xb;

  CdistHammingBenchmark()
    : super(
        'DISTANCE Track| cdist Hamming                                   [shape=500x100 vs 500x100]',
      );

  @override
  void setup() {
    xa = randint([500, 100], low: 0, high: 2, dtype: DType.int32, seed: 42);
    xb = randint([500, 100], low: 0, high: 2, dtype: DType.int32, seed: 43);
  }

  @override
  void run() {
    final res = cdist(xa, xb, metric: DistanceMetric.hamming);
    res.dispose();
  }

  @override
  void teardown() {
    xa.dispose();
    xb.dispose();
  }
}

// ============================================================================
// MAIN SUITE RUNNER ENTRYPOINT
// ============================================================================

void main() {
  // Limit OpenBLAS execution to 1 thread to avoid parallel thread context switch
  // overhead on lightweight/small matrices in the benchmark suite.
  setNumThreads(1);

  print(
    '============================================================================',
  );
  print(
    '         ndarray ALL-INCLUSIVE PERFORMANCE BENCHMARK SUITE MASTER          ',
  );
  print(
    '============================================================================',
  );
  print('Establishing high-precision baseline metrics pre-optimization...\n');

  print('--- TRACK A: RANDOM DISTRIBUTIONS & RNG SOLVERS ---');
  const NormalDistributionBenchmark().report();
  const PoissonDistributionBenchmark().report();
  const BinomialDistributionBenchmark().report();

  print('\n--- TRACK B: NATIVE C HEAP SORTING & SEARCHING BROADCASTS ---');
  NativeQSortContiguousBenchmark().report();
  NativeQSortRandomBenchmark().report();
  BooleanMaskUnpackingBenchmark().report();
  ArgsortBenchmark().report();
  TernaryWhereBroadcastingBenchmark().report();

  print(
    '\n--- TRACK C: OPENBLAS LINEAR ALGEBRA & NATIVE POCKETFFT SIGNALS ---',
  );
  LapackMatrixInversionBenchmark().report();
  QrDecompositionBenchmark().report();
  SvdDecompositionBenchmark().report();
  NativeFftTransformBenchmark().report();
  CholeskyDecompositionBenchmark().report();
  MatmulBenchmark().report();

  print(
    '\n--- TRACK D: UNIVERSAL UFUNCS, REDUCTIONS & MEMORY STRIDES (SECTION 9 TARGET) ---',
  );
  ElementwiseAddBenchmark().report();
  ScalarAdditionBroadcastBenchmark().report();
  SinUfuncBenchmark().report();
  CosUfuncBenchmark().report();
  ExpUfuncBenchmark().report();
  SumReductionBenchmark().report();
  ZerosBenchmark().report();
  ConcatenateBenchmark().report();
  ClipUfuncBenchmark().report();
  ContiguousViewFlattenBenchmark().report();
  ContiguousViewSumBenchmark().report();
  StridedElementwiseAddBenchmark().report();

  print('\n--- TRACK E: DISTANCE METRICS (pdist & cdist) ---');
  PdistEuclideanBenchmark().report();
  PdistCosineBenchmark().report();
  PdistHammingBenchmark().report();
  CdistEuclideanBenchmark().report();
  CdistCosineBenchmark().report();
  CdistHammingBenchmark().report();

  print(
    '\n============================================================================',
  );
  print(
    'Exhaustive Master Baseline Performance Benchmarks completed successfully.',
  );
  print(
    '============================================================================',
  );
}
