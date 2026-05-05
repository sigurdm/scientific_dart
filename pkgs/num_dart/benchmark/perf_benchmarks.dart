import 'dart:math' as standard_math;
import 'dart:typed_data';
import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:num_dart/num_dart.dart';

// ============================================================================
// 1. PROBABILITY DISTRIBUTIONS & RNG TRACK (Section 2)
// ============================================================================

class NormalDistributionBenchmark extends BenchmarkBase {
  const NormalDistributionBenchmark()
    : super(
        'RNG Track  | Seeded normal() (Gaussian samples)       [size=50,000]',
      );

  @override
  void run() {
    // Distributions are top-level functions taking shape and math.Random objects for seeding!
    normal([50000], random: standard_math.Random(42));
  }
}

class PoissonDistributionBenchmark extends BenchmarkBase {
  const PoissonDistributionBenchmark()
    : super(
        'RNG Track  | Seeded poisson() (Knuth vs Gaussian)      [size=20,000]',
      );

  @override
  void run() {
    // lam >= 30.0 triggers Gaussian Normal approximation track natively
    poisson([20000], lam: 35.0, random: standard_math.Random(42));
  }
}

class BinomialDistributionBenchmark extends BenchmarkBase {
  const BinomialDistributionBenchmark()
    : super(
        'RNG Track  | Seeded binomial() (Bernoulli vs Normal)   [size=20,000]',
      );

  @override
  void run() {
    // n >= 50 triggers Normal distribution approximation track natively
    binomial([20000], 60, 0.4, random: standard_math.Random(42));
  }
}

// ============================================================================
// 2. SORTING & SEARCHING TRACK (Section 7)
// ============================================================================

class NativeQSortContiguousBenchmark extends BenchmarkBase {
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
    sort(target);
  }

  @override
  void teardown() {
    target.dispose();
  }
}

class TernaryWhereBroadcastingBenchmark extends BenchmarkBase {
  late NDArray<bool> cond;
  late NDArray<double> x;
  late NDArray<double> y;

  TernaryWhereBroadcastingBenchmark()
    : super(
        'SORT Track | Ternary where() 3-Way Broadcasting       [shape=100x100]',
      );

  @override
  void setup() {
    cond = NDArray.zeros([100, 100], DType.boolean);
    x = NDArray.ones([100], DType.float64); // 1D vector stretching
    y = NDArray.ones([100, 100], DType.float64);
  }

  @override
  void run() {
    where(cond, x, y);
  }

  @override
  void teardown() {
    cond.dispose();
    x.dispose();
    y.dispose();
  }
}

// ============================================================================
// 3. ADVANCED LINEAR ALGEBRA & SIGNALS TRACK (Section 6)
// ============================================================================

class LapackMatrixInversionBenchmark extends BenchmarkBase {
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
    inv(a);
  }

  @override
  void teardown() {
    a.dispose();
  }
}

class QrDecompositionBenchmark extends BenchmarkBase {
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
    res['Q']!.dispose();
    res['R']!.dispose();
  }

  @override
  void teardown() {
    a.dispose();
  }
}

class SvdDecompositionBenchmark extends BenchmarkBase {
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
    res['U']!.dispose();
    res['S']!.dispose();
    res['Vh']!.dispose();
  }

  @override
  void teardown() {
    a.dispose();
  }
}

class NativeFftTransformBenchmark extends BenchmarkBase {
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
    fft(signal);
  }

  @override
  void teardown() {
    signal.dispose();
  }
}

// ============================================================================
// 4. UNIVERSAL UFUNCS, REDUCTIONS & MEMORY TRACK (Section 9 Target)
// ============================================================================

class ElementwiseAddBenchmark extends BenchmarkBase {
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

class ScalarAdditionBroadcastBenchmark extends BenchmarkBase {
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

class SinUfuncBenchmark extends BenchmarkBase {
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

class SumReductionBenchmark extends BenchmarkBase {
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

// ============================================================================
// MAIN SUITE RUNNER ENTRYPOINT
// ============================================================================

void main() {
  print(
    '============================================================================',
  );
  print(
    '         num_dart ALL-INCLUSIVE PERFORMANCE BENCHMARK SUITE MASTER          ',
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
  TernaryWhereBroadcastingBenchmark().report();

  print(
    '\n--- TRACK C: OPENBLAS LINEAR ALGEBRA & NATIVE POCKETFFT SIGNALS ---',
  );
  LapackMatrixInversionBenchmark().report();
  QrDecompositionBenchmark().report();
  SvdDecompositionBenchmark().report();
  NativeFftTransformBenchmark().report();

  print(
    '\n--- TRACK D: UNIVERSAL UFUNCS, REDUCTIONS & MEMORY STRIDES (SECTION 9 TARGET) ---',
  );
  ElementwiseAddBenchmark().report();
  ScalarAdditionBroadcastBenchmark().report();
  SinUfuncBenchmark().report();
  SumReductionBenchmark().report();

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
