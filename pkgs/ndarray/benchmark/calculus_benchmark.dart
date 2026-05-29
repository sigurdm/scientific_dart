import 'package:benchmark_harness/benchmark_harness.dart';
import 'package:ndarray/ndarray.dart';

class Trapz1DBenchmark extends BenchmarkBase {
  late NDArray<double> y;
  Trapz1DBenchmark() : super('Calculus | trapz 1D (Float64) [size=1,000,000]');

  @override
  void setup() {
    y = NDArray.zeros([1000000], DType.float64);
    for (var i = 0; i < 1000000; i++) {
      y.data[i] = i.toDouble();
    }
  }

  @override
  void run() {
    final res = trapz(y);
    res.dispose();
  }

  @override
  void teardown() {
    y.dispose();
  }
}

class Gradient1DBenchmark extends BenchmarkBase {
  late NDArray<double> f;
  Gradient1DBenchmark()
    : super('Calculus | gradient 1D (Float64) [size=1,000,000]');

  @override
  void setup() {
    f = NDArray.zeros([1000000], DType.float64);
    for (var i = 0; i < 1000000; i++) {
      f.data[i] = i.toDouble() * i.toDouble();
    }
  }

  @override
  void run() {
    final res = gradient(f);
    res.dispose();
  }

  @override
  void teardown() {
    f.dispose();
  }
}

class Gradient2DBenchmark extends BenchmarkBase {
  late NDArray<double> f;
  Gradient2DBenchmark()
    : super('Calculus | gradient 2D (Float64) [size=1,000x1,000]');

  @override
  void setup() {
    f = NDArray.zeros([1000, 1000], DType.float64);
    for (var i = 0; i < 1000000; i++) {
      f.data[i] = i.toDouble();
    }
  }

  @override
  void run() {
    // Calculates partial derivative along axis 0
    final res = gradient(f, axis: 0);
    res.dispose();
  }

  @override
  void teardown() {
    f.dispose();
  }
}

void main() {
  Trapz1DBenchmark().report();
  Gradient1DBenchmark().report();
  Gradient2DBenchmark().report();
}
