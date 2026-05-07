import 'dart:math' as math;
import 'dart:math' show Random;
import 'dart:ffi' as ffi;
import 'numdart_bindings.dart';
import 'ndarray.dart';
import 'operations.dart';

/// Generates an array with random values uniformly distributed in the half-open interval `[0.0, 1.0)`.
///
/// **Preconditions:**
/// - [dtype] must be a floating point type (`DType.float32` or `DType.float64`).
///
/// **Throws:**
/// - [ArgumentError] if the provided [dtype] is not a supported floating point type.
///
/// **Performance considerations:**
/// - Algorithmic time complexity is $O(N)$ and space complexity is $O(N)$, where $N$ is the total size of
///   the generated array (product of [shape] dimensions).
/// - Offloads element generation directly to high-speed C FFI vector functions (`v_uniform_double` / `v_uniform_float`),
///   yielding ultra-high performance and avoiding Dart loop context switching overhead.
///
/// **Example:**
/// {@example /example/random_example.dart lang=dart}
///
/// Refer to the [Uniform Distribution Reference](https://en.wikipedia.org/wiki/Continuous_uniform_distribution)
/// for details on continuous uniform distributions.
///
/// By default, uses Dart's standard [Random] class, which is not cryptographically secure.
/// You can pass a secure random object via the [random] parameter if needed.
NDArray<double> uniform(
  List<int> shape, {
  DType dtype = DType.float64,
  Random? random,
}) {
  if (dtype != DType.float32 && dtype != DType.float64) {
    throw ArgumentError('uniform only supports float types for now');
  }
  final arr = NDArray<double>.create(shape, dtype);
  final rand = random ?? Random();
  final len = arr.data.length;
  final seed = rand.nextInt(4294967296);

  if (dtype == DType.float64) {
    v_uniform_double(arr.pointer.cast<ffi.Double>(), len, seed);
  } else {
    v_uniform_float(arr.pointer.cast<ffi.Float>(), len, seed);
  }
  return arr;
}

/// Generates an array with random integer values uniformly distributed in the half-open interval `[low, high)`.
///
/// **Preconditions:**
/// - [dtype] must be an integer type (`DType.int32` or `DType.int64`).
/// - [low] must be strictly less than [high].
///
/// **Throws:**
/// - [ArgumentError] if [dtype] is not a supported integer type.
/// - [ArgumentError] if [low] is greater than or equal to [high].
///
/// **Performance considerations:**
/// - Algorithmic time complexity is $O(N)$ and space complexity is $O(N)$, where $N$ is the total size of
///   the generated array (product of [shape] dimensions).
/// - Offloads element generation directly to high-speed C FFI vector functions (`v_randint_int64` / `v_randint_int32`),
///   providing optimal random integers generation speed.
///
/// **Example:**
/// {@example /example/random_example.dart lang=dart}
///
/// Refer to the [Discrete Uniform Distribution Reference](https://en.wikipedia.org/wiki/Discrete_uniform_distribution)
/// for details on discrete uniform distributions.
///
/// By default, uses Dart's standard [Random] class, which is not cryptographically secure.
/// You can pass a secure random object via the [random] parameter if needed.
NDArray<int> randint(
  List<int> shape, {
  required int low,
  required int high,
  DType dtype = DType.int64,
  Random? random,
}) {
  if (dtype != DType.int32 && dtype != DType.int64) {
    throw ArgumentError('randint only supports integer types');
  }
  if (low >= high) {
    throw ArgumentError('low must be less than high');
  }
  final arr = NDArray<int>.create(shape, dtype);
  final rand = random ?? Random();
  final len = arr.data.length;
  final seed = rand.nextInt(4294967296);

  if (dtype == DType.int64) {
    v_randint_int64(arr.pointer.cast<ffi.Int64>(), len, low, high, seed);
  } else {
    v_randint_int32(arr.pointer.cast<ffi.Int32>(), len, low, high, seed);
  }
  return arr;
}

/// Draw random samples from a normal (Gaussian) distribution.
///
/// This function corresponds to NumPy's `random.normal` function.
///
/// **Box-Muller Optimization**:
/// It utilizes the mathematical Box-Muller transform to generate two independent
/// standard normal random scalars (`z0` and `z1`) from two uniform ones simultaneously,
/// cutting the costly transcendental CPU trig/log function calls by exactly 50%.
///
/// **Preconditions:**
/// - [scale] (standard deviation) must be strictly positive.
/// - [dtype] must be a floating point type (`float32` or `float64`).
///
/// **Throws:**
/// - [ArgumentError] if [dtype] is not a supported floating point type.
/// - [ArgumentError] if [scale] is less than or equal to 0.0.
///
/// **Performance considerations:**
/// - Algorithmic time complexity is $O(N)$ and space complexity is $O(N)$, where $N$ is the total size of
///   the generated array.
/// - Offloads element generation directly to high-speed C FFI vector functions (`v_normal_double` / `v_normal_float`),
///   combining Box-Muller math with native vector speed.
///
/// **Example:**
/// {@example /example/random_example.dart lang=dart}
///
/// Refer to the [Normal Distribution Reference](https://en.wikipedia.org/wiki/Normal_distribution)
/// for details on standard Gaussian distributions.
NDArray<double> normal(
  List<int> shape, {
  double loc = 0.0,
  double scale = 1.0,
  DType dtype = DType.float64,
  Random? random,
}) {
  if (dtype != DType.float32 && dtype != DType.float64) {
    throw ArgumentError(
      'normal only supports floating point dtypes (float32/float64)',
    );
  }
  if (scale <= 0.0) {
    throw ArgumentError(
      'scale (standard deviation) must be strictly positive (was $scale)',
    );
  }

  final arr = NDArray<double>.create(shape, dtype);
  final rand = random ?? Random();
  final len = arr.data.length;

  final seed = rand.nextInt(4294967296);
  if (dtype == DType.float64) {
    v_normal_double(arr.pointer.cast<ffi.Double>(), len, loc, scale, seed);
  } else {
    v_normal_float(arr.pointer.cast<ffi.Float>(), len, loc, scale, seed);
  }

  return arr;
}

/// Draw samples from an exponential distribution.
///
/// This function corresponds to NumPy's `random.exponential` function.
/// It uses Inverse Transform Sampling to extract exponential variables.
///
/// **Preconditions:**
/// - [scale] (the inverse of the rate parameter lambda, i.e., 1/lambda) must be strictly positive.
/// - [dtype] must be a floating point type (`float32` or `float64`).
///
/// **Throws:**
/// - [ArgumentError] if [dtype] is not a supported floating point type.
/// - [ArgumentError] if [scale] (or 1 / lam) is non-positive.
///
/// **Performance considerations:**
/// - Algorithmic time complexity is $O(N)$ and space complexity is $O(N)$, where $N$ is the total size of
///   the generated array.
///
/// **Example:**
/// {@example /example/random_example.dart lang=dart}
///
/// Refer to the [Exponential Distribution Reference](https://en.wikipedia.org/wiki/Exponential_distribution)
/// for details on exponential variables.
NDArray<double> exponential(
  List<int> shape, {
  double scale = 1.0,
  double? lam,
  DType dtype = DType.float64,
  Random? random,
}) {
  if (dtype != DType.float32 && dtype != DType.float64) {
    throw ArgumentError(
      'exponential only supports floating point dtypes (float32/float64)',
    );
  }
  final targetScale = lam != null ? 1.0 / lam : scale;
  if (targetScale <= 0.0) {
    throw ArgumentError(
      'scale parameter (or 1 / lam) must be strictly positive (was $targetScale)',
    );
  }

  final arr = NDArray<double>.create(shape, dtype);
  final rand = random ?? Random();

  for (var i = 0; i < arr.data.length; i++) {
    final u = rand.nextDouble();
    arr.data[i] = -targetScale * math.log(1.0 - u);
  }

  return arr;
}

/// Draw samples from a Poisson distribution.
///
/// This function corresponds to NumPy's `random.poisson` function.
///
/// **Dual-Track Algorithms**:
/// - For small lambda (`lam < 30.0`), it executes Knuth's precise inversion algorithm.
/// - For large lambda (`lam >= 30.0`), Knuth's method averages `lam` steps per element
///   and suffers severe numerical float underflow. To avoid stalls and underflows, it
///   automatically switches to high-speed **Gaussian Approximation** with continuity correction.
///
/// **Preconditions:**
/// - [lam] (lambda, the rate/mean) must be strictly positive.
/// - [dtype] must be an integer type (`int32` or `int64`).
///
/// **Throws:**
/// - [ArgumentError] if [dtype] is not a supported integer type.
/// - [ArgumentError] if [lam] is less than or equal to 0.0.
///
/// **Performance considerations:**
/// - Algorithmic time complexity is $O(N)$ and space complexity is $O(N)$, where $N$ is the total size of
///   the generated array.
/// - For small [lam] (< 30.0), Knuth's method iterates an average of `lam` times per element, making the runtime
///   dependent on the rate, whereas the Gaussian approximation runs in stable $O(1)$ steps per element.
///
/// **Example:**
/// {@example /example/random_example.dart lang=dart}
///
/// Refer to the [Poisson Distribution Reference](https://en.wikipedia.org/wiki/Poisson_distribution)
/// for details on Poisson processes.
NDArray<int> poisson(
  List<int> shape, {
  double lam = 1.0,
  DType dtype = DType.int64,
  Random? random,
}) {
  if (dtype != DType.int32 && dtype != DType.int64) {
    throw ArgumentError('poisson only supports integer dtypes (int32/int64)');
  }
  if (lam <= 0.0) {
    throw ArgumentError('lambda must be strictly positive (was $lam)');
  }

  final arr = NDArray<int>.create(shape, dtype);
  final rand = random ?? Random();

  if (lam < 30.0) {
    final limit = math.exp(-lam);
    for (var i = 0; i < arr.data.length; i++) {
      var k = 0;
      var p = 1.0;
      do {
        k++;
        p *= rand.nextDouble();
      } while (p > limit);
      arr.data[i] = k - 1;
    }
  } else {
    final stddev = math.sqrt(lam);
    final len = arr.data.length;
    var i = 0;
    while (i < len) {
      var u1 = rand.nextDouble();
      while (u1 == 0.0) {
        u1 = rand.nextDouble();
      }
      final u2 = rand.nextDouble();

      final mag = math.sqrt(-2.0 * math.log(u1));
      final angle = 2.0 * math.pi * u2;

      final z0 = mag * math.cos(angle);
      final z1 = mag * math.sin(angle);

      final val0 = (lam + stddev * z0).round();
      arr.data[i] = val0 < 0 ? 0 : val0;

      if (i + 1 < len) {
        final val1 = (lam + stddev * z1).round();
        arr.data[i + 1] = val1 < 0 ? 0 : val1;
      }
      i += 2;
    }
  }

  return arr;
}

/// Draw samples from a Binomial distribution.
///
/// This function corresponds to NumPy's `random.binomial` function.
///
/// **Dual-Track Algorithms**:
/// - For small `n < 50`, it directly simulates the Bernoulli trials (counts successes of [n]
///   independent random tests).
/// - For large `n >= 50`, counting `n` trials gets slow ($O(n)$). It triggers an optimized
///   **Normal Distribution Approximation** with mean `n*p` and standard deviation `sqrt(n*p*(1-p))`
///   for high-speed probabilistic simulations.
///
/// **Preconditions:**
/// - [n] (number of trials) must be non-negative.
/// - [p] (success probability) must be in the interval `[0.0, 1.0]`.
/// - [dtype] must be an integer type (`int32` or `int64`).
///
/// **Throws:**
/// - [ArgumentError] if [dtype] is not a supported integer type.
/// - [ArgumentError] if [n] is negative.
/// - [ArgumentError] if [p] is less than 0.0 or greater than 1.0.
///
/// **Performance considerations:**
/// - Algorithmic time complexity is $O(N)$ and space complexity is $O(N)$, where $N$ is the total size of
///   the generated array.
/// - For small [n] (< 50), Bernoulli simulation runs in $O(n)$ loops per element. For large [n], the Normal
///   distribution approximation executes in stable $O(1)$ steps per element, avoiding performance degradation.
///
/// **Example:**
/// {@example /example/random_example.dart lang=dart}
///
/// Refer to the [Binomial Distribution Reference](https://en.wikipedia.org/wiki/Binomial_distribution)
/// for details on independent Bernoulli trials.
NDArray<int> binomial(
  List<int> shape, {
  required int n,
  required double p,
  DType dtype = DType.int64,
  Random? random,
}) {
  if (dtype != DType.int32 && dtype != DType.int64) {
    throw ArgumentError('binomial only supports integer dtypes (int32/int64)');
  }
  if (n < 0) {
    throw ArgumentError('number of trials n must be non-negative (was $n)');
  }
  if (p < 0.0 || p > 1.0) {
    throw ArgumentError(
      'success probability p must be between 0.0 and 1.0 (was $p)',
    );
  }

  final arr = NDArray<int>.create(shape, dtype);
  final rand = random ?? Random();

  if (n == 0) {
    arr.data.fillRange(0, arr.data.length, 0);
    return arr;
  }

  if (n < 50) {
    for (var i = 0; i < arr.data.length; i++) {
      var successes = 0;
      for (var t = 0; t < n; t++) {
        if (rand.nextDouble() < p) {
          successes++;
        }
      }
      arr.data[i] = successes;
    }
  } else {
    final mean = n * p;
    final stddev = math.sqrt(n * p * (1.0 - p));
    final len = arr.data.length;

    if (stddev == 0.0) {
      arr.data.fillRange(0, len, mean.round());
    } else {
      var i = 0;
      while (i < len) {
        var u1 = rand.nextDouble();
        while (u1 == 0.0) {
          u1 = rand.nextDouble();
        }
        final u2 = rand.nextDouble();

        final mag = math.sqrt(-2.0 * math.log(u1));
        final angle = 2.0 * math.pi * u2;

        final z0 = mag * math.cos(angle);
        final z1 = mag * math.sin(angle);

        final val0 = (mean + stddev * z0).round();
        arr.data[i] = val0.clamp(0, n);

        if (i + 1 < len) {
          final val1 = (mean + stddev * z1).round();
          arr.data[i + 1] = val1.clamp(0, n);
        }
        i += 2;
      }
    }
  }

  return arr;
}

/// Draw random samples from a multivariate normal (Gaussian) distribution.
///
/// This corresponds to NumPy's `random.multivariate_normal` function.
///
/// **Mathematical Mechanics**:
/// The multivariate normal distribution is defined by a mean vector [mean] ($\mu$) of size $D$
/// and a symmetric, positive-definite covariance matrix [cov] ($\Sigma$) of size $D \times D$.
///
/// To draw a sample $X \sim \mathcal{N}(\mu, \Sigma)$:
/// 1. Computes the Cholesky factorization of the covariance matrix $\Sigma = L \cdot L^T$,
///    where $L$ is a lower triangular factor.
/// 2. Draws standard independent normal vectors $Z \sim \mathcal{N}(0, I)$ of size $D$.
/// 3. Returns the linearly transformed sample $X = \mu + Z \cdot L^T$ natively using
///    zero-copy BLAS matrix multiplication (`matmul()`) and broadcasted upcast addition (`add()`)!
///
/// **Preconditions:**
/// - [mean] must be a 1-dimensional vector of size $D$.
/// - [cov] must be a square 2-dimensional symmetric, positive-definite covariance matrix of size $D \times D$.
/// - If provided, [size] must be a valid shape list (e.g. `[N]`).
///
/// **Throws:**
/// - [ArgumentError] if [mean] is not 1D or [cov] is not 2D and square.
/// - [ArgumentError] if [mean] first dimension does not match [cov] dimensions.
/// - [ArgumentError] if [cov] is not symmetric positive-definite.
///
/// **Performance considerations:**
/// - Leverages high-speed LAPACK Cholesky solver and native CBLAS double/float matrix multiplication,
///   yielding spectacular compiled execution speeds.
///
/// **Example:**
/// ```dart
/// final mean = NDArray.fromList([1.0, 2.0], [2], DType.float64);
/// final cov = NDArray.fromList([1.0, 0.0, 0.0, 1.0], [2, 2], DType.float64);
/// final samples = multivariateNormal(mean, cov, size: [1000]);
/// print(samples.shape); // [1000, 2]
/// ```
NDArray multivariateNormal(
  NDArray mean,
  NDArray cov, {
  List<int>? size,
  Random? random,
}) {
  if (mean.shape.length != 1) {
    throw ArgumentError(
      'mean must be a 1-dimensional vector (was ${mean.shape})',
    );
  }
  if (cov.shape.length != 2 || cov.shape[0] != cov.shape[1]) {
    throw ArgumentError(
      'cov must be a 2-dimensional square matrix (was ${cov.shape})',
    );
  }
  final d = mean.shape[0];
  if (cov.shape[0] != d) {
    throw ArgumentError(
      'mean dimension ($d) must match cov dimensions (${cov.shape[0]}x${cov.shape[1]})',
    );
  }

  // 1. LAPACK Cholesky factorization: Sigma = L * L^T
  final choleskyFactors = cholesky(cov);
  final l = choleskyFactors['L']! as NDArray<double>;

  final sampleShape = <int>[];
  if (size != null) {
    sampleShape.addAll(size);
  }
  final sampleCount = sampleShape.isEmpty
      ? 1
      : sampleShape.reduce((a, b) => a * b);

  // 2. Draw independent standard normals Z
  final zShape = [...sampleShape, d];
  final z = normal(zShape, dtype: cov.dtype, random: random);

  // 3. Transform: X = Z * L^T + mean
  final lT = l.transpose();

  // We need to reshape or broadcast Z to 2D if sampleShape rank > 1
  final z2D = z.reshape([sampleCount, d]);
  final x2D = add(matmul(z2D, lT), mean);

  l.dispose();
  lT.dispose();
  z.dispose();
  z2D.dispose();

  // Reshape back to final output shape: [...size, d]
  final finalShape = [...sampleShape, d];
  return x2D.reshape(finalShape);
}
