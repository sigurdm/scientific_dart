import 'dart:math' as math;
import 'dart:math' show Random;
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'numdart_bindings.dart';
import 'ndarray.dart';
import 'operations.dart';

bool _listEquals(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

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
NDArray<T> uniform<T extends num>(
  List<int> shape, {
  DType<T>? dtype,
  int? seed,
  NDArray<T>? into,
  bool secure = false,
}) {
  final resolvedDType = dtype ?? (into?.dtype ?? DType.float64 as DType<T>);
  if (!identical(resolvedDType, DType.float32) &&
      !identical(resolvedDType, DType.float64)) {
    throw ArgumentError('uniform only supports float types for now');
  }
  if (into != null) {
    if (!_listEquals(into.shape, shape) || into.dtype != resolvedDType) {
      throw ArgumentError('Incompatible into buffer shape or dtype.');
    }
  }
  final arr = into ?? NDArray<T>.create(shape, resolvedDType);
  final len = arr.data.length;

  if (secure) {
    if (identical(resolvedDType, DType.float64)) {
      v_secure_uniform_double(arr.pointer.cast<ffi.Double>(), len);
    } else {
      v_secure_uniform_float(arr.pointer.cast<ffi.Float>(), len);
    }
    return arr;
  }

  final seedVal = seed ?? Random().nextInt(4294967296);

  if (identical(resolvedDType, DType.float64)) {
    v_uniform_double(arr.pointer.cast<ffi.Double>(), len, seedVal);
  } else {
    v_uniform_float(arr.pointer.cast<ffi.Float>(), len, seedVal);
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
NDArray<T> randint<T extends num>(
  List<int> shape, {
  required int low,
  required int high,
  DType<T>? dtype,
  int? seed,
  NDArray<T>? into,
  bool secure = false,
}) {
  final resolvedDType = dtype ?? (into?.dtype ?? DType.int64 as DType<T>);
  if (!identical(resolvedDType, DType.int32) &&
      !identical(resolvedDType, DType.int64)) {
    throw ArgumentError('randint only supports integer types');
  }
  if (low >= high) {
    throw ArgumentError('low must be less than high');
  }
  if (into != null) {
    if (!_listEquals(into.shape, shape) || into.dtype != resolvedDType) {
      throw ArgumentError('Incompatible into buffer shape or dtype.');
    }
  }
  final arr = into ?? NDArray<T>.create(shape, resolvedDType);
  final len = arr.data.length;

  if (secure) {
    if (identical(resolvedDType, DType.int64)) {
      v_secure_randint_int64(arr.pointer.cast<ffi.Int64>(), len, low, high);
    } else {
      v_secure_randint_int32(arr.pointer.cast<ffi.Int32>(), len, low, high);
    }
    return arr;
  }

  final seedVal = seed ?? Random().nextInt(4294967296);

  if (identical(resolvedDType, DType.int64)) {
    v_randint_int64(arr.pointer.cast<ffi.Int64>(), len, low, high, seedVal);
  } else {
    v_randint_int32(arr.pointer.cast<ffi.Int32>(), len, low, high, seedVal);
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
NDArray<T> normal<T extends num>(
  List<int> shape, {
  double loc = 0.0,
  double scale = 1.0,
  DType<T>? dtype,
  int? seed,
  NDArray<T>? into,
  bool secure = false,
}) {
  final resolvedDType = dtype ?? (into?.dtype ?? DType.float64 as DType<T>);
  if (!identical(resolvedDType, DType.float32) &&
      !identical(resolvedDType, DType.float64)) {
    throw ArgumentError(
      'normal only supports floating point dtypes (float32/float64)',
    );
  }
  if (scale <= 0.0) {
    throw ArgumentError(
      'scale (standard deviation) must be strictly positive (was $scale)',
    );
  }

  if (into != null) {
    if (!_listEquals(into.shape, shape) || into.dtype != resolvedDType) {
      throw ArgumentError('Incompatible into buffer shape or dtype.');
    }
  }
  final arr = into ?? NDArray<T>.create(shape, resolvedDType);
  final len = arr.data.length;

  if (secure) {
    if (identical(resolvedDType, DType.float64)) {
      v_secure_normal_double(arr.pointer.cast<ffi.Double>(), len, loc, scale);
    } else {
      v_secure_normal_float(arr.pointer.cast<ffi.Float>(), len, loc, scale);
    }
    return arr;
  }

  final seedVal = seed ?? Random().nextInt(4294967296);

  if (identical(resolvedDType, DType.float64)) {
    v_normal_double(arr.pointer.cast<ffi.Double>(), len, loc, scale, seedVal);
  } else {
    v_normal_float(arr.pointer.cast<ffi.Float>(), len, loc, scale, seedVal);
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
NDArray<T> exponential<T extends num>(
  List<int> shape, {
  double scale = 1.0,
  double? lam,
  DType<T>? dtype,
  int? seed,
  NDArray<T>? into,
  bool secure = false,
}) {
  final resolvedDType = dtype ?? (into?.dtype ?? DType.float64 as DType<T>);
  if (!identical(resolvedDType, DType.float32) &&
      !identical(resolvedDType, DType.float64)) {
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

  if (into != null) {
    if (!_listEquals(into.shape, shape) || into.dtype != resolvedDType) {
      throw ArgumentError('Incompatible into buffer shape or dtype.');
    }
  }
  final arr = into ?? NDArray<T>.create(shape, resolvedDType);
  final len = arr.data.length;

  if (secure) {
    if (identical(resolvedDType, DType.float64)) {
      v_secure_uniform_double(arr.pointer.cast<ffi.Double>(), len);
      final data = arr.data as Float64List;
      for (var i = 0; i < len; i++) {
        var u = data[i];
        if (u >= 1.0) u = 0.9999999999999999;
        data[i] = -targetScale * math.log(1.0 - u);
      }
    } else {
      v_secure_uniform_float(arr.pointer.cast<ffi.Float>(), len);
      final data = arr.data as Float32List;
      for (var i = 0; i < len; i++) {
        var u = data[i];
        if (u >= 1.0) u = 0.999999;
        data[i] = -targetScale * math.log(1.0 - u);
      }
    }
    return arr;
  }

  final seedVal = seed ?? Random().nextInt(4294967296);

  if (identical(resolvedDType, DType.float64)) {
    v_uniform_double(arr.pointer.cast<ffi.Double>(), len, seedVal);
    final data = arr.data as Float64List;
    for (var i = 0; i < len; i++) {
      var u = data[i];
      if (u >= 1.0) u = 0.9999999999999999;
      data[i] = -targetScale * math.log(1.0 - u);
    }
  } else {
    v_uniform_float(arr.pointer.cast<ffi.Float>(), len, seedVal);
    final data = arr.data as Float32List;
    for (var i = 0; i < len; i++) {
      var u = data[i];
      if (u >= 1.0) u = 0.999999;
      data[i] = -targetScale * math.log(1.0 - u);
    }
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
NDArray<T> poisson<T extends num>(
  List<int> shape, {
  double lam = 1.0,
  DType<T>? dtype,
  int? seed,
  NDArray<T>? into,
  bool secure = false,
}) {
  final resolvedDType = dtype ?? (into?.dtype ?? DType.int64 as DType<T>);
  if (!identical(resolvedDType, DType.int32) &&
      !identical(resolvedDType, DType.int64)) {
    throw ArgumentError('poisson only supports integer dtypes (int32/int64)');
  }
  if (lam <= 0.0) {
    throw ArgumentError('lambda must be strictly positive (was $lam)');
  }

  if (into != null) {
    if (!_listEquals(into.shape, shape) || into.dtype != resolvedDType) {
      throw ArgumentError('Incompatible into buffer shape or dtype.');
    }
  }
  final arr = into ?? NDArray<T>.create(shape, resolvedDType);
  final len = arr.data.length;

  final seedVal = secure
      ? Random.secure().nextInt(4294967296)
      : (seed ?? Random().nextInt(4294967296));

  if (identical(resolvedDType, DType.int64)) {
    v_poisson_int64(arr.pointer.cast<ffi.Int64>(), len, lam, seedVal);
  } else {
    v_poisson_int32(arr.pointer.cast<ffi.Int32>(), len, lam, seedVal);
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
NDArray<T> binomial<T extends num>(
  List<int> shape, {
  required int n,
  required double p,
  DType<T>? dtype,
  int? seed,
  NDArray<T>? into,
  bool secure = false,
}) {
  final resolvedDType = dtype ?? (into?.dtype ?? DType.int64 as DType<T>);
  if (!identical(resolvedDType, DType.int32) &&
      !identical(resolvedDType, DType.int64)) {
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

  if (into != null) {
    if (!_listEquals(into.shape, shape) || into.dtype != resolvedDType) {
      throw ArgumentError('Incompatible into buffer shape or dtype.');
    }
  }
  final arr = into ?? NDArray<T>.create(shape, resolvedDType);
  final len = arr.data.length;

  final seedVal = secure
      ? Random.secure().nextInt(4294967296)
      : (seed ?? Random().nextInt(4294967296));

  if (identical(resolvedDType, DType.int64)) {
    v_binomial_int64(arr.pointer.cast<ffi.Int64>(), len, n, p, seedVal);
  } else {
    v_binomial_int32(arr.pointer.cast<ffi.Int32>(), len, n, p, seedVal);
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
NDArray<T> multivariateNormal<T extends num>(
  NDArray mean,
  NDArray cov, {
  List<int>? size,
  DType<T>? dtype,
  int? seed,
  NDArray<T>? into,
  bool secure = false,
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

  final resolvedDType = dtype ?? (into?.dtype ?? DType.float64 as DType<T>);
  if (!identical(resolvedDType, DType.float32) &&
      !identical(resolvedDType, DType.float64)) {
    throw ArgumentError(
      'multivariateNormal only supports floating point dtypes (float32/float64)',
    );
  }

  final sampleShape = <int>[];
  if (size != null) {
    sampleShape.addAll(size);
  }
  final finalShape = [...sampleShape, d];
  if (into != null) {
    if (!_listEquals(into.shape, finalShape) || into.dtype != resolvedDType) {
      throw ArgumentError('Incompatible into buffer shape or dtype.');
    }
  }

  return NDArray.scope(() {
    final choleskyFactors = cholesky(cov);
    final l = choleskyFactors['L']!;

    final sampleShape = <int>[];
    if (size != null) {
      sampleShape.addAll(size);
    }
    final sampleCount = sampleShape.isEmpty
        ? 1
        : sampleShape.reduce((a, b) => a * b);

    final zShape = [...sampleShape, d];
    final z = normal(zShape, dtype: resolvedDType, seed: seed, secure: secure);
    final lT = l.transpose();

    final z2D = z.reshape([sampleCount, d]);
    final x2D =
        into?.reshape([sampleCount, d]) ??
        NDArray<T>.create([sampleCount, d], resolvedDType);
    add(matmul(z2D, lT), mean, out: x2D);

    if (into != null) {
      return into;
    }
    final result = x2D.reshape(finalShape);
    return result.detachToParentScope();
  });
}

/// Draw samples from a multinomial distribution.
///
/// This corresponds to NumPy's `random.multinomial` function.
///
/// **Mathematical Mechanics**:
/// The multinomial distribution is a generalization of the binomial distribution.
/// A trial has $K$ possible categorical outcomes, each with a probability $p_j$ specified in [pvals].
///
/// To draw a sample of shape `[...size, K]`:
/// 1. Computes the cumulative probability distribution (CDF) of [pvals].
/// 2. For each output coordinate block, performs [n] trials:
///    - Draws a standard uniform variable $U \sim \mathcal{U}(0, 1)$.
///    - Uses binary/linear search to find the first outcome index $j$ where $U \le \text{CDF}[j]$.
///    - Increments the count of category $j$ for that sample.
///
/// **Preconditions:**
/// - [n] must be strictly non-negative ($\ge 0$).
/// - [pvals] must be a 1-dimensional vector of probabilities. The probabilities must sum to approximately 1.0.
/// - If provided, [size] must be a valid shape list.
///
/// **Throws:**
/// - [ArgumentError] if [n] is negative, or if [pvals] is not a 1D vector.
/// - [ArgumentError] if [pvals] contains negative probabilities, or if their sum exceeds 1.0 by a significant tolerance.
///
/// **Example:**
/// ```dart
/// final pvals = NDArray.fromList([0.2, 0.5, 0.3], [3], DType.float64);
/// final samples = multinomial(10, pvals, size: [1000]);
/// print(samples.shape); // [1000, 3]
/// ```
NDArray<T> multinomial<T extends num>(
  int n,
  NDArray pvals, {
  List<int>? size,
  DType<T>? dtype,
  int? seed,
  NDArray<T>? into,
  bool secure = false,
}) {
  if (n < 0) {
    throw ArgumentError('n trials must be non-negative (was $n)');
  }
  if (pvals.shape.length != 1) {
    throw ArgumentError(
      'pvals must be a 1-dimensional probability vector (was ${pvals.shape})',
    );
  }

  final resolvedDType = dtype ?? (into?.dtype ?? DType.int32 as DType<T>);
  if (!identical(resolvedDType, DType.int32) &&
      !identical(resolvedDType, DType.int64)) {
    throw ArgumentError(
      'multinomial only supports integer dtypes (int32/int64)',
    );
  }

  final k = pvals.shape[0];
  final rand = secure ? Random.secure() : Random();

  final cdf = List<double>.filled(k, 0.0);
  var sumP = 0.0;
  for (var i = 0; i < k; i++) {
    final p = (pvals.data[i] as num).toDouble();
    if (p < 0.0) {
      throw ArgumentError(
        'pvals must contain non-negative probabilities (was $p at index $i)',
      );
    }
    sumP += p;
    cdf[i] = sumP;
  }

  if ((sumP - 1.0).abs() > 1e-3) {
    for (var i = 0; i < k; i++) {
      cdf[i] /= sumP;
    }
  }

  final sampleShape = <int>[];
  if (size != null) {
    sampleShape.addAll(size);
  }
  final sampleCount = sampleShape.isEmpty
      ? 1
      : sampleShape.reduce((a, b) => a * b);

  final finalShape = [...sampleShape, k];
  if (into != null) {
    if (!_listEquals(into.shape, finalShape) || into.dtype != resolvedDType) {
      throw ArgumentError('Incompatible into buffer shape or dtype.');
    }
  }
  final result =
      into ?? NDArray<T>.create(finalShape, resolvedDType, zeroInit: true);
  if (into != null) {
    result.fill(0);
  }

  for (var s = 0; s < sampleCount; s++) {
    final offset = s * k;
    for (var t = 0; t < n; t++) {
      final u = rand.nextDouble();
      var outcome = k - 1;
      for (var j = 0; j < k; j++) {
        if (u <= cdf[j]) {
          outcome = j;
          break;
        }
      }
      result.data[offset + outcome] =
          ((result.data[offset + outcome] as num) + 1) as T;
    }
  }

  return result;
}
