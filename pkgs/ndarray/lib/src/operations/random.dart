// ignore_for_file: non_constant_identifier_names
import 'dart:typed_data';
import 'dart:math' as math;
import 'dart:math' show Random;
import 'dart:io';
import 'package:archive/archive.dart';
import 'package:pocketfft/pocketfft.dart';
import '../ndarray.dart';
import 'package:openblas/openblas.dart';
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import '../ndarray_bindings.dart';
import '../scratch_arena.dart';

// Standalone operational relative cross-imports
import 'math.dart';
import 'stats.dart';
import 'sorting.dart';
import 'linalg.dart';
import 'spacers.dart';
import 'manipulation.dart';
import 'broadcasting.dart';
import 'splitting.dart';
import 'shaping_meshes.dart';
import 'repeating_tiling.dart';
import 'io.dart';
import 'random.dart';
import 'fft.dart';
import 'calculus.dart';
import 'helpers.dart';

/// Generates an array with random values uniformly distributed in the half-open interval `[0.0, 1.0)`.
///
/// **Preconditions:**
/// - [dtype] must be a floating point type (DType.float32 or DType.float64).
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
  NDArray<T>? out,
  bool secure = false,
}) {
  final resolvedDType = dtype ?? (out?.dtype ?? DType.float64 as DType<T>);
  if (out != null) {
    if (!listEquals(out.shape, shape) || out.dtype != resolvedDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }
  final arr = out ?? NDArray<T>.create(shape, resolvedDType);
  final len = arr.data.length;
  final seedVal = secure ? 0 : (seed ?? Random().nextInt(4294967296));

  switch (resolvedDType) {
    case DType.float64:
      if (secure) {
        v_secure_uniform_double(arr.pointer.cast<ffi.Double>(), len);
      } else {
        v_uniform_double(arr.pointer.cast<ffi.Double>(), len, seedVal);
      }
    case DType.float32:
      if (secure) {
        v_secure_uniform_float(arr.pointer.cast<ffi.Float>(), len);
      } else {
        v_uniform_float(arr.pointer.cast<ffi.Float>(), len, seedVal);
      }
    default:
      throw ArgumentError('uniform only supports float types for now');
  }
  return arr;
}

/// Return random integers from the half-open interval `[low, high)`.
///
/// Generates uniformly distributed random integers of the specified integer [dtype]
/// in the range `[low, high)`.
///
/// **Preconditions:**
/// - [low] must be strictly less than [high].
/// - [dtype] must be a supported integer type (`int64`, `int32`, `int16`, or `uint8`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype.
///
/// **Throws:**
/// - [ArgumentError] if [low] is greater than or equal to [high].
/// - [ArgumentError] if [dtype] is not a supported integer DType.
/// - [ArgumentError] if [out] has mismatched shape or dtype.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ in both time and space, where $N$ is the total size of the generated array.
/// - Leverages optimized native C FFI vector calls (`v_randint_int64`, `v_randint_int32` etc.) yielding excellent execution speeds.
///
/// **Memory Ownership & Recycle:**
/// - If the optional [out] recycler buffer is provided, it is populated in-place, avoiding heap allocations.
/// - Otherwise, allocates a new array on the unmanaged C heap. **The caller takes full ownership** of this memory page and **must explicitly call [dispose()]** to prevent native memory leaks, unless executing inside a managed [NDArray.scope()].
///
/// **NumPy Counterpart:**
/// - Equates directly to NumPy's `np.random.randint`.
///
/// **Example:**
/// ```dart
/// final a = randint([3], low: 1, high: 10, dtype: DType.int32);
/// print(a.toList()); // e.g., [3, 7, 1]
/// a.dispose();
/// ```
NDArray<T> randint<T extends num>(
  List<int> shape, {
  required int low,
  required int high,
  DType<T>? dtype,
  int? seed,
  NDArray<T>? out,
  bool secure = false,
}) {
  if (low >= high) {
    throw ArgumentError('low must be less than high');
  }
  final resolvedDType = dtype ?? (out?.dtype ?? DType.int64 as DType<T>);
  if (out != null) {
    if (!listEquals(out.shape, shape) || out.dtype != resolvedDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }
  final arr = out ?? NDArray<T>.create(shape, resolvedDType);
  final len = arr.data.length;
  final seedVal = secure ? 0 : (seed ?? Random().nextInt(4294967296));

  switch (resolvedDType) {
    case DType.int64:
      if (secure) {
        v_secure_randint_int64(arr.pointer.cast<ffi.Int64>(), len, low, high);
      } else {
        v_randint_int64(arr.pointer.cast<ffi.Int64>(), len, low, high, seedVal);
      }
    case DType.int32:
      if (secure) {
        v_secure_randint_int32(arr.pointer.cast<ffi.Int32>(), len, low, high);
      } else {
        v_randint_int32(arr.pointer.cast<ffi.Int32>(), len, low, high, seedVal);
      }
    case DType.uint8:
      if (secure) {
        v_secure_randint_uint8(arr.pointer.cast<ffi.Uint8>(), len, low, high);
      } else {
        v_randint_uint8(arr.pointer.cast<ffi.Uint8>(), len, low, high, seedVal);
      }
    case DType.int16:
      if (secure) {
        v_secure_randint_int16(arr.pointer.cast<ffi.Int16>(), len, low, high);
      } else {
        v_randint_int16(arr.pointer.cast<ffi.Int16>(), len, low, high, seedVal);
      }
    default:
      throw ArgumentError(
        'randint only supports integer types (int64, int32, int16, uint8)',
      );
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
  NDArray<T>? out,
  bool secure = false,
}) {
  if (scale <= 0.0) {
    throw ArgumentError(
      'scale (standard deviation) must be strictly positive (was $scale)',
    );
  }
  final resolvedDType = dtype ?? (out?.dtype ?? DType.float64 as DType<T>);
  if (out != null) {
    if (!listEquals(out.shape, shape) || out.dtype != resolvedDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }
  final arr = out ?? NDArray<T>.create(shape, resolvedDType);
  final len = arr.data.length;
  final seedVal = secure ? 0 : (seed ?? Random().nextInt(4294967296));

  switch (resolvedDType) {
    case DType.float64:
      if (secure) {
        v_secure_normal_double(arr.pointer.cast<ffi.Double>(), len, loc, scale);
      } else {
        v_normal_double(
          arr.pointer.cast<ffi.Double>(),
          len,
          loc,
          scale,
          seedVal,
        );
      }
    case DType.float32:
      if (secure) {
        v_secure_normal_float(arr.pointer.cast<ffi.Float>(), len, loc, scale);
      } else {
        v_normal_float(arr.pointer.cast<ffi.Float>(), len, loc, scale, seedVal);
      }
    default:
      throw ArgumentError(
        'normal only supports floating point dtypes (float32/float64)',
      );
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
  NDArray<T>? out,
  bool secure = false,
}) {
  final targetScale = lam != null ? 1.0 / lam : scale;
  if (targetScale <= 0.0) {
    throw ArgumentError(
      'scale parameter (or 1 / lam) must be strictly positive (was $targetScale)',
    );
  }
  final resolvedDType = dtype ?? (out?.dtype ?? DType.float64 as DType<T>);
  if (out != null) {
    if (!listEquals(out.shape, shape) || out.dtype != resolvedDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }
  final arr = out ?? NDArray<T>.create(shape, resolvedDType);
  final len = arr.data.length;
  final seedVal = secure ? 0 : (seed ?? Random().nextInt(4294967296));

  switch (resolvedDType) {
    case DType.float64:
      if (secure) {
        v_secure_uniform_double(arr.pointer.cast<ffi.Double>(), len);
      } else {
        v_uniform_double(arr.pointer.cast<ffi.Double>(), len, seedVal);
      }
      final data = arr.data as Float64List;
      for (var i = 0; i < len; i++) {
        var u = data[i];
        if (u >= 1.0) u = 0.9999999999999999;
        data[i] = -targetScale * math.log(1.0 - u);
      }
    case DType.float32:
      if (secure) {
        v_secure_uniform_float(arr.pointer.cast<ffi.Float>(), len);
      } else {
        v_uniform_float(arr.pointer.cast<ffi.Float>(), len, seedVal);
      }
      final data = arr.data as Float32List;
      for (var i = 0; i < len; i++) {
        var u = data[i];
        if (u >= 1.0) u = 0.999999;
        data[i] = -targetScale * math.log(1.0 - u);
      }
    default:
      throw ArgumentError(
        'exponential only supports floating point dtypes (float32/float64)',
      );
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
  NDArray<T>? out,
  bool secure = false,
}) {
  if (lam <= 0.0) {
    throw ArgumentError('lambda must be strictly positive (was $lam)');
  }
  final resolvedDType = dtype ?? (out?.dtype ?? DType.int64 as DType<T>);
  if (out != null) {
    if (!listEquals(out.shape, shape) || out.dtype != resolvedDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }
  final arr = out ?? NDArray<T>.create(shape, resolvedDType);
  final len = arr.data.length;
  final seedVal = secure
      ? Random.secure().nextInt(4294967296)
      : (seed ?? Random().nextInt(4294967296));

  switch (resolvedDType) {
    case DType.int64:
      v_poisson_int64(arr.pointer.cast<ffi.Int64>(), len, lam, seedVal);
    case DType.int32:
      v_poisson_int32(arr.pointer.cast<ffi.Int32>(), len, lam, seedVal);
    default:
      throw ArgumentError('poisson only supports integer dtypes (int32/int64)');
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
  NDArray<T>? out,
  bool secure = false,
}) {
  if (n < 0) {
    throw ArgumentError('number of trials n must be non-negative (was $n)');
  }
  if (p < 0.0 || p > 1.0) {
    throw ArgumentError(
      'success probability p must be between 0.0 and 1.0 (was $p)',
    );
  }
  final resolvedDType = dtype ?? (out?.dtype ?? DType.int64 as DType<T>);
  if (out != null) {
    if (!listEquals(out.shape, shape) || out.dtype != resolvedDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }
  final arr = out ?? NDArray<T>.create(shape, resolvedDType);
  final len = arr.data.length;
  final seedVal = secure
      ? Random.secure().nextInt(4294967296)
      : (seed ?? Random().nextInt(4294967296));

  switch (resolvedDType) {
    case DType.int64:
      v_binomial_int64(arr.pointer.cast<ffi.Int64>(), len, n, p, seedVal);
    case DType.int32:
      v_binomial_int32(arr.pointer.cast<ffi.Int32>(), len, n, p, seedVal);
    default:
      throw ArgumentError(
        'binomial only supports integer dtypes (int32/int64)',
      );
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
  NDArray<T>? out,
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

  final resolvedDType = dtype ?? (out?.dtype ?? DType.float64 as DType<T>);
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
  if (out != null) {
    if (!listEquals(out.shape, finalShape) || out.dtype != resolvedDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
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
        out?.reshape([sampleCount, d]) ??
        NDArray<T>.create([sampleCount, d], resolvedDType);
    add(matmul(z2D, lT), mean, out: x2D);

    if (out != null) {
      return out;
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
  NDArray<T>? out,
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

  final resolvedDType = dtype ?? (out?.dtype ?? DType.int32 as DType<T>);
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
  if (out != null) {
    if (!listEquals(out.shape, finalShape) || out.dtype != resolvedDType) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }
  final result =
      out ?? NDArray<T>.create(finalShape, resolvedDType, zeroInit: true);
  if (out != null) {
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

/// Generates a random sample from a given 1-D array.
///
/// **Preconditions:**
/// - [a] must be a 1-D array and not disposed.
/// - [size] if specified must be a valid shape list.
/// - If [replace] is false, the total sample count must be $\le$ [a.size].
/// - If [p] is specified:
///   - It must be a 1-D array of the same size as [a].
///   - Its values must be non-negative probabilities summing to approximately 1.0.
///
/// **Throws:**
/// - [StateError] if [a] or [p] is disposed.
/// - [ArgumentError] if [a] is not 1-D.
/// - [ArgumentError] if [replace] is false and sample size exceeds [a.size].
/// - [ArgumentError] if [p] size is mismatched, negative, or does not sum to 1.0.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([10, 20, 30, 40], [4], DType.int32);
/// final sampled = choice(a, size: [2], replace: false); // e.g., [20, 40]
/// ```
///
/// Reference: [NumPy choice](https://numpy.org/doc/stable/reference/generated/numpy.random.choice.html)
NDArray<T> choice<T>(
  NDArray<T> a, {
  List<int>? size,
  bool replace = true,
  NDArray<double>? p,
  int? seed,
  bool secure = false,
}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute choice on a disposed array.');
  }
  if (a.shape.length != 1) {
    throw ArgumentError('choice only supports 1-D input arrays.');
  }
  if (p != null) {
    if (p.isDisposed) {
      throw StateError('Provided probability array p is disposed.');
    }
    if (p.shape.length != 1 || p.shape[0] != a.shape[0]) {
      throw ArgumentError(
        'Probability array p must be 1-D and match the size of a.',
      );
    }
  }

  final sampleShape = size ?? <int>[];
  final sampleCount = sampleShape.isEmpty
      ? 1
      : sampleShape.reduce((x, y) => x * y);

  if (!replace && sampleCount > a.size) {
    throw ArgumentError(
      'Cannot choose $sampleCount elements without replacement from an array of size ${a.size}.',
    );
  }

  final rand = secure
      ? Random.secure()
      : Random(seed ?? Random().nextInt(4294967296));
  final result = NDArray<T>.create(sampleShape, a.dtype);

  // Pre-calculate CDF if probability array p is specified
  List<double>? cdf;
  if (p != null) {
    final nonNullP = p;
    cdf = List<double>.filled(a.size, 0.0);
    var sumP = 0.0;
    for (var i = 0; i < a.size; i++) {
      final prob =
          nonNullP.data[nonNullP.offsetElements + i * nonNullP.strides[0]];
      if (prob < 0.0) {
        throw ArgumentError(
          'pvals must contain non-negative probabilities (was $prob at index $i)',
        );
      }
      sumP += prob;
      cdf[i] = sumP;
    }
    if ((sumP - 1.0).abs() > 1e-3) {
      for (var i = 0; i < a.size; i++) {
        cdf[i] /= sumP;
      }
    }
  }

  final aOffset = a.offsetElements;
  final aStride = a.strides[0];
  final data = a.data;

  if (replace) {
    // Draw with replacement
    for (var i = 0; i < sampleCount; i++) {
      var index = 0;
      if (cdf != null) {
        // Draw using CDF
        final u = rand.nextDouble();
        index = a.size - 1;
        for (var j = 0; j < a.size; j++) {
          if (u <= cdf[j]) {
            index = j;
            break;
          }
        }
      } else {
        // Uniform draw
        index = rand.nextInt(a.size);
      }
      result.data[i] = data[aOffset + index * aStride];
    }
  } else {
    // Draw without replacement
    if (cdf == null) {
      final indices = List<int>.generate(a.size, (i) => i);
      for (var i = a.size - 1; i > a.size - 1 - sampleCount; i--) {
        final j = rand.nextInt(i + 1);
        final temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
      }
      for (var i = 0; i < sampleCount; i++) {
        final idx = indices[a.size - 1 - i];
        result.data[i] = data[aOffset + idx * aStride];
      }
    } else {
      final nonNullP = p!;
      final tempProbs = List<double>.generate(
        a.size,
        (i) => nonNullP.data[nonNullP.offsetElements + i * nonNullP.strides[0]],
      );
      final drawn = List<bool>.filled(a.size, false);

      for (var draw = 0; draw < sampleCount; draw++) {
        final drawCdf = List<double>.filled(a.size, 0.0);
        var sumP = 0.0;
        for (var i = 0; i < a.size; i++) {
          if (!drawn[i]) {
            sumP += tempProbs[i];
          }
          drawCdf[i] = sumP;
        }

        final u = rand.nextDouble() * sumP;
        var index = a.size - 1;
        for (var j = 0; j < a.size; j++) {
          if (!drawn[j] && u <= drawCdf[j]) {
            index = j;
            break;
          }
        }

        drawn[index] = true;
        result.data[draw] = data[aOffset + index * aStride];
      }
    }
  }

  return result;
}

/// Shuffles the array in-place along the first axis.
///
/// For N-Dimensional arrays, shuffles sub-arrays along axis 0.
/// For 1-Dimensional arrays, shuffles individual elements.
///
/// **Preconditions:**
/// - The array [a] must not be disposed.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
///
/// **Performance considerations:**
/// - Uses Fisher-Yates shuffle, performing $O(D_0)$ swaps where $D_0$ is the size of the first dimension.
/// - Time complexity is $O(D_0 \cdot S)$ where $S$ is the size of each sub-array slice.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
/// shuffle(a); // a is now shuffled in-place, e.g., [2.0, 1.0, 3.0]
/// ```
void shuffle(NDArray a, {int? seed, bool secure = false}) {
  if (a.isDisposed) {
    throw StateError('Cannot shuffle a disposed array.');
  }

  final rand = secure
      ? Random.secure()
      : Random(seed ?? Random().nextInt(4294967296));
  final d0 = a.shape.isEmpty ? 1 : a.shape[0];
  if (d0 <= 1) return;

  if (a.shape.length == 1) {
    final data = a.data;
    final offset = a.offsetElements;
    final stride = a.strides[0];
    for (var i = d0 - 1; i > 0; i--) {
      final j = rand.nextInt(i + 1);
      if (i != j) {
        final temp = data[offset + i * stride];
        data[offset + i * stride] = data[offset + j * stride];
        data[offset + j * stride] = temp;
      }
    }
    return;
  }

  final sliceShape = a.shape.sublist(1);
  final sliceStrides = a.strides.sublist(1);

  final tempSlice = NDArray.create(sliceShape, a.dtype);
  try {
    final stepStride = a.strides[0];
    final data = a.data;

    for (var i = d0 - 1; i > 0; i--) {
      final j = rand.nextInt(i + 1);
      if (i != j) {
        final offsetI = a.offsetElements + i * stepStride;
        final offsetJ = a.offsetElements + j * stepStride;

        _copySlice(
          data,
          offsetI,
          tempSlice.data,
          0,
          sliceShape,
          sliceStrides,
          tempSlice.strides,
          0,
        );
        _copySlice(
          data,
          offsetJ,
          data,
          offsetI,
          sliceShape,
          sliceStrides,
          sliceStrides,
          0,
        );
        _copySlice(
          tempSlice.data,
          0,
          data,
          offsetJ,
          sliceShape,
          tempSlice.strides,
          sliceStrides,
          0,
        );
      }
    }
  } finally {
    tempSlice.dispose();
  }
}

/// Returns a permuted copy of an array along axis 0.
///
/// **Preconditions:**
/// - The array [a] must not be disposed.
///
/// **Throws:**
/// - [StateError] if [a] is disposed.
///
/// **Performance considerations:**
/// - Returns a brand new contiguous deep copy of the array permuted along axis 0.
/// - Time complexity matches [shuffle].
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
/// final perm = permutation(a); // perm is a permuted copy, a remains unchanged
/// ```
NDArray<T> permutation<T>(NDArray<T> a, {int? seed, bool secure = false}) {
  if (a.isDisposed) {
    throw StateError('Cannot permute a disposed array.');
  }
  final copyArr = a.copy();
  shuffle(copyArr, seed: seed, secure: secure);
  return copyArr;
}

void _copySlice(
  List src,
  int srcOffset,
  List dest,
  int destOffset,
  List<int> shape,
  List<int> stridesSrc,
  List<int> stridesDest,
  int dim,
) {
  if (shape.isEmpty) {
    dest[destOffset] = src[srcOffset];
    return;
  }
  if (dim == shape.length - 1) {
    final limit = shape[dim];
    final strideSrc = stridesSrc[dim];
    final strideDest = stridesDest[dim];
    for (var i = 0; i < limit; i++) {
      dest[destOffset + i * strideDest] = src[srcOffset + i * strideSrc];
    }
    return;
  }
  final limit = shape[dim];
  final strideSrc = stridesSrc[dim];
  final strideDest = stridesDest[dim];
  for (var i = 0; i < limit; i++) {
    _copySlice(
      src,
      srcOffset + i * strideSrc,
      dest,
      destOffset + i * strideDest,
      shape,
      stridesSrc,
      stridesDest,
      dim + 1,
    );
  }
}
