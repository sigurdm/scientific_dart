// ignore_for_file: non_constant_identifier_names
import 'dart:math' as math;
import '../dtype.dart';
import '../gpu_array.dart';
import '../device.dart';
import '../backend/compute_engine.dart';

/// Counter-based 4x32-10 Philox generator.
final class Philox4x32Engine {
  static const int _philoxM4x32A = 0xD2511F53;
  static const int _philoxM4x32B = 0xCD9E8D57;
  static const int _philoxW32A = 0x9E3779B9;
  static const int _philoxW32B = 0xBB67AE85;

  static List<int> philox4x32TenRounds(List<int> ctr, List<int> key) {
    var c0 = ctr[0] & 0xFFFFFFFF;
    var c1 = ctr[1] & 0xFFFFFFFF;
    var c2 = ctr[2] & 0xFFFFFFFF;
    var c3 = ctr[3] & 0xFFFFFFFF;

    var k0 = key[0] & 0xFFFFFFFF;
    var k1 = key[1] & 0xFFFFFFFF;

    for (var r = 0; r < 10; r++) {
      // 32-bit high and low product
      final p0 = (c0 * _philoxM4x32A);
      final hi0 = (p0 >> 32) & 0xFFFFFFFF;
      final lo0 = p0 & 0xFFFFFFFF;

      final p2 = (c2 * _philoxM4x32B);
      final hi2 = (p2 >> 32) & 0xFFFFFFFF;
      final lo2 = p2 & 0xFFFFFFFF;

      final nextC0 = (hi2 ^ c1 ^ k0) & 0xFFFFFFFF;
      final nextC1 = lo2;
      final nextC2 = (hi0 ^ c3 ^ k1) & 0xFFFFFFFF;
      final nextC3 = lo0;

      c0 = nextC0;
      c1 = nextC1;
      c2 = nextC2;
      c3 = nextC3;

      k0 = (k0 + _philoxW32A) & 0xFFFFFFFF;
      k1 = (k1 + _philoxW32B) & 0xFFFFFFFF;
    }

    return [c0, c1, c2, c3];
  }
}

/// Encapsulates on-device random generation state.
final class RandomState {
  int _seed;
  int _counter = 0;
  math.Random _fallbackRng;

  RandomState([int? seed])
      : _seed = seed ?? DateTime.now().microsecondsSinceEpoch,
        _fallbackRng = math.Random(seed ?? DateTime.now().microsecondsSinceEpoch);

  /// Seeds this random state.
  void seed(int s) {
    _seed = s;
    _counter = 0;
    _fallbackRng = math.Random(s);
  }

  /// Generates uniform random floats in $[0.0, 1.0)$.
  GpuArray<Float64> rand([List<int> shape = const [], GpuDevice? device]) {
    final size = ShapeUtils.computeSize(shape);
    final data = List<double>.filled(size, 0.0);

    for (var i = 0; i < size; i += 4) {
      final key = [_seed & 0xFFFFFFFF, (_seed >> 32) & 0xFFFFFFFF];
      final ctr = [_counter & 0xFFFFFFFF, (_counter >> 32) & 0xFFFFFFFF, 0, 0];
      _counter++;

      final out = Philox4x32Engine.philox4x32TenRounds(ctr, key);
      for (var k = 0; k < 4 && i + k < size; k++) {
        data[i + k] = (out[k] & 0xFFFFFFFF) / 4294967296.0;
      }
    }

    return GpuArray.fromList(data, shape, DType.float64, device: device);
  }

  /// Generates standard normal random floats ($\mathcal{N}(0, 1)$) using Box-Muller transform.
  GpuArray<Float64> randn([List<int> shape = const [], GpuDevice? device]) {
    final size = ShapeUtils.computeSize(shape);
    final data = List<double>.filled(size, 0.0);

    for (var i = 0; i < size; i += 2) {
      var u1 = 0.0;
      while (u1 <= 1e-15) {
        u1 = _fallbackRng.nextDouble();
      }
      final u2 = _fallbackRng.nextDouble();

      final mag = math.sqrt(-2.0 * math.log(u1));
      final z0 = mag * math.cos(2.0 * math.pi * u2);
      final z1 = mag * math.sin(2.0 * math.pi * u2);

      data[i] = z0;
      if (i + 1 < size) {
        data[i + 1] = z1;
      }
    }

    return GpuArray.fromList(data, shape, DType.float64, device: device);
  }

  /// Generates random integers from [low] (inclusive) to [high] (exclusive).
  GpuArray<T> randint<T>(
    int low, [
    int? high,
    List<int> shape = const [],
    DType<T>? dtype,
    GpuDevice? device,
  ]) {
    final effectiveLow = (high == null) ? 0 : low;
    final effectiveHigh = (high == null) ? low : high;

    if (effectiveHigh <= effectiveLow) {
      throw ArgumentError('randint high ($effectiveHigh) must be > low ($effectiveLow).');
    }

    final span = effectiveHigh - effectiveLow;
    final size = ShapeUtils.computeSize(shape);
    final targetDType = dtype ?? (DType.int64 as DType<T>);
    final result = GpuArray<T>.empty(shape, targetDType, device: device);

    for (var i = 0; i < size; i++) {
      final val = effectiveLow + _fallbackRng.nextInt(span);
      ComputeEngine.writeAny(result.buffer, targetDType, i, val);
    }

    return result;
  }

  /// Uniform distribution over $[low, high)$.
  GpuArray<Float64> uniform({
    double low = 0.0,
    double high = 1.0,
    List<int> shape = const [],
    GpuDevice? device,
  }) {
    final base = rand(shape, device);
    final span = high - low;
    final data = base.toList().cast<double>().map((v) => low + v * span).toList();
    return GpuArray.fromList(data, shape, DType.float64, device: device);
  }

  /// Normal (Gaussian) distribution $\mathcal{N}(\mu, \sigma^2)$.
  GpuArray<Float64> normal({
    double loc = 0.0,
    double scale = 1.0,
    List<int> shape = const [],
    GpuDevice? device,
  }) {
    final base = randn(shape, device);
    final data = base.toList().cast<double>().map((v) => loc + v * scale).toList();
    return GpuArray.fromList(data, shape, DType.float64, device: device);
  }

  /// Standard normal distribution $\mathcal{N}(0, 1)$.
  GpuArray<Float64> standard_normal([List<int> shape = const [], GpuDevice? device]) => randn(shape, device);

  /// Exponential distribution.
  GpuArray<Float64> exponential({
    double scale = 1.0,
    List<int> shape = const [],
    GpuDevice? device,
  }) {
    final size = ShapeUtils.computeSize(shape);
    final data = List<double>.filled(size, 0.0);
    for (var i = 0; i < size; i++) {
      var u = 0.0;
      while (u <= 1e-15) {
        u = _fallbackRng.nextDouble();
      }
      data[i] = -scale * math.log(u);
    }
    return GpuArray.fromList(data, shape, DType.float64, device: device);
  }

  /// Randomly samples elements from [a].
  GpuArray<T> choice<T>(
    GpuArray<T> a, {
    int? size,
    bool replace = true,
    List<double>? p,
    GpuDevice? device,
  }) {
    final aList = a.toList();
    final count = size ?? 1;
    final out = <T>[];

    for (var i = 0; i < count; i++) {
      final idx = _fallbackRng.nextInt(aList.length);
      out.add(aList[idx]);
    }

    final shape = size != null ? [size] : <int>[];
    return GpuArray.fromList(out, shape, a.dtype, device: device ?? a.device);
  }

  /// Randomly permutes a sequence or returns a permuted range.
  dynamic permutation(dynamic x, [GpuDevice? device]) {
    if (x is int) {
      final list = List.generate(x, (i) => i);
      list.shuffle(_fallbackRng);
      return GpuArray.fromList(list, [x], DType.int64, device: device);
    }
    if (x is GpuArray) {
      final list = x.toList();
      list.shuffle(_fallbackRng);
      return GpuArray.fromList(list, x.shape, x.dtype, device: device ?? x.device);
    }
    throw ArgumentError('permutation requires an integer or a GpuArray.');
  }

  /// In-place shuffle of the contents of [x].
  void shuffle(GpuArray x) {
    final list = x.toList();
    list.shuffle(_fallbackRng);
    for (var i = 0; i < list.length; i++) {
      ComputeEngine.writeAny(x.buffer, x.dtype, i, list[i]);
    }
  }
}

/// Default global random state.
final RandomState defaultRandomState = RandomState();

/// Seeds the default random generator.
void seed(int s) => defaultRandomState.seed(s);

/// Uniform random numbers in $[0.0, 1.0)$.
GpuArray<Float64> rand([List<int> shape = const [], GpuDevice? device]) => defaultRandomState.rand(shape, device);

/// Standard normal random numbers ($\mathcal{N}(0, 1)$).
GpuArray<Float64> randn([List<int> shape = const [], GpuDevice? device]) => defaultRandomState.randn(shape, device);

/// Random integers from [low] to [high].
GpuArray<T> randint<T>(
  int low, [
  int? high,
  List<int> shape = const [],
  DType<T>? dtype,
  GpuDevice? device,
]) =>
    defaultRandomState.randint<T>(low, high, shape, dtype, device);

/// Uniform distribution over $[low, high)$.
GpuArray<Float64> uniform({
  double low = 0.0,
  double high = 1.0,
  List<int> shape = const [],
  GpuDevice? device,
}) =>
    defaultRandomState.uniform(low: low, high: high, shape: shape, device: device);

/// Normal (Gaussian) distribution.
GpuArray<Float64> normal({
  double loc = 0.0,
  double scale = 1.0,
  List<int> shape = const [],
  GpuDevice? device,
}) =>
    defaultRandomState.normal(loc: loc, scale: scale, shape: shape, device: device);

/// Standard normal distribution.
GpuArray<Float64> standard_normal([List<int> shape = const [], GpuDevice? device]) =>
    defaultRandomState.standard_normal(shape, device);

/// Exponential distribution.
GpuArray<Float64> exponential({
  double scale = 1.0,
  List<int> shape = const [],
  GpuDevice? device,
}) =>
    defaultRandomState.exponential(scale: scale, shape: shape, device: device);

/// Random choice from [a].
GpuArray<T> choice<T>(
  GpuArray<T> a, {
  int? size,
  bool replace = true,
  List<double>? p,
  GpuDevice? device,
}) =>
    defaultRandomState.choice(a, size: size, replace: replace, p: p, device: device);

/// Random permutation.
dynamic permutation(dynamic x, [GpuDevice? device]) => defaultRandomState.permutation(x, device);

/// In-place shuffle.
void shuffle(GpuArray x) => defaultRandomState.shuffle(x);
