import 'dart:math' as math;
import '../gpu_array.dart';

/// Base class for all parameter optimizers.
abstract class Optimizer {
  /// The parameters to optimize.
  final List<GpuArray> params;

  /// Learning rate.
  double lr;

  Optimizer(this.params, {required this.lr});

  /// Performs a single optimization step (parameter update).
  void step();

  /// Clears the gradients of all optimized [params].
  void zeroGrad() {
    for (final p in params) {
      p.zeroGrad();
    }
  }
}

/// Stochastic Gradient Descent (SGD) optimizer with momentum and weight decay.
class SGD extends Optimizer {
  final double momentum;
  final double weightDecay;
  final bool nesterov;

  final Map<GpuArray, GpuArray> _velocity = {};

  SGD(
    super.params, {
    required super.lr,
    this.momentum = 0.0,
    this.weightDecay = 0.0,
    this.nesterov = false,
  });

  @override
  void step() {
    for (final p in params) {
      final grad = p.grad;
      if (grad == null) continue;

      var dP = grad;
      if (weightDecay != 0.0) {
        dP = dP + (p * weightDecay);
      }

      if (momentum != 0.0) {
        var v = _velocity[p];
        if (v == null) {
          v = GpuArray.zeros(p.shape, p.dtype, device: p.device);
          _velocity[p] = v;
        }

        // v = momentum * v + dP
        final newV = (v * momentum) + dP;
        _velocity[p] = newV;

        if (nesterov) {
          dP = dP + (newV * momentum);
        } else {
          dP = newV;
        }
      }

      // p = p - lr * dP (in-place update directly on buffer)
      final pFlat = p.toNDArray();
      final pList = pFlat.toList().cast<double>();
      final dpFlat = dP.toNDArray();
      final dpList = dpFlat.toList().cast<double>();

      final updated = <double>[];
      for (var i = 0; i < pList.length; i++) {
        updated.add(pList[i] - lr * dpList[i]);
      }

      final newP = GpuArray.fromList(updated, p.shape, p.dtype, device: p.device);
      newP.buffer.copyToBuffer(p.buffer, p.byteSize);

      pFlat.dispose();
      dpFlat.dispose();
      newP.dispose();
    }
  }
}

/// Adam optimizer (Adaptive Moment Estimation).
class Adam extends Optimizer {
  final double beta1;
  final double beta2;
  final double eps;
  final double weightDecay;

  int _stepCount = 0;
  final Map<GpuArray, GpuArray> _m = {};
  final Map<GpuArray, GpuArray> _v = {};

  Adam(
    super.params, {
    required super.lr,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.eps = 1e-8,
    this.weightDecay = 0.0,
  });

  @override
  void step() {
    _stepCount++;
    final biasCorrection1 = 1.0 - math.pow(beta1, _stepCount);
    final biasCorrection2 = 1.0 - math.pow(beta2, _stepCount);

    for (final p in params) {
      final grad = p.grad;
      if (grad == null) continue;

      var g = grad;
      if (weightDecay != 0.0) {
        g = g + (p * weightDecay);
      }

      var m = _m[p];
      if (m == null) {
        m = GpuArray.zeros(p.shape, p.dtype, device: p.device);
        _m[p] = m;
      }

      var v = _v[p];
      if (v == null) {
        v = GpuArray.zeros(p.shape, p.dtype, device: p.device);
        _v[p] = v;
      }

      // m = beta1 * m + (1 - beta1) * g
      final newM = (m * beta1) + (g * (1.0 - beta1));
      _m[p] = newM;

      // v = beta2 * v + (1 - beta2) * (g * g)
      final newV = (v * beta2) + ((g * g) * (1.0 - beta2));
      _v[p] = newV;

      // update = (m / biasCorrection1) / (sqrt(v / biasCorrection2) + eps)
      final mHat = newM * (1.0 / biasCorrection1);
      final vHat = newV * (1.0 / biasCorrection2);
      final denom = vHat.sqrt() + eps;
      final step = (mHat / denom) * lr;

      final pFlat = p.toNDArray();
      final pList = pFlat.toList().cast<double>();
      final stepFlat = step.toNDArray();
      final stepList = stepFlat.toList().cast<double>();

      final updated = <double>[];
      for (var i = 0; i < pList.length; i++) {
        updated.add(pList[i] - stepList[i]);
      }

      final newP = GpuArray.fromList(updated, p.shape, p.dtype, device: p.device);
      newP.buffer.copyToBuffer(p.buffer, p.byteSize);

      pFlat.dispose();
      stepFlat.dispose();
      newP.dispose();
    }
  }
}

/// AdamW optimizer (Decoupled Weight Decay Adam).
class AdamW extends Optimizer {
  final double beta1;
  final double beta2;
  final double eps;
  final double weightDecay;

  int _stepCount = 0;
  final Map<GpuArray, GpuArray> _m = {};
  final Map<GpuArray, GpuArray> _v = {};

  AdamW(
    super.params, {
    required super.lr,
    this.beta1 = 0.9,
    this.beta2 = 0.999,
    this.eps = 1e-8,
    this.weightDecay = 0.01,
  });

  @override
  void step() {
    _stepCount++;
    final biasCorrection1 = 1.0 - math.pow(beta1, _stepCount);
    final biasCorrection2 = 1.0 - math.pow(beta2, _stepCount);

    for (final p in params) {
      final grad = p.grad;
      if (grad == null) continue;

      // Weight decay step decoupled from gradient update
      if (weightDecay != 0.0) {
        final pDecayed = p * (1.0 - lr * weightDecay);
        pDecayed.buffer.copyToBuffer(p.buffer, p.byteSize);
      }

      var m = _m[p];
      if (m == null) {
        m = GpuArray.zeros(p.shape, p.dtype, device: p.device);
        _m[p] = m;
      }

      var v = _v[p];
      if (v == null) {
        v = GpuArray.zeros(p.shape, p.dtype, device: p.device);
        _v[p] = v;
      }

      final newM = (m * beta1) + (grad * (1.0 - beta1));
      _m[p] = newM;

      final newV = (v * beta2) + ((grad * grad) * (1.0 - beta2));
      _v[p] = newV;

      final mHat = newM * (1.0 / biasCorrection1);
      final vHat = newV * (1.0 / biasCorrection2);
      final denom = vHat.sqrt() + eps;
      final step = (mHat / denom) * lr;

      final pFlat = p.toNDArray();
      final pList = pFlat.toList().cast<double>();
      final stepFlat = step.toNDArray();
      final stepList = stepFlat.toList().cast<double>();

      final updated = <double>[];
      for (var i = 0; i < pList.length; i++) {
        updated.add(pList[i] - stepList[i]);
      }

      final newP = GpuArray.fromList(updated, p.shape, p.dtype, device: p.device);
      newP.buffer.copyToBuffer(p.buffer, p.byteSize);

      pFlat.dispose();
      stepFlat.dispose();
      newP.dispose();
    }
  }
}
