// ignore_for_file: non_constant_identifier_names
import 'dart:math' as math;
import '../dtype.dart';
import '../gpu_array.dart';
import '../autograd/autograd.dart';

/// Applies the Rectified Linear Unit function element-wise: $\text{ReLU}(x) = \max(0, x)$.
GpuArray relu(GpuArray input) {
  final mask = input.greater(0.0).astype(input.dtype);
  final out = input * mask;
  if (isGradEnabled && input.requiresGrad) {
    out.requiresGrad = true;
    out.gradFn = ReluBackward(input);
  }
  return out;
}

/// Applies the Sigmoid element-wise: $\sigma(x) = \frac{1}{1 + e^{-x}}$.
GpuArray sigmoid(GpuArray input) {
  // sigmoid(x) = 1 / (1 + exp(-x))
  final expNeg = input.negate().exp();
  final out = (expNeg + 1.0).pow(-1.0);
  if (isGradEnabled && input.requiresGrad) {
    out.requiresGrad = true;
    out.gradFn = SigmoidBackward(out);
  }
  return out;
}

/// Applies the Hyperbolic Tangent element-wise: $\tanh(x)$.
GpuArray tanh(GpuArray input) {
  final out = input.tanh();
  if (isGradEnabled && input.requiresGrad) {
    out.requiresGrad = true;
    out.gradFn = TanhBackward(out);
  }
  return out;
}

/// Applies Gaussian Error Linear Unit (GELU) activation:
/// $\text{GELU}(x) = 0.5x \left(1 + \tanh\left(\sqrt{2/\pi}\left(x + 0.044715 x^3\right)\right)\right)$.
GpuArray gelu(GpuArray input) {
  final sqrt2OverPi = math.sqrt(2.0 / math.pi);
  final xCubed = input * input * input * 0.044715;
  final inner = (input + xCubed) * sqrt2OverPi;
  final tanhInner = inner.tanh();
  final onePlusTanh = tanhInner + 1.0;
  return input * onePlusTanh * 0.5;
}

/// Applies the Sigmoid Linear Unit (SiLU / Swish) activation: $\text{SiLU}(x) = x \cdot \sigma(x)$.
GpuArray silu(GpuArray input) {
  return input * sigmoid(input);
}

/// Applies Softmax function to an n-dimensional input tensor along [axis].
GpuArray softmax(GpuArray input, {int axis = -1}) {
  final maxVal = input.max(axis: axis, keepDims: true);
  final expX = (input - maxVal).exp();
  final sumExp = expX.sum(axis: axis, keepDims: true);
  return expX / sumExp;
}

/// Applies Log-Softmax function to an n-dimensional input tensor along [axis].
GpuArray log_softmax(GpuArray input, {int axis = -1}) {
  final maxVal = input.max(axis: axis, keepDims: true);
  final expX = (input - maxVal).exp();
  final sumExp = expX.sum(axis: axis, keepDims: true);
  return (input - maxVal) - sumExp.log();
}

/// Measures the Mean Squared Error (squared L2 norm) between [input] and [target].
GpuArray mse_loss(
  GpuArray input,
  GpuArray target, {
  String reduction = 'mean',
}) {
  final diff = input - target;
  final sq = diff * diff;
  if (reduction == 'mean') {
    return sq.mean();
  } else if (reduction == 'sum') {
    return sq.sum();
  }
  return sq;
}

/// Computes the cross entropy loss between input logits and target class indices.
GpuArray cross_entropy(
  GpuArray logits,
  GpuArray targets, {
  String reduction = 'mean',
}) {
  // logits: [N, C], targets: [N] (integers)
  final logProbs = log_softmax(logits, axis: -1);
  final numSamples = logits.shape[0];
  final numClasses = logits.shape[1];

  final targetList = targets.toList().cast<int>();
  final logProbsFlat = logProbs.toNDArray();
  final logProbsList = logProbsFlat.toList().cast<num>().map((e) => e.toDouble()).toList();

  final losses = <double>[];
  for (var i = 0; i < numSamples; i++) {
    final targetClass = targetList[i];
    final logP = logProbsList[i * numClasses + targetClass];
    losses.add(-logP);
  }
  logProbsFlat.dispose();

  final lossArray = GpuArray.fromList(losses, [numSamples], DType.float64, device: logits.device);

  if (reduction == 'mean') {
    return lossArray.mean();
  } else if (reduction == 'sum') {
    return lossArray.sum();
  }
  return lossArray;
}
