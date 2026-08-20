import 'dart:math' as math;
import '../dtype.dart';
import '../gpu_array.dart';
import '../device.dart';
import '../random/random.dart' as rng;
import '../autograd/autograd.dart';
import 'module.dart';
import 'functional.dart' as f;

/// Applies an affine linear transformation to the incoming data: $y = x A^T + b$.
class Linear extends Module {
  final int inFeatures;
  final int outFeatures;
  final bool hasBias;

  late final GpuArray weight;
  late final GpuArray? bias;

  Linear(
    this.inFeatures,
    this.outFeatures, {
    this.hasBias = true,
    GpuDevice? device,
  }) {
    final dev = device ?? GpuDevice.defaultDevice;
    final k = 1.0 / math.sqrt(inFeatures);

    final rawW = rng.uniform(
      low: -k,
      high: k,
      shape: [outFeatures, inFeatures],
      device: dev,
    );
    weight = registerParameter(
      'weight',
      GpuArray.fromList(
        rawW.toList(),
        [outFeatures, inFeatures],
        DType.float64,
        device: dev,
        requiresGrad: true,
      ),
    );

    if (hasBias) {
      final rawB = rng.uniform(
        low: -k,
        high: k,
        shape: [outFeatures],
        device: dev,
      );
      bias = registerParameter(
        'bias',
        GpuArray.fromList(
          rawB.toList(),
          [outFeatures],
          DType.float64,
          device: dev,
          requiresGrad: true,
        ),
      );
    } else {
      bias = null;
    }
  }

  @override
  GpuArray forward(GpuArray input) {
    final wT = weight.swapaxes(-1, -2);
    final output = input.matmul(wT);
    if (bias != null) {
      return output + bias!;
    }
    return output;
  }
}

/// Applies a 2D convolution over an input signal composed of several input planes.
class Conv2d extends Module {
  final int inChannels;
  final int outChannels;
  final int kernelSize;
  final int stride;
  final int padding;
  final bool hasBias;

  late final GpuArray weight;
  late final GpuArray? bias;

  Conv2d(
    this.inChannels,
    this.outChannels,
    this.kernelSize, {
    this.stride = 1,
    this.padding = 0,
    this.hasBias = true,
    GpuDevice? device,
  }) {
    final dev = device ?? GpuDevice.defaultDevice;
    final k = 1.0 / math.sqrt(inChannels * kernelSize * kernelSize);

    final rawW = rng.uniform(
      low: -k,
      high: k,
      shape: [outChannels, inChannels, kernelSize, kernelSize],
      device: dev,
    );
    weight = registerParameter(
      'weight',
      GpuArray.fromList(
        rawW.toList(),
        [outChannels, inChannels, kernelSize, kernelSize],
        DType.float64,
        device: dev,
        requiresGrad: true,
      ),
    );

    if (hasBias) {
      final rawB = rng.uniform(
        low: -k,
        high: k,
        shape: [outChannels],
        device: dev,
      );
      bias = registerParameter(
        'bias',
        GpuArray.fromList(
          rawB.toList(),
          [outChannels],
          DType.float64,
          device: dev,
          requiresGrad: true,
        ),
      );
    } else {
      bias = null;
    }
  }

  @override
  GpuArray forward(GpuArray input) {
    // Input: [N, C_in, H, W]
    final batchSize = input.shape[0];
    final inC = input.shape[1];
    final inH = input.shape[2];
    final inW = input.shape[3];

    final outH = ((inH + 2 * padding - kernelSize) ~/ stride) + 1;
    final outW = ((inW + 2 * padding - kernelSize) ~/ stride) + 1;

    final inputND = input.toNDArray();
    final inputList = inputND
        .toList()
        .cast<num>()
        .map((e) => e.toDouble())
        .toList();
    inputND.dispose();

    final weightND = weight.toNDArray();
    final weightList = weightND
        .toList()
        .cast<num>()
        .map((e) => e.toDouble())
        .toList();
    weightND.dispose();

    List<double>? biasList;
    if (bias != null) {
      final biasND = bias!.toNDArray();
      biasList = biasND.toList().cast<num>().map((e) => e.toDouble()).toList();
      biasND.dispose();
    }

    final outSize = batchSize * outChannels * outH * outW;
    final outList = List<double>.filled(outSize, 0.0);

    for (var b = 0; b < batchSize; b++) {
      for (var oc = 0; oc < outChannels; oc++) {
        final bVal = biasList != null ? biasList[oc] : 0.0;
        for (var oh = 0; oh < outH; oh++) {
          for (var ow = 0; ow < outW; ow++) {
            var sum = bVal;
            final ihStart = oh * stride - padding;
            final iwStart = ow * stride - padding;

            for (var ic = 0; ic < inC; ic++) {
              for (var kh = 0; kh < kernelSize; kh++) {
                final ih = ihStart + kh;
                if (ih < 0 || ih >= inH) continue;
                for (var kw = 0; kw < kernelSize; kw++) {
                  final iw = iwStart + kw;
                  if (iw < 0 || iw >= inW) continue;

                  final inIdx = ((b * inC + ic) * inH + ih) * inW + iw;
                  final wIdx =
                      ((oc * inC + ic) * kernelSize + kh) * kernelSize + kw;
                  sum += inputList[inIdx] * weightList[wIdx];
                }
              }
            }

            final outIdx = ((b * outChannels + oc) * outH + oh) * outW + ow;
            outList[outIdx] = sum;
          }
        }
      }
    }

    final out = GpuArray.fromList(
      outList,
      [batchSize, outChannels, outH, outW],
      input.dtype,
      device: input.device,
    );

    if (isGradEnabled &&
        (input.requiresGrad ||
            weight.requiresGrad ||
            (bias != null && bias!.requiresGrad))) {
      out.requiresGrad = true;
      out.gradFn = Conv2dBackward(
        input: input,
        weight: weight,
        bias: bias,
        stride: stride,
        padding: padding,
        kernelSize: kernelSize,
      );
    }

    return out;
  }
}

/// Applies Layer Normalization over a mini-batch of inputs.
class LayerNorm extends Module {
  final List<int> normalizedShape;
  final double eps;

  late final GpuArray weight;
  late final GpuArray bias;

  LayerNorm(this.normalizedShape, {this.eps = 1e-5, GpuDevice? device}) {
    final dev = device ?? GpuDevice.defaultDevice;
    weight = registerParameter(
      'weight',
      GpuArray.ones(
        normalizedShape,
        DType.float64,
        device: dev,
        requiresGrad: true,
      ),
    );
    bias = registerParameter(
      'bias',
      GpuArray.zeros(
        normalizedShape,
        DType.float64,
        device: dev,
        requiresGrad: true,
      ),
    );
  }

  @override
  GpuArray forward(GpuArray input) {
    final mean = input.mean(axis: -1, keepDims: true);
    final variance = ((input - mean) * (input - mean)).mean(
      axis: -1,
      keepDims: true,
    );
    final normalized = (input - mean) / ((variance + eps).sqrt());
    return normalized * weight + bias;
  }
}

/// During training, randomly zeroes some of the elements of the input tensor with probability [p].
class Dropout extends Module {
  final double p;

  Dropout({this.p = 0.5});

  @override
  GpuArray forward(GpuArray input) {
    if (!isTraining || p == 0.0) return input;
    final mask = rng
        .rand(input.shape, input.device)
        .greater(p)
        .astype(input.dtype);
    final scale = 1.0 / (1.0 - p);
    return input * mask * scale;
  }
}

/// A simple lookup table that stores embeddings of a fixed dictionary and size.
class Embedding extends Module {
  final int numEmbeddings;
  final int embeddingDim;

  late final GpuArray weight;

  Embedding(this.numEmbeddings, this.embeddingDim, {GpuDevice? device}) {
    final dev = device ?? GpuDevice.defaultDevice;
    final rawW = rng.randn([numEmbeddings, embeddingDim], dev);
    weight = registerParameter(
      'weight',
      GpuArray.fromList(
        rawW.toList(),
        [numEmbeddings, embeddingDim],
        DType.float64,
        device: dev,
        requiresGrad: true,
      ),
    );
  }

  @override
  GpuArray forward(GpuArray indices) {
    final idxList = indices.toList().cast<int>();
    final outRows = <dynamic>[];
    for (final idx in idxList) {
      final row = weight[idx];
      outRows.addAll(row.toList());
    }
    final outShape = [...indices.shape, embeddingDim];
    final out = GpuArray.fromList(
      outRows,
      outShape,
      weight.dtype,
      device: indices.device,
    );
    if (isGradEnabled && weight.requiresGrad) {
      out.requiresGrad = true;
      out.gradFn = EmbeddingBackward(
        weight,
        indices,
        numEmbeddings,
        embeddingDim,
      );
    }
    return out;
  }
}

/// Applies ReLU activation as a module.
class ReLU extends Module {
  @override
  GpuArray forward(GpuArray input) => f.relu(input);
}

/// Applies GELU activation as a module.
class GELU extends Module {
  @override
  GpuArray forward(GpuArray input) => f.gelu(input);
}

/// Applies Sigmoid activation as a module.
class Sigmoid extends Module {
  @override
  GpuArray forward(GpuArray input) => f.sigmoid(input);
}

/// Applies Tanh activation as a module.
class Tanh extends Module {
  @override
  GpuArray forward(GpuArray input) => f.tanh(input);
}
