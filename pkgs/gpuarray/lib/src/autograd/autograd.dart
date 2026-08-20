// ignore_for_file: non_constant_identifier_names
import '../gpu_array.dart';

/// Global flag controlling whether gradient tracking is enabled.
bool _gradEnabled = true;

/// Returns true if autograd recording is currently enabled.
bool get isGradEnabled => _gradEnabled;

/// Disables gradient calculation during execution of [body].
R no_grad<R>(R Function() body) {
  final prev = _gradEnabled;
  _gradEnabled = false;
  try {
    return body();
  } finally {
    _gradEnabled = prev;
  }
}

/// Abstract base class for backward computation graph nodes.
abstract class GradFn {
  /// Name of the operation.
  String get name;

  /// Input nodes/tensors connected to this operation.
  List<GpuArray> get inputs;

  /// Computes the vector-Jacobian product (VJP) given upstream [gradOutput].
  List<GpuArray?> backward(GpuArray gradOutput);
}

/// Utility for reducing gradients across broadcasted dimensions.
GpuArray _unbroadcast(GpuArray grad, List<int> targetShape) {
  if (grad.shape == targetShape) return grad;

  var res = grad;
  final rankDiff = res.rank - targetShape.length;

  for (var i = 0; i < rankDiff; i++) {
    res = res.sum(axis: 0);
  }

  for (var i = 0; i < targetShape.length; i++) {
    if (targetShape[i] == 1 && res.shape[i] > 1) {
      res = res.sum(axis: i, keepDims: true);
    }
  }

  return res;
}

/// Backward node for Addition: $y = a + b$.
class AddBackward extends GradFn {
  final GpuArray a;
  final GpuArray b;

  AddBackward(this.a, this.b);

  @override
  String get name => 'AddBackward';

  @override
  List<GpuArray> get inputs => [a, b];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    final gradA = a.requiresGrad ? _unbroadcast(gradOutput, a.shape) : null;
    final gradB = b.requiresGrad ? _unbroadcast(gradOutput, b.shape) : null;
    return [gradA, gradB];
  }
}

/// Backward node for Subtraction: $y = a - b$.
class SubBackward extends GradFn {
  final GpuArray a;
  final GpuArray b;

  SubBackward(this.a, this.b);

  @override
  String get name => 'SubBackward';

  @override
  List<GpuArray> get inputs => [a, b];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    final gradA = a.requiresGrad ? _unbroadcast(gradOutput, a.shape) : null;
    final gradB = b.requiresGrad ? _unbroadcast(gradOutput.negate(), b.shape) : null;
    return [gradA, gradB];
  }
}

/// Backward node for Multiplication: $y = a \cdot b$.
class MulBackward extends GradFn {
  final GpuArray a;
  final GpuArray b;

  MulBackward(this.a, this.b);

  @override
  String get name => 'MulBackward';

  @override
  List<GpuArray> get inputs => [a, b];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    final gradA = a.requiresGrad ? _unbroadcast(gradOutput * b, a.shape) : null;
    final gradB = b.requiresGrad ? _unbroadcast(gradOutput * a, b.shape) : null;
    return [gradA, gradB];
  }
}

/// Backward node for Division: $y = a / b$.
class DivBackward extends GradFn {
  final GpuArray a;
  final GpuArray b;

  DivBackward(this.a, this.b);

  @override
  String get name => 'DivBackward';

  @override
  List<GpuArray> get inputs => [a, b];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    final gradA = a.requiresGrad ? _unbroadcast(gradOutput / b, a.shape) : null;
    final gradB = b.requiresGrad
        ? _unbroadcast((gradOutput * a.negate()) / (b * b), b.shape)
        : null;
    return [gradA, gradB];
  }
}

/// Backward node for Matrix Multiplication: $Y = A B$.
class MatmulBackward extends GradFn {
  final GpuArray a;
  final GpuArray b;

  MatmulBackward(this.a, this.b);

  @override
  String get name => 'MatmulBackward';

  @override
  List<GpuArray> get inputs => [a, b];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    // dL/dA = G @ B^T
    // dL/dB = A^T @ G
    GpuArray? gradA;
    GpuArray? gradB;

    if (a.requiresGrad) {
      final bT = b.swapaxes(-1, -2);
      gradA = gradOutput.matmul(bT);
    }
    if (b.requiresGrad) {
      final aT = a.swapaxes(-1, -2);
      gradB = aT.matmul(gradOutput);
    }

    return [gradA, gradB];
  }
}

/// Backward node for Sum Reduction: $y = \sum a$.
class SumBackward extends GradFn {
  final GpuArray a;
  final int? axis;
  final bool keepDims;

  SumBackward(this.a, {this.axis, this.keepDims = false});

  @override
  String get name => 'SumBackward';

  @override
  List<GpuArray> get inputs => [a];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!a.requiresGrad) return [null];

    var grad = gradOutput;
    if (axis != null && !keepDims) {
      grad = grad.unsqueeze(axis!);
    }

    // Broadcast grad to a.shape
    final ones = GpuArray.ones(a.shape, a.dtype, device: a.device);
    return [grad * ones];
  }
}

/// Backward node for Mean Reduction: $y = \text{mean}(a)$.
class MeanBackward extends GradFn {
  final GpuArray a;
  final int? axis;
  final bool keepDims;

  MeanBackward(this.a, {this.axis, this.keepDims = false});

  @override
  String get name => 'MeanBackward';

  @override
  List<GpuArray> get inputs => [a];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!a.requiresGrad) return [null];

    final n = (axis == null) ? a.size : a.shape[axis! < 0 ? axis! + a.rank : axis!];
    final scale = 1.0 / n;

    var grad = gradOutput * scale;
    if (axis != null && !keepDims) {
      grad = grad.unsqueeze(axis!);
    }

    final ones = GpuArray.ones(a.shape, a.dtype, device: a.device);
    return [grad * ones];
  }
}

/// Backward node for ReLU: $y = \max(0, x)$.
class ReluBackward extends GradFn {
  final GpuArray x;

  ReluBackward(this.x);

  @override
  String get name => 'ReluBackward';

  @override
  List<GpuArray> get inputs => [x];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    if (!x.requiresGrad) return [null];
    final mask = x.greater(0.0).astype(x.dtype);
    return [gradOutput * mask];
  }
}

/// Backward node for Sigmoid: $y = \sigma(x) = \frac{1}{1 + e^{-x}}$.
class SigmoidBackward extends GradFn {
  final GpuArray y; // saved output

  SigmoidBackward(this.y);

  @override
  String get name => 'SigmoidBackward';

  @override
  List<GpuArray> get inputs => [y];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    // dy/dx = y * (1 - y)
    final oneMinusY = y.negate() + 1.0;
    return [gradOutput * y * oneMinusY];
  }
}

/// Backward node for Tanh: $y = \tanh(x)$.
class TanhBackward extends GradFn {
  final GpuArray y; // saved output

  TanhBackward(this.y);

  @override
  String get name => 'TanhBackward';

  @override
  List<GpuArray> get inputs => [y];

  @override
  List<GpuArray?> backward(GpuArray gradOutput) {
    // dy/dx = 1 - y^2
    final oneMinusYSq = (y * y).negate() + 1.0;
    return [gradOutput * oneMinusYSq];
  }
}

/// Executes reverse-mode automatic differentiation starting from [root].
void runBackward(GpuArray root, [GpuArray? gradient, bool retainGraph = false]) {
  if (!root.requiresGrad) {
    throw StateError('Cannot call backward() on a tensor with requiresGrad = false.');
  }

  final seedGrad = gradient ??
      ((root.rank == 0 || root.size == 1)
          ? GpuArray.ones(root.shape, root.dtype, device: root.device)
          : throw ArgumentError('grad can be implicitly created only for scalar outputs'));

  root.grad = (root.grad == null) ? seedGrad : (root.grad! + seedGrad);

  // Topological sort of computational DAG
  final orderedNodes = <GpuArray>[];
  final visited = <GpuArray>{};

  void buildTopo(GpuArray node) {
    if (visited.contains(node)) return;
    visited.add(node);

    final fn = node.gradFn;
    if (fn != null) {
      for (final input in fn.inputs) {
        if (input.requiresGrad) {
          buildTopo(input);
        }
      }
    }
    orderedNodes.add(node);
  }

  buildTopo(root);

  // Traverse in reverse topological order
  for (var i = orderedNodes.length - 1; i >= 0; i--) {
    final node = orderedNodes[i];
    final fn = node.gradFn;
    final nodeGrad = node.grad;

    if (fn != null && nodeGrad != null) {
      final inputGrads = fn.backward(nodeGrad);
      final inputs = fn.inputs;

      for (var j = 0; j < inputs.length; j++) {
        final inp = inputs[j];
        final g = (j < inputGrads.length) ? inputGrads[j] : null;

        if (inp.requiresGrad && g != null) {
          inp.grad = (inp.grad == null) ? g : (inp.grad! + g);
        }
      }

      if (!retainGraph) {
        node.gradFn = null;
      }
    }
  }
}
