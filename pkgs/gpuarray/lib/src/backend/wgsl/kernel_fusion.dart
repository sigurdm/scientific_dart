import 'dart:convert';
import 'dart:typed_data';
import 'wgsl_types.dart';
import 'wgsl_templates.dart';
import '../../dtype.dart';
import '../../gpu_array.dart';
import '../../serialization/webgpu_pipeline.dart';

/// Base class for all expression nodes in a fused kernel computation graph.
typedef GpuExpr = Expr;
typedef GpuVarExpr = VarExpr;
typedef GpuConstExpr = ConstExpr;
typedef GpuScalarParamExpr = ScalarParamExpr;
typedef GpuLoopExpr = LoopExpr;

abstract class Expr {
  const Expr();

  /// Creates a variable reference representing an input tensor buffer.
  static VarExpr variable(
    String name, {
    int bindingIndex = 0,
    WgslDType dtype = WgslDType.float32,
  }) => VarExpr(name, bindingIndex: bindingIndex, dtype: dtype);

  /// Creates a literal numerical constant.
  static ConstExpr constant(double value) => ConstExpr(value);

  /// Creates a runtime scalar uniform parameter.
  static ScalarParamExpr scalar(String name, {double defaultValue = 0.0}) =>
      ScalarParamExpr(name, defaultValue: defaultValue);

  /// Creates a functional loop AST node with state variables, dynamic condition, and step function.
  static LoopExpr loop({
    required List<Object> initialValues,
    required Object maxIterations,
    required Expr Function(List<Expr> state, Expr iter) condition,
    required List<Expr> Function(List<Expr> state, Expr iter) step,
    Expr Function(List<Expr> state)? result,
    String? name,
  }) => LoopExpr(
    initialValues: initialValues,
    maxIterations: maxIterations,
    condition: condition,
    step: step,
    result: result,
    name: name,
  );

  /// Converts any num or Expr object into an [Expr].
  static Expr from(Object value) {
    if (value is Expr) return value;
    if (value is num) return ConstExpr(value.toDouble());
    throw ArgumentError(
      'Cannot convert $value of type ${value.runtimeType} to Expr',
    );
  }

  /// Evaluates to the WGSL code snippet for this subexpression.
  String toWgsl();

  /// Collects all input variable references used in this expression tree.
  Set<VarExpr> get variables;

  /// Collects all scalar uniform parameters used in this expression tree.
  Set<ScalarParamExpr> get scalarParams;

  /// Computes the depth of the expression tree.
  int get depth;

  /// Total count of AST nodes in this expression subtree.
  int get nodeCount;

  /// Generates a normalized structural fingerprint string for caching.
  String toFingerprint();

  // Operator overloads for building expression trees fluently
  Expr operator +(Object other) => BinaryOpExpr('add', this, Expr.from(other));

  Expr operator -(Object other) => BinaryOpExpr('sub', this, Expr.from(other));

  Expr operator *(Object other) => BinaryOpExpr('mul', this, Expr.from(other));

  Expr operator /(Object other) => BinaryOpExpr('div', this, Expr.from(other));

  Expr operator -() => UnaryOpExpr('negate', this);

  Expr pow(Object exponent) => BinaryOpExpr('pow', this, Expr.from(exponent));

  Expr max(Object other) => BinaryOpExpr('max', this, Expr.from(other));

  Expr min(Object other) => BinaryOpExpr('min', this, Expr.from(other));

  Expr equal(Object other) => BinaryOpExpr('eq', this, Expr.from(other));

  Expr notEqual(Object other) => BinaryOpExpr('neq', this, Expr.from(other));

  Expr greaterThan(Object other) => BinaryOpExpr('gt', this, Expr.from(other));

  Expr lessThan(Object other) => BinaryOpExpr('lt', this, Expr.from(other));

  Expr greaterEqual(Object other) =>
      BinaryOpExpr('gte', this, Expr.from(other));

  Expr lessEqual(Object other) => BinaryOpExpr('lte', this, Expr.from(other));

  // Common Unary activations and math functions
  Expr relu() => UnaryOpExpr('relu', this);
  Expr silu() => UnaryOpExpr('silu', this);
  Expr gelu() => UnaryOpExpr('gelu', this);
  Expr sigmoid() => UnaryOpExpr('sigmoid', this);
  Expr tanh() => UnaryOpExpr('tanh', this);
  Expr exp() => UnaryOpExpr('exp', this);
  Expr log() => UnaryOpExpr('log', this);
  Expr sqrt() => UnaryOpExpr('sqrt', this);
  Expr rsqrt() => UnaryOpExpr('rsqrt', this);
  Expr abs() => UnaryOpExpr('abs', this);
  Expr sin() => UnaryOpExpr('sin', this);
  Expr cos() => UnaryOpExpr('cos', this);
  Expr tan() => UnaryOpExpr('tan', this);
  Expr floor() => UnaryOpExpr('floor', this);
  Expr ceil() => UnaryOpExpr('ceil', this);
  Expr round() => UnaryOpExpr('round', this);
  Expr reciprocal() => UnaryOpExpr('reciprocal', this);
  Expr hardswish() => UnaryOpExpr('hardswish', this);
  Expr softplus() => UnaryOpExpr('softplus', this);
  Expr mish() => UnaryOpExpr('mish', this);

  /// Clamps expression between [minVal] and [maxVal].
  Expr clamp(Object minVal, Object maxVal) =>
      TernaryOpExpr('clamp', this, Expr.from(minVal), Expr.from(maxVal));

  /// Selects [thenExpr] if this expression evaluates to > 0.0, else [elseExpr].
  Expr where(Object thenExpr, Object elseExpr) =>
      TernaryOpExpr('where', this, Expr.from(thenExpr), Expr.from(elseExpr));
}

/// Represents an input tensor variable buffer.
final class VarExpr extends Expr {
  final String name;
  final int bindingIndex;
  final WgslDType dtype;

  const VarExpr(
    this.name, {
    this.bindingIndex = 0,
    this.dtype = WgslDType.float32,
  });

  @override
  String toWgsl() => '${name}_val';

  @override
  Set<VarExpr> get variables => {this};

  @override
  Set<ScalarParamExpr> get scalarParams => {};

  @override
  int get depth => 1;

  @override
  int get nodeCount => 1;

  @override
  String toFingerprint() => 'var($name:$bindingIndex:${dtype.wgslType})';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is VarExpr &&
          runtimeType == other.runtimeType &&
          name == other.name &&
          bindingIndex == other.bindingIndex &&
          dtype == other.dtype;

  @override
  int get hashCode => Object.hash(name, bindingIndex, dtype);

  @override
  String toString() => 'VarExpr($name, binding: $bindingIndex)';
}

/// Represents a constant floating-point literal.
final class ConstExpr extends Expr {
  final double value;

  const ConstExpr(this.value);

  @override
  String toWgsl() {
    final s = value.toString();
    return (s.contains('.') || s.contains('e') || s.contains('E'))
        ? '${s}f'
        : '$s.0f';
  }

  @override
  Set<VarExpr> get variables => {};

  @override
  Set<ScalarParamExpr> get scalarParams => {};

  @override
  int get depth => 1;

  @override
  int get nodeCount => 1;

  @override
  String toFingerprint() => 'const($value)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ConstExpr &&
          runtimeType == other.runtimeType &&
          value == other.value;

  @override
  int get hashCode => value.hashCode;

  @override
  String toString() => 'ConstExpr($value)';
}

/// Represents a dynamic runtime scalar uniform parameter.
final class ScalarParamExpr extends Expr {
  final String name;
  final double defaultValue;

  const ScalarParamExpr(this.name, {this.defaultValue = 0.0});

  @override
  String toWgsl() => 'uniforms.$name';

  @override
  Set<VarExpr> get variables => {};

  @override
  Set<ScalarParamExpr> get scalarParams => {this};

  @override
  int get depth => 1;

  @override
  int get nodeCount => 1;

  @override
  String toFingerprint() => 'scalar($name)';

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ScalarParamExpr &&
          runtimeType == other.runtimeType &&
          name == other.name;

  @override
  int get hashCode => name.hashCode;

  @override
  String toString() => 'ScalarParamExpr($name)';
}

/// Represents a unary operation applied to an expression.
final class UnaryOpExpr extends Expr {
  final String op;
  final Expr child;

  const UnaryOpExpr(this.op, this.child);

  @override
  String toWgsl() =>
      WgslTemplates.getWgslUnaryExpression(op, '(${child.toWgsl()})');

  @override
  Set<VarExpr> get variables => child.variables;

  @override
  Set<ScalarParamExpr> get scalarParams => child.scalarParams;

  @override
  int get depth => 1 + child.depth;

  @override
  int get nodeCount => 1 + child.nodeCount;

  @override
  String toFingerprint() => '$op(${child.toFingerprint()})';

  @override
  String toString() => 'UnaryOpExpr($op, $child)';
}

/// Represents a binary operation between two expressions.
final class BinaryOpExpr extends Expr {
  final String op;
  final Expr left;
  final Expr right;

  const BinaryOpExpr(this.op, this.left, this.right);

  @override
  String toWgsl() =>
      '(${WgslTemplates.getWgslOpExpression(op, left.toWgsl(), right.toWgsl())})';

  @override
  Set<VarExpr> get variables => {...left.variables, ...right.variables};

  @override
  Set<ScalarParamExpr> get scalarParams => {
    ...left.scalarParams,
    ...right.scalarParams,
  };

  @override
  int get depth => 1 + (left.depth > right.depth ? left.depth : right.depth);

  @override
  int get nodeCount => 1 + left.nodeCount + right.nodeCount;

  @override
  String toFingerprint() =>
      '$op(${left.toFingerprint()}, ${right.toFingerprint()})';

  @override
  String toString() => 'BinaryOpExpr($op, $left, $right)';
}

/// Represents a ternary operation (e.g. clamp or select/where).
final class TernaryOpExpr extends Expr {
  final String op;
  final Expr first;
  final Expr second;
  final Expr third;

  const TernaryOpExpr(this.op, this.first, this.second, this.third);

  @override
  String toWgsl() {
    switch (op) {
      case 'clamp':
        return 'clamp(${first.toWgsl()}, ${second.toWgsl()}, ${third.toWgsl()})';
      case 'where':
        return 'select(${third.toWgsl()}, ${second.toWgsl()}, (${first.toWgsl()}) > 0.0)';
      default:
        throw ArgumentError('Unsupported ternary op: $op');
    }
  }

  @override
  Set<VarExpr> get variables => {
    ...first.variables,
    ...second.variables,
    ...third.variables,
  };

  @override
  Set<ScalarParamExpr> get scalarParams => {
    ...first.scalarParams,
    ...second.scalarParams,
    ...third.scalarParams,
  };

  @override
  int get depth =>
      1 +
      [first.depth, second.depth, third.depth].reduce((a, b) => a > b ? a : b);

  @override
  int get nodeCount => 1 + first.nodeCount + second.nodeCount + third.nodeCount;

  @override
  String toFingerprint() =>
      '$op(${first.toFingerprint()}, ${second.toFingerprint()}, ${third.toFingerprint()})';

  @override
  String toString() => 'TernaryOpExpr($op, $first, $second, $third)';
}

/// Represents a functional bounded loop node in the expression graph.
final class LoopExpr extends Expr {
  final List<Expr> initialValues;
  final Expr maxIterations;
  final Expr conditionExpr;
  final List<Expr> stepExprs;
  final Expr resultExpr;
  final List<VarExpr> stateVars;
  final VarExpr iterVar;
  final String functionName;

  LoopExpr._({
    required this.initialValues,
    required this.maxIterations,
    required this.conditionExpr,
    required this.stepExprs,
    required this.resultExpr,
    required this.stateVars,
    required this.iterVar,
    required this.functionName,
  });

  static int _loopCounter = 0;

  factory LoopExpr({
    required List<Object> initialValues,
    required Object maxIterations,
    required Expr Function(List<Expr> state, Expr iter) condition,
    required List<Expr> Function(List<Expr> state, Expr iter) step,
    Expr Function(List<Expr> state)? result,
    String? name,
  }) {
    final parsedInit = initialValues.map(Expr.from).toList();
    final parsedMax = Expr.from(maxIterations);
    final fnName = name ?? 'fused_loop_${_loopCounter++}';
    final stateVars = List.generate(
      parsedInit.length,
      (i) => VarExpr('${fnName}_s$i'),
    );
    final iterVar = VarExpr('${fnName}_iter');

    final cond = condition(stateVars, iterVar);
    final nextSteps = step(stateVars, iterVar);
    if (nextSteps.length != parsedInit.length) {
      throw ArgumentError(
        'step() returned ${nextSteps.length} values, expected ${parsedInit.length} to match initialValues.',
      );
    }
    final res = result != null ? result(stateVars) : stateVars.first;

    return LoopExpr._(
      initialValues: parsedInit,
      maxIterations: parsedMax,
      conditionExpr: cond,
      stepExprs: nextSteps,
      resultExpr: res,
      stateVars: stateVars,
      iterVar: iterVar,
      functionName: fnName,
    );
  }

  @override
  String toWgsl() {
    final args = variables.toList();
    args.sort((a, b) => a.bindingIndex.compareTo(b.bindingIndex));
    final argList = args.map((v) => '${v.name}_val').join(', ');
    return '$functionName($argList)';
  }

  /// Generates the standalone WGSL helper function for this loop.
  String generateWgslFunction() {
    final args = variables.toList();
    args.sort((a, b) => a.bindingIndex.compareTo(b.bindingIndex));
    final params = args.map((v) => '${v.name}_val: f32').join(', ');

    final initStatements = StringBuffer();
    for (var i = 0; i < initialValues.length; i++) {
      initStatements.writeln(
        '  var ${stateVars[i].name}_val: f32 = ${initialValues[i].toWgsl()};',
      );
    }

    final stepStatements = StringBuffer();
    for (var i = 0; i < stepExprs.length; i++) {
      stepStatements.writeln(
        '    let _next_${stateVars[i].name} = ${stepExprs[i].toWgsl()};',
      );
    }
    for (var i = 0; i < stepExprs.length; i++) {
      stepStatements.writeln(
        '    ${stateVars[i].name}_val = _next_${stateVars[i].name};',
      );
    }

    return '''
fn $functionName($params) -> f32 {
$initStatements  var ${iterVar.name}_val: f32 = 0.0;
  let _max_iter: f32 = ${maxIterations.toWgsl()};
  while (${iterVar.name}_val < _max_iter && ((${conditionExpr.toWgsl()}) > 0.0)) {
$stepStatements    ${iterVar.name}_val += 1.0;
  }
  return ${resultExpr.toWgsl()};
}
''';
  }

  @override
  Set<VarExpr> get variables {
    final localNames = {...stateVars.map((v) => v.name), iterVar.name};
    final allVars = {
      for (final v in initialValues) ...v.variables,
      ...maxIterations.variables,
      ...conditionExpr.variables,
      for (final s in stepExprs) ...s.variables,
      ...resultExpr.variables,
    };
    return allVars.where((v) => !localNames.contains(v.name)).toSet();
  }

  @override
  Set<ScalarParamExpr> get scalarParams => {
    for (final v in initialValues) ...v.scalarParams,
    ...maxIterations.scalarParams,
    ...conditionExpr.scalarParams,
    for (final s in stepExprs) ...s.scalarParams,
    ...resultExpr.scalarParams,
  };

  @override
  int get depth =>
      1 +
      [
        ...initialValues.map((e) => e.depth),
        maxIterations.depth,
        conditionExpr.depth,
        ...stepExprs.map((e) => e.depth),
        resultExpr.depth,
      ].reduce((a, b) => a > b ? a : b);

  @override
  int get nodeCount =>
      1 +
      initialValues.fold<int>(0, (sum, e) => sum + e.nodeCount) +
      maxIterations.nodeCount +
      conditionExpr.nodeCount +
      stepExprs.fold<int>(0, (sum, e) => sum + e.nodeCount) +
      resultExpr.nodeCount;

  @override
  String toFingerprint() =>
      'loop(${initialValues.map((e) => e.toFingerprint()).join(",")};max:${maxIterations.toFingerprint()};cond:${conditionExpr.toFingerprint()};step:${stepExprs.map((e) => e.toFingerprint()).join(",")};res:${resultExpr.toFingerprint()})';

  @override
  String toString() =>
      'LoopExpr($functionName, inits: $initialValues, max: $maxIterations)';
}

/// Descriptor that holds the full configuration for compiling a fused kernel.
final class FusedKernelDescriptor {
  final String name;
  final Expr expression;
  final List<VarExpr> inputs;
  final List<ScalarParamExpr> scalarParams;
  final WgslDType outputDType;
  final bool isStrided;

  FusedKernelDescriptor({
    required this.name,
    Expr? expression,
    Expr? outputExpr,
    List<VarExpr>? inputs,
    List<ScalarParamExpr>? scalarParams,
    this.outputDType = WgslDType.float32,
    this.isStrided = false,
  }) : expression = expression ?? outputExpr ?? (throw ArgumentError('Either expression or outputExpr must be provided.')),
       inputs = inputs ?? _sortVariables((expression ?? outputExpr!).variables),
       scalarParams = scalarParams ?? (expression ?? outputExpr!).scalarParams.toList();

  static List<VarExpr> _sortVariables(Set<VarExpr> vars) {
    final list = vars.toList();
    list.sort((a, b) => a.bindingIndex.compareTo(b.bindingIndex));
    return list;
  }

  static Set<LoopExpr> _collectLoops(Expr expr) {
    final loops = <LoopExpr>{};
    void walk(Expr e) {
      if (e is LoopExpr) {
        loops.add(e);
        for (final init in e.initialValues) {
          walk(init);
        }
        walk(e.maxIterations);
        walk(e.conditionExpr);
        for (final s in e.stepExprs) {
          walk(s);
        }
        walk(e.resultExpr);
      } else if (e is UnaryOpExpr) {
        walk(e.child);
      } else if (e is BinaryOpExpr) {
        walk(e.left);
        walk(e.right);
      } else if (e is TernaryOpExpr) {
        walk(e.first);
        walk(e.second);
        walk(e.third);
      }
    }

    walk(expr);
    return loops;
  }

  /// Produces a deterministic unique cache key for this kernel.
  String generateCacheKey() =>
      'fused_${expression.toFingerprint()}_${outputDType.wgslType}_strided:$isStrided';

  /// Generates the resource binding descriptors.
  List<WgslBinding> createBindings() {
    final bindings = <WgslBinding>[];
    var bindIdx = 0;

    for (final v in inputs) {
      bindings.add(
        WgslBinding(
          group: 0,
          binding: bindIdx++,
          name: v.name,
          dtype: v.dtype,
          access: WgslBufferAccess.read,
        ),
      );
    }

    // Output destination buffer
    bindings.add(
      WgslBinding(
        group: 0,
        binding: bindIdx++,
        name: 'dst',
        dtype: outputDType,
        access: WgslBufferAccess.readWrite,
      ),
    );

    // Uniforms struct
    bindings.add(
      WgslBinding(
        group: 0,
        binding: bindIdx++,
        name: isStrided ? 'meta' : 'uniforms',
        isUniform: true,
        customTypeName: isStrided ? 'StridedMetadata' : 'FusedUniforms',
      ),
    );

    return bindings;
  }

  /// Generates the full WGSL compute shader source code for this fused kernel.
  String generateWgslSource({int workgroupSize = 256}) {
    final bindings = createBindings();
    final wgSize = WgslWorkgroupSize(workgroupSize, 1, 1);
    final bufferDeclarations = bindings
        .map((b) => b.toWgslDeclaration())
        .join('\n');
    final loopFunctions = _collectLoops(
      expression,
    ).map((l) => l.generateWgslFunction()).join('\n');

    if (!isStrided) {
      // Contiguous 1D fast path
      final loadStatements = inputs
          .map((v) => '  let ${v.name}_val = ${v.name}[idx];')
          .join('\n');

      final uniformFields = StringBuffer('  total_elements: u32,\n');
      for (final sp in scalarParams) {
        uniformFields.writeln('  ${sp.name}: f32,');
      }
      final fieldCount = 1 + scalarParams.length;
      final padNeeded = (4 - (fieldCount % 4)) % 4;
      for (var i = 0; i < padNeeded; i++) {
        uniformFields.writeln('  pad$i: u32,');
      }

      return '''
// WGSL JIT Fused Compute Shader: $name (Contiguous)
// Expression: ${expression.toFingerprint()}
struct FusedUniforms {
$uniformFields}

$bufferDeclarations

${WgslTemplates.mathHelpers}

$loopFunctions

@compute ${wgSize.toAttribute()}
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  var idx = global_id.x;
  let stride = num_workgroups.x * ${workgroupSize}u;
  while (idx < uniforms.total_elements) {
$loadStatements
    let result = ${expression.toWgsl()};
    dst[idx] = result;
    idx += stride;
  }
}
''';
    } else {
      // Strided multidimensional path
      final loadStatements = inputs
          .asMap()
          .entries
          .map((entry) {
            final offsetVar = entry.key == 0
                ? 'off_a'
                : (entry.key == 1 ? 'off_b' : 'off_a');
            return '  let ${entry.value.name}_val = ${entry.value.name}[$offsetVar];';
          })
          .join('\n');

      return '''
// WGSL JIT Fused Compute Shader: $name (Strided)
// Expression: ${expression.toFingerprint()}
${WgslTemplates.stridedHeader}

$bufferDeclarations

${WgslTemplates.mathHelpers}

$loopFunctions

@compute ${wgSize.toAttribute()}
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  if (idx >= meta.total_elements) {
    return;
  }

  var off_a: u32 = 0u;
  var off_b: u32 = 0u;
  var off_dst: u32 = 0u;
  flat_to_strided_offsets(idx, meta, &off_a, &off_b, &off_dst);

$loadStatements
  let result = ${expression.toWgsl()};
  dst[off_dst] = result;
}
''';
    }
  }

  /// Packages this fused kernel and its input tensors into an interactive [WebGpuWidget].
  WebGpuWidget createBrowserWidget({
    required List<dynamic> inputArrays,
    required List<dynamic> outputShape,
    String? title,
    List<dynamic> sliders = const [],
    bool renderToCanvas = false,
    int canvasWidth = 512,
    int canvasHeight = 512,
    String colorMap = 'viridis',
  }) {
    final parsedInputs = inputArrays.map((e) => e as GpuArray).toList();
    final parsedShape = outputShape.map((e) => (e as num).toInt()).toList();
    final parsedSliders = sliders.map((e) => e as WebGpuSlider).toList();

    final wgslSource = generateWgslSource();
    final inputPayloads = <GpuBufferPayload>[];

    for (var i = 0; i < parsedInputs.length; i++) {
      final arr = parsedInputs[i];
      final rawND = arr.toNDArray();
      final rawList = rawND.toList();
      final f32List = Float32List.fromList(
        rawList.map((e) => (e as num).toDouble()).toList(),
      );
      final base64Payload = base64Encode(f32List.buffer.asUint8List());
      rawND.dispose();

      inputPayloads.add(
        GpuBufferPayload(
          bindingIndex: i,
          name: inputs.length > i ? inputs[i].name : 'input_$i',
          dtype: arr.dtype,
          shape: arr.shape,
          base64Data: base64Payload,
          sizeInBytes: arr.buffer.sizeInBytes,
        ),
      );
    }

    final totalOut = parsedShape.reduce((a, b) => a * b);
    final outBytes = totalOut * outputDType.byteSize;

    final outputPayload = GpuBufferPayload(
      bindingIndex: parsedInputs.length,
      name: 'dst',
      dtype: DType.values.byName(
        outputDType.wgslType == 'f32' ? 'float32' : 'float16',
      ),
      shape: parsedShape,
      isOutput: true,
      sizeInBytes: outBytes,
    );

    final scalarList = scalarParams.toList();
    final uniformWords = <int>[totalOut];
    final byteData = ByteData(4);
    for (final sp in scalarList) {
      final matchingSlider = parsedSliders.cast<WebGpuSlider?>().firstWhere(
        (s) => s?.name == sp.name,
        orElse: () => null,
      );
      final initialVal = matchingSlider?.initialValue ?? sp.defaultValue;
      if (matchingSlider?.isInteger == true) {
        uniformWords.add(initialVal.round());
      } else {
        byteData.setFloat32(0, initialVal, Endian.little);
        uniformWords.add(byteData.getUint32(0, Endian.little));
      }
    }
    while (uniformWords.length % 4 != 0) {
      uniformWords.add(0);
    }

    final resolvedSliders = parsedSliders.map((slider) {
      if (slider.uniformWordIndex == 0) {
        final paramIdx = scalarList.indexWhere((sp) => sp.name == slider.name);
        if (paramIdx != -1) {
          return WebGpuSlider(
            name: slider.name,
            label: slider.label,
            min: slider.min,
            max: slider.max,
            initialValue: slider.initialValue,
            step: slider.step,
            isInteger: slider.isInteger,
            uniformWordIndex: paramIdx + 1,
          );
        }
      }
      return slider;
    }).toList();

    final pkg = GpuComputePipelinePackage(
      name: name,
      wgslCode: wgslSource,
      inputs: inputPayloads,
      output: outputPayload,
      uniforms: uniformWords,
      sliders: resolvedSliders,
      renderToCanvas: renderToCanvas,
      canvasWidth: canvasWidth,
      canvasHeight: canvasHeight,
      colorMap: colorMap,
    );

    return WebGpuWidget(pkg, title: title ?? name);
  }
}
