import 'wgsl_types.dart';
import 'wgsl_templates.dart';

/// Base class for all expression nodes in a fused kernel computation graph.
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
    required this.expression,
    List<VarExpr>? inputs,
    List<ScalarParamExpr>? scalarParams,
    this.outputDType = WgslDType.float32,
    this.isStrided = false,
  }) : inputs = inputs ?? _sortVariables(expression.variables),
       scalarParams = scalarParams ?? expression.scalarParams.toList();

  static List<VarExpr> _sortVariables(Set<VarExpr> vars) {
    final list = vars.toList();
    list.sort((a, b) => a.bindingIndex.compareTo(b.bindingIndex));
    return list;
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
}
