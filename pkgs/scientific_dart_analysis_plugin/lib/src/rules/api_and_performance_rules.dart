import 'package:analyzer/analysis_rule/analysis_rule.dart';
import 'package:analyzer/analysis_rule/rule_context.dart';
import 'package:analyzer/analysis_rule/rule_visitor_registry.dart';
import 'package:analyzer/dart/ast/ast.dart';
import 'package:analyzer/dart/ast/token.dart';
import 'package:analyzer/dart/ast/visitor.dart';
import 'package:analyzer/error/error.dart';

import '../type_utils.dart';

// =============================================================================
// 1. ndarray_equality_operator
// =============================================================================

/// Flags `a == b` and `a != b` when both operands are `NDArray`s.
final class EqualityOperatorRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_equality_operator',
    'NDArray.operator == checks reference identity (identical(a, b)), not '
        'structural or element-wise equality.',
    correctionMessage:
        'Use a.equals(b), arrayEqual(a, b), or allClose(a, b) for structural '
        'equality, equal(a, b) for element-wise NDArray<Boolean>, or '
        'identical(a, b) if reference identity is intended.',
    severity: DiagnosticSeverity.WARNING,
  );

  EqualityOperatorRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Avoid using == or != on NDArray instances; use .equals(), '
            'arrayEqual(), allClose(), equal(), or identical().',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _EqualityOperatorVisitor(this);
    registry.addBinaryExpression(this, visitor);
  }
}

final class _EqualityOperatorVisitor extends SimpleAstVisitor<void> {
  final EqualityOperatorRule rule;

  _EqualityOperatorVisitor(this.rule);

  @override
  void visitBinaryExpression(BinaryExpression node) {
    final op = node.operator.type;
    if (op != TokenType.EQ_EQ && op != TokenType.BANG_EQ) return;
    if (node.leftOperand is NullLiteral || node.rightOperand is NullLiteral) {
      return;
    }
    if (isNDArrayType(node.leftOperand.staticType) &&
        isNDArrayType(node.rightOperand.staticType)) {
      rule.reportAtNode(node);
    }
  }
}

// =============================================================================
// 2. ndarray_uint64_signed_comparison
// =============================================================================

/// Flags relational comparisons (`<`, `<=`, `>`, `>=`, `.compareTo`) on scalar
/// elements extracted from an `NDArray<Uint64>`.
final class Uint64SignedComparisonRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_uint64_signed_comparison',
    'Dart int is signed 64-bit; relational operators (<, <=, >, >=) on '
        'NDArray<Uint64> elements treat values >= 2^63 as negative.',
    correctionMessage:
        'Use uint64Compare(a, b) (from package:ndarray/ndarray.dart) for '
        'unsigned 64-bit comparisons.',
    severity: DiagnosticSeverity.WARNING,
  );

  Uint64SignedComparisonRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Use uint64Compare() instead of signed relational operators on '
            'NDArray<Uint64> elements.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _Uint64SignedComparisonVisitor(this);
    registry.addBinaryExpression(this, visitor);
    registry.addMethodInvocation(this, visitor);
  }
}

final class _Uint64SignedComparisonVisitor extends SimpleAstVisitor<void> {
  final Uint64SignedComparisonRule rule;

  _Uint64SignedComparisonVisitor(this.rule);

  @override
  void visitBinaryExpression(BinaryExpression node) {
    final op = node.operator.type;
    if (op != TokenType.LT &&
        op != TokenType.LT_EQ &&
        op != TokenType.GT &&
        op != TokenType.GT_EQ) {
      return;
    }
    final decls = _collectEnclosingDeclarations(node);
    if (_isUint64ElementExpression(node.leftOperand, decls) ||
        _isUint64ElementExpression(node.rightOperand, decls)) {
      rule.reportAtNode(node);
    }
  }

  @override
  void visitMethodInvocation(MethodInvocation node) {
    if (node.methodName.name != 'compareTo') return;
    final target = node.realTarget;
    if (target == null) return;
    final decls = _collectEnclosingDeclarations(node);
    final args = node.argumentList.arguments;
    if (_isUint64ElementExpression(target, decls) ||
        (args.isNotEmpty &&
            _isUint64ElementExpression(args.first.argumentExpression, decls))) {
      rule.reportAtNode(node);
    }
  }

  SubtreeDeclarations _collectEnclosingDeclarations(AstNode node) {
    final body = node.thisOrAncestorOfType<FunctionBody>();
    return body != null
        ? SubtreeDeclarations.collect(body)
        : SubtreeDeclarations();
  }

  bool _isUint64ElementExpression(
    Expression expr,
    SubtreeDeclarations decls, {
    int depth = 0,
  }) {
    if (depth > 4) return false;
    var unwrapped = unwrapParenthesized(expr);
    while (unwrapped is AsExpression) {
      unwrapped = unwrapParenthesized(unwrapped.expression);
    }
    if (unwrapped is IndexExpression) {
      if (isNDArrayUint64Type(unwrapped.realTarget.staticType)) {
        return true;
      }
    } else if (unwrapped is PropertyAccess) {
      if (unwrapped.propertyName.name == 'scalar' &&
          isNDArrayUint64Type(unwrapped.realTarget.staticType)) {
        return true;
      }
    } else if (unwrapped is PrefixedIdentifier) {
      if (unwrapped.identifier.name == 'scalar' &&
          isNDArrayUint64Type(unwrapped.prefix.staticType)) {
        return true;
      }
    } else if (unwrapped is SimpleIdentifier) {
      final element = unwrapped.element;
      if (element != null) {
        final decl = decls.variableDeclarations[element];
        final init = decl?.initializer;
        if (init != null) {
          return _isUint64ElementExpression(init, decls, depth: depth + 1);
        }
      }
    }
    return false;
  }
}

// =============================================================================
// 3. ndarray_broadcast_view_as_out
// =============================================================================

/// Flags passing a `broadcastTo` view as an `out:` argument.
final class BroadcastViewAsOutRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_broadcast_view_as_out',
    'Passing a broadcastTo() view as an out: buffer writes multiple output '
        'elements to the same stride-0 memory address.',
    correctionMessage:
        'Allocate a writable array (e.g., zeros(...) or empty(...)) or call '
        '.copy() on the broadcast view before passing it as out:.',
    severity: DiagnosticSeverity.WARNING,
  );

  BroadcastViewAsOutRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Do not pass read-only broadcast views (stride == 0) as out: '
            'arguments.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _BroadcastViewAsOutVisitor(this);
    registry.addNamedArgument(this, visitor);
  }
}

final class _BroadcastViewAsOutVisitor extends SimpleAstVisitor<void> {
  final BroadcastViewAsOutRule rule;

  _BroadcastViewAsOutVisitor(this.rule);

  @override
  void visitNamedArgument(NamedArgument node) {
    if (node.name.lexeme != 'out') return;
    final body = node.thisOrAncestorOfType<FunctionBody>();
    final decls = body != null
        ? SubtreeDeclarations.collect(body)
        : SubtreeDeclarations();
    if (isBroadcastViewExpression(node.argumentExpression, decls)) {
      rule.reportAtNode(node.argumentExpression);
    }
  }
}

// =============================================================================
// 4. ndarray_raw_generic_type
// =============================================================================

/// Flags bare `NDArray` type annotations that omit an explicit `DTypeTag` type
/// argument (`NDArray<Float64>`, `NDArray<T>`, etc.).
final class RawGenericTypeRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_raw_generic_type',
    'Raw NDArray type annotation erases the compile-time DTypeSpec tag, '
        'degrading .scalar to dynamic and disabling static return-type '
        'inference.',
    correctionMessage:
        'Specify a concrete DTypeSpec type argument such as NDArray<Float64>, '
        'NDArray<Float32>, NDArray<Int32>, or a generic type parameter '
        'NDArray<T>.',
    severity: DiagnosticSeverity.INFO,
  );

  RawGenericTypeRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Always specify a DTypeTag type argument on NDArray<T> type '
            'annotations.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _RawGenericTypeVisitor(this);
    registry.addNamedType(this, visitor);
  }
}

final class _RawGenericTypeVisitor extends SimpleAstVisitor<void> {
  final RawGenericTypeRule rule;

  _RawGenericTypeVisitor(this.rule);

  @override
  void visitNamedType(NamedType node) {
    if (node.name.lexeme != 'NDArray') return;
    if (node.typeArguments != null) return;
    if (!isNDArrayType(node.type)) return;

    // Allow `is NDArray` / `is! NDArray` runtime type tests where Dart does
    // not check generic type arguments when narrowing from Object?.
    if (node.parent is IsExpression) return;

    // Allow constructor invocations (`NDArray.zeros(...)`, `NDArray.scope(...)`)
    // where `NDArray` is parsed as a NamedType inside ConstructorName or target.
    if (node.parent is ConstructorName) return;

    rule.reportAtNode(node);
  }
}

// =============================================================================
// 5. ndarray_hot_loop_element_indexing
// =============================================================================

/// Flags scalar indexing `arr[[i, j]]` inside nested loops or calling
/// `.toList()` on an `NDArray`.
final class HotLoopElementIndexingRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_hot_loop_element_indexing',
    'Element-by-element NDArray indexing inside nested loops or .toList() heap '
        'conversion allocates per-element Dart objects and crosses FFI on each '
        'scalar.',
    correctionMessage:
        'Prefer vectorized NDArray operations, slicing views, or NDIter for '
        'multi-dimensional traversal.',
    severity: DiagnosticSeverity.INFO,
  );

  HotLoopElementIndexingRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Avoid per-element NDArray[[i, j]] indexing in nested loops and '
            '.toList() heap dumps.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _HotLoopElementIndexingVisitor(this);
    registry.addIndexExpression(this, visitor);
    registry.addMethodInvocation(this, visitor);
  }
}

final class _HotLoopElementIndexingVisitor extends SimpleAstVisitor<void> {
  final HotLoopElementIndexingRule rule;

  _HotLoopElementIndexingVisitor(this.rule);

  @override
  void visitIndexExpression(IndexExpression node) {
    if (!isNDArrayType(node.realTarget.staticType)) return;
    if (_enclosingLoopDepth(node) >= 2) {
      rule.reportAtNode(node);
    }
  }

  @override
  void visitMethodInvocation(MethodInvocation node) {
    if (node.methodName.name == 'toList' &&
        isNDArrayType(node.realTarget?.staticType)) {
      rule.reportAtNode(node);
    }
  }

  int _enclosingLoopDepth(AstNode node) {
    var depth = 0;
    AstNode? current = node.parent;
    while (current != null && current is! FunctionBody) {
      if (current is ForStatement ||
          current is WhileStatement ||
          current is DoStatement) {
        depth++;
      }
      current = current.parent;
    }
    return depth;
  }
}
