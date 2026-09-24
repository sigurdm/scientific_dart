import 'package:analyzer/analysis_rule/analysis_rule.dart';
import 'package:analyzer/analysis_rule/rule_context.dart';
import 'package:analyzer/analysis_rule/rule_visitor_registry.dart';
import 'package:analyzer/dart/ast/ast.dart';
import 'package:analyzer/dart/ast/visitor.dart';
import 'package:analyzer/dart/element/element.dart';
import 'package:analyzer/error/error.dart';

import '../type_utils.dart';

// =============================================================================
// 11. ndarray_0d_reduction_indexing_and_leak
// =============================================================================

/// Flags non-empty coordinate indexing `r[[0]]` on a 0-dimensional reduction
/// result (which throws `RangeError` at runtime) and unscoped `sum(a).scalar`
/// calls (which leak the 0-D native buffer).
final class ZeroDimReductionIndexingAndLeakRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_0d_reduction_indexing_and_leak',
    'Axis-less reductions return a 0-dimensional (shape []) NDArray: indexing '
        'with non-empty coordinates throws a RangeError, and calling '
        'sum(a).scalar outside NDArray.scope leaks the 0-D native buffer.',
    correctionMessage:
        'Use .scalar inside NDArray.scope (or dispose the 0-D reduction array '
        'after reading .scalar).',
    severity: DiagnosticSeverity.WARNING,
  );

  ZeroDimReductionIndexingAndLeakRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Prevent rank-0 RangeError when indexing axis-less reductions and '
            'off-heap leaks when chaining .scalar outside NDArray.scope.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _ZeroDimReductionVisitor(this);
    registry.addIndexExpression(this, visitor);
    registry.addPropertyAccess(this, visitor);
  }
}

final class _ZeroDimReductionVisitor extends SimpleAstVisitor<void> {
  final ZeroDimReductionIndexingAndLeakRule rule;

  _ZeroDimReductionVisitor(this.rule);

  @override
  void visitIndexExpression(IndexExpression node) {
    final body = node.thisOrAncestorOfType<FunctionBody>();
    final decls = body != null
        ? SubtreeDeclarations.collect(body)
        : SubtreeDeclarations();
    if (!isAxislessReductionExpression(node.realTarget, decls)) return;

    final indexExpr = unwrapParenthesized(node.index);
    // Indexing a 0-D array with `[0]` (non-empty list) or an int `0` fails at
    // runtime because rank == 0 requires `[]` or `.scalar`.
    if (indexExpr is ListLiteral && indexExpr.elements.isEmpty) return;
    rule.reportAtNode(node);
  }

  @override
  void visitPropertyAccess(PropertyAccess node) {
    if (node.propertyName.name != 'scalar') return;
    if (isInsideActiveScope(node)) return;
    final target = unwrapParenthesized(node.realTarget);
    if (target is MethodInvocation &&
        isAxislessReductionExpression(target, SubtreeDeclarations())) {
      rule.reportAtNode(node);
    }
  }
}

// =============================================================================
// 12. scoped_resource_chained_intermediate_leak
// =============================================================================

/// Flags nested `ScopedResource` allocations (`add(multiply(a, b), c)` or
/// `(a * b) + c`) outside `NDArray.scope` / `ResourceScope.scope`, where the
/// intermediate native buffer has no variable reference and cannot be disposed.
final class ScopedResourceChainedIntermediateLeakRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'scoped_resource_chained_intermediate_leak',
    'Chaining ScopedResource allocations outside NDArray.scope / '
        'ResourceScope.scope leaks the intermediate off-heap buffer.',
    correctionMessage:
        'Wrap the compound expression in NDArray.returning(() => ...) or '
        'NDArray.scope(() { ... }), or bind and dispose the intermediate.',
    severity: DiagnosticSeverity.WARNING,
  );

  ScopedResourceChainedIntermediateLeakRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Avoid anonymous intermediate ScopedResource allocations outside '
            'an active ResourceScope.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _ChainedIntermediateLeakVisitor(this);
    registry.addMethodInvocation(this, visitor);
    registry.addBinaryExpression(this, visitor);
  }
}

final class _ChainedIntermediateLeakVisitor extends SimpleAstVisitor<void> {
  final ScopedResourceChainedIntermediateLeakRule rule;

  _ChainedIntermediateLeakVisitor(this.rule);

  @override
  void visitMethodInvocation(MethodInvocation node) {
    if (isInsideActiveScope(node)) return;
    if (!isScopedResourceType(node.staticType)) return;
    if (isScopeInvocation(node) || isReturningInvocation(node)) return;
    if (node.methodName.name == 'where') return;

    for (final arg in node.argumentList.arguments) {
      if (arg is NamedArgument && arg.name.lexeme == 'out') continue;
      final argExpr = arg.argumentExpression;
      if (isFreshScopedResourceAllocation(argExpr)) {
        rule.reportAtNode(argExpr);
      }
    }
  }

  @override
  void visitBinaryExpression(BinaryExpression node) {
    if (isInsideActiveScope(node)) return;
    if (!isScopedResourceType(node.staticType)) return;

    if (isFreshScopedResourceAllocation(node.leftOperand)) {
      rule.reportAtNode(node.leftOperand);
    }
    if (isFreshScopedResourceAllocation(node.rightOperand)) {
      rule.reportAtNode(node.rightOperand);
    }
  }
}

// =============================================================================
// 13. nditer_coords_aliasing_or_mutation
// =============================================================================

/// Flags storing `NDIter.coords` into a collection without copying or mutating
/// `NDIter.coords` in-place (`iter.coords[0] = ...`).
final class NDIterCoordsAliasingOrMutationRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'nditer_coords_aliasing_or_mutation',
    'NDIter.coords returns an internal List<int> that moveNext() mutates '
        'in-place; storing or mutating it directly aliases or corrupts '
        'iterator coordinates.',
    correctionMessage:
        'Copy the coordinates with List.of(iter.coords) or [...iter.coords] '
        'before storing, and never mutate iter.coords in-place.',
    severity: DiagnosticSeverity.WARNING,
  );

  NDIterCoordsAliasingOrMutationRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Do not store or mutate NDIter.coords without copying via '
            'List.of(iter.coords).',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _NDIterCoordsVisitor(this);
    registry.addPropertyAccess(this, visitor);
    registry.addPrefixedIdentifier(this, visitor);
  }
}

final class _NDIterCoordsVisitor extends SimpleAstVisitor<void> {
  final NDIterCoordsAliasingOrMutationRule rule;

  _NDIterCoordsVisitor(this.rule);

  @override
  void visitPropertyAccess(PropertyAccess node) {
    if (node.propertyName.name == 'coords' &&
        isNDIterType(node.realTarget.staticType)) {
      _checkCoordsUsage(node);
    }
  }

  @override
  void visitPrefixedIdentifier(PrefixedIdentifier node) {
    if (node.identifier.name == 'coords' &&
        isNDIterType(node.prefix.staticType)) {
      _checkCoordsUsage(node);
    }
  }

  void _checkCoordsUsage(Expression coordsNode) {
    final parent = coordsNode.parent;

    // 1. In-place index mutation: `iter.coords[0] = ...`
    if (parent is IndexExpression && parent.target == coordsNode) {
      final grandParent = parent.parent;
      if (grandParent is AssignmentExpression &&
          grandParent.leftHandSide == parent) {
        rule.reportAtNode(grandParent);
        return;
      }
    }

    // 2. Mutating List method on `iter.coords`: `.clear()`, `.add()`, `.sort()`
    if (parent is MethodInvocation && parent.realTarget == coordsNode) {
      const mutatingMethods = {
        'add',
        'addAll',
        'clear',
        'fillRange',
        'insert',
        'insertAll',
        'remove',
        'removeAt',
        'removeLast',
        'removeRange',
        'removeWhere',
        'replaceRange',
        'retainWhere',
        'setAll',
        'setRange',
        'shuffle',
        'sort',
      };
      if (mutatingMethods.contains(parent.methodName.name)) {
        rule.reportAtNode(parent);
        return;
      }
    }

    // 3. Passed directly into collection `.add(iter.coords)` or `map[...]`
    // (as opposed to `List.of(iter.coords)` or `List.from(iter.coords)`).
    if (parent is ArgumentList) {
      final invocation = parent.parent;
      if (invocation is MethodInvocation) {
        final fnName = invocation.methodName.name;
        if (fnName == 'add' || fnName == 'insert' || fnName == 'putIfAbsent') {
          rule.reportAtNode(coordsNode);
        }
      }
    }
  }
}

// =============================================================================
// 14. ndarray_overlapping_view_out_or_where
// =============================================================================

/// Flags passing an `out:` argument that aliases the `where:` mask or is a
/// slice/transpose/flip view of one of the operation's input arrays.
final class OverlappingViewOutOrWhereRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_overlapping_view_out_or_where',
    'Passing an out: array that aliases the where: mask or is a non-identical '
        'view (slice/transpose/flip) of an input array corrupts in-place '
        'execution.',
    correctionMessage:
        'Use an independent output buffer or pass the exact input array '
        '(out: a) when in-place execution is supported.',
    severity: DiagnosticSeverity.WARNING,
  );

  OverlappingViewOutOrWhereRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Prevent memory aliasing between out: and where: or non-identical '
            'views of input arrays.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _OverlappingViewOutOrWhereVisitor(this);
    registry.addMethodInvocation(this, visitor);
  }
}

final class _OverlappingViewOutOrWhereVisitor extends SimpleAstVisitor<void> {
  final OverlappingViewOutOrWhereRule rule;

  _OverlappingViewOutOrWhereVisitor(this.rule);

  @override
  void visitMethodInvocation(MethodInvocation node) {
    Expression? outExpr;
    Expression? whereExpr;
    final positionalInputs = <Expression>[];

    for (final arg in node.argumentList.arguments) {
      if (arg is NamedArgument) {
        final name = arg.name.lexeme;
        if (name == 'out') {
          outExpr = arg.argumentExpression;
        } else if (name == 'where') {
          whereExpr = arg.argumentExpression;
        }
      } else {
        positionalInputs.add(arg.argumentExpression);
      }
    }

    if (outExpr == null) return;
    final body = node.thisOrAncestorOfType<FunctionBody>();
    final decls = body != null
        ? SubtreeDeclarations.collect(body)
        : SubtreeDeclarations();

    final outTrace = traceRootArrayAndView(outExpr, decls);
    if (outTrace.rootElement == null) return;

    // Case 1: `out:` shares root buffer with `where:` mask.
    if (whereExpr != null) {
      final whereTrace = traceRootArrayAndView(whereExpr, decls);
      if (whereTrace.rootElement == outTrace.rootElement) {
        rule.reportAtNode(outExpr);
        return;
      }
    }

    // Case 2: `out:` shares root buffer with a positional input AND at least
    // one of them went through a view (`slice`, `transpose`, `flip`, etc.).
    for (final input in positionalInputs) {
      final inTrace = traceRootArrayAndView(input, decls);
      if (inTrace.rootElement == outTrace.rootElement &&
          (outTrace.throughView || inTrace.throughView)) {
        rule.reportAtNode(outExpr);
        return;
      }
    }
  }
}

// =============================================================================
// 15. ndarray_lost_mutation_on_copy
// =============================================================================

/// Flags mutating an inline copy-producing method (`a.flatten().fill(0)`,
/// `a.copy()[0] = 1`, `a.astype(dt).setSlice(...)`) in an expression statement
/// where the copy is immediately discarded.
final class LostMutationOnCopyRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_lost_mutation_on_copy',
    'Mutating the result of .flatten(), .copy(), or .astype() in a statement '
        'modifies and leaks a temporary copy while leaving the original '
        'NDArray unchanged.',
    correctionMessage:
        'Use a view method such as .ravel() or .slice() to mutate the '
        'original array in-place, or assign the copy to a variable.',
    severity: DiagnosticSeverity.WARNING,
  );

  LostMutationOnCopyRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Do not call in-place mutating operations (.fill, .setSlice, []=) '
            'on discarded copies (.flatten(), .copy(), .astype()).',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _LostMutationOnCopyVisitor(this);
    registry.addExpressionStatement(this, visitor);
  }
}

final class _LostMutationOnCopyVisitor extends SimpleAstVisitor<void> {
  final LostMutationOnCopyRule rule;

  _LostMutationOnCopyVisitor(this.rule);

  @override
  void visitExpressionStatement(ExpressionStatement node) {
    final expr = unwrapParenthesized(node.expression);

    // Case 1: `a.flatten().fill(0)` or `a.copy().setSlice(...)`
    if (expr is MethodInvocation) {
      final methodName = expr.methodName.name;
      if (methodName == 'fill' || methodName == 'setSlice') {
        final target = expr.realTarget;
        if (target != null && _isInlineCopyInvocation(target)) {
          rule.reportAtNode(expr);
        }
      }
    }

    // Case 2: `a.flatten()[[0]] = 42`
    if (expr is AssignmentExpression) {
      final lhs = expr.leftHandSide;
      if (lhs is IndexExpression && _isInlineCopyInvocation(lhs.realTarget)) {
        rule.reportAtNode(expr);
      }
    }
  }

  bool _isInlineCopyInvocation(Expression expr) {
    final unwrapped = unwrapParenthesized(expr);
    if (unwrapped is! MethodInvocation) return false;
    final name = unwrapped.methodName.name;
    if (name == 'flatten' || name == 'copy') {
      return isNDArrayType(unwrapped.staticType);
    }
    if (name == 'astype' && isNDArrayType(unwrapped.staticType)) {
      return identityCastSourceExpression(unwrapped) == null;
    }
    return false;
  }
}

// =============================================================================
// 16. ndarray_from_pointer_dangling_arena
// =============================================================================

/// Flags returning an `NDArray.fromPointer(ptr, ...)` from a `try`/`finally`
/// block that frees `ptr` (`calloc.free(ptr)`, `malloc.free(ptr)`, or
/// `ScratchArena.reset(...)`) or from a `using((arena) => ...)` callback.
final class FromPointerDanglingArenaRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_from_pointer_dangling_arena',
    'NDArray.fromPointer wraps raw C memory without copying; returning it '
        'from a block that frees the pointer or resets ScratchArena creates '
        'an immediate use-after-free.',
    correctionMessage:
        'Return .copy() (detached if inside a scope) before the pointer is '
        'freed, or transfer pointer ownership via customNativeFinalizer.',
    severity: DiagnosticSeverity.WARNING,
  );

  FromPointerDanglingArenaRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Do not return an NDArray.fromPointer view over memory freed in a '
            'finally block or arena callback.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _FromPointerDanglingArenaVisitor(this);
    registry.addTryStatement(this, visitor);
    registry.addMethodInvocation(this, visitor);
  }
}

final class _FromPointerDanglingArenaVisitor extends SimpleAstVisitor<void> {
  final FromPointerDanglingArenaRule rule;

  _FromPointerDanglingArenaVisitor(this.rule);

  @override
  void visitTryStatement(TryStatement node) {
    final finallyBlock = node.finallyBlock;
    if (finallyBlock == null) return;
    final finallySrc = finallyBlock.toSource();
    if (!finallySrc.contains('.free(') &&
        !finallySrc.contains('ScratchArena.reset(')) {
      return;
    }

    final decls = SubtreeDeclarations.collect(node.body);
    node.body.accept(_TryReturnFromPointerFinder(rule, decls));
  }

  @override
  void visitMethodInvocation(MethodInvocation node) {
    if (node.methodName.name != 'using') return;
    final args = node.argumentList.arguments;
    if (args.isEmpty) return;
    final callback = unwrapParenthesized(args.first.argumentExpression);
    if (callback is! FunctionExpression) return;
    final decls = SubtreeDeclarations.collect(callback);
    callback.body.accept(_TryReturnFromPointerFinder(rule, decls));
  }
}

final class _TryReturnFromPointerFinder extends RecursiveAstVisitor<void> {
  final FromPointerDanglingArenaRule rule;
  final SubtreeDeclarations decls;

  _TryReturnFromPointerFinder(this.rule, this.decls);

  @override
  void visitReturnStatement(ReturnStatement node) {
    final expr = node.expression;
    if (expr != null && _isFromPointerWithoutCopy(expr, decls)) {
      rule.reportAtNode(expr);
    }
  }

  @override
  void visitExpressionFunctionBody(ExpressionFunctionBody node) {
    final expr = node.expression;
    if (_isFromPointerWithoutCopy(expr, decls)) {
      rule.reportAtNode(expr);
    }
  }

  bool _isFromPointerWithoutCopy(
    Expression expr,
    SubtreeDeclarations decls, {
    int depth = 0,
  }) {
    if (depth > 4) return false;
    final unwrapped = unwrapParenthesized(expr);
    if (unwrapped is MethodInvocation) {
      if (unwrapped.methodName.name == 'copy') return false;
      if (unwrapped.methodName.name == 'fromPointer') return true;
      if (unwrapped.methodName.name == 'detachToParentScope' ||
          unwrapped.methodName.name == 'detachFromScope') {
        final target = unwrapped.realTarget;
        if (target != null) {
          return _isFromPointerWithoutCopy(target, decls, depth: depth + 1);
        }
      }
    } else if (unwrapped is InstanceCreationExpression) {
      if (unwrapped.constructorName.name?.name == 'fromPointer') {
        return true;
      }
    } else if (unwrapped is SimpleIdentifier) {
      final element = unwrapped.element;
      if (element != null) {
        final init = decls.variableDeclarations[element]?.initializer;
        if (init != null) {
          return _isFromPointerWithoutCopy(init, decls, depth: depth + 1);
        }
      }
    }
    return false;
  }
}

// =============================================================================
// 17. scoped_resource_unawaited_in_scope
// =============================================================================

/// Flags unawaited `Future`-returning calls inside `NDArray.scope` /
/// `ResourceScope.scope` that reference a local `ScopedResource`, because
/// `scope.dispose()` runs before the unawaited `Future` completes.
final class UnawaitedAsyncInScopeRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'scoped_resource_unawaited_in_scope',
    'Invoking an unawaited Future-returning operation on a local '
        'ScopedResource inside NDArray.scope allows the scope to exit and '
        'dispose the resource while the async operation is still running.',
    correctionMessage:
        'Make the NDArray.scope callback async and await (or return) the '
        'Future.',
    severity: DiagnosticSeverity.WARNING,
  );

  UnawaitedAsyncInScopeRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Await or return Futures that access local ScopedResources inside '
            'NDArray.scope.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _UnawaitedAsyncInScopeVisitor(this);
    registry.addMethodInvocation(this, visitor);
  }
}

final class _UnawaitedAsyncInScopeVisitor extends SimpleAstVisitor<void> {
  final UnawaitedAsyncInScopeRule rule;

  _UnawaitedAsyncInScopeVisitor(this.rule);

  @override
  void visitMethodInvocation(MethodInvocation node) {
    if (!isScopeInvocation(node)) return;
    final args = node.argumentList.arguments;
    if (args.isEmpty) return;
    final callback = unwrapParenthesized(args.first.argumentExpression);
    if (callback is! FunctionExpression) return;

    final decls = SubtreeDeclarations.collect(callback);
    callback.body.accept(
      _UnawaitedFutureStatementFinder(rule, callback, decls),
    );
  }
}

final class _UnawaitedFutureStatementFinder extends RecursiveAstVisitor<void> {
  final UnawaitedAsyncInScopeRule rule;
  final FunctionExpression scopeCallback;
  final SubtreeDeclarations decls;

  _UnawaitedFutureStatementFinder(this.rule, this.scopeCallback, this.decls);

  @override
  void visitFunctionExpression(FunctionExpression node) {
    if (!identical(node, scopeCallback)) return;
    super.visitFunctionExpression(node);
  }

  @override
  void visitExpressionStatement(ExpressionStatement node) {
    final expr = unwrapParenthesized(node.expression);
    if (expr is AwaitExpression) return;
    if (expr is MethodInvocation && expr.methodName.name == 'unawaited') {
      if (_referencesLocalScopedResource(expr)) {
        rule.reportAtNode(expr);
      }
      return;
    }
    final type = expr.staticType;
    if (type != null && (type.isDartAsyncFuture || type.isDartAsyncFutureOr)) {
      if (_referencesLocalScopedResource(expr)) {
        rule.reportAtNode(expr);
      }
    }
  }

  bool _referencesLocalScopedResource(AstNode expr) {
    final finder = _LocalScopedResourceRefFinder(decls.elements);
    expr.accept(finder);
    return finder.found;
  }
}

final class _LocalScopedResourceRefFinder extends RecursiveAstVisitor<void> {
  final Set<Element> localElements;
  bool found = false;

  _LocalScopedResourceRefFinder(this.localElements);

  @override
  void visitSimpleIdentifier(SimpleIdentifier node) {
    final elem = node.element;
    if (elem != null &&
        localElements.contains(elem) &&
        isScopedResourceType(node.staticType)) {
      found = true;
    }
  }
}

// =============================================================================
// 18. symbolic_lambdify_or_subs_in_loop
// =============================================================================

/// Flags calling `expr.lambdify(...)` or `expr.subs(...).evalf()` inside a
/// `for`/`while`/`do` loop when `expr` was declared outside the loop.
final class SymbolicLambdifyInLoopRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'symbolic_lambdify_or_subs_in_loop',
    'Calling lambdify() or subs().evalf() on a loop-invariant symbolic '
        'expression inside a loop recompiles or re-traverses the symbolic AST '
        'on every iteration.',
    correctionMessage:
        'Hoist `final fn = expr.lambdify(vars);` outside the loop and '
        'evaluate `fn.callScalar(...)` or `fn.callArray(...)`.',
    severity: DiagnosticSeverity.INFO,
  );

  SymbolicLambdifyInLoopRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Hoist loop-invariant symbolic lambdify() compilation outside '
            'loops.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _SymbolicLambdifyInLoopVisitor(this);
    registry.addMethodInvocation(this, visitor);
  }
}

final class _SymbolicLambdifyInLoopVisitor extends SimpleAstVisitor<void> {
  final SymbolicLambdifyInLoopRule rule;

  _SymbolicLambdifyInLoopVisitor(this.rule);

  @override
  void visitMethodInvocation(MethodInvocation node) {
    final methodName = node.methodName.name;
    if (methodName != 'lambdify' && methodName != 'subs') return;
    final loopNode = _findEnclosingLoop(node);
    if (loopNode == null) return;
    final loopDecls = SubtreeDeclarations.collect(loopNode);

    final target = node.realTarget != null
        ? unwrapParenthesized(node.realTarget!)
        : (node.argumentList.arguments.isNotEmpty
              ? unwrapParenthesized(
                  node.argumentList.arguments.first.argumentExpression,
                )
              : null);
    if (target is SimpleIdentifier) {
      final elem = target.element;
      if (elem != null && !loopDecls.elements.contains(elem)) {
        rule.reportAtNode(node);
      }
    }
  }

  AstNode? _findEnclosingLoop(AstNode node) {
    AstNode? current = node.parent;
    while (current != null && current is! FunctionBody) {
      if (current is ForStatement ||
          current is WhileStatement ||
          current is DoStatement) {
        return current;
      }
      current = current.parent;
    }
    return null;
  }
}

// =============================================================================
// 19. ndarray_where_both_branches_eager_alloc
// =============================================================================

/// Flags calling `where(cond, branchA, branchB)` outside `NDArray.scope` when
/// `branchA` or `branchB` is an inline `NDArray`-allocating call, since both
/// branches are eagerly allocated and immediately leaked.
final class WhereEagerBranchAllocationRule extends AnalysisRule {
  static const LintCode code = LintCode(
    'ndarray_where_both_branches_eager_alloc',
    'where(condition, x, y) eagerly evaluates both x and y; passing inline '
        'NDArray allocations outside NDArray.scope leaks the temporary branch '
        'buffers.',
    correctionMessage:
        'Wrap the where(...) call in NDArray.returning(() => ...) or '
        'NDArray.scope(() { ... }).',
    severity: DiagnosticSeverity.WARNING,
  );

  WhereEagerBranchAllocationRule()
    : super(
        name: code.lowerCaseName,
        description:
            'Wrap eager branch allocations in where(condition, x, y) inside '
            'NDArray.scope.',
      );

  @override
  DiagnosticCode get diagnosticCode => code;

  @override
  void registerNodeProcessors(
    RuleVisitorRegistry registry,
    RuleContext context,
  ) {
    final visitor = _WhereEagerBranchVisitor(this);
    registry.addMethodInvocation(this, visitor);
  }
}

final class _WhereEagerBranchVisitor extends SimpleAstVisitor<void> {
  final WhereEagerBranchAllocationRule rule;

  _WhereEagerBranchVisitor(this.rule);

  @override
  void visitMethodInvocation(MethodInvocation node) {
    if (node.methodName.name != 'where') return;
    final args = node.argumentList.arguments;
    if (args.length < 3) return;
    if (!isNDArrayType(args[0].argumentExpression.staticType)) return;

    final branchX = args[1].argumentExpression;
    final branchY = args[2].argumentExpression;
    final xAlloc = isFreshScopedResourceAllocation(branchX);
    final yAlloc = isFreshScopedResourceAllocation(branchY);
    if ((xAlloc && yAlloc) ||
        (!isInsideActiveScope(node) && (xAlloc || yAlloc))) {
      rule.reportAtNode(node);
    }
  }
}
