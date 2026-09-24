import 'dart:io';

import 'package:analysis_server_plugin/registry.dart';
import 'package:analysis_server_plugin/src/correction/fix_generators.dart';
import 'package:analyzer/analysis_rule/analysis_rule.dart';
import 'package:analyzer/dart/analysis/analysis_context_collection.dart';
import 'package:analyzer/dart/analysis/results.dart';
import 'package:analyzer/diagnostic/diagnostic.dart';
import 'package:analyzer/error/error.dart';
import 'package:analyzer/src/lint/config.dart';
import 'package:scientific_dart_analysis_plugin/scientific_dart_analysis_plugin.dart';
import 'package:test/test.dart';

Directory _findWorkspaceRoot() {
  var dir = Directory.current;
  while (true) {
    if (File('${dir.path}/pubspec.yaml').existsSync() &&
        Directory('${dir.path}/pkgs/ndarray').existsSync()) {
      return dir;
    }
    final parent = dir.parent;
    if (parent.path == dir.path) {
      throw StateError('Could not locate math_workspace root directory.');
    }
    dir = parent;
  }
}

final class _FakePluginRegistry implements PluginRegistry {
  final List<AbstractAnalysisRule> warningRules = [];
  final List<AbstractAnalysisRule> lintRules = [];
  final Map<DiagnosticCode, List<ProducerGenerator>> fixes = {};
  final List<ProducerGenerator> assists = [];

  @override
  Iterable<AbstractAnalysisRule> enabled(Map<String, RuleConfig> ruleConfigs) =>
      [...warningRules, ...lintRules];

  @override
  void registerAssist(ProducerGenerator generator) {
    assists.add(generator);
  }

  @override
  void registerFixForRule(DiagnosticCode code, ProducerGenerator generator) {
    fixes.putIfAbsent(code, () => []).add(generator);
  }

  @override
  void registerLintRule(AbstractAnalysisRule rule) {
    lintRules.add(rule);
  }

  @override
  void registerWarningRule(AbstractAnalysisRule rule) {
    warningRules.add(rule);
  }
}

void main() {
  late Directory workspaceRoot;
  late Directory scratchDir;
  late AnalysisContextCollection collection;

  setUpAll(() {
    workspaceRoot = _findWorkspaceRoot();
    scratchDir = Directory(
      '${workspaceRoot.path}/pkgs/scientific_dart_analysis_plugin/test/_fixtures',
    )..createSync(recursive: true);
    collection = AnalysisContextCollection(includedPaths: [scratchDir.path]);
  });

  tearDownAll(() {
    if (scratchDir.existsSync()) {
      scratchDir.deleteSync(recursive: true);
    }
  });

  var counter = 0;
  Future<List<Diagnostic>> analyzeCode(
    String source, {
    List<AnalysisRule>? rules,
  }) async {
    final file = File('${scratchDir.path}/case_${counter++}.dart');
    file.writeAsStringSync(source);
    final context = collection.contextFor(file.path);
    context.changeFile(file.path);
    await context.applyPendingFileChanges();
    final result = await context.currentSession.getResolvedUnit(file.path);
    if (result is! ResolvedUnitResult) {
      fail('Expected ResolvedUnitResult, got $result');
    }
    final compileErrors = result.diagnostics.where(
      (d) => d.diagnosticCode.type == DiagnosticType.COMPILE_TIME_ERROR,
    );
    expect(
      compileErrors,
      isEmpty,
      reason: 'Fixture had compile errors:\n${compileErrors.join('\n')}',
    );
    return runScientificDartLintsOnUnit(result, rules: rules);
  }

  group('Plugin Registration', () {
    test(
      'ScientificDartAnalysisPlugin registers all 19 rules and quick fixes',
      () {
        final registry = _FakePluginRegistry();
        final plugin = ScientificDartAnalysisPlugin();
        plugin.register(registry);

        final registeredNames = registry.warningRules
            .map((r) => r.name)
            .toSet();
        expect(
          registeredNames,
          containsAll([
            'ndarray_unescaped_scope_return',
            'ndarray_view_lifecycle_misuse',
            'ndarray_equality_operator',
            'ndarray_loop_reassignment_leak',
            'ndarray_isolate_capture_and_borrow',
            'ndarray_uint64_signed_comparison',
            'ndarray_broadcast_view_as_out',
            'ndarray_identity_cast_dispose',
            'ndarray_raw_generic_type',
            'ndarray_hot_loop_element_indexing',
            'ndarray_0d_reduction_indexing_and_leak',
            'scoped_resource_chained_intermediate_leak',
            'nditer_coords_aliasing_or_mutation',
            'ndarray_overlapping_view_out_or_where',
            'ndarray_lost_mutation_on_copy',
            'ndarray_from_pointer_dangling_arena',
            'scoped_resource_unawaited_in_scope',
            'symbolic_lambdify_or_subs_in_loop',
            'ndarray_where_both_branches_eager_alloc',
          ]),
        );
        expect(registry.fixes, isNotEmpty);
      },
    );
  });

  group('Scope & Lifecycle Rules', () {
    test(
      '1. ndarray_unescaped_scope_return catches unescaped returns and allows detached returns',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

NDArray<Float64> badReturn() {
  return NDArray.scope(() {
    final a = NDArray.zeros([4], DType.float64);
    return sin(a); // VIOLATION 1: unescaped fresh array
  });
}

NDArray<Float64> badLocalReturn() {
  return NDArray.scope(() {
    final a = NDArray.ones([4], DType.float64);
    return a; // VIOLATION 2: unescaped local variable
  });
}

NDArray<Float64> goodDetachedReturn() {
  return NDArray.scope(() {
    final a = NDArray.zeros([4], DType.float64);
    return sin(a).detachToParentScope(); // OK
  });
}

NDArray<Float64> goodReturningHelper() {
  return NDArray.returning(() {
    final a = NDArray.zeros([4], DType.float64);
    return sin(a); // OK: NDArray.returning detaches automatically
  });
}

NDArray<Float64> goodOuterParamReturn(NDArray<Float64> out) {
  return NDArray.scope(() {
    final temp = NDArray.ones([4], DType.float64);
    add(temp, temp, out: out);
    return out; // OK: `out` was declared outside the scope
  });
}
''',
          rules: [UnescapedScopeReturnRule()],
        );

        expect(diagnostics, hasLength(2));
        expect(
          diagnostics.every(
            (d) =>
                d.diagnosticCode.lowerCaseName ==
                'ndarray_unescaped_scope_return',
          ),
          isTrue,
        );
      },
    );

    test(
      '2. ndarray_view_lifecycle_misuse catches view return in NDArray.returning and view detach/dispose',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

NDArray<Float64> badReturningSlice() {
  return NDArray.returning(() {
    final a = NDArray.zeros([4, 4], DType.float64);
    return a.slice([Slice(start: 0, stop: 2)]); // VIOLATION 1: returning view from NDArray.returning
  });
}

NDArray<Float64> badDetachTranspose() {
  return NDArray.scope(() {
    final a = NDArray.ones([3, 3], DType.float64);
    final t = a.transpose();
    return t.detachToParentScope(); // VIOLATION 2: detaching a transposed view
  });
}

void badDisposeReshape() {
  final a = NDArray.ones([6], DType.float64);
  a.reshape([2, 3]).dispose(); // VIOLATION 3: disposing a view is a no-op
  a.dispose();
}

NDArray<Float64> goodCopyBeforeReturn() {
  return NDArray.returning(() {
    final a = NDArray.zeros([4, 4], DType.float64);
    return a.slice([Slice(start: 0, stop: 2)]).copy(); // OK: .copy() materializes an owning array
  });
}
''',
          rules: [ViewLifecycleMisuseRule()],
        );

        expect(diagnostics, hasLength(3));
        expect(
          diagnostics.every(
            (d) =>
                d.diagnosticCode.lowerCaseName ==
                'ndarray_view_lifecycle_misuse',
          ),
          isTrue,
        );
      },
    );

    test(
      '3. ndarray_loop_reassignment_leak flags unscoped loop reassignment without out: or dispose()',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

void leakLoop(NDArray<Float64> delta) {
  var x = NDArray.zeros([10], DType.float64);
  for (var i = 0; i < 10; i++) {
    x = add(x, delta); // VIOLATION: leaks previous `x` every iteration
  }
  x.dispose();
}

void safeInPlaceLoop(NDArray<Float64> delta) {
  final x = NDArray.zeros([10], DType.float64);
  for (var i = 0; i < 10; i++) {
    add(x, delta, out: x); // OK
  }
  x.dispose();
}

void safeDisposedLoop(NDArray<Float64> delta) {
  var x = NDArray.zeros([10], DType.float64);
  for (var i = 0; i < 10; i++) {
    final prev = x;
    x = add(prev, delta);
    prev.dispose(); // OK: previous buffer explicitly disposed
  }
  x.dispose();
}
''',
          rules: [LoopReassignmentLeakRule()],
        );

        expect(diagnostics, hasLength(1));
        expect(
          diagnostics.first.diagnosticCode.lowerCaseName,
          equals('ndarray_loop_reassignment_leak'),
        );
      },
    );

    test(
      '4. ndarray_identity_cast_dispose flags unguarded .dispose() on astype(copy: false)',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

void badCastDispose(NDArray<Float64> a) {
  final casted = a.astype(DType.float64, copy: false);
  casted.dispose(); // VIOLATION: disposes `a` when dtype already matches!
}

void goodGuardedCastDispose(NDArray<Float64> a) {
  final casted = a.astype(DType.float64, copy: false);
  if (!identical(casted, a)) {
    casted.dispose(); // OK: guarded by !identical
  }
}
''',
          rules: [IdentityCastDisposeRule()],
        );

        expect(diagnostics, hasLength(1));
        expect(
          diagnostics.first.diagnosticCode.lowerCaseName,
          equals('ndarray_identity_cast_dispose'),
        );
      },
    );

    test(
      '5. ndarray_isolate_capture_and_borrow flags raw NDArray capture and sync toSendableBorrow',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'dart:isolate';
import 'package:ndarray/ndarray.dart';

Future<void> badIsolateCapture() async {
  final a = NDArray.ones([10], DType.float64);
  await Isolate.run(() {
    return a.size; // VIOLATION 1: raw NDArray `a` captured across isolate
  });
  a.dispose();
}

void badSyncScopeBorrow() {
  NDArray.scope(() {
    final a = NDArray.ones([10], DType.float64);
    final borrowed = a.toSendableBorrow(); // VIOLATION 2: synchronous scope exits immediately!
    Isolate.run(() => borrowed.materializeView().size);
  });
}

Future<int> goodAsyncAwaitedBorrow() async {
  return await NDArray.scope(() async {
    final a = NDArray.ones([10], DType.float64);
    final borrowed = a.toSendableBorrow(); // OK: scope is async and awaits Isolate.run
    return await Isolate.run(() => borrowed.materializeView().size);
  });
}
''',
          rules: [IsolateCaptureAndBorrowRule()],
        );

        expect(diagnostics, hasLength(2));
        expect(
          diagnostics.every(
            (d) =>
                d.diagnosticCode.lowerCaseName ==
                'ndarray_isolate_capture_and_borrow',
          ),
          isTrue,
        );
      },
    );
  });

  group('API & Performance Rules', () {
    test(
      '6. ndarray_equality_operator flags == and != between NDArrays, allows null and .equals()',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

bool checkArrays(NDArray<Float64> a, NDArray<Float64> b, NDArray<Float64>? maybeNull) {
  final badEq = a == b; // VIOLATION 1
  final badNeq = a != b; // VIOLATION 2
  final okNull = maybeNull == null; // OK
  final okEquals = a.equals(b); // OK
  final okIdentical = identical(a, b); // OK
  return badEq && badNeq && okNull && okEquals && okIdentical;
}
''',
          rules: [EqualityOperatorRule()],
        );

        expect(diagnostics, hasLength(2));
        expect(
          diagnostics.every(
            (d) =>
                d.diagnosticCode.lowerCaseName == 'ndarray_equality_operator',
          ),
          isTrue,
        );
      },
    );

    test(
      '7. ndarray_uint64_signed_comparison flags <, <=, >, >= on NDArray<Uint64> elements',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

bool checkUint64(NDArray<Uint64> u64, NDArray<Int64> i64) {
  final badIndexCmp = (u64[[0]] as int) > 0; // VIOLATION 1
  final elem = u64.scalar;
  final badScalarCmp = elem < 100; // VIOLATION 2
  final okUint64Compare = uint64Compare(u64.scalar, 0) > 0; // OK
  final okInt64Cmp = i64.scalar > 0; // OK: Int64 is signed
  return badIndexCmp && badScalarCmp && okUint64Compare && okInt64Cmp;
}
''',
          rules: [Uint64SignedComparisonRule()],
        );

        expect(diagnostics, hasLength(2));
        expect(
          diagnostics.every(
            (d) =>
                d.diagnosticCode.lowerCaseName ==
                'ndarray_uint64_signed_comparison',
          ),
          isTrue,
        );
      },
    );

    test(
      '8. ndarray_broadcast_view_as_out flags broadcastTo passed as out:',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

void testBroadcastOut(NDArray<Float64> a) {
  final bcast = broadcastTo(NDArray.zeros([1], DType.float64), [4]);
  sin(a, out: bcast); // VIOLATION: stride-0 broadcast view as out
}
''',
          rules: [BroadcastViewAsOutRule()],
        );

        expect(diagnostics, hasLength(1));
        expect(
          diagnostics.first.diagnosticCode.lowerCaseName,
          equals('ndarray_broadcast_view_as_out'),
        );
      },
    );

    test(
      '9. ndarray_raw_generic_type flags bare NDArray type annotations',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

void rawParam(NDArray a, NDArray<Float64> typed) {
  // `NDArray a` is VIOLATION 1; `is NDArray` type check is allowed.
  if (typed is NDArray) {
    return;
  }
}
''',
          rules: [RawGenericTypeRule()],
        );

        expect(diagnostics, hasLength(1));
        expect(
          diagnostics.first.diagnosticCode.lowerCaseName,
          equals('ndarray_raw_generic_type'),
        );
      },
    );

    test(
      '10. ndarray_hot_loop_element_indexing flags nested loop indexing and .toList()',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

double sumSlow(NDArray<Float64> m) {
  final dump = m.toList(); // VIOLATION 1: .toList() heap dump
  var total = 0.0;
  for (var i = 0; i < 3; i++) {
    for (var j = 0; j < 3; j++) {
      total += m[[i, j]] as double; // VIOLATION 2: nested loop element indexing
    }
  }
  return total + dump.length;
}
''',
          rules: [HotLoopElementIndexingRule()],
        );

        expect(diagnostics, hasLength(2));
        expect(
          diagnostics.every(
            (d) =>
                d.diagnosticCode.lowerCaseName ==
                'ndarray_hot_loop_element_indexing',
          ),
          isTrue,
        );
      },
    );
  });

  group('Advanced Memory, View, & Symbolic Rules', () {
    test(
      '11. ndarray_0d_reduction_indexing_and_leak catches non-empty coordinate indexing on 0-D reductions and unscoped .scalar',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

double bad0d(NDArray<Float64> a) {
  final r = sum(a);
  final x = r[[0]] as double; // VIOLATION 1: non-empty indexing on 0-D reduction
  final y = max(a).scalar; // VIOLATION 2: unscoped .scalar leaks 0-D buffer
  final okEmpty = r[[]] as double; // OK
  r.dispose();
  return x + y + okEmpty;
}

double goodScopedScalar(NDArray<Float64> a) {
  return NDArray.scope(() => sum(a).scalar); // OK: inside NDArray.scope
}

double goodAxisReduction(NDArray<Float64> a) {
  final r = sum(a, axis: 0);
  final x = r[[0]] as double; // OK: 1-D reduction when axis is specified
  r.dispose();
  return x;
}
''',
          rules: [ZeroDimReductionIndexingAndLeakRule()],
        );

        expect(diagnostics, hasLength(2));
        expect(
          diagnostics.every(
            (d) =>
                d.diagnosticCode.lowerCaseName ==
                'ndarray_0d_reduction_indexing_and_leak',
          ),
          isTrue,
        );
      },
    );

    test(
      '12. scoped_resource_chained_intermediate_leak catches nested allocations outside scope',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

NDArray<Float64> badChained(NDArray<Float64> a, NDArray<Float64> b, NDArray<Float64> c) {
  final r1 = add(multiply(a, b), c); // VIOLATION 1: multiply(a, b) leaked
  final r2 = (a * b) + c; // VIOLATION 2: (a * b) leaked
  r1.dispose();
  return r2;
}

NDArray<Float64> goodScopedChained(NDArray<Float64> a, NDArray<Float64> b, NDArray<Float64> c) {
  return NDArray.returning(() => (a * b) + c); // OK: inside NDArray.returning
}
''',
          rules: [ScopedResourceChainedIntermediateLeakRule()],
        );

        expect(diagnostics, hasLength(2));
        expect(
          diagnostics.every(
            (d) =>
                d.diagnosticCode.lowerCaseName ==
                'scoped_resource_chained_intermediate_leak',
          ),
          isTrue,
        );
      },
    );

    test(
      '13. nditer_coords_aliasing_or_mutation catches storing or mutating NDIter.coords directly',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

void badIterCoords(NDArray<Float64> a) {
  final iter = NDIter(a);
  final saved = <List<int>>[];
  while (iter.moveNext()) {
    saved.add(iter.coords); // VIOLATION 1: aliasing mutable coords list
    iter.coords[0] = 99; // VIOLATION 2: mutating iterator coords in-place
    saved.add(List.of(iter.coords)); // OK: defensive copy
  }
}
''',
          rules: [NDIterCoordsAliasingOrMutationRule()],
        );

        expect(diagnostics, hasLength(2));
        expect(
          diagnostics.every(
            (d) =>
                d.diagnosticCode.lowerCaseName ==
                'nditer_coords_aliasing_or_mutation',
          ),
          isTrue,
        );
      },
    );

    test(
      '14. ndarray_overlapping_view_out_or_where catches overlapping views between inputs and out:',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

void badOverlap(NDArray<Float64> a) {
  final t = a.transpose();
  add(a, t, out: a); // VIOLATION 1: t is a transposed view of out: a
  subtract(a.slice([Slice(start: 1)]), a, out: a); // VIOLATION 2: slice overlaps out: a
  add(a, a, out: a); // OK: exact same stride/offset in-place
}
''',
          rules: [OverlappingViewOutOrWhereRule()],
        );

        expect(diagnostics, hasLength(2));
        expect(
          diagnostics.every(
            (d) =>
                d.diagnosticCode.lowerCaseName ==
                'ndarray_overlapping_view_out_or_where',
          ),
          isTrue,
        );
      },
    );

    test(
      '15. ndarray_lost_mutation_on_copy catches mutating temporary copies',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

void badCopyMutation(NDArray<Float64> a) {
  a.astype(DType.float32)[[0]] = 1.0; // VIOLATION 1: index assignment on astype copy
  a.flatten().fill(0.0); // VIOLATION 2: .fill() on flatten() copy
  a.slice([Slice(start: 0, stop: 2)]).fill(0.0); // OK: slice() returns a view
}
''',
          rules: [LostMutationOnCopyRule()],
        );

        expect(diagnostics, hasLength(2));
        expect(
          diagnostics.every(
            (d) =>
                d.diagnosticCode.lowerCaseName ==
                'ndarray_lost_mutation_on_copy',
          ),
          isTrue,
        );
      },
    );

    test(
      '16. ndarray_from_pointer_dangling_arena catches NDArray.fromPointer escaping ScratchArena',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'dart:ffi' as ffi;
import 'package:ndarray/ndarray.dart';

NDArray<Float64> badArenaReturn() {
  final marker = ScratchArena.marker;
  try {
    final ptr = ScratchArena.allocate<ffi.Double>(32);
    return NDArray.fromPointer(ptr.cast(), [4], DType.float64); // VIOLATION: dangling pointer
  } finally {
    ScratchArena.reset(marker);
  }
}

NDArray<Float64> goodArenaCopyReturn() {
  final marker = ScratchArena.marker;
  try {
    final ptr = ScratchArena.allocate<ffi.Double>(32);
    return NDArray.fromPointer(ptr.cast(), [4], DType.float64).copy(); // OK: copied before arena unwinds
  } finally {
    ScratchArena.reset(marker);
  }
}
''',
          rules: [FromPointerDanglingArenaRule()],
        );

        expect(diagnostics, hasLength(1));
        expect(
          diagnostics.first.diagnosticCode.lowerCaseName,
          equals('ndarray_from_pointer_dangling_arena'),
        );
      },
    );

    test(
      '17. scoped_resource_unawaited_in_scope catches unawaited Future statements in NDArray.scope',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'dart:async';
import 'package:ndarray/ndarray.dart';

Future<void> helperAsync(NDArray<Float64> a) async {}

void badScopeAsync() {
  NDArray.scope(() {
    final a = NDArray.zeros([4], DType.float64);
    helperAsync(a); // VIOLATION 1: unawaited Future expression statement using local ScopedResource
    unawaited(helperAsync(a)); // VIOLATION 2: explicitly unawaited Future using local ScopedResource
  });
}

Future<void> goodScopeAsync() async {
  await NDArray.scope(() async {
    final a = NDArray.zeros([4], DType.float64);
    await helperAsync(a); // OK: awaited inside async scope
  });
}
''',
          rules: [UnawaitedAsyncInScopeRule()],
        );

        expect(diagnostics, hasLength(2));
        expect(
          diagnostics.every(
            (d) =>
                d.diagnosticCode.lowerCaseName ==
                'scoped_resource_unawaited_in_scope',
          ),
          isTrue,
        );
      },
    );

    test(
      '18. symbolic_lambdify_or_subs_in_loop catches Expr.lambdify and Expr.subs() inside loops',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:symbolic_dart/symbolic_dart.dart';

double badSymbolicLoop(Expr expr, Expr x) {
  var sum = 0.0;
  for (var i = 0; i < 10; i++) {
    final fn = expr.lambdify([x]); // VIOLATION 1: lambdify inside loop
    sum += fn.callScalar([i.toDouble()]);
    final subbed = expr.subs({x: Expr.real(i.toDouble())}); // VIOLATION 2: subs() on loop-invariant expr inside loop
    subbed.dispose();
  }
  return sum;
}

double goodSymbolicHoisted(Expr expr, Expr x) {
  final fn = expr.lambdify([x]); // OK: hoisted outside loop
  var sum = 0.0;
  for (var i = 0; i < 10; i++) {
    sum += fn.callScalar([i.toDouble()]);
  }
  return sum;
}
''',
          rules: [SymbolicLambdifyInLoopRule()],
        );

        expect(diagnostics, hasLength(2));
        expect(
          diagnostics.every(
            (d) =>
                d.diagnosticCode.lowerCaseName ==
                'symbolic_lambdify_or_subs_in_loop',
          ),
          isTrue,
        );
      },
    );

    test(
      '19. ndarray_where_both_branches_eager_alloc catches eager branch allocations in where()',
      () async {
        final diagnostics = await analyzeCode(
          '''
import 'package:ndarray/ndarray.dart';

NDArray<Float64> badWhere(NDArray<Boolean> cond, NDArray<Float64> a, NDArray<Float64> b) {
  return NDArray.returning(() {
    return where(cond, sin(a), cos(b)) as NDArray<Float64>; // VIOLATION: both branches eagerly allocate
  });
}

NDArray<Float64> goodWhere(NDArray<Boolean> cond, NDArray<Float64> a, NDArray<Float64> b) {
  return NDArray.returning(() {
    return where(cond, a, b) as NDArray<Float64>; // OK: existing arrays
  });
}
''',
          rules: [WhereEagerBranchAllocationRule()],
        );

        expect(diagnostics, hasLength(1));
        expect(
          diagnostics.first.diagnosticCode.lowerCaseName,
          equals('ndarray_where_both_branches_eager_alloc'),
        );
      },
    );
  });
}
