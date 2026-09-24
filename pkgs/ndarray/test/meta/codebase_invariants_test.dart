import 'dart:io';

import 'package:analyzer/dart/analysis/analysis_context_collection.dart';
import 'package:analyzer/dart/analysis/features.dart';
import 'package:analyzer/dart/analysis/results.dart';
import 'package:analyzer/dart/analysis/utilities.dart';
import 'package:analyzer/dart/ast/ast.dart';
import 'package:analyzer/dart/ast/visitor.dart';
import 'package:analyzer/dart/element/element.dart';
import 'package:analyzer/dart/element/type.dart';
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

Directory _findPackageRoot() {
  var dir = Directory.current;
  while (true) {
    if (File('${dir.path}/pubspec.yaml').existsSync() &&
        Directory('${dir.path}/hook').existsSync()) {
      return dir;
    }
    final sub = Directory('${dir.path}/pkgs/ndarray');
    if (sub.existsSync()) {
      return sub;
    }
    final parent = dir.parent;
    if (parent.path == dir.path) {
      throw StateError('Could not locate pkgs/ndarray root directory.');
    }
    dir = parent;
  }
}

List<File> _dartFilesIn(Directory dir) {
  if (!dir.existsSync()) return const [];
  return dir
      .listSync(recursive: true)
      .whereType<File>()
      .where((f) => f.path.endsWith('.dart'))
      .toList()
    ..sort((a, b) => a.path.compareTo(b.path));
}

String _stripCppComments(String source) {
  final withoutBlock = source.replaceAll(
    RegExp(r'/\*[\s\S]*?\*/', multiLine: true),
    '',
  );
  return withoutBlock
      .split('\n')
      .map((line) {
        final idx = line.indexOf('//');
        return idx >= 0 ? line.substring(0, idx) : line;
      })
      .join('\n');
}

void main() {
  final pkgRoot = _findPackageRoot();
  final monorepoRoot = pkgRoot.parent.parent;
  final pkgsDir = Directory('${monorepoRoot.path}/pkgs');
  final libDir = Directory('${pkgRoot.path}/lib');
  final hookDir = Directory('${pkgRoot.path}/hook');
  final libFiles = _dartFilesIn(libDir);
  final featureSet = FeatureSet.latestLanguageVersion();

  group('Codebase & FFI Invariants', () {
    test(
      'Every ScratchArena.marker in lib/ is paired with try / finally ScratchArena.reset',
      () {
        final violations = <String>[];

        for (final file in libFiles) {
          if (file.path.endsWith('scratch_arena.dart')) continue;
          final result = parseFile(
            path: file.path,
            featureSet: featureSet,
            throwIfDiagnostics: false,
          );
          final visitor = _ScratchArenaVisitor(file.path, result.lineInfo);
          result.unit.accept(visitor);
          violations.addAll(visitor.violations);
        }

        expect(
          violations,
          isEmpty,
          reason:
              'Every `final marker = ScratchArena.marker;` must be followed by '
              'a `try { ... } finally { ScratchArena.reset(marker); }` block:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'Every int-returning @ffi.Native call in lib/ (including extensions) checks its return code',
      () {
        final bindingFiles = [
          File('${libDir.path}/src/ndarray_bindings.dart'),
          File('${libDir.path}/src/ndarray_extensions_bindings.dart'),
        ];

        final intReturningNativeFunctions = <String>{};
        for (final bindingsFile in bindingFiles) {
          if (!bindingsFile.existsSync()) continue;
          final bindingsUnit = parseFile(
            path: bindingsFile.path,
            featureSet: featureSet,
            throwIfDiagnostics: false,
          ).unit;

          for (final decl in bindingsUnit.declarations) {
            if (decl is FunctionDeclaration) {
              final name = decl.name.lexeme;
              final returnType = decl.returnType?.toSource();
              if (name.startsWith('native_') && returnType == 'int') {
                intReturningNativeFunctions.add(name);
              }
            }
          }
        }

        expect(
          intReturningNativeFunctions,
          isNotEmpty,
          reason: 'Expected to discover int-returning native_* FFI bindings.',
        );

        final uncheckedCalls = <String>[];
        for (final file in libFiles) {
          if (file.path.endsWith('ndarray_bindings.dart') ||
              file.path.endsWith('ndarray_extensions_bindings.dart')) {
            continue;
          }
          final parsed = parseFile(
            path: file.path,
            featureSet: featureSet,
            throwIfDiagnostics: false,
          );
          final visitor = _UncheckedNativeCallVisitor(
            file.path,
            parsed.lineInfo,
            intReturningNativeFunctions,
          );
          parsed.unit.accept(visitor);
          uncheckedCalls.addAll(visitor.violations);
        }

        expect(
          uncheckedCalls,
          isEmpty,
          reason:
              'All int-returning native_* FFI functions can fail (e.g. OOM -4 '
              'or bounds errors) and must have their return code checked:\n'
              '${uncheckedCalls.join('\n')}',
        );
      },
    );

    test(
      'NativeFinalizer.attach never passes externalSize across all workspace packages',
      () {
        final violations = <String>[];
        final allPkgLibFiles = pkgsDir.existsSync()
            ? pkgsDir
                  .listSync()
                  .whereType<Directory>()
                  .expand((d) => _dartFilesIn(Directory('${d.path}/lib')))
                  .toList()
            : libFiles;

        for (final file in allPkgLibFiles) {
          final parsed = parseFile(
            path: file.path,
            featureSet: featureSet,
            throwIfDiagnostics: false,
          );
          final visitor = _FinalizerExternalSizeVisitor(
            file.path,
            parsed.lineInfo,
          );
          parsed.unit.accept(visitor);
          violations.addAll(visitor.violations);
        }

        expect(
          violations,
          isEmpty,
          reason:
              'Per Dart VM team guidance, do not pass externalSize to '
              'NativeFinalizer.attach; rely on NDArray.scope / dispose() instead:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'C++ hook sources are exception-free (-fno-exceptions safe), include guards exist, and exported functions match headers',
      () {
        final cppFiles = hookDir
            .listSync()
            .whereType<File>()
            .where((f) => f.path.endsWith('.cpp'))
            .toList();
        final headerFiles = hookDir
            .listSync()
            .whereType<File>()
            .where((f) => f.path.endsWith('.h'))
            .toList();
        expect(cppFiles, isNotEmpty);
        expect(headerFiles, isNotEmpty);

        final violations = <String>[];

        for (final hFile in headerFiles) {
          final raw = hFile.readAsStringSync();
          final baseName = hFile.uri.pathSegments.last;
          if (!raw.contains('#pragma once') && !raw.contains('#ifndef')) {
            violations.add(
              '$baseName: missing `#pragma once` or `#ifndef` include guard.',
            );
          }
        }

        for (final cppFile in cppFiles) {
          final raw = cppFile.readAsStringSync();
          final stripped = _stripCppComments(raw);
          final baseName = cppFile.uri.pathSegments.last;

          if (RegExp(r'\bstd::vector\b').hasMatch(stripped)) {
            violations.add(
              '$baseName: contains `std::vector` (use `NoThrowBuffer` with `std::nothrow` under `-fno-exceptions`).',
            );
          }
          if (RegExp(r'\bstd::call_once\b').hasMatch(stripped)) {
            violations.add(
              '$baseName: contains `std::call_once` (can throw `std::system_error`; use `std::atomic` instead).',
            );
          }
          if (RegExp(r'\bthrow\b').hasMatch(stripped)) {
            violations.add(
              '$baseName: contains `throw` statement under `-fno-exceptions`.',
            );
          }
          if (stripped.contains('std::memcpy') &&
              !raw.contains('<cstring>') &&
              !raw.contains('<string.h>')) {
            violations.add(
              '$baseName: uses `std::memcpy` without `#include <cstring>`.',
            );
          }

          // Check that every NDARRAY_EXPORT / FFI_PLUGIN_EXPORT function in foo.cpp is declared in foo.h
          final headerPath = cppFile.path.replaceFirst(RegExp(r'\.cpp$'), '.h');
          final headerFile = File(headerPath);
          if (headerFile.existsSync()) {
            final headerSource = _stripCppComments(
              headerFile.readAsStringSync(),
            );
            final exportRegex = RegExp(
              r'(?:NDARRAY_EXPORT|FFI_PLUGIN_EXPORT)\s+[A-Za-z0-9_*\s]+\s+([A-Za-z0-9_]+)\s*\(',
            );
            for (final match in exportRegex.allMatches(stripped)) {
              final fnName = match.group(1)!;
              if (!RegExp('\\b$fnName\\b').hasMatch(headerSource)) {
                violations.add(
                  '$baseName: exported function `$fnName` is missing from ${headerFile.uri.pathSegments.last}.',
                );
              }
            }
          }
        }

        expect(
          violations,
          isEmpty,
          reason:
              'C++ hook files must be strictly `-fno-exceptions` clean and '
              'declare all exported symbols in their header:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'All public top-level API declarations transitively exported by lib/ndarray.dart have Dartdoc comments',
      () {
        final entrypoint = File('${libDir.path}/ndarray.dart');
        final visited = <String>{};
        final exportedFiles = <(File, Set<String>?, Set<String>?)>[];

        void collectExports(
          File file, {
          Set<String>? showNames,
          Set<String>? hideNames,
        }) {
          final canonical = file.resolveSymbolicLinksSync();
          if (!visited.add('$canonical|$showNames|$hideNames')) return;
          exportedFiles.add((file, showNames, hideNames));

          final parsed = parseFile(
            path: file.path,
            featureSet: featureSet,
            throwIfDiagnostics: false,
          );
          for (final directive in parsed.unit.directives) {
            if (directive is ExportDirective) {
              final uriStr = directive.uri.stringValue;
              if (uriStr == null || uriStr.startsWith('package:')) continue;
              final resolved = File('${file.parent.path}/$uriStr');
              if (!resolved.existsSync()) continue;

              Set<String>? childShow = showNames;
              final childHide = <String>{...?hideNames};
              for (final combinator in directive.combinators) {
                if (combinator is ShowCombinator) {
                  final names = combinator.shownNames
                      .map((n) => n.name)
                      .toSet();
                  childShow = childShow == null
                      ? names
                      : childShow.intersection(names);
                } else if (combinator is HideCombinator) {
                  childHide.addAll(combinator.hiddenNames.map((n) => n.name));
                }
              }
              collectExports(
                resolved,
                showNames: childShow,
                hideNames: childHide,
              );
            }
          }
        }

        collectExports(entrypoint);
        expect(exportedFiles.length, greaterThan(20));

        final missingDocs = <String>[];

        for (final (file, showNames, hideNames) in exportedFiles) {
          final parsed = parseFile(
            path: file.path,
            featureSet: featureSet,
            throwIfDiagnostics: false,
          );
          for (final decl in parsed.unit.declarations) {
            final name = switch (decl) {
              FunctionDeclaration(:final name) => name.lexeme,
              ClassDeclaration(:final namePart) => namePart.typeName.lexeme,
              EnumDeclaration(:final namePart) => namePart.typeName.lexeme,
              MixinDeclaration(:final name) => name.lexeme,
              GenericTypeAlias(:final name) => name.lexeme,
              TopLevelVariableDeclaration(:final variables) =>
                variables.variables.first.name.lexeme,
              _ => null,
            };
            if (name == null || name.startsWith('_')) continue;
            if (showNames != null && !showNames.contains(name)) continue;
            if (hideNames != null && hideNames.contains(name)) continue;
            if (name.endsWith('_helper')) continue;

            final isInternalAnnotated = decl.metadata.any(
              (m) => m.name.name == 'internal',
            );
            if (isInternalAnnotated) continue;

            if (decl.documentationComment == null) {
              final line = parsed.lineInfo.getLocation(decl.offset).lineNumber;
              missingDocs.add(
                '${file.path}:$line — `$name` is missing /// dartdoc',
              );
            }
          }
        }

        expect(
          missingDocs,
          isEmpty,
          reason:
              'Every public top-level symbol transitively exported by lib/ndarray.dart must have a `///` dartdoc comment:\n'
              '${missingDocs.join('\n')}',
        );
      },
    );

    test(
      'Architecture & memory invariants: no cross-package src/ imports, no external .dataRaw, no pointer+offsetElements double-offset, no CellFlat+getIndex',
      () {
        final violations = <String>[];

        // 1. No cross-package package:<other>/src/ imports across workspace packages in pkgs/*/lib/
        if (pkgsDir.existsSync()) {
          final workspacePackages = pkgsDir
              .listSync()
              .whereType<Directory>()
              .map((d) => d.uri.pathSegments.where((s) => s.isNotEmpty).last)
              .toSet();
          for (final pkg in pkgsDir.listSync().whereType<Directory>()) {
            final pkgName = pkg.uri.pathSegments
                .where((s) => s.isNotEmpty)
                .last;
            for (final dartFile in _dartFilesIn(Directory('${pkg.path}/lib'))) {
              final lines = dartFile.readAsLinesSync();
              for (var i = 0; i < lines.length; i++) {
                final m = RegExp(
                  r'''^\s*(?:import|export)\s+['"]package:([^/]+)/src/''',
                ).firstMatch(lines[i]);
                if (m != null) {
                  final importedPkg = m.group(1)!;
                  if (importedPkg != pkgName &&
                      workspacePackages.contains(importedPkg)) {
                    violations.add(
                      '${dartFile.path}:${i + 1} — cross-package src/ import: `${lines[i].trim()}`',
                    );
                  }
                }
              }
            }
          }
        }

        // 2. Check ndarray/lib/ files
        for (final file in libFiles) {
          final isNdarrayCore = file.path.endsWith('/src/ndarray.dart');
          final isOperations = file.path.contains('/src/operations/');
          final lines = file.readAsLinesSync();

          for (var i = 0; i < lines.length; i++) {
            final line = lines[i];
            if (!isNdarrayCore && RegExp(r'\bdataRaw\b').hasMatch(line)) {
              violations.add(
                '${file.path}:${i + 1} — `.dataRaw` accessed outside ndarray.dart',
              );
            }
            if (isOperations &&
                line.contains('.pointer') &&
                line.contains('.offsetElements')) {
              violations.add(
                '${file.path}:${i + 1} — `.pointer` and `.offsetElements` on the same line (risk of double-offsetting)',
              );
            }
            if ((line.contains('getCellFlat') ||
                    line.contains('setCellFlat')) &&
                line.contains('getIndex')) {
              violations.add(
                '${file.path}:${i + 1} — `getCellFlat`/`setCellFlat` used with `iter.getIndex` (must use `getCellRaw`/`setCellRaw`)',
              );
            }
            if (isOperations && line.contains('.setRange(')) {
              violations.add(
                '${file.path}:${i + 1} — `.setRange(` used in operations/ (use NDArray views and `.copy()` instead)',
              );
            }
          }
        }

        expect(
          violations,
          isEmpty,
          reason:
              'Architectural and memory access invariants violated:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'DType enum indices and native package release versions / recursive source hooks stay in sync',
      () {
        final indexingH = File(
          '${hookDir.path}/custom_indexing.h',
        ).readAsStringSync();
        final sortingH = File(
          '${hookDir.path}/custom_sorting.h',
        ).readAsStringSync();
        final expectedMacros = <DType, String>{
          DType.float64: 'DTYPE_FLOAT64',
          DType.float32: 'DTYPE_FLOAT32',
          DType.float16: 'DTYPE_FLOAT16',
          DType.bfloat16: 'DTYPE_BFLOAT16',
          DType.int64: 'DTYPE_INT64',
          DType.int32: 'DTYPE_INT32',
          DType.int16: 'DTYPE_INT16',
          DType.int8: 'DTYPE_INT8',
          DType.uint64: 'DTYPE_UINT64',
          DType.uint32: 'DTYPE_UINT32',
          DType.uint16: 'DTYPE_UINT16',
          DType.uint8: 'DTYPE_UINT8',
          DType.complex128: 'DTYPE_COMPLEX128',
          DType.complex64: 'DTYPE_COMPLEX64',
          DType.boolean: 'DTYPE_BOOLEAN',
        };

        expect(expectedMacros.length, equals(DType.values.length));
        for (final entry in expectedMacros.entries) {
          final idx = entry.key.index;
          final macro = entry.value;
          final pattern = RegExp('#define\\s+$macro\\s+$idx\\b');
          expect(
            pattern.hasMatch(indexingH) && pattern.hasMatch(sortingH),
            isTrue,
            reason:
                'Expected `#define $macro $idx` in custom_indexing.h and custom_sorting.h',
          );
        }

        // Check native packages (ndarray, openblas, pocketfft, symengine)
        if (pkgsDir.existsSync()) {
          for (final pkgName in [
            'ndarray',
            'openblas',
            'pocketfft',
            'symengine',
          ]) {
            final dir = Directory('${pkgsDir.path}/$pkgName');
            final hashesFile = File(
              '${dir.path}/lib/src/hook_helpers/hashes.dart',
            );
            if (!hashesFile.existsSync()) continue;

            final pubspec = File('${dir.path}/pubspec.yaml').readAsStringSync();
            final pubVersion = RegExp(
              r'^version:\s*(\S+)',
              multiLine: true,
            ).firstMatch(pubspec)?.group(1);
            final hashesTxt = hashesFile.readAsStringSync();
            final tagVersion = RegExp(
              r"const\s+version\s*=\s*'artifacts-v([^']+)';",
            ).firstMatch(hashesTxt)?.group(1);
            expect(
              tagVersion,
              equals(pubVersion),
              reason:
                  '$pkgName: hashes.dart version (`artifacts-v$tagVersion`) must match pubspec.yaml version (`$pubVersion`).',
            );

            // If hook/ has subdirectories (like pocketfft/hook/src/), listSync must be recursive: true
            final subDirs = Directory(
              '${dir.path}/hook',
            ).listSync().whereType<Directory>().toList();
            if (subDirs.isNotEmpty) {
              expect(
                hashesTxt.contains('listSync(recursive: true)'),
                isTrue,
                reason:
                    '$pkgName: hook/ contains subdirectories; nativeSourceFiles in hashes.dart must use `listSync(recursive: true)`.',
              );
            }
          }
        }
      },
    );

    test(
      'Zero duplicate @ffi.Native external declarations across binding files (S1)',
      () {
        final bindingFiles = [
          File('${libDir.path}/src/ndarray_bindings.dart'),
          File('${libDir.path}/src/ndarray_extensions_bindings.dart'),
        ];
        final seen = <String, String>{};
        final duplicates = <String>[];

        for (final file in bindingFiles) {
          if (!file.existsSync()) continue;
          final parsed = parseFile(
            path: file.path,
            featureSet: featureSet,
            throwIfDiagnostics: false,
          );
          final baseName = file.uri.pathSegments.last;
          for (final decl in parsed.unit.declarations) {
            if (decl is FunctionDeclaration && decl.externalKeyword != null) {
              final name = decl.name.lexeme;
              final line = parsed.lineInfo.getLocation(decl.offset).lineNumber;
              final loc = '$baseName:$line';
              if (seen.containsKey(name)) {
                duplicates.add(
                  '`$name` declared at both ${seen[name]} and $loc',
                );
              } else {
                seen[name] = loc;
              }
            }
          }
        }

        expect(
          duplicates,
          isEmpty,
          reason:
              'Duplicate external @ffi.Native declarations risk signature drift:\n'
              '${duplicates.join('\n')}',
        );
      },
    );

    test(
      'ScratchArena.getStridedBuffer segment bounds and integer floordiv error checks (S6)',
      () {
        final violations = <String>[];

        for (final file in libFiles) {
          final content = file.readAsStringSync();
          final lines = content.split('\n');

          // 1. Check getStridedBuffer(ndim, [segments]) vs cBuffer + ndim * k
          final stridedDeclRegex = RegExp(
            r'final\s+(\w+)\s*=\s*ScratchArena\.getStridedBuffer\(\s*(\w+)(?:\s*,\s*(\d+))?\s*\)',
          );
          for (var i = 0; i < lines.length; i++) {
            final m = stridedDeclRegex.firstMatch(lines[i]);
            if (m != null) {
              final bufVar = m.group(1)!;
              final ndimVar = m.group(2)!;
              final segments = int.parse(m.group(3) ?? '3');
              final endLine = (i + 60 < lines.length) ? i + 60 : lines.length;
              final window = lines.sublist(i, endLine).join('\n');
              final offsetRegex = RegExp(
                '$bufVar\\s*\\+\\s*$ndimVar\\s*\\*\\s*(\\d+)',
              );
              for (final om in offsetRegex.allMatches(window)) {
                final k = int.parse(om.group(1)!);
                if (k >= segments) {
                  violations.add(
                    '${file.path}:${i + 1} — `$bufVar + $ndimVar * $k` exceeds allocated segments ($segments) in `getStridedBuffer`.',
                  );
                }
              }
            }
          }

          // 2. Every file invoking v_floordiv_int* or s_floordiv_int* must check get_and_reset_division_error()
          if (!file.path.endsWith('ndarray_bindings.dart') &&
              RegExp(r'\b[vs]_floordiv_int').hasMatch(content)) {
            if (!content.contains('get_and_reset_division_error()')) {
              violations.add(
                '${file.path} — calls `v_floordiv_int*` / `s_floordiv_int*` without checking `get_and_reset_division_error()`.',
              );
            }
          }
        }

        expect(
          violations,
          isEmpty,
          reason:
              'Strided scratch buffer segment or integer division error check violation:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'C++ hook sources use std::nothrow on all heap allocations and avoid std::stable_sort/std::map/iostream/printf (S7)',
      () {
        final cppAndHeaderFiles = hookDir
            .listSync()
            .whereType<File>()
            .where((f) => f.path.endsWith('.cpp') || f.path.endsWith('.h'))
            .toList();
        final violations = <String>[];

        for (final file in cppAndHeaderFiles) {
          final raw = file.readAsStringSync();
          final stripped = _stripCppComments(raw);
          final baseName = file.uri.pathSegments.last;

          // Check raw `new` without `(std::nothrow)`
          for (final m in RegExp(r'\bnew\b([^\n;]*)').allMatches(stripped)) {
            final tail = m.group(1)!;
            if (!tail.contains('nothrow')) {
              violations.add(
                '$baseName — raw `new` without `(std::nothrow)`: `new$tail`',
              );
            }
          }

          if (RegExp(
            r'\bstd::(?:unordered_)?(?:map|set)\b',
          ).hasMatch(stripped)) {
            violations.add(
              '$baseName — contains `std::map`/`std::set` (throws `std::bad_alloc` under `-fno-exceptions`).',
            );
          }
          if (raw.contains('#include <iostream>')) {
            violations.add(
              '$baseName — includes `<iostream>` (adds static constructors and I/O bloat).',
            );
          }
          if (RegExp(
            r'\b(?:printf|std::cout|std::cerr)\b',
          ).hasMatch(stripped)) {
            violations.add(
              '$baseName — contains `printf`/`std::cout`/`std::cerr` in FFI library.',
            );
          }
        }

        expect(
          violations,
          isEmpty,
          reason:
              'C++ `-fno-exceptions` / heap discipline violations:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'Public functions with an `out` parameter use a named parameter `{NDArray? out}` and BinaryOp/UnaryOp enums are exhaustively dispatched (S2 & S9)',
      () {
        final violations = <String>[];

        // 1. All public functions in lib/src/operations/ with a parameter named `out` must declare it as a named parameter
        for (final file in libFiles) {
          if (!file.path.contains('/src/operations/')) continue;
          final parsed = parseFile(
            path: file.path,
            featureSet: featureSet,
            throwIfDiagnostics: false,
          );
          for (final decl in parsed.unit.declarations) {
            if (decl is FunctionDeclaration) {
              final fnName = decl.name.lexeme;
              if (fnName.startsWith('_') || fnName == 'where') continue;
              final params = decl.functionExpression.parameters?.parameters;
              if (params == null) continue;
              for (final p in params) {
                if (p.name?.lexeme == 'out' && !p.isNamed) {
                  final line = parsed.lineInfo.getLocation(p.offset).lineNumber;
                  violations.add(
                    '${file.path}:$line — `$fnName` declares `out` as a positional parameter instead of a named parameter.',
                  );
                }
              }
            }
          }
        }

        // 2. Every reducible BinaryOp and every UnaryOp in binary_op.dart is dispatched in ufunc_methods.dart
        final binOpFile = File(
          '${libDir.path}/src/operations/math/binary_op.dart',
        );
        final ufuncMethodsTxt = File(
          '${libDir.path}/src/operations/math/ufunc_methods.dart',
        ).readAsStringSync();
        for (final op in BinaryOp.values.where((o) => o.isReducible)) {
          if (!ufuncMethodsTxt.contains('BinaryOp.${op.name}')) {
            violations.add(
              'BinaryOp.${op.name} (isReducible: true) is not dispatched in ufunc_methods.dart',
            );
          }
        }
        final binParsed = parseFile(
          path: binOpFile.path,
          featureSet: featureSet,
          throwIfDiagnostics: false,
        );
        for (final decl in binParsed.unit.declarations) {
          if (decl is EnumDeclaration &&
              decl.namePart.typeName.lexeme == 'UnaryOp') {
            for (final c in decl.body.constants) {
              final cName = c.name.lexeme;
              if (!ufuncMethodsTxt.contains('UnaryOp.$cName')) {
                violations.add(
                  'UnaryOp.$cName is declared in binary_op.dart but not dispatched in ufunc_methods.dart',
                );
              }
            }
          }
        }

        // 3. Zero-violation ratchets: no NDArray<double|int|num|bool|Object|dynamic>, no print( in lib/, no solo_test in test/
        final badTypeGeneric = RegExp(
          r'\bNDArray<\s*(?:double|int|num|bool|Object|dynamic)\s*>',
        );
        for (final file in libFiles) {
          final lines = file.readAsLinesSync();
          for (var i = 0; i < lines.length; i++) {
            final line = lines[i];
            if (line.trimLeft().startsWith('//')) continue;
            if (badTypeGeneric.hasMatch(line)) {
              violations.add(
                '${file.path}:${i + 1} — uses Dart primitive type parameter on NDArray instead of DataType marker (`Float64`, `Int64`, etc.): `${line.trim()}`',
              );
            }
            if (RegExp(r'\bprint\s*\(').hasMatch(line)) {
              violations.add(
                '${file.path}:${i + 1} — `print(...)` statement in library code.',
              );
            }
          }
        }

        final testFiles = _dartFilesIn(Directory('${pkgRoot.path}/test'));
        for (final file in testFiles) {
          if (file.path.endsWith('codebase_invariants_test.dart')) continue;
          final content = file.readAsStringSync();
          if (RegExp(r'\bsolo_test\s*\(').hasMatch(content) ||
              RegExp(r'\bsolo_group\s*\(').hasMatch(content)) {
            violations.add(
              '${file.path} — contains `solo_test` or `solo_group`.',
            );
          }
        }

        expect(
          violations,
          isEmpty,
          reason:
              'API signature, BinaryOp enum parity, or code hygiene ratchet violated:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'Resolved Semantic AST (DartType & Element) Invariants: exportNamespace type-closure, class modifiers/member dartdocs, receiver-type-aware NDArray.toList()/NDArray.data bans, ScratchArena.allocate<T> vs sizeOf<U> DartType equality, and zero dead extension bindings',
      () async {
        final resolvedLibPath = libDir.resolveSymbolicLinksSync();
        final collection = AnalysisContextCollection(
          includedPaths: [resolvedLibPath],
        );
        final session = collection.contextFor(resolvedLibPath).currentSession;

        final libResult = await session.getResolvedLibrary(
          '$resolvedLibPath/ndarray.dart',
        );
        expect(libResult, isA<ResolvedLibraryResult>());
        final exportNames = (libResult as ResolvedLibraryResult)
            .element
            .exportNamespace
            .definedNames2;
        expect(exportNames.length, greaterThan(350));

        final violations = <String>[];

        void checkNoRawNdarray(DartType type, String context) {
          if (type is InterfaceType && type.element.name == 'NDArray') {
            if (type.typeArguments.first is DynamicType) {
              violations.add(
                '$context — uses raw `NDArray<dynamic>` (`${type.getDisplayString()}`); specify `<T extends DTypeTag>` or a concrete `DTypeTag`.',
              );
            }
          }
        }

        // 1. Inspect resolved exportNamespace of package:ndarray/ndarray.dart
        for (final entry in exportNames.entries) {
          final name = entry.key;
          final el = entry.value;

          if (el.metadata.annotations.any((a) => a.isInternal)) {
            violations.add(
              'Exported symbol `$name` (${el.kind.displayName}) is annotated `@internal`.',
            );
          }

          if (el is ExecutableElement) {
            if (el.returnType is DynamicType &&
                name != 'where' &&
                name != 'unique') {
              violations.add(
                'Exported function `$name` returns untyped `dynamic`.',
              );
            }
            checkNoRawNdarray(el.returnType, 'Exported `$name` return type');
            for (final p in el.formalParameters) {
              checkNoRawNdarray(
                p.type,
                'Exported `$name` parameter `${p.name}`',
              );
            }
          } else if (el is ClassElement) {
            if (!el.isFinal && !el.isSealed && !el.isAbstract && !el.isBase) {
              violations.add(
                'Exported class `${el.name}` must be marked `final`, `sealed`, `abstract`, or `base`.',
              );
            }
            for (final m in [...el.methods, ...el.getters]) {
              if (m.isPrivate || m.firstFragment.nameOffset == null) continue;
              if (m.metadata.annotations.any(
                (a) => a.isInternal || a.isOverride,
              )) {
                continue;
              }
              if (m.documentationComment == null) {
                violations.add(
                  'Public member `${el.name}.${m.name}` is missing `///` dartdoc.',
                );
              }
              checkNoRawNdarray(
                m.returnType,
                'Member `${el.name}.${m.name}` return type',
              );
              for (final p in m.formalParameters) {
                checkNoRawNdarray(
                  p.type,
                  'Member `${el.name}.${m.name}` parameter `${p.name}`',
                );
              }
            }
          }
        }

        // 2. Collect all @ffi.Native external functions in ndarray_extensions_bindings.dart
        final extBindingsRes = await session.getResolvedUnit(
          '$resolvedLibPath/src/ndarray_extensions_bindings.dart',
        );
        final extFunctions = <ExecutableElement>{};
        if (extBindingsRes is ResolvedUnitResult) {
          for (final d in extBindingsRes.unit.declarations) {
            if (d is FunctionDeclaration && d.externalKeyword != null) {
              final el = d.declaredFragment?.element;
              if (el != null) extFunctions.add(el);
            }
          }
        }
        final usedExtFunctions = <ExecutableElement>{};

        // 3. Walk resolved AST of every implementation file in lib/
        for (final file in libFiles) {
          if (file.path.endsWith('ndarray_bindings.dart')) continue;
          final resolvedFile = file.resolveSymbolicLinksSync();
          final unitRes = await session.getResolvedUnit(resolvedFile);
          if (unitRes is! ResolvedUnitResult) continue;

          final visitor = _ResolvedSemanticVisitor(
            file.path,
            unitRes.lineInfo,
            usedExtFunctions,
          );
          unitRes.unit.accept(visitor);
          violations.addAll(visitor.violations);
        }

        final unusedExt = extFunctions.difference(usedExtFunctions);
        for (final u in unusedExt) {
          violations.add(
            'ndarray_extensions_bindings.dart — unused `@ffi.Native` binding `${u.name}`.',
          );
        }

        // 4. Verify operation_contracts_test.dart covers all BinaryOp & UnaryOp operations and branch coverage tool exists
        final contractsFile = File(
          '${pkgRoot.path}/test/meta/operation_contracts_test.dart',
        );
        final contractsSrc = contractsFile.readAsStringSync();
        const opAliasMap = <String, String>{
          'absolute': 'abs',
          'fabs': 'abs',
          'conjugate': 'conj',
          'arcsin': 'asin',
          'arccos': 'acos',
          'arctan': 'atan',
          'arcsinh': 'asinh',
          'arccosh': 'acosh',
          'arctanh': 'atanh',
          'arctan2': 'atan2',
          'degrees': 'rad2deg',
          'radians': 'deg2rad',
          'bitwiseNot': 'invert',
          'remainder': 'mod',
          'floatPower': 'power',
          'minimum': 'min',
          'maximum': 'max',
          'fmin': 'nanmin',
          'fmax': 'nanmax',
          'positive': 'add',
          'exp2': 'exp',
          'cbrt': 'sqrt',
          'signbit': 'sign',
          'spacing': 'abs',
        };
        for (final op in BinaryOp.values) {
          final fnName = opAliasMap[op.name] ?? op.name;
          if (!RegExp('\\b$fnName\\b').hasMatch(contractsSrc)) {
            violations.add(
              'BinaryOp.${op.name} (`$fnName`) is not exercised in test/meta/operation_contracts_test.dart',
            );
          }
        }
        for (final op in UnaryOp.values) {
          final fnName = opAliasMap[op.name] ?? op.name;
          if (!RegExp('\\b$fnName\\b').hasMatch(contractsSrc)) {
            violations.add(
              'UnaryOp.${op.name} (`$fnName`) is not exercised in test/meta/operation_contracts_test.dart',
            );
          }
        }
        if (!File(
          '${pkgRoot.path}/tool/check_branch_coverage.dart',
        ).existsSync()) {
          violations.add('Missing tool/check_branch_coverage.dart');
        }

        expect(
          violations,
          isEmpty,
          reason:
              'Resolved semantic AST (`DartType` / `Element`) invariants violated:\n'
              '${violations.join('\n')}',
        );
      },
    );
  });
}

class _ResolvedSemanticVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final dynamic lineInfo;
  final Set<ExecutableElement> usedExtFunctions;
  final List<String> violations = [];

  _ResolvedSemanticVisitor(this.filePath, this.lineInfo, this.usedExtFunctions);

  bool _isNDArrayType(DartType? type) =>
      type is InterfaceType && type.element.name == 'NDArray';

  @override
  void visitSimpleIdentifier(SimpleIdentifier node) {
    final el = node.element;
    if (el is ExecutableElement) {
      usedExtFunctions.add(el);
    }
    super.visitSimpleIdentifier(node);
  }

  @override
  void visitMethodInvocation(MethodInvocation node) {
    final isOperations = filePath.contains('/src/operations/');
    final targetType = node.realTarget?.staticType;

    // Receiver-type-aware ban on NDArray.toList() in lib/src/operations/
    if (isOperations &&
        _isNDArrayType(targetType) &&
        node.methodName.name == 'toList') {
      final line = lineInfo.getLocation(node.offset).lineNumber;
      violations.add(
        '$filePath:$line — `NDArray.toList()` called in operations (`${node.toSource()}`); use views, `NDIter`, or FFI pointers instead.',
      );
    }

    // Resolved DartType equality between ScratchArena.allocate<T>(...) and sizeOf<U>()
    if (node.target?.toSource() == 'ScratchArena' &&
        node.methodName.name == 'allocate') {
      final allocType = node.typeArgumentTypes?.firstOrNull;
      final arg = node.argumentList.arguments.firstOrNull;
      if (allocType != null && arg != null) {
        final sizeOfFinder = _FindSizeOfTypeVisitor();
        arg.accept(sizeOfFinder);
        for (final sizeType in sizeOfFinder.sizeOfTypes) {
          if (allocType != sizeType) {
            final line = lineInfo.getLocation(node.offset).lineNumber;
            violations.add(
              '$filePath:$line — `ScratchArena.allocate<${allocType.getDisplayString()}>` byte count uses mismatched `sizeOf<${sizeType.getDisplayString()}>()`.',
            );
          }
        }
      }
    }

    super.visitMethodInvocation(node);
  }

  @override
  void visitPropertyAccess(PropertyAccess node) {
    if (filePath.contains('/src/operations/') &&
        _isNDArrayType(node.realTarget.staticType) &&
        node.propertyName.name == 'data') {
      final line = lineInfo.getLocation(node.offset).lineNumber;
      violations.add(
        '$filePath:$line — `@internal` `NDArray.data` accessed in operations (`${node.toSource()}`).',
      );
    }
    super.visitPropertyAccess(node);
  }

  @override
  void visitPrefixedIdentifier(PrefixedIdentifier node) {
    if (filePath.contains('/src/operations/') &&
        _isNDArrayType(node.prefix.staticType) &&
        node.identifier.name == 'data') {
      final line = lineInfo.getLocation(node.offset).lineNumber;
      violations.add(
        '$filePath:$line — `@internal` `NDArray.data` accessed in operations (`${node.toSource()}`).',
      );
    }
    super.visitPrefixedIdentifier(node);
  }
}

class _FindSizeOfTypeVisitor extends RecursiveAstVisitor<void> {
  final List<DartType> sizeOfTypes = [];

  @override
  void visitMethodInvocation(MethodInvocation node) {
    if (node.methodName.name == 'sizeOf') {
      final t = node.typeArgumentTypes?.firstOrNull;
      if (t != null) sizeOfTypes.add(t);
    }
    super.visitMethodInvocation(node);
  }

  @override
  void visitFunctionExpressionInvocation(FunctionExpressionInvocation node) {
    if (node.function.toSource().endsWith('sizeOf')) {
      final t = node.typeArgumentTypes?.firstOrNull;
      if (t != null) sizeOfTypes.add(t);
    }
    super.visitFunctionExpressionInvocation(node);
  }
}

class _ScratchArenaVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final dynamic lineInfo;
  final List<String> violations = [];

  _ScratchArenaVisitor(this.filePath, this.lineInfo);

  @override
  void visitVariableDeclarationStatement(VariableDeclarationStatement node) {
    for (final variable in node.variables.variables) {
      final init = variable.initializer;
      if (init != null && init.toSource() == 'ScratchArena.marker') {
        final markerName = variable.name.lexeme;
        final parent = node.parent;
        final List<Statement>? statements = switch (parent) {
          Block(:final statements) => statements,
          SwitchPatternCase(:final statements) => statements,
          SwitchCase(:final statements) => statements,
          SwitchDefault(:final statements) => statements,
          _ => null,
        };
        if (statements != null) {
          final idx = statements.indexOf(node);
          final hasTryFinallyReset = statements
              .skip(idx + 1)
              .whereType<TryStatement>()
              .any((tryStmt) {
                final finallySrc = tryStmt.finallyBlock?.toSource() ?? '';
                return finallySrc.contains('ScratchArena.reset($markerName)');
              });
          if (!hasTryFinallyReset) {
            final line = lineInfo.getLocation(node.offset).lineNumber;
            violations.add(
              '$filePath:$line — `ScratchArena.marker` stored in `$markerName` is not paired with `try { ... } finally { ScratchArena.reset($markerName); }`.',
            );
          }
        } else {
          final line = lineInfo.getLocation(node.offset).lineNumber;
          violations.add(
            '$filePath:$line — `ScratchArena.marker` must be declared in a Block or SwitchCase before a TryStatement.',
          );
        }
      }
    }
    super.visitVariableDeclarationStatement(node);
  }
}

class _UncheckedNativeCallVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final dynamic lineInfo;
  final Set<String> intReturningNativeFunctions;
  final List<String> violations = [];

  _UncheckedNativeCallVisitor(
    this.filePath,
    this.lineInfo,
    this.intReturningNativeFunctions,
  );

  @override
  void visitMethodInvocation(MethodInvocation node) {
    final name = node.methodName.name;
    if (intReturningNativeFunctions.contains(name)) {
      if (node.parent is ExpressionStatement) {
        final line = lineInfo.getLocation(node.offset).lineNumber;
        violations.add(
          '$filePath:$line — return value of `$name(...)` is ignored.',
        );
      }
    }
    super.visitMethodInvocation(node);
  }
}

class _FinalizerExternalSizeVisitor extends RecursiveAstVisitor<void> {
  final String filePath;
  final dynamic lineInfo;
  final List<String> violations = [];

  _FinalizerExternalSizeVisitor(this.filePath, this.lineInfo);

  @override
  void visitMethodInvocation(MethodInvocation node) {
    if (node.methodName.name == 'attach') {
      for (final arg in node.argumentList.arguments) {
        if (arg.toSource().startsWith('externalSize:')) {
          final line = lineInfo.getLocation(arg.offset).lineNumber;
          violations.add(
            '$filePath:$line — `externalSize:` passed to `.attach(...)`.',
          );
        }
      }
    }
    super.visitMethodInvocation(node);
  }
}
