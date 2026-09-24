import 'dart:io';

import 'package:analyzer/dart/analysis/features.dart';
import 'package:analyzer/dart/analysis/utilities.dart';
import 'package:analyzer/dart/ast/ast.dart';
import 'package:analyzer/dart/ast/visitor.dart';
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
      'Every int-returning native_* FFI call in lib/ checks its return code',
      () {
        final bindingsFile = File('${libDir.path}/src/ndarray_bindings.dart');
        expect(bindingsFile.existsSync(), isTrue);

        final bindingsUnit = parseFile(
          path: bindingsFile.path,
          featureSet: featureSet,
          throwIfDiagnostics: false,
        ).unit;

        final intReturningNativeFunctions = <String>{};
        for (final decl in bindingsUnit.declarations) {
          if (decl is FunctionDeclaration) {
            final name = decl.name.lexeme;
            final returnType = decl.returnType?.toSource();
            if (name.startsWith('native_') && returnType == 'int') {
              intReturningNativeFunctions.add(name);
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
          if (file.path.endsWith('ndarray_bindings.dart')) continue;
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
      'NativeFinalizer.attach never passes externalSize (VM GC thrashing guard)',
      () {
        final violations = <String>[];
        for (final file in libFiles) {
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
              'Do not pass externalSize to NativeFinalizer.attach (causes GC '
              'thrashing on large buffers); rely on NDArray.scope / dispose() instead:\n'
              '${violations.join('\n')}',
        );
      },
    );

    test(
      'C++ hook sources are exception-free (-fno-exceptions safe) and match headers',
      () {
        final cppFiles = hookDir
            .listSync()
            .whereType<File>()
            .where((f) => f.path.endsWith('.cpp'))
            .toList();
        expect(cppFiles, isNotEmpty);

        final violations = <String>[];

        for (final cppFile in cppFiles) {
          final raw = cppFile.readAsStringSync();
          final stripped = _stripCppComments(raw);
          final baseName = cppFile.uri.pathSegments.last;

          // Indexing & NPZ I/O return integer status codes and must use NoThrowBuffer.
          if ((baseName == 'custom_indexing.cpp' || baseName == 'npz_io.cpp') &&
              RegExp(r'\bstd::vector\b').hasMatch(stripped)) {
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

          // Check that every FFI_PLUGIN_EXPORT function in foo.cpp is declared in foo.h
          final headerPath = cppFile.path.replaceFirst(RegExp(r'\.cpp$'), '.h');
          final headerFile = File(headerPath);
          if (headerFile.existsSync()) {
            final headerSource = _stripCppComments(
              headerFile.readAsStringSync(),
            );
            final exportRegex = RegExp(
              r'FFI_PLUGIN_EXPORT\s+[A-Za-z0-9_*\s]+\s+([A-Za-z0-9_]+)\s*\(',
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
      'All public top-level API declarations exported by lib/ndarray.dart have Dartdoc comments',
      () {
        final entrypoint = File('${libDir.path}/ndarray.dart');
        final entryParsed = parseFile(
          path: entrypoint.path,
          featureSet: featureSet,
          throwIfDiagnostics: false,
        );

        final exportedFiles = <File>[entrypoint];
        for (final directive in entryParsed.unit.directives) {
          if (directive is ExportDirective) {
            final uriStr = directive.uri.stringValue;
            if (uriStr != null && !uriStr.startsWith('package:')) {
              final resolved = File('${libDir.path}/$uriStr');
              if (resolved.existsSync()) {
                exportedFiles.add(resolved);
              }
            }
          }
        }

        final missingDocs = <String>[];

        for (final file in exportedFiles) {
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
            // Skip internal C-style reduction helpers (e.g., r_quantile_helper).
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
              'Every public top-level symbol exported by lib/ndarray.dart must have a `///` dartdoc comment:\n'
              '${missingDocs.join('\n')}',
        );
      },
    );
  });
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
