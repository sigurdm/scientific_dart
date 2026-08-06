import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:path/path.dart' as p;
import 'package:vm_service/vm_service.dart';
import 'package:vm_service/vm_service_io.dart';
import 'package:analyzer/dart/analysis/utilities.dart';
import 'package:analyzer/dart/ast/ast.dart';
import 'package:notebook/src/lsp_client.dart';
import 'package:notebook/src/kernel_helper.dart';
import 'package:dart_style/dart_style.dart';

class DeclaredSymbolResult {
  final String symbol;
  final bool isVariable;
  DeclaredSymbolResult(this.symbol, this.isVariable);
}

class CompletionItem {
  final String label;
  final String type;
  final String? detail;

  CompletionItem({required this.label, required this.type, this.detail});

  Map<String, dynamic> toJson() => {
    'label': label,
    'type': type,
    if (detail != null) 'detail': detail,
  };
}

class NotebookKernel {
  final String workspaceDir;
  final String dartSdkPath;

  Process? _process;
  VmService? _service;
  String? _isolateId;
  String? _rootLibId;
  String? _workspaceLibId;

  LspClient? _lspClient;
  int _workspaceVersion = 1;

  final Set<String> _imports = {
    "import 'package:ndarray/ndarray.dart';",
    "import 'dart:math' as math;",
  };
  final Map<String, String> _definitions = {};

  NotebookKernel({required this.workspaceDir, required this.dartSdkPath}) {
    _writeWorkspace();
  }

  Future<void> start() async {
    await _startKernelOnly();

    _lspClient = LspClient(dartSdkPath: dartSdkPath, rootPath: workspaceDir);
    try {
      await _lspClient!.start();
      final workspaceFile = File(
        p.join(workspaceDir, 'lib', 'src', 'workspace.dart'),
      );
      final fileUri = p.toUri(workspaceFile.path).toString();
      _lspClient!.didOpen(
        fileUri,
        workspaceFile.existsSync() ? workspaceFile.readAsStringSync() : '',
      );
    } catch (e) {
      print('Warning: Failed to start LSP client: $e');
    }
  }

  Future<void> _startKernelOnly() async {
    final dartExecutable = p.join(dartSdkPath, 'bin', 'dart');

    _process = await Process.start(dartExecutable, [
      '--enable-vm-service=0',
      'bin/kernel.dart',
    ], workingDirectory: workspaceDir);

    final uriCompleter = Completer<String>();

    _process!.stdout
        .transform(utf8.decoder)
        .transform(const LineSplitter())
        .listen((line) {
          final match = RegExp(
            r'The Dart VM service is listening on (http://\S+/)',
          ).firstMatch(line);
          if (match != null && !uriCompleter.isCompleted) {
            uriCompleter.complete(match.group(1)!);
          }
        });

    _process!.stderr
        .transform(utf8.decoder)
        .transform(const LineSplitter())
        .listen((line) {
          final match = RegExp(
            r'The Dart VM service is listening on (http://\S+/)',
          ).firstMatch(line);
          if (match != null && !uriCompleter.isCompleted) {
            uriCompleter.complete(match.group(1)!);
          }
        });

    final vmServiceUri = await uriCompleter.future.timeout(
      const Duration(seconds: 10),
      onTimeout: () => throw TimeoutException(
        'Failed to find VM Service URI from kernel process',
      ),
    );

    final wsUri = vmServiceUri.replaceFirst('http://', 'ws://') + 'ws';
    _service = await vmServiceConnectUri(wsUri);

    var vm = await _service!.getVM();
    IsolateRef? mainIsolateRef;
    final timeout = DateTime.now().add(const Duration(seconds: 10));

    while (mainIsolateRef == null) {
      if (DateTime.now().isAfter(timeout)) {
        throw TimeoutException(
          'Timed out waiting for main isolate in kernel process.',
        );
      }
      vm = await _service!.getVM();
      for (var isolateRef in vm.isolates ?? []) {
        if (isolateRef.name == 'main' || vm.isolates!.length == 1) {
          mainIsolateRef = isolateRef;
          break;
        }
      }
      if (mainIsolateRef == null) {
        await Future.delayed(const Duration(milliseconds: 100));
      }
    }

    _isolateId = mainIsolateRef.id!;

    final isolateTimeout = DateTime.now().add(const Duration(seconds: 10));
    var isolate = await _service!.getIsolate(_isolateId!);
    while (isolate.runnable != true || isolate.rootLib == null) {
      if (DateTime.now().isAfter(isolateTimeout)) {
        throw TimeoutException(
          'Timed out waiting for runnable isolate rootLib.',
        );
      }
      await Future.delayed(const Duration(milliseconds: 100));
      isolate = await _service!.getIsolate(_isolateId!);
    }

    _rootLibId = isolate.rootLib!.id!;
    _updateWorkspaceLibId(isolate);
  }

  Future<void> _restartKernelProcess() async {
    await _service?.dispose();
    _process?.kill();
    _process = null;
    _service = null;

    await _startKernelOnly();
    await _reloadWorkspace();
  }

  Future<String> _handleAddDependency(String pkgName) async {
    final dartExecutable = p.join(dartSdkPath, 'bin', 'dart');
    final result = await Process.run(dartExecutable, [
      'pub',
      'add',
      pkgName,
    ], workingDirectory: workspaceDir);

    if (result.exitCode != 0) {
      return 'Failed to add dependency "$pkgName":\n${result.stderr}';
    }

    try {
      await _restartKernelProcess();
      return 'Successfully added package "$pkgName" and reloaded environment.';
    } catch (e) {
      return 'Added package "$pkgName", but failed to reload kernel: $e';
    }
  }

  Future<void> _ensurePackageInstalled(String pkgName) async {
    final pubspecFile = File(p.join(workspaceDir, 'pubspec.yaml'));
    if (pubspecFile.existsSync()) {
      final content = pubspecFile.readAsStringSync();
      if (content.contains('$pkgName:')) return;
    }
    await _handleAddDependency(pkgName);
  }

  void _updateWorkspaceLibId(Isolate isolate) {
    for (var lib in isolate.libraries ?? []) {
      if (lib.uri != null && lib.uri!.endsWith('workspace.dart')) {
        _workspaceLibId = lib.id!;
        return;
      }
    }
    throw StateError('Could not find workspace.dart library in isolate');
  }

  Future<void> stop() async {
    await _lspClient?.stop();
    await _service?.dispose();
    _process?.kill();
  }

  /// Formats Dart [code] using dartfmt ([DartFormatter]).
  String formatCode(String code) {
    final trimmed = code.trim();
    if (trimmed.isEmpty) return trimmed;
    try {
      final formatter = DartFormatter(
        languageVersion: DartFormatter.latestLanguageVersion,
      );
      return formatter.format(trimmed).trim();
    } catch (_) {
      try {
        final formatter = DartFormatter(
          languageVersion: DartFormatter.latestLanguageVersion,
        );
        return formatter.formatStatement(trimmed).trim();
      } catch (_) {
        return trimmed;
      }
    }
  }

  Future<String> execute(String code) async {
    code = formatCode(code);
    if (code.isEmpty) return '';

    final pubAddMatch = RegExp(
      r'^(?:%|%)?(?:pub\s+add|add)\s+([\w\d_\-]+)',
    ).firstMatch(code);
    if (pubAddMatch != null) {
      final pkgName = pubAddMatch.group(1)!;
      return await _handleAddDependency(pkgName);
    }

    if (RegExp(r'^import\s+').hasMatch(code)) {
      final pkgMatch = RegExp(
        r"^import\s+[']package:([\w\d_\-]+)/",
      ).firstMatch(code);
      if (pkgMatch != null) {
        final pkgName = pkgMatch.group(1)!;
        if (pkgName != 'ndarray' && pkgName != 'notebook') {
          await _ensurePackageInstalled(pkgName);
        }
      }
      _imports.add(code);
      await _reloadWorkspace();
      return 'Imported: $code';
    }

    final declResult = _getDeclaredSymbolWithAnalyzer(code);
    if (declResult != null) {
      final symbol = declResult.symbol;
      var fullDecl = code.trim();
      if (declResult.isVariable) {
        if (!fullDecl.endsWith(';')) fullDecl += ';';
        _definitions[symbol] = fullDecl;
        await _reloadWorkspace();

        try {
          final evalResult = await _service!.evaluate(
            _isolateId!,
            _workspaceLibId!,
            symbol,
          );
          final valStr = await _formatResult(evalResult);

          String printed = '';
          try {
            final capObj = await _service!.evaluate(
              _isolateId!,
              _workspaceLibId!,
              'getCapturedOutput()',
            );
            if (capObj is InstanceRef && capObj.valueAsString != null) {
              printed = capObj.valueAsString!.trim();
            }
          } catch (_) {}

          final header = 'Declared variable $symbol\nValue: $valStr';
          return printed.isNotEmpty ? '$printed\n$header' : header;
        } catch (_) {
          return 'Declared variable $symbol';
        }
      } else {
        if (!fullDecl.endsWith(';') && !fullDecl.endsWith('}')) fullDecl += ';';
        _definitions[symbol] = fullDecl;
        await _reloadWorkspace();
        return 'Declared: $symbol';
      }
    }

    final prevDefs = Map<String, String>.from(_definitions);
    final transformRes = _transformCellCode(code);
    for (var def in transformRes.topLevelDefinitions) {
      final name = def.split(' ')[1].replaceAll(';', '');
      if (!_definitions.containsKey(name)) {
        _definitions[name] = def;
      }
    }
    try {
      await _reloadWorkspace();
    } catch (e) {
      _definitions.clear();
      _definitions.addAll(prevDefs);
      _writeWorkspace();
      try {
        await _service!.reloadSources(_isolateId!);
      } catch (_) {}
      rethrow;
    }

    try {
      try {
        await _service!.evaluate(
          _isolateId!,
          _workspaceLibId!,
          'clearCapturedOutput()',
        );
      } catch (_) {}

      final evalExpr =
          'evalInNotebookZone(() {\n${transformRes.cellBodyCode}\n})';
      final result = await _service!.evaluate(
        _isolateId!,
        _workspaceLibId!,
        evalExpr,
      );

      final outputs = <CellOutputItem>[];

      try {
        final capObj = await _service!.evaluate(
          _isolateId!,
          _workspaceLibId!,
          'getCapturedOutputsJson()',
        );
        String? jsonStr;
        if (capObj is InstanceRef) {
          if (capObj.kind == InstanceKind.kString && capObj.id != null) {
            final fullObj = await _service!.getObject(_isolateId!, capObj.id!);
            if (fullObj is Instance && fullObj.valueAsString != null) {
              jsonStr = fullObj.valueAsString;
            }
          }
          jsonStr ??= capObj.valueAsString;
        }
        if (jsonStr != null) {
          final decoded = jsonDecode(jsonStr) as List;
          for (var item in decoded) {
            outputs.add(
              CellOutputItem.fromJson(Map<String, dynamic>.from(item)),
            );
          }
        }
      } catch (e) {
        print('Error decoding captured outputs: $e');
      }

      final resultItem = await _formatResultToOutputItem(result);
      if (resultItem != null) {
        outputs.add(resultItem);
      }

      return jsonEncode(outputs.map((e) => e.toJson()).toList());
    } catch (e) {
      rethrow;
    }
  }

  Future<List<CompletionItem>> getCompletions(
    String code,
    int cursorOffset,
  ) async {
    if (_lspClient != null) {
      try {
        final workspaceFile = File(
          p.join(workspaceDir, 'lib', 'src', 'workspace.dart'),
        );
        final fileUri = p.toUri(workspaceFile.path).toString();
        final baseContent = workspaceFile.existsSync()
            ? workspaceFile.readAsStringSync()
            : '';

        final prefix = '$baseContent\n\nvoid __eval_dummy__() {\n';
        final fullContent = '$prefix$code\n}';

        _lspClient!.didChange(fileUri, fullContent, ++_workspaceVersion);

        final targetOffset =
            prefix.length +
            (cursorOffset <= code.length ? cursorOffset : code.length);
        final lines = fullContent.substring(0, targetOffset).split('\n');
        final line = lines.length - 1;
        final character = lines.last.length;

        final lspResults = await _lspClient!.getCompletions(
          fileUri,
          line,
          character,
        );

        _lspClient!.didChange(fileUri, baseContent, ++_workspaceVersion);

        if (lspResults.isNotEmpty) {
          return lspResults
              .map(
                (r) => CompletionItem(
                  label: r.label,
                  type: r.kind,
                  detail: r.detail,
                ),
              )
              .toList();
        }
      } catch (_) {
        // Fall back
      }
    }

    return _getLocalCompletions(code, cursorOffset);
  }

  Future<String?> getHover(String code, int cursorOffset) async {
    if (_lspClient == null) return null;
    try {
      final workspaceFile = File(
        p.join(workspaceDir, 'lib', 'src', 'workspace.dart'),
      );
      final fileUri = p.toUri(workspaceFile.path).toString();
      final baseContent = workspaceFile.existsSync()
          ? workspaceFile.readAsStringSync()
          : '';

      final prefix = '$baseContent\n\nvoid __eval_dummy__() {\n';
      final fullContent = '$prefix$code\n}';

      _lspClient!.didChange(fileUri, fullContent, ++_workspaceVersion);

      final targetOffset =
          prefix.length +
          (cursorOffset <= code.length ? cursorOffset : code.length);
      final lines = fullContent.substring(0, targetOffset).split('\n');
      final line = lines.length - 1;
      final character = lines.last.length;

      final hoverResult = await _lspClient!.getHover(fileUri, line, character);

      _lspClient!.didChange(fileUri, baseContent, ++_workspaceVersion);

      return hoverResult;
    } catch (_) {
      return null;
    }
  }

  List<CompletionItem> _getLocalCompletions(String code, int cursorOffset) {
    if (cursorOffset > code.length) cursorOffset = code.length;
    final textBeforeCursor = code.substring(0, cursorOffset);

    // Check if user is typing after a dot e.g. "a." or "NDArray."
    final dotMatch = RegExp(
      r'([\w\d_$]+)\.([\w\d_$]*)$',
    ).firstMatch(textBeforeCursor);
    if (dotMatch != null) {
      final target = dotMatch.group(1)!;
      final prefix = dotMatch.group(2)!;

      final items = <CompletionItem>[];
      if (target == 'NDArray') {
        items.addAll([
          CompletionItem(
            label: 'fromList',
            type: 'constructor',
            detail:
                'factory NDArray.fromList(List list, List<int> shape, DType dtype)',
          ),
          CompletionItem(
            label: 'zeros',
            type: 'constructor',
            detail: 'factory NDArray.zeros(List<int> shape, DType dtype)',
          ),
          CompletionItem(
            label: 'ones',
            type: 'constructor',
            detail: 'factory NDArray.ones(List<int> shape, DType dtype)',
          ),
          CompletionItem(
            label: 'arange',
            type: 'constructor',
            detail:
                'factory NDArray.arange(num start, num stop, [num step, DType dtype])',
          ),
          CompletionItem(
            label: 'linspace',
            type: 'constructor',
            detail:
                'factory NDArray.linspace(num start, num stop, int num, [DType dtype])',
          ),
          CompletionItem(
            label: 'scope',
            type: 'method',
            detail: 'static T scope<T>(T Function() fn)',
          ),
        ]);
      } else if (target == 'DType') {
        items.addAll([
          CompletionItem(
            label: 'float64',
            type: 'property',
            detail: 'DType<Float64>',
          ),
          CompletionItem(
            label: 'float32',
            type: 'property',
            detail: 'DType<Float32>',
          ),
          CompletionItem(
            label: 'int64',
            type: 'property',
            detail: 'DType<Int64>',
          ),
          CompletionItem(
            label: 'int32',
            type: 'property',
            detail: 'DType<Int32>',
          ),
          CompletionItem(
            label: 'int16',
            type: 'property',
            detail: 'DType<Int16>',
          ),
          CompletionItem(
            label: 'uint8',
            type: 'property',
            detail: 'DType<Uint8>',
          ),
          CompletionItem(
            label: 'complex128',
            type: 'property',
            detail: 'DType<Complex128>',
          ),
          CompletionItem(
            label: 'complex64',
            type: 'property',
            detail: 'DType<Complex64>',
          ),
          CompletionItem(
            label: 'boolean',
            type: 'property',
            detail: 'DType<bool>',
          ),
        ]);
      } else {
        items.addAll([
          CompletionItem(
            label: 'shape',
            type: 'property',
            detail: 'List<int> shape',
          ),
          CompletionItem(
            label: 'dtype',
            type: 'property',
            detail: 'DType<T> dtype',
          ),
          CompletionItem(label: 'size', type: 'property', detail: 'int size'),
          CompletionItem(label: 'rank', type: 'property', detail: 'int rank'),
          CompletionItem(
            label: 'isContiguous',
            type: 'property',
            detail: 'bool isContiguous',
          ),
          CompletionItem(
            label: 'isSquare',
            type: 'property',
            detail: 'bool isSquare',
          ),
          CompletionItem(label: 'scalar', type: 'property', detail: 'T scalar'),
          CompletionItem(
            label: 'transposed',
            type: 'property',
            detail: 'NDArray<T> transposed',
          ),
          CompletionItem(
            label: 'copy()',
            type: 'method',
            detail: 'NDArray<T> copy()',
          ),
          CompletionItem(
            label: 'slice()',
            type: 'method',
            detail: 'NDArray<T> slice(List selectors)',
          ),
          CompletionItem(
            label: 'reshape()',
            type: 'method',
            detail: 'NDArray<T> reshape(List<int> newShape)',
          ),
          CompletionItem(
            label: 'transpose()',
            type: 'method',
            detail: 'NDArray<T> transpose([List<int> axes])',
          ),
          CompletionItem(
            label: 'flatten()',
            type: 'method',
            detail: 'NDArray<T> flatten()',
          ),
          CompletionItem(
            label: 'ravel()',
            type: 'method',
            detail: 'NDArray<T> ravel()',
          ),
          CompletionItem(
            label: 'dispose()',
            type: 'method',
            detail: 'void dispose()',
          ),
        ]);
      }
      return items
          .where((i) => i.label.toLowerCase().startsWith(prefix.toLowerCase()))
          .toList();
    }

    final wordMatch = RegExp(r'([\w\d_$]+)$').firstMatch(textBeforeCursor);
    final prefix = wordMatch != null ? wordMatch.group(1)! : '';

    final items = <CompletionItem>[];
    for (var entry in _definitions.entries) {
      final symbol = entry.key;
      final isVar = entry.value.startsWith('var ');
      items.add(
        CompletionItem(
          label: symbol,
          type: isVar ? 'variable' : 'function',
          detail: 'User defined in workspace',
        ),
      );
    }

    items.addAll([
      CompletionItem(
        label: 'sum',
        type: 'function',
        detail: 'NDArray sum(NDArray a, {int? axis, NDArray? out})',
      ),
      CompletionItem(
        label: 'mean',
        type: 'function',
        detail: 'NDArray mean(NDArray a, {int? axis, NDArray? out})',
      ),
      CompletionItem(
        label: 'min',
        type: 'function',
        detail: 'NDArray min(NDArray a, {int? axis, NDArray? out})',
      ),
      CompletionItem(
        label: 'max',
        type: 'function',
        detail: 'NDArray max(NDArray a, {int? axis, NDArray? out})',
      ),
      CompletionItem(
        label: 'std',
        type: 'function',
        detail: 'NDArray std(NDArray a, {int? axis, NDArray? out})',
      ),
      CompletionItem(
        label: 'dot',
        type: 'function',
        detail: 'NDArray dot(NDArray a, NDArray b)',
      ),
      CompletionItem(
        label: 'matmul',
        type: 'function',
        detail: 'NDArray matmul(NDArray a, NDArray b)',
      ),
      CompletionItem(
        label: 'sin',
        type: 'function',
        detail: 'NDArray sin(NDArray a)',
      ),
      CompletionItem(
        label: 'cos',
        type: 'function',
        detail: 'NDArray cos(NDArray a)',
      ),
      CompletionItem(
        label: 'exp',
        type: 'function',
        detail: 'NDArray exp(NDArray a)',
      ),
      CompletionItem(
        label: 'log',
        type: 'function',
        detail: 'NDArray log(NDArray a)',
      ),
      CompletionItem(
        label: 'abs',
        type: 'function',
        detail: 'NDArray abs(NDArray a)',
      ),
      CompletionItem(
        label: 'sqrt',
        type: 'function',
        detail: 'NDArray sqrt(NDArray a)',
      ),
      CompletionItem(
        label: 'NDArray',
        type: 'class',
        detail: 'Multi-dimensional array',
      ),
      CompletionItem(
        label: 'Image',
        type: 'class',
        detail: 'Renderable graphic image wrapper for NDArray',
      ),
      CompletionItem(
        label: 'DType',
        type: 'enum',
        detail: 'Data type specifier',
      ),
      CompletionItem(label: 'var', type: 'keyword'),
      CompletionItem(label: 'final', type: 'keyword'),
      CompletionItem(label: 'const', type: 'keyword'),
      CompletionItem(label: 'int', type: 'type'),
      CompletionItem(label: 'double', type: 'type'),
      CompletionItem(label: 'bool', type: 'type'),
      CompletionItem(label: 'String', type: 'type'),
      CompletionItem(label: 'List', type: 'type'),
    ]);

    return items
        .where((i) => i.label.toLowerCase().startsWith(prefix.toLowerCase()))
        .toList();
  }

  Future<void> _reloadWorkspace() async {
    _writeWorkspace();
    final reloadReport = await _service!.reloadSources(_isolateId!);
    if (reloadReport.success != true) {
      throw StateError('Failed to reload workspace: ${reloadReport.toJson()}');
    }
  }

  void _writeWorkspace() {
    final workspaceFile = File(
      p.join(workspaceDir, 'lib', 'src', 'workspace.dart'),
    );
    final buffer = StringBuffer();
    buffer.writeln('// ignore_for_file: unused_import, unused_element');
    buffer.writeln('// Auto-generated workspace. Do not edit.');
    final defaultImports = {
      "import 'package:notebook/src/kernel_helper.dart';",
      "import 'package:ndarray/ndarray.dart';",
    };
    for (var imp in defaultImports) {
      buffer.writeln(imp);
    }
    for (var imp in _imports) {
      if (!defaultImports.contains(imp.trim())) {
        buffer.writeln(imp);
      }
    }
    buffer.writeln();
    for (var def in _definitions.values) {
      buffer.writeln(def);
      buffer.writeln();
    }
    final content = buffer.toString();
    workspaceFile.writeAsStringSync(content);
    if (_lspClient != null) {
      final fileUri = p.toUri(workspaceFile.path).toString();
      _lspClient!.didChange(fileUri, content, ++_workspaceVersion);
    }
  }

  DeclaredSymbolResult? _getDeclaredSymbolWithAnalyzer(String code) {
    try {
      final trimmed = code.trim();
      var parseResult = parseString(
        content: trimmed,
        throwIfDiagnostics: false,
      );
      if (parseResult.errors.isNotEmpty) {
        parseResult = parseString(
          content: '$trimmed;',
          throwIfDiagnostics: false,
        );
      }
      if (parseResult.errors.isNotEmpty) {
        return null;
      }

      final unit = parseResult.unit;
      if (unit.declarations.isEmpty) {
        return null;
      }

      final decl = unit.declarations.first;

      if (decl is ClassDeclaration) {
        // ignore: undefined_getter
        return DeclaredSymbolResult(
          (decl.namePart as dynamic).typeName.lexeme,
          false,
        );
      } else if (decl is EnumDeclaration) {
        // ignore: undefined_getter
        return DeclaredSymbolResult(
          (decl.namePart as dynamic).typeName.lexeme,
          false,
        );
      } else if (decl is FunctionDeclaration) {
        return DeclaredSymbolResult(decl.name.lexeme, false);
      } else if (decl is MixinDeclaration) {
        return DeclaredSymbolResult(decl.name.lexeme, false);
      } else if (decl is ExtensionDeclaration) {
        if (decl.name != null) {
          return DeclaredSymbolResult(decl.name!.lexeme, false);
        }
      } else if (decl is TopLevelVariableDeclaration) {
        final variables = decl.variables.variables;
        if (variables.isNotEmpty) {
          return DeclaredSymbolResult(variables.first.name.lexeme, true);
        }
      }
    } catch (_) {
      // Ignored
    }
    return null;
  }

  CellTransformationResult _transformCellCode(String code) {
    final trimmed = code.trim();
    var wrapper = 'dynamic __evalCell() {\n$trimmed\n}';
    var parseResult = parseString(content: wrapper, throwIfDiagnostics: false);
    if (parseResult.errors.isNotEmpty) {
      wrapper = 'dynamic __evalCell() {\n$trimmed;\n}';
      parseResult = parseString(content: wrapper, throwIfDiagnostics: false);
    }
    final unit = parseResult.unit;

    if (unit.declarations.isEmpty) {
      return CellTransformationResult(
        [],
        'dynamic __evalCell() {\n$trimmed\n}',
      );
    }

    final firstDecl = unit.declarations.first;
    if (firstDecl is! FunctionDeclaration) {
      return CellTransformationResult(
        [],
        'dynamic __evalCell() {\n$trimmed\n}',
      );
    }

    final body = firstDecl.functionExpression.body;
    if (body is! BlockFunctionBody) {
      return CellTransformationResult(
        [],
        'dynamic __evalCell() {\n$trimmed\n}',
      );
    }

    final statements = body.block.statements;
    final topLevelDefs = <String>[];
    final bodyBuffer = StringBuffer();

    for (var i = 0; i < statements.length; i++) {
      final stmt = statements[i];
      final isLast = (i == statements.length - 1);

      if (stmt is VariableDeclarationStatement) {
        for (var v in stmt.variables.variables) {
          final varName = v.name.lexeme;
          topLevelDefs.add('dynamic $varName;');
          if (v.initializer != null) {
            bodyBuffer.writeln('$varName = ${v.initializer!.toSource()};');
          } else {
            bodyBuffer.writeln('// $varName');
          }
        }
      } else if (isLast && stmt is ExpressionStatement) {
        final expr = stmt.expression;
        final exprStr = expr.toSource();
        final isVoid =
            (expr is MethodInvocation && expr.methodName.name == 'print');
        if (!isVoid && !exprStr.startsWith('return ')) {
          bodyBuffer.writeln('return $exprStr;');
        } else {
          bodyBuffer.writeln(stmt.toSource());
        }
      } else {
        bodyBuffer.writeln(stmt.toSource());
      }
    }

    return CellTransformationResult(topLevelDefs, bodyBuffer.toString());
  }

  Future<CellOutputItem?> _formatResultToOutputItem(dynamic response) async {
    if (response is InstanceRef) {
      if (response.kind == InstanceKind.kNull ||
          response.classRef?.name == 'Null' ||
          response.classRef?.name == 'void') {
        return null;
      }
      try {
        final libId = _workspaceLibId ?? _rootLibId!;
        final formattedRef = await _service!.evaluate(
          _isolateId!,
          libId,
          'prettyFormat(x)',
          scope: {'x': response.id!},
        );
        String? strVal;
        if (formattedRef is InstanceRef) {
          if (formattedRef.kind == InstanceKind.kString &&
              formattedRef.id != null) {
            final fullObj = await _service!.getObject(
              _isolateId!,
              formattedRef.id!,
            );
            if (fullObj is Instance && fullObj.valueAsString != null) {
              strVal = fullObj.valueAsString!;
            }
          }
          strVal ??= formattedRef.valueAsString;
        }
        if (strVal != null) {
          final trimmed = strVal.trim();
          if (trimmed.startsWith('<') &&
              (trimmed.endsWith('>') ||
                  trimmed.contains('/>') ||
                  trimmed.contains('</'))) {
            return CellOutputItem('text/html', strVal);
          } else {
            return CellOutputItem('text/plain', strVal);
          }
        }
      } catch (_) {}
    }
    final fallbackStr = await _formatResult(response);
    if (fallbackStr == 'null' || fallbackStr.isEmpty) return null;
    final trimmed = fallbackStr.trim();
    if (trimmed.startsWith('<') &&
        (trimmed.endsWith('>') ||
            trimmed.contains('/>') ||
            trimmed.contains('</'))) {
      return CellOutputItem('text/html', fallbackStr);
    }
    return CellOutputItem('text/plain', fallbackStr);
  }

  Future<String> _formatResult(Response response) async {
    if (response is InstanceRef) {
      if (response.kind == InstanceKind.kString && response.id != null) {
        try {
          final fullObj = await _service!.getObject(_isolateId!, response.id!);
          if (fullObj is Instance && fullObj.valueAsString != null) {
            return fullObj.valueAsString!;
          }
        } catch (_) {}
      }
      if (response.valueAsString != null &&
          !response.valueAsString!.startsWith('Instance of ')) {
        return response.valueAsString!;
      }

      // Attempt to run prettyFormat in the isolate
      try {
        final libId = _workspaceLibId ?? _rootLibId!;
        final formattedRef = await _service!.evaluate(
          _isolateId!,
          libId,
          'prettyFormat(x)',
          scope: {'x': response.id!},
        );
        if (formattedRef is InstanceRef) {
          if (formattedRef.kind == InstanceKind.kString &&
              formattedRef.id != null) {
            final fullObj = await _service!.getObject(
              _isolateId!,
              formattedRef.id!,
            );
            if (fullObj is Instance && fullObj.valueAsString != null) {
              return fullObj.valueAsString!;
            }
          }
          if (formattedRef.valueAsString != null) {
            return formattedRef.valueAsString!;
          }
        }
      } catch (_) {}

      // Fallback: evaluate "$x" directly in the isolate
      try {
        final strRef = await _service!.evaluate(
          _isolateId!,
          _workspaceLibId ?? _rootLibId!,
          '"\$x"',
          scope: {'x': response.id!},
        );
        if (strRef is InstanceRef) {
          if (strRef.valueAsString != null) {
            return strRef.valueAsString!;
          }
          if (strRef.id != null) {
            final fullObj = await _service!.getObject(_isolateId!, strRef.id!);
            if (fullObj is Instance && fullObj.valueAsString != null) {
              return fullObj.valueAsString!;
            }
          }
        }
      } catch (_) {}

      return 'Instance of ${response.classRef?.name} (id: ${response.id})';
    } else if (response is ErrorRef) {
      return 'Error: ${response.message}';
    } else {
      return response.toString();
    }
  }
}

/// Result of transforming multi-statement notebook cell code into executable workspace structures.
final class CellTransformationResult {
  /// Top-level variable definitions to register in the workspace.
  final List<String> topLevelDefinitions;

  /// The transformed body code for cell execution.
  final String cellBodyCode;

  CellTransformationResult(this.topLevelDefinitions, this.cellBodyCode);
}
