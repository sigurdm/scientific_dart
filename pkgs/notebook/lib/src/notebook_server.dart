import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:path/path.dart' as p;
import 'package:notebook/src/notebook_kernel.dart';
import 'package:markdown/markdown.dart' as md;

class NotebookServer {
  final String workspaceDir;
  final String dartSdkPath;
  final int port;

  HttpServer? _server;
  NotebookKernel? _kernel;

  NotebookServer({
    required this.workspaceDir,
    required this.dartSdkPath,
    this.port = 8080,
  });

  List<Map<String, dynamic>> _sessionCells = [];

  File get _sessionFile => File(p.join(workspaceDir, 'notebook_session.json'));

  void _loadSessionData() {
    try {
      if (_sessionFile.existsSync()) {
        final content = _sessionFile.readAsStringSync();
        if (content.trim().isNotEmpty) {
          final decoded = jsonDecode(content);
          if (decoded is List) {
            _sessionCells = List<Map<String, dynamic>>.from(
              decoded.map((e) => Map<String, dynamic>.from(e as Map)),
            );
          }
        }
      }
    } catch (e) {
      print('Warning: Failed to load session file: $e');
    }
  }

  void _saveSessionData([List<dynamic>? cells]) {
    try {
      if (cells != null) {
        _sessionCells = List<Map<String, dynamic>>.from(
          cells.map((e) => Map<String, dynamic>.from(e as Map)),
        );
      }
      _sessionFile.writeAsStringSync(jsonEncode(_sessionCells));
    } catch (e) {
      print('Warning: Failed to save session file: $e');
    }
  }

  void _updateOrAddCell({
    required String id,
    String? code,
    String? output,
    bool? isError,
    bool? evaluated,
  }) {
    final index = _sessionCells.indexWhere((c) => c['id'] == id);
    if (index != -1) {
      if (code != null) _sessionCells[index]['code'] = code;
      if (output != null) _sessionCells[index]['output'] = output;
      if (isError != null) _sessionCells[index]['isError'] = isError;
      if (evaluated != null) _sessionCells[index]['evaluated'] = evaluated;
    } else {
      _sessionCells.add({
        'id': id,
        'code': code ?? '',
        'output': output ?? '',
        'isError': isError ?? false,
        'evaluated': evaluated ?? false,
      });
    }
    _saveSessionData();
  }

  Future<void> start() async {
    _loadSessionData();

    _kernel = NotebookKernel(
      workspaceDir: workspaceDir,
      dartSdkPath: dartSdkPath,
    );

    print('Starting notebook kernel...');
    await _kernel!.start();
    print('Kernel started successfully.');

    _server = await HttpServer.bind(InternetAddress.anyIPv4, port);
    print('Server listening on http://localhost:$port');

    _server!.listen((HttpRequest request) async {
      if (request.uri.path == '/ws') {
        if (WebSocketTransformer.isUpgradeRequest(request)) {
          final socket = await WebSocketTransformer.upgrade(request);
          _handleWebSocket(socket);
        } else {
          request.response.statusCode = HttpStatus.badRequest;
          request.response.close();
        }
      } else {
        var requestedPath = request.uri.path;
        if (requestedPath == '/') requestedPath = '/index.html';
        final relativePath = requestedPath.startsWith('/')
            ? requestedPath.substring(1)
            : requestedPath;
        final targetFile = File(p.join(workspaceDir, 'web', relativePath));

        if (await targetFile.exists()) {
          final ext = p.extension(targetFile.path).toLowerCase();
          switch (ext) {
            case '.wasm':
              request.response.headers.set('content-type', 'application/wasm');
              break;
            case '.mjs':
            case '.js':
              request.response.headers.set('content-type', 'text/javascript');
              break;
            case '.html':
              request.response.headers.contentType = ContentType.html;
              break;
            case '.css':
              request.response.headers.set('content-type', 'text/css');
              break;
            default:
              request.response.headers.contentType = ContentType.binary;
          }
          await request.response.addStream(targetFile.openRead());
        } else {
          request.response.statusCode = HttpStatus.notFound;
          request.response.write('404 Not Found');
        }
        await request.response.close();
      }
    });
  }

  void _handleWebSocket(WebSocket socket) {
    print('Client connected to WebSocket.');
    socket.add(jsonEncode({'type': 'init_session', 'cells': _sessionCells}));

    socket.listen(
      (data) async {
        try {
          final msg = jsonDecode(data as String) as Map<String, dynamic>;
          final msgType = msg['type'];

          if (msgType == 'save_session') {
            if (msg['cells'] is List) {
              _saveSessionData(msg['cells'] as List);
            }
          } else if (msgType == 'update_cell') {
            final cellId = msg['id'] as String;
            final code = msg['code'] as String?;
            _updateOrAddCell(id: cellId, code: code, evaluated: false);
          } else if (msgType == 'delete_cell') {
            final cellId = msg['id'] as String;
            _sessionCells.removeWhere((c) => c['id'] == cellId);
            _saveSessionData();
          } else if (msgType == 'execute') {
            final reqId = msg['id'] as String;
            final code = msg['code'] as String;
            final cellId = msg['cellId'] as String? ?? reqId.split('-')[0];

            _updateOrAddCell(
              id: cellId,
              code: code,
              output: 'Running...',
              evaluated: false,
            );

            try {
              final formattedCode = _kernel!.formatCode(code);
              final output = await _kernel!.execute(code);
              List outputsList;
              try {
                outputsList = jsonDecode(output) as List;
              } catch (_) {
                outputsList = [
                  {'mimeType': 'text/plain', 'data': output},
                ];
              }
              _updateOrAddCell(
                id: cellId,
                code: formattedCode,
                output: output,
                isError: false,
                evaluated: true,
              );
              socket.add(
                jsonEncode({
                  'type': 'result',
                  'id': reqId,
                  'output': output,
                  'outputs': outputsList,
                  'formattedCode': formattedCode,
                  'isError': false,
                }),
              );
            } catch (e) {
              final errStr = '$e';
              _updateOrAddCell(
                id: cellId,
                code: code,
                output: errStr,
                isError: true,
                evaluated: false,
              );
              socket.add(
                jsonEncode({
                  'type': 'result',
                  'id': reqId,
                  'output': errStr,
                  'outputs': [
                    {'mimeType': 'text/plain', 'data': errStr},
                  ],
                  'isError': true,
                }),
              );
            }
          } else if (msgType == 'complete') {
            final reqId = msg['id'] as String;
            final code = msg['code'] as String;
            final offset = (msg['offset'] as num?)?.toInt() ?? code.length;

            final completions = await _kernel!.getCompletions(code, offset);
            socket.add(
              jsonEncode({
                'type': 'completion',
                'id': reqId,
                'completions': completions.map((c) => c.toJson()).toList(),
              }),
            );
          } else if (msgType == 'hover') {
            final reqId = msg['id'] as String;
            final code = msg['code'] as String;
            final offset = (msg['offset'] as num?)?.toInt() ?? code.length;

            final hoverInfo = await _kernel!.getHover(code, offset);
            final hoverHtml = hoverInfo != null
                ? md.markdownToHtml(
                    hoverInfo,
                    extensionSet: md.ExtensionSet.gitHubFlavored,
                  )
                : null;
            socket.add(
              jsonEncode({
                'type': 'hover',
                'id': reqId,
                'hover': hoverInfo,
                'hoverHtml': hoverHtml,
              }),
            );
          }
        } catch (e) {
          print('Error handling WebSocket message: $e');
        }
      },
      onDone: () {
        print('Client disconnected from WebSocket.');
      },
    );
  }

  Future<void> stop() async {
    await _server?.close();
    await _kernel?.stop();
    print('Notebook server stopped.');
  }
}
