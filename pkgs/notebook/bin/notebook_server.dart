import 'dart:io';
import 'package:path/path.dart' as p;
import 'package:notebook/src/notebook_server.dart';

void main(List<String> args) async {
  print('=== Dart NDArray Notebook Web Server ===');

  final dartSdkPath = p.dirname(p.dirname(Platform.resolvedExecutable));
  final workspaceDir = p.dirname(p.dirname(Platform.script.toFilePath()));

  int port = 8080;
  if (args.isNotEmpty) {
    port = int.tryParse(args[0]) ?? 8080;
  }

  print('SDK Path: $dartSdkPath');
  print('Workspace Dir: $workspaceDir');

  final server = NotebookServer(
    workspaceDir: workspaceDir,
    dartSdkPath: dartSdkPath,
    port: port,
  );

  try {
    await server.start();
    print(
      '\n🚀 Notebook is live! Open http://localhost:$port in your browser.',
    );
    print('Press Ctrl+C or kill process to terminate.\n');

    // Keep process alive
    await ProcessSignal.sigint.watch().first;
  } catch (e, stack) {
    print('Error running notebook server: $e');
    print(stack);
  } finally {
    print('\nStopping server...');
    await server.stop();
    print('Goodbye.');
  }
}
