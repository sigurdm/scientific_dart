import 'dart:io';
import 'package:path/path.dart' as p;
import 'package:notebook/src/notebook_kernel.dart';

void main() async {
  print('Starting Dart Notebook Kernel...');

  final dartSdkPath = p.dirname(p.dirname(Platform.resolvedExecutable));
  final workspaceDir = p.dirname(p.dirname(Platform.script.toFilePath()));

  print('SDK Path: $dartSdkPath');
  print('Workspace Dir: $workspaceDir');

  final kernel = NotebookKernel(
    workspaceDir: workspaceDir,
    dartSdkPath: dartSdkPath,
  );

  try {
    await kernel.start();
    print('Kernel active. Type your expressions or declarations.');
    print('Type "exit" to quit.\n');

    while (true) {
      stdout.write('dart_notebook> ');
      final line = stdin.readLineSync();
      if (line == null) break;

      final trimmed = line.trim();
      if (trimmed == 'exit' || trimmed == 'quit') {
        break;
      }
      if (trimmed.isEmpty) continue;

      try {
        final result = await kernel.execute(trimmed);
        if (result.isNotEmpty) {
          print(result);
        }
      } catch (e) {
        print('Error: $e');
      }
      print('');
    }
  } catch (e, stack) {
    print('Failed to run notebook: $e');
    print(stack);
  } finally {
    print('Stopping kernel...');
    await kernel.stop();
    print('Goodbye.');
  }
}
