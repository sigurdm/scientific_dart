import 'dart:io';
import 'package:path/path.dart' as p;
import 'package:test/test.dart';
import 'package:notebook/notebook.dart';

void main() {
  late NotebookKernel kernel;

  setUp(() async {
    final workspaceDir = Directory.current.path;
    final sdkPath =
        Platform.environment['DART_SDK'] ??
        p.dirname(p.dirname(Platform.resolvedExecutable));

    kernel = NotebookKernel(workspaceDir: workspaceDir, dartSdkPath: sdkPath);
    await kernel.start();
  });

  tearDown(() async {
    await kernel.stop();
  });

  test('evaluates Image(NDArray) returning rendered BMP data URL', () async {
    await kernel.execute('''
var imgData = NDArray.fromList([
  1.0, 0.0, 0.0,
  0.0, 1.0, 0.0,
  0.0, 0.0, 1.0,
  1.0, 1.0, 0.0
], [2, 2, 3], DType.float64);
''');

    final result = await kernel.execute('Image(imgData)');
    expect(result, contains('data:image/bmp;base64,'));
  });

  test('evaluates 2D grayscale Image(NDArray)', () async {
    await kernel.execute('''
var grayData = NDArray.fromList([
  0.0, 0.5,
  0.8, 1.0
], [2, 2], DType.float64);
''');

    final result = await kernel.execute('Image(grayData)');
    expect(result, contains('data:image/bmp;base64,'));
  });
}
