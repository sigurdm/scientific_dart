import 'dart:io';
import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) {
      return;
    }

    final packageName = input.packageName;
    final os = input.config.code.targetOS;
    final cCompiler = input.config.code.cCompiler;

    final libName = os == OS.windows
        ? 'libtuner.dll'
        : (os == OS.macOS ? 'libtuner.dylib' : 'libtuner.so');

    final outputDir = Directory.fromUri(input.outputDirectory);
    if (!outputDir.existsSync()) {
      outputDir.createSync(recursive: true);
    }
    final libFile = File.fromUri(outputDir.uri.resolve(libName));

    final compilerPath = cCompiler?.compiler.toFilePath() ?? 'cc';

    final compileArgs = <String>[
      '-shared',
      '-fPIC',
      '-O3',
      input.packageRoot.resolve('hook/tuner_bridge.c').toFilePath(),
      '-o',
      libFile.path,
      '-lm',
    ];

    if (os == OS.linux) {
      compileArgs.add('-lpthread');
      compileArgs.add('-ldl');
    } else if (os == OS.macOS) {
      compileArgs.addAll([
        '-framework',
        'CoreAudio',
        '-framework',
        'AudioToolbox',
        '-framework',
        'AudioUnit',
        '-framework',
        'CoreFoundation',
      ]);
    }

    final res = await Process.run(compilerPath, compileArgs);
    if (res.exitCode != 0) {
      throw StateError('Guitar tuner bridge compilation failed: ${res.stderr}');
    }

    output.assets.code.add(
      CodeAsset(
        package: packageName,
        name: 'tuner_bridge',
        linkMode: DynamicLoadingBundled(),
        file: libFile.uri,
      ),
    );
  });
}
