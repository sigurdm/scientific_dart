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
        ? 'libnumdart.dll'
        : (os == OS.macOS ? 'libnumdart.dylib' : 'libnumdart.so');

    final outputDir = Directory.fromUri(input.outputDirectory);
    if (!outputDir.existsSync()) {
      outputDir.createSync(recursive: true);
    }
    final libFile = File.fromUri(outputDir.uri.resolve(libName));

    // Compile custom C code if shared binary asset is absent
    if (!libFile.existsSync()) {
      final compilerPath = cCompiler?.compiler?.toFilePath() ?? 'cc';
      print('Compiling num_dart custom C extensions using compiler: $compilerPath');

      final compileArgs = <String>[
        '-shared',
        '-fPIC',
        '-O3',
        input.packageRoot.resolve('hook/custom_sorting.c').toFilePath(),
        '-o',
        libFile.path,
      ];

      final res = await Process.run(compilerPath, compileArgs);
      if (res.exitCode != 0) {
        throw StateError('num_dart native C extensions compilation failed: ${res.stderr}');
      }
      print('Compiled num_dart shared library successfully at: ${libFile.path}');
    }

    // Register the dynamic CodeAsset in the hooks pipeline under package asset ID namespace
    if (libFile.existsSync()) {
      output.assets.code.add(
        CodeAsset(
          package: packageName,
          name: 'num_dart',
          linkMode: DynamicLoadingBundled(),
          file: libFile.uri,
        ),
      );
      print('Registered num_dart native custom extensions code asset successfully.');
    }
  });
}
