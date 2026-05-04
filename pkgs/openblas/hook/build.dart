import 'dart:io';
import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) {
      return;
    }

    final openblas = OpenBlasBinary.forBuild(input);
    switch (openblas) {
      case PrecompiledBinary():
        print('Precompiled binary not supported yet.');
        break;
      case CompileOpenBlas(:final sourceUrl):
        final packageName = input.packageName;
        final os = input.config.code.targetOS;
        final arch = input.config.code.targetArchitecture;
        final cCompiler = input.config.code.cCompiler;

        print('Building for OS: $os, Architecture: $arch');

        String openBlasTarget = 'GENERIC'; // Default
        if (arch == Architecture.arm64) {
          openBlasTarget = 'ARMV8';
        } else if (arch == Architecture.arm) {
          openBlasTarget = 'ARMV7';
        } else if (arch == Architecture.ia32) {
          openBlasTarget = 'ATOM';
        }

        final outputDir = Directory.fromUri(input.outputDirectory);
        if (!outputDir.existsSync()) {
          outputDir.createSync(recursive: true);
        }

        final tarballPath = outputDir.uri
            .resolve('OpenBLAS-0.3.33.tar.gz')
            .toFilePath();
        final extractDir = outputDir.uri
            .resolve('OpenBLAS-0.3.33/')
            .toFilePath();

        final libName = os == OS.windows
            ? 'libopenblas.dll'
            : (os == OS.macOS ? 'libopenblas.dylib' : 'libopenblas.so');
        final libFile = File(
          outputDir.uri.resolve('OpenBLAS-0.3.33/$libName').toFilePath(),
        );

        if (!libFile.existsSync()) {
          print('Downloading OpenBLAS release...');
          final downloadResult = await Process.run('curl', [
            '-L',
            sourceUrl,
            '-o',
            tarballPath,
          ]);
          if (downloadResult.exitCode != 0) {
            print('Failed to download OpenBLAS: ${downloadResult.stderr}');
            exit(1);
          }

          print('Extracting OpenBLAS...');
          final extractResult = await Process.run('tar', [
            '-xzf',
            tarballPath,
            '-C',
            outputDir.uri.toFilePath(),
          ]);
          if (extractResult.exitCode != 0) {
            print('Failed to extract OpenBLAS: ${extractResult.stderr}');
            exit(1);
          }

          print('Building OpenBLAS with target $openBlasTarget...');
          final makeArgs = <String>['TARGET=$openBlasTarget', 'USE_THREAD=1'];

          if (cCompiler != null) {
            makeArgs.add('CC=${cCompiler.compiler.toFilePath()}');
            makeArgs.add('AR=${cCompiler.archiver.toFilePath()}');
            print('Using cross-compiler: ${cCompiler.compiler.toFilePath()}');
          } else {
            print('No cross-compiler provided. Using host compiler.');
          }

          final buildResult = await Process.run(
            'make',
            makeArgs,
            workingDirectory: extractDir,
          );
          if (buildResult.exitCode != 0) {
            print('Failed to build OpenBLAS: ${buildResult.stderr}');
            exit(1);
          }
        }

        if (libFile.existsSync()) {
          output.assets.code.add(
            CodeAsset(
              package: packageName,
              name: 'openblas',
              linkMode: DynamicLoadingBundled(),
              file: libFile.uri,
            ),
          );
          print('Using built OpenBLAS library at ${libFile.path}');
        } else {
          print('Built library not found at ${libFile.path}');
        }
        break;
      case ExternalOpenBlas():
        print('External OpenBLAS not supported yet.');
        break;
    }
  });
}

sealed class OpenBlasBinary {
  OpenBlasBinary._();

  factory OpenBlasBinary.forBuild(dynamic input) {
    return CompileOpenBlas(
      'https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.33/OpenBLAS-0.3.33.tar.gz',
    );
  }
}

class PrecompiledBinary extends OpenBlasBinary {
  PrecompiledBinary() : super._();
}

class CompileOpenBlas extends OpenBlasBinary {
  final String sourceUrl;
  CompileOpenBlas(this.sourceUrl) : super._();
}

class ExternalOpenBlas extends OpenBlasBinary {
  ExternalOpenBlas() : super._();
}
