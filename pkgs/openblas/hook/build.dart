import 'dart:io';
import 'dart:typed_data';
import 'package:archive/archive.dart';
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
          final client = HttpClient();
          List<int> tarGzBytes;
          try {
            final request = await client.getUrl(Uri.parse(sourceUrl));
            final response = await request.close();
            if (response.statusCode != 200) {
              throw HttpException(
                'Failed to download OpenBLAS: status ${response.statusCode}',
              );
            }
            final bytesBuilder = BytesBuilder();
            await for (final chunk in response) {
              bytesBuilder.add(chunk);
            }
            tarGzBytes = bytesBuilder.toBytes();
          } finally {
            client.close();
          }

          print('Extracting OpenBLAS...');
          final unzippedBytes = GZipDecoder().decodeBytes(tarGzBytes);
          final archive = TarDecoder().decodeBytes(unzippedBytes);

          for (final file in archive) {
            final outPath = outputDir.uri.resolve(file.name).toFilePath();
            if (file.isFile) {
              final outFile = File(outPath);
              outFile.createSync(recursive: true);
              outFile.writeAsBytesSync(file.content as List<int>, flush: true);
            } else {
              Directory(outPath).createSync(recursive: true);
            }
          }

          print('Building OpenBLAS with target $openBlasTarget...');
          final makeArgs = <String>[
            '-j${Platform.numberOfProcessors}',
            'TARGET=$openBlasTarget',
            'USE_THREAD=1',
          ];

          if (cCompiler != null) {
            makeArgs.add('CC=${cCompiler.compiler.toFilePath()}');
            makeArgs.add('AR=${cCompiler.archiver.toFilePath()}');
            print('Using cross-compiler: ${cCompiler.compiler.toFilePath()}');
          } else {
            print('No cross-compiler provided. Using host compiler.');
          }

          // Restore executable permissions for OpenBLAS build scripts (lost during Dart TarDecoder extraction)
          await Process.run('chmod', [
            '-R',
            '+x',
            '.',
          ], workingDirectory: extractDir);

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
          output.dependencies.add(libFile.uri);
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
