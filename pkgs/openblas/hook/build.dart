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
        final packageName = input.packageName;
        final os = input.config.code.targetOS;
        final cCompiler = input.config.code.cCompiler;
        final compilerPath =
            cCompiler?.compiler.toFilePath() ?? (os == OS.windows ? 'cl' : 'cc');

        final compilerLower = compilerPath.toLowerCase();
        final isGNU =
            compilerLower.contains('gcc') ||
            compilerLower.contains('clang') ||
            compilerLower.contains('g++');
        final isMSVC = os == OS.windows && !isGNU;

        if (os != OS.windows) {
          throw UnimplementedError(
            'Precompiled binaries only supported on Windows for now.',
          );
        }

        final outputDir = Directory.fromUri(input.outputDirectory);
        if (!outputDir.existsSync()) {
          outputDir.createSync(recursive: true);
        }

        final zipUrl =
            'https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.33/OpenBLAS-0.3.33-x64.zip';
        final extractDir = outputDir.uri.resolve('OpenBLAS-precompiled/');
        final extractDirFile = Directory.fromUri(extractDir);

        if (!extractDirFile.existsSync()) {
          print('Downloading precompiled OpenBLAS zip...');
          final client = HttpClient();
          List<int> zipBytes;
          try {
            final request = await client.getUrl(Uri.parse(zipUrl));
            final response = await request.close();
            if (response.statusCode != 200) {
              throw HttpException(
                'Failed to download OpenBLAS zip: status ${response.statusCode}',
              );
            }
            final bytesBuilder = BytesBuilder();
            await for (final chunk in response) {
              bytesBuilder.add(chunk);
            }
            zipBytes = bytesBuilder.toBytes();
          } finally {
            client.close();
          }

          print('Extracting precompiled OpenBLAS zip...');
          final archive = ZipDecoder().decodeBytes(zipBytes);
          for (final file in archive) {
            final outPath = extractDir.resolve(file.name).toFilePath();
            if (file.isFile) {
              final outFile = File(outPath);
              outFile.createSync(recursive: true);
              outFile.writeAsBytesSync(file.content as List<int>, flush: true);
            } else {
              Directory(outPath).createSync(recursive: true);
            }
          }
        }

        // Locate files
        final dllFile = File.fromUri(extractDir.resolve('bin/libopenblas.dll'));
        final headersDir = extractDir.resolve('include/');
        final libDir = extractDir.resolve('lib/');

        final String openblasLibName;
        if (isMSVC) {
          openblasLibName = 'libopenblas.lib';
        } else {
          openblasLibName = 'libopenblas.dll.a';
        }
        final openblasLibFile = File.fromUri(libDir.resolve(openblasLibName));

        if (!dllFile.existsSync()) {
          throw StateError('Expected DLL not found at: ${dllFile.path}');
        }
        if (!openblasLibFile.existsSync()) {
          throw StateError(
            'Expected import library not found at: ${openblasLibFile.path}',
          );
        }

        // Register OpenBLAS binary
        output.assets.code.add(
          CodeAsset(
            package: packageName,
            name: 'openblas',
            linkMode: DynamicLoadingBundled(),
            file: dllFile.uri,
          ),
        );
        output.dependencies.add(
          input.packageRoot.resolve('hook/custom_extensions.c'),
        );

        // Compile custom extensions
        final extLibName = 'libopenblas_extensions.dll'; // We are on Windows
        final extLibFile = File(outputDir.uri.resolve(extLibName).toFilePath());

        final List<String> compileArgs;
        if (isMSVC) {
          compileArgs = [
            '/LD',
            '/O2',
            '/EHsc',
            '/I${headersDir.toFilePath()}',
            input.packageRoot.resolve('hook/custom_extensions.c').toFilePath(),
            '/Fe:${extLibFile.path}',
            openblasLibFile.path,
          ];
        } else {
          compileArgs = [
            '-shared',
            '-fPIC',
            '-O3',
            '-I${headersDir.toFilePath()}',
            input.packageRoot.resolve('hook/custom_extensions.c').toFilePath(),
            '-o',
            extLibFile.path,
            openblasLibFile.path,
          ];
        }

        print(
          'Compiling custom extensions with: $compilerPath ${compileArgs.join(' ')}',
        );
        final extRes = await Process.run(compilerPath, compileArgs);
        if (extRes.exitCode != 0) {
          throw StateError(
            'Failed to compile custom extensions: ${extRes.stderr}',
          );
        }
        print('Compiled custom extensions successfully at: ${extLibFile.path}');

        output.assets.code.add(
          CodeAsset(
            package: packageName,
            name: 'openblas_extensions',
            linkMode: DynamicLoadingBundled(),
            file: extLibFile.uri,
          ),
        );
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
          output.dependencies.add(
            input.packageRoot.resolve('hook/custom_extensions.c'),
          );
          print('Using built OpenBLAS library at ${libFile.path}');

          // Compile custom extensions!
          final extLibName = os == OS.windows
              ? 'libopenblas_extensions.dll'
              : (os == OS.macOS
                    ? 'libopenblas_extensions.dylib'
                    : 'libopenblas_extensions.so');
          final extLibFile = File(
            outputDir.uri.resolve(extLibName).toFilePath(),
          );
          final compilerPath =
              cCompiler?.compiler.toFilePath() ?? (os == OS.windows ? 'cl' : 'cc');
          final isMSVC =
              os == OS.windows && compilerPath.toLowerCase().contains('cl');

          final compileArgs = isMSVC
              ? [
                  '/LD',
                  '/O2',
                  '/EHsc',
                  '/I${extractDir}lapack-netlib/LAPACKE/include',
                  input.packageRoot
                      .resolve('hook/custom_extensions.c')
                      .toFilePath(),
                  '/Fe:${extLibFile.path}',
                  '/link',
                  '/LIBPATH:$extractDir',
                  'libopenblas.lib',
                ]
              : [
                  '-shared',
                  '-fPIC',
                  '-O3',
                  '-I${extractDir}lapack-netlib/LAPACKE/include',
                  input.packageRoot
                      .resolve('hook/custom_extensions.c')
                      .toFilePath(),
                  '-o',
                  extLibFile.path,
                  '-L$extractDir',
                  '-Wl,-rpath,$extractDir',
                  '-lopenblas',
                  '-lm',
                ];

          print(
            'Compiling custom extensions with: $compilerPath ${compileArgs.join(' ')}',
          );
          final extRes = await Process.run(compilerPath, compileArgs);
          if (extRes.exitCode != 0) {
            print('Failed to compile custom extensions: ${extRes.stderr}');
            exit(1);
          }
          print(
            'Compiled custom extensions successfully at: ${extLibFile.path}',
          );

          output.assets.code.add(
            CodeAsset(
              package: packageName,
              name: 'openblas_extensions',
              linkMode: DynamicLoadingBundled(),
              file: extLibFile.uri,
            ),
          );
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

  factory OpenBlasBinary.forBuild(BuildInput input) {
    if (input.config.code.targetOS == OS.windows) {
      return PrecompiledBinary();
    }
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
