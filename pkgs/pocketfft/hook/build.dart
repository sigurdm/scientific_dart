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

    final srcDir = Directory.fromUri(input.packageRoot.resolve('hook/src/'));
    if (!srcDir.existsSync()) {
      srcDir.createSync(recursive: true);
    }

    // 1. Download and extract KissFFT if not present
    final logHeader = File.fromUri(srcDir.uri.resolve('kiss_fft_log.h'));
    if (!logHeader.existsSync()) {
      print('Downloading KissFFT source files archive from GitHub...');
      final client = HttpClient();
      try {
        final request = await client.getUrl(
          Uri.parse(
            'https://github.com/mborgerding/kissfft/archive/6e9e673e420c4bf47d4a60c57c578f93e4ec192f.tar.gz',
          ),
        );
        final response = await request.close();
        if (response.statusCode != 200) {
          throw HttpException(
            'Failed to download KissFFT archive: status ${response.statusCode}',
          );
        }

        final bytesBuilder = BytesBuilder();
        await for (final chunk in response) {
          bytesBuilder.add(chunk);
        }
        final tarGzBytes = bytesBuilder.toBytes();

        final unzippedBytes = GZipDecoder().decodeBytes(tarGzBytes);
        final archive = TarDecoder().decodeBytes(unzippedBytes);

        for (final file in archive) {
          if (file.isFile) {
            final baseName = file.name.split('/').last;
            if (baseName.endsWith('.c') || baseName.endsWith('.h')) {
              final outFile = File.fromUri(srcDir.uri.resolve(baseName));
              outFile.writeAsBytesSync(file.content as List<int>, flush: true);
              print('Extracted: $baseName');
            }
          }
        }
      } finally {
        client.close();
      }
    }

    // 2. Get cross-compiler or host compiler from modern input config
    final packageName = input.packageName;
    final os = input.config.code.targetOS;
    final cCompiler = input.config.code.cCompiler;

    final libName = os == OS.windows
        ? 'libpocketfft.dll'
        : (os == OS.macOS ? 'libpocketfft.dylib' : 'libpocketfft.so');

    final outputDir = Directory.fromUri(input.outputDirectory);
    if (!outputDir.existsSync()) {
      outputDir.createSync(recursive: true);
    }
    final libFile = File.fromUri(outputDir.uri.resolve(libName));

    // 3. Compile plain-C source code using Process.run for extreme reliability
    final compilerPath =
        cCompiler?.compiler.toFilePath() ?? (os == OS.windows ? 'cl' : 'cc');
    print('Compiling pocketfft plain C files via compiler: $compilerPath');

    final compilerLower = compilerPath.toLowerCase();
    final isGNU =
        compilerLower.contains('gcc') ||
        compilerLower.contains('clang') ||
        compilerLower.contains('g++');
    final isMSVC = os == OS.windows && !isGNU;
    final compileArgs = isMSVC
        ? <String>[
            '/LD',
            '/O2',
            '/EHsc',
            '/Dkiss_fft_scalar=double',
            '/I',
            srcDir.uri.toFilePath(),
            srcDir.uri.resolve('kiss_fft.c').toFilePath(),
            srcDir.uri.resolve('kiss_fftr.c').toFilePath(),
            srcDir.uri.resolve('kiss_fftnd.c').toFilePath(),
            '/Fe:${libFile.path}',
          ]
        : <String>[
            '-shared',
            '-fPIC',
            '-O3',
            '-ffast-math',
            '-Dkiss_fft_scalar=double',
            '-I',
            srcDir.uri.toFilePath(),
            srcDir.uri.resolve('kiss_fft.c').toFilePath(),
            srcDir.uri.resolve('kiss_fftr.c').toFilePath(),
            srcDir.uri.resolve('kiss_fftnd.c').toFilePath(),
            '-o',
            libFile.path,
            if (os != OS.windows) '-lm',
          ];

    print('Environment PATH: ${Platform.environment['PATH']}');
    print('Environment INCLUDE: ${Platform.environment['INCLUDE']}');
    print('Environment LIB: ${Platform.environment['LIB']}');

    final res = await Process.run(compilerPath, compileArgs);
    if (res.exitCode != 0) {
      throw StateError(
        'PocketFFT native C compilation failed (exit ${res.exitCode}):\n'
        'stdout: ${res.stdout}\n'
        'stderr: ${res.stderr}',
      );
    }
    print('Compiled shared library binary successfully at: ${libFile.path}');

    // 4. Register the dynamic CodeAsset in the hooks pipeline
    if (libFile.existsSync()) {
      output.assets.code.add(
        CodeAsset(
          package: packageName,
          name: 'pocketfft',
          linkMode: DynamicLoadingBundled(),
          file: libFile.uri,
        ),
      );
      output.dependencies.add(srcDir.uri.resolve('kiss_fft.c'));
      output.dependencies.add(srcDir.uri.resolve('kiss_fftr.c'));
      output.dependencies.add(srcDir.uri.resolve('kiss_fftnd.c'));
      output.dependencies.add(srcDir.uri.resolve('kiss_fft.h'));
      output.dependencies.add(srcDir.uri.resolve('kiss_fft_log.h'));
      output.dependencies.add(srcDir.uri.resolve('kiss_fftnd.h'));
      output.dependencies.add(srcDir.uri.resolve('kiss_fftr.h'));
      output.dependencies.add(srcDir.uri.resolve('_kiss_fft_guts.h'));
      print('Registered pocketfft native dynamic code asset successfully.');
    }
  });
}
