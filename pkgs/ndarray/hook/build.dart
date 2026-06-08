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
        ? 'libndarray.dll'
        : (os == OS.macOS ? 'libndarray.dylib' : 'libndarray.so');

    final outputDir = Directory.fromUri(input.outputDirectory);
    if (!outputDir.existsSync()) {
      outputDir.createSync(recursive: true);
    }
    final libFile = File.fromUri(outputDir.uri.resolve(libName));

    final compilerPath = cCompiler?.compiler.toFilePath() ?? 'cc';
    var cppCompilerPath = compilerPath;
    if (compilerPath.endsWith('gcc')) {
      cppCompilerPath =
          '${compilerPath.substring(0, compilerPath.length - 3)}g++';
    } else if (compilerPath.endsWith('clang')) {
      cppCompilerPath = '$compilerPath++';
    } else if (compilerPath.endsWith('cc')) {
      cppCompilerPath = 'c++';
    } else if (compilerPath.contains('gcc-')) {
      cppCompilerPath = compilerPath.replaceAll('gcc-', 'g++-');
    } else if (compilerPath.contains('clang-')) {
      cppCompilerPath = compilerPath.replaceAll('clang-', 'clang++-');
    }

    print(
      'Compiling ndarray custom C++ extensions using compiler: $cppCompilerPath',
    );

    final isMSVC =
        os == OS.windows && compilerPath.toLowerCase().contains('cl');
    if (isMSVC) {
      final ufuncsObj = outputDir.uri.resolve('custom_ufuncs.obj').toFilePath();
      final sortingObj = outputDir.uri
          .resolve('custom_sorting.obj')
          .toFilePath();

      var res = await Process.run(compilerPath, [
        '/c',
        '/O2',
        '/EHsc',
        '/I${input.packageRoot.toFilePath()}',
        input.packageRoot.resolve('hook/custom_ufuncs.c').toFilePath(),
        '/Fo:$ufuncsObj',
      ]);
      if (res.exitCode != 0) {
        throw StateError('Ufuncs compilation failed: ${res.stderr}');
      }

      res = await Process.run(cppCompilerPath, [
        '/c',
        '/O2',
        '/EHsc',
        '/I${input.packageRoot.toFilePath()}',
        '/I${input.packageRoot.resolve('third_party/highway').toFilePath()}',
        input.packageRoot.resolve('hook/custom_sorting.cpp').toFilePath(),
        '/Fo:$sortingObj',
      ]);
      if (res.exitCode != 0) {
        throw StateError('Sorting compilation failed: ${res.stderr}');
      }

      res = await Process.run(cppCompilerPath, [
        '/LD',
        ufuncsObj,
        sortingObj,
        input.packageRoot
            .resolve('third_party/highway/build/libhwy_contrib.a')
            .toFilePath(),
        input.packageRoot
            .resolve('third_party/highway/build/libhwy.a')
            .toFilePath(),
        '/Fe:${libFile.path}',
      ]);
      if (res.exitCode != 0) throw StateError('Linking failed: ${res.stderr}');
    } else {
      final ufuncsObj = outputDir.uri.resolve('custom_ufuncs.o').toFilePath();
      final sortingObj = outputDir.uri.resolve('custom_sorting.o').toFilePath();

      print('Compiling custom_ufuncs.c...');
      var res = await Process.run(compilerPath, [
        '-c',
        '-fPIC',
        '-O3',
        '-I${input.packageRoot.toFilePath()}',
        input.packageRoot.resolve('hook/custom_ufuncs.c').toFilePath(),
        '-o',
        ufuncsObj,
      ]);
      if (res.exitCode != 0) {
        throw StateError('Ufuncs compilation failed: ${res.stderr}');
      }

      print('Compiling custom_sorting.cpp...');
      res = await Process.run(cppCompilerPath, [
        '-c',
        '-fPIC',
        '-O3',
        '-I${input.packageRoot.toFilePath()}',
        '-I${input.packageRoot.resolve('third_party/highway').toFilePath()}',
        input.packageRoot.resolve('hook/custom_sorting.cpp').toFilePath(),
        '-o',
        sortingObj,
      ]);
      if (res.exitCode != 0) {
        throw StateError('Sorting compilation failed: ${res.stderr}');
      }

      print('Linking shared library...');
      res = await Process.run(cppCompilerPath, [
        '-shared',
        '-fPIC',
        ufuncsObj,
        sortingObj,
        input.packageRoot
            .resolve('third_party/highway/build/libhwy_contrib.a')
            .toFilePath(),
        input.packageRoot
            .resolve('third_party/highway/build/libhwy.a')
            .toFilePath(),
        '-o',
        libFile.path,
        '-lm',
      ]);
      if (res.exitCode != 0) throw StateError('Linking failed: ${res.stderr}');
    }
    print('Compiled ndarray shared library successfully at: ${libFile.path}');

    // Register the dynamic CodeAsset in the hooks pipeline under package asset ID namespace
    if (libFile.existsSync()) {
      output.assets.code.add(
        CodeAsset(
          package: packageName,
          name: 'ndarray',
          linkMode: DynamicLoadingBundled(),
          file: libFile.uri,
        ),
      );
      output.dependencies.add(
        input.packageRoot.resolve('hook/custom_sorting.cpp'),
      );
      output.dependencies.add(
        input.packageRoot.resolve('hook/custom_sorting.h'),
      );
      output.dependencies.add(
        input.packageRoot.resolve('hook/custom_ufuncs.c'),
      );
      output.dependencies.add(
        input.packageRoot.resolve('hook/custom_ufuncs.h'),
      );
      output.dependencies.add(
        input.packageRoot.resolve('third_party/timsort/timsort.h'),
      );
      print(
        'Registered ndarray native custom extensions code asset successfully.',
      );
    }
  });
}
