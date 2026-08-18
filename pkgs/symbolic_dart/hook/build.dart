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

    if (os != OS.linux && os != OS.macOS) {
      throw UnimplementedError(
        'symbolic_dart native build currently only supports Linux and macOS.',
      );
    }

    final outputDir = Directory.fromUri(input.outputDirectory);
    if (!outputDir.existsSync()) {
      outputDir.createSync(recursive: true);
    }

    final installDir = Directory.fromUri(outputDir.uri.resolve('install'));
    final libPrefix = os == OS.macOS ? 'lib' : 'lib';
    final ext = os == OS.macOS ? '.dylib' : '.so';

    final flintLib = File(
      installDir.uri.resolve('lib/${libPrefix}flint$ext').toFilePath(),
    );
    final symengineLib = File(
      installDir.uri.resolve('lib/${libPrefix}symengine$ext').toFilePath(),
    );
    final mpfrLib = File(
      installDir.uri.resolve('lib/${libPrefix}mpfr$ext').toFilePath(),
    );

    // Check if MPFR header is on system or needs local download
    final hasSystemMpfrHeader =
        File('/usr/include/mpfr.h').existsSync() ||
        File('/usr/include/x86_64-linux-gnu/mpfr.h').existsSync() ||
        File('/usr/include/aarch64-linux-gnu/mpfr.h').existsSync();

    // Fast local workspace cache from pre-compiled scratch directory
    final prebuiltDir = Directory(
      '/usr/local/google/home/sigurdm/projects/math/scratch/test_stack/install',
    );
    if (prebuiltDir.existsSync() && !flintLib.existsSync()) {
      if (!installDir.existsSync()) installDir.createSync(recursive: true);
      final libDir = Directory(installDir.uri.resolve('lib').toFilePath());
      if (!libDir.existsSync()) libDir.createSync(recursive: true);
      await Process.run('cp', [
        '-r',
        prebuiltDir.uri.resolve('lib').toFilePath(),
        installDir.path,
      ]);
      await Process.run('cp', [
        '-r',
        prebuiltDir.uri.resolve('include').toFilePath(),
        installDir.path,
      ]);
    }

    if (!flintLib.existsSync() || !symengineLib.existsSync()) {
      print('Building SymEngine & FLINT native dependencies in $installDir...');

      final srcDir = Directory.fromUri(outputDir.uri.resolve('src'));
      if (!srcDir.existsSync()) {
        srcDir.createSync(recursive: true);
      }

      // 1. Download & build MPFR if headers missing
      if (!hasSystemMpfrHeader && !mpfrLib.existsSync()) {
        print('Downloading MPFR 4.2.1...');
        final mpfrTar = File(
          srcDir.uri.resolve('mpfr-4.2.1.tar.gz').toFilePath(),
        );
        if (!mpfrTar.existsSync()) {
          final res = await Process.run('curl', [
            '-sSL',
            'https://ftp.gnu.org/gnu/mpfr/mpfr-4.2.1.tar.gz',
            '-o',
            mpfrTar.path,
          ]);
          if (res.exitCode != 0) {
            throw StateError('Failed to download MPFR: ${res.stderr}');
          }
        }

        final mpfrExtract = Directory(
          srcDir.uri.resolve('mpfr-4.2.1').toFilePath(),
        );
        if (!mpfrExtract.existsSync()) {
          await Process.run('tar', [
            '-xzf',
            mpfrTar.path,
          ], workingDirectory: srcDir.path);
        }

        print('Configuring and compiling MPFR...');
        final gmpInc = File('/usr/include/x86_64-linux-gnu/gmp.h').existsSync()
            ? '/usr/include/x86_64-linux-gnu'
            : '/usr/include';
        final confRes = await Process.run('./configure', [
          '--prefix=${installDir.path}',
          '--with-gmp-include=$gmpInc',
          '--disable-static',
          '--enable-shared',
        ], workingDirectory: mpfrExtract.path);
        if (confRes.exitCode != 0) {
          throw StateError(
            'MPFR configure failed:\n${confRes.stdout}\n${confRes.stderr}',
          );
        }

        final makeRes = await Process.run('make', [
          '-j${Platform.numberOfProcessors}',
          'install',
        ], workingDirectory: mpfrExtract.path);
        if (makeRes.exitCode != 0) {
          throw StateError(
            'MPFR make install failed:\n${makeRes.stdout}\n${makeRes.stderr}',
          );
        }
      }

      // 2. Download & build FLINT 3.1.3
      if (!flintLib.existsSync()) {
        print('Downloading FLINT 3.1.3...');
        final flintTar = File(
          srcDir.uri.resolve('flint-3.1.3.tar.gz').toFilePath(),
        );
        if (!flintTar.existsSync()) {
          final res = await Process.run('curl', [
            '-sSL',
            'https://github.com/flintlib/flint/releases/download/v3.1.3/flint-3.1.3.tar.gz',
            '-o',
            flintTar.path,
          ]);
          if (res.exitCode != 0) {
            throw StateError('Failed to download FLINT: ${res.stderr}');
          }
        }

        final flintExtract = Directory(
          srcDir.uri.resolve('flint-3.1.3').toFilePath(),
        );
        if (!flintExtract.existsSync()) {
          await Process.run('tar', [
            '-xzf',
            flintTar.path,
          ], workingDirectory: srcDir.path);
        }

        print('Configuring and compiling FLINT 3.1.3...');
        final gmpInc = File('/usr/include/x86_64-linux-gnu/gmp.h').existsSync()
            ? '/usr/include/x86_64-linux-gnu'
            : '/usr/include';
        final confArgs = [
          '--prefix=${installDir.path}',
          '--with-gmp-include=$gmpInc',
          '--disable-static',
          '--enable-shared',
        ];
        if (!hasSystemMpfrHeader) {
          confArgs.add('--with-mpfr=${installDir.path}');
        }

        final confRes = await Process.run(
          './configure',
          confArgs,
          workingDirectory: flintExtract.path,
        );
        if (confRes.exitCode != 0) {
          throw StateError(
            'FLINT configure failed:\n${confRes.stdout}\n${confRes.stderr}',
          );
        }

        final makeRes = await Process.run('make', [
          '-j${Platform.numberOfProcessors}',
          'install',
        ], workingDirectory: flintExtract.path);
        if (makeRes.exitCode != 0) {
          throw StateError(
            'FLINT make install failed:\n${makeRes.stdout}\n${makeRes.stderr}',
          );
        }
      }

      // 3. Download & build SymEngine 0.11.2
      if (!symengineLib.existsSync()) {
        print('Downloading SymEngine 0.11.2...');
        final symTar = File(
          srcDir.uri.resolve('symengine-0.11.2.tar.gz').toFilePath(),
        );
        if (!symTar.existsSync()) {
          final res = await Process.run('curl', [
            '-sSL',
            'https://github.com/symengine/symengine/releases/download/v0.11.2/symengine-0.11.2.tar.gz',
            '-o',
            symTar.path,
          ]);
          if (res.exitCode != 0) {
            throw StateError('Failed to download SymEngine: ${res.stderr}');
          }
        }

        final symExtract = Directory(
          srcDir.uri.resolve('symengine-0.11.2').toFilePath(),
        );
        if (!symExtract.existsSync()) {
          await Process.run('tar', [
            '-xzf',
            symTar.path,
          ], workingDirectory: srcDir.path);
        }

        final symBuild = Directory(
          symExtract.uri.resolve('build').toFilePath(),
        );
        symBuild.createSync(recursive: true);

        print('Configuring SymEngine CMake...');
        final pkgConfigPath = installDir.uri
            .resolve('lib/pkgconfig')
            .toFilePath();
        final gmpInc = File('/usr/include/x86_64-linux-gnu/gmp.h').existsSync()
            ? '/usr/include/x86_64-linux-gnu'
            : '/usr/include';

        final cmakeArgs = [
          '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
          '-DBUILD_SHARED_LIBS=ON',
          '-DBUILD_TESTS=OFF',
          '-DBUILD_BENCHMARKS=OFF',
          '-DWITH_GMP=ON',
          '-DGMP_INCLUDE_DIR=$gmpInc',
          '-DWITH_FLINT=ON',
          '-DFLINT_INCLUDE_DIR=${installDir.uri.resolve('include').toFilePath()}',
          '-DFLINT_LIBRARY=${flintLib.path}',
          '-DCMAKE_INSTALL_PREFIX=${installDir.path}',
          '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON',
          '-DCMAKE_INSTALL_RPATH=\$ORIGIN',
        ];

        if (!hasSystemMpfrHeader) {
          cmakeArgs.addAll([
            '-DWITH_MPFR=ON',
            '-DMPFR_INCLUDE_DIR=${installDir.uri.resolve('include').toFilePath()}',
            '-DMPFR_LIBRARY=${mpfrLib.path}',
          ]);
        }

        final cmakeRes = await Process.run(
          'cmake',
          cmakeArgs,
          workingDirectory: symBuild.path,
          environment: {'PKG_CONFIG_PATH': pkgConfigPath},
        );
        if (cmakeRes.exitCode != 0) {
          throw StateError(
            'SymEngine CMake failed:\n${cmakeRes.stdout}\n${cmakeRes.stderr}',
          );
        }

        print('Compiling SymEngine...');
        final makeRes = await Process.run('make', [
          '-j${Platform.numberOfProcessors}',
          'install',
        ], workingDirectory: symBuild.path);
        if (makeRes.exitCode != 0) {
          throw StateError(
            'SymEngine make failed:\n${makeRes.stdout}\n${makeRes.stderr}',
          );
        }
      }
    }

    if (!flintLib.existsSync()) {
      throw StateError('FLINT library not found at ${flintLib.path}');
    }
    if (!symengineLib.existsSync()) {
      throw StateError('SymEngine library not found at ${symengineLib.path}');
    }

    final flint19Lib = File(
      installDir.uri.resolve('lib/${libPrefix}flint$ext.19').toFilePath(),
    );
    if (!flint19Lib.existsSync()) {
      flintLib.copySync(flint19Lib.path);
    }

    output.assets.code.add(
      CodeAsset(
        package: packageName,
        name: 'flint',
        linkMode: DynamicLoadingBundled(),
        file: flint19Lib.uri,
      ),
    );

    output.assets.code.add(
      CodeAsset(
        package: packageName,
        name: 'symengine',
        linkMode: DynamicLoadingBundled(),
        file: symengineLib.uri,
      ),
    );
  });
}
