import 'dart:io';
import 'dart:typed_data';

import 'package:archive/archive.dart';
import 'package:code_assets/code_assets.dart';
import 'package:crypto/crypto.dart';
import 'package:hooks/hooks.dart';
import 'package:pocketfft/src/hook_helpers/build_options.dart';
import 'package:pocketfft/src/hook_helpers/hashes.dart';

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) {
      return;
    }

    final BuildOptions buildOptions;
    try {
      buildOptions = BuildOptions.fromDefines(input.userDefines);
    } catch (e) {
      throw ArgumentError(BuildOptions.usageError(e));
    }
    print('pocketfft build options: $buildOptions');

    final buildMode = switch (buildOptions.buildMode) {
      BuildModeEnum.fetch => FetchMode(input),
      BuildModeEnum.local => LocalMode(input, buildOptions.localPath),
      BuildModeEnum.source => SourceMode(input, buildOptions.checkoutPath),
    };

    final builtLibrary = await buildMode.build();

    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: 'pocketfft',
        linkMode: DynamicLoadingBundled(),
        file: builtLibrary,
      ),
    );
    output.dependencies.addAll(buildMode.dependencies);
    output.dependencies.add(input.packageRoot.resolve('pubspec.yaml'));
  });
}

String _canonicalLibName(OS os) => os == OS.windows
    ? 'libpocketfft.dll'
    : ((os == OS.macOS || os == OS.iOS)
          ? 'libpocketfft.dylib'
          : 'libpocketfft.so');

sealed class BuildMode {
  final BuildInput input;

  const BuildMode(this.input);

  List<Uri> get dependencies;

  Future<Uri> build();
}

final class FetchMode extends BuildMode {
  FetchMode(super.input);

  @override
  Future<Uri> build() async {
    final os = input.config.code.targetOS;
    final arch = input.config.code.targetArchitecture;
    final artifactName = pocketfftArtifactName(os, arch);
    final expectedHash = fileHashes[(os, arch)];

    if (expectedHash == null || expectedHash.startsWith('00000000')) {
      throw StateError(
        'No prebuilt pocketfft binary hash is pinned for ($os, $arch) in release $version.\n'
        '${BuildOptions.usageError('Switch to `buildMode: source` or `buildMode: local`.')}',
      );
    }

    final libName = _canonicalLibName(os);
    final cachedLibrary = File.fromUri(
      input.outputDirectoryShared
          .resolve('pocketfft-$version/${os.name}-${arch.name}/')
          .resolve(libName),
    );

    if (await cachedLibrary.exists()) {
      final cachedHash = sha256
          .convert(await cachedLibrary.readAsBytes())
          .toString();
      if (cachedHash == expectedHash) {
        print('Using cached pocketfft binary from ${cachedLibrary.path}.');
        return cachedLibrary.uri;
      }
    }

    final remoteUri = Uri.parse(
      'https://github.com/$repository/releases/download/$version/$artifactName',
    );
    print('Fetching prebuilt pocketfft binary from $remoteUri...');
    final bytes = await _downloadBytesWithRedirects(remoteUri);
    final actualHash = sha256.convert(bytes).toString();
    if (actualHash != expectedHash) {
      throw StateError(
        'SHA-256 mismatch for prebuilt pocketfft binary at $remoteUri:\n'
        'Expected: $expectedHash\n'
        'Actual:   $actualHash',
      );
    }

    await cachedLibrary.parent.create(recursive: true);
    await cachedLibrary.writeAsBytes(bytes, flush: true);
    return cachedLibrary.uri;
  }

  @override
  List<Uri> get dependencies => const [];
}

final class LocalMode extends BuildMode {
  final Uri? localPath;

  LocalMode(super.input, this.localPath);

  File _resolveLocalFile() {
    if (localPath == null) {
      throw ArgumentError(
        '`localPath` is not set in `hooks.user_defines.pocketfft` '
        '(or `LOCAL_POCKETFFT_BINARY` environment variable).',
      );
    }
    final os = input.config.code.targetOS;
    final entityPath = localPath!.toFilePath(windows: Platform.isWindows);
    if (FileSystemEntity.isDirectorySync(entityPath)) {
      final candidate = File.fromUri(
        Directory(entityPath).uri.resolve(_canonicalLibName(os)),
      );
      if (candidate.existsSync()) return candidate;
      final artifactCandidate = File.fromUri(
        Directory(entityPath).uri.resolve(
          pocketfftArtifactName(os, input.config.code.targetArchitecture),
        ),
      );
      if (artifactCandidate.existsSync()) return artifactCandidate;
      throw FileSystemException(
        'Could not find ${_canonicalLibName(os)} in localPath directory.',
        entityPath,
      );
    }
    final file = File(entityPath);
    if (!file.existsSync()) {
      throw FileSystemException(
        'Could not find local pocketfft binary.',
        entityPath,
      );
    }
    return file;
  }

  @override
  Future<Uri> build() async {
    final sourceFile = _resolveLocalFile();
    final targetUri = input.outputDirectory.resolve(
      _canonicalLibName(input.config.code.targetOS),
    );
    final targetFile = File.fromUri(targetUri);
    await targetFile.parent.create(recursive: true);
    await sourceFile.copy(targetFile.path);
    return targetFile.uri;
  }

  @override
  List<Uri> get dependencies => [_resolveLocalFile().uri];
}

final class SourceMode extends BuildMode {
  final Uri? checkoutPath;
  final List<Uri> _recordedDependencies = [];

  SourceMode(super.input, this.checkoutPath);

  @override
  Future<Uri> build() async {
    final rootUri = checkoutPath ?? input.packageRoot;
    final packageSrcUri = rootUri.resolve('hook/src/');
    final usePackageSrc = File.fromUri(
      packageSrcUri.resolve('kiss_fft_log.h'),
    ).existsSync();
    final srcDir = usePackageSrc
        ? Directory.fromUri(packageSrcUri)
        : Directory.fromUri(input.outputDirectory.resolve('kissfft_src/'));

    final logHeader = File.fromUri(srcDir.uri.resolve('kiss_fft_log.h'));
    if (!logHeader.existsSync()) {
      if (!srcDir.existsSync()) {
        srcDir.createSync(recursive: true);
      }
      print('Downloading KissFFT source files archive from GitHub...');
      final tarGzBytes = await _downloadBytesWithRedirects(
        Uri.parse(
          'https://github.com/mborgerding/kissfft/archive/6e9e673e420c4bf47d4a60c57c578f93e4ec192f.tar.gz',
        ),
      );

      final actualHash = sha256.convert(tarGzBytes).toString();
      const expectedHash =
          '3da5fb17fa446f5368a7e9c71e2ae6a1a29a9940f7afc499a3e70224a17c95e5';
      if (actualHash != expectedHash) {
        throw StateError(
          'SHA-256 mismatch for KissFFT archive: expected $expectedHash, got $actualHash',
        );
      }

      final unzippedBytes = GZipDecoder().decodeBytes(tarGzBytes);
      final archive = TarDecoder().decodeBytes(unzippedBytes);

      final safeSrcPrefix = srcDir.path.endsWith(Platform.pathSeparator)
          ? srcDir.path
          : '${srcDir.path}${Platform.pathSeparator}';

      for (final file in archive) {
        if (file.isFile) {
          final cleanName = file.name.replaceAll('\\', '/');
          final baseName = cleanName.split('/').last;
          if (baseName.contains('..') ||
              baseName.contains('/') ||
              baseName.contains('\\')) {
            throw FormatException(
              'Invalid filename in KissFFT archive: ${file.name}',
            );
          }
          if (baseName.endsWith('.c') || baseName.endsWith('.h')) {
            final outFile = File.fromUri(srcDir.uri.resolve(baseName));
            if (!outFile.path.startsWith(safeSrcPrefix)) {
              throw FormatException(
                'Path traversal attempt in KissFFT archive: ${file.name}',
              );
            }
            outFile.writeAsBytesSync(file.content as List<int>, flush: true);
          }
        }
      }
    }

    final os = input.config.code.targetOS;
    final arch = input.config.code.targetArchitecture;
    final cCompiler = input.config.code.cCompiler;

    final libName = _canonicalLibName(os);
    final outputDir = Directory.fromUri(input.outputDirectory);
    if (!outputDir.existsSync()) {
      outputDir.createSync(recursive: true);
    }
    final libFile = File.fromUri(outputDir.uri.resolve(libName));

    final compilerPath =
        cCompiler?.compiler.toFilePath() ?? (os == OS.windows ? 'cl' : 'cc');
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
            '/link',
            '/EXPORT:kiss_fft_alloc',
            '/EXPORT:kiss_fft',
            '/EXPORT:kiss_fftr_alloc',
            '/EXPORT:kiss_fftr',
            '/EXPORT:kiss_fftri',
            '/EXPORT:kiss_fftnd_alloc',
            '/EXPORT:kiss_fftnd',
          ]
        : <String>[
            if (os == OS.macOS || os == OS.iOS) ...[
              '-arch',
              arch == Architecture.arm64 ? 'arm64' : 'x86_64',
              '-Wl,-install_name,@rpath/$libName',
            ],
            '-shared',
            '-fPIC',
            '-O3',
            '-ffast-math',
            if (os == OS.android) '-Wl,-z,max-page-size=16384',
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

    final runEnv = <String, String>{...Platform.environment};
    if (isMSVC) {
      final msvcEnv = await getMSVCEnvironment(arch);
      for (final key in ['INCLUDE', 'LIB', 'LIBPATH']) {
        final val = msvcEnv[key] ?? msvcEnv[key.toLowerCase()];
        if (val != null) {
          runEnv[key] = val;
        }
      }
    }

    final res = await Process.run(
      compilerPath,
      compileArgs,
      environment: runEnv,
    );
    if (res.exitCode != 0) {
      throw StateError(
        'PocketFFT native C compilation failed (exit ${res.exitCode}):\n'
        'stdout: ${res.stdout}\n'
        'stderr: ${res.stderr}',
      );
    }

    if (usePackageSrc) {
      for (final srcFile in const [
        'kiss_fft.c',
        'kiss_fftr.c',
        'kiss_fftnd.c',
        'kiss_fft.h',
        'kiss_fft_log.h',
        'kiss_fftnd.h',
        'kiss_fftr.h',
        '_kiss_fft_guts.h',
      ]) {
        _recordedDependencies.add(srcDir.uri.resolve(srcFile));
      }
    }

    return libFile.uri;
  }

  @override
  List<Uri> get dependencies => _recordedDependencies;
}

Future<Uint8List> _downloadBytesWithRedirects(Uri url) async {
  final client = HttpClient();
  try {
    var currentUrl = url;
    for (var redirectCount = 0; redirectCount < 5; redirectCount++) {
      final request = await client.getUrl(currentUrl);
      final response = await request.close();
      if (response.statusCode >= 300 &&
          response.statusCode < 400 &&
          response.headers.value(HttpHeaders.locationHeader) != null) {
        currentUrl = currentUrl.resolve(
          response.headers.value(HttpHeaders.locationHeader)!,
        );
        continue;
      }
      if (response.statusCode != 200) {
        throw HttpException(
          'Failed to download $currentUrl (HTTP ${response.statusCode})',
        );
      }
      final builder = BytesBuilder(copy: false);
      await for (final chunk in response) {
        builder.add(chunk);
      }
      return builder.takeBytes();
    }
    throw HttpException('Too many redirects while downloading $url');
  } finally {
    client.close();
  }
}

Future<Map<String, String>> getMSVCEnvironment(Architecture targetArch) async {
  if (!Platform.isWindows) return {};

  String vswherePath = 'vswhere.exe';
  final programFilesX86 =
      Platform.environment['ProgramFiles(x86)'] ?? 'C:\\Program Files (x86)';
  final defaultVswhere =
      '$programFilesX86\\Microsoft Visual Studio\\Installer\\vswhere.exe';
  if (await File(defaultVswhere).exists()) {
    vswherePath = defaultVswhere;
  }

  try {
    final vswhereRes = await Process.run(vswherePath, [
      '-latest',
      '-property',
      'installationPath',
    ]);
    if (vswhereRes.exitCode != 0) return {};

    final vsPath = vswhereRes.stdout.toString().trim();
    if (vsPath.isEmpty) return {};

    final vcvarsPath = '$vsPath\\VC\\Auxiliary\\Build\\vcvarsall.bat';
    if (!await File(vcvarsPath).exists()) return {};

    final vcvarsArch = targetArch == Architecture.arm64
        ? 'arm64'
        : (targetArch == Architecture.ia32 ? 'x86' : 'amd64');

    final tempDir = Directory.systemTemp;
    final tempFile = File(
      '${tempDir.path}\\get_msvc_env_${DateTime.now().millisecondsSinceEpoch}.bat',
    );
    await tempFile.writeAsString(
      '@echo off\ncall "$vcvarsPath" $vcvarsArch\nset\n',
    );
    final envRes = await Process.run('cmd.exe', ['/c', tempFile.path]);
    try {
      await tempFile.delete();
    } catch (_) {}

    if (envRes.exitCode != 0) return {};

    final envMap = <String, String>{};
    for (final line in envRes.stdout.toString().split('\n')) {
      final parts = line.split('=');
      if (parts.length >= 2) {
        final key = parts[0].trim();
        final value = parts.sublist(1).join('=').trim();
        if (key.isNotEmpty) {
          envMap[key] = value;
        }
      }
    }
    return envMap;
  } catch (_) {
    return {};
  }
}
