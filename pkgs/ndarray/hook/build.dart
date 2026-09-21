import 'dart:io';
import 'dart:typed_data';

import 'package:code_assets/code_assets.dart';
import 'package:crypto/crypto.dart';
import 'package:hooks/hooks.dart';
import 'package:ndarray/src/hook_helpers/build_options.dart';
import 'package:ndarray/src/hook_helpers/hashes.dart';

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
    print('ndarray build options: $buildOptions');

    final currentSourceHash = computeNativeSourceHash(input.packageRoot);
    if (currentSourceHash != nativeSourceHash &&
        buildOptions.buildMode == BuildModeEnum.source) {
      print(
        'WARNING: Native sources in package:${input.packageName}/hook/ '
        '(${currentSourceHash.substring(0, 12)}) differ from prebuilt release '
        '$version (${nativeSourceHash.substring(0, 12)}). '
        'Remember to build & attest new release artifacts and run '
        '`dart tool/regenerate_hashes.dart <tag>` before publishing.',
      );
    }

    final buildMode = switch (buildOptions.buildMode) {
      BuildModeEnum.fetch => FetchMode(input),
      BuildModeEnum.local => LocalMode(input, buildOptions.localPath),
      BuildModeEnum.source => SourceMode(input, buildOptions.checkoutPath),
    };

    final builtLibrary = await buildMode.build();

    output.assets.code.add(
      CodeAsset(
        package: input.packageName,
        name: 'ndarray',
        linkMode: DynamicLoadingBundled(),
        file: builtLibrary,
      ),
    );
    output.dependencies.addAll(buildMode.dependencies);
    output.dependencies.add(input.packageRoot.resolve('pubspec.yaml'));
  });
}

String _canonicalLibName(OS os) => os == OS.windows
    ? 'libndarray.dll'
    : ((os == OS.macOS || os == OS.iOS) ? 'libndarray.dylib' : 'libndarray.so');

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
    final currentSourceHash = computeNativeSourceHash(input.packageRoot);
    if (currentSourceHash != nativeSourceHash) {
      throw StateError(
        'Prebuilt ndarray binary for release $version is out of date with native sources in hook/!\n'
        'Pinned nativeSourceHash: $nativeSourceHash\n'
        'Current hook/ hash:      $currentSourceHash\n'
        'If you are the package author, build and attest new release artifacts (.github/workflows/artifacts.yml) and run:\n'
        '  dart tool/regenerate_hashes.dart <new-release-tag>\n'
        '${BuildOptions.usageError('Switch to `buildMode: source` while developing native code.')}',
      );
    }

    final os = input.config.code.targetOS;
    final arch = input.config.code.targetArchitecture;
    final artifactName = ndarrayArtifactName(os, arch);
    final expectedHash = fileHashes[(os, arch)];

    if (expectedHash == null || expectedHash.startsWith('00000000')) {
      throw StateError(
        'No prebuilt ndarray binary hash is pinned for ($os, $arch) in release $version.\n'
        '${BuildOptions.usageError('Switch to `buildMode: source` or `buildMode: local`.')}',
      );
    }

    final libName = _canonicalLibName(os);
    final cachedLibrary = File.fromUri(
      input.outputDirectoryShared
          .resolve('ndarray-$version/${os.name}-${arch.name}/')
          .resolve(libName),
    );

    if (await cachedLibrary.exists()) {
      final cachedHash = sha256
          .convert(await cachedLibrary.readAsBytes())
          .toString();
      if (cachedHash == expectedHash) {
        print('Using cached ndarray binary from ${cachedLibrary.path}.');
        return cachedLibrary.uri;
      }
    }

    final remoteUri = Uri.parse(
      'https://github.com/$repository/releases/download/$version/$artifactName',
    );
    print('Fetching prebuilt ndarray binary from $remoteUri...');
    final bytes = await _downloadBytesWithRedirects(remoteUri);
    final actualHash = sha256.convert(bytes).toString();
    if (actualHash != expectedHash) {
      throw StateError(
        'SHA-256 mismatch for prebuilt ndarray binary at $remoteUri:\n'
        'Expected: $expectedHash\n'
        'Actual:   $actualHash',
      );
    }

    await cachedLibrary.parent.create(recursive: true);
    await cachedLibrary.writeAsBytes(bytes, flush: true);
    return cachedLibrary.uri;
  }

  @override
  List<Uri> get dependencies => [
    for (final file in nativeSourceFiles(input.packageRoot)) file.uri,
  ];
}

final class LocalMode extends BuildMode {
  final Uri? localPath;

  LocalMode(super.input, this.localPath);

  File _resolveLocalFile() {
    if (localPath == null) {
      throw ArgumentError(
        '`localPath` is not set in `hooks.user_defines.ndarray` '
        '(or `LOCAL_NDARRAY_BINARY` environment variable).',
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
          ndarrayArtifactName(os, input.config.code.targetArchitecture),
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
        'Could not find local ndarray binary.',
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

  SourceMode(super.input, this.checkoutPath);

  Uri get _root => checkoutPath ?? input.packageRoot;

  @override
  Future<Uri> build() async {
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
    final msvcEnv = <String, String>{
      ...Platform.environment,
      if (isMSVC) ...await getMSVCEnvironment(arch),
    };

    var cppCompilerPath = compilerPath;
    if (isMSVC) {
      for (final candidate in const [
        r'C:\Program Files\LLVM\bin\clang-cl.exe',
        'clang-cl.exe',
      ]) {
        try {
          final check = await Process.run(candidate, [
            '--version',
          ], environment: msvcEnv);
          if (check.exitCode == 0) {
            cppCompilerPath = candidate;
            break;
          }
        } catch (_) {}
      }
    } else if (compilerPath.endsWith('gcc')) {
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

    final highwayDir = _root.resolve('third_party/highway/');
    final legacyHwyDir = Directory.fromUri(outputDir.uri.resolve('hwy_build'));
    final highwayBuildDir = legacyHwyDir.existsSync()
        ? legacyHwyDir
        : Directory.fromUri(
            input.outputDirectoryShared.resolve(
              'hwy_build-${os.name}-${arch.name}/',
            ),
          );

    final String hwyLibName = isMSVC ? 'hwy.lib' : 'libhwy.a';
    final String hwyContribLibName = isMSVC
        ? 'hwy_contrib.lib'
        : 'libhwy_contrib.a';

    File resolveHwyLib(String name) {
      final direct = File.fromUri(highwayBuildDir.uri.resolve(name));
      if (direct.existsSync()) return direct;
      final release = File.fromUri(
        highwayBuildDir.uri.resolve('Release/$name'),
      );
      if (release.existsSync()) return release;
      return direct;
    }

    if (!resolveHwyLib(hwyLibName).existsSync() ||
        !resolveHwyLib(hwyContribLibName).existsSync()) {
      print('Highway static libraries not found. Compiling highway...');
      if (!highwayBuildDir.existsSync()) {
        highwayBuildDir.createSync(recursive: true);
      }

      final cmakeRes = await Process.run(
        'cmake',
        [
          if (isMSVC) ...['-G', 'NMake Makefiles'],
          '-DCMAKE_BUILD_TYPE=Release',
          '-DCMAKE_CXX_STANDARD=17',
          '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
          '-DHWY_ENABLE_TESTS=OFF',
          '-DHWY_ENABLE_EXAMPLES=OFF',
          '-DHWY_COMPILE_ONLY_STATIC=ON',
          if (arch == Architecture.x64)
            isMSVC
                ? '-DCMAKE_CXX_FLAGS=/arch:AVX2 /DHWY_COMPILE_ONLY_STATIC=1'
                : '-DCMAKE_CXX_FLAGS=-mavx2 -mfma -mf16c -DHWY_COMPILE_ONLY_STATIC=1'
          else
            isMSVC
                ? '-DCMAKE_CXX_FLAGS=/DHWY_COMPILE_ONLY_STATIC=1'
                : '-DCMAKE_CXX_FLAGS=-DHWY_COMPILE_ONLY_STATIC=1',
          if (cCompiler != null) ...[
            '-DCMAKE_C_COMPILER=$compilerPath',
            '-DCMAKE_CXX_COMPILER=$cppCompilerPath',
          ],
          if (os == OS.macOS || os == OS.iOS)
            '-DCMAKE_OSX_ARCHITECTURES=${arch == Architecture.arm64 ? 'arm64' : 'x86_64'}',
          highwayDir.toFilePath(),
        ],
        workingDirectory: highwayBuildDir.path,
        environment: msvcEnv,
      );

      if (cmakeRes.exitCode != 0) {
        throw StateError(
          'CMake failed for highway (exit ${cmakeRes.exitCode}):\n'
          'stdout: ${cmakeRes.stdout}\n'
          'stderr: ${cmakeRes.stderr}',
        );
      }

      final buildRes = await Process.run(
        'cmake',
        [
          '--build',
          '.',
          '--target',
          'hwy',
          'hwy_contrib',
          if (!isMSVC) '--parallel',
        ],
        workingDirectory: highwayBuildDir.path,
        environment: msvcEnv,
      );

      if (buildRes.exitCode != 0) {
        throw StateError(
          'Build failed for highway (exit ${buildRes.exitCode}):\n'
          'stdout: ${buildRes.stdout}\n'
          'stderr: ${buildRes.stderr}',
        );
      }
    }

    final libhwy = resolveHwyLib(hwyLibName);
    final libhwyContrib = resolveHwyLib(hwyContribLibName);

    if (isMSVC) {
      final sharedObjDir = Directory.fromUri(
        input.outputDirectoryShared.resolve(
          'ndarray-objs-${os.name}-${arch.name}/',
        ),
      );
      if (!sharedObjDir.existsSync()) {
        sharedObjDir.createSync(recursive: true);
      }
      final ufuncsObj = sharedObjDir.uri
          .resolve('custom_ufuncs.obj')
          .toFilePath();
      final sortingObj = sharedObjDir.uri
          .resolve('custom_sorting.obj')
          .toFilePath();
      final indexingObj = sharedObjDir.uri
          .resolve('custom_indexing.obj')
          .toFilePath();
      final minizObj = sharedObjDir.uri.resolve('miniz.obj').toFilePath();
      final npzIoObj = sharedObjDir.uri.resolve('npz_io.obj').toFilePath();
      final winBuiltinsSrc = sharedObjDir.uri
          .resolve('win_builtins.c')
          .toFilePath();
      final winBuiltinsObj = sharedObjDir.uri
          .resolve('win_builtins.obj')
          .toFilePath();

      await File(winBuiltinsSrc).writeAsString('''
typedef unsigned __int128 uint128_t;
typedef __int128 int128_t;

uint128_t __udivti3(uint128_t n, uint128_t d) {
  if (d == 0) return 0;
  uint128_t q = 0;
  uint128_t r = 0;
  for (int i = 127; i >= 0; i--) {
    r = (r << 1) | ((n >> i) & 1);
    if (r >= d) {
      r -= d;
      q |= ((uint128_t)1 << i);
    }
  }
  return q;
}

int128_t __divti3(int128_t a, int128_t b) {
  int neg = 0;
  uint128_t ua = (uint128_t)a;
  uint128_t ub = (uint128_t)b;
  if (a < 0) {
    ua = -ua;
    neg ^= 1;
  }
  if (b < 0) {
    ub = -ub;
    neg ^= 1;
  }
  uint128_t uq = __udivti3(ua, ub);
  return neg ? -(int128_t)uq : (int128_t)uq;
}
''');

      Future<void> runMsvcCompile(
        String label,
        String exe,
        List<String> args,
      ) async {
        final res = await Process.run(exe, args, environment: msvcEnv);
        if (res.exitCode != 0) {
          throw StateError(
            '$label compilation failed:\nstdout: ${res.stdout}\nstderr: ${res.stderr}',
          );
        }
      }

      await Future.wait([
        runMsvcCompile('Ufuncs', cppCompilerPath, [
          '/c',
          '/std:c++17',
          '/bigobj',
          '/O2',
          '/MD',
          '/EHsc',
          if (arch == Architecture.x64) '/arch:AVX2',
          '/D_USE_MATH_DEFINES',
          '/DNOMINMAX',
          '/DVECTORIZED_TARGETS=',
          '/I${_root.toFilePath()}',
          _root.resolve('hook/custom_ufuncs.cpp').toFilePath(),
          '/Fo:$ufuncsObj',
        ]),
        runMsvcCompile('Sorting', cppCompilerPath, [
          '/c',
          '/std:c++17',
          '/bigobj',
          '/O2',
          '/MD',
          '/EHsc',
          if (arch == Architecture.x64) '/arch:AVX2',
          '/D_USE_MATH_DEFINES',
          '/DNOMINMAX',
          '/DHWY_COMPILE_ONLY_STATIC=1',
          '/I${_root.toFilePath()}',
          '/I${_root.resolve('third_party/highway/').toFilePath()}',
          _root.resolve('hook/custom_sorting.cpp').toFilePath(),
          '/Fo:$sortingObj',
        ]),
        runMsvcCompile('Indexing', cppCompilerPath, [
          '/c',
          '/std:c++17',
          '/bigobj',
          '/O2',
          '/MD',
          '/EHsc',
          if (arch == Architecture.x64) '/arch:AVX2',
          '/D_USE_MATH_DEFINES',
          '/DNOMINMAX',
          '/I${_root.toFilePath()}',
          _root.resolve('hook/custom_indexing.cpp').toFilePath(),
          '/Fo:$indexingObj',
        ]),
        runMsvcCompile('miniz', compilerPath, [
          '/c',
          '/O2',
          '/MD',
          '/I${_root.toFilePath()}',
          _root.resolve('third_party/miniz/miniz.c').toFilePath(),
          '/Fo:$minizObj',
        ]),
        runMsvcCompile('npz_io', cppCompilerPath, [
          '/c',
          '/std:c++17',
          '/bigobj',
          '/O2',
          '/MD',
          '/EHsc',
          '/DNOMINMAX',
          '/I${_root.toFilePath()}',
          _root.resolve('hook/npz_io.cpp').toFilePath(),
          '/Fo:$npzIoObj',
        ]),
        runMsvcCompile('win_builtins', cppCompilerPath, [
          '/c',
          '/O2',
          '/MD',
          winBuiltinsSrc,
          '/Fo:$winBuiltinsObj',
        ]),
      ]);

      final allExports = [
        ...extractExportsFromBindings(
          _root.resolve('lib/src/ndarray_bindings.dart').toFilePath(),
        ),
        ...extractExportsFromBindings(
          _root
              .resolve('lib/src/ndarray_extensions_bindings.dart')
              .toFilePath(),
        ),
      ];

      final defFile = File(
        outputDir.uri.resolve('libndarray.def').toFilePath(),
      );
      await defFile.writeAsString(
        ['LIBRARY libndarray', 'EXPORTS', ...allExports].join('\n'),
      );

      final res = await Process.run(cppCompilerPath, [
        '/LD',
        '/MD',
        ufuncsObj,
        sortingObj,
        indexingObj,
        minizObj,
        npzIoObj,
        winBuiltinsObj,
        libhwyContrib.path,
        libhwy.path,
        '/Fe:${libFile.path}',
        '/link',
        '/def:${defFile.path}',
      ], environment: msvcEnv);
      if (res.exitCode != 0) {
        throw StateError(
          'Linking failed:\nstdout: ${res.stdout}\nstderr: ${res.stderr}',
        );
      }
    } else {
      final sharedObjDir = Directory.fromUri(
        input.outputDirectoryShared.resolve(
          'ndarray-objs-${os.name}-${arch.name}/',
        ),
      );
      if (!sharedObjDir.existsSync()) {
        sharedObjDir.createSync(recursive: true);
      }
      final ufuncsObj = sharedObjDir.uri
          .resolve('custom_ufuncs.o')
          .toFilePath();
      final sortingObj = sharedObjDir.uri
          .resolve('custom_sorting.o')
          .toFilePath();
      final indexingObj = sharedObjDir.uri
          .resolve('custom_indexing.o')
          .toFilePath();
      final minizObj = sharedObjDir.uri.resolve('miniz.o').toFilePath();
      final npzIoObj = sharedObjDir.uri.resolve('npz_io.o').toFilePath();

      String computeInputDigest(String src) {
        final bytes = BytesBuilder(copy: false);
        bytes.add(File(src).readAsBytesSync());
        for (final header in const [
          'hook/custom_indexing.h',
          'hook/custom_sorting.h',
          'hook/custom_ufuncs.h',
          'hook/npz_io.h',
        ]) {
          final hF = File(_root.resolve(header).toFilePath());
          if (hF.existsSync()) {
            bytes.add(hF.readAsBytesSync());
          }
        }
        return sha256.convert(bytes.takeBytes()).toString();
      }

      Future<bool> compileIfNeeded(
        String label,
        String src,
        String obj,
        String exe,
        List<String> args,
      ) async {
        final objF = File(obj);
        final hashF = File('$obj.sha256');
        final digest = computeInputDigest(src);
        if (objF.existsSync() &&
            hashF.existsSync() &&
            hashF.readAsStringSync().trim() == digest) {
          return false;
        }
        final res = await Process.run(exe, args);
        if (res.exitCode != 0) {
          throw StateError('$label compilation failed: ${res.stderr}');
        }
        await hashF.writeAsString(digest);
        return true;
      }

      final ufuncsSrc = _root.resolve('hook/custom_ufuncs.cpp').toFilePath();
      final sortingSrc = _root.resolve('hook/custom_sorting.cpp').toFilePath();
      final indexingSrc = _root
          .resolve('hook/custom_indexing.cpp')
          .toFilePath();
      final minizSrc = _root.resolve('third_party/miniz/miniz.c').toFilePath();
      final npzIoSrc = _root.resolve('hook/npz_io.cpp').toFilePath();

      final compiledAny = await Future.wait([
        compileIfNeeded('Ufuncs', ufuncsSrc, ufuncsObj, cppCompilerPath, [
          if (os == OS.macOS || os == OS.iOS) ...[
            '-arch',
            arch == Architecture.arm64 ? 'arm64' : 'x86_64',
          ],
          '-std=c++17',
          '-c',
          '-fPIC',
          '-O2',
          if (arch == Architecture.x64) ...['-mavx2', '-mfma', '-mf16c'],
          '-DVECTORIZED_TARGETS=',
          '-fno-math-errno',
          '-I${_root.toFilePath()}',
          ufuncsSrc,
          '-o',
          ufuncsObj,
        ]),
        compileIfNeeded('Sorting', sortingSrc, sortingObj, cppCompilerPath, [
          if (os == OS.macOS || os == OS.iOS) ...[
            '-arch',
            arch == Architecture.arm64 ? 'arm64' : 'x86_64',
          ],
          '-std=c++17',
          '-c',
          '-fPIC',
          '-O2',
          if (arch == Architecture.x64) ...['-mavx2', '-mfma', '-mf16c'],
          '-DHWY_COMPILE_ONLY_STATIC=1',
          '-fno-math-errno',
          '-I${_root.toFilePath()}',
          '-I${_root.resolve('third_party/highway/').toFilePath()}',
          sortingSrc,
          '-o',
          sortingObj,
        ]),
        compileIfNeeded('Indexing', indexingSrc, indexingObj, cppCompilerPath, [
          if (os == OS.macOS || os == OS.iOS) ...[
            '-arch',
            arch == Architecture.arm64 ? 'arm64' : 'x86_64',
          ],
          '-std=c++17',
          '-c',
          '-fPIC',
          '-O2',
          if (arch == Architecture.x64) ...['-mavx2', '-mfma', '-mf16c'],
          '-fno-math-errno',
          '-I${_root.toFilePath()}',
          indexingSrc,
          '-o',
          indexingObj,
        ]),
        compileIfNeeded('miniz', minizSrc, minizObj, compilerPath, [
          if (os == OS.macOS || os == OS.iOS) ...[
            '-arch',
            arch == Architecture.arm64 ? 'arm64' : 'x86_64',
          ],
          '-c',
          '-fPIC',
          '-O3',
          '-I${_root.toFilePath()}',
          minizSrc,
          '-o',
          minizObj,
        ]),
        compileIfNeeded('npz_io', npzIoSrc, npzIoObj, cppCompilerPath, [
          if (os == OS.macOS || os == OS.iOS) ...[
            '-arch',
            arch == Architecture.arm64 ? 'arm64' : 'x86_64',
          ],
          '-std=c++17',
          '-c',
          '-fPIC',
          '-O3',
          '-I${_root.toFilePath()}',
          npzIoSrc,
          '-o',
          npzIoObj,
        ]),
      ]);

      final needsLink = !libFile.existsSync() || compiledAny.any((c) => c);

      if (needsLink) {
        final res = await Process.run(cppCompilerPath, [
          if (os == OS.macOS || os == OS.iOS) ...[
            '-arch',
            arch == Architecture.arm64 ? 'arm64' : 'x86_64',
            '-Wl,-install_name,@rpath/$libName',
          ],
          '-shared',
          '-fPIC',
          if (os == OS.android) '-Wl,-z,max-page-size=16384',
          ufuncsObj,
          sortingObj,
          indexingObj,
          minizObj,
          npzIoObj,
          libhwyContrib.path,
          libhwy.path,
          '-o',
          libFile.path,
          if (os != OS.windows) '-lm',
        ]);
        if (res.exitCode != 0) {
          throw StateError('Linking failed: ${res.stderr}');
        }
      }
    }

    return libFile.uri;
  }

  @override
  List<Uri> get dependencies => [
    _root.resolve('hook/custom_sorting.cpp'),
    _root.resolve('hook/custom_sorting.h'),
    _root.resolve('hook/custom_ufuncs.cpp'),
    _root.resolve('hook/custom_ufuncs.h'),
    _root.resolve('hook/custom_indexing.cpp'),
    _root.resolve('hook/custom_indexing.h'),
    _root.resolve('hook/npz_io.cpp'),
    _root.resolve('hook/npz_io.h'),
    _root.resolve('third_party/miniz/miniz.c'),
    _root.resolve('third_party/miniz/miniz.h'),
    _root.resolve('third_party/timsort/timsort.h'),
  ];
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

List<String> extractExportsFromBindings(String bindingsPath) {
  final file = File(bindingsPath);
  if (!file.existsSync()) return [];

  final content = file.readAsStringSync();
  final regex = RegExp(r'external\s+[\w\d_<>.]+\s+(\w+)\s*\(');

  final exports = <String>[];
  for (final match in regex.allMatches(content)) {
    final name = match.group(1);
    if (name != null && !exports.contains(name)) {
      exports.add(name);
    }
  }
  return exports;
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
