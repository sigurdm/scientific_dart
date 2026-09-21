import 'dart:io';

import 'package:code_assets/code_assets.dart';
import 'package:crypto/crypto.dart';
import 'package:hooks/hooks.dart';
import 'package:pocketfft/src/hook_helpers/build_options.dart';
import 'package:pocketfft/src/hook_helpers/hashes.dart';
import 'package:test/test.dart';

void main() {
  group('BuildOptions & hashes', () {
    test('defaults to BuildModeEnum.fetch when user_defines is empty', () {
      final inputBuilder = BuildInputBuilder()
        ..setupShared(
          packageRoot: Directory.current.uri,
          packageName: 'pocketfft',
          outputDirectoryShared: Directory.systemTemp.uri,
          outputFile: Directory.systemTemp.uri.resolve('out.json'),
        )
        ..config.setupBuild(linkingEnabled: false);
      final input = BuildInput(inputBuilder.json);

      // Only if env var is not overriding
      if (!Platform.environment.containsKey('POCKETFFT_BUILD_MODE') &&
          !Platform.environment.containsKey('SCIENTIFIC_DART_BUILD_MODE')) {
        final options = BuildOptions.fromDefines(input.userDefines);
        expect(options.buildMode, equals(BuildModeEnum.fetch));
      }
    });

    test('parses buildMode: local and resolves relative localPath', () {
      final baseUri = Directory.current.uri;
      final inputBuilder = BuildInputBuilder()
        ..setupShared(
          packageRoot: baseUri,
          packageName: 'pocketfft',
          outputDirectoryShared: Directory.systemTemp.uri,
          outputFile: Directory.systemTemp.uri.resolve('out.json'),
          userDefines: PackageUserDefines(
            workspacePubspec: PackageUserDefinesSource(
              defines: const {
                'buildMode': 'local',
                'localPath': 'dist/libpocketfft.so',
              },
              basePath: baseUri,
            ),
          ),
        )
        ..config.setupBuild(linkingEnabled: false);
      final input = BuildInput(inputBuilder.json);

      if (!Platform.environment.containsKey('POCKETFFT_BUILD_MODE') &&
          !Platform.environment.containsKey('SCIENTIFIC_DART_BUILD_MODE')) {
        final options = BuildOptions.fromDefines(input.userDefines);
        expect(options.buildMode, equals(BuildModeEnum.local));
        expect(
          options.localPath,
          equals(baseUri.resolve('dist/libpocketfft.so')),
        );
      }
    });

    test('throws ArgumentError on unknown buildMode', () {
      final baseUri = Directory.current.uri;
      final inputBuilder = BuildInputBuilder()
        ..setupShared(
          packageRoot: baseUri,
          packageName: 'pocketfft',
          outputDirectoryShared: Directory.systemTemp.uri,
          outputFile: Directory.systemTemp.uri.resolve('out.json'),
          userDefines: PackageUserDefines(
            workspacePubspec: PackageUserDefinesSource(
              defines: const {'buildMode': 'invalid_mode'},
              basePath: baseUri,
            ),
          ),
        )
        ..config.setupBuild(linkingEnabled: false);
      final input = BuildInput(inputBuilder.json);

      if (!Platform.environment.containsKey('POCKETFFT_BUILD_MODE') &&
          !Platform.environment.containsKey('SCIENTIFIC_DART_BUILD_MODE')) {
        expect(
          () => BuildOptions.fromDefines(input.userDefines),
          throwsArgumentError,
        );
      }
    });

    test('pocketfftArtifactName formats canonical artifact names', () {
      expect(
        pocketfftArtifactName(OS.linux, Architecture.x64),
        equals('pocketfft-linux-x64.so'),
      );
      expect(
        pocketfftArtifactName(OS.macOS, Architecture.arm64),
        equals('pocketfft-macos-arm64.dylib'),
      );
      expect(
        pocketfftArtifactName(OS.windows, Architecture.x64),
        equals('pocketfft-windows-x64.dll'),
      );
      expect(fileHashes.containsKey((OS.linux, Architecture.x64)), isTrue);
      expect(sha256.convert(const [1, 2, 3]).toString().length, equals(64));
    });
  });
}
