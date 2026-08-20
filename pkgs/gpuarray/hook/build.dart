import 'dart:io';
import 'dart:typed_data';
import 'package:archive/archive.dart';
import 'package:code_assets/code_assets.dart';
import 'package:crypto/crypto.dart';
import 'package:hooks/hooks.dart';

const String _wgpuVersion = 'v29.0.1.1';
const String _baseUrl =
    'https://github.com/gfx-rs/wgpu-native/releases/download/$_wgpuVersion';

final class _WgpuAssetConfig {
  final String zipName;
  final String sha256;
  final String libName;

  const _WgpuAssetConfig({
    required this.zipName,
    required this.sha256,
    required this.libName,
  });
}

_WgpuAssetConfig? _resolveAsset(OS os, Architecture arch) {
  if (os == OS.linux) {
    if (arch == Architecture.x64) {
      return const _WgpuAssetConfig(
        zipName: 'wgpu-linux-x86_64-release.zip',
        sha256:
            '95a4d90c071005a98d03eab348beaa6b07e16eb00d1dcdb9f8348f75eb97ec5a',
        libName: 'libwgpu_native.so',
      );
    } else if (arch == Architecture.arm64) {
      return const _WgpuAssetConfig(
        zipName: 'wgpu-linux-aarch64-release.zip',
        sha256:
            '015fcdf1dbae82e614a783cc38017e5399ae0927a889fe9b69c9b664bc61b47a',
        libName: 'libwgpu_native.so',
      );
    }
  } else if (os == OS.macOS) {
    if (arch == Architecture.arm64) {
      return const _WgpuAssetConfig(
        zipName: 'wgpu-macos-aarch64-release.zip',
        sha256:
            'a5797a37b1adf720bcd5dcffb291edbbd5b7b14be0a3874c28e6393a655a7a3e',
        libName: 'libwgpu_native.dylib',
      );
    } else if (arch == Architecture.x64) {
      return const _WgpuAssetConfig(
        zipName: 'wgpu-macos-x86_64-release.zip',
        sha256:
            '8e2f7378548ddd0e2cf21e7d864dda46e953f0af724855a33778b85ead206d41',
        libName: 'libwgpu_native.dylib',
      );
    }
  } else if (os == OS.windows) {
    if (arch == Architecture.x64) {
      return const _WgpuAssetConfig(
        zipName: 'wgpu-windows-x86_64-msvc-release.zip',
        sha256:
            '7e67d7445c42aeb85e30f88930fd8d7d83ee769e3390aeb1ada75ebf3cf78132',
        libName: 'wgpu_native.dll',
      );
    } else if (arch == Architecture.arm64) {
      return const _WgpuAssetConfig(
        zipName: 'wgpu-windows-aarch64-msvc-release.zip',
        sha256:
            '4a876421a8c1e5fe72f849b3722214280fe485cb1c56f77f8b0c82414be5b29f',
        libName: 'wgpu_native.dll',
      );
    } else if (arch == Architecture.ia32) {
      return const _WgpuAssetConfig(
        zipName: 'wgpu-windows-i686-msvc-release.zip',
        sha256:
            'ad59d4eadfcfe667999a37e096cc551ecf3f56c387b5a7fd5f61baebf105f54a',
        libName: 'wgpu_native.dll',
      );
    }
  } else if (os == OS.android) {
    if (arch == Architecture.arm64) {
      return const _WgpuAssetConfig(
        zipName: 'wgpu-android-aarch64-release.zip',
        sha256:
            '721741f1b05a20c1738166bedf7a5efb2ba4b382da689526d3fc33de22bdd573',
        libName: 'libwgpu_native.so',
      );
    } else if (arch == Architecture.arm) {
      return const _WgpuAssetConfig(
        zipName: 'wgpu-android-armv7-release.zip',
        sha256:
            'f9d76c77b3fda3f7121476884eb16ec067f7dada83276298a3cc8bf6a8403d60',
        libName: 'libwgpu_native.so',
      );
    } else if (arch == Architecture.x64) {
      return const _WgpuAssetConfig(
        zipName: 'wgpu-android-x86_64-release.zip',
        sha256:
            'ef16fc0644bf0e308a39ac4516742da8e22d8c201d3a542cc5baf533d272c491',
        libName: 'libwgpu_native.so',
      );
    } else if (arch == Architecture.ia32) {
      return const _WgpuAssetConfig(
        zipName: 'wgpu-android-i686-release.zip',
        sha256:
            '593b94875bc4fcc1506ea0b6714dd12b96b7c852921caa63f45eb61517793312',
        libName: 'libwgpu_native.so',
      );
    }
  } else if (os == OS.iOS) {
    if (arch == Architecture.arm64) {
      return const _WgpuAssetConfig(
        zipName: 'wgpu-ios-aarch64-release.zip',
        sha256:
            'e36c9913b9e5095a530fa9121c50b16a4e3dd020e1eebf601f2f47ce24d56941',
        libName: 'libwgpu_native.dylib',
      );
    } else if (arch == Architecture.x64) {
      return const _WgpuAssetConfig(
        zipName: 'wgpu-ios-x86_64-simulator-release.zip',
        sha256:
            '94f67e1b268e8dd31b8e59b32f211ac469f09ed7950fceee52bd84f0623da3d9',
        libName: 'libwgpu_native.dylib',
      );
    }
  }
  return null;
}

Future<Uint8List> _downloadWithRedirects(String url) async {
  final client = HttpClient();
  try {
    var currentUrl = Uri.parse(url);
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
          'Failed to download wgpu-native archive from $currentUrl (HTTP ${response.statusCode})',
        );
      }
      final builder = BytesBuilder();
      await for (final chunk in response) {
        builder.add(chunk);
      }
      return builder.toBytes();
    }
    throw HttpException('Too many redirects while downloading $url');
  } finally {
    client.close();
  }
}

void main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) {
      return;
    }

    final os = input.config.code.targetOS;
    final arch = input.config.code.targetArchitecture;
    final asset = _resolveAsset(os, arch);

    if (asset == null) {
      print(
        'Warning: No prebuilt wgpu-native release configuration for $os $arch. '
        'Hardware acceleration will fall back to CPU simulation mode.',
      );
      return;
    }

    final outputDir = Directory.fromUri(input.outputDirectory);
    if (!outputDir.existsSync()) {
      outputDir.createSync(recursive: true);
    }

    final extractDir = Directory.fromUri(
      outputDir.uri.resolve('wgpu-native-$_wgpuVersion/'),
    );
    if (!extractDir.existsSync()) {
      extractDir.createSync(recursive: true);
    }

    final libFile = File(extractDir.uri.resolve(asset.libName).toFilePath());

    if (!libFile.existsSync()) {
      final downloadUrl = '$_baseUrl/${asset.zipName}';
      print('Downloading wgpu-native release from $downloadUrl...');
      final zipBytes = await _downloadWithRedirects(downloadUrl);

      // Validate cryptographic SHA-256 hash
      final actualSha256 = sha256.convert(zipBytes).toString().toLowerCase();
      final expectedSha256 = asset.sha256.toLowerCase();
      if (actualSha256 != expectedSha256) {
        throw StateError(
          'Security Error: SHA-256 hash mismatch for ${asset.zipName}!\n'
          'Expected: $expectedSha256\n'
          'Actual:   $actualSha256',
        );
      }
      print('SHA-256 hash verified successfully for ${asset.zipName}.');

      // Extract archive
      print('Extracting ${asset.zipName} to ${extractDir.path}...');
      final archive = ZipDecoder().decodeBytes(zipBytes);
      for (final file in archive) {
        if (file.isFile) {
          final baseName = file.name.split('/').last;
          final outFile = File(extractDir.uri.resolve(baseName).toFilePath());
          outFile.writeAsBytesSync(file.content as List<int>, flush: true);
        }
      }
    }

    if (libFile.existsSync()) {
      print('Registering wgpu_native CodeAsset: ${libFile.path}');
      output.assets.code.add(
        CodeAsset(
          package: input.packageName,
          name: 'wgpu_native',
          linkMode: DynamicLoadingBundled(),
          file: libFile.uri,
        ),
      );
    } else {
      print(
        'Warning: Expected library file not found at ${libFile.path} after extraction.',
      );
    }
  });
}
