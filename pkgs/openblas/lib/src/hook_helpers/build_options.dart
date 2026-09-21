import 'dart:io';

import 'package:hooks/hooks.dart';

/// Build mode for resolving the native `openblas` libraries in `hook/build.dart`.
enum BuildModeEnum {
  /// Fetch precompiled and provenance-attested binaries from GitHub Releases.
  fetch,

  /// Use locally existing binaries specified via `localPath` (and optional `localExtensionsPath`).
  local,

  /// Compile OpenBLAS and/or custom extensions on the host machine.
  source,
}

/// Configuration options for the `openblas` native build hook.
final class BuildOptions {
  /// Selected build mode (`fetch`, `local`, or `source`).
  final BuildModeEnum buildMode;

  /// Path to a directory containing `libopenblas` and `libopenblas_extensions`
  /// (or path to `libopenblas` directly when [localExtensionsPath] is also provided).
  final Uri? localPath;

  /// Optional path to `libopenblas_extensions` when [localPath] points to the
  /// `libopenblas` file rather than a directory.
  final Uri? localExtensionsPath;

  /// Path to a local package/source checkout when [buildMode] is [BuildModeEnum.source].
  final Uri? checkoutPath;

  /// Creates a [BuildOptions] configuration.
  const BuildOptions({
    required this.buildMode,
    this.localPath,
    this.localExtensionsPath,
    this.checkoutPath,
  });

  /// Parses [BuildOptions] from `pubspec.yaml` `hooks.user_defines.openblas`
  /// with optional environment variable overrides (`OPENBLAS_BUILD_MODE`,
  /// `SCIENTIFIC_DART_BUILD_MODE`, `LOCAL_OPENBLAS_BINARY`,
  /// `LOCAL_OPENBLAS_EXTENSIONS_BINARY`, `LOCAL_OPENBLAS_CHECKOUT`).
  ///
  /// It is an error if `buildMode` is not one of `fetch`, `local`, or `source`.
  factory BuildOptions.fromDefines(HookInputUserDefines defines) {
    final envMode =
        Platform.environment['OPENBLAS_BUILD_MODE'] ??
        Platform.environment['SCIENTIFIC_DART_BUILD_MODE'];
    final rawMode = envMode ?? defines['buildMode'];
    final buildMode = switch (rawMode) {
      null || 'fetch' => BuildModeEnum.fetch,
      'local' => BuildModeEnum.local,
      'source' || 'checkout' => BuildModeEnum.source,
      final other => throw ArgumentError(
        'Unknown buildMode "$other" for package:openblas.',
      ),
    };

    final envLocalPath = Platform.environment['LOCAL_OPENBLAS_BINARY'];
    Uri? localPath;
    if (envLocalPath != null && envLocalPath.isNotEmpty) {
      localPath = FileSystemEntity.isDirectorySync(envLocalPath)
          ? Uri.directory(envLocalPath)
          : Uri.file(envLocalPath);
    } else {
      localPath = defines.path('localPath');
    }

    final envLocalExtPath =
        Platform.environment['LOCAL_OPENBLAS_EXTENSIONS_BINARY'];
    final localExtensionsPath =
        envLocalExtPath != null && envLocalExtPath.isNotEmpty
        ? Uri.file(envLocalExtPath)
        : defines.path('localExtensionsPath');

    final envCheckoutPath = Platform.environment['LOCAL_OPENBLAS_CHECKOUT'];
    final checkoutPath = envCheckoutPath != null && envCheckoutPath.isNotEmpty
        ? Uri.directory(envCheckoutPath)
        : defines.path('checkoutPath');

    return BuildOptions(
      buildMode: buildMode,
      localPath: localPath,
      localExtensionsPath: localExtensionsPath,
      checkoutPath: checkoutPath,
    );
  }

  /// Returns a formatted usage message for `pubspec.yaml` configuration.
  static String usageError(Object error) =>
      '''
Error: $error

Set the build mode for `openblas` with either `fetch`, `local`, or `source` in your workspace `pubspec.yaml`:

* fetch: Download the precompiled binaries from GitHub Releases (verified via SHA-256 & SLSA provenance).
```yaml
hooks:
  user_defines:
    openblas:
      buildMode: fetch
```

* local: Use locally existing binaries from a directory or explicit file paths.
```yaml
hooks:
  user_defines:
    openblas:
      buildMode: local
      localPath: path/to/dir_or_libopenblas.so
      localExtensionsPath: path/to/libopenblas_extensions.so # optional if localPath is a directory
```

* source: Compile OpenBLAS / Accelerate wrappers and custom extensions on the host machine.
```yaml
hooks:
  user_defines:
    openblas:
      buildMode: source
```
''';

  @override
  String toString() =>
      'BuildOptions(buildMode: ${buildMode.name}, localPath: $localPath, '
      'localExtensionsPath: $localExtensionsPath, checkoutPath: $checkoutPath)';
}
