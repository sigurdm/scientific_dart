import 'dart:io';

import 'package:hooks/hooks.dart';

/// Build mode for resolving the native `ndarray` library in `hook/build.dart`.
enum BuildModeEnum {
  /// Fetch precompiled and provenance-attested binary from GitHub Releases.
  fetch,

  /// Use a locally existing binary specified via `localPath`.
  local,

  /// Compile the native library from C/C++ source on the host machine.
  source,
}

/// Configuration options for the `ndarray` native build hook.
final class BuildOptions {
  /// Selected build mode (`fetch`, `local`, or `source`).
  final BuildModeEnum buildMode;

  /// Whether [buildMode] was explicitly specified via user defines or environment variables.
  final bool isExplicit;

  /// Path to a prebuilt dynamic library when [buildMode] is [BuildModeEnum.local].
  final Uri? localPath;

  /// Path to a local package/source checkout when [buildMode] is [BuildModeEnum.source].
  final Uri? checkoutPath;

  /// Creates a [BuildOptions] configuration.
  const BuildOptions({
    required this.buildMode,
    this.isExplicit = false,
    this.localPath,
    this.checkoutPath,
  });

  /// Parses [BuildOptions] from `pubspec.yaml` `hooks.user_defines.ndarray`
  /// with optional environment variable overrides (`NDARRAY_BUILD_MODE`,
  /// `SCIENTIFIC_DART_BUILD_MODE`, `LOCAL_NDARRAY_BINARY`,
  /// `LOCAL_NDARRAY_CHECKOUT`).
  ///
  /// It is an error if `buildMode` is not one of `fetch`, `local`, or `source`.
  factory BuildOptions.fromDefines(HookInputUserDefines defines) {
    final envMode =
        Platform.environment['NDARRAY_BUILD_MODE'] ??
        Platform.environment['SCIENTIFIC_DART_BUILD_MODE'];
    final rawMode = envMode ?? defines['buildMode'];
    final buildMode = switch (rawMode) {
      null || 'fetch' => BuildModeEnum.fetch,
      'local' => BuildModeEnum.local,
      'source' || 'checkout' => BuildModeEnum.source,
      final other => throw ArgumentError(
        'Unknown buildMode "$other" for package:ndarray.',
      ),
    };

    final envLocalPath = Platform.environment['LOCAL_NDARRAY_BINARY'];
    final localPath = envLocalPath != null && envLocalPath.isNotEmpty
        ? Uri.file(envLocalPath)
        : defines.path('localPath');

    final envCheckoutPath = Platform.environment['LOCAL_NDARRAY_CHECKOUT'];
    final checkoutPath = envCheckoutPath != null && envCheckoutPath.isNotEmpty
        ? Uri.directory(envCheckoutPath)
        : defines.path('checkoutPath');

    return BuildOptions(
      buildMode: buildMode,
      isExplicit: rawMode != null,
      localPath: localPath,
      checkoutPath: checkoutPath,
    );
  }

  /// Returns a formatted usage message for `pubspec.yaml` configuration.
  static String usageError(Object error) =>
      '''
Error: $error

Set the build mode for `ndarray` with either `fetch`, `local`, or `source` in your workspace `pubspec.yaml`:

* fetch: Download the precompiled binary from GitHub Releases (verified via SHA-256 & SLSA provenance).
```yaml
hooks:
  user_defines:
    ndarray:
      buildMode: fetch
```

* local: Use a locally existing binary (or set `LOCAL_NDARRAY_BINARY`).
```yaml
hooks:
  user_defines:
    ndarray:
      buildMode: local
      localPath: path/to/libndarray.so
```

* source: Compile a fresh library from C/C++ source on the host machine.
```yaml
hooks:
  user_defines:
    ndarray:
      buildMode: source
```
''';

  @override
  String toString() =>
      'BuildOptions(buildMode: ${buildMode.name}, localPath: $localPath, checkoutPath: $checkoutPath)';
}
