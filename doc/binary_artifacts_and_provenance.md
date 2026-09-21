# Prebuilt Binary Artifacts & SLSA Provenance (`intl4x` style)

The native packages in this workspace (`pocketfft`, `openblas`, `ndarray`, `symbolic_dart`, and `gpuarray`) use Dart Native Assets (`hook/build.dart`) modeled after `package:intl4x` / `package:icu4x`.

## Build Modes (`fetch`, `local`, `source`)

Each package supports three build modes configured via `hooks.user_defines` in the workspace root `pubspec.yaml` (or overridden via environment variables):

1. **`fetch` (default for package consumers)**:
   Downloads prebuilt dynamic libraries for `(targetOS, targetArchitecture)` from GitHub Releases (`https://github.com/sigurdm/scientific_dart/releases/download/<version>/<artifact>`), caches them in `input.outputDirectoryShared`, and verifies their SHA-256 digest against the pinned `fileHashes` table in `lib/src/hook_helpers/hashes.dart`.
   ```yaml
   hooks:
     user_defines:
       ndarray:
         buildMode: fetch
       openblas:
         buildMode: fetch
       pocketfft:
         buildMode: fetch
   ```

2. **`local` (offline / custom prebuilt binaries)**:
   Uses locally existing dynamic library files from `localPath` (or environment variables `LOCAL_NDARRAY_BINARY`, `LOCAL_OPENBLAS_BINARY`, `LOCAL_POCKETFFT_BINARY`).
   ```yaml
   hooks:
     user_defines:
       ndarray:
         buildMode: local
         localPath: path/to/libndarray.so
       openblas:
         buildMode: local
         localPath: path/to/directory_containing_openblas_libs/
       pocketfft:
         buildMode: local
         localPath: path/to/libpocketfft.so
   ```

3. **`source` (default when developing inside this workspace)**:
   Compiles the native C/C++/Fortran libraries from source on the host machine. The root `pubspec.yaml` of this repository configures `buildMode: source` so local changes to C/C++ files in `hook/` are compiled automatically during development.
   ```yaml
   hooks:
     user_defines:
       ndarray:
         buildMode: source
       openblas:
         buildMode: source
       pocketfft:
         buildMode: source
   ```

### Environment Variable Overrides

You can override the build mode without editing `pubspec.yaml`:
- `SCIENTIFIC_DART_BUILD_MODE=fetch|local|source` (applies to all packages)
- `NDARRAY_BUILD_MODE=fetch|local|source` + `LOCAL_NDARRAY_BINARY=/path/to/lib`
- `OPENBLAS_BUILD_MODE=fetch|local|source` + `LOCAL_OPENBLAS_BINARY=/path/to/dir`
- `POCKETFFT_BUILD_MODE=fetch|local|source` + `LOCAL_POCKETFFT_BINARY=/path/to/lib`

---

## Releasing New Prebuilt Artifacts with SLSA Provenance

1. **Tag and push an artifact release** (or trigger `.github/workflows/artifacts.yml` via `workflow_dispatch`):
   ```bash
   git tag artifacts-v0.0.2
   git push origin artifacts-v0.0.2
   ```
2. **GitHub Actions (`.github/workflows/artifacts.yml`)**:
   - Compiles the native binaries across `linux-x64`, `linux-arm64`, `macos-arm64`, `macos-x64`, and `windows-x64` using `dart tool/build_artifacts.dart`.
   - Generates `SHA256SUMS.txt`.
   - Signs SLSA v1.0 build provenance attestations for all binaries using `actions/attest-build-provenance@v2` (Sigstore/Fulcio) and bundles `provenance.intoto.jsonl`.
   - Publishes the binaries and provenance bundle to the GitHub Release.
3. **Verify Provenance & Update Pinned Hashes**:
   Run `tool/regenerate_hashes.dart`:
   ```bash
   dart tool/regenerate_hashes.dart artifacts-v0.0.2
   ```
   This script downloads every release artifact, runs `gh attestation verify <artifact> --repo sigurdm/scientific_dart` to cryptographically verify its SLSA provenance, computes the SHA-256 digest, and updates `lib/src/hook_helpers/hashes.dart` in `pocketfft`, `ndarray`, and `openblas`.
4. **Commit the updated `hashes.dart` files** and publish the packages.
