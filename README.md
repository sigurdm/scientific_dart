# Math Workspace Monorepo

Packages for scientific computing in Dart.

## Repository Structure

The repository is structured as a standard Dart workspace monorepo under `pubspec.yaml`:

- **[pkgs/num_dart](file:///usr/local/google/home/sigurdm/projects/math/pkgs/num_dart)**: The core scientific tensor package. It supports n-dimensional arrays (`NDArray`), universal element-wise ufuncs, advanced multidimensional stride slicing, logical reductions, and Marsaglia-RNG distributions.
- **[pkgs/openblas](file:///usr/local/google/home/sigurdm/projects/math/pkgs/openblas)**: Minimalistic FFI bindings layer wrapping OpenBLAS cblas and LAPACK headers. Includes compile hooks to build OpenBLAS from source tarballs.
- **[pkgs/pocketfft](file:///usr/local/google/home/sigurdm/projects/math/pkgs/pocketfft)**: Native FFI wrappers around KissFFT mixed-radix discrete Fourier transform plans.

## Development Guidelines

### 1. Packages Resolution
To fetch dependencies and prepare the workspace for all packages at once:
```bash
dart pub get
```

### 2. Code formatting & Analyzer
Ensure formatting and analyzer guidelines pass perfectly before pushing to a pull request:
```bash
dart format .
dart analyze
```

### 3. Executing Unit Tests
To run all tests from the root workspace folder:
```bash
dart test pkgs/*
```

### 4. Generating Coverage Reports
To measure test coverage metrics inside `num_dart`, navigate to `pkgs/num_dart` and run:
```bash
dart tool/generate_coverage.dart
```

## License
This workspace is licensed under the **[Apache License, Version 2.0](file:///usr/local/google/home/sigurdm/projects/math/LICENSE)**.
