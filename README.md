# Math Workspace Monorepo

High-performance numerical, scientific, GPU-accelerated, and symbolic computing packages for Dart.

## Repository Structure

This repository is structured as a Dart workspace monorepo configured under [`pubspec.yaml`](pubspec.yaml):

### Core Scientific & Numerical Packages

- **[`pkgs/ndarray`](pkgs/ndarray)**: The core scientific N-dimensional array and tensor computing library (`NDArray`). Features native C/C++17 & Google Highway SIMD kernels, OpenBLAS/LAPACK linear algebra (`matmul`, `svd`, `qr`, `cholesky`, `eigh`/`eig`, `lstsq`, `pinv`, `solve`), Einstein summation (`einsum`) & `tensordot`, Google Highway SIMD sorting (`sort`, `argsort`, `partition`, `argpartition`), spectral transforms & DSP (`fft`, `rfft`, `convolve`, `correlate`, windows), numerical optimization & root finding (`minimize`, `root_scalar`), spatial distance metrics (`pdist`, `cdist`), orthogonal polynomials, zero-copy NumPy `.npy`/`.npz` streaming I/O, zero-copy strided views, and deterministic scoped memory management (`NDArray.scope`).
- **[`pkgs/gpuarray`](pkgs/gpuarray)**: GPU-accelerated N-dimensional array computing with compute shaders and seamless zero-copy/streaming interoperability with `NDArray`.
- **[`pkgs/ndarray_ma`](pkgs/ndarray_ma)**: Masked arrays (`MaskedArray`) for `ndarray`, enabling element-wise operations, reductions, and statistical analysis over datasets with missing, invalid, or masked entries.
- **[`pkgs/resource_scope`](pkgs/resource_scope)**: Zone-based automatic scoped resource management (`ResourceScope`, `NDArray.scope`) for deterministic FFI and native C-heap memory disposal in Dart.
- **[`pkgs/symbolic_dart`](pkgs/symbolic_dart)**: Symbolic mathematics and Computer Algebra System (CAS) library for Dart powered by native C/C++ bindings (SymEngine & FLINT), with symbolic-to-numerical `ndarray` evaluation.

### Native Backend Bindings

- **[`pkgs/openblas`](pkgs/openblas)**: Low-level FFI bindings wrapping OpenBLAS CBLAS and LAPACK headers, complete with automated Dart Native Assets compile hooks that build OpenBLAS across Linux, macOS, and Windows.
- **[`pkgs/pocketfft`](pkgs/pocketfft)**: Native AOT FFI bindings around PocketFFT/KissFFT mixed-radix discrete Fourier transform plans for fast 1D and multidimensional complex and real FFTs.

### Applications & Tooling

- **[`pkgs/notebook`](pkgs/notebook)**: Interactive notebook and REPL interface for Dart, `ndarray`, and `symbolic_dart`.
- **[`pkgs/code_editor`](pkgs/code_editor)**: Code editor component supporting syntax highlighting and interactive evaluation for the notebook environment.
- **[`pkgs/guitar_tuner`](pkgs/guitar_tuner)**: Real-time CLI guitar tuner demonstrating live audio capture (ALSA) and spectral pitch detection using `ndarray`.

---

## Development Guidelines

### 1. Environment & SDK Setup
Ensure you are using a Dart SDK supporting Dart Workspaces and Native Assets (`^3.10.0`, or `master` channel for workspace development). Fetch dependencies across all workspace packages at once from the repository root:
```bash
dart pub get
```

### 2. Code Formatting & Static Analysis
Ensure formatting and static analysis pass with zero warnings or errors before submitting changes:
```bash
dart format .
dart analyze
```

### 3. Executing Unit Tests
To run unit tests across all packages from the workspace root:
```bash
dart test
```
Or target a specific package (e.g., `pkgs/ndarray`):
```bash
dart test pkgs/ndarray
```

### 4. Generating Coverage Reports
To measure test coverage metrics inside `ndarray`, navigate to `pkgs/ndarray` and run:
```bash
dart tool/generate_coverage.dart
```

---

## License

This workspace is licensed under the **[Apache License, Version 2.0](LICENSE)**.

## Disclaimer

This is not an official Google product.
