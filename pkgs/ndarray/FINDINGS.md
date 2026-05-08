# Codebase Quality & Enhancements Review - FINDINGS.md

This file logs architectural improvements and hidden flaws discovered during autonomous code-review loops.

---

## 🚀 Section 1: Critical Performance Bottlenecks

### 1.1 Reductions & Statistics Bloat
- **Issue**: `binomial()`, `exponential()`, and `poisson()` for small parameters use raw Dart JIT loops with billions of `rand.nextDouble()` calls, stalling simulations.
- **Recommended Tweak**: Collapse these into unified streaming C kernels in `custom_ufuncs.c`. For statistics, use single-pass algorithms. For random sampling, move the entire loop into AOT C space.

---

## 🛠️ Section 2: Architectural & Memory Safety Gaps

### 2.1 Linear Algebra & LAPACK Integration
- **Issue**: `det()`, `eig()`, `qr()`, and `svd()` lack ND stack support, violating NumPy conventions for tensor shapes like `[Batch, N, N]`.
- **Issue**: `matmul()` lacks support for Float32, Complex64, and Complex128 BLAS routines, forcing slow/double-precision conversions.
- **Issue**: Missing standard solvers like `linalg.solve`, `linalg.lstsq`, `linalg.norm`, and `linalg.pinv`.
- **Recommended Tweak**: Expand the FFI bridge to expose the full suite of LAPACK routines and refactor high-level methods to handle ND-stack broadcasting.

### 2.2 Advanced Indexing
- **Issue**: The recursive advanced indexing walker (`_copyAdvancedRecursive`) is functional but further C-level optimization for extreme ranks is an option.
- **Recommended Tweak**: Offload advanced indexing walks to a native C odometer kernel.

---

## 🧪 Section 3: NumPy Compatibility Roadmap (Missing Features)

### 3.1 Universal Functions (ufuncs)
- **Math**: `diff` (still pending).
- **Fourier Transforms**:
  - Multi-dimensional `axis` support inside `fft()` and `ifft()`. Currently, our FFT transforms are hardcoded to execute along the final axis (`a.shape.last`). Adding an `axis` parameter (default `-1`) and transposing dimensions internally before/after FFI plan runs would achieve full standard NumPy `np.fft.fft(a, axis=axis)` compatibility!

### 3.2 Array Manipulation
- **Shaping**: `mgrid`/`ogrid`, `asStrided`, `slidingWindowView`.

### 3.3 Statistics & Sorting
- **Sorting**: `partition`, `argpartition`, `unique`, stable sort support (`kind` parameter).
- **Searching**: `searchsorted`, `ravel_multi_index`/`unravel_index`.

### 3.4 Random & DType
- **Sampling**: `choice`, `shuffle`, `permutation`, `multinomial`.
- **Types**: Expansion to `uint8` and `int16` for image/audio processing.

### 3.5 Advanced Linear Algebra & Vector Calculus (Roadmap)
- **Tensors & Matrices**:
  - `matrix_power(NDArray a, int n)`: Raise a square 2D matrix to integer power `n` using binary exponentiation.
  - `kron(NDArray a, NDArray b)`: Kronecker product of two arrays.
- **Vector Calculus**:
  - `cross(NDArray a, NDArray b)`: Vector cross product in 3D space.
  - `outer(NDArray a, NDArray b)`: Vector outer product.
- **Solvers & Norms**:
  - `norm(NDArray a, {dynamic ord, int? axis})`: Calculate vector/matrix norms. Expose standard L1, L2 (Frobenius), and Chebyshev infinity norms, supporting Axis-wise reductions cleanly!

---

## ✨ Section 4: Usability & Ergonomics
- **Recommended Tweak**: Expand `operator []=` to handle more complex NumPy-style selection objects (e.g. mixed lists).

---

## 🏗️ Section 5: DevOps & Build Hazards
- **Issue**: **OpenBLAS compilation latency**. Building from source takes 5-10 minutes. Needs precompiled binary distribution.
- **Issue**: **Windows MSVC breakage**. Hardcoded GCC flags in `pocketfft` build hook prevent Windows compilation.
