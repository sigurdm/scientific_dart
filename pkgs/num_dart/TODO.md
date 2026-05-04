# TODO - NumPy Features for num_dart

This file tracks foundational features missing in `num_dart` relative to NumPy.

## 1. Advanced Indexing and Slicing
- [x] Support rich multi-dimensional slicing syntax (e.g., `arr[1:3, ::2]`) without manual stride calculation.
- [x] Support integer array indexing (via `take` method).
- [x] Support boolean masking (via `applyMask` method).
- [x] Support "Fancy Indexing": indexing with integer arrays (e.g., `arr[[0, 2, 4]]`).
- [x] Integrate boolean masking into standard indexing syntax (e.g., `arr[arr > 0]`).

## 2. Array Creation Utilities
- [x] Add `arange` for range-based creation.
- [x] Add `linspace` for evenly spaced values.
- [x] Add `eye` or `identity` for identity matrices.
- [x] Create a random module for generating arrays with random values.
- [x] Expand random module: `normal` (Gaussian), `poisson`, `binomial`, `exponential` distributions.

## 3. Shape Manipulation
- [x] Add `reshape` to change array shape without copying data.
- [x] Add `transpose` to permute dimensions (swapping strides).
- [x] Add `flatten` / `ravel` to collapse to 1D.
- [x] Add `concatenate`, `vstack`, `hstack` to join arrays.
- [x] Add utilities: `swapaxes`, `moveaxis`, `squeeze`, `expand_dims`, `tile`, `repeat`.

## 4. Universal Functions (ufuncs) & Broadcasting
- [x] Add fast element-wise math functions: `sin`, `cos`, `exp`, `log`, `sqrt`.
- [x] Expand ufuncs: `tan`, `atan2`, `sinh`, `cosh`, `tanh`, `abs`, `ceil`, `floor`, `round`, `clip`.
- [x] Support broadcasting in comparison operators (`>`, `<`, `>=`, `<=`, `eq`).
- [x] Logical operations: `logical_and`, `logical_or`, `logical_not`, `logical_xor`.

## 5. Reductions along Axes
- [x] Add `axis` support to `sum`.
- [x] Add reductions: `mean`, `min`, `max`.
- [x] Add reductions: `prod`, `std`, `var`.

## 5.5 Complex numbers
- [x] Add support for complex numbers.

## 6. Linear Algebra
- [x] Add matrix inversion (`inv`).
- [x] Add determinant (`det`).
- [x] Add solver for linear systems.
- [x] Add eigen decompositions.
- [x] Optimize `inv` using LAPACK (`dgetrf` + `dgetri`) instead of pure Dart.
- [x] Support high-dimensional (ND) broadcasting in `matmul` (stacks of matrices).
- [x] Add SVD (Singular Value Decomposition), QR, and Cholesky decompositions.
- [x] Add FFT (Fast Fourier Transform) implementation via native `pocketfft`.
- [x] Created sibling `package:pocketfft` for automated FFI native assets builds.

## 7. Sorting, Searching & Counting
- [x] Sorting: `sort`, `argsort`, `partition`, `argpartition`.
- [x] Searching: `where`, `nonzero`, `argmax`, `argmin`, `count_nonzero`.

## 8. Interoperability & I/O
- [x] Support `.npy` and `.npz` file formats for compatibility with Python.
- [x] Direct conversion to/from Dart `TypedData` with minimal copying.

## 9. Performance & Infrastructure
- [x] Optimize element-wise ops to iterate on C memory directly (avoid `toList()` copies) - C vector kernels + odometer walks.
- [x] Expand SIMD usage beyond `float32` addition (via compiler autovectorization loops).
- [x] Add `DType.bool` or efficient `uint8` masking for boolean operations (via `DType.boolean` + `BoolList`).
- [x] Create a comprehensive benchmark suite comparing `num_dart` element-wise operations, reductions, and linear algebra performance directly against Python's NumPy.
- [ ] Set up test-coverage measurement, and achieve 100% or close coverage.