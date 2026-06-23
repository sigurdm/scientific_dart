# Codebase Quality & Enhancements Review - FINDINGS.md

This file logs architectural improvements, optimization ideas, and feature gaps relative to the reference NumPy library.


## 🛠️ Section 2: Architectural & Memory Safety Gaps & Correctness

(Completed integer absolute value and exponentiation FFI optimizations)

---

## 🧪 Section 3: NumPy Compatibility Roadmap (Missing Features)

### 3.1 Universal Functions (ufuncs)
- **Special Functions**:
  - `i0()` (modified Bessel function of the first kind, order 0) for filter windowing.
(Completed log2, log10, reciprocal, positive, and sinc ufuncs)

### 3.2 Array Manipulation & Geometry
(Completed shaping, repeating, rearranging, and splitting features)


### 3.3 Statistics & Sorting
- **Binning & Histograms**: `histogram()` (1D/2D/ND binning), `bincount()` (integer occurrences), and `digitize()` (bin indices assignment).
- **Correlation & Covariance**: `cov()` (covariance matrix estimation) and `corrcoef()` (Pearson correlation coefficients).
(Completed average, ptp, sorting, partitioning, and searchsorted features)

### 3.4 Random & DType
(Completed sampling choice, shuffle, permutation, and media types expansion)

### 3.6 Calculus & Cumulative Accumulations
- all done

### 3.7 Vectorized Logical Reductions
- all done

### 3.9 Progressive Scientific Generators
- all done

### 3.14 Structured Masked Arrays (ma category)
- (Completed in a separate package)

### 3.16 Schur and Hessenberg linalg Decompositions
- **Advanced Control Theory Solvers**: Exposing native LAPACK-bound solvers `linalg.schur()` and `linalg.hessenberg()` to support advanced control systems design and numeric eigenvalue search algorithms.

### 3.20 Broad-Boundary Array Clipping (clip)
(Completed native broadcasting array bounds clipping feature)

### 3.21 Advanced Linear Algebra (`linalg`)
- **Symmetric/Hermitian Eigenvalue Solvers**: `eigh()` and `eigvalsh()` for faster, more stable eigenvalue/vector computation of symmetric/complex Hermitian matrices.
- **Fast Eigenvalues**: `eigvals()` to compute only eigenvalues (without eigenvectors) much faster than `eig()`.
- **Log-Determinant**: `slogdet()` (sign and log-determinant) to prevent numerical underflow/overflow on large matrices.

---

## ✨ Section 4: Usability & Ergonomics

### 4.1 operator []= Selector Expansion
- **Issue**: `operator []=` is currently constrained and can be expanded to handle more complex NumPy-style selection objects (e.g. mixed lists).

---

## 🏗️ Section 5: DevOps & Build Hazards
- **Issue**: **OpenBLAS compilation latency**. Building from source takes 5-10 minutes. Needs precompiled binary distribution.
- **Issue**: **Windows MSVC breakage**. Hardcoded GCC flags in `pocketfft` build hook prevent Windows compilation.
