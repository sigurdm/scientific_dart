# Codebase Quality & Enhancements Review - FINDINGS.md

This file logs architectural improvements, optimization ideas, and feature gaps relative to the reference NumPy library, discovered during autonomous review loops.

---

## 🚀 Section 1: Critical Performance Bottlenecks

### 1.1 Reductions & Statistics Bloat
- **Issue**: `binomial()`, `exponential()`, and `poisson()` for small parameters use raw Dart JIT loops with billions of `rand.nextDouble()` calls, stalling simulations.
- **Recommended Tweak**: Collapse these into unified streaming C kernels in `custom_ufuncs.c`. For statistics, use single-pass algorithms. For random sampling, move the entire loop into AOT C space.

---

## 🛠️ Section 2: Architectural & Memory Safety Gaps

### 2.1 Linear Algebra & LAPACK Integration
- **Issue**: `det()`, `eig()`, `qr()`, and `svd()` lack ND-stack support, violating NumPy conventions for tensor shapes like `[Batch, N, N]`.
- **Issue**: `matmul()` lacks support for Complex64 (`cblas_cgemm`) and Complex128 (`cblas_zgemm`) BLAS routines. While Float32 (`cblas_sgemm`) and Float64 (`cblas_dgemm`) are bound and FFI-offloaded, complex matrix multiplications are still falling back to slower iterative loops.
- **Issue**: Missing premium solvers like pseudo-inverse `linalg.pinv`, least-squares solver `linalg.lstsq`, matrix power `linalg.matrix_power`, and optimal matrix multiplication chain order optimizer `linalg.multi_dot`.
- **Recommended Tweak**: Expose `cblas_cgemm` and `cblas_zgemm` via the FFI bridge. Refactor linear algebra algorithms to handle ND-stack broadcasting. Add advanced LAPACK solver bindings for pseudo-inverses and least-squares equations.

### 2.2 Advanced Indexing & Iteration Structures
- **Issue**: The recursive advanced indexing walker (`_copyAdvancedRecursive`) is functional but further C-level optimization for extreme ranks is an option.
- **Issue**: Missing structured multi-dimensional iteration utilities equivalent to NumPy's `nditer` and `ndenumerate`. Currently, hot loops are forced to nested dimension walks or explicit flat-index strided conversions, causing VM boundary crossing overhead and list allocations.
- **Recommended Tweak**: Offload advanced indexing walks to a native C odometer kernel. Design a zero-allocation `nditer`-like class in Dart that yields strided coordinate buffers directly to consumer math closures.

---

## 🧪 Section 3: NumPy Compatibility Roadmap (Missing Features)

### 3.1 Universal Functions (ufuncs)
- **Math & Trigonometry**:
  - `diff` (difference calculation along a given axis).
  - Trigonometric functions: `tan()`, inverse trig `asin()`, `acos()`, `atan()`, and `atan2(y, x)`.
  - Hyperbolic functions: `sinh()`, `cosh()`, `tanh()`, `asinh()`, `acosh()`, `atanh()`.
  - Power & logarithmic: `square()`, element-wise `power()`, modulo `remainder()` / `mod()`, and combined division/modulo `divmod()`.
  - Floating-point classification: `isnan()`, `isinf()`, `isfinite()`, sign copier `copysign()`.
- **Logical Operations (Vectorized)**:
  - `logical_and()`, `logical_or()`, `logical_not()`, `logical_xor()`.
- **Bitwise Operations**:
  - Vectorized bitwise ufuncs for integer data types (`int32`, `int64`, `uint8`, `int16`): `bitwise_and()`, `bitwise_or()`, `bitwise_xor()`, bitwise negation `invert()`, shifts `left_shift()` and `right_shift()`.
- **Fourier Transforms**:
  - Multi-dimensional `axis` support inside `fft()` and `ifft()`. Currently, our FFT transforms are hardcoded to execute along the final axis (`a.shape.last`). Adding an `axis` parameter (default `-1`) and transposing dimensions internally before/after FFI plan runs would achieve full standard NumPy `np.fft.fft(a, axis=axis)` compatibility!
  - Spectrogram shifts: `fftshift()` and `ifftshift()`.

### 3.2 Array Manipulation & Geometry
- **Shaping & Meshes**: `mgrid`/`ogrid`, `asStrided`, `slidingWindowView`.
- **Repeating & Tiling**: Vector repeat `repeat()` and grid tiling `tile()`.
- **Rearranging**: Axis roll `roll()`, flips `flip()`, `fliplr()`, and `flipud()`.
- **Splitting**: Block splitting `split()`, `array_split()`, `hsplit()`, and `vsplit()`.
- **Triangular Masks**: Lower triangular matrix extraction `tril()` and upper triangular matrix extraction `triu()` (extremely vital for causal mask scaling inside LLM / attention block layers).

### 3.3 Statistics & Sorting
- **Sorting**: Partial sorting `partition()` and index partial sorting `argpartition()` (extremely high performance benefit for top-K filtering), stable sorting indicator `kind` parameter inside `sort()`.
- **Searching**: Binary search insertion `searchsorted()`.

### 3.4 Random & DType
- **Sampling**: `choice()` (random selection from an array), `shuffle()` (in-place shuffling), and `permutation()`.
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

### 3.6 Calculus & Cumulative Accumulations
- **Cumulative ufuncs**: `cumsum(a, {int? axis})` (cumulative sum) and `cumprod(a, {int? axis})` (cumulative product) to enable robust DSP time-series integrals calculations.
- **Calculus Solvers**: N-Dimensional gradients `gradient(f)` and trapezoidal integrals solver `trapz(y, x)`.

### 3.7 Vectorized Logical Reductions
- **Axis-wise logical reductions**: `all(a, {int? axis})` (check if all elements along axis evaluate to true) and `any(a, {int? axis})` (check if any element along axis evaluates to true) over numeric or boolean tensor structures.

### 3.8 Tabular Tabular Record Arrays (Heterogeneous Structured Data)
- **Structured Fields Support**: Exposing composite data types matching standard C structs (equivalent to NumPy's `np.recarray` and structured dtype schemas), allowing heterogeneous fields elements to be packaged and walked sequentially inside unmanaged heap bytes segments.

### 3.9 Progressive Scientific Generators
- **Geometric Progressive Spacers**: Exposing logarithmic spacer `logspace()` and geometric spacer `geomspace()` generators to match high-end scientific computing specifications.

---

## ✨ Section 4: Usability & Ergonomics

### 4.1 operator []= Selector Expansion
- **Issue**: `operator []=` is currently constrained and can be expanded to handle more complex NumPy-style selection objects (e.g. mixed lists).

---

## 🏗️ Section 5: DevOps & Build Hazards
- **Issue**: **OpenBLAS compilation latency**. Building from source takes 5-10 minutes. Needs precompiled binary distribution.
- **Issue**: **Windows MSVC breakage**. Hardcoded GCC flags in `pocketfft` build hook prevent Windows compilation.
