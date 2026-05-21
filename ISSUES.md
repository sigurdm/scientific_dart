# Codebase Quality & Enhancements Review - FINDINGS.md

This file logs architectural improvements, optimization ideas, and feature gaps relative to the reference NumPy library.
When a task is completed, describe it in DONE.md file and remove it from this file.


## 🛠️ Section 2: Architectural & Memory Safety Gaps

(No outstanding architectural issues)

---

## 🧪 Section 3: NumPy Compatibility Roadmap (Missing Features)

### 3.1 Universal Functions (ufuncs)

### 3.2 Array Manipulation & Geometry
(Completed shaping, repeating, rearranging, and splitting features)


### 3.3 Statistics & Sorting
(Completed sorting, partitioning, and searchsorted features)

### 3.4 Random & DType
- **Sampling**: `choice()` (random selection from an array), `shuffle()` (in-place shuffling), and `permutation()`.
- **Types**: Expansion to `uint8` and `int16` for image/audio processing.

### 3.5 Advanced Linear Algebra & Vector Calculus (Roadmap)
- **Tensors & Matrices**:
  - `kron(NDArray a, NDArray b)`: Kronecker product of two arrays.
- **Vector Calculus**:
  - `cross(NDArray a, NDArray b)`: Vector cross product in 3D space.
  - `outer(NDArray a, NDArray b)`: Vector outer product.
- **Solvers & Norms**:
  - `norm(NDArray a, {dynamic ord, int? axis})`: Calculate vector/matrix norms. Expose standard L1, L2 (Frobenius), and Chebyshev infinity norms, supporting Axis-wise reductions cleanly!

### 3.6 Calculus & Cumulative Accumulations
- **Cumulative ufuncs**: (cumsum() and cumprod() ufuncs have been fully resolved!)
- **Calculus Solvers**: N-Dimensional gradients `gradient(f)` and trapezoidal integrals solver `trapz(y, x)`.

### 3.7 Vectorized Logical Reductions
- **Axis-wise logical reductions**: `all(a, {int? axis})` (check if all elements along axis evaluate to true) and `any(a, {int? axis})` (check if any element along axis evaluate to true) over numeric or boolean tensor structures.

### 3.9 Progressive Scientific Generators
- **Geometric Progressive Spacers**: Exposing logarithmic spacer `logspace()` and geometric spacer `geomspace()` generators to match high-end scientific computing specifications.

### 3.11 1D Set Operations
- **Categorical Elements Matching**: Vectorized 1D set operations equivalent to standard Python sets: `intersect1d()`, `setdiff1d()`, `setxor1d()`, `union1d()`, and element check `isin()` to accelerate classification preprocessing filters.

### 3.12 Statistical Percentiles & Medians
- **Descriptive Statistics**: Exposing robust Descriptive statistics solvers: `median(a, {int? axis})` (median calculation), `percentile(a, q, {int? axis})`, and `quantile(a, q, {int? axis})` along axes to compute IQR, outliers, and normalization parameters natively.

### 3.13 1D Linear Interpolation
- **Data Alignment Solver**: Exposing piece-wise linear interpolation solver `interp(x, xp, fp)` to align uneven scientific timeseries records natively.

### 3.14 Structured Masked Arrays (ma category)
- **Robust Missing Data Handling**: Exposing masked array wrappers (similar to standard `numpy.ma` package) to dynamically package arrays with boolean masks, allowing ufuncs and reductions to automatically bypass invalid/corrupted records natively.

### 3.15 Quantitative Financial ufuncs
- **Corporate & Quantitative Solvers**: Exposing high-speed financial universal functions such as Net Present Value `npv()`, Internal Rate of Return `irr()`, Future Value `fv()`, and Present Value `pv()`.

### 3.16 Schur and Hessenberg linalg Decompositions
- **Advanced Control Theory Solvers**: Exposing native LAPACK-bound solvers `linalg.schur()` and `linalg.hessenberg()` to support advanced control systems design and numeric eigenvalue search algorithms.

### 3.17 Axis-Specific Matrix Splitting
- **Standard Multi-Dimensional Segmentation**: Adding axis-specific sub-array splitting helpers `hsplit()`, `vsplit()`, and `dsplit()` to align with standard NumPy dataset segmentations.

### 3.18 N-Dimensional Padding (pad category)
- **Edge Border Manipulations**: Exposing high-performance vectorized N-Dimensional padding `pad(a, pad_width, {mode: 'constant', constant_values: 0})` to support edge reflection, border replication, and zero-padding, which are extremely vital for Convolutional Neural Network (CNN) boundary gates, signal windowing, and numeric differential equation boundaries.

### 3.20 Broad-Boundary Array Clipping (clip)
- **Broadcasting min/max bounds**: Expanding `clip(a, min, max)` to support **array bounds** that broadcast natively against the input array, allowing spatial envelope capping instead of rigid scalar limits.

---

## ✨ Section 4: Usability & Ergonomics

### 4.1 operator []= Selector Expansion
- **Issue**: `operator []=` is currently constrained and can be expanded to handle more complex NumPy-style selection objects (e.g. mixed lists).

---

## 🏗️ Section 5: DevOps & Build Hazards
- **Issue**: **OpenBLAS compilation latency**. Building from source takes 5-10 minutes. Needs precompiled binary distribution.
- **Issue**: **Windows MSVC breakage**. Hardcoded GCC flags in `pocketfft` build hook prevent Windows compilation.
