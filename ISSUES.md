# Codebase Quality & Enhancements Review - FINDINGS.md

This file logs architectural improvements, optimization ideas, and feature gaps relative to the reference NumPy library.


## 🛠️ Section 2: Architectural & Memory Safety Gaps & Correctness

(Completed integer absolute value and exponentiation FFI optimizations)

---

## 🧪 Section 3: NumPy Compatibility Roadmap (Missing Features)

### 3.1 Universal Functions (ufuncs)
(Completed log2, log10, reciprocal, positive, sinc, and i0 ufuncs)

### 3.2 Array Manipulation & Geometry
(Completed shaping, repeating, rearranging, and splitting features)


### 3.3 Statistics & Sorting
(Completed binning, histograms, covariance, correlation, average, ptp, sorting, partitioning, and searchsorted features)

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
(Completed Schur and Hessenberg decompositions)

### 3.20 Broad-Boundary Array Clipping (clip)
(Completed native broadcasting array bounds clipping feature)

### 3.21 Advanced Linear Algebra (`linalg`)
(Completed eigh, eigvalsh, eigvals, and slogdet linear algebra functions)

### 3.22 Einstein Summation (`einsum`) & Tensor Contractions
- (Completed einsum and tensordot; kron remaining)

### 3.23 Polynomial Module
- (Completed polyfit, polyval, roots, and orthogonal series evaluations)

### 3.24 Advanced Indexing Operations
- (Completed take_along_axis, put_along_axis, select, and choose)

### 3.25 N-Dimensional Convolutions & Spatial Filtering
- (Completed convolve2d, correlate, and N-D stencil/FFT convolution)

### 3.26 Scientific Optimization & Root Finding
- (Completed 1D root finders brentq/newton/secant and multivariate minimizers Nelder-Mead/L-BFGS)

### 3.27 Expanded Data Type (DType) Support
- **Feature**: Add missing integer types (`int8`, `uint16`, `uint32`, `uint64`), floating-point types (`float16`, `bfloat16`), fixed-width strings/bytes (`string_`), datetime types (`datetime64`, `timedelta64`), and structured / record arrays (`recarray`).
- **Details**: Extend core C FFI and Dart wrappers to support fixed-width string buffers, time unit encodings, and heterogeneous structured record layouts.

### 3.28 Ellipsis Indexing (`...`) & Multi-Array Indexing
- **Feature**: Support Ellipsis slicing across arbitrary dimensions dynamically, and implement full N-dimensional index array broadcasting ("fancy indexing").
- **Details**: Expand dimension resolution during indexing to interpret `...` across non-explicit axes and compute broadcasted coordinate indexing offsets.

### 3.29 Universal Function (`ufunc`) Capabilities
- (Completed where= masking parameter, reduce, accumulate, reduceat, outer, and at ufunc methods)

### 3.30 Modern BitGenerator Random Generator API
- **Feature**: Implement a modern `Generator` API powered by BitGenerators (PCG64, Philox, SFC64) to replace legacy `dart:math Random` distribution sampling.
- **Details**: Provide high-performance BitGenerator bitstream engines and statistical distribution samplers (Normal, Gamma, Beta, etc.) matching `numpy.random.Generator`.

### 3.31 Extended Matrix Functions & Tensor Solvers (`linalg`)
- **Feature**: Implement matrix exponential/logarithm/square-root (`expm`, `logm`, `sqrtm`), generalized matrix ufuncs (`gufuncs`), and tensor solvers (`tensorsolve`, `tensorinv`).
- **Details**: Interface with LAPACK/OpenBLAS routines for matrix functions, implement tensor inverse/solver decompositions, and provide GUFunc signature dispatching.

### 3.32 Multi-Dimensional Real & Hermite FFT Routines (`fft`)
- **Feature**: Implement 2D/N-D real FFTs (`rfft2`, `irfft2`, `rfftn`, `irfftn`) and Hermite FFTs (`hfft`, `ihfft`).
- **Details**: Extend 1D real FFT algorithms to N-dimensional real-to-complex and Hermite-symmetric array transformations along arbitrary axis subsets.

### 3.33 Object-Oriented Polynomial API
- **Feature**: Implement OO polynomial classes (`Polynomial`, `Chebyshev`, `Legendre`, `Hermite`) with `.fit()` routines and domain mapping.
- **Details**: Provide class hierarchy for orthogonal polynomial series, series arithmetic, domain/window mapping, and least-squares fitting against sample points.

---

## ✨ Section 4: Usability & Ergonomics

### 4.1 operator []= Selector Expansion
- **Issue**: `operator []=` is currently constrained and can be expanded to handle more complex NumPy-style selection objects (e.g. mixed lists and index tuples).

---

## 🏗️ Section 5: DevOps & Build Hazards
(Completed resolving OpenBLAS compilation latency on Windows and Windows MSVC compilation, linking, and test runtime hazards)
