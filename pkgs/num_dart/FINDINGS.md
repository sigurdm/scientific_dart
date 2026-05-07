# Codebase Quality & Enhancements Review - FINDINGS.md

This file logs architectural improvements and hidden flaws discovered during autonomous code-review loops.

---

## 🚀 Section 1: Critical Performance Bottlenecks

### 1.1 Reductions & Statistics Bloat
- **Issue**: `variance()` and `std()` (#1) trigger a chain of intermediate element-wise operations (subtract, multiply, cast) leading to **4x memory amplification** and 4x loop passes.
- **Issue**: `binomial()` (#4), `exponential()` (#56), and `poisson()` (#57) for small parameters use raw Dart JIT loops with billions of `rand.nextDouble()` calls, stalling simulations.
- **Issue**: `_countNonzeroRecursive()` (#13) and `nonzero()` (#5) create massive heap allocation churn due to dynamic list growth and recursive coordinate mapping.
- **Recommended Tweak**: Collapse these into unified streaming C kernels in `custom_ufuncs.c`. For statistics, use single-pass algorithms. For random sampling, move the entire loop into AOT C space.

### 1.2 View & Manipulation Overhead
- **Issue**: `flatten()` (#14) and `ravel()` (on strided views) suffer a **double-allocation and double-copy penalty** because they route through `toList()` and `fromList()`.
- **Issue**: `det()` (#59) and `svd()` fallback to slow `toList()` copies for non-contiguous views.
- **Issue**: `operator ==` and `hashCode` (#58) - While `operator ==` is optimized, `hashCode` still loops in Dart space for large arrays.
- **Recommended Tweak**: Implement low-level C FFI flattening/walking kernels. For `hashCode`, use a native rolling hash (e.g., xxHash) acting directly on C memory.

---

## 🛠️ Section 2: Architectural & Memory Safety Gaps

### 2.1 Linear Algebra & LAPACK Integration
- **Issue**: `det()` (#3) lacks ND stack support, violating NumPy conventions for tensor shapes like `[Batch, N, N]`.
- **Issue**: `matmul()` (#27) lacks support for Float32, Complex64, and Complex128 BLAS routines, forcing slow/double-precision conversions.
- **Issue**: Missing standard solvers like `linalg.solve` (#18), `lstsq` (#18), `linalg.norm` (#21), and `linalg.pinv` (#28).
- **Recommended Tweak**: Expand the FFI bridge to expose the full suite of LAPACK routines and refactor high-level methods to handle ND-stack broadcasting.

### 2.2 Memory Management & Safety
- **Issue**: **Test suite memory leakage** (#16). Almost all tests allocate FFI memory without calling `.dispose()`, leading to heap growth during long test runs.
- **Issue**: `NDArray.zeros()` (#7, #22) uses `malloc` + `memset`. Using `calloc` would be more efficient for the OS via demand-paging.
- **Recommended Tweak**: Harden tests with `addTearDown(() => arr.dispose())`. Refactor `NDArray.create` to support `calloc`.

### 2.3 Broadcasting & Advanced Indexing
- **Issue**: `setByMask()` (#47) lacks support for broadcasting multi-dimensional array assignments.
- **Issue**: `concatenate()` (#48) lacks implicit axis expansion (e.g., concatenating a 1D vector to a 2D matrix).
- **Recommended Tweak**: Align broadcasting logic in these methods with NumPy standards.

---

## 🧪 Section 3: NumPy Compatibility Roadmap (Missing Features)

### 3.1 Universal Functions (ufuncs)
- **Math**: `power` (#33), `hypot` (#41), `sign` (#44), `fmod`/`remainder` (#46), `diff` (#39).
- **Trig/Hyperbolic**: `asin`, `acos`, `atan`, `atan2`, `asinh`, `acosh`, `atanh` (#20, #32).
- **High-Precision**: `log1p`, `expm1` (#43).
- **Bitwise**: `bitwise_and`, `bitwise_or`, `bitwise_xor`, `left_shift`, `right_shift` (#54).

### 3.2 Array Manipulation
- **Shaping**: `broadcast_to` (#34), `meshgrid`/`mgrid`/`ogrid` (#52), `asStrided` (#50), `slidingWindowView` (#61).
- **Joining/Splitting**: `dstack`, `column_stack` (#19, #35), `split`, `hsplit`, `vsplit`, `dsplit` (#29).
- **Reorientation**: `flip`, `fliplr`, `flipud`, `rot90`, `roll` (#30).

### 3.3 Statistics & Sorting
- **Reductions**: `nanmin`, `nanmax` (#36), `percentile`, `quantile` (#37), `cumsum`, `cumprod` (#53).
- **Sorting**: `partition`, `argpartition` (#45), `unique` (#31), stable sort support (`kind` parameter) (#49).
- **Searching**: `searchsorted` (#40), `ravel_multi_index`/`unravel_index` (#42).

### 3.4 Random & DType
- **Sampling**: `choice`, `shuffle`, `permutation` (#26), `multinomial` (#60).
- **Types**: Expansion to `uint8` and `int16` (#51) for image/audio processing.

---

## 🏗️ Section 4: DevOps & Build Hazards
- **Issue**: **OpenBLAS compilation latency** (#15). Building from source takes 5-10 minutes. Needs precompiled binary distribution.
- **Issue**: **Windows MSVC breakage** (#23). Hardcoded GCC flags in `pocketfft` build hook prevent Windows compilation.
