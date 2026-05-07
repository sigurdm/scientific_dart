# Codebase Quality & Enhancements Review - FINDINGS.md

This file logs architectural improvements and hidden flaws discovered during autonomous code-review loops.

---

## 🚀 Section 1: Critical Performance Bottlenecks

### 1.1 Reductions & Statistics Bloat
- **Issue**: `variance()` and `std()` trigger a chain of intermediate element-wise operations (subtract, multiply, cast) leading to **4x memory amplification** and 4x loop passes.
- **Issue**: `binomial()`, `exponential()`, and `poisson()` for small parameters use raw Dart JIT loops with billions of `rand.nextDouble()` calls, stalling simulations.
- **Issue**: `_countNonzeroRecursive()` and `nonzero()` create massive heap allocation churn due to dynamic list growth and recursive coordinate mapping.
- **Recommended Tweak**: Collapse these into unified streaming C kernels in `custom_ufuncs.c`. For statistics, use single-pass algorithms. For random sampling, move the entire loop into AOT C space.

### 1.2 View & Manipulation Overhead
- **Issue**: `flatten()` and `ravel()` (on strided views) suffer a **double-allocation and double-copy penalty** because they route through `toList()` and `fromList()`.
- **Issue**: `det()`, `qr()`, and `svd()` fallback to slow `toList()` copies for non-contiguous views and often use redundant intermediate `List.from(a.data)` allocations.
- **Issue**: `operator ==` and `hashCode` - While `operator ==` is optimized, `hashCode` still loops in Dart space for large arrays.
- **Recommended Tweak**: Implement low-level C FFI flattening/walking kernels. For `hashCode`, use a native rolling hash acting directly on C memory.

---

## 🛠️ Section 2: Architectural & Memory Safety Gaps

### 2.1 Linear Algebra & LAPACK Integration
- **Issue**: `det()`, `eig()`, `qr()`, and `svd()` lack ND stack support, violating NumPy conventions for tensor shapes like `[Batch, N, N]`.
- **Issue**: `matmul()` lacks support for Float32, Complex64, and Complex128 BLAS routines, forcing slow/double-precision conversions.
- **Issue**: Missing standard solvers like `linalg.solve`, `linalg.lstsq`, `linalg.norm`, and `linalg.pinv`.
- **Recommended Tweak**: Expand the FFI bridge to expose the full suite of LAPACK routines and refactor high-level methods to handle ND-stack broadcasting.

### 2.2 Memory Management & Safety
- **Issue**: **Test suite memory leakage**. Almost all tests allocate FFI memory without calling `.dispose()`, leading to heap growth during long test runs.
- **Issue**: **TypedData buffer offset safety in Serialization**. Inside `io.dart`, `.buffer.asUint8List()` is called directly on temporary TypedData lists during serialization without specifying `offsetInBytes` and `lengthInBytes`, posing a future regression hazard if Dart's internal TypedData buffer allocations or views align differently.
- **Recommended Tweak**: Harden tests with `addTearDown(() => arr.dispose())`. Update `io.dart` serialization code to use `.buffer.asUint8List(list.offsetInBytes, list.lengthInBytes)` for absolute safety.

### 2.3 Broadcasting & Advanced Indexing
- **Issue**: `setByMask()` lacks support for broadcasting multi-dimensional array assignments.
- **Issue**: `concatenate()` lacks implicit axis expansion (e.g., concatenating a 1D vector to a 2D matrix).
- **Issue**: Advanced indexing (e.g., `a[[0, 1], [0, 1]]`) does not support broadcasting multiple index arrays against each other to select arbitrary coordinate sets.
- **Issue**: The recursive advanced indexing walker (`_copyAdvancedRecursive`) is slow and prone to stack depth limits on high-rank tensors.
- **Recommended Tweak**: Align broadcasting and indexing logic with NumPy standards. Offload advanced indexing walks to a native C odometer kernel.

---

## 🧪 Section 3: NumPy Compatibility Roadmap (Missing Features)

### 3.1 Universal Functions (ufuncs)
- **Math**: `power`, `hypot`, `sign`, `fmod`/`remainder`, `diff`.
- **Trig/Hyperbolic**: `asin`, `acos`, `atan`, `atan2`, `asinh`, `acosh`, `atanh`.
- **High-Precision**: `log1p`, `expm1`.
- **Bitwise**: `bitwise_and`, `bitwise_or`, `bitwise_xor`, `left_shift`, `right_shift`.
  - **bitwise detail**: We can implement `bitwise_and(NDArray a, NDArray b)`, `bitwise_or()`, and `bitwise_xor()` for integer dtypes (`int32` and `int64`). These will walk strides recursively and apply operators: `a.data[offsetA] & b.data[offsetB]`, `|`, and `^`. It will throw an `UnsupportedError` if called on float or complex dtypes, matching standard NumPy behaviour exactly!

### 3.2 Array Manipulation
- **Shaping**: `broadcast_to`, `meshgrid`/`mgrid`/`ogrid`, `asStrided`, `slidingWindowView`.
- **Joining/Splitting**: `dstack`, `column_stack`, `split`, `hsplit`, `vsplit`, `dsplit`.
- **Reorientation**: `flip`, `fliplr`, `flipud`, `rot90`, `roll`.

### 3.3 Statistics & Sorting
- **Reductions**: `nanmin`, `nanmax`, `percentile`, `quantile`, `cumsum`, `cumprod`.
  - **percentile/quantile detail**: We can implement `percentile(NDArray a, double q, {int? axis})` and `quantile(NDArray a, double q, {int? axis})` by sorting slices using our high-speed Timsort solver. We can then apply linear interpolation:
    $idx = q \times (N - 1)$ (for quantile, $q \in [0, 1]$)
    $low = \lfloor idx \rfloor$, $high = \lceil idx \rceil$
    $val = arr[low] + (idx - low) \times (arr[high] - arr[low])$.
    This achieves 100% standard NumPy parity and is highly optimized!
- **Sorting**: `partition`, `argpartition`, `unique`, stable sort support (`kind` parameter).
- **Searching**: `searchsorted`, `ravel_multi_index`/`unravel_index`.

### 3.4 Random & DType
- **Sampling**: `choice`, `shuffle`, `permutation`, `multinomial`.
- **Types**: Expansion to `uint8` and `int16` for image/audio processing.
- **Consistency**: Reduction results (e.g., `variance`) sometimes force `float64` even for `float32` inputs, causing precision/DType inconsistency.

---

## ✨ Section 4: Usability & Ergonomics (Resolved)
- **Status**: **RESOLVED**. Implemented full operator overloading for arithmetic (+, -, *, /, ~/, %), bitwise (&, |, ^, ~, <<, >>), and unary (-) operations. These wrap existing and new ufuncs with full broadcasting support for both array-array and array-scalar operands.
- **Issue**: `operator []` does not support `Slice` or `Indices` objects directly, and lacks equivalents for `NewAxis` or `Ellipsis`.
- **Recommended Tweak**: Expand `operator []` and `operator []=` to handle complex NumPy-style selection objects.

---

## 🏗️ Section 5: DevOps & Build Hazards
- **Issue**: **OpenBLAS compilation latency**. Building from source takes 5-10 minutes. Needs precompiled binary distribution.
- **Issue**: **Windows MSVC breakage**. Hardcoded GCC flags in `pocketfft` build hook prevent Windows compilation.
