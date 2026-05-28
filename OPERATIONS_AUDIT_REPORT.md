# Ndarray Package Operations Suite Audit Report

**Date:** 2026-05-28  
**Auditors:** Antigravity (Lead AI Coding Assistant) & 14 Specialized Reviewer Subagents  
**Target Scope:** All 14 modular standalone operational libraries under `lib/src/operations/`  
**Status:** Complete (All audits completed concurrently)  

---

## 1. Executive Summary

This audit presents a comprehensive quality, performance, memory, and correctness review of the restructured `ndarray` operations suite. By running 14 concurrent subagent audits across every single operational library, we have identified **7 critical, previously hidden correctness and stability bugs**, along with systematic performance bottlenecks, memory allocation violations, and type-safety gaps.

This report groups all findings into clear, actionable remediation categories to prepare for structured, iterative fixing.

---

## 2. Critical Correctness & Stability Bugs (Remediated in Phase 1)

We uncovered 7 critical bugs that cause runtime crashes, silent data corruption, or incorrect math results. These have all been successfully fixed and verified in **Phase 1**.

### 2.1 Fallback Path Data Corruption & `RangeError` (Math)
*   **Location:** `math.dart` (Applies to `sinh`, `cosh`, `tanh`, `sqrt`, `square`, `exp`, `log`, etc.)
*   **Root Cause:** When a math operation was invoked on a non-contiguous sliced view (where `isContiguous` is `false`), it fell back to a pure Dart loop. Instead of walking the logical elements via strides, the loop indexed sequentially using physical indices `a.data[i]`, reading incorrect memory offsets or writing past the bounds of the result array throwing `RangeError`.
*   **Fix:** Refactored all fallback loops to use the generic strided `unaryOp` walker instead of sequential physical loops. Verified via `test/sinh_bug_repro_test.dart`.

### 2.2 Silent Zeros in Integer `matmul` (Linalg)
*   **Location:** `linalg.dart` (at the FFI matrix multiplication walker)
*   **Root Cause:** In the FFI matrix multiplication walker, there was absolutely no implementation for integer `targetDType` (e.g., `DType.int32`, `DType.int64`), meaning it silently returned a zero-filled matrix.
*   **Fix:** Added an optimized JIT-speed integer matrix multiplication fallback to `matmul`'s FFI closure. Verified via `test/matmul_bug_repro_test.dart`.

### 2.3 Integer `eig` TypeError Crash (Linalg)
*   **Location:** `linalg.dart` (eigenvalue solver `eig`)
*   **Root Cause:** Calling `eig` on integer matrices caused a runtime `TypeError` because the code attempted to cast raw `Int32List`/`Int64List` directly to `Float64List` for packing.
*   **Fix:** Refactored `eig`'s internal slice buffer to allocate double-precision `Float64` buffers for all integer matrices, upcasting them safely.

### 2.4 Empty 1D Array Sorting Crash (Sorting)
*   **Location:** `sorting.dart` (in `sort` and `argsort`)
*   **Root Cause:** Attempting to sort an empty 1D array (shape `[0]`) caused a crash with `IntegerDivisionByZeroException` due to an unchecked `0 ~/ 0` in the pivot/partition calculations.
*   **Fix:** Added upfront `a.size == 0` guards in both `sort` and `argsort` to return immediately without executing partition calculations. Verified via `test/empty_sort_bug_repro_test.dart`.

### 2.5 Boolean `argmax`/`argmin` TypeError Crash (Sorting)
*   **Location:** `sorting.dart` (in `argmax` and `argmin`)
*   **Root Cause:** The fallback `else` block assumed the array has type `double` and cast the data directly to `List<double>`, throwing `_TypeError` when called on `boolean` arrays.
*   **Fix:** Added explicit support for `boolean` DTypes in both flat loops and the recursive multidimensional reduction walker `argMinMaxRecursive` in `helpers.dart` (mapping `bool` to `0`/`1` for numeric comparison). Verified via `test/boolean_argminmax_bug_repro_test.dart`.

### 2.6 Boolean `cumsum` Reductions Crash (Stats)
*   **Location:** `stats.dart` (in `cumsum`/`cumprod` flat & FFI paths)
*   **Root Cause:** `cumsum` on Boolean arrays was completely broken: the flat path threw `NoSuchMethodError` (due to `+` on `bool` values) and the strided FFI fallback path threw `TypeError` (casting `bool` directly to `num`).
*   **Fix:** Enabled generic return type promotion `<T, R>` and mapped the Boolean cumulative paths to output `Int32` arrays, casting `bool` to `0`/`1` during accumulation. Verified via `test/boolean_cumsum_bug_repro_test.dart`.

### 2.7 Silent Failure with Integer Step Spacing (Calculus)
*   **Location:** `calculus.dart` (in `trapz` and `gradient`)
*   **Root Cause:** The `StepSpacing` value was strictly type-checked with `value is double`. If a user passed an integer step (e.g. `Spacing.step(1)`), the FFI dispatches were silently skipped.
*   **Fix:** Checked if `value is num` and converted it using `.toDouble()` to support integer spacing. Verified via `test/integer_spacing_bug_repro_test.dart`.

---

## 3. Performance & FFI Optimizations (C Intrinsics)

Several files fall back to slow, element-wise pure Dart loops even when the input arrays are contiguous.

1.  **FFT Contiguous Copying:** In `fft.dart`, data is copied element-by-element between C pointers (`pin`/`pout`) and Dart `NDArray.data` via JIT loops.
    *   *Optimization:* For contiguous layouts, use native `memcpy` (via FFI) to copy the entire memory blocks instantly. For real inputs, write a C intrinsic to expand real values to complex (`imag = 0.0`) at C speed.
2.  **Missing `s_flatten_int16` Binding:** `_copyStridedToContiguous` throws `UnimplementedError` for `int16` because `s_flatten_int16` is missing in `ndarray_bindings.dart`.
    *   *Optimization:* Bind and implement `s_flatten_int16` to enable C-speed contiguous copies for 16-bit integer arrays.
3.  **I/O Non-Contiguous Saving:** Saving a non-contiguous array falls back to a slow, double-copying Dart loop (`a.toList()`).
    *   *Optimization:* Call `a.copy()` (which runs at FFI C-speed) and dump its raw memory pointer directly, bypassing Dart-speed list serializers completely.
4.  **Recursive Traversal in Manipulation:** `copyConcatenateRecursive` and `copyStackRecursive` run cell-by-cell Dart loops.
    *   *Optimization:* Port the strided concatenation/stacking coordinate traversals to C FFI to perform block memory transfers directly on the C heap.

---

## 4. Memory Management Violations (ScratchArena)

According to `AGENTS.md` guidelines: **"Always use ScratchArena for temporary allocations."** We have identified multiple instances of raw `malloc`/`calloc` allocations for shape and strides FFI buffers that bypass the arena.

### Identified Violations:
-   **`calculus.dart`**: Pivots and shape buffers allocated via raw `malloc` ([calculus.dart:149-151](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/calculus.dart#L149-L151)).
-   **`manipulation.dart` / `helpers.dart`**: Strides and shape arrays allocated via `malloc` ([helpers.dart:1078-1080](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/helpers.dart#L1078-L1080)).
-   **`stats.dart`**: Cumulative operation shape and strides allocations bypass the arena.
-   **`sorting.dart`**: `partition` and `argpartition` FFI indices buffers use raw `malloc` ([sorting.dart:1054-1058](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/sorting.dart#L1054-L1058)).
-   **`ndarray.dart` (`_copyStridedToContiguous`)**: Uses raw `malloc` for FFI shape/stride arrays ([ndarray.dart:916-917](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/ndarray.dart#L916-L917)).

### Proposed Fix:
Migrate all FFI buffer allocations to the pre-allocated arena:
```dart
final marker = ScratchArena.marker;
try {
  final cShape = ScratchArena.allocate<ffi.Int>(rank * ffi.sizeOf<ffi.Int>());
  // Execute FFI Calls...
} finally {
  ScratchArena.reset(marker);
}
```

---

## 5. Incomplete Generic Type Safety & Style Gaps

To enforce maximum compile-time validation, we must eliminate raw `NDArray` (resolving to `NDArray<dynamic>`) and ungeneric `DType` parameters.

### 5.1 Raw `NDArray` Signatures to Fix
-   **Linalg**: `inv`, `solve`, `pinv`, `matrix_power` signatures must use `NDArray<T>` instead of raw `NDArray`.
-   **Manipulation**: `stack`, `expand_dims`, `squeeze`, `slidingWindowView` must be refactored to `NDArray<T>`.
-   **I/O**: `save`, `load`, `savez`, `loadz` must utilize typed generics.

### 5.2 Non-Exhaustive DType Switches (If-Else Chains)
Long `if-else if` chains violate enum exhaustiveness.
-   **Violations:** `uniform`, `randint`, `normal`, `exponential`, `poisson`, `binomial` (in `random.dart`) and the spacing dispatches (in `calculus.dart`) use if-else chains.
-   **Remediation:** Refactor them to use strict `switch (dtype)` blocks to ensure compile-time safety for all 9 supported dtypes.

---

## 6. Major Test Coverage Gaps

The following testing voids leave significant portions of the operations codebase unverified:

1.  **Spacers**: `logspaceGrid` and `geomspaceGrid` have **zero unit tests** in the entire test directory.
2.  **Broadcasting**: `broadcastTo` stretching, success, and disposed exceptions have **zero dedicated unit tests**.
3.  **Calculus**: Zero tests for non-contiguous views (stride safety), user-provided `out` recycler buffers, or empty coordinate ranges.
4.  **I/O**: Zero tests for I/O serialization of `uint8`, `int16`, and `boolean` arrays.
5.  **Stats**: Zero tests for cumulative reductions (`cumsum`, etc.) on `boolean`, `uint8`, or `int16` layouts.

---

## 7. Conclusion & Action Plan

This audit presents a comprehensive checklist to complete the operational suite optimization:
*   **Phase 1: Critical Correctness Bug Fixes** (100% Completed).
*   **Phase 2: ScratchArena Migration** (Next up).
*   **Phase 3: Type Safety & Style Cleanup** (Pending).
*   **Phase 4: Test Coverage Expansion** (Pending).
