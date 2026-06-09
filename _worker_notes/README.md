# Phase 3 Synthesis - Worker Notes

This workspace contains the synthesized solution for Phase 3 (Type Safety & FFI Operations Refactoring) in the `ndarray` package. It combines the strengths of both Worker 0 and Worker 1.

## What Works
- **Type Safety in Math Fallbacks**: Functions `floor_divide`, `remainder`, `sin`, and `abs` handle `uint8` and `int16` arrays without throwing `TypeError`.
- **Negative Operator**: Fixed silent fallthrough for `uint8`/`int16` types using `switch` dispatch.
- **Linalg Det Type Consistency**: `det` returns correct type (`double` for float32/float64, `Complex` for complex64/complex128).
- **SVD/QR Integer Validation**: Correctly throws `ArgumentError` for integer inputs.
- **Complex SVD & Pinv**: SVD supports complex types and returns real singular values `S` as `NDArray<double>`. `pinv` uses this to compute pseudo-inverse correctly.
- **Generic Refactoring**: Raw `NDArray` usages have been replaced with `NDArray<T>` in `fft`, `ifft`, `nanmean`, `atan2`, `negative`, `abs`, `isnan`, `isinf`, and all logical operations.
- **Switch Dispatch**: `if-else if` chains on `dtype`/`targetDType` have been replaced with `switch` in all 14 requested functions.
- **Generic isclose/allclose/norm**: Merged from Worker 0 to ensure strict type safety. `isclose` and `allclose` are generic, as well as `norm` and its helpers.

## Setup Requirements
The package uses a custom C++ extension that links against Google Highway. The static libraries for Highway (`libhwy.a` and `libhwy_contrib.a`) must be present in `pkgs/ndarray/third_party/highway/build/`.
If they are missing (e.g. in a clean worktree since they are gitignored), you must build them first or copy them from a workspace that has them.
To build them (assuming cmake and make are available):
```bash
cd pkgs/ndarray/third_party/highway
mkdir -p build
cd build
cmake ..
make
```
In this workspace, they have been copied from Worker 1's workspace to allow immediate testing.

## How to Run Tests
To run all tests in the package:
```bash
cd pkgs/ndarray
dart pub get
dart test
```

To run specific type safety tests:
```bash
cd pkgs/ndarray
dart test test/phase3_type_safety_test.dart
dart test test/type_safety_fixes_test.dart
```

All 710 tests are passing.
