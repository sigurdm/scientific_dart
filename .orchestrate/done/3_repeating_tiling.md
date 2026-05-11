# Repeating & Tiling Implementation Walkthrough

Implemented the `repeat()` and `tile()` operations for the `ndarray` package in a clean, isolated, and fully compliant manner.

## Implementation Details

### `repeat()`
- File: `pkgs/ndarray/lib/src/repeating_tiling.dart`
- Signature: `NDArray<T> repeat<T>(NDArray<T> a, dynamic repeats, {int? axis, NDArray<T>? out})`
- Behavior:
  - If `axis` is null, flattens the array first and repeats elements.
  - Normalizes negative axes.
  - Supports scalar `int`, `List<int>`, and `NDArray<int>` for `repeats`.
  - Ensures `repeats` values are non-negative.
  - Validates `out` parameter shape and dtype if provided.
  - Uses an elegant recursive coordinate-walking algorithm to copy repeated elements, naturally handling any dimensionality.
  - Cleans up any temporary copies (e.g., from `flatten()`) safely.

### `tile()`
- File: `pkgs/ndarray/lib/src/repeating_tiling.dart`
- Signature: `NDArray<T> tile<T>(NDArray<T> a, dynamic reps, {NDArray<T>? out})`
- Behavior:
  - Promotes input array dimensionality or prepends 1s to reps if ranks don't match, matching NumPy behavior.
  - Supports scalar `int`, `List<int>`, and `NDArray<int>` for `reps`.
  - Validates `out` parameter shape and dtype if provided.
  - Uses an extremely elegant recursive coordinate walker with `%` (modulo) operations to copy tiled elements safely across arbitrary dimensions.
  - Cleans up temporary reshaped views/copies safely.

### Integration
- Exported the new operations from `pkgs/ndarray/lib/ndarray.dart`.
- Removed the old duplicate implementation of `repeat` and `tile` from `pkgs/ndarray/lib/src/operations.dart` to ensure single source of truth.

## Testing & Quality Assurance
- Created comprehensive tests in `pkgs/ndarray/test/repeating_tiling_test.dart` covering:
  - 1D, 2D, 3D arrays.
  - Scalar and list-based repeats/reps.
  - Negative axes.
  - Axis = null (flattening behavior).
  - Edge cases: `0` repeats/reps (returns empty array or excludes elements).
  - Mismatched or negative inputs (proper exception checks).
  - `out` parameter validation (success and error paths).
- All 465 tests in the repository (including the new ones) pass successfully.
- Formatting and analysis checks are completely clean on the modified files.
