# Task 6: Documentation Cleanup for ndarray

## Work Performed
1.  Identified widespread incorrect backticks in Dartdoc examples across `pkgs/ndarray/lib/src/`.
    *   Example: `` `NDArray<double>`.ones(...) `` instead of `NDArray<double>.ones(...)`.
    *   Example: `` `DType.float64);` `` instead of `DType.float64);`.
2.  Automated the cleanup using a Dart script to:
    *   Remove backticks from type fragments in code blocks.
    *   Fix broken triple backticks caused by previous regex replacements.
    *   Wrap type fragments with angle brackets (e.g., `NDArray<T>`, `List<int>`) in backticks within prose to avoid "unintended HTML" analyzer warnings.
3.  Improved `NDArray.fromList` documentation:
    *   Added a total size check to ensure the input list length matches the requested shape.
    *   Added `ArgumentError` documentation and implementation.
4.  Added missing Dartdoc to extensions in `extensions.dart`.
5.  Verified fixes by running `dart analyze` and checking generated docs (manually via `read_file`).

## Files Modified
*   `pkgs/ndarray/lib/src/ndarray.dart`
*   `pkgs/ndarray/lib/src/operations.dart`
*   `pkgs/ndarray/lib/src/broadcasting.dart`
*   `pkgs/ndarray/lib/src/extensions.dart`
*   `pkgs/ndarray/lib/src/io.dart`
*   `pkgs/ndarray/lib/src/random.dart`
*   `pkgs/ndarray/lib/src/fft.dart`
