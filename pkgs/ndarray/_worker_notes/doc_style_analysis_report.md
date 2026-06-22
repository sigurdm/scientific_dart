# Comprehensive Dartdoc Style Analysis Report: `ndarray` Package

This report provides a detailed synthesis of all findings from the style consistency, formality, and guideline adherence analysis of public dartdoc comments within the `ndarray` package (`pkgs/ndarray/lib/src/`). The analysis has been verified against the actual codebase.

---

## 1. Executive Summary & Key Violations

The `ndarray` package contains comprehensive and detailed documentation, but exhibits systemic inconsistencies that deviate from [Effective Dart: Writing](https://dart.dev/effective-dart/documentation#writing) guidelines. The most common issues identified and verified are:

1. **Widespread Imperative Verbs**: Over 60 public functions start their documentation summaries with imperative verbs (e.g., "Compute", "Return", "Draw", "Integrate", "Calculate") instead of the required third-person singular form (e.g., "Computes", "Returns", "Draws", "Integrates", "Calculates").
2. **Inconsistent Section Formatting**: Preconditions, exceptions, examples, and performance considerations use highly inconsistent formats (e.g., H3 headers `###` vs. bold text `**` vs. plain text `Preconditions:`). Inconsistencies occur even within the same file (e.g., `operations/sorting.dart`).
3. **Invalid Code Block Backticks**: Markdown code blocks (` ```dart ... ``` `) frequently enclose type arguments in nested backticks (e.g., `` `NDArray<double>` ``). This results in literal backticks rendering in the output and is invalid Dart syntax inside code blocks.
4. **Duplicate & Conflicting Documentation**: Multiple methods (e.g., `setByMask`, `applyMask`, `slice`) have duplicated paragraphs or erroneous descriptions (e.g., describing boolean masks as binary `0` and `1` when the API requires `true` and `false`).
5. **Syntax Errors in Macros**: The `@example` macro is used without the required enclosing curly braces (`{}`) in `operations/distance.dart`.
6. **Incorrect Link Syntax**: Methods are linked using parentheses, such as `[dispose()]` instead of `[dispose]`, which prevents Dartdoc from resolving the references properly.
7. **Missing Documentation**: The public function `cumsum` in `operations/stats.dart` has no dartdoc comments.

---

## 2. Detailed Verification and Findings

### 2.1. Imperative Phrasing (Effective Dart: Writing Violations)
The first sentence of a dartdoc comment must start with a third-person singular verb. The following files/methods violate this rule:
* **"Compute"** instead of **"Computes"**:
  - `operations/stats.dart`: `sum` (L106), `prod` (L234), `mean` (L471), `std` (L649), `nanvar` (L714), `nanstd` (L817), `min` (L876), `nanmin` (L1028), `max` (L1216), `nanmax` (L1368), `cumprod` (L1610), `cummin` (L1677), `cummax` (L1736), `variance` (L1795), `nanmean` (L1892), `quantile` (L2031), `percentile` (L2239), `median` (L2264).
  - `operations/linalg.dart`: `inv` (L699), `det` (L915), `eig` (L1362), `pinv` (L1693), `cholesky` (L1875).
  - `operations/math.dart`: Over 40 ufuncs, including `sqrt` (L37), `square` (L95), `sin` (L249), `cos` (L389), `exp` (L532), `log` (L630), `tan` (L854), and cumulative operations like `cumsum` (L6340).
* **"Return"** instead of **"Returns"**:
  - `operations/sorting.dart`: `where` (L957).
  - `operations/random.dart`: `randint` (L76).
* **"Draw"** instead of **"Draws"**:
  - `operations/random.dart`: `normal` (L169), `exponential` (L252), `poisson` (L337), `binomial` (L408), `multivariateNormal` (L488), `multinomial` (L607).
* **"Integrate"** instead of **"Integrates"**:
  - `operations/calculus.dart`: `trapz` (L49).
* **"Calculate"** instead of **"Calculates"**:
  - `operations/calculus.dart`: `gradient` (L386), `gradientArray` (L753).
* **"Find"** instead of **"Finds"**:
  - `operations/sorting.dart`: `searchsorted` (L704).
* **"Manually free"** instead of **"Manually frees"**:
  - `ndarray.dart`: `dispose` (L2689).

### 2.2. Informal Language & Slang
* **`**Gotchas:**`** Header: Informal; should be replaced with `**Edge cases:**` or `**Limitations:**`.
  - `ndarray.dart` (L103, L403, L605)
  - `operations/stats.dart` (L878, L1218)
  - `operations/math.dart` (L48)
* **"100% unmanaged"**: Too informal.
  - `ndarray.dart` (L261): `/// // 100% unmanaged, survives the scope block!` -> Should be "completely unmanaged".
* **"fancy row stack..." / "fancy index..."**: "Fancy" is informal terminology borrowed from Python.
  - `ndarray.dart` (L1508, L1547, L1576) -> Should be "advanced indexing".
* **"3x Footprint Hazard"**: Dramatized technical documentation.
  - `operations/io.dart` (L475): `Memory Consideration Warning (3x Footprint Hazard)` -> Should be "Memory overhead".

### 2.3. Section Header Inconsistency
Header styles fluctuate between Markdown H3 (`###`), bold (`**`), and plain text, even within the same file:
* **Exceptions / Throws**:
  - `**Throws:**` is used in most functions.
  - `**Exceptions:**` is used in `ndarray.dart` comparisons (e.g. `operator >` L1903).
  - `**Exceptions Thrown:**` is used in `operations/sorting.dart` (`findIndex` L1842).
  - `### Throws` is used in `operations/sorting.dart` (`searchsorted` L719).
  - `### Exceptions` is used in `operations/padding.dart` (`pad` L205).
* **Preconditions**:
  - `**Preconditions:**` is used in most files.
  - `### Preconditions` is used in `operations/sorting.dart` (`searchsorted` L712) and `operations/padding.dart` (`pad` L197).
  - `Preconditions:` (plain text) is used in `operations/padding.dart` (`PadWidth.all` L31, `PadWidth.axes` L44, `StatLength` constructors L122, L137) and `ndarray.dart` (`BooleanMask` L2845).
* **Performance**:
  - `**Performance considerations:**` (lowercase 'c') is used in `operations/stats.dart`, `operations/broadcasting.dart`, `nditer.dart`.
  - `**Performance Considerations:**` (capitalized 'C') is used in `ndarray.dart` and `operations/sorting.dart` (`findIndex` L1849).
  - `### Performance Considerations` is used in `operations/sorting.dart` (`searchsorted` L723) and `operations/padding.dart` (`pad` L210).
  - `**Performance:**` is used in `ndarray.dart` comparisons (e.g. `operator >` L1907).
* **Examples**:
  - `**Example:**` is used in most files.
  - `### Example` is used in `operations/padding.dart` (`pad` L219).
  - `### Inline Example:` is used in `operations/sorting.dart` (`searchsorted` L727) and `operations/spacers.dart` (`SearchSide` L38).

### 2.4. Markdown Syntax & Code Blocks
* **Backticks Inside Code Blocks**:
  In multiple files, inside code blocks, generic parameters are unnecessarily surrounded by backticks, e.g. `` `NDArray<double>` ``:
  - `ndarray.dart` (L102, L399, L533, L547, L568, L601, L914, L1153)
  - `operations/broadcasting.dart` (L44, L45)
  - `operations/linalg.dart` (L1195, L1196, L2037, L2281)
  - `operations/stats.dart` (L727, L830, L1906)
  - `operations/math.dart` (L740, L3404, L3475)
* **Math Complexity Formatting**:
  Complexity is documented as `$O(N)$` (LaTeX format) in most files (e.g. `broadcasting.dart`, `calculus.dart`, `io.dart`, `math.dart`, `stats.dart`), but as `` `O(N)` `` (backtick format) in `ndarray.dart` (L1909, L1936, L1962, L1982, L2018).

### 2.5. Parameter and Macro Formatting Errors
* **Missing curly braces around `@example`**:
  - `operations/distance.dart` (L100, L202): `/// @example /example/distance_example.dart` -> Should be `/// {@example /example/distance_example.dart}`.
* **Incorrect Parameter References**:
  Methods are referenced as `[dispose()]` instead of `[dispose]`:
  - `operations/manipulation.dart` (L176, L227, L310, L406)
  - `operations/io.dart` (L151, L484)
  - `operations/random.dart` (L97)
  - `operations/calculus.dart` (L428, L793)
  - `operations/math.dart` (L5344, L5345)
  - `operations/shaping_meshes.dart` (L106)
  - `operations/sorting.dart` (L984)

### 2.6. Redundant, Verbose, or Inaccurate Explanations
* **`setByMask` in `ndarray.dart` (L1407-1424)**:
  - The documentation block is duplicated and contradictory. One block states: `Modifies elements where the provided boolean binary [mask] contains 1` and `[mask] values must be binary (0 or 1)`. Another block correctly states `contains true`. Since the mask is an `NDArray<bool>`, it cannot contain `1` or `0`.
  - The documentation also refers to parameter `[value]` instead of the actual signature variable `values`.
* **`applyMask` in `ndarray.dart` (L2289)**:
  - Documentation states: `The [mask] array must have elements with value 0 or 1. Returns a 1D array containing the elements where the mask is 1.` -> Incorrect since it accepts `NDArray<bool>`.
* **`slice` in `ndarray.dart` (L2029-2041)**:
  - Contains duplicate summaries: `Returns a view of this array...` (L2029) and `Returns a view or copy of the array...` (L2038).
* **`linspace` & `linspaceWithStep` in `operations/spacers.dart` (L80, L114)**:
  - Parameter list uses capitalized "True": `If True...` -> Should be "If true".
* **`linspace` in `operations/spacers.dart` (L68-70)**:
  - Redundant summary statements: `Returns an array with evenly spaced values over a specified interval.` followed by `Returns [numSamples] evenly spaced samples...`.

---

## 3. Summary of Files Requiring Modification

| File Path | Functions / Classes | Summary of Action Needed |
| :--- | :--- | :--- |
| [`ndarray.dart`](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/ndarray.dart) | `NDArray`, `setByMask`, `applyMask`, `slice` | Remove nested backticks in examples, standardize preconditions/exceptions headings, replace "fancy" indexing and "100%" unmanaged slang, fix boolean mask descriptions (`true`/`false` instead of `0`/`1`), deduplicate `slice` and `setByMask` comments, fix `$O(N)$` LaTeX math formatting. |
| [`nditer.dart`](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/nditer.dart) | `NDIter`, `NDEnumerate`, `getIndex` | Fix typo ("NDIter reuse" -> "NDIter reuses"), document `RangeError` on `getIndex`, and deduplicate class/constructor throws. |
| [`operations/stats.dart`](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/stats.dart) | `cumsum`, `sum`, `mean`, `std`, `min`, `max`, etc. | **Add missing dartdoc comments for `cumsum` (L1556)**, change "Compute" to "Computes", remove inline backticks in code blocks, replace "Gotchas" headers. |
| [`operations/math.dart`](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/math.dart) | All element-wise ufuncs | Change "Compute" to "Computes" in 40+ functions, remove inline backticks in code blocks, fix `[dispose()]` references, replace "Gotchas" headers. |
| [`operations/linalg.dart`](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/linalg.dart) | `inv`, `det`, `eig`, `pinv`, `cholesky` | Change "Compute" to "Computes" in all functions, remove inline backticks in code blocks. |
| [`operations/random.dart`](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/random.dart) | `randint`, `normal`, `exponential`, etc. | Change "Return/Draw" to "Returns/Draws", fix `[dispose()]` references. |
| [`operations/calculus.dart`](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/calculus.dart) | `trapz`, `gradient`, `gradientArray` | Change "Integrate/Calculate" to "Integrates/Calculates", fix `[dispose()]` references. |
| [`operations/sorting.dart`](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/sorting.dart) | `searchsorted`, `where`, `findIndex` | Change "Find/Return" to "Finds/Returns", standardize `searchsorted` H3 headers to bold headers, standardize `findIndex` "Exceptions Thrown" heading. |
| [`operations/distance.dart`](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/distance.dart) | `pdist`, `cdist` | **Fix example macro syntax (missing `{}` on `@example`)**. |
| [`operations/repeating_tiling.dart`](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/repeating_tiling.dart) | `tile`, `repeat` | Fix Python terminology (replace `a.ndim` with `a.rank`). |
| [`operations/spacers.dart`](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/spacers.dart) | `linspace`, `linspaceWithStep` | Deduplicate summaries, replace "True" with "true", convert parameter list to inline description. |
| [`operations/interpolation.dart`](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/interpolation.dart) | `interp` | Expand noun-phrase summary to full sentence, convert parameter list to inline description. |
| [`operations/padding.dart`](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/padding.dart) | `pad`, `PadWidth`, `StatLength` | Standardize H3 headers to bold headers, standardize plain text `Preconditions:` to bold headers. |
| [`operations/broadcasting.dart`](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/broadcasting.dart) | `broadcast`, `broadcastTo` | Remove nested backticks in examples, standardize headings. |
| [`operations/io.dart`](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/io.dart) | `save`, `load`, etc. | Replace "3x Footprint Hazard" slang, fix `[dispose()]` references. |
| [`operations/shaping_meshes.dart`](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/shaping_meshes.dart) | `asStrided`, `mgrid`, `ogrid` | Fix `[dispose()]` references. |
| [`operations/manipulation.dart`](file:///usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/manipulation.dart) | `squeeze`, `expandDims`, `stack`, etc. | Fix `[dispose()]` references. |
