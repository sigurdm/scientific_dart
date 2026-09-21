# Phantom dtype tag migration — working notes

**Branch:** `spike/phantom-dtype-tags`. Temporary file; delete before merging.

## What changed

`NDArray<T>`'s type parameter used to be the **element type** (`double`, `int`,
`Complex`, `bool`) dressed up in `extension type` wrappers named `Float64`,
`Int32`, ... Because extension types are **erased**, `NDArray<Float64>` and
`NDArray<Float32>` were the *same runtime type*, so every `as NDArray<X>` cast
in the package silently succeeded and lied.

Now the type parameter is a **reified phantom tag**:

```dart
sealed class DTypeTag<E> {}

abstract final class Float64 extends DTypeTag<double> {}
abstract final class Float32 extends DTypeTag<double> {}
abstract final class Int32   extends DTypeTag<int> {}
abstract final class Complex128 extends DTypeTag<Complex> {}
abstract final class Boolean extends DTypeTag<bool> {}
// ... one per dtype, 15 in total

typedef AnyFloat   = DTypeTag<double>;   // float16/bfloat16/float32/float64
typedef AnyInt     = DTypeTag<int>;      // int8..int64, uint8..uint64
typedef AnyComplex = DTypeTag<Complex>;  // complex64/complex128
typedef AnyReal    = DTypeTag<num>;      // AnyFloat + AnyInt
typedef AnyDType   = DTypeTag<Object?>;  // everything, incl. Boolean

final class NDArray<T extends AnyDType> { ... }
enum DType<T extends AnyDType> { ... }
```

Tags are `abstract` and never instantiated. Generics are covariant, so
`NDArray<Float32> <: NDArray<AnyFloat> <: NDArray<AnyReal> <: NDArray<AnyDType>`.

## Element types

`NDArray`'s class members can no longer name the element type, so they take and
return `Object?` and are suffixed `Untyped` / `Raw`. The nice names live on an
extension that recovers `E` from the tag's `DTypeTag<E>` bound:

```dart
extension NDArrayElements<T extends DTypeTag<E>, E> on NDArray<T> {
  List<E> get data;
  E get scalar;
  List<E> toList();
  E    getCell(List<int> coords);      void setCell(List<int>, E);
  E    getCellRaw(int rawOffset);      void setCellRaw(int, E);
  E    getCellFlat(int flatIndex);     void setCellFlat(int, E);
  void fill(E value);
}
```

| use this (typed)   | class member (untyped, `Object?`) |
|--------------------|-----------------------------------|
| `data`             | `dataRaw`                         |
| `scalar`           | `scalarRaw`                       |
| `toList()`         | `toListRaw()`                     |
| `getCell`          | `getCellUntyped`                  |
| `setCell`          | `setCellUntyped`                  |
| `getCellRaw`       | `getCellRawUntyped`               |
| `setCellRaw`       | `setCellRawUntyped`               |
| `getCellFlat`      | `getCellFlatUntyped`              |
| `setCellFlat`      | `setCellFlatUntyped`              |
| `fill`             | `fillUntyped`                     |

On `NDArray<Float64>` you get `double`; on `NDArray<AnyInt>` you get `int`; in
code generic over `T extends AnyDType` you get `Object?` — which is honest,
since such code genuinely does not know the element type.

## Translation table for fixing errors

| old                        | new                                               |
|----------------------------|---------------------------------------------------|
| `NDArray<double>`          | `NDArray<Float64>` if it really is float64, else `NDArray<AnyFloat>` |
| `NDArray<int>`             | `NDArray<Int32>` / `NDArray<Int64>` for results, `NDArray<AnyInt>` for inputs |
| `NDArray<bool>`            | `NDArray<Boolean>`                                |
| `NDArray<Complex>`         | `NDArray<AnyComplex>`                             |
| `NDArray<num>`             | `NDArray<AnyReal>`                                |
| `NDArray<Object>` / `<dynamic>` | `NDArray<AnyDType>`                          |
| `<T extends Object>`       | `<T extends AnyDType>`                            |
| `<T extends num>`          | `<T extends AnyReal>`                             |
| `<T extends Complex>`      | `<T extends AnyComplex>`                          |
| `Float64(x)`, `Int32(x)` … | just `x` (the wrappers are gone)                  |
| `Complex64(re, im)`        | `Complex(re, im)`                                 |
| `value as T` (element)     | drop the cast; the value is `Object?`             |
| `ComplexList<Complex128>`  | `ComplexList` (no longer generic)                 |

## Rules

1. **Do not** put an `extends AnyDType` bound on a type parameter that is not a
   dtype tag — e.g. the `R` of `NDArray.scope<R>(R Function())` or of
   `_withWrappedScalar<R>`, which stand for arbitrary results.
2. **Keep the runtime `out.dtype == ...` validations.** Where two parameters
   share a type variable, Dart silently takes the least upper bound instead of
   reporting a conflict, so the static system narrows the error space but does
   not close it.
3. Prefer the narrowest honest tag. A function that only ever produces float64
   should say `NDArray<Float64>`, not `NDArray<AnyFloat>`.
4. Never "fix" an error by casting a tag away (`as NDArray<Float64>` on
   something that isn't). That reintroduces exactly the bug this migration
   removes. If a cast is unavoidable, the surrounding code should have already
   checked `dtype`.
