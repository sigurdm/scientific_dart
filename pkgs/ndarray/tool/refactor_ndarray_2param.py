import re

def refactor_ndarray():
    filepath = '/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/ndarray.dart'
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. Remove operators block first
    content = content.replace("import 'operations.dart' as ops;", "")
    
    start_anchor = '  NDArray _wrapScalar(dynamic value'
    end_anchor = '(x, y) => x == y,\n    );\n    return result;\n  }'
    
    start_idx = content.find(start_anchor)
    if start_idx != -1:
        end_idx = content.find(end_anchor, start_idx)
        if end_idx != -1:
            end_idx += len(end_anchor)
            content = content[:start_idx] + content[end_idx:]
            print("Removed operators from ndarray.dart")

    # Define markers
    markers = """
/// Base marker interface for all types that can be stored in an [NDArray].
abstract interface class Marker {}

/// Marker interface for numeric elements (integers, floats, complex).
abstract interface class NumericMarker implements Marker {}

/// Marker interface for floating-point elements.
abstract interface class FloatingMarker implements NumericMarker {}

/// Marker interface for integer elements.
abstract interface class IntegerMarker implements NumericMarker {}

/// Marker interface for complex number elements.
abstract interface class ComplexMarker implements NumericMarker {}

/// Represents a boolean element in an [NDArray].
abstract interface class BoolMarker implements Marker {}

/// Marker for double-precision float (64-bit).
final class Float64Marker implements FloatingMarker {}

/// Marker for single-precision float (32-bit).
final class Float32Marker implements FloatingMarker {}

/// Marker for 64-bit signed integer.
final class Int64Marker implements IntegerMarker {}

/// Marker for 32-bit signed integer.
final class Int32Marker implements IntegerMarker {}

/// Marker for 8-bit unsigned integer.
final class Uint8Marker implements IntegerMarker {}

/// Marker for 16-bit signed integer.
final class Int16Marker implements IntegerMarker {}

/// Marker for double-precision complex (128-bit).
final class Complex128Marker implements ComplexMarker {}

/// Marker for single-precision complex (64-bit).
final class Complex64Marker implements ComplexMarker {}

/// Marker for boolean.
final class BooleanMarker implements BoolMarker {}
"""

    # Revert Complex64 and Complex128 extension types (remove them)
    content = re.sub(r"extension type const Complex64._.*?}\n", "", content, flags=re.DOTALL)
    content = re.sub(r"extension type const Complex128._.*?}\n", "", content, flags=re.DOTALL)

    # Revert ComplexList to non-generic
    content = content.replace(
        "final class ComplexList<T extends Complex> extends ListBase<T> {",
        "final class ComplexList extends ListBase<Complex> {"
    )
    content = re.sub(
        r"  T operator\s*\[\]\(int index\)\s*\{\s*return Complex\(_list\[index \* 2\], _list\[index \* 2 \+ 1\]\) as T;\s*\}",
        "  Complex operator [](int index) {\n    return Complex(_list[index * 2], _list[index * 2 + 1]);\n  }",
        content
    )
    content = content.replace(
        "  void operator []=(int index, T value) {",
        "  void operator []=(int index, Complex value) {"
    )

    # Insert markers before DType enum specifically
    content = content.replace(
        "/// Supported data types for the elements of an [NDArray].\nenum DType<T> {",
        markers + "\n/// Supported data types for the elements of an [NDArray].\nenum DType<T, M extends Marker> {"
    )

    # Update DType values
    content = content.replace(
        "float32<Float32>('float32', 4, '<f4'),",
        "float32<double, Float32Marker>('float32', 4, '<f4'),"
    )
    content = content.replace(
        "float64<Float64>('float64', 8, '<f8'),",
        "float64<double, Float64Marker>('float64', 8, '<f8'),"
    )
    content = content.replace(
        "complex64<Complex64>('complex64', 8, '<c8'),",
        "complex64<Complex, Complex64Marker>('complex64', 8, '<c8'),"
    )
    content = content.replace(
        "complex128<Complex128>('complex128', 16, '<c16'),",
        "complex128<Complex, Complex128Marker>('complex128', 16, '<c16'),"
    )
    content = content.replace(
        "uint8<Uint8>('uint8', 1, '|u1'),",
        "uint8<int, Uint8Marker>('uint8', 1, '|u1'),"
    )
    content = content.replace(
        "int16<Int16>('int16', 2, '<i2'),",
        "int16<int, Int16Marker>('int16', 2, '<i2'),"
    )
    content = content.replace(
        "int32<Int32>('int32', 4, '<i4'),",
        "int32<int, Int32Marker>('int32', 4, '<i4'),"
    )
    content = content.replace(
        "int64<Int64>('int64', 8, '<i8'),",
        "int64<int, Int64Marker>('int64', 8, '<i8'),"
    )
    content = content.replace(
        "boolean<bool>('boolean', 1, '|b1');",
        "boolean<bool, BooleanMarker>('boolean', 1, '|b1');"
    )

    # Update NDArray definition: NDArray<T, M extends Marker>
    content = content.replace(
        "final class NDArray<T> implements ffi.Finalizable {",
        "final class NDArray<T, M extends Marker> implements ffi.Finalizable {"
    )

    # Update NDArray.dtype field
    content = content.replace(
        "  final DType<T> dtype;",
        "  final DType<T, M> dtype;"
    )

    # Update DType<T> to DType<T, M> in signatures
    content = content.replace("DType<T> dtype", "DType<T, M> dtype")
    content = content.replace("DType<T>? dtype", "DType<T, M>? dtype")
    content = content.replace("DType<R> targetDType", "DType<R, MR> targetDType")
    content = content.replace("final DType<T> resolvedDType", "final DType<T, M> resolvedDType")
    content = content.replace("as DType<T>", "as DType<T, M>")

    # Replace NDArray<bool> with NDArray<bool, BooleanMarker>
    content = content.replace("NDArray<bool>", "NDArray<bool, BooleanMarker>")

    # Replace NDArray<int> with NDArray<int, IntegerMarker> in signatures and bodies
    content = content.replace("NDArray<int> indices", "NDArray<int, IntegerMarker> indices")
    content = content.replace("NDArray<int> spec", "NDArray<int, IntegerMarker> spec")
    content = content.replace("NDArray<int> strides", "NDArray<int, IntegerMarker> strides")
    content = content.replace("spec is NDArray<int>", "spec is NDArray<int, IntegerMarker>")
    content = content.replace("NDArray<int>.fromList", "NDArray<int, Int32Marker>.fromList")

    # Replace NDArray<double> with NDArray<double, FloatingMarker> where appropriate
    content = content.replace("NDArray<double> view", "NDArray<double, FloatingMarker> view")
    content = content.replace("NDArray<double>? spacingArray", "NDArray<double, FloatingMarker>? spacingArray")
    content = content.replace("NDArray<double>.fromList", "NDArray<double, Float64Marker>.fromList")
    content = content.replace("NDArray<double>.zeros", "NDArray<double, Float64Marker>.zeros")
    content = content.replace("NDArray<double>.ones", "NDArray<double, Float64Marker>.ones")
    content = content.replace("NDArray<double>.arange", "NDArray<double, Float64Marker>.arange")
    content = content.replace("NDArray<double>.eye", "NDArray<double, Float64Marker>.eye")
    content = content.replace("NDArray<double>.create", "NDArray<double, Float64Marker>.create")

    # Replace NDArray<Complex> with NDArray<Complex, ComplexMarker>
    content = content.replace("NDArray<Complex> complexView", "NDArray<Complex, ComplexMarker> complexView")
    content = content.replace("NDArray<Complex>.create", "NDArray<Complex, Complex128Marker>.create")
    content = content.replace("NDArray<Complex>.fromList", "NDArray<Complex, Complex128Marker>.fromList")

    # Generic method signatures returning NDArray
    self_methods = [
        "view", "reshape", "transpose", "slice", "copy", "flatten", 
        "diagonal", "getRow", "getCol", "applyMask", "detachFromScope", 
        "detachToParentScope", "ravel", "take", "expandDims", "squeeze", 
        "swapaxes", "moveaxis"
    ]
    for method in self_methods:
        content = content.replace(f"NDArray<T> {method}", f"NDArray<T, M> {method}")
        # Also handle get transposed: NDArray<T> get transposed
        content = content.replace(f"NDArray<T> get {method}", f"NDArray<T, M> get {method}")

    # Specific fix for get transposed
    content = content.replace("NDArray<T> get transposed", "NDArray<T, M> get transposed")

    # Specific fix for copy out param: copy({NDArray<T>? out})
    content = content.replace("copy({NDArray<T>? out})", "copy({NDArray<T, M>? out})")

    content = content.replace("NDArray<T> otherArr", "NDArray<T, M> otherArr")
    content = content.replace("NDArray<T> other", "NDArray<T, M> other")
    content = content.replace("NDArray<T> values", "NDArray<T, M> values")
    content = content.replace("NDArray<T> dest", "NDArray<T, M> dest")
    content = content.replace("NDArray<T> src", "NDArray<T, M> src")
    content = content.replace("NDArray<T> result", "NDArray<T, M> result")
    content = content.replace("NDArray<T> a", "NDArray<T, M> a")
    content = content.replace("NDArray<T> b", "NDArray<T, M> b")
    content = content.replace("NDArray<T> parent", "NDArray<T, M> parent")

    # value is NDArray<T> check
    content = content.replace("value is NDArray<T>", "value is NDArray<T, M>")

    # Generic constructor calls
    content = content.replace("NDArray<T>.create", "NDArray<T, M>.create")
    content = content.replace("NDArray<T>._", "NDArray<T, M>._")
    content = content.replace("NDArray<T>.fromList", "NDArray<T, M>.fromList")
    content = content.replace("NDArray<T>.zeros", "NDArray<T, M>.zeros")

    # Boolean constructor calls
    content = content.replace("NDArray<bool>.create", "NDArray<bool, BooleanMarker>.create")
    content = content.replace("NDArray<bool>._", "NDArray<bool, BooleanMarker>._")
    content = content.replace("NDArray<bool>.fromList", "NDArray<bool, BooleanMarker>.fromList")

    # Double constructor calls
    content = content.replace("NDArray<double>.create", "NDArray<double, Float64Marker>.create")
    content = content.replace("NDArray<double>._", "NDArray<double, Float64Marker>._")
    content = content.replace("NDArray<double>.fromList", "NDArray<double, Float64Marker>.fromList")

    # Complex constructor calls
    content = content.replace("NDArray<Complex>.create", "NDArray<Complex, Complex128Marker>.create")
    content = content.replace("NDArray<Complex>._", "NDArray<Complex, Complex128Marker>._")
    content = content.replace("NDArray<Complex>.fromList", "NDArray<Complex, Complex128Marker>.fromList")

    # Int constructor calls
    content = content.replace("NDArray<int>.create", "NDArray<int, Int64Marker>.create")
    content = content.replace("NDArray<int>._", "NDArray<int, Int64Marker>._")
    content = content.replace("NDArray<int>.fromList", "NDArray<int, Int64Marker>.fromList")

    # castNDArray fix
    content = content.replace(
        "NDArray<R> castNDArray<R>(DType<R> targetDType)",
        "NDArray<R, MR> castNDArray<R, MR extends Marker>(DType<R, MR> targetDType)"
    )
    content = content.replace(
        "final result = NDArray<R>.create(shape, targetDType);",
        "final result = NDArray<R, MR>.create(shape, targetDType);"
    )

    # ComplexList cleanup in NDArray.create
    content = content.replace("ComplexList<Complex128>", "ComplexList")
    content = content.replace("ComplexList<Complex64>", "ComplexList")

    # Fix other wrong type arguments in ndarray.dart
    content = content.replace("DType<dynamic>", "DType<dynamic, Marker>")
    content = content.replace("NDArray<dynamic>", "NDArray<dynamic, Marker>")

    # Clean up operations/broadcasting.dart import if it is unused
    content = content.replace("import 'operations/broadcasting.dart';", "")

    with open(filepath, 'w') as f:
        f.write(content)
    print("Completed refactoring of ndarray.dart")

if __name__ == '__main__':
    refactor_ndarray()
