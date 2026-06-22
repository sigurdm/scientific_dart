# Script to fix constructor calls in ndarray.dart
import re

def fix_constructors():
    filepath = '/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/ndarray.dart'
    with open(filepath, 'r') as f:
        content = f.read()

    # Generic constructor calls
    content = content.replace("NDArray<T>.create", "NDArray<T, M>.create")
    content = content.replace("NDArray<T>._", "NDArray<T, M>._")
    content = content.replace("NDArray<T>.fromList", "NDArray<T, M>.fromList")

    # Boolean constructor calls
    content = content.replace("NDArray<bool>.create", "NDArray<bool, BooleanMarker>.create")
    content = content.replace("NDArray<bool>._", "NDArray<bool, BooleanMarker>._")
    content = content.replace("NDArray<bool>.fromList", "NDArray<bool, BooleanMarker>.fromList")

    # Double constructor calls (usually Float64 by default)
    content = content.replace("NDArray<double>.create", "NDArray<double, Float64Marker>.create")
    content = content.replace("NDArray<double>._", "NDArray<double, Float64Marker>._")
    content = content.replace("NDArray<double>.fromList", "NDArray<double, Float64Marker>.fromList")

    # Complex constructor calls (Complex128 by default)
    content = content.replace("NDArray<Complex>.create", "NDArray<Complex, Complex128Marker>.create")
    content = content.replace("NDArray<Complex>._", "NDArray<Complex, Complex128Marker>._")
    content = content.replace("NDArray<Complex>.fromList", "NDArray<Complex, Complex128Marker>.fromList")

    # Int constructor calls (Int64 by default)
    content = content.replace("NDArray<int>.create", "NDArray<int, Int64Marker>.create")
    content = content.replace("NDArray<int>._", "NDArray<int, Int64Marker>._")
    content = content.replace("NDArray<int>.fromList", "NDArray<int, Int64Marker>.fromList")

    # Reshape, transpose, slice helper methods returning NDArray<T, M>
    # We also have generic methods like `NDArray<R, MR> reshape<R, MR extends Marker>(...)`?
    # No, reshape is NOT changing type.
    # But wait, we have `castNDArray`:
    # `NDArray<R> castNDArray<R>(DType<R> targetDType)`
    # This should become:
    # `NDArray<R, MR> castNDArray<R, MR extends Marker>(DType<R, MR> targetDType)`
    content = content.replace(
        "NDArray<R> castNDArray<R>(DType<R> targetDType)",
        "NDArray<R, MR> castNDArray<R, MR extends Marker>(DType<R, MR> targetDType)"
    )
    content = content.replace(
        "final result = NDArray<R>.create(shape, targetDType);",
        "final result = NDArray<R, MR>.create(shape, targetDType);"
    )

    with open(filepath, 'w') as f:
        f.write(content)
    print("Fixed constructors in ndarray.dart")

if __name__ == '__main__':
    fix_constructors()
