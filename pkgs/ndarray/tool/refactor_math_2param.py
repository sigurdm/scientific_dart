import re

def refactor_math():
    filepath = '/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/math.dart'
    with open(filepath, 'r') as f:
        content = f.read()

    # 1. Refactor generic unary ufuncs
    # NDArray<R> sin<T, R>(NDArray<T> a, {NDArray<R>? out})
    unary_pattern = r'NDArray<R>\s+(\w+)<T,\s*R>\(NDArray<T>\s+a,\s*\{\s*NDArray<R>\?\s*out\s*\}\)'
    unary_replacement = r'NDArray<R, MR> \1<T, MT extends Marker, R, MR extends Marker>(NDArray<T, MT> a, {NDArray<R, MR>? out})'
    content, count = re.subn(unary_pattern, unary_replacement, content)
    print(f"Refactored {count} generic unary ufunc signatures")

    # 2. Refactor double-returning unary ufuncs (T extends num)
    # NDArray<double> sinh<T extends num>(NDArray<T> a, {NDArray<double>? out})
    double_unary_pattern = r'NDArray<double>\s+(\w+)<T\s+extends\s+num>\(NDArray<T>\s+a,\s*\{\s*NDArray<double>\?\s*out\s*\}\)'
    double_unary_replacement = r'NDArray<double, MR> \1<T extends num, MT extends Marker, MR extends FloatingMarker>(NDArray<T, MT> a, {NDArray<double, MR>? out})'
    content, count = re.subn(double_unary_pattern, double_unary_replacement, content)
    print(f"Refactored {count} double-returning unary ufunc signatures")

    # 3. Refactor binary ufuncs
    # NDArray<R> add<Ta, Tb, R>(NDArray<Ta> a, NDArray<Tb> b, {NDArray<R>? out})
    binary_pattern = r'NDArray<R>\s+(\w+)<Ta,\s*Tb,\s*R>\(NDArray<Ta>\s+a,\s*NDArray<Tb>\s+b,\s*\{\s*NDArray<R>\?\s*out\s*\}\)'
    binary_replacement = r'NDArray<R, MR> \1<Ta, Ma extends Marker, Tb, Mb extends Marker, R, MR extends Marker>(NDArray<Ta, Ma> a, NDArray<Tb, Mb> b, {NDArray<R, MR>? out})'
    content, count = re.subn(binary_pattern, binary_replacement, content)
    print(f"Refactored {count} binary ufunc signatures")

    # 4. Refactor int-returning binary ufuncs (like bitwise operations? No, they are usually generic or return same type).
    # Let's check if there are others.

    # Apply general replacements
    content = content.replace("NDArray<double>", "NDArray<double, Float64Marker>")
    content = content.replace("NDArray<int>", "NDArray<int, Int64Marker>")
    content = content.replace("NDArray<Complex>", "NDArray<Complex, Complex128Marker>")
    content = content.replace("NDArray<bool>", "NDArray<bool, BooleanMarker>")

    content = content.replace("NDArray<R>.create", "NDArray<R, MR>.create")
    content = content.replace("NDArray<R> result", "NDArray<R, MR> result")
    content = content.replace("NDArray<R>? out", "NDArray<R, MR>? out")
    content = content.replace("NDArray<R>? result", "NDArray<R, MR>? result")

    content = content.replace("DType<R> targetDType", "DType<R, MR> targetDType")
    content = content.replace("DType<R> dtype", "DType<R, MR> dtype")

    # We also need to fix DType<double> and DType<Complex> casts inside math.dart.
    # E.g. `as DType<double>` -> `as DType<double, Float64Marker>`
    content = content.replace("as DType<double>", "as DType<double, Float64Marker>")
    content = content.replace("as DType<Complex>", "as DType<Complex, Complex128Marker>")
    
    # We also have `DType<dynamic>` which we replaced in ndarray.dart, but here we might have `DType<dynamic>` too.
    # Actually, we should replace `DType<dynamic>` with `DType<dynamic, Marker>` here too if it exists.
    content = content.replace("DType<dynamic>", "DType<dynamic, Marker>")

    with open(filepath, 'w') as f:
        f.write(content)
    print("Updated math.dart")

if __name__ == '__main__':
    refactor_math()
