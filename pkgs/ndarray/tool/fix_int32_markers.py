import re

def fix_file(filepath, replacements):
    print(f"Fixing {filepath}...")
    with open(filepath, 'r') as f:
        content = f.read()
    for pattern, replacement in replacements.items():
        content = re.sub(pattern, replacement, content)
    with open(filepath, 'w') as f:
        f.write(content)

def main():
    # 1. helpers.dart
    # Replace NDArray<int, Int64Marker> counts/dest with Int32Marker
    # in nanReduceRecursive, countNonzeroRecursive, argMinMaxRecursive
    fix_file(
        '/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/helpers.dart',
        {
            r'NDArray<int, Int64Marker> counts': 'NDArray<int, Int32Marker> counts',
            r'NDArray<int, Int64Marker> dest': 'NDArray<int, Int32Marker> dest',
        }
    )

    # 2. stats.dart
    # Line 1683: final counts = NDArray<int, Int64Marker>.zeros(newShape, DType.int32);
    fix_file(
        '/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/stats.dart',
        {
            r'NDArray<int, Int64Marker>\.zeros\(newShape, DType\.int32\)': 'NDArray<int, Int32Marker>.zeros(newShape, DType.int32)',
            r'final counts = NDArray<int, Int64Marker>': 'final counts = NDArray<int, Int32Marker>',
        }
    )

    # 3. sorting.dart
    # Almost all Int64Marker should be Int32Marker in sorting.dart
    # because they are all paired with DType.int32 (except maybe some others? We checked they are all DType.int32).
    # Wait, count_nonzero has: NDArray<int, Int64Marker> count_nonzero<T, MT extends Marker>(NDArray<T, MT> a, {int? axis, NDArray<int, Int64Marker>? out})
    # and it uses DType.int32 too.
    # So we can just replace Int64Marker with Int32Marker globally in sorting.dart!
    fix_file(
        '/usr/local/google/home/sigurdm/projects/math/pkgs/ndarray/lib/src/operations/sorting.dart',
        {
            r'Int64Marker': 'Int32Marker',
        }
    )

if __name__ == '__main__':
    main()
