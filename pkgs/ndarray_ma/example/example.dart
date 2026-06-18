import 'package:ndarray/ndarray.dart';
import 'package:ndarray_ma/ndarray_ma.dart';

void main() {
  // Always use NDArray.scope for memory management when working with arrays
  NDArray.scope(() {
    print('--- MaskedArray Example ---');

    // 1. Creation
    final data = NDArray.fromList(
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
      [2, 3],
      DType.float64,
    );
    final mask = NDArray.fromList(
      [false, true, false, true, false, false],
      [2, 3],
      DType.boolean,
    );

    final marr = MaskedArray(data, mask, fillValue: -999.0);
    print('Original MaskedArray:');
    print(marr.filled().toList()); // Prints array with masked elements filled

    // 2. Element Access
    print('\nElement at [0, 0] (unmasked): ${marr[[0, 0]]}');
    print('Element at [0, 1] (masked): ${marr[[0, 1]]}'); // Should be null

    // 3. Slicing
    final slice = marr[0]; // View of row 0
    print('\nSlice (row 0):');
    print(slice.filled().toList());

    // 4. Arithmetic (Bypasses masked elements and propagates masks)
    final other = MaskedArray.ones([2, 3], DType.float64).multiply(10.0);
    final sum = marr.add(other);
    print('\nSum (marr + 10.0):');
    print(sum.filled().toList());

    // 5. Reductions (Ignore masked elements)
    print('\nReductions:');
    print('Sum along axis 0 (columns): ${sum.sum(axis: 0).filled().toList()}');
    print('Mean (overall): ${sum.mean().scalar}');

    // 6. Compression
    print('\nCompressed (unmasked elements only):');
    print(marr.compressed().toList());
  });
}
