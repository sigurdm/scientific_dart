import 'package:ndarray/ndarray.dart';

void main() {
  NDArray.scope(() {
    print('=== unique ===');
    final a = NDArray<DTypeTag>.fromList([1, 2, 2, 3, 1, 4], [6], DType.int32);
    final u = unique(a);
    print('Input: ${a.toList()}');
    print('Unique: ${u.toList()}'); // [1, 2, 3, 4]

    print('\n=== intersect1d ===');
    final ar1 = NDArray<DTypeTag>.fromList([1, 3, 4, 3], [4], DType.int32);
    final ar2 = NDArray<DTypeTag>.fromList([3, 1, 2, 1], [4], DType.int32);
    final intersection = intersect1d(ar1, ar2);
    print('Array 1: ${ar1.toList()}');
    print('Array 2: ${ar2.toList()}');
    print('Intersection: ${intersection.toList()}'); // [1, 3]

    print('\n=== setdiff1d ===');
    final sd1 = NDArray<DTypeTag>.fromList([1, 2, 3, 2, 4], [5], DType.int32);
    final sd2 = NDArray<DTypeTag>.fromList([2, 3, 5], [3], DType.int32);
    final diff = setdiff1d(sd1, sd2);
    print('Array 1: ${sd1.toList()}');
    print('Array 2: ${sd2.toList()}');
    print('Difference (1 - 2): ${diff.toList()}'); // [1, 4]

    print('\n=== setxor1d ===');
    final sx1 = NDArray<DTypeTag>.fromList([1, 2, 3], [3], DType.int32);
    final sx2 = NDArray<DTypeTag>.fromList([2, 3, 4], [3], DType.int32);
    final xor = setxor1d(sx1, sx2);
    print('Array 1: ${sx1.toList()}');
    print('Array 2: ${sx2.toList()}');
    print('XOR: ${xor.toList()}'); // [1, 4]

    print('\n=== union1d ===');
    final un1 = NDArray<DTypeTag>.fromList([1, 2, 3], [3], DType.int32);
    final un2 = NDArray<DTypeTag>.fromList([2, 3, 4, 5], [4], DType.int32);
    final union = union1d(un1, un2);
    print('Array 1: ${un1.toList()}');
    print('Array 2: ${un2.toList()}');
    print('Union: ${union.toList()}'); // [1, 2, 3, 4, 5]

    print('\n=== isin ===');
    final element = NDArray<DTypeTag>.fromList(
      [1, 2, 3, 4, 2, 1],
      [6],
      DType.int32,
    );
    final testElements = NDArray<DTypeTag>.fromList([2, 4], [2], DType.int32);
    final mask = isin(element, testElements);
    print('Element: ${element.toList()}');
    print('Test Elements: ${testElements.toList()}');
    print('Is In: ${mask.toList()}'); // [false, true, false, true, true, false]
  });
}
