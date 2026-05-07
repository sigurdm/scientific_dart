import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('NDArray Boolean Data Type (DType.boolean) Tests', () {
    test('Create empty boolean array via zeros factory', () => NDArray.scope(() {
      final arr = NDArray<bool>.zeros([2, 3], DType.boolean);

      expect(arr.shape, [2, 3]);
      expect(arr.dtype, DType.boolean);
      expect(arr.data.length, 6);

      // Check true default zeros allocation (false)
      expect(arr.data, [false, false, false, false, false, false]);
    }));

    test('Mutating boolean values via BoolList indexing operators',
        () => NDArray.scope(() {
      final arr = NDArray<bool>.zeros([4], DType.boolean);

      arr.data[0] = true;
      arr.data[2] = true;

      expect(arr.data[0], true);
      expect(arr.data[1], false);
      expect(arr.data[2], true);
      expect(arr.data[3], false);

      expect(arr.toList(), [true, false, true, false]);
    }));

    test('Create boolean array filled with true via ones factory',
        () => NDArray.scope(() {
      final arr = NDArray<bool>.ones([5], DType.boolean);

      expect(arr.shape, [5]);
      expect(arr.dtype, DType.boolean);
      expect(arr.toList(), [true, true, true, true, true]);
    }));

    test('Boolean fromList contiguous block copying', () => NDArray.scope(() {
      final input = [true, false, false, true];
      final arr = NDArray.fromList(input, [2, 2], DType.boolean);

      expect(arr.shape, [2, 2]);
      expect(arr.dtype, DType.boolean);
      expect(arr.toList(), [true, false, false, true]);
    }));

    test('Zero-copy boolean sliced views memory sharing',
        () => NDArray.scope(() {
      final parent = NDArray.fromList(
        [true, false, true, false],
        [4],
        DType.boolean,
      );

      // Slice a view on the latter elements (index 2 to 4 -> [true, false])
      final view = parent.slice([Slice(start: 2, stop: 4)]);

      expect(view.shape, [2]);
      expect(view.dtype, DType.boolean);
      expect(view.toList(), [true, false]);

      // Mutate via the view and check if parent reflects it instantly!
      view.data[1] = true; // view -> [true, true]

      expect(parent.toList(), [true, false, true, true]);
      // Success! Confirms zero-copy C heap pointer sharing works flawlessly on boolean bytes!
    }));
  });
}
