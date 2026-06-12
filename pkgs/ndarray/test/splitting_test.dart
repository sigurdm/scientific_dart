import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Splitting Tests', () {
    group('split & array_split Tests', () {
      test('basic 1D equal split', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
          final splits = split(a, 2);

          expect(splits.length, 2);
          expect(splits[0].shape, [2]);
          expect(splits[0].toList(), [1, 2]);
          expect(splits[1].shape, [2]);
          expect(splits[1].toList(), [3, 4]);

          // Modifying sub-array view affects original
          splits[0].setCell([0], 99);
          expect(a.getCell([0]), 99);
        });
      });

      test('split unequal division throws', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          expect(() => split(a, 2), throwsArgumentError);
        });
      });

      test('array_split unequal division splits as equal as possible', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final splits = array_split(a, 2);

          expect(splits.length, 2);
          expect(splits[0].shape, [2]);
          expect(splits[0].toList(), [1, 2]);
          expect(splits[1].shape, [1]);
          expect(splits[1].toList(), [3]);
        });
      });

      test('split_at with list of indices', () {
        NDArray.scope(() {
          final a = NDArray.fromList([10, 20, 30, 40, 50], [5], DType.int32);
          final splits = split_at(a, [1, 3]);

          expect(splits.length, 3);
          expect(splits[0].toList(), [10]);
          expect(splits[1].toList(), [20, 30]);
          expect(splits[2].toList(), [40, 50]);
        });
      });

      test('split along specific axis', () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [2, 4],
            DType.int32,
          );

          // Split columns (axis 1) into 2 equal parts
          final splits = split(a, 2, axis: 1);
          expect(splits.length, 2);
          expect(splits[0].toList(), [1, 2, 5, 6]);
          expect(splits[1].toList(), [3, 4, 7, 8]);
        });
      });
    });

    group('hsplit & hsplit_at Tests', () {
      test('hsplit 2D array', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final splits = hsplit(a, 2);

          expect(splits.length, 2);
          expect(splits[0].toList(), [1, 3]);
          expect(splits[1].toList(), [2, 4]);
        });
      });

      test('hsplit_at 2D array', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final splits = hsplit_at(a, [1]);

          expect(splits.length, 2);
          expect(splits[0].toList(), [1, 3]);
          expect(splits[1].toList(), [2, 4]);
        });
      });

      test('hsplit 1D array splits along axis 0', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
          final splits = hsplit(a, 2);

          expect(splits.length, 2);
          expect(splits[0].toList(), [1, 2]);
          expect(splits[1].toList(), [3, 4]);
        });
      });
    });

    group('vsplit & vsplit_at Tests', () {
      test('vsplit 2D array', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final splits = vsplit(a, 2);

          expect(splits.length, 2);
          expect(splits[0].toList(), [1, 2]);
          expect(splits[1].toList(), [3, 4]);
        });
      });

      test('vsplit_at 2D array', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final splits = vsplit_at(a, [1]);

          expect(splits.length, 2);
          expect(splits[0].toList(), [1, 2]);
          expect(splits[1].toList(), [3, 4]);
        });
      });

      test('vsplit 1D array throws', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          expect(() => vsplit(a, 2), throwsArgumentError);
        });
      });
    });

    group('dsplit & dsplit_at Tests', () {
      test('dsplit 3D array equal split', () {
        NDArray.scope(() {
          final a = NDArray.fromList(List.generate(16, (i) => i + 1), [
            2,
            2,
            4,
          ], DType.int32);
          final splits = dsplit(a, 2);

          expect(splits.length, 2);
          expect(splits[0].shape, [2, 2, 2]);
          expect(splits[0].toList(), [1, 2, 5, 6, 9, 10, 13, 14]);
          expect(splits[1].shape, [2, 2, 2]);
          expect(splits[1].toList(), [3, 4, 7, 8, 11, 12, 15, 16]);

          // Zero-copy check: mutating sub-array affects original
          splits[0].setCell([0, 0, 0], 99);
          expect(a.getCell([0, 0, 0]), 99);
        });
      });

      test('dsplit_at 3D array at indices', () {
        NDArray.scope(() {
          final a = NDArray.fromList(List.generate(16, (i) => i + 1), [
            2,
            2,
            4,
          ], DType.int32);
          final splits = dsplit_at(a, [1, 3]);

          expect(splits.length, 3);
          expect(splits[0].shape, [2, 2, 1]);
          expect(splits[0].toList(), [1, 5, 9, 13]);
          expect(splits[1].shape, [2, 2, 2]);
          expect(splits[1].toList(), [2, 3, 6, 7, 10, 11, 14, 15]);
          expect(splits[2].shape, [2, 2, 1]);
          expect(splits[2].toList(), [4, 8, 12, 16]);
        });
      });

      test('dsplit rank < 3 throws', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          expect(() => dsplit(a, 2), throwsArgumentError);
        });
      });

      test('dsplit_at rank < 3 throws', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          expect(() => dsplit_at(a, [1]), throwsArgumentError);
        });
      });

      test('dsplit unequal division throws', () {
        NDArray.scope(() {
          final a = NDArray.fromList(List.generate(12, (i) => i + 1), [
            2,
            2,
            3,
          ], DType.int32);
          expect(() => dsplit(a, 2), throwsArgumentError);
        });
      });

      test('dsplit disposed array throws', () {
        NDArray.scope(() {
          final a = NDArray.fromList(List.generate(16, (i) => i + 1), [
            2,
            2,
            4,
          ], DType.int32);
          a.dispose();
          expect(() => dsplit(a, 2), throwsStateError);
          expect(() => dsplit_at(a, [1]), throwsStateError);
        });
      });
    });
  });
}
