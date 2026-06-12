import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('NDArray Rearranging Tests', () {
    group('flip Tests', () {
      test(
        'Basic flip 1D Float64',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
          final flipped = flip(a);
          expect(flipped.shape, [4]);
          expect(flipped.toList(), [4.0, 3.0, 2.0, 1.0]);

          // Verify it is a zero-copy view
          flipped.setCell([0], Float64(99.0));
          expect(a.getCell([3]).value, 99.0);
        }),
      );

      test(
        'Flip 2D along specific axes',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float64,
          );

          // Flip all axes
          final flippedAll = flip(a);
          expect(flippedAll.toList(), [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

          // Flip axis 0 only
          final flipped0 = flip(a, axis: 0);
          expect(flipped0.toList(), [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);

          // Flip axis 1 only
          final flipped1 = flip(a, axis: 1);
          expect(flipped1.toList(), [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);

          // Flip negative axis
          final flippedNeg = flip(a, axis: -1);
          expect(flippedNeg.toList(), [3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
        }),
      );

      test(
        'Flip 3D with list of axes',
        () => NDArray.scope(() {
          final a = NDArray.fromList(List.generate(8, (i) => i + 1), [
            2,
            2,
            2,
          ], DType.int32);
          // original:
          // [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
          // Flip axis 0 and 2
          final flipped = flip(a, axis: [0, 2]);
          // expected:
          // [[[6, 5], [8, 7]], [[2, 1], [4, 3]]]
          expect(flipped.toList(), [6, 5, 8, 7, 2, 1, 4, 3]);
        }),
      );

      test(
        'Flip invalid axis throws',
        () => NDArray.scope(() {
          final a = NDArray.create([2, 2], DType.float64);
          expect(() => flip(a, axis: 2), throwsRangeError);
          expect(() => flip(a, axis: -3), throwsRangeError);
          expect(() => flip(a, axis: 'invalid'), throwsArgumentError);
        }),
      );
    });

    group('fliplr and flipud Tests', () {
      test(
        'fliplr 2D Int32',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final flipped = fliplr(a);
          expect(flipped.toList(), [2, 1, 4, 3]);
        }),
      );

      test(
        'fliplr invalid dimensions throws',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          expect(() => fliplr(a), throwsArgumentError);
        }),
      );

      test(
        'flipud 2D Int32',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final flipped = flipud(a);
          expect(flipped.toList(), [3, 4, 1, 2]);
        }),
      );

      test(
        'flipud invalid dimensions throws',
        () => NDArray.scope(() {
          final a = NDArray.fromList([5], [], DType.int32);
          expect(() => flipud(a), throwsArgumentError);
        }),
      );
    });

    group('roll Tests', () {
      test(
        'Basic roll 1D flat',
        () => NDArray.scope(() {
          final a = NDArray.fromList([10, 20, 30, 40, 50], [5], DType.int32);

          // Roll positive shift
          final rolledPos = roll(a, 2);
          expect(rolledPos.toList(), [40, 50, 10, 20, 30]);

          // Roll negative shift
          final rolledNeg = roll(a, -2);
          expect(rolledNeg.toList(), [30, 40, 50, 10, 20]);

          // Roll larger than size
          final rolledLarge = roll(a, 7); // 7 % 5 = 2
          expect(rolledLarge.toList(), [40, 50, 10, 20, 30]);

          // Verify it is a copy (independent memory)
          rolledPos.setCell([0], Int32(99));
          expect(a.getCell([0]).value, 10);
        }),
      );

      test(
        'Roll 2D with specific axis',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [3, 3],
            DType.int32,
          );

          // Roll axis null flattens first
          final rolledFlat = roll(a, 2);
          expect(rolledFlat.shape, [3, 3]);
          expect(rolledFlat.toList(), [8, 9, 1, 2, 3, 4, 5, 6, 7]);

          // Roll axis 0
          final rolledAxis0 = roll(a, 1, axis: 0);
          expect(rolledAxis0.toList(), [7, 8, 9, 1, 2, 3, 4, 5, 6]);

          // Roll axis 1
          final rolledAxis1 = roll(a, 1, axis: 1);
          expect(rolledAxis1.toList(), [3, 1, 2, 6, 4, 5, 9, 7, 8]);
        }),
      );

      test(
        'Roll with multiple shifts and axes',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1, 2, 3, 4, 5, 6, 7, 8, 9],
            [3, 3],
            DType.int32,
          );

          // Roll list shift and list axis
          final rolledList = roll(a, [1, 1], axis: [0, 1]);
          // axis 0 shift 1: [7, 8, 9, 1, 2, 3, 4, 5, 6]
          // axis 1 shift 1: [9, 7, 8, 3, 1, 2, 6, 4, 5]
          expect(rolledList.toList(), [9, 7, 8, 3, 1, 2, 6, 4, 5]);

          // Roll single shift with multiple axes
          final rolledSingleShift = roll(a, 1, axis: [0, 1]);
          expect(rolledSingleShift.toList(), [9, 7, 8, 3, 1, 2, 6, 4, 5]);
        }),
      );

      test(
        'Roll invalid parameters throw',
        () => NDArray.scope(() {
          final a = NDArray.create([2, 2], DType.float64);
          expect(() => roll(a, [1], axis: [0, 1]), throwsArgumentError);
          expect(() => roll(a, [1, 2], axis: [0]), throwsArgumentError);
          expect(() => roll(a, 'invalid', axis: 0), throwsArgumentError);
          expect(() => roll(a, 1, axis: 'invalid'), throwsArgumentError);
          expect(() => roll(a, 1, axis: 2), throwsRangeError);
        }),
      );
    });

    group('Extension methods Tests', () {
      test(
        'All rearranging extensions',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);

          expect(flip(a).toList(), [4, 3, 2, 1]);
          expect(fliplr(a).toList(), [2, 1, 4, 3]);
          expect(flipud(a).toList(), [3, 4, 1, 2]);
          expect(roll(a, 1, axis: 0).toList(), [3, 4, 1, 2]);
        }),
      );
    });
  });
}
