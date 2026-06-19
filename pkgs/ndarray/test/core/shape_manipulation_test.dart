import 'dart:math' as math;
import 'dart:typed_data';
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

List<int> rep(dynamic v) {
  if (v is int) return [v];
  if (v is List<int>) return v;
  throw ArgumentError();
}

void main() {
  group('Shape Manipulation Core Tests', () {
    group('Shape Manipulation Tests', () {
      group('expandDims Tests', () {
        test(
          'Basic expandDims at front (axis 0)',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final b = a.expandDims(0);
            expect(b.shape, [1, 2, 2]);
            expect(b.strides, [2, 2, 1]); // Strides are correct for [1, 2, 2]
            expect(b.toList(), [1.0, 2.0, 3.0, 4.0]);

            // Verify it shares memory (view behavior)
            b.data[0] = Float64(99.0);
            expect(a.data[0], 99.0);
          }),
        );

        test(
          'expandDims in the middle (axis 1)',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final b = a.expandDims(1);
            expect(b.shape, [2, 1, 2]);
            expect(b.strides, [2, 1, 1]);
            expect(b.toList(), [1.0, 2.0, 3.0, 4.0]);
          }),
        );

        test(
          'expandDims at the end (axis 2)',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final b = a.expandDims(2);
            expect(b.shape, [2, 2, 1]);
            expect(b.strides, [2, 1, 1]);
            expect(b.toList(), [1.0, 2.0, 3.0, 4.0]);
          }),
        );

        test(
          'expandDims negative axis (-1)',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            // -1 refers to the position right after the last dimension (equivalent to axis 2)
            final b = a.expandDims(-1);
            expect(b.shape, [2, 2, 1]);
          }),
        );

        test(
          'expandDims negative axis (-3)',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            // -3 refers to the very front (equivalent to axis 0)
            final b = a.expandDims(-3);
            expect(b.shape, [1, 2, 2]);
          }),
        );

        test(
          'expandDims invalid axis throws',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 2], DType.float64);
            expect(() => a.expandDims(3), throwsRangeError);
            expect(() => a.expandDims(-4), throwsRangeError);
          }),
        );
      });

      group('squeeze Tests', () {
        test(
          'Squeeze all size 1 axes',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              1,
              2,
              1,
            ], DType.float64);
            final b = a.squeeze();
            expect(b.shape, [2]);
            expect(b.strides, [1]);
            expect(b.toList(), [1.0, 2.0]);

            // View verification
            b.data[0] = Float64(42.0);
            expect(a.data[0], 42.0);
          }),
        );

        test(
          'Squeeze specific int axis',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              1,
              2,
              1,
            ], DType.float64);
            final b = a.squeeze(axis: 0);
            expect(b.shape, [2, 1]);
            expect(b.strides, [1, 1]);
          }),
        );

        test(
          'Squeeze specific negative axis',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              1,
              2,
              1,
            ], DType.float64);
            final b = a.squeeze(axis: -1);
            expect(b.shape, [1, 2]);
            expect(b.strides, [2, 1]);
          }),
        );

        test(
          'Squeeze specific List<int> axes',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              1,
              2,
              1,
            ], DType.float64);
            final b = a.squeeze(axis: [0, 2]);
            expect(b.shape, [2]);
            expect(b.strides, [1]);
          }),
        );

        test(
          'Squeeze axis with size > 1 throws',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              1,
              2,
              1,
            ], DType.float64);
            expect(() => a.squeeze(axis: 1), throwsArgumentError);
          }),
        );

        test(
          'Squeeze invalid axis range throws',
          () => NDArray.scope(() {
            final a = NDArray.create([1, 2], DType.float64);
            expect(() => a.squeeze(axis: 2), throwsRangeError);
            expect(() => a.squeeze(axis: -3), throwsRangeError);
          }),
        );

        test(
          'Squeeze 1x1 array to 0-D scalar array',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([5.5]), [
              1,
              1,
            ], DType.float64);
            final b = a.squeeze();
            expect(b.shape, isEmpty);
            expect(b.strides, isEmpty);
            expect(b[[]], 5.5);
          }),
        );
      });

      group('swapaxes Tests', () {
        test(
          'Basic swapaxes in 2D',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
              [2, 3],
              DType.float64,
            );
            // shape: [2, 3], strides: [3, 1]
            final b = a.swapaxes(0, 1);
            expect(b.shape, [3, 2]);
            expect(b.strides, [1, 3]);
            // Original elements:
            // Row 0: 1, 2, 3
            // Row 1: 4, 5, 6
            // Swapped matrix:
            // [1, 4]
            // [2, 5]
            // [3, 6]
            expect(b.toList(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

            // View verification
            b.data[0] = Float64(11.0); // updates a.data[0]
            expect(a.data[0], 11.0);
          }),
        );

        test(
          'swapaxes negative index',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 3, 4], DType.float64);
            final b = a.swapaxes(-3, -1); // swaps 0 and 2
            expect(b.shape, [4, 3, 2]);
          }),
        );

        test(
          'swapaxes same axis is no-op view',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              2,
            ], DType.float64);
            final b = a.swapaxes(0, 0);
            expect(b.shape, [2]);
            expect(b.toList(), [1.0, 2.0]);
          }),
        );

        test(
          'swapaxes out of bounds throws',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 2], DType.float64);
            expect(() => a.swapaxes(0, 2), throwsRangeError);
            expect(() => a.swapaxes(-3, 1), throwsRangeError);
          }),
        );
      });

      group('moveaxis Tests', () {
        test(
          'Move single axis 0 to 2 in 3D',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 3, 4], DType.float64);
            final b = a.moveaxis(0, 2);
            expect(b.shape, [3, 4, 2]);
            // original strides were [12, 4, 1]
            // order becomes [1, 2, 0], so strides should be [4, 1, 12]
            expect(b.strides, [4, 1, 12]);
          }),
        );

        test(
          'Move single axis negative indexes (-1 to 0)',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 3, 4], DType.float64);
            final b = a.moveaxis(-1, 0); // moves axis 2 to 0
            expect(b.shape, [4, 2, 3]);
            expect(b.strides, [1, 12, 4]);
          }),
        );

        test(
          'Move multiple axes lists',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 3, 4], DType.float64);
            // Move axis 0 -> 1 and axis 1 -> 2
            final b = a.moveaxis([0, 1], [1, 2]);
            expect(b.shape, [
              4,
              2,
              3,
            ]); // axis 2 becomes 0, axis 0 becomes 1, axis 1 becomes 2
          }),
        );

        test(
          'moveaxis invalid lengths or duplicates throw',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 3, 4], DType.float64);
            expect(() => a.moveaxis([0, 1], [2]), throwsArgumentError);
            expect(() => a.moveaxis([0, 0], [1, 2]), throwsArgumentError);
            expect(() => a.moveaxis([0, 1], [2, 2]), throwsArgumentError);
          }),
        );
      });

      group('tile Tests', () {
        test(
          'Tile 1D array with int scalar reps',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              2,
            ], DType.float64);
            final b = tile(a, rep(3));
            expect(b.shape, [6]);
            expect(b.toList(), [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

            // Verify it allocates new memory (copy behavior)
            b.data[0] = Float64(99.0);
            expect(a.data[0], 1.0);
          }),
        );

        test(
          'Tile 1D array with 1D List reps',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              2,
            ], DType.float64);
            final b = tile(a, rep([2]));
            expect(b.shape, [4]);
            expect(b.toList(), [1.0, 2.0, 1.0, 2.0]);
          }),
        );

        test(
          'Tile 1D array with higher-rank 2D List reps',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
              2,
            ], DType.float64);
            final b = tile(
              a,
              rep([2, 3]),
            ); // shape promoted from [2] to [1, 2], target [2*1, 3*2] = [2, 6]
            expect(b.shape, [2, 6]);
            expect(b.toList(), [
              1.0, 2.0, 1.0, 2.0, 1.0, 2.0, // row 0
              1.0, 2.0, 1.0, 2.0, 1.0, 2.0, // row 1
            ]);
          }),
        );

        test(
          'Tile 2D array with list reps',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final b = tile(
              a,
              rep([2, 1]),
            ); // repeat rows 2 times, columns 1 time
            expect(b.shape, [4, 2]);
            expect(b.toList(), [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
          }),
        );

        test(
          'Tile with negative or invalid reps throws',
          () => NDArray.scope(() {
            final a = NDArray.create([2], DType.float64);
            expect(() => tile(a, rep(-1)), throwsArgumentError);
            expect(() => tile(a, rep([2, -3])), throwsArgumentError);
            expect(
              () => tile(a, 'invalid' as dynamic),
              throwsA(isA<TypeError>()),
            );
            expect(() => tile(a, 3.5 as dynamic), throwsA(isA<TypeError>()));
          }),
        );
      });

      group('repeat Tests', () {
        test(
          'Repeat elements of flat 1D array with scalar int',
          () => NDArray.scope(() {
            final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
              3,
            ], DType.float64);
            final b = repeat(a, rep(2));
            expect(b.shape, [6]);
            expect(b.toList(), [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);

            // Copy verification
            b.data[0] = Float64(99.0);
            expect(a.data[0], 1.0);
          }),
        );

        test(
          'Repeat automatically flattens high-rank array if axis is null',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final b = repeat(a, rep(2)); // axis = null
            expect(b.shape, [8]);
            expect(b.toList(), [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
          }),
        );

        test(
          'Repeat along axis 0 with scalar int',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            final b = repeat(a, rep(2), axis: 0);
            expect(b.shape, [4, 2]);
            expect(b.toList(), [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
          }),
        );

        test(
          'Repeat along axis 1 with custom List<int> counts',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            // Repeat column 0 -> 2 times, column 1 -> 1 time. total cols = 3
            final b = repeat(a, rep([2, 1]), axis: 1);
            expect(b.shape, [2, 3]);
            expect(b.toList(), [
              1.0, 1.0, 2.0, // row 0
              3.0, 3.0, 4.0, // row 1
            ]);
          }),
        );

        test(
          'Repeat with an entry of 0 counts (deletes slice)',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
              [2, 2],
              DType.float64,
            );
            // Repeat column 0 -> 0 times, column 1 -> 2 times.
            final b = repeat(a, rep([0, 2]), axis: 1);
            expect(b.shape, [2, 2]);
            expect(b.toList(), [2.0, 2.0, 4.0, 4.0]);
          }),
        );

        test(
          'Repeat mismatched list counts throws',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 2], DType.float64);
            expect(
              () => repeat(a, rep([1, 2, 3]), axis: 1),
              throwsArgumentError,
            );
          }),
        );

        test(
          'Repeat negative counts throws',
          () => NDArray.scope(() {
            final a = NDArray.create([3], DType.float64);
            expect(() => repeat(a, rep(-2)), throwsArgumentError);
            expect(() => repeat(a, rep([1, -1, 0])), throwsArgumentError);
          }),
        );

        test(
          'Repeat invalid axis throws',
          () => NDArray.scope(() {
            final a = NDArray.create([2, 2], DType.float64);
            expect(() => repeat(a, rep(2), axis: 2), throwsRangeError);
          }),
        );
      });

      group('NDArray deep duplicate copy() Tests', () {
        test(
          'copy() contiguous float64 array',
          () => NDArray.scope(() {
            final a = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
            final b = a.copy();

            expect(b.shape, [3]);
            expect(b.dtype, DType.float64);
            expect(b.toList(), [10.0, 20.0, 30.0]);

            b.data[0] = Float64(99.0);
            expect(a.data[0], 10.0);
          }),
        );

        test(
          'copy() non-contiguous transposed strided float64 array view',
          () => NDArray.scope(() {
            final parent = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );

            final view = parent.transposed;
            expect(view.isContiguous, false);

            final b = view.copy();
            expect(b.shape, [2, 2]);
            expect(b.isContiguous, true);
            expect(b.toList(), [1.0, 3.0, 2.0, 4.0]);

            b.data[0] = Float64(99.0);
            expect(parent.data[0], 1.0);
          }),
        );

        test(
          'disposed array throws StateError',
          () => NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            a.dispose();
            expect(() => a.copy(), throwsStateError);
          }),
        );

        test(
          'top-level copy() contiguous float64 array',
          () => NDArray.scope(() {
            final a = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
            final b = copy(a);

            expect(b.shape, [3]);
            expect(b.dtype, DType.float64);
            expect(b.toList(), [10.0, 20.0, 30.0]);

            b.data[0] = Float64(99.0);
            expect(a.data[0], 10.0);
          }),
        );

        test(
          'top-level copy() disposed array throws StateError',
          () => NDArray.scope(() {
            final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            a.dispose();
            expect(() => copy(a), throwsStateError);
          }),
        );
      });
    });

    group('Rearranging Tests', () {
      group('flip Tests', () {
        test(
          'Basic flip 1D Float64',
          () => NDArray.scope(() {
            final a = NDArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [4],
              DType.float64,
            );
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
            splits[0].setCell([0], Int32(99));
            expect(a.getCell([0]).value, 99);
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
            splits[0].setCell([0, 0, 0], Int32(99));
            expect(a.getCell([0, 0, 0]).value, 99);
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

    group('Stack Tests', () {
      test(
        'stack() basic 1D arrays to 2D along axis 0',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          final b = NDArray.fromList([3, 4], [2], DType.int32);

          final s0 = stack([a, b], axis: 0);
          expect(s0.shape, [2, 2]);
          expect(s0.dtype, DType.int32);
          expect(s0.toList(), [1, 2, 3, 4]);
        }),
      );

      test(
        'stack() basic 1D arrays to 2D along axis 1',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          final b = NDArray.fromList([3, 4], [2], DType.int32);

          final s1 = stack([a, b], axis: 1);
          expect(s1.shape, [2, 2]);
          expect(s1.toList(), [1, 3, 2, 4]);
        }),
      );

      test(
        'stack() multidimensional 2D matrices checks',
        () => NDArray.scope(() {
          // two matrices of shape [2, 3]
          final a = NDArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], DType.int32);
          final b = NDArray.fromList(
            [10, 20, 30, 40, 50, 60],
            [2, 3],
            DType.int32,
          );

          // Stacks into shape [2, 2, 3] along axis 0
          final s0 = stack([a, b], axis: 0);
          expect(s0.shape, [2, 2, 3]);
          expect(s0.toList(), [1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60]);

          // Stacks into shape [2, 2, 3] along axis 1
          final s1 = stack([a, b], axis: 1);
          expect(s1.shape, [2, 2, 3]);
          expect(s1.toList(), [1, 2, 3, 10, 20, 30, 4, 5, 6, 40, 50, 60]);
        }),
      );

      test(
        'stack() resolved negative target axis',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          final b = NDArray.fromList([3, 4], [2], DType.int32);

          final sNeg = stack([a, b], axis: -1); // targets axis 1
          expect(sNeg.shape, [2, 2]);
          expect(sNeg.toList(), [1, 3, 2, 4]);
        }),
      );

      test(
        'stack() validation errors throws exceptions',
        () => NDArray.scope(() {
          // Empty list throws ArgumentError
          expect(() => stack(<NDArray<Object>>[]), throwsArgumentError);

          final a = NDArray.fromList([1, 2], [2], DType.int32);
          final wrongShape = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final wrongDType = NDArray.fromList(
            Float64List.fromList([1.0, 2.0]),
            [2],
            DType.float64,
          );

          // Mismatch shape/DType throws ArgumentError
          expect(() => stack([a, wrongShape]), throwsArgumentError);
          expect(() => stack([a, wrongDType]), throwsArgumentError);

          // Out of bounds axis throws RangeError
          expect(() => stack([a, a], axis: 5), throwsRangeError);
          expect(() => stack([a, a], axis: -5), throwsRangeError);
        }),
      );

      test(
        'disposed arrays throw StateError',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          a.dispose();
          expect(() => stack([a, a]), throwsStateError);
        }),
      );
    });

    group('Repeating & Tiling Tests', () {
      group('repeat Tests', () {
        test('1D array, scalar repeats', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
            final r = repeat(a, rep(2));
            expect(r.shape, [6]);
            expect(r.toList(), [1, 1, 2, 2, 3, 3]);
          });
        });

        test('1D array, list repeats', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
            final r = repeat(a, rep([1, 2, 0]));
            expect(r.shape, [3]);
            expect(r.toList(), [1, 2, 2]);
          });
        });

        test('2D array, axis = 0', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final r = repeat(a, rep(2), axis: 0);
            expect(r.shape, [4, 2]);
            expect(r.toList(), [1, 2, 1, 2, 3, 4, 3, 4]);
          });
        });

        test('2D array, axis = 1', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final r = repeat(a, rep([2, 1]), axis: 1);
            expect(r.shape, [2, 3]);
            expect(r.toList(), [1, 1, 2, 3, 3, 4]);
          });
        });

        test('2D array, negative axis', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final r = repeat(a, rep(2), axis: -1);
            expect(r.shape, [2, 4]);
            expect(r.toList(), [1, 1, 2, 2, 3, 3, 4, 4]);
          });
        });

        test('2D array, axis = null (flatten)', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final r = repeat(a, rep(2));
            expect(r.shape, [8]);
            expect(r.toList(), [1, 1, 2, 2, 3, 3, 4, 4]);
          });
        });

        test('3D array, repeat along axis 1', () {
          NDArray.scope(() {
            final a = NDArray.fromList(List.generate(8, (i) => i + 1), [
              2,
              2,
              2,
            ], DType.int32);
            final r = repeat(a, rep([2, 1]), axis: 1);
            expect(r.shape, [2, 3, 2]);
            expect(r.toList(), [
              1, 2, 1, 2, 3, 4, // block 0
              5, 6, 5, 6, 7, 8, // block 1
            ]);
          });
        });

        test('Edge case: repeats with 0 (deletes elements)', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
            final r = repeat(a, rep(0));
            expect(r.shape, [0]);
            expect(r.toList(), <int>[]);
          });
        });

        test('Precondition checks', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
            expect(() => repeat(a, rep([-1, 1, 1])), throwsArgumentError);
            expect(() => repeat(a, rep([1, -1, 1])), throwsArgumentError);
            expect(() => repeat(a, rep([1, 2])), throwsArgumentError);
            expect(() => repeat(a, rep(2), axis: 2), throwsRangeError);
            expect(() => repeat(a, rep(2), axis: -2), throwsRangeError);
          });
        });

        test('out parameter: correct shape/dtype', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
            final out = NDArray<int>.create([6], DType.int32);
            final r = repeat(a, rep(2), out: out);
            expect(identical(r, out), true);
            expect(r.toList(), [1, 1, 2, 2, 3, 3]);
          });
        });

        test('out parameter: incorrect shape/dtype', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
            final outWrongShape = NDArray<int>.create([5], DType.int32);
            final outWrongDType = NDArray<double>.create([6], DType.float64);

            expect(
              () => repeat(a, rep(2), out: outWrongShape),
              throwsArgumentError,
            );
            expect(
              () => repeat(a, rep(2), out: outWrongDType),
              throwsArgumentError,
            );
          });
        });

        test('Different dtypes (float64, complex128, boolean)', () {
          NDArray.scope(() {
            final f = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            final rF = repeat(f, rep(2));
            expect(rF.dtype, DType.float64);
            expect(rF.toList(), [1.0, 1.0, 2.0, 2.0]);

            final c = NDArray<Complex>.fromList(
              [Complex(1, 2), Complex(3, 4)],
              [2],
              DType.complex128,
            );
            final rC = repeat(c, rep(2));
            expect(rC.dtype, DType.complex128);
            expect(rC.toList(), [
              Complex(1, 2),
              Complex(1, 2),
              Complex(3, 4),
              Complex(3, 4),
            ]);

            final b = NDArray.fromList([true, false], [2], DType.boolean);
            final rB = repeat(b, rep(2));
            expect(rB.dtype, DType.boolean);
            expect(rB.toList(), [true, true, false, false]);
          });
        });

        test('Non-contiguous strided view input', () {
          NDArray.scope(() {
            final parent = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final view = parent.transposed; // [[1, 3], [2, 4]]
            expect(view.isContiguous, false);

            final r = repeat(view, rep(2), axis: 0);
            expect(r.shape, [4, 2]);
            expect(r.toList(), [1, 3, 1, 3, 2, 4, 2, 4]);
          });
        });
      });

      group('tile Tests', () {
        test('1D array, scalar reps', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2], [2], DType.int32);
            final t = tile(a, rep(3));
            expect(t.shape, [6]);
            expect(t.toList(), [1, 2, 1, 2, 1, 2]);
          });
        });

        test('1D array, list reps', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2], [2], DType.int32);
            final t = tile(a, rep([3]));
            expect(t.shape, [6]);
            expect(t.toList(), [1, 2, 1, 2, 1, 2]);
          });
        });

        test('2D array, reps of same rank', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final t = tile(a, rep([2, 3]));
            expect(t.shape, [4, 6]);
            expect(t.toList(), [
              1,
              2,
              1,
              2,
              1,
              2,
              3,
              4,
              3,
              4,
              3,
              4,
              1,
              2,
              1,
              2,
              1,
              2,
              3,
              4,
              3,
              4,
              3,
              4,
            ]);
          });
        });

        test('2D array, reps with prepended 1s (reps.length < rank)', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final t = tile(a, rep([2])); // treated as [1, 2]
            expect(t.shape, [2, 4]);
            expect(t.toList(), [1, 2, 1, 2, 3, 4, 3, 4]);
          });
        });

        test('2D array, reps with expanded rank (reps.length > rank)', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final t = tile(a, rep([2, 1, 2])); // a is promoted to [1, 2, 2]
            expect(t.shape, [2, 2, 4]);
            expect(t.toList(), [
              1,
              2,
              1,
              2,
              3,
              4,
              3,
              4,
              1,
              2,
              1,
              2,
              3,
              4,
              3,
              4,
            ]);
          });
        });

        test('Edge case: reps with 0 (returns empty array)', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2], [2], DType.int32);
            final t = tile(a, rep(0));
            expect(t.shape, [0]);
            expect(t.toList(), <int>[]);
          });
        });

        test('Precondition checks', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2], [2], DType.int32);
            expect(() => tile(a, rep([-1])), throwsArgumentError);
            expect(() => tile(a, rep([1, -1])), throwsArgumentError);
          });
        });

        test('out parameter: correct shape/dtype', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2], [2], DType.int32);
            final out = NDArray<int>.create([4], DType.int32);
            final t = tile(a, rep(2), out: out);
            expect(identical(t, out), true);
            expect(t.toList(), [1, 2, 1, 2]);
          });
        });

        test('out parameter: incorrect shape/dtype', () {
          NDArray.scope(() {
            final a = NDArray.fromList([1, 2], [2], DType.int32);
            final outWrongShape = NDArray<int>.create([5], DType.int32);
            final outWrongDType = NDArray<double>.create([4], DType.float64);

            expect(
              () => tile(a, rep(2), out: outWrongShape),
              throwsArgumentError,
            );
            expect(
              () => tile(a, rep(2), out: outWrongDType),
              throwsArgumentError,
            );
          });
        });

        test('Different dtypes (float64, complex128, boolean)', () {
          NDArray.scope(() {
            final f = NDArray.fromList([1.0, 2.0], [2], DType.float64);
            final tF = tile(f, rep(2));
            expect(tF.dtype, DType.float64);
            expect(tF.toList(), [1.0, 2.0, 1.0, 2.0]);

            final c = NDArray<Complex>.fromList(
              [Complex(1, 2), Complex(3, 4)],
              [2],
              DType.complex128,
            );
            final tC = tile(c, rep(2));
            expect(tC.dtype, DType.complex128);
            expect(tC.toList(), [
              Complex(1, 2),
              Complex(3, 4),
              Complex(1, 2),
              Complex(3, 4),
            ]);

            final b = NDArray.fromList([true, false], [2], DType.boolean);
            final tB = tile(b, rep(2));
            expect(tB.dtype, DType.boolean);
            expect(tB.toList(), [true, false, true, false]);
          });
        });

        test('Non-contiguous strided view input', () {
          NDArray.scope(() {
            final parent = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
            final view = parent.transposed; // [[1, 3], [2, 4]]
            expect(view.isContiguous, false);

            final t = tile(view, rep([2, 1])); // tile rows 2x, cols 1x
            expect(t.shape, [4, 2]);
            expect(t.toList(), [1, 3, 2, 4, 1, 3, 2, 4]);
          });
        });
      });
    });

    group('Padding Tests', () {
      test(
        'Constant Padding - 1D - Uniform',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = pad(
            a,
            PadWidth.all(2),
            mode: PaddingMode.constant,
            constantValues: PadValues.all(9),
          );
          expect(r.shape, [7]);
          expect(r.toList(), [9, 9, 1, 2, 3, 9, 9]);
        }),
      );

      test(
        'Constant Padding - 2D - Per Axis',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final r = pad(
            a,
            PadWidth.axes([(1, 1), (2, 2)]),
            mode: PaddingMode.constant,
            constantValues: PadValues.axes([(8, 8), (9, 9)]),
          );
          expect(r.shape, [4, 6]);
          expect(r.toList(), [
            9,
            9,
            8,
            8,
            9,
            9,
            9,
            9,
            1,
            2,
            9,
            9,
            9,
            9,
            3,
            4,
            9,
            9,
            9,
            9,
            8,
            8,
            9,
            9,
          ]);
        }),
      );

      test(
        'Constant Mode - 2D Int with different before/after',
        () => NDArray.scope(() {
          final arr = NDArray<int>.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final padded = pad(
            arr,
            PadWidth.axes([(1, 2), (2, 1)]),
            mode: PaddingMode.constant,
            constantValues: PadValues.axes([(10, 20), (30, 40)]),
          );
          expect(padded.shape, [5, 5]);
          expect(padded.toList(), [
            30,
            30,
            10,
            10,
            40,
            30,
            30,
            1,
            2,
            40,
            30,
            30,
            3,
            4,
            40,
            30,
            30,
            20,
            20,
            40,
            30,
            30,
            20,
            20,
            40,
          ]);
        }),
      );

      test(
        'Edge Padding - 1D',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = pad(a, PadWidth.axes([(2, 3)]), mode: PaddingMode.edge);
          expect(r.shape, [8]);
          expect(r.toList(), [1, 1, 1, 2, 3, 3, 3, 3]);
        }),
      );

      test(
        'Edge Padding - 2D',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final r = pad(a, PadWidth.all(1), mode: PaddingMode.edge);
          expect(r.shape, [4, 4]);
          expect(r.toList(), [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]);
        }),
      );

      test(
        'Wrap Padding',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = pad(a, PadWidth.axes([(2, 2)]), mode: PaddingMode.wrap);
          expect(r.shape, [7]);
          expect(r.toList(), [2, 3, 1, 2, 3, 1, 2]);
        }),
      );

      test(
        'Reflect Padding - 1D',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = pad(a, PadWidth.all(2), mode: PaddingMode.reflect);
          expect(r.shape, [7]);
          expect(r.toList(), [3, 2, 1, 2, 3, 2, 1]);
        }),
      );

      test(
        'Reflect Padding - Large padding',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = pad(a, PadWidth.all(5), mode: PaddingMode.reflect);
          expect(r.shape, [13]);
          expect(r.toList(), [2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2]);
        }),
      );

      test(
        'Symmetric Padding',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = pad(a, PadWidth.all(2), mode: PaddingMode.symmetric);
          expect(r.shape, [7]);
          expect(r.toList(), [2, 1, 1, 2, 3, 3, 2]);
        }),
      );

      test(
        'Symmetric Mode - 1D Large Pad',
        () => NDArray.scope(() {
          final arr = NDArray<double>.fromList(
            [1.0, 2.0, 3.0],
            [3],
            DType.float64,
          );
          final padded = pad(arr, PadWidth.all(5), mode: PaddingMode.symmetric);
          expect(padded.toList(), [
            2.0,
            3.0,
            3.0,
            2.0,
            1.0,
            1.0,
            2.0,
            3.0,
            3.0,
            2.0,
            1.0,
            1.0,
            2.0,
          ]);
        }),
      );

      test(
        'Linear Ramp Padding',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final r = pad(
            a,
            PadWidth.all(2),
            mode: PaddingMode.linearRamp,
            endValues: PadValues.all(5.0, 7.0),
          );
          expect(r.shape, [7]);
          expect(r.toList(), [5.0, 3.0, 1.0, 2.0, 3.0, 5.0, 7.0]);
        }),
      );

      test(
        'Linear Ramp Padding - Int',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = pad(
            a,
            PadWidth.all(2),
            mode: PaddingMode.linearRamp,
            endValues: PadValues.all(5, 7),
          );
          expect(r.shape, [7]);
          expect(r.toList(), [5, 3, 1, 2, 3, 5, 7]);
        }),
      );

      test(
        'Stats Padding - Maximum',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 5, 3, 9, 2], [5], DType.int32);
          final r = pad(a, PadWidth.all(2), mode: PaddingMode.maximum);
          expect(r.shape, [9]);
          expect(r.toList(), [9, 9, 1, 5, 3, 9, 2, 9, 9]);
        }),
      );

      test(
        'Stats Padding - Maximum with Window',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 5, 3, 9, 2], [5], DType.int32);
          final r = pad(
            a,
            PadWidth.all(2),
            mode: PaddingMode.maximum,
            statLength: StatLength.all(3),
          );
          expect(r.shape, [9]);
          expect(r.toList(), [5, 5, 1, 5, 3, 9, 2, 9, 9]);
        }),
      );

      test(
        'Stats Padding - Mean',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
          final r = pad(a, PadWidth.all(1), mode: PaddingMode.mean);
          expect(r.shape, [6]);
          expect(r.toList(), [2.5, 1.0, 2.0, 3.0, 4.0, 2.5]);
        }),
      );

      test(
        'Stats Padding - Median - Odd',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 5, 3], [3], DType.int32);
          final r = pad(a, PadWidth.all(1), mode: PaddingMode.median);
          expect(r.shape, [5]);
          expect(r.toList(), [3, 1, 5, 3, 3]);
        }),
      );

      test(
        'Stats Padding - Median - Even',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 5.0, 3.0, 9.0], [4], DType.float64);
          final r = pad(a, PadWidth.all(1), mode: PaddingMode.median);
          expect(r.shape, [6]);
          expect(r.toList(), [4.0, 1.0, 5.0, 3.0, 9.0, 4.0]);
        }),
      );

      test(
        'Complex Padding - Constant',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [Complex(1, 2), Complex(3, 4)],
            [2],
            DType.complex128,
          );
          final r = pad(
            a,
            PadWidth.all(1),
            mode: PaddingMode.constant,
            constantValues: PadValues.all(Complex(9, 9)),
          );
          expect(r.shape, [4]);
          expect(r.toList(), [
            Complex(9, 9),
            Complex(1, 2),
            Complex(3, 4),
            Complex(9, 9),
          ]);
        }),
      );

      test(
        'Complex Padding - Median (Independent)',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [Complex(1, 6), Complex(5, 2), Complex(3, 4)],
            [3],
            DType.complex128,
          );
          final r = pad(a, PadWidth.all(1), mode: PaddingMode.median);
          expect(r.shape, [5]);
          expect(r.toList(), [
            Complex(3, 4),
            Complex(1, 6),
            Complex(5, 2),
            Complex(3, 4),
            Complex(3, 4),
          ]);
        }),
      );

      test(
        'Complex Padding - Min/Max (Lexicographical)',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [Complex(1, 10), Complex(2, 1), Complex(2, 2)],
            [3],
            DType.complex128,
          );
          final rMin = pad(a, PadWidth.all(1), mode: PaddingMode.minimum);
          final rMax = pad(a, PadWidth.all(1), mode: PaddingMode.maximum);

          expect(rMin.toList(), [
            Complex(1, 10),
            Complex(1, 10),
            Complex(2, 1),
            Complex(2, 2),
            Complex(1, 10),
          ]);
          expect(rMax.toList(), [
            Complex(2, 2),
            Complex(1, 10),
            Complex(2, 1),
            Complex(2, 2),
            Complex(2, 2),
          ]);
        }),
      );

      test(
        'Boolean Padding',
        () => NDArray.scope(() {
          final a = NDArray.fromList([true, false, true], [3], DType.boolean);
          final r = pad(
            a,
            PadWidth.all(1),
            mode: PaddingMode.constant,
            constantValues: PadValues.all(true),
          );
          expect(r.shape, [5]);
          expect(r.toList(), [true, true, false, true, true]);
        }),
      );

      test(
        'Zero Padding',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = pad(a, PadWidth.all(0));
          expect(r.shape, [3]);
          expect(r.toList(), [1, 2, 3]);
          expect(r, isNot(same(a)));
        }),
      );

      test(
        'Out Parameter Reuse',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final out = NDArray<int>.zeros([5], DType.int32);
          final r = pad(
            a,
            PadWidth.all(1),
            mode: PaddingMode.constant,
            constantValues: PadValues.all(9),
            out: out,
          );
          expect(r, same(out));
          expect(out.toList(), [9, 1, 2, 3, 9]);
        }),
      );

      test(
        'Preconditions - Disposed',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          a.dispose();
          expect(() => pad(a, PadWidth.all(1)), throwsStateError);
        }),
      );

      test(
        'Preconditions - 0-D Array',
        () => NDArray.scope(() {
          final a = NDArray<int>.fromList([1], [], DType.int32);
          expect(() => pad(a, PadWidth.all(1)), throwsArgumentError);
        }),
      );
    });

    group('Spacers Tests', () {
      group('linspace', () {
        test('standard linspace', () {
          final a = linspace(0.0, 1.0, 5);
          expect(a.shape, [5]);
          for (var i = 0; i < 5; i++) {
            expect(a.data[i], closeTo(i * 0.25, 1e-10));
          }
        });

        test('linspace without endpoint', () {
          final a = linspace(0.0, 1.0, 5, endpoint: false);
          expect(a.shape, [5]);
          for (var i = 0; i < 5; i++) {
            expect(a.data[i], closeTo(i * 0.2, 1e-10));
          }
        });

        test('integer linspace', () {
          final a = linspace<int>(0, 10, 5, dtype: DType.int64);
          expect(a.dtype, DType.int64);
          expect(a.data, [
            0,
            2,
            5,
            7,
            10,
          ]); // interpolated: 0, 2.5, 5, 7.5, 10 -> truncated to int
        });
      });

      group('logspace', () {
        test('standard logspace (base 10)', () {
          final a = logspace(0.0, 2.0, 3); // 10^0, 10^1, 10^2
          expect(a.shape, [3]);
          expect(a.data[0], closeTo(1.0, 1e-10));
          expect(a.data[1], closeTo(10.0, 1e-10));
          expect(a.data[2], closeTo(100.0, 1e-10));
        });

        test('logspace base 2', () {
          final a = logspace(0.0, 4.0, 5, base: 2.0); // 2^0, 2^1, 2^2, 2^3, 2^4
          expect(a.data, [1.0, 2.0, 4.0, 8.0, 16.0]);
        });

        test('logspace without endpoint', () {
          final a = logspace(0.0, 3.0, 3, endpoint: false); // 10^0, 10^1, 10^2
          expect(a.data[0], closeTo(1.0, 1e-10));
          expect(a.data[1], closeTo(10.0, 1e-10));
          expect(a.data[2], closeTo(100.0, 1e-10));
        });
      });

      group('geomspace', () {
        test('standard geomspace', () {
          final a = geomspace(1.0, 1000.0, 4);
          expect(a.data[0], closeTo(1.0, 1e-10));
          expect(a.data[1], closeTo(10.0, 1e-10));
          expect(a.data[2], closeTo(100.0, 1e-10));
          expect(a.data[3], closeTo(1000.0, 1e-10));
        });

        test('negative geomspace', () {
          final a = geomspace(-1.0, -1000.0, 4);
          expect(a.data[0], closeTo(-1.0, 1e-10));
          expect(a.data[1], closeTo(-10.0, 1e-10));
          expect(a.data[2], closeTo(-100.0, 1e-10));
          expect(a.data[3], closeTo(-1000.0, 1e-10));
        });

        test('geomspace without endpoint', () {
          final a = geomspace(1.0, 1000.0, 3, endpoint: false);
          expect(a.data[0], closeTo(1.0, 1e-10));
          expect(a.data[1], closeTo(10.0, 1e-10));
          expect(a.data[2], closeTo(100.0, 1e-10));
        });

        test('geomspace invalid signs', () {
          expect(() => geomspace(-1.0, 10.0, 5), throwsArgumentError);
          expect(() => geomspace(1.0, -10.0, 5), throwsArgumentError);
          expect(() => geomspace(0.0, 10.0, 5), throwsArgumentError);
        });
      });

      group('Complex spacers', () {
        test('linspace complex', () {
          final a = linspace<Complex>(Complex(0, 0), Complex(1, 1), 3);
          expect(a.dtype, DType.complex128);
          expect(a.data[0], Complex(0, 0));
          expect(a.data[1], Complex(0.5, 0.5));
          expect(a.data[2], Complex(1, 1));
        });

        test('logspace complex', () {
          final a = logspace<Complex>(
            Complex(0, 0),
            Complex(0, 2),
            3,
            base: 10.0,
          );
          expect(a.data[0], Complex(1, 0));
          final expected1 = Complex(
            math.cos(math.log(10)),
            math.sin(math.log(10)),
          );
          final val1 = a.data[1];
          expect(val1.real, closeTo(expected1.real, 1e-10));
          expect(val1.imag, closeTo(expected1.imag, 1e-10));
        });

        test('geomspace complex', () {
          final a = geomspace<Complex>(Complex(1, 0), Complex(-1, 0), 3);
          expect(a.data[0], Complex(1, 0));
          final val1 = a.data[1];
          expect(val1.real, closeTo(0, 1e-10));
          expect(val1.imag, closeTo(1, 1e-10));
          final val2 = a.data[2];
          expect(val2.real, closeTo(-1, 1e-10));
          expect(val2.imag, closeTo(0, 1e-10));
        });
      });

      group('Broadcasting spacers', () {
        test('linspace broadcasting', () {
          final start = NDArray.fromList([0.0, 10.0], [2], DType.float64);
          final stop = NDArray.fromList([1.0, 11.0], [2], DType.float64);
          // axis=0 (default) -> [numSamples, 2]
          final a = linspaceGrid(start, stop, 3);
          expect(a.shape, [3, 2]);
          // row 0: [0, 10]
          // row 1: [0.5, 10.5]
          // row 2: [1, 11]
          expect(a.data, [0.0, 10.0, 0.5, 10.5, 1.0, 11.0]);
        });

        test('linspace axis support', () {
          final start = NDArray.fromList([0.0, 10.0], [2], DType.float64);
          final stop = NDArray.fromList([1.0, 11.0], [2], DType.float64);
          // axis=1 -> [2, numSamples]
          final a = linspaceGrid(start, stop, 3, axis: 1);
          expect(a.shape, [2, 3]);
          // row 0: [0, 0.5, 1]
          // row 1: [10, 10.5, 11]
          expect(a.data, [0.0, 0.5, 1.0, 10.0, 10.5, 11.0]);
        });

        test('linspaceGridWithStep', () {
          final start = NDArray.fromList([0.0, 10.0], [2], DType.float64);
          final stop = NDArray.fromList([1.0, 12.0], [2], DType.float64);
          final (a, step) = linspaceGridWithStep(start, stop, 3);
          expect(a.shape, [3, 2]);
          expect(step.shape, [2]);
          expect(step.data, [0.5, 1.0]);
        });
      });

      group('Step retrieval', () {
        test('linspaceWithStep', () {
          final (a, step) = linspaceWithStep(0.0, 10.0, 5);
          expect(step, 2.5);
          expect(a.data, [0.0, 2.5, 5.0, 7.5, 10.0]);
        });
      });
    });
  });
}
