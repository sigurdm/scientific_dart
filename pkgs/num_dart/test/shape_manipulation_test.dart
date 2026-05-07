import 'package:num_dart/num_dart.dart';
import 'package:test/test.dart';
import 'dart:typed_data';

void main() {
  group('NDArray Shape Manipulation Tests', () {
    group('expandDims Tests', () {
      test('Basic expandDims at front (axis 0)', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = a.expandDims(0);
        addTearDown(b.dispose);
        expect(b.shape, [1, 2, 2]);
        expect(b.strides, [2, 2, 1]); // Strides are correct for [1, 2, 2]
        expect(b.toList(), [1.0, 2.0, 3.0, 4.0]);

        // Verify it shares memory (view behavior)
        b.data[0] = 99.0;
        expect(a.data[0], 99.0);
      });

      test('expandDims in the middle (axis 1)', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = a.expandDims(1);
        addTearDown(b.dispose);
        expect(b.shape, [2, 1, 2]);
        expect(b.strides, [2, 1, 1]);
        expect(b.toList(), [1.0, 2.0, 3.0, 4.0]);
      });

      test('expandDims at the end (axis 2)', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = a.expandDims(2);
        addTearDown(b.dispose);
        expect(b.shape, [2, 2, 1]);
        expect(b.strides, [2, 1, 1]);
        expect(b.toList(), [1.0, 2.0, 3.0, 4.0]);
      });

      test('expandDims negative axis (-1)', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        addTearDown(a.dispose);
        // -1 refers to the position right after the last dimension (equivalent to axis 2)
        final b = a.expandDims(-1);
        addTearDown(b.dispose);
        expect(b.shape, [2, 2, 1]);
      });

      test('expandDims negative axis (-3)', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        addTearDown(a.dispose);
        // -3 refers to the very front (equivalent to axis 0)
        final b = a.expandDims(-3);
        addTearDown(b.dispose);
        expect(b.shape, [1, 2, 2]);
      });

      test('expandDims invalid axis throws', () {
        final a = NDArray.create([2, 2], DType.float64);
        addTearDown(a.dispose);
        expect(() => a.expandDims(3), throwsRangeError);
        expect(() => a.expandDims(-4), throwsRangeError);
      });
    });

    group('squeeze Tests', () {
      test('Squeeze all size 1 axes', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          1,
          2,
          1,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = a.squeeze();
        addTearDown(b.dispose);
        expect(b.shape, [2]);
        expect(b.strides, [1]);
        expect(b.toList(), [1.0, 2.0]);

        // View verification
        b.data[0] = 42.0;
        expect(a.data[0], 42.0);
      });

      test('Squeeze specific int axis', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          1,
          2,
          1,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = a.squeeze(axis: 0);
        addTearDown(b.dispose);
        expect(b.shape, [2, 1]);
        expect(b.strides, [1, 1]);
      });

      test('Squeeze specific negative axis', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          1,
          2,
          1,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = a.squeeze(axis: -1);
        addTearDown(b.dispose);
        expect(b.shape, [1, 2]);
        expect(b.strides, [2, 1]);
      });

      test('Squeeze specific List<int> axes', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          1,
          2,
          1,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = a.squeeze(axis: [0, 2]);
        addTearDown(b.dispose);
        expect(b.shape, [2]);
        expect(b.strides, [1]);
      });

      test('Squeeze axis with size > 1 throws', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          1,
          2,
          1,
        ], DType.float64);
        addTearDown(a.dispose);
        expect(() => a.squeeze(axis: 1), throwsArgumentError);
      });

      test('Squeeze invalid axis range throws', () {
        final a = NDArray.create([1, 2], DType.float64);
        addTearDown(a.dispose);
        expect(() => a.squeeze(axis: 2), throwsRangeError);
        expect(() => a.squeeze(axis: -3), throwsRangeError);
      });

      test('Squeeze 1x1 array to 0-D scalar array', () {
        final a = NDArray.fromList(Float64List.fromList([5.5]), [
          1,
          1,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = a.squeeze();
        addTearDown(b.dispose);
        expect(b.shape, isEmpty);
        expect(b.strides, isEmpty);
        expect(b[[]], 5.5);
      });
    });

    group('swapaxes Tests', () {
      test('Basic swapaxes in 2D', () {
        final a = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
          [2, 3],
          DType.float64,
        );
        addTearDown(a.dispose);
        // shape: [2, 3], strides: [3, 1]
        final b = a.swapaxes(0, 1);
        addTearDown(b.dispose);
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
        b.data[0] = 11.0; // updates a.data[0]
        expect(a.data[0], 11.0);
      });

      test('swapaxes negative index', () {
        final a = NDArray.create([2, 3, 4], DType.float64);
        addTearDown(a.dispose);
        final b = a.swapaxes(-3, -1); // swaps 0 and 2
        addTearDown(b.dispose);
        expect(b.shape, [4, 3, 2]);
      });

      test('swapaxes same axis is no-op view', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          2,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = a.swapaxes(0, 0);
        addTearDown(b.dispose);
        expect(b.shape, [2]);
        expect(b.toList(), [1.0, 2.0]);
      });

      test('swapaxes out of bounds throws', () {
        final a = NDArray.create([2, 2], DType.float64);
        addTearDown(a.dispose);
        expect(() => a.swapaxes(0, 2), throwsRangeError);
        expect(() => a.swapaxes(-3, 1), throwsRangeError);
      });
    });

    group('moveaxis Tests', () {
      test('Move single axis 0 to 2 in 3D', () {
        final a = NDArray.create([2, 3, 4], DType.float64);
        addTearDown(a.dispose);
        final b = a.moveaxis(0, 2);
        addTearDown(b.dispose);
        expect(b.shape, [3, 4, 2]);
        // original strides were [12, 4, 1]
        // order becomes [1, 2, 0], so strides should be [4, 1, 12]
        expect(b.strides, [4, 1, 12]);
      });

      test('Move single axis negative indexes (-1 to 0)', () {
        final a = NDArray.create([2, 3, 4], DType.float64);
        addTearDown(a.dispose);
        final b = a.moveaxis(-1, 0); // moves axis 2 to 0
        addTearDown(b.dispose);
        expect(b.shape, [4, 2, 3]);
        expect(b.strides, [1, 12, 4]);
      });

      test('Move multiple axes lists', () {
        final a = NDArray.create([2, 3, 4], DType.float64);
        addTearDown(a.dispose);
        // Move axis 0 -> 1 and axis 1 -> 2
        final b = a.moveaxis([0, 1], [1, 2]);
        addTearDown(b.dispose);
        expect(b.shape, [
          4,
          2,
          3,
        ]); // axis 2 becomes 0, axis 0 becomes 1, axis 1 becomes 2
      });

      test('moveaxis invalid lengths or duplicates throw', () {
        final a = NDArray.create([2, 3, 4], DType.float64);
        addTearDown(a.dispose);
        expect(() => a.moveaxis([0, 1], [2]), throwsArgumentError);
        expect(() => a.moveaxis([0, 0], [1, 2]), throwsArgumentError);
        expect(() => a.moveaxis([0, 1], [2, 2]), throwsArgumentError);
      });
    });

    group('tile Tests', () {
      test('Tile 1D array with int scalar reps', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          2,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = tile(a, 3);
        addTearDown(b.dispose);
        expect(b.shape, [6]);
        expect(b.toList(), [1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);

        // Verify it allocates new memory (copy behavior)
        b.data[0] = 99.0;
        expect(a.data[0], 1.0);
      });

      test('Tile 1D array with 1D List reps', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          2,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = tile(a, [2]);
        addTearDown(b.dispose);
        expect(b.shape, [4]);
        expect(b.toList(), [1.0, 2.0, 1.0, 2.0]);
      });

      test('Tile 1D array with higher-rank 2D List reps', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          2,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = tile(a, [
          2,
          3,
        ]); // shape promoted from [2] to [1, 2], target [2*1, 3*2] = [2, 6]
        addTearDown(b.dispose);
        expect(b.shape, [2, 6]);
        expect(b.toList(), [
          1.0, 2.0, 1.0, 2.0, 1.0, 2.0, // row 0
          1.0, 2.0, 1.0, 2.0, 1.0, 2.0, // row 1
        ]);
      });

      test('Tile 2D array with list reps', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = tile(a, [2, 1]); // repeat rows 2 times, columns 1 time
        addTearDown(b.dispose);
        expect(b.shape, [4, 2]);
        expect(b.toList(), [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]);
      });

      test('Tile with negative or invalid reps throws', () {
        final a = NDArray.create([2], DType.float64);
        addTearDown(a.dispose);
        expect(() => tile(a, -1), throwsArgumentError);
        expect(() => tile(a, [2, -3]), throwsArgumentError);
        expect(() => tile(a, 'invalid'), throwsArgumentError);
        expect(() => tile(a, 3.5), throwsArgumentError);
      });
    });

    group('repeat Tests', () {
      test('Repeat elements of flat 1D array with scalar int', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
          3,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = repeat(a, 2);
        addTearDown(b.dispose);
        expect(b.shape, [6]);
        expect(b.toList(), [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);

        // Copy verification
        b.data[0] = 99.0;
        expect(a.data[0], 1.0);
      });

      test('Repeat automatically flattens high-rank array if axis is null', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = repeat(a, 2); // axis = null
        addTearDown(b.dispose);
        expect(b.shape, [8]);
        expect(b.toList(), [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
      });

      test('Repeat along axis 0 with scalar int', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = repeat(a, 2, axis: 0);
        addTearDown(b.dispose);
        expect(b.shape, [4, 2]);
        expect(b.toList(), [1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0]);
      });

      test('Repeat along axis 1 with custom List<int> counts', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        addTearDown(a.dispose);
        // Repeat column 0 -> 2 times, column 1 -> 1 time. total cols = 3
        final b = repeat(a, [2, 1], axis: 1);
        addTearDown(b.dispose);
        expect(b.shape, [2, 3]);
        expect(b.toList(), [
          1.0, 1.0, 2.0, // row 0
          3.0, 3.0, 4.0, // row 1
        ]);
      });

      test('Repeat with an entry of 0 counts (deletes slice)', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        addTearDown(a.dispose);
        // Repeat column 0 -> 0 times, column 1 -> 2 times.
        final b = repeat(a, [0, 2], axis: 1);
        addTearDown(b.dispose);
        expect(b.shape, [2, 2]);
        expect(b.toList(), [2.0, 2.0, 4.0, 4.0]);
      });

      test('Repeat mismatched list counts throws', () {
        final a = NDArray.create([2, 2], DType.float64);
        addTearDown(a.dispose);
        expect(() => repeat(a, [1, 2, 3], axis: 1), throwsArgumentError);
      });

      test('Repeat negative counts throws', () {
        final a = NDArray.create([3], DType.float64);
        addTearDown(a.dispose);
        expect(() => repeat(a, -2), throwsArgumentError);
        expect(() => repeat(a, [1, -1, 0]), throwsArgumentError);
      });

      test('Repeat invalid axis throws', () {
        final a = NDArray.create([2, 2], DType.float64);
        addTearDown(a.dispose);
        expect(() => repeat(a, 2, axis: 2), throwsRangeError);
      });
    });

    group('NDArray deep duplicate copy() Tests', () {
      test('copy() contiguous float64 array', () {
        final a = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
        addTearDown(a.dispose);
        final b = a.copy();
        addTearDown(b.dispose);

        expect(b.shape, [3]);
        expect(b.dtype, DType.float64);
        expect(b.toList(), [10.0, 20.0, 30.0]);

        b.data[0] = 99.0;
        expect(a.data[0], 10.0);
      });

      test('copy() non-contiguous transposed strided float64 array view', () {
        final parent = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        addTearDown(parent.dispose);

        final view = parent.transposed;
        addTearDown(view.dispose);
        expect(view.isContiguous, false);

        final b = view.copy();
        addTearDown(b.dispose);
        expect(b.shape, [2, 2]);
        expect(b.isContiguous, true);
        expect(b.toList(), [1.0, 3.0, 2.0, 4.0]);

        b.data[0] = 99.0;
        expect(parent.data[0], 1.0);
      });

      test('disposed array throws StateError', () {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        a.dispose();
        expect(() => a.copy(), throwsStateError);
      });

      test('top-level copy() contiguous float64 array', () {
        final a = NDArray.fromList([10.0, 20.0, 30.0], [3], DType.float64);
        addTearDown(a.dispose);
        final b = copy(a);
        addTearDown(b.dispose);

        expect(b.shape, [3]);
        expect(b.dtype, DType.float64);
        expect(b.toList(), [10.0, 20.0, 30.0]);

        b.data[0] = 99.0;
        expect(a.data[0], 10.0);
      });

      test('top-level copy() disposed array throws StateError', () {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        a.dispose();
        expect(() => copy(a), throwsStateError);
      });
    });
  });
}
