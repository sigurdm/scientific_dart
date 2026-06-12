import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('NDIter and NDEnumerate Tests', () {
    test(
      'NDIter single 2D array walk',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [10, 20, 30, 40, 50, 60],
          [2, 3],
          DType.int32,
        );
        final iter = NDIter(a);

        final coordsList = <List<int>>[];
        final indicesList = <int>[];

        while (iter.moveNext()) {
          coordsList.add(List<int>.from(iter.coords));
          indicesList.add(iter.index);
        }

        expect(coordsList, [
          [0, 0],
          [0, 1],
          [0, 2],
          [1, 0],
          [1, 1],
          [1, 2],
        ]);

        expect(indicesList, [0, 1, 2, 3, 4, 5]);
        expect(a.data[indicesList[0]], 10);
        expect(a.data[indicesList[5]], 60);
      }),
    );

    test(
      'NDIter single 0D scalar and 1D array edge cases',
      () => NDArray.scope(() {
        // 1D Array
        final a = NDArray.fromList([99, 100], [2], DType.int32);
        final iter1D = NDIter(a);
        final coords1D = <List<int>>[];
        final indices1D = <int>[];

        while (iter1D.moveNext()) {
          coords1D.add(List<int>.from(iter1D.coords));
          indices1D.add(iter1D.index);
        }
        expect(coords1D, [
          [0],
          [1],
        ]);
        expect(indices1D, [0, 1]);

        // 0D Scalar
        final s = NDArray.fromList([42], [], DType.int32);
        final iter0D = NDIter(s);
        expect(iter0D.moveNext(), true);
        expect(iter0D.coords, isEmpty);
        expect(iter0D.index, 0);
        expect(iter0D.moveNext(), false);
      }),
    );

    test(
      'NDIter zero-allocation coords list reuse check',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
        final iter = NDIter(a);

        expect(iter.moveNext(), true);
        final firstCoordsRef = iter.coords;

        expect(iter.moveNext(), true);
        final secondCoordsRef = iter.coords;

        // Verify that NDIter returns the exact same identical in-memory List object (reused in-place)
        expect(identical(firstCoordsRef, secondCoordsRef), true);
        expect(
          firstCoordsRef[0],
          1,
        ); // The original reference's contents have been updated!
      }),
    );

    test(
      'NDIter.broadcast2 dual-array simultaneous walk',
      () => NDArray.scope(() {
        final a = NDArray.fromList([10, 20], [2, 1], DType.int32); // [2, 1]
        final b = NDArray.fromList([1, 2, 3], [1, 3], DType.int32); // [1, 3]

        final iter = NDIter.broadcast2(a, b);

        final coordsList = <List<int>>[];
        final aIndices = <int>[];
        final bIndices = <int>[];

        while (iter.moveNext()) {
          coordsList.add(List<int>.from(iter.coords));
          aIndices.add(iter.getIndex(0));
          bIndices.add(iter.getIndex(1));
        }

        // Common broadcasted shape: [2, 3]
        expect(coordsList, [
          [0, 0],
          [0, 1],
          [0, 2],
          [1, 0],
          [1, 1],
          [1, 2],
        ]);

        // Check aIndices (shape [2, 1] stretched along dim 1)
        expect(aIndices, [
          0,
          0,
          0,
          1,
          1,
          1,
        ]); // Row 0 (offset 0) then Row 1 (offset 1)

        // Check bIndices (shape [1, 3] stretched along dim 0)
        expect(bIndices, [0, 1, 2, 0, 1, 2]); // Col 0, 1, 2 repeated
      }),
    );

    test(
      'NDIter.broadcast multi-array simultaneous walk',
      () => NDArray.scope(() {
        final a = NDArray.fromList([100], [1], DType.int32);
        final b = NDArray.fromList([1, 2], [2], DType.int32);
        final c = NDArray.fromList([10, 20], [2], DType.int32);

        final iter = NDIter.broadcast([a, b, c]);

        final coordsList = <List<int>>[];
        final aIdx = <int>[];
        final bIdx = <int>[];
        final cIdx = <int>[];

        while (iter.moveNext()) {
          coordsList.add(List<int>.from(iter.coords));
          aIdx.add(iter.getIndex(0));
          bIdx.add(iter.getIndex(1));
          cIdx.add(iter.getIndex(2));
        }

        expect(coordsList, [
          [0],
          [1],
        ]);

        expect(aIdx, [0, 0]);
        expect(bIdx, [0, 1]);
        expect(cIdx, [0, 1]);
      }),
    );

    test(
      'NDIter.broadcast incompatible shapes throw ArgumentError',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 2], [2], DType.int32);
        final b = NDArray.fromList([1, 2, 3], [3], DType.int32);

        expect(() => NDIter.broadcast2(a, b), throwsArgumentError);
        expect(() => NDIter.broadcast([a, b]), throwsArgumentError);
      }),
    );

    test(
      'disposed array throws StateError',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 2], [2], DType.int32);
        a.dispose();

        expect(() => NDIter(a), throwsStateError);
      }),
    );

    test(
      'NDEnumerate coordinates and value enumeration',
      () => NDArray.scope(() {
        final a = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.int32);
        final en = NDEnumerate<int>(a);

        final coordsList = <List<int>>[];
        final valuesList = <int>[];

        while (en.moveNext()) {
          coordsList.add(List<int>.from(en.coords));
          valuesList.add(en.value);
        }

        expect(coordsList, [
          [0, 0],
          [0, 1],
          [1, 0],
          [1, 1],
        ]);

        expect(valuesList, [10, 20, 30, 40]);
      }),
    );
  });
}
