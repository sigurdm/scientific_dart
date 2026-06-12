import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';
import 'dart:typed_data';

void main() {
  group('NDArray Sorting, Searching & Counting Tests', () {
    group('Native QSort direct sort tests', () {
      test(
        'Sort contiguous Float64 row slices',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(
            Float64List.fromList([3.0, 1.0, 2.0, 9.0, 5.0, 7.0]),
            [2, 3],
            DType.float64,
          );

          final sorted = sort(mat, axis: 1);
          expect(sorted.shape, [2, 3]);
          expect(sorted.dtype, DType.float64);
          // Verifies each row is independently sorted in C memory!
          expect(sorted.toList(), [1.0, 2.0, 3.0, 5.0, 7.0, 9.0]);
        }),
      );

      test(
        'Sort contiguous Int32 array',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([5, -2, 10, 0]), [
            4,
          ], DType.int32);
          final s = sort(a);
          expect(s.toList(), [-2, 0, 5, 10]);
        }),
      );

      test(
        'Sort contiguous Int64 array',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int64List.fromList([100, 50, 200]), [
            3,
          ], DType.int64);
          final s = sort(a);
          expect(s.toList(), [50, 100, 200]);
        }),
      );
    });

    group('Complex Lexicographical Sorting tests', () {
      test(
        'Sort Complex128 array lexicographically',
        () => NDArray.scope(() {
          final a = NDArray.create([4], DType.complex128);
          // Real part compared first, then Imaginary part if reals are equal!
          a.data[0] = Complex(2.0, 5.0);
          a.data[1] = Complex(1.0, 10.0);
          a.data[2] = Complex(
            2.0,
            1.0,
          ); // shares real=2.0 but imag=1.0 is smaller
          a.data[3] = Complex(0.0, 0.0);

          final s = sort(a);
          expect(s.shape, [4]);
          expect(s.dtype, DType.complex128);
          expect(s.data[0], Complex(0.0, 0.0));
          expect(s.data[1], Complex(1.0, 10.0));
          expect(s.data[2], Complex(2.0, 1.0)); // comes *before* Complex(2,5)
          expect(s.data[3], Complex(2.0, 5.0));
        }),
      );
    });

    group('Axis-Swapping & Internal Axis Sort tests', () {
      test(
        'Sort 2D matrix along column axis 0',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(
            Float64List.fromList([5.0, 1.0, 4.0, 2.0, 6.0, 3.0]),
            [2, 3],
            DType.float64,
          );

          // Sorting columns across rows!
          final s = sort(mat, axis: 0);
          expect(s.shape, [2, 3]);
          expect(s.toList(), [
            2.0, 1.0, 3.0, // column 0: min(5,2)=2. col 2: min(4,3)=3
            5.0, 6.0, 4.0, // col 0: max(5,2)=5. col 2: max(4,3)=4
          ]);
        }),
      );

      test(
        'Sort 3D tensor along an internal axis',
        () => NDArray.scope(() {
          // shape [2, 2, 2]
          final tensor = NDArray.fromList(
            Float64List.fromList([
              10.0, 2.0, // slice 0, row 0
              5.0, 8.0, // slice 0, row 1
              3.0, 1.0, // slice 1, row 0
              7.0, 4.0, // slice 1, row 1
            ]),
            [2, 2, 2],
            DType.float64,
          );

          final s = sort(tensor, axis: 1); // sort along row axis
          expect(s.shape, [2, 2, 2]);
          expect(s.toList(), [
            5.0, 2.0, // row 0 vs 1 elements sorted
            10.0, 8.0,
            3.0, 1.0,
            7.0, 4.0,
          ]);
        }),
      );

      test(
        'Sort non-contiguous sliced array view',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(
            Float64List.fromList([10.0, 2.0, 3.0, 4.0, 20.0, 6.0]),
            [2, 3],
            DType.float64,
          );

          // Create a non-contiguous view: column 0 (elements 10.0 and 4.0)
          final colView = mat.slice([Slice.all(), Index(0)]);
          expect(colView.isContiguous, false);

          final s = sort(colView);
          expect(s.shape, [2]);
          expect(s.toList(), [4.0, 10.0]); // sorted perfectly!
        }),
      );
    });

    group('argsort (Indirect index sorting) tests', () {
      test(
        'Argsort 1D Float64 list',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([40.0, 10.0, 30.0, 20.0]),
            [4],
            DType.float64,
          );
          final indices = argsort(a);
          expect(indices.shape, [4]);
          expect(indices.dtype, DType.int32);
          expect(indices.toList(), [
            1,
            3,
            2,
            0,
          ]); // 10(idx 1) < 20(idx 3) < 30(idx 2) < 40(idx 0)
        }),
      );

      test(
        'Argsort complex array indirect ranking',
        () => NDArray.scope(() {
          final a = NDArray.create([3], DType.complex128);
          a.data[0] = Complex(5.0, 0.0);
          a.data[1] = Complex(2.0, 3.0);
          a.data[2] = Complex(2.0, 1.0);

          final idx = argsort(a);
          expect(idx.toList(), [
            2,
            1,
            0,
          ]); // Complex(2,1) < Complex(2,3) < Complex(5,0)
        }),
      );

      test(
        'Argsort float32, int32, and int64 lists',
        () => NDArray.scope(() {
          final f32 = NDArray.fromList(
            Float32List.fromList([4.0, 1.0, 3.0, 2.0]),
            [4],
            DType.float32,
          );
          final resF32 = argsort(f32);
          expect(resF32.toList(), [1, 3, 2, 0]);

          final i32 = NDArray.fromList(Int32List.fromList([40, 10, 30, 20]), [
            4,
          ], DType.int32);
          final resI32 = argsort(i32);
          expect(resI32.toList(), [1, 3, 2, 0]);

          final i64 = NDArray.fromList(
            Int64List.fromList([400, 100, 300, 200]),
            [4],
            DType.int64,
          );
          final resI64 = argsort(i64);
          expect(resI64.toList(), [1, 3, 2, 0]);
        }),
      );

      test(
        'Argsort scalar 0D array returns single index 0',
        () => NDArray.scope(() {
          final scalar = NDArray.fromList([99.0], [], DType.float64);
          final idx = argsort(scalar);
          expect(idx.shape, []);
          expect(idx.toList(), [0]);
        }),
      );

      test(
        'Argsort invalid target axis throws RangeError',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          expect(() => argsort(a, axis: 5), throwsRangeError);
          expect(() => argsort(a, axis: -5), throwsRangeError);
        }),
      );
    });

    group('count_nonzero & nonzero tests', () {
      test(
        'count_nonzero flat and axis reduced',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(Int32List.fromList([0, 5, 0, 2, 0, 3]), [
            2,
            3,
          ], DType.int32);

          expect(count_nonzero(mat).scalar, 3);

          final c0 = count_nonzero(mat, axis: 0); // count along columns
          expect(c0.shape, [3]);
          expect(c0.toList(), [1, 1, 1]); // col0 has 2, col1 has 5, col2 has 3
        }),
      );

      test(
        'nonzero indices tuple coordinate arrays',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(Int32List.fromList([0, 9, 0, 4, 0, 0]), [
            2,
            3,
          ], DType.int32);

          final coords = nonzero(mat);
          expect(coords.length, 2); // one per dimension
          expect(coords[0].data, [0, 1]); // row coordinates
          expect(coords[1].data, [
            1,
            0,
          ]); // col coordinates -> elements are at (0,1) and (1,0)
        }),
      );
    });

    group('where (Ternary Select & SIMD) tests', () {
      test(
        'where condition select with scalar/array broadcasting',
        () => NDArray.scope(() {
          final cond = NDArray.fromList(
            [true, false, true, false],
            [4],
            DType.boolean,
          );
          final x = NDArray.fromList(
            Float64List.fromList([10.0, 20.0, 30.0, 40.0]),
            [4],
            DType.float64,
          );
          final y = NDArray.fromList(
            Float64List.fromList([-5.0, -6.0, -7.0, -8.0]),
            [4],
            DType.float64,
          );

          final result = where(cond, x, y) as NDArray;
          expect(result.dtype, DType.float64);
          expect(result.shape, [4]);
          expect(result.toList(), [10.0, -6.0, 30.0, -8.0]);
        }),
      );

      test(
        'where behaves as nonzero if x and y are omitted',
        () => NDArray.scope(() {
          final cond = NDArray.fromList(
            [false, true, false, true],
            [4],
            DType.boolean,
          );
          final res = where(cond) as List<NDArray<int, Int32Marker>>;
          expect(res.length, 1);
          expect(res[0].data, [1, 3]);
        }),
      );

      test(
        'where with integer inputs',
        () => NDArray.scope(() {
          final cond = NDArray.fromList([true, false], [2], DType.boolean);
          final x = NDArray.fromList([10, 20], [2], DType.int32);
          final y = NDArray.fromList([100, 200], [2], DType.int32);

          final res = where(cond, x, y) as NDArray;
          expect(res.dtype, DType.int32);
          expect(res.toList(), [10, 200]);
        }),
      );

      test(
        'where with Complex inputs',
        () => NDArray.scope(() {
          final cond = NDArray.fromList([true, false], [2], DType.boolean);
          final x = NDArray.create([2], DType.complex128);
          x.data[0] = Complex(1.0, 1.0);
          x.data[1] = Complex(2.0, 2.0);

          final y = NDArray.create([2], DType.complex128);
          y.data[0] = Complex(10.0, 10.0);
          y.data[1] = Complex(20.0, 20.0);

          final res = where(cond, x, y) as NDArray;
          expect(res.dtype, DType.complex128);
          expect(res.data[0], Complex(1.0, 1.0));
          expect(res.data[1], Complex(20.0, 20.0));
        }),
      );
    });

    group('argmax / argmin (Extremes Reductions) tests', () {
      test(
        'Global flat argmax and argmin',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([10.0, 50.0, 5.0, 50.0, 20.0]),
            [5],
            DType.float64,
          );

          // In NumPy duplicate max extremes return the *first* occurrence!
          // max is 50.0 at index 1 and 3. NumPy returns 1!
          expect(argmax(a).scalar, 1);
          expect(argmin(a).scalar, 2); // min is 5.0 at index 2
        }),
      );

      test(
        'Argmax along axis reduction in 2D',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(
            Float64List.fromList([10.0, 30.0, 5.0, 40.0, 15.0, 60.0]),
            [2, 3],
            DType.float64,
          );

          final am0 = argmax(mat, axis: 0); // cols max row indices
          expect(am0.shape, [3]);
          expect(am0.toList(), [
            1,
            0,
            1,
          ]); // col0 max is row 1(40), col1 max row 0(30), col2 max row 1(60)
        }),
      );
    });

    group('SortKind custom quicksort/heapsort/mergesort/stable tests', () {
      test(
        'Sort with quicksort',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [5.0, 3.0, 8.0, 1.0, 2.0],
            [5],
            DType.float64,
          );
          final s = sort(a, kind: SortKind.quicksort);
          expect(s.toList(), [1.0, 2.0, 3.0, 5.0, 8.0]);
        }),
      );

      test(
        'Sort with heapsort',
        () => NDArray.scope(() {
          final a = NDArray.fromList([5, 3, 8, 1, 2], [5], DType.int32);
          final s = sort(a, kind: SortKind.heapsort);
          expect(s.toList(), [1, 2, 3, 5, 8]);
        }),
      );

      test(
        'Sort with stable mergesort',
        () => NDArray.scope(() {
          final a = NDArray.fromList([5, 3, 8, 1, 2], [5], DType.int64);
          final s = sort(a, kind: SortKind.mergesort);
          expect(s.toList(), [1, 2, 3, 5, 8]);
        }),
      );

      test(
        'Argsort with different kinds',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [5.0, 3.0, 8.0, 1.0, 2.0],
            [5],
            DType.float64,
          );
          final idxQ = argsort(a, kind: SortKind.quicksort);
          final idxH = argsort(a, kind: SortKind.heapsort);
          final idxM = argsort(a, kind: SortKind.mergesort);
          expect(idxQ.toList(), [3, 4, 1, 0, 2]);
          expect(idxH.toList(), [3, 4, 1, 0, 2]);
          expect(idxM.toList(), [3, 4, 1, 0, 2]);
        }),
      );

      test(
        'Boolean sort correctness',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [true, false, true, false, true],
            [5],
            DType.boolean,
          );
          final s = sort(a);
          expect(s.toList(), [false, false, true, true, true]);
        }),
      );
    });

    group('partition & argpartition tests', () {
      test(
        'Simple partition with single kth',
        () => NDArray.scope(() {
          final a = NDArray.fromList([3.0, 4.0, 2.0, 1.0], [4], DType.float64);
          final p = partition(a, 1);
          expect(p.data[1], 2.0);
          expect(p.data[0] <= 2.0, true);
          expect(p.data[2] >= 2.0, true);
          expect(p.data[3] >= 2.0, true);
        }),
      );

      test(
        'Partition with multiple kth',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [9, 5, 7, 1, 4, 2, 8, 6, 3],
            [9],
            DType.int32,
          );
          final p = partition(a, [2, 6]);
          expect(p.data[2], 3);
          expect(p.data[6], 7);
          for (var i = 0; i < 2; i++) {
            expect(p.data[i] <= 3, true);
          }
          for (var i = 3; i < 6; i++) {
            expect(p.data[i] >= 3 && p.data[i] <= 7, true);
          }
          for (var i = 7; i < 9; i++) {
            expect(p.data[i] >= 7, true);
          }
        }),
      );

      test(
        'Partition non-contiguous view',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(
            Float64List.fromList([5.0, 1.0, 4.0, 2.0, 6.0, 3.0]),
            [2, 3],
            DType.float64,
          );
          final colView = mat.slice([Slice.all(), Index(0)]);
          final p = partition(colView, 0);
          expect(p.toList(), [2.0, 5.0]);
        }),
      );

      test(
        'Argpartition with single kth',
        () => NDArray.scope(() {
          final a = NDArray.fromList([3.0, 4.0, 2.0, 1.0], [4], DType.float64);
          final idx = argpartition(a, 1);
          expect(a.data[idx.data[1]], 2.0);
          expect(a.data[idx.data[0]] <= 2.0, true);
          expect(a.data[idx.data[2]] >= 2.0, true);
          expect(a.data[idx.data[3]] >= 2.0, true);
        }),
      );

      test(
        'Boolean partition',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [true, false, true, false],
            [4],
            DType.boolean,
          );
          final p = partition(a, 1);
          expect(p.toList(), [false, false, true, true]);
        }),
      );
    });

    group('searchsorted tests', () {
      test(
        'Simple searchsorted numeric inputs',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [5],
            DType.float64,
          );

          final idxL = searchsorted(
            a,
            NDArray.fromList([2.5, 3.0, 5.5], [3], DType.float64),
            side: SearchSide.left,
          );
          expect(idxL.toList(), [2, 2, 5]);

          final idxR = searchsorted(
            a,
            NDArray.fromList([2.5, 3.0, 5.5], [3], DType.float64),
            side: SearchSide.right,
          );
          expect(idxR.toList(), [2, 3, 5]);
        }),
      );

      test(
        'searchsorted with sorter',
        () => NDArray.scope(() {
          final a = NDArray.fromList([3.0, 1.0, 2.0], [3], DType.float64);
          final sorter = argsort(a);

          final idxL = searchsorted(
            a,
            NDArray.fromList([1.5, 2.0], [2], DType.float64),
            side: SearchSide.left,
            sorter: sorter,
          );
          expect(idxL.toList(), [1, 1]);
        }),
      );

      test(
        'searchsorted multidimensional input shape matching',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
          final v = NDArray.fromList(
            [1.5, 2.5, 0.5, 3.5],
            [2, 2],
            DType.float64,
          );

          final idx = searchsorted(a, v);
          expect(idx.shape, [2, 2]);
          expect(idx.toList(), [1, 2, 0, 3]);
        }),
      );

      test(
        'searchsorted complex and boolean support',
        () => NDArray.scope(() {
          final aComp = NDArray.create([3], DType.complex128);
          aComp.data[0] = Complex(1.0, 1.0);
          aComp.data[1] = Complex(2.0, 2.0);
          aComp.data[2] = Complex(3.0, 3.0);

          final vComp = NDArray.create([2], DType.complex128);
          vComp.data[0] = Complex(1.5, 1.5);
          vComp.data[1] = Complex(2.0, 2.0);

          final idxComp = searchsorted(aComp, vComp, side: SearchSide.left);
          expect(idxComp.toList(), [1, 1]);

          final aBool = NDArray.fromList(
            [false, false, true, true],
            [4],
            DType.boolean,
          );
          final vBool = NDArray.fromList([false, true], [2], DType.boolean);
          final idxBool = searchsorted(aBool, vBool, side: SearchSide.left);
          expect(idxBool.toList(), [0, 2]);
        }),
      );
    });
  });
}
