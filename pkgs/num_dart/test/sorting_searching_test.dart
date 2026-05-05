import 'package:num_dart/num_dart.dart';
import 'package:test/test.dart';
import 'dart:typed_data';

void main() {
  group('NDArray Sorting, Searching & Counting Tests', () {
    group('Native QSort direct sort tests', () {
      test('Sort contiguous Float64 row slices', () {
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
      });

      test('Sort contiguous Int32 array', () {
        final a = NDArray.fromList(Int32List.fromList([5, -2, 10, 0]), [
          4,
        ], DType.int32);
        final s = sort(a);
        expect(s.toList(), [-2, 0, 5, 10]);
      });

      test('Sort contiguous Int64 array', () {
        final a = NDArray.fromList(Int64List.fromList([100, 50, 200]), [
          3,
        ], DType.int64);
        final s = sort(a);
        expect(s.toList(), [50, 100, 200]);
      });
    });

    group('Complex Lexicographical Sorting tests', () {
      test('Sort Complex128 array lexicographically', () {
        final a = NDArray<Complex>.create([4], DType.complex128);
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
      });
    });

    group('Axis-Swapping & Internal Axis Sort tests', () {
      test('Sort 2D matrix along column axis 0', () {
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
      });

      test('Sort 3D tensor along an internal axis', () {
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
      });

      test('Sort non-contiguous sliced array view', () {
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
      });
    });

    group('argsort (Indirect index sorting) tests', () {
      test('Argsort 1D Float64 list', () {
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
      });

      test('Argsort complex array indirect ranking', () {
        final a = NDArray<Complex>.create([3], DType.complex128);
        a.data[0] = Complex(5.0, 0.0);
        a.data[1] = Complex(2.0, 3.0);
        a.data[2] = Complex(2.0, 1.0);

        final idx = argsort(a);
        expect(idx.toList(), [
          2,
          1,
          0,
        ]); // Complex(2,1) < Complex(2,3) < Complex(5,0)
      });
    });

    group('count_nonzero & nonzero tests', () {
      test('count_nonzero flat and axis reduced', () {
        final mat = NDArray.fromList(Int32List.fromList([0, 5, 0, 2, 0, 3]), [
          2,
          3,
        ], DType.int32);

        expect(count_nonzero(mat), 3);

        final c0 =
            count_nonzero(mat, axis: 0) as NDArray<int>; // count along columns
        expect(c0.shape, [3]);
        expect(c0.toList(), [1, 1, 1]); // col0 has 2, col1 has 5, col2 has 3
      });

      test('nonzero indices tuple coordinate arrays', () {
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
      });
    });

    group('where (Ternary Select & SIMD) tests', () {
      test('where condition select with scalar/array broadcasting', () {
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
      });

      test('where behaves as nonzero if x and y are omitted', () {
        final cond = NDArray.fromList(
          [false, true, false, true],
          [4],
          DType.boolean,
        );
        final res = where(cond) as List<NDArray<int>>;
        expect(res.length, 1);
        expect(res[0].data, [1, 3]);
      });
    });

    group('argmax / argmin (Extremes Reductions) tests', () {
      test('Global flat argmax and argmin', () {
        final a = NDArray.fromList(
          Float64List.fromList([10.0, 50.0, 5.0, 50.0, 20.0]),
          [5],
          DType.float64,
        );

        // In NumPy duplicate max extremes return the *first* occurrence!
        // max is 50.0 at index 1 and 3. NumPy returns 1!
        expect(argmax(a), 1);
        expect(argmin(a), 2); // min is 5.0 at index 2
      });

      test('Argmax along axis reduction in 2D', () {
        final mat = NDArray.fromList(
          Float64List.fromList([10.0, 30.0, 5.0, 40.0, 15.0, 60.0]),
          [2, 3],
          DType.float64,
        );

        final am0 =
            argmax(mat, axis: 0) as NDArray<int>; // cols max row indices
        expect(am0.shape, [3]);
        expect(am0.toList(), [
          1,
          0,
          1,
        ]); // col0 max is row 1(40), col1 max row 0(30), col2 max row 1(60)
      });
    });
  });
}
