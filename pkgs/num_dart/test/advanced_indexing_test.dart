import 'package:num_dart/num_dart.dart';
import 'package:test/test.dart';
import 'dart:typed_data';

void main() {
  group('NDArray Advanced Indexing, Fancy Lists & Mask Overloads Tests', () {
    group('Explicit Static-Typed Addressing Methods tests', () {
      test('getCell and setCell basic checks', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);

        expect(a.getCell([0, 0]), 1.0);
        expect(a.getCell([0, 1]), 2.0);

        a.setCell([1, 1], 40.0);
        expect(a.getCell([1, 1]), 40.0);
        expect(a.toList(), [1.0, 2.0, 3.0, 40.0]);
      });

      test('setByMask explicit boolean mask mutation scalar clipping', () {
        final arr = NDArray.fromList(
          Float64List.fromList([-5.0, 10.0, -2.5, 4.0]),
          [4],
          DType.float64,
        );
        final mask = arr < 0.0; // returns binary mask array

        // Explicit scalar clip
        arr.setByMask(mask, 0.0);
        expect(arr.toList(), [0.0, 10.0, 0.0, 4.0]);
      });

      test('setIndicesScalar and setIndices fancy explicit row mutations', () {
        final mat = NDArray.fromList(
          Int32List.fromList([1, 1, 1, 2, 2, 2, 3, 3, 3]),
          [3, 3],
          DType.int32,
        );

        final targetRows = NDArray.fromList([0, 2], [2], DType.int32);

        // Overwrite row 0 and row 2 to 9
        mat.setIndicesScalar(targetRows, 9, axis: 0);
        expect(mat.toList(), [9, 9, 9, 2, 2, 2, 9, 9, 9]);
      });
    });

    group(
      'Polymorphic Overload [] and []= Syntax tests (NumPy Equivalence)',
      () {
        test('operator [] single int extracts direct sub-matrix row views', () {
          final a = NDArray.fromList(
            Int32List.fromList(List.generate(12, (i) => i)),
            [3, 4],
            DType.int32,
          );

          // NumPy: a[1] extracts row 1 view
          final r1 = a[1];
          expect(r1.shape, [4]);
          expect(r1.toList(), [4, 5, 6, 7]);
        });

        test('operator [] List<int> extracts fancy row stacks', () {
          final a = NDArray.fromList(
            Int32List.fromList(List.generate(12, (i) => i)),
            [3, 4],
            DType.int32,
          );

          // NumPy: a[[0, 2]] extracts row 0 and row 2 stacked!
          final fancy =
              a[[
                [0, 2],
              ]];
          expect(fancy.shape, [2, 4]);
          expect(fancy.toList(), [
            0, 1, 2, 3, // row 0
            8, 9, 10, 11, // row 2
          ]);
        });

        test(
          'operator [] NDArray boolean criteria filters and flattens to 1D',
          () {
            final a = NDArray.fromList(
              Float64List.fromList([1.0, -2.0, 3.0, -4.0]),
              [4],
              DType.float64,
            );

            // NumPy: a[a > 0] returns positive elements flattened vector!
            final positives = a[a > 0.0];
            expect(positives.shape, [2]);
            expect(positives.toList(), [1.0, 3.0]);
          },
        );

        test('operator []= Boolean Mask assignments clips in-place', () {
          final mat = NDArray.fromList(
            Float64List.fromList([10.0, -1.0, 20.0, -3.0]),
            [4],
            DType.float64,
          );

          // NumPy: mat[mat < 0.0] = 0.0
          mat[mat < 0.0] = 0.0;
          expect(mat.toList(), [10.0, 0.0, 20.0, 0.0]);
        });

        test(
          'operator []= Fancy list index assignment mutates specific rows stack',
          () {
            final mat = NDArray.arange(
              0.0,
              6.0,
              dtype: DType.float64,
            ).reshape([3, 2]);
            // [[0,1], [2,3], [4,5]]

            // Mutate row 0 and row 2 in-place to scalar 99.0
            mat[[
                  [0, 2],
                ]] =
                99.0;
            expect(mat.toList(), [
              99.0, 99.0, // row 0 overwritten
              2.0, 3.0,
              99.0, 99.0, // row 2 overwritten
            ]);
          },
        );
      },
    );
  });
}
