import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';
import 'dart:typed_data';

void main() {
  group(
    'NDArray High-Dimensional (ND) matmul Stack Broadcasting & 1D Promotions Tests',
    () {
      group('Standard 3D Batch Matmul tests', () {
        test(
          'Verify uniform batch matrix stack multiply [2, 2, 2] x [2, 2, 2] -> [2, 2, 2]',
          () => NDArray.scope(() {
            // Two matrices per stack.
            // Batch 0, Mat A: [[1, 2], [3, 4]]
            // Batch 1, Mat A: [[5, 6], [7, 8]]
            final a = NDArray.fromList(
              Float64List.fromList([
                1.0, 2.0, 3.0, 4.0, // batch 0
                5.0, 6.0, 7.0, 8.0, // batch 1
              ]),
              [2, 2, 2],
              DType.float64,
            );

            // Batch 0, Mat B: identity [[1, 0], [0, 1]]
            // Batch 1, Mat B: all twos [[2, 2], [2, 2]]
            final b = NDArray.fromList(
              Float64List.fromList([
                1.0, 0.0, 0.0, 1.0, // batch 0 identity
                2.0, 2.0, 2.0, 2.0, // batch 1 twos
              ]),
              [2, 2, 2],
              DType.float64,
            );

            final res = matmul(a, b);

            expect(res.shape, [2, 2, 2]);
            expect(res.dtype, DType.float64);

            // Batch 0 result: A * identity => [[1, 2], [3, 4]]
            expect(res.data.sublist(0, 4), [1.0, 2.0, 3.0, 4.0]);

            // Batch 1 result: [[5,6],[7,8]] * [[2,2],[2,2]] => [[22, 22], [30, 30]]
            expect(res.data.sublist(4, 8), [22.0, 22.0, 30.0, 30.0]);
          }),
        );
      });

      group('Asymmetric Stack Shape Broadcasting tests', () {
        test(
          'Verify broadcast mapping stretching [2, 1, 2, 2] x [3, 2, 2] -> [2, 3, 2, 2]',
          () => NDArray.scope(() {
            // a stack shape: [2, 1], b stack shape: [3], pads b to [1, 3], combined stack: [2, 3]!
            final a = NDArray.fromList(
              Float64List.fromList([
                1.0, 0.0, 0.0, 1.0, // block 0 (identity)
                2.0, 0.0, 0.0, 2.0, // block 1 (scaled twos identity)
              ]),
              [2, 1, 2, 2],
              DType.float64,
            );

            final b = NDArray.fromList(
              Float64List.fromList([
                1.0, 2.0, 3.0, 4.0, // sub-block 0
                5.0, 6.0, 7.0, 8.0, // sub-block 1
                9.0, 10.0, 11.0, 12.0, // sub-block 2
              ]),
              [3, 2, 2],
              DType.float64,
            );

            final res = matmul(a, b);

            expect(res.shape, [2, 3, 2, 2]);

            // a[0, 0] is identity [1,0;0,1], multiplies all 3 blocks of b => returns them exactly!
            expect(res.data.sublist(0, 4), [1.0, 2.0, 3.0, 4.0]); // b[0]
            expect(res.data.sublist(4, 8), [5.0, 6.0, 7.0, 8.0]); // b[1]
            expect(res.data.sublist(8, 12), [9.0, 10.0, 11.0, 12.0]); // b[2]

            // a[1, 0] is double identity [2,0;0,2], multiplies all 3 blocks of b => returns them doubled!
            expect(res.data.sublist(12, 16), [2.0, 4.0, 6.0, 8.0]);
            expect(res.data.sublist(16, 20), [10.0, 12.0, 14.0, 16.0]);
            expect(res.data.sublist(20, 24), [18.0, 20.0, 22.0, 24.0]);
          }),
        );
      });

      group('1D Vector Promotions & Dot Products tests', () {
        test(
          'Verify pure 1D Vector Dot Product [3] x [3] -> [] (0D Scalar)',
          () => NDArray.scope(() {
            final v1 = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
              3,
            ], DType.float64);
            final v2 = NDArray.fromList(Float64List.fromList([4.0, 5.0, 6.0]), [
              3,
            ], DType.float64);

            final dotRes = matmul(v1, v2);
            // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32.0
            expect(dotRes.shape, <int>[]); // 0D scalar array
            expect(dotRes.data[0], 32.0);
          }),
        );

        test('Verify Matrix-Vector multiplication [2, 3] x [3] -> [2]', () => NDArray.scope(() {
          final mat = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            [2, 3],
            DType.float64,
          );
          final vec = NDArray.fromList(Float64List.fromList([1.0, 1.0, 1.0]), [
            3,
          ], DType.float64);

          final res = matmul(mat, vec);
          // row 0: 1+2+3 = 6.0, row 1: 4+5+6 = 15.0
          expect(res.shape, [2]);
          expect(res.toList(), [6.0, 15.0]);
        }));

        test('Verify Vector-Matrix multiplication [3] x [3, 2] -> [2]', () => NDArray.scope(() {
          final vec = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
            3,
          ], DType.float64);
          final mat = NDArray.fromList(
            Float64List.fromList([1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            [3, 2],
            DType.float64,
          );

          final res = matmul(vec, mat);
          // col 0: 1*1 + 2*0 + 3*1 = 4.0
          // col 1: 1*0 + 2*1 + 3*1 = 5.0
          expect(res.shape, [2]);
          expect(res.toList(), [4.0, 5.0]);
        }));

        test(
          'matmul() throws ArgumentError on incompatible 1D vector dot dimensions',
          () => NDArray.scope(() {
            final v1 = NDArray<double>.fromList(
              Float64List.fromList([1.0, 2.0]),
              [2],
              DType.float64,
            );
            final v2 = NDArray<double>.fromList(
              Float64List.fromList([1.0, 2.0, 3.0]),
              [3],
              DType.float64,
            );
            expect(() => matmul(v1, v2), throwsArgumentError);
          }),
        );

        test(
          'matmul() throws ArgumentError on incompatible inner dimensions',
          () => NDArray.scope(() {
            final a = NDArray<double>.zeros([2, 3], DType.float64);
            final b = NDArray<double>.zeros([2, 2], DType.float64);
            expect(() => matmul(a, b), throwsArgumentError);
          }),
        );
      });

      group('Float32 Single-Precision matmul tests', () {
        test('Verify Float32 1D Vector Dot Product sdot', () => NDArray.scope(() {
          final v1 = NDArray<double>.fromList(
            Float32List.fromList([1.0, 2.0]),
            [2],
            DType.float32,
          );
          final v2 = NDArray<double>.fromList(
            Float32List.fromList([3.0, 4.0]),
            [2],
            DType.float32,
          );

          final res = matmul(v1, v2);
          expect(res.shape, []);
          expect(res.dtype, DType.float32);
          expect(res.data[0], closeTo(11.0, 1e-5));
        }));

        test('Verify Float32 2D Matrix Multiply sgemm', () => NDArray.scope(() {
          final a = NDArray<double>.fromList(
            Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float32,
          );
          final b = NDArray<double>.fromList(
            Float32List.fromList([5.0, 6.0, 7.0, 8.0]),
            [2, 2],
            DType.float32,
          );

          final res = matmul(a, b);
          expect(res.shape, [2, 2]);
          expect(res.dtype, DType.float32);
          expect(res.toList(), [19.0, 22.0, 43.0, 50.0]);
        }));
      });
    },
  );
}
