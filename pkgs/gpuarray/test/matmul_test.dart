import 'package:test/test.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Matrix Multiplication', () {
    test('1D vector dot product', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final b = GpuArray.fromList([4.0, 5.0, 6.0], [3], DType.float64);

        final dot = a.dot(b);
        expect(dot.scalar, equals(32.0)); // 1*4 + 2*5 + 3*6 = 32
      });
    });

    test('2D matrix multiplication', () {
      ResourceScope.scope(() {
        // A: 2x3
        final a = GpuArray.fromList(
          [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
          ],
          [2, 3],
          DType.float64,
        );

        // B: 3x2
        final b = GpuArray.fromList(
          [
            [7.0, 8.0],
            [9.0, 1.0],
            [2.0, 3.0],
          ],
          [3, 2],
          DType.float64,
        );

        // Result: 2x2
        // [1*7+2*9+3*2, 1*8+2*1+3*3] = [31, 19]
        // [4*7+5*9+6*2, 4*8+5*1+6*3] = [85, 55]
        final c = a.matmul(b);
        expect(c.shape, equals([2, 2]));
        expect(
          c.toNestedList(),
          equals([
            [31.0, 19.0],
            [85.0, 55.0],
          ]),
        );
      });
    });

    test('Batched 3D matrix multiplication', () {
      ResourceScope.scope(() {
        // 2 batches of 2x2 matrices
        final a = GpuArray.fromList(
          [
            [
              [1.0, 2.0],
              [3.0, 4.0],
            ],
            [
              [5.0, 6.0],
              [7.0, 8.0],
            ],
          ],
          [2, 2, 2],
          DType.float64,
        );

        // Identity 2x2
        final eye = GpuArray.fromList(
          [
            [1.0, 0.0],
            [0.0, 1.0],
          ],
          [1, 2, 2],
          DType.float64,
        );

        final res = a.matmul(eye);
        expect(res.shape, equals([2, 2, 2]));
        expect(
          res.toNestedList(),
          equals([
            [
              [1.0, 2.0],
              [3.0, 4.0],
            ],
            [
              [5.0, 6.0],
              [7.0, 8.0],
            ],
          ]),
        );
      });
    });

    test('Shape mismatch throws GpuShapeMismatchException', () {
      ResourceScope.scope(() {
        final a = GpuArray.zeros([2, 3], DType.float64);
        final b = GpuArray.zeros([4, 2], DType.float64);

        expect(() => a.matmul(b), throwsA(isA<GpuShapeMismatchException>()));
      });
    });
  });
}
