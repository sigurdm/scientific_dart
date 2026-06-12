import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';
import 'dart:typed_data';

void main() {
  group('NDArray Determinant Tests', () {
    test(
      '2D Matrix Determinant (Backward Compatibility)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final d = det(a);
        expect(d.shape, []);
        expect(d.data[0], closeTo(-2.0, 1e-9));
      }),
    );

    test(
      '3D Stack Matrix Determinant',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          Float64List.fromList([
            1.0, 2.0, 3.0, 4.0, // Matrix 0: det = -2
            5.0, 6.0, 7.0, 8.0, // Matrix 1: det = -2
          ]),
          [2, 2, 2],
          DType.float64,
        );

        final d = det(a);

        expect(d.shape, [2]);
        expect(d.data[0], closeTo(-2.0, 1e-9));
        expect(d.data[1], closeTo(-2.0, 1e-9));
      }),
    );

    test(
      '2D Complex128 Matrix Determinant',
      () => NDArray.scope(() {
        // Matrix: [[1+1i, 2-1i],
        //          [3+0i, 4+2i]]
        // det = (1+1i)*(4+2i) - (2-1i)*3
        //     = (2 + 6i) - (6 - 3i) = -4 + 9i
        final a = NDArray.fromList(
          [
            Complex(1.0, 1.0),
            Complex(2.0, -1.0),
            Complex(3.0, 0.0),
            Complex(4.0, 2.0),
          ],
          [2, 2],
          DType.complex128,
        );

        final d = det<Complex>(a);

        expect(d.shape, []);
        expect(d.data[0].real, closeTo(-4.0, 1e-9));
        expect(d.data[0].imag, closeTo(9.0, 1e-9));
      }),
    );

    test(
      '3D Stacked Complex64 Matrix Determinant',
      () => NDArray.scope(() {
        // Matrix 0: [[1+1i, 2-1i], [3+0i, 4+2i]] -> det = -4 + 9i
        // Matrix 1: [[2+0i, 0+1i], [0-1i, 3+0i]] -> det = 6 - (i * -i) = 6 - 1 = 5
        final a = NDArray.fromList(
          [
            // Matrix 0
            Complex(1.0, 1.0), Complex(2.0, -1.0),
            Complex(3.0, 0.0), Complex(4.0, 2.0),
            // Matrix 1
            Complex(2.0, 0.0), Complex(0.0, 1.0),
            Complex(0.0, -1.0), Complex(3.0, 0.0),
          ],
          [2, 2, 2],
          DType.complex64,
        );

        final d = det<Complex>(a);

        expect(d.shape, [2]);
        expect(d.data[0].real, closeTo(-4.0, 1e-4));
        expect(d.data[0].imag, closeTo(9.0, 1e-4));
        expect(d.data[1].real, closeTo(5.0, 1e-4));
        expect(d.data[1].imag, closeTo(0.0, 1e-4));
      }),
    );
  });
}
