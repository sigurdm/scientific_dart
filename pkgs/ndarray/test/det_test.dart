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
  });
}
