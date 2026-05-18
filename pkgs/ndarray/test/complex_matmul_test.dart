import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('NDArray Complex Matrix Multiplication Tests', () {
    test(
      'Complex128 Matrix Multiplication',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [
            Complex(1.0, 2.0),
            Complex(3.0, 4.0),
            Complex(5.0, 6.0),
            Complex(7.0, 8.0),
          ],
          [2, 2],
          DType.complex128,
        );

        final b = NDArray.fromList(
          [
            Complex(9.0, 10.0),
            Complex(11.0, 12.0),
            Complex(13.0, 14.0),
            Complex(15.0, 16.0),
          ],
          [2, 2],
          DType.complex128,
        );

        // a * b = [ [ (1+2i)(9+10i) + (3+4i)(13+14i), (1+2i)(11+12i) + (3+4i)(15+16i) ],
        //           [ (5+6i)(9+10i) + (7+8i)(13+14i), (5+6i)(11+12i) + (7+8i)(15+16i) ] ]
        //
        // (1+2i)(9+10i) = 9 + 10i + 18i - 20 = -11 + 28i
        // (3+4i)(13+14i) = 39 + 42i + 52i - 56 = -17 + 94i
        // Sum = -28 + 122i

        final result = matmul(a, b);

        expect(result.shape, [2, 2]);
        expect(result.dtype, DType.complex128);

        final Complex c00 = result.data[0] as Complex;
        expect(c00.real, closeTo(-28.0, 1e-9));
        expect(c00.imag, closeTo(122.0, 1e-9));
      }),
    );

    test(
      'Complex64 Matrix Multiplication',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [
            Complex(1.0, 2.0),
            Complex(3.0, 4.0),
            Complex(5.0, 6.0),
            Complex(7.0, 8.0),
          ],
          [2, 2],
          DType.complex64,
        );

        final b = NDArray.fromList(
          [
            Complex(9.0, 10.0),
            Complex(11.0, 12.0),
            Complex(13.0, 14.0),
            Complex(15.0, 16.0),
          ],
          [2, 2],
          DType.complex64,
        );

        final result = matmul(a, b);

        expect(result.shape, [2, 2]);
        expect(result.dtype, DType.complex64);

        final Complex c00 = result.data[0] as Complex;
        expect(c00.real, closeTo(-28.0, 1e-5));
        expect(c00.imag, closeTo(122.0, 1e-5));
      }),
    );

    test(
      'Complex128 Transposed Matrix Multiplication',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [
            Complex(1.0, 2.0),
            Complex(3.0, 4.0),
            Complex(5.0, 6.0),
            Complex(7.0, 8.0),
          ],
          [2, 2],
          DType.complex128,
        );

        final b = NDArray.fromList(
          [
            Complex(9.0, 10.0),
            Complex(11.0, 12.0),
            Complex(13.0, 14.0),
            Complex(15.0, 16.0),
          ],
          [2, 2],
          DType.complex128,
        );

        final result = matmul(a.transpose(), b);

        expect(result.shape, [2, 2]);
        expect(result.dtype, DType.complex128);

        final Complex c00 = result.data[0] as Complex;
        expect(c00.real, closeTo(-30.0, 1e-9));
        expect(c00.imag, closeTo(176.0, 1e-9));
      }),
    );

    test(
      'Complex128 Sliced Matrix Multiplication',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [
            Complex(1.0, 2.0),
            Complex(3.0, 4.0),
            Complex(5.0, 6.0),
            Complex(7.0, 8.0),
            Complex(9.0, 10.0),
            Complex(11.0, 12.0),
          ],
          [2, 3],
          DType.complex128,
        );

        final b = NDArray.fromList(
          [
            Complex(9.0, 10.0),
            Complex(11.0, 12.0),
            Complex(13.0, 14.0),
            Complex(15.0, 16.0),
            Complex(17.0, 18.0),
            Complex(19.0, 20.0),
          ],
          [3, 2],
          DType.complex128,
        );

        final aSlice = a.slice([const Slice(), const Slice(start: 0, stop: 2)]);
        final bSlice = b.slice([const Slice(start: 0, stop: 2), const Slice()]);

        final result = matmul(aSlice, bSlice);

        expect(result.shape, [2, 2]);
        expect(result.dtype, DType.complex128);

        final Complex c00 = result.data[0] as Complex;
        expect(c00.real, closeTo(-28.0, 1e-9));
        expect(c00.imag, closeTo(122.0, 1e-9));
      }),
    );
  });
}
