import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';

void main() {
  group('Type Safety Fixes (Phase 3)', () {
    test('floor_divide and remainder with uint8 and int16', () {
      NDArray.scope(() {
        final a = NDArray.fromList([10, 20, 30, 40], [4], DType.uint8);
        final b = NDArray.fromList([3, 3, 3, 3], [4], DType.uint8);

        // floor_divide
        final resDiv = floor_divide(a, b);
        expect(resDiv.dtype, DType.uint8);
        expect(resDiv.toList(), [3, 6, 10, 13]);

        // remainder
        final resRem = remainder(a, b);
        expect(resRem.dtype, DType.uint8);
        expect(resRem.toList(), [1, 2, 0, 1]);

        // int16
        final a16 = NDArray.fromList([10, -20, 30, -40], [4], DType.int16);
        final b16 = NDArray.fromList([3, 3, 3, 3], [4], DType.int16);

        final resDiv16 = floor_divide(a16, b16);
        expect(resDiv16.dtype, DType.int16);
        expect(resDiv16.toList(), [3, -7, 10, -14]);

        final resRem16 = remainder(a16, b16);
        expect(resRem16.dtype, DType.int16);
        expect(resRem16.toList(), [1, 1, 0, 2]);
      });
    });

    test('sin and abs with uint8 and int16', () {
      NDArray.scope(() {
        final a = NDArray.fromList([0, 30, 90], [3], DType.uint8);

        final resSin = sin(a);
        expect(resSin.dtype, DType.float64);
        expect(resSin.toList()[0], closeTo(0.0, 1e-5));

        // abs
        final a16 = NDArray.fromList([-10, 0, 20], [3], DType.int16);
        final resAbs = abs(a16);
        expect(resAbs.dtype, DType.int16);
        expect(resAbs.toList(), [10, 0, 20]);
      });
    });

    test('negative with uint8 and int16 (no silent fallthrough)', () {
      NDArray.scope(() {
        final a = NDArray.fromList([-10, 0, 20], [3], DType.int16);
        final resNeg = negative(a);
        expect(resNeg.dtype, DType.int16);
        expect(resNeg.toList(), [10, 0, -20]);

        final a8 = NDArray.fromList([10, 0, 20], [3], DType.uint8);
        final resNeg8 = negative(a8);
        expect(resNeg8.dtype, DType.uint8);
        expect(resNeg8.toList(), [246, 0, 236]);
      });
    });

    test('det type consistency for float32', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float32);
        final d = det(a);
        expect(d.dtype, DType.float32);
        expect(d.toList()[0], closeTo(-2.0, 1e-5));
      });
    });

    test('svd and qr throw ArgumentError for integer inputs', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        expect(() => svd(a), throwsArgumentError);
        expect(() => qr(a), throwsArgumentError);

        final a8 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.uint8);
        expect(() => svd(a8), throwsArgumentError);
        expect(() => qr(a8), throwsArgumentError);
      });
    });

    test('complex SVD and pinv', () {
      NDArray.scope(() {
        // complex128
        final a = NDArray.fromList(
          [Complex(1, 2), Complex(3, 4), Complex(5, 6), Complex(7, 8)],
          [2, 2],
          DType.complex128,
        );

        final svdRes = svd(a);
        expect(svdRes.U.dtype, DType.complex128);
        expect(svdRes.S.dtype, DType.float64);
        expect(svdRes.Vh.dtype, DType.complex128);

        final sDiag = diag(svdRes.S);
        final NDArray<Complex> uS = matmul(svdRes.U, sDiag);
        final reconstructed = matmul(uS, svdRes.Vh);

        expect(allClose(reconstructed, a, rtol: 1e-5, atol: 1e-5), isTrue);

        // complex64
        final a64 = NDArray.fromList(
          [Complex(1, 2), Complex(3, 4), Complex(5, 6), Complex(7, 8)],
          [2, 2],
          DType.complex64,
        );

        final svdRes64 = svd(a64);
        expect(svdRes64.U.dtype, DType.complex64);
        expect(svdRes64.S.dtype, DType.float32);
        expect(svdRes64.Vh.dtype, DType.complex64);

        final sDiag64 = diag(svdRes64.S);
        final NDArray<Complex> uS64 = matmul(svdRes64.U, sDiag64);
        final reconstructed64 = matmul(uS64, svdRes64.Vh);
        expect(
          allClose(reconstructed64, a64, rtol: 1e-3, atol: 1e-3),
          isTrue,
        ); // Lower tolerance for float32

        // complex pinv
        final aPinv = pinv(a);
        expect(aPinv.dtype, DType.complex128);

        final aPinvA = matmul(a, aPinv);
        final aPinvAA = matmul(aPinvA, a);
        expect(allClose(aPinvAA, a, rtol: 1e-5, atol: 1e-5), isTrue);

        final aPinv64 = pinv(a64);
        expect(aPinv64.dtype, DType.complex64);
      });
    });
  });
}
