import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';
import 'dart:math' as math;
import 'dart:typed_data';

void main() {
  group('NDArray Advanced LAPACK Linear Algebra Decompositions Tests', () {
    group('LAPACK Inversion (inv) tests', () {
      test(
        'Invert Float64 square matrix and check identity reconstruction',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([4.0, 7.0, 2.0, 6.0]),
            [2, 2],
            DType.float64,
          );

          final aInv = inv(a);
          expect(aInv.shape, [2, 2]);
          expect(aInv.dtype, DType.float64);

          // Check exact values (approx 0.6, -0.7, -0.2, 0.4)
          expect(aInv.data[0], closeTo(0.6, 1e-6));
          expect(aInv.data[1], closeTo(-0.7, 1e-6));
          expect(aInv.data[2], closeTo(-0.2, 1e-6));
          expect(aInv.data[3], closeTo(0.4, 1e-6));
        }),
      );

      test(
        'Invert Float32 matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float32,
          );

          final aInv = inv(a);
          expect(aInv.dtype, DType.float32);
          // expected inverse for [[1,2],[3,4]] is [[-2, 1], [1.5, -0.5]]
          expect(aInv.data[0], closeTo(-2.0, 1e-5));
          expect(aInv.data[1], closeTo(1.0, 1e-5));
          expect(aInv.data[2], closeTo(1.5, 1e-5));
          expect(aInv.data[3], closeTo(-0.5, 1e-5));
        }),
      );

      test(
        'Singular matrix throws ArgumentError',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 2.0, 4.0]),
            [2, 2],
            DType.float64,
          ); // dependent rows, det = 0

          expect(() => inv(a), throwsArgumentError);
        }),
      );

      test(
        'Singular Float32 matrix throws ArgumentError',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float32List.fromList([1.0, 2.0, 2.0, 4.0]),
            [2, 2],
            DType.float32,
          ); // dependent rows, det = 0

          expect(() => inv(a), throwsArgumentError);
        }),
      );
    });

    group('Cholesky Decomposition tests', () {
      test(
        'Cholesky lower triangular factor and L * L^T reconstruction',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([
              4.0,
              12.0,
              -16.0,
              12.0,
              37.0,
              -43.0,
              -16.0,
              -43.0,
              98.0,
            ]),
            [3, 3],
            DType.float64,
          );

          final l = cholesky(a);

          expect(l.shape, [3, 3]);

          // Check that the strictly upper triangular elements are zeroed out!
          expect(l.data[1], 0.0);
          expect(l.data[2], 0.0);
          expect(l.data[5], 0.0);

          // Check exact expected lower triangular values
          // L = [[2, 0, 0], [6, 1, 0], [-8, 5, 3]]
          expect(l.data[0], closeTo(2.0, 1e-6));
          expect(l.data[3], closeTo(6.0, 1e-6));
          expect(l.data[4], closeTo(1.0, 1e-6));
          expect(l.data[6], closeTo(-8.0, 1e-6));
          expect(l.data[7], closeTo(5.0, 1e-6));
          expect(l.data[8], closeTo(3.0, 1e-6));
        }),
      );

      test(
        'Cholesky Float32 matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float32List.fromList([4.0, 2.0, 2.0, 2.0]),
            [2, 2],
            DType.float32,
          );
          final l = cholesky(a);
          expect(l.dtype, DType.float32);
          // L = [[2, 0], [1, 1]]
          expect(l.data[0], closeTo(2.0, 1e-5));
          expect(l.data[1], 0.0);
          expect(l.data[2], closeTo(1.0, 1e-5));
          expect(l.data[3], closeTo(1.0, 1e-5));
        }),
      );

      test(
        'Cholesky Complex128 matrix and check exact factor values',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(4.0, 0.0),
              Complex(2.0, 1.0),
              Complex(2.0, -1.0),
              Complex(3.0, 0.0),
            ],
            [2, 2],
            DType.complex128,
          );
          final l = cholesky(a);
          expect(l.dtype, DType.complex128);

          // Expected: L = [[2.0, 0.0], [1.0 - 0.5i, sqrt(1.75)]]
          expect(l.data[0].real, closeTo(2.0, 1e-9));
          expect(l.data[0].imag, closeTo(0.0, 1e-9));

          expect(l.data[1].real, closeTo(0.0, 1e-9));
          expect(l.data[1].imag, closeTo(0.0, 1e-9));

          expect(l.data[2].real, closeTo(1.0, 1e-9));
          expect(l.data[2].imag, closeTo(-0.5, 1e-9));

          expect(l.data[3].real, closeTo(math.sqrt(1.75), 1e-9));
          expect(l.data[3].imag, closeTo(0.0, 1e-9));
        }),
      );

      test(
        'Cholesky Complex64 matrix and check exact factor values',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [
              Complex(4.0, 0.0),
              Complex(2.0, 1.0),
              Complex(2.0, -1.0),
              Complex(3.0, 0.0),
            ],
            [2, 2],
            DType.complex64,
          );
          final l = cholesky(a);
          expect(l.dtype, DType.complex64);

          // Expected: L = [[2.0, 0.0], [1.0 - 0.5i, sqrt(1.75)]]
          expect(l.data[0].real, closeTo(2.0, 1e-5));
          expect(l.data[0].imag, closeTo(0.0, 1e-5));

          expect(l.data[1].real, closeTo(0.0, 1e-5));
          expect(l.data[1].imag, closeTo(0.0, 1e-5));

          expect(l.data[2].real, closeTo(1.0, 1e-5));
          expect(l.data[2].imag, closeTo(-0.5, 1e-5));

          expect(l.data[3].real, closeTo(math.sqrt(1.75), 1e-5));
          expect(l.data[3].imag, closeTo(0.0, 1e-5));
        }),
      );

      test(
        'Cholesky with pre-allocated contiguous out parameter',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([
              4.0,
              12.0,
              -16.0,
              12.0,
              37.0,
              -43.0,
              -16.0,
              -43.0,
              98.0,
            ]),
            [3, 3],
            DType.float64,
          );

          final outL = NDArray<double>.zeros([3, 3], DType.float64);
          final res = cholesky(a, out: outL);
          expect(identical(res, outL), true);

          // Check that the strictly upper triangular elements are zeroed out!
          expect(outL.data[1], 0.0);
          expect(outL.data[2], 0.0);
          expect(outL.data[5], 0.0);

          // Check exact expected lower triangular values
          expect(outL.data[0], closeTo(2.0, 1e-6));
          expect(outL.data[3], closeTo(6.0, 1e-6));
          expect(outL.data[4], closeTo(1.0, 1e-6));
          expect(outL.data[6], closeTo(-8.0, 1e-6));
          expect(outL.data[7], closeTo(5.0, 1e-6));
          expect(outL.data[8], closeTo(3.0, 1e-6));
        }),
      );
    });

    group('QR Decomposition tests', () {
      test(
        'QR orthogonal Q and upper triangular R verification',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([
              12.0,
              -51.0,
              4.0,
              6.0,
              167.0,
              -68.0,
              -4.0,
              24.0,
              -41.0,
            ]),
            [3, 3],
            DType.float64,
          );

          final res = qr(a);
          final q = res.Q;
          final r = res.R;

          expect(q.shape, [3, 3]);
          expect(r.shape, [3, 3]);

          // Verify R is upper triangular: elements under diagonal are exactly 0
          expect(r.data[3], 0.0);
          expect(r.data[6], 0.0);
          expect(r.data[7], 0.0);

          // Verify Q is orthogonal: Q * Q^T should equal Identity Matrix!
          // To perform matrix multiplication, we could loop or verify dot products.
          // Let's just do a custom check: row 0 dot row 0 should be 1.0, row 0 dot row 1 should be 0.0!
          double dotRow(int rA, int rB) {
            var s = 0.0;
            for (var i = 0; i < 3; i++) {
              s += q.data[rA * 3 + i] * q.data[rB * 3 + i];
            }
            return s;
          }

          expect(dotRow(0, 0), closeTo(1.0, 1e-9));
          expect(dotRow(1, 1), closeTo(1.0, 1e-9));
          expect(dotRow(2, 2), closeTo(1.0, 1e-9));
          expect(dotRow(0, 1), closeTo(0.0, 1e-9));
          expect(dotRow(0, 2), closeTo(0.0, 1e-9));
        }),
      );

      test(
        'QR Float32 matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float32List.fromList([
              12.0,
              -51.0,
              4.0,
              6.0,
              167.0,
              -68.0,
              -4.0,
              24.0,
              -41.0,
            ]),
            [3, 3],
            DType.float32,
          );
          final res = qr(a);
          expect(res.Q.dtype, DType.float32);
          expect(res.R.dtype, DType.float32);
          expect(res.R.data[3], 0.0);
        }),
      );

      test(
        'QR non-contiguous view matrix',
        () => NDArray.scope(() {
          final parent = NDArray.fromList(
            Float64List.fromList([
              12.0,
              6.0,
              -4.0,
              -51.0,
              167.0,
              24.0,
              4.0,
              -68.0,
              -41.0,
            ]),
            [3, 3],
            DType.float64,
          );
          final view = parent.transpose();
          expect(view.isContiguous, false);
          final res = qr(view);
          expect(res.Q.shape, [3, 3]);
          expect(res.R.shape, [3, 3]);
          expect(res.R.data[3], 0.0);
        }),
      );

      test(
        'QR with out parameter recycling buffers',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([
              12.0,
              -51.0,
              4.0,
              6.0,
              167.0,
              -68.0,
              -4.0,
              24.0,
              -41.0,
            ]),
            [3, 3],
            DType.float64,
          );

          final qBuffer = NDArray<double>.zeros([3, 3], DType.float64);
          final rBuffer = NDArray<double>.zeros([3, 3], DType.float64);

          final res = qr(a, out: (Q: qBuffer, R: rBuffer));

          expect(identical(res.Q, qBuffer), true);
          expect(identical(res.R, rBuffer), true);

          // Verify correctness of QR values in the recycled buffers
          expect(res.R.data[3], 0.0);
          expect(res.R.data[6], 0.0);
          expect(res.R.data[7], 0.0);
        }),
      );

      test(
        'QRRecordDispose extension',
        () => NDArray.scope(() {
          // Check the extension method on records
          final a2 = NDArray.fromList(
            Float64List.fromList([
              12.0,
              -51.0,
              4.0,
              6.0,
              167.0,
              -68.0,
              -4.0,
              24.0,
              -41.0,
            ]),
            [3, 3],
            DType.float64,
          );
          final res2 = qr(a2);
          expect(res2.Q.isDisposed, false);
          expect(res2.R.isDisposed, false);

          res2.dispose();
          expect(res2.Q.isDisposed, true);
          expect(res2.R.isDisposed, true);
        }),
      );

      test(
        'QR throws ArgumentError on incompatible shape of out parameter',
        () => NDArray.scope(() {
          final a = NDArray.zeros([3, 3], DType.float64);

          // Wrong shape for Q
          final qBufferWrongShape = NDArray<double>.zeros([
            2,
            3,
          ], DType.float64);
          final rBuffer = NDArray<double>.zeros([3, 3], DType.float64);
          expect(
            () => qr(a, out: (Q: qBufferWrongShape, R: rBuffer)),
            throwsArgumentError,
          );

          // Wrong shape for R
          final qBuffer = NDArray<double>.zeros([3, 3], DType.float64);
          final rBufferWrongShape = NDArray<double>.zeros([
            3,
            2,
          ], DType.float64);
          expect(
            () => qr(a, out: (Q: qBuffer, R: rBufferWrongShape)),
            throwsArgumentError,
          );
        }),
      );
    });

    group('SVD Decomposition tests', () {
      test(
        'SVD full matrices factorizations checks',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            [3, 2],
            DType.float64,
          );

          final res = svd(a);
          final u = res.U;
          final s = res.S;
          final vh = res.Vh;

          expect(u.shape, [3, 3]); // full matrix
          expect(s.shape, [2]); // min(3, 2) vector
          expect(vh.shape, [2, 2]); // full matrix

          // Singular values in S must be sorted in descending order by LAPACK rules!
          expect(s.data[0], greaterThan(s.data[1]));
          expect(s.data[0], greaterThan(0.0));
        }),
      );

      test(
        'SVD Float32 matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            [3, 2],
            DType.float32,
          );
          final res = svd(a);
          expect(res.U.dtype, DType.float32);
          expect(res.S.dtype, DType.float32);
          expect(res.Vh.dtype, DType.float32);
          expect(res.U.shape, [3, 3]);
        }),
      );

      test(
        'SVD non-contiguous view matrix',
        () => NDArray.scope(() {
          final parent = NDArray.fromList(
            Float64List.fromList([1.0, 3.0, 5.0, 2.0, 4.0, 6.0]),
            [2, 3],
            DType.float64,
          );
          final view = parent.transpose();
          expect(view.isContiguous, false);
          final res = svd(view);
          expect(res.U.shape, [3, 3]);
          expect(res.S.shape, [2]);
        }),
      );
    });

    group('Determinant (det) tests', () {
      test(
        'Determinant of Float64 2D square matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float64,
          );
          expect(det(a).scalar, closeTo(-2.0, 1e-9));
        }),
      );

      test(
        'Determinant of Float32 2D square matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float32List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float32,
          );
          expect(det(a).scalar, closeTo(-2.0, 1e-5));
        }),
      );

      test(
        'det() throws ArgumentError on non-square matrix',
        () => NDArray.scope(() {
          final a = NDArray<double>.zeros([2, 3], DType.float64);
          expect(() => det(a), throwsArgumentError);
        }),
      );
    });

    group('Linear Solver (solve) tests', () {
      test(
        'Solve Float64 system of equations',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([3.0, 1.0, 1.0, 2.0]),
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(Float64List.fromList([9.0, 8.0]), [
            2,
            1,
          ], DType.float64);
          final x = solve(a, b);
          expect(x.shape, [2, 1]);
          expect(x.data[0], closeTo(2.0, 1e-9));
          expect(x.data[1], closeTo(3.0, 1e-9));
        }),
      );

      test(
        'Solve Float32 system of equations',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float32List.fromList([3.0, 1.0, 1.0, 2.0]),
            [2, 2],
            DType.float32,
          );
          final b = NDArray.fromList(Float32List.fromList([9.0, 8.0]), [
            2,
            1,
          ], DType.float32);
          final x = solve(a, b);
          expect(x.shape, [2, 1]);
          expect(x.data[0], closeTo(2.0, 1e-5));
          expect(x.data[1], closeTo(3.0, 1e-5));
        }),
      );

      test(
        'Solve Complex128 system of equations',
        () => NDArray.scope(() {
          final a = NDArray<Complex>.create([2, 2], DType.complex128);
          a.data[0] = Complex(3.0, 0.0);
          a.data[1] = Complex(1.0, 0.0);
          a.data[2] = Complex(1.0, 0.0);
          a.data[3] = Complex(2.0, 0.0);

          final b = NDArray<Complex>.create([2, 1], DType.complex128);
          b.data[0] = Complex(9.0, 0.0);
          b.data[1] = Complex(8.0, 0.0);

          final x = solve(a, b);
          expect(x.dtype, DType.complex128);
          expect(x.data[0], Complex(2.0, 0.0));
          expect(x.data[1], Complex(3.0, 0.0));
        }),
      );

      test(
        'Singular matrix solve throws ArgumentError',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 2.0, 4.0]),
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(Float64List.fromList([9.0, 8.0]), [
            2,
            1,
          ], DType.float64);
          expect(() => solve(a, b), throwsArgumentError);
        }),
      );

      test(
        'Solve integer matrices system throws ArgumentError',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([3, 1, 1, 2]), [
            2,
            2,
          ], DType.int32);
          final b = NDArray.fromList(Int32List.fromList([9, 8]), [
            2,
            1,
          ], DType.int32);
          expect(() => solve(a, b), throwsArgumentError);
        }),
      );

      test(
        'Solve Float64 system of equations with out parameter',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([3.0, 1.0, 1.0, 2.0]),
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(Float64List.fromList([9.0, 8.0]), [
            2,
            1,
          ], DType.float64);
          final outBuffer = NDArray<double>.zeros([2, 1], DType.float64);
          final x = solve(a, b, out: outBuffer);
          expect(identical(x, outBuffer), true);
          expect(x.shape, [2, 1]);
          expect(x.data[0], closeTo(2.0, 1e-9));
          expect(x.data[1], closeTo(3.0, 1e-9));
        }),
      );

      test(
        'Solve Complex64 system of equations',
        () => NDArray.scope(() {
          final a = NDArray<Complex>.create([2, 2], DType.complex64);
          a.data[0] = Complex(3.0, 0.0);
          a.data[1] = Complex(1.0, 0.0);
          a.data[2] = Complex(1.0, 0.0);
          a.data[3] = Complex(2.0, 0.0);

          final b = NDArray<Complex>.create([2, 1], DType.complex64);
          b.data[0] = Complex(9.0, 0.0);
          b.data[1] = Complex(8.0, 0.0);

          final x = solve(a, b);
          expect(x.dtype, DType.complex64);
          expect(x.data[0].real, closeTo(2.0, 1e-5));
          expect(x.data[1].real, closeTo(3.0, 1e-5));
        }),
      );
    });

    group('Eigenvalues (eig) tests', () {
      test(
        'Eigenvalues/eigenvectors of Float64 2D square matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([0.0, 1.0, -2.0, -3.0]),
            [2, 2],
            DType.float64,
          );
          final (eigenvalues: w, eigenvectors: vr) = eig(a);

          expect(w.shape, [2]);
          expect(vr.shape, [2, 2]);

          expect(w.data[0].real, closeTo(-1.0, 1e-9));
          expect(w.data[0].imag, closeTo(0.0, 1e-9));
          expect(w.data[1].real, closeTo(-2.0, 1e-9));
          expect(w.data[1].imag, closeTo(0.0, 1e-9));
        }),
      );

      test(
        'Eigenvalues of Float32 2D square matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float32List.fromList([0.0, 1.0, -2.0, -3.0]),
            [2, 2],
            DType.float32,
          );
          final (eigenvalues: w, eigenvectors: _) = eig(a);
          expect(w.dtype, DType.complex64);
          expect(w.data[0].real, closeTo(-1.0, 1e-5));
          expect(w.data[1].real, closeTo(-2.0, 1e-5));
        }),
      );

      test(
        'Eigenvalues/eigenvectors of Complex128 matrix',
        () => NDArray.scope(() {
          final a = NDArray<Complex>.create([2, 2], DType.complex128);
          a.data[0] = Complex(0.0, 0.0);
          a.data[1] = Complex(1.0, 0.0);
          a.data[2] = Complex(-2.0, 0.0);
          a.data[3] = Complex(-3.0, 0.0);

          final (eigenvalues: w, eigenvectors: _) = eig(a);
          expect(w.dtype, DType.complex128);
          expect(w.data[0].real, closeTo(-1.0, 1e-9));
          expect(w.data[1].real, closeTo(-2.0, 1e-9));
        }),
      );

      test(
        'Eigenvalues/eigenvectors of Complex64 matrix',
        () => NDArray.scope(() {
          final a = NDArray<Complex>.create([2, 2], DType.complex64);
          a.data[0] = Complex(0.0, 0.0);
          a.data[1] = Complex(1.0, 0.0);
          a.data[2] = Complex(-2.0, 0.0);
          a.data[3] = Complex(-3.0, 0.0);

          final (eigenvalues: w, eigenvectors: _) = eig(a);
          expect(w.dtype, DType.complex64);
          expect(w.data[0].real, closeTo(-1.0, 1e-5));
          expect(w.data[1].real, closeTo(-2.0, 1e-5));
        }),
      );

      test(
        'Eigenvalues/eigenvectors with out parameter recycler',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([0.0, 1.0, -2.0, -3.0]),
            [2, 2],
            DType.float64,
          );
          final outW = NDArray<Complex>.zeros([2], DType.complex128);
          final outVR = NDArray<Complex>.zeros([2, 2], DType.complex128);

          final (eigenvalues: w, eigenvectors: vr) = eig(
            a,
            out: (eigenvalues: outW, eigenvectors: outVR),
          );

          expect(identical(w, outW), true);
          expect(identical(vr, outVR), true);

          expect(w.data[0].real, closeTo(-1.0, 1e-9));
          expect(w.data[1].real, closeTo(-2.0, 1e-9));
        }),
      );
    });

    group('Linear Algebra Validation & Error Exceptions tests', () {
      test(
        'cholesky() throws ArgumentError on non-square matrix',
        () => NDArray.scope(() {
          final a = NDArray.zeros([2, 3], DType.float64);
          expect(() => cholesky(a), throwsArgumentError);
        }),
      );

      test(
        'cholesky() throws ArgumentError on non positive-definite matrix',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([-1.0, 0.0, 0.0, -1.0]),
            [2, 2],
            DType.float64,
          );
          expect(() => cholesky(a), throwsArgumentError);
        }),
      );

      test(
        'cholesky() throws ArgumentError on unsupported dtype (int32)',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([4, 2, 2, 2]), [
            2,
            2,
          ], DType.int32);
          expect(() => cholesky(a), throwsArgumentError);
        }),
      );

      test(
        'qr() throws ArgumentError on non-2D tensor',
        () => NDArray.scope(() {
          final a = NDArray.zeros([3], DType.float64);
          expect(() => qr(a), throwsArgumentError);
        }),
      );

      test(
        'svd() throws ArgumentError on non-2D tensor',
        () => NDArray.scope(() {
          final a = NDArray.zeros([3], DType.float64);
          expect(() => svd(a), throwsArgumentError);
        }),
      );

      test(
        'eig() throws ArgumentError on non-square matrix',
        () => NDArray.scope(() {
          final a = NDArray.zeros([2, 3], DType.float64);
          expect(() => eig(a), throwsArgumentError);
        }),
      );

      test(
        'eig() throws UnimplementedError on boolean matrix',
        () => NDArray.scope(() {
          final a = NDArray.zeros([2, 2], DType.boolean);
          expect(() => eig(a), throwsUnimplementedError);
        }),
      );

      test(
        'solve() throws ArgumentError on non-square matrix a',
        () => NDArray.scope(() {
          final a = NDArray.zeros([2, 3], DType.float64);
          final b = NDArray.zeros([2], DType.float64);
          expect(() => solve(a, b), throwsArgumentError);
        }),
      );

      test(
        'solve() throws ArgumentError on mismatched b first dimension',
        () => NDArray.scope(() {
          final a = NDArray.zeros([2, 2], DType.float64);
          final b = NDArray.zeros([3], DType.float64);
          expect(() => solve(a, b), throwsArgumentError);
        }),
      );

      test(
        'solve() throws ArgumentError on empty/scalar b array',
        () => NDArray.scope(() {
          final a = NDArray.zeros([2, 2], DType.float64);
          final b = NDArray.zeros([], DType.float64);
          expect(() => solve(a, b), throwsArgumentError);
        }),
      );
    });

    group('Least-Squares Solver (lstsq) tests', () {
      test(
        'Solve over-determined system (m > n)',
        () => NDArray.scope(() {
          // y = c + m * x
          // Data points: (1, 2), (2, 3.9), (3, 6.1)
          // A = [[1, 1], [1, 2], [1, 3]]
          // B = [2.0, 3.9, 6.1]
          final a = NDArray.fromList(
            [1.0, 1.0, 1.0, 2.0, 1.0, 3.0],
            [3, 2],
            DType.float64,
          );
          final b = NDArray.fromList([2.0, 3.9, 6.1], [3], DType.float64);

          final res = lstsq(a, b);

          expect(res.rank, 2);
          expect(res.s.shape, [2]);
          expect(res.residuals.shape, [1]);

          // Analytical solution: c = -0.1, m = 2.05
          expect(res.x.data[0], closeTo(-0.1, 1e-9));
          expect(res.x.data[1], closeTo(2.05, 1e-9));

          // Residual: sum of squared errors = 0.05^2 + (-0.1)^2 + 0.05^2 = 0.0025 + 0.01 + 0.0025 = 0.015
          expect(res.residuals.data[0], closeTo(0.015, 1e-9));
        }),
      );

      test(
        'Solve under-determined system (m < n) returns minimum norm solution',
        () => NDArray.scope(() {
          // A = [[1.0, 2.0, 3.0]]
          // B = [6.0]
          final a = NDArray.fromList([1.0, 2.0, 3.0], [1, 3], DType.float64);
          final b = NDArray.fromList([6.0], [1], DType.float64);

          final res = lstsq(a, b);

          expect(res.rank, 1);
          expect(res.residuals.shape, [0]); // no residuals computed

          // Minimum norm solution: x_i = 6 * a_i / sum(a_i^2) = 6 * a_i / 14
          expect(res.x.data[0], closeTo(6.0 / 14.0, 1e-9));
          expect(res.x.data[1], closeTo(12.0 / 14.0, 1e-9));
          expect(res.x.data[2], closeTo(18.0 / 14.0, 1e-9));
        }),
      );

      test(
        'Solve full rank square matrix matching solve()',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [3.0, 1.0, 1.0, 2.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList([9.0, 8.0], [2], DType.float64);

          final res = lstsq(a, b);
          final xSolve = solve(a, b);

          expect(res.rank, 2);
          expect(res.residuals.shape, [
            0,
          ]); // square full rank -> empty residuals
          expect(res.x.data[0], closeTo(xSolve.data[0], 1e-9));
          expect(res.x.data[1], closeTo(xSolve.data[1], 1e-9));
        }),
      );

      test(
        'lstsq with float32 support',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float32,
          );
          final b = NDArray.fromList([5.0, 11.0], [2], DType.float32);

          final res = lstsq(a, b);
          expect(res.x.dtype, DType.float32);
          expect(res.x.data[0], closeTo(1.0, 1e-6));
          expect(res.x.data[1], closeTo(2.0, 1e-6));
        }),
      );

      test(
        'Complex128 least-squares solver',
        () => NDArray.scope(() {
          final a = NDArray<Complex>.create([2, 2], DType.complex128);
          a.data[0] = Complex(3.0, 0.0);
          a.data[1] = Complex(1.0, 0.0);
          a.data[2] = Complex(1.0, 0.0);
          a.data[3] = Complex(2.0, 0.0);

          final b = NDArray<Complex>.create([2, 1], DType.complex128);
          b.data[0] = Complex(9.0, 0.0);
          b.data[1] = Complex(8.0, 0.0);

          final res = lstsq(a, b);
          expect(res.x.dtype, DType.complex128);
          expect(res.residuals.dtype, DType.float64);
          expect(res.x.data[0].real, closeTo(2.0, 1e-9));
          expect(res.x.data[0].imag, closeTo(0.0, 1e-9));
          expect(res.x.data[1].real, closeTo(3.0, 1e-9));
          expect(res.x.data[1].imag, closeTo(0.0, 1e-9));
        }),
      );

      test(
        'lstsq using out recycler buffer',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 1.0, 1.0, 2.0, 1.0, 3.0],
            [3, 2],
            DType.float64,
          );
          final b = NDArray.fromList([2.0, 3.9, 6.1], [3], DType.float64);
          final outBuffer = NDArray<double>.zeros([2], DType.float64);

          final res = lstsq(a, b, out: outBuffer);

          expect(identical(res.x, outBuffer), true);
          expect(res.x.data[0], closeTo(-0.1, 1e-9));
          expect(res.x.data[1], closeTo(2.05, 1e-9));
        }),
      );
      test(
        'lstsq with integer matrices throws ArgumentError',
        () => NDArray.scope(() {
          final a = NDArray.fromList([1, 1, 1, 2, 1, 3], [3, 2], DType.int32);
          final b = NDArray.fromList([2, 4, 6], [3], DType.int32);
          expect(() => lstsq(a, b), throwsArgumentError);
        }),
      );
    });

    group('Optimal Matrix Chain Product (multi_dot) tests', () {
      test(
        'Multiply 3 matrices with optimal parenthesization',
        () => NDArray.scope(() {
          // shapes: A[2, 10] * B[10, 5] * C[5, 3] -> result [2, 3]
          final a = NDArray.ones([2, 10], DType.float64);
          final b = NDArray.ones([10, 5], DType.float64);
          final c = NDArray.ones([5, 3], DType.float64);

          final res = multi_dot([a, b, c]);

          expect(res.shape, [2, 3]);
          // Each cell = sum_{k} (1 * sum_{m} (1 * 1)) = 10 * 5 = 50
          for (var i = 0; i < 6; i++) {
            expect(res.data[i], 50.0);
          }
        }),
      );

      test(
        'multi_dot with 1D squeezing at first array',
        () => NDArray.scope(() {
          // A[10] * B[10, 5] * C[5, 3] -> result [3]
          final a = NDArray.ones([10], DType.float64);
          final b = NDArray.ones([10, 5], DType.float64);
          final c = NDArray.ones([5, 3], DType.float64);

          final res = multi_dot([a, b, c]);

          expect(res.shape, [3]);
          for (var i = 0; i < 3; i++) {
            expect(res.data[i], 50.0);
          }
        }),
      );

      test(
        'multi_dot with 1D squeezing at last array',
        () => NDArray.scope(() {
          // A[2, 10] * B[10, 5] * C[5] -> result [2]
          final a = NDArray.ones([2, 10], DType.float64);
          final b = NDArray.ones([10, 5], DType.float64);
          final c = NDArray.ones([5], DType.float64);

          final res = multi_dot([a, b, c]);

          expect(res.shape, [2]);
          for (var i = 0; i < 2; i++) {
            expect(res.data[i], 50.0);
          }
        }),
      );

      test(
        'multi_dot with squeezing at both first and last arrays',
        () => NDArray.scope(() {
          // A[10] * B[10, 5] * C[5] -> result [] (scalar)
          final a = NDArray.ones([10], DType.float64);
          final b = NDArray.ones([10, 5], DType.float64);
          final c = NDArray.ones([5], DType.float64);

          final res = multi_dot([a, b, c]);

          expect(res.shape, []);
          expect(res.scalar, 50.0);
        }),
      );

      test(
        'multi_dot using out recycler buffer',
        () => NDArray.scope(() {
          final a = NDArray.ones([2, 10], DType.float64);
          final b = NDArray.ones([10, 5], DType.float64);
          final c = NDArray.ones([5, 3], DType.float64);

          final out = NDArray<double>.zeros([2, 3], DType.float64);
          final res = multi_dot([a, b, c], out: out);

          expect(res == out, true);
          expect(out.shape, [2, 3]);
          for (var i = 0; i < 6; i++) {
            expect(out.data[i], 50.0);
          }
        }),
      );

      test(
        'Incompatible dimensions throw ArgumentError',
        () => NDArray.scope(() {
          final a = NDArray.ones([2, 10], DType.float64);
          final b = NDArray.ones([9, 5], DType.float64); // 9 != 10
          final c = NDArray.ones([5, 3], DType.float64);

          expect(() => multi_dot([a, b, c]), throwsArgumentError);
        }),
      );
    });

    group('Stacked/Batch Linear Algebra Tests', () {
      test(
        'Stacked det() double and float32',
        () => NDArray.scope(() {
          // Double precision stacked matrix [2, 2, 2]
          final a64 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2, 2, 2],
            DType.float64,
          );

          final d64 = det(a64);
          expect(d64.shape, [2]);
          expect(d64.dtype, DType.float64);
          expect(d64.data[0], closeTo(-2.0, 1e-9));
          expect(d64.data[1], closeTo(-2.0, 1e-9));

          // Float32 precision stacked matrix [2, 2, 2]
          final a32 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            [2, 2, 2],
            DType.float32,
          );

          final d32 = det(a32);
          expect(d32.shape, [2]);
          expect(d32.data[0], closeTo(-2.0, 1e-5));
          expect(d32.data[1], closeTo(-2.0, 1e-5));

          // High rank 4D stack det [2, 2, 2, 2]
          final a4d = NDArray.fromList(
            List<double>.generate(
              16,
              (i) => (i % 4) == 0 ? 2.0 : ((i % 4) == 3 ? 3.0 : 1.0),
            ),
            [2, 2, 2, 2],
            DType.float64,
          ); // Each matrix is [[2, 1], [1, 3]] -> det = 5.0
          final d4d = det(a4d);
          expect(d4d.shape, [2, 2]);
          for (var i = 0; i < 4; i++) {
            expect(d4d.data[i], closeTo(5.0, 1e-9));
          }
        }),
      );

      test(
        'Stacked eig()',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [0.0, 1.0, -2.0, -3.0, 0.0, 1.0, -2.0, -3.0],
            [2, 2, 2],
            DType.float64,
          );

          final (eigenvalues: w, eigenvectors: vr) = eig(a);

          expect(w.shape, [2, 2]);
          expect(vr.shape, [2, 2, 2]);
          expect(w.dtype, DType.complex128);
          expect(vr.dtype, DType.complex128);

          // Slice 0
          expect(w.data[0].real, closeTo(-1.0, 1e-9));
          expect(w.data[1].real, closeTo(-2.0, 1e-9));
          // Slice 1
          expect(w.data[2].real, closeTo(-1.0, 1e-9));
          expect(w.data[3].real, closeTo(-2.0, 1e-9));
        }),
      );

      test(
        'Stacked qr()',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            [
              12.0,
              -51.0,
              4.0,
              6.0,
              167.0,
              -68.0,
              -4.0,
              24.0,
              -41.0,
              12.0,
              -51.0,
              4.0,
              6.0,
              167.0,
              -68.0,
              -4.0,
              24.0,
              -41.0,
            ],
            [2, 3, 3],
            DType.float64,
          );

          final res = qr(a);
          final q = res.Q;
          final r = res.R;

          expect(q.shape, [2, 3, 3]);
          expect(r.shape, [2, 3, 3]);

          // Verify upper triangular slice 0 and 1
          expect(r.data[3], 0.0); // r[0, 1, 0]
          expect(r.data[6], 0.0); // r[0, 2, 0]
          expect(r.data[7], 0.0); // r[0, 2, 1]

          expect(r.data[12], 0.0); // r[1, 1, 0]
          expect(r.data[15], 0.0); // r[1, 2, 0]
          expect(r.data[16], 0.0); // r[1, 2, 1]

          // Verify Q slice 0 orthogonality: dot product of row 0 and row 1 is 0
          double dotRow(NDArray mat, int slice, int rA, int rB) {
            var s = 0.0;
            final offset = slice * 9;
            for (var i = 0; i < 3; i++) {
              s +=
                  mat.data[offset + rA * 3 + i] * mat.data[offset + rB * 3 + i];
            }
            return s;
          }

          expect(dotRow(q, 0, 0, 0), closeTo(1.0, 1e-9));
          expect(dotRow(q, 0, 0, 1), closeTo(0.0, 1e-9));
          expect(dotRow(q, 1, 0, 0), closeTo(1.0, 1e-9));
          expect(dotRow(q, 1, 0, 1), closeTo(0.0, 1e-9));
        }),
      );

      test(
        'Stacked svd() tall and wide',
        () => NDArray.scope(() {
          // 1. Tall matrix svd: [2, 3, 2]
          final aTall = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3, 2],
            DType.float64,
          );

          final resTall = svd(aTall);
          final uT = resTall.U;
          final sT = resTall.S;
          final vhT = resTall.Vh;

          expect(uT.shape, [2, 3, 3]);
          expect(sT.shape, [2, 2]);
          expect(vhT.shape, [2, 2, 2]);
          expect(sT.data[0], greaterThan(sT.data[1]));
          expect(sT.data[2], greaterThan(sT.data[3]));

          // 2. Wide matrix svd: [2, 2, 3] (tests transposed wide matrix path under batch)
          final aWide = NDArray.fromList(
            [1.0, 3.0, 5.0, 2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 2.0, 4.0, 6.0],
            [2, 2, 3],
            DType.float64,
          );

          final resWide = svd(aWide);
          final uW = resWide.U;
          final sW = resWide.S;
          final vhW = resWide.Vh;

          expect(uW.shape, [2, 2, 2]);
          expect(sW.shape, [2, 2]);
          expect(vhW.shape, [2, 3, 3]);
        }),
      );

      test(
        'Stacked operations with strided inputs (views)',
        () => NDArray.scope(() {
          final parent = NDArray.fromList(
            [1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0],
            [2, 2, 2],
            DType.float64,
          );

          // Take the transpose of the last two axes
          final axes = [0, 2, 1];
          final view = parent.transpose(
            axes,
          ); // Shape [2, 2, 2], non-contiguous
          expect(view.isContiguous, false);

          // Det of non-contiguous stacked view
          final d = det(view);
          expect(d.shape, [2]);
          // slice 0: [[1, 2], [3, 4]] -> det = -2
          // slice 1: [[5, 6], [7, 8]] -> det = -2
          expect(d.data[0], closeTo(-2.0, 1e-9));
          expect(d.data[1], closeTo(-2.0, 1e-9));
        }),
      );
    });

    group('Advanced Linear Algebra (kron, outer, cross, norm) tests', () {
      test(
        'Kronecker product (kron) tests',
        () => NDArray.scope(() {
          // 1D vector kron
          final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);
          final res1 = kron(a, b);
          expect(res1.shape, [4]);
          expect(res1.dtype, DType.float64);
          expect(res1.toList(), [3.0, 4.0, 6.0, 8.0]);

          // 2D matrix kron
          final a2 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b2 = NDArray.fromList(
            [0.0, 5.0, 6.0, 7.0],
            [2, 2],
            DType.float64,
          );
          final res2 = kron(a2, b2);
          expect(res2.shape, [4, 4]);
          expect(res2.toList(), [
            0.0,
            5.0,
            0.0,
            10.0,
            6.0,
            7.0,
            12.0,
            14.0,
            0.0,
            15.0,
            0.0,
            20.0,
            18.0,
            21.0,
            24.0,
            28.0,
          ]);

          // Strided view inputs
          final viewA = a2.transpose();
          final resView = kron(viewA, b2);
          expect(resView.shape, [4, 4]);

          // Complex types
          final ac = NDArray.fromList(
            [Complex(1.0, 2.0)],
            [1],
            DType.complex128,
          );
          final bc = NDArray.fromList(
            [Complex(3.0, 4.0)],
            [1],
            DType.complex128,
          );
          final resc = kron(ac, bc);
          expect(resc.dtype, DType.complex128);
          expect(resc.data[0].real, -5.0);
          expect(resc.data[0].imag, 10.0);
        }),
      );

      test(
        'Vector Outer Product (outer) tests',
        () => NDArray.scope(() {
          // Basic outer product
          final u = NDArray.fromList([1, 2], [2], DType.int32);
          final v = NDArray.fromList([3, 4, 5], [3], DType.int32);
          final res = outer(u, v);
          expect(res.shape, [2, 3]);
          expect(res.toList(), [3, 4, 5, 6, 8, 10]);

          // Complex outer product
          final uc = NDArray.fromList(
            [Complex(1.0, 0.0)],
            [1],
            DType.complex128,
          );
          final vc = NDArray.fromList(
            [Complex(0.0, 2.0)],
            [1],
            DType.complex128,
          );
          final resc = outer(uc, vc);
          expect(resc.dtype, DType.complex128);
          expect(resc.data[0].real, 0.0);
          expect(resc.data[0].imag, 2.0);
        }),
      );

      test(
        'Vector Cross Product (cross) tests',
        () => NDArray.scope(() {
          // 3D cross product
          final v1 = NDArray.fromList([1.0, 0.0, 0.0], [3], DType.float64);
          final v2 = NDArray.fromList([0.0, 1.0, 0.0], [3], DType.float64);
          final res3d = cross(v1, v2);
          expect(res3d.shape, [3]);
          expect(res3d.toList(), [0.0, 0.0, 1.0]);

          // 2D cross product (returns scalar/z-component)
          final u1 = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final u2 = NDArray.fromList([3.0, 4.0], [2], DType.float64);
          final res2d = cross(u1, u2);
          expect(res2d.shape, []);
          expect(res2d.data[0], -2.0);

          // Stacked/multidimensional cross product
          final sa = NDArray.fromList(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [2, 3],
            DType.float64,
          );
          final sb = NDArray.fromList(
            [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [2, 3],
            DType.float64,
          );
          final resStack = cross(sa, sb);
          expect(resStack.shape, [2, 3]);
          expect(resStack.toList(), [0.0, 0.0, 1.0, 0.0, 0.0, -1.0]);
        }),
      );

      test(
        'Vector/Matrix Norm (norm) tests',
        () => NDArray.scope(() {
          // 1D Vector Norms
          final x = NDArray.fromList(
            [1.0, -2.0, 3.0, -4.0],
            [4],
            DType.float64,
          );
          expect(norm(x, ord: 1).data[0], 10.0);
          expect(norm(x, ord: 2).data[0], closeTo(math.sqrt(30.0), 1e-9));
          expect(norm(x, ord: double.infinity).data[0], 4.0);
          expect(norm(x, ord: double.negativeInfinity).data[0], 1.0);

          // Matrix Norms
          final m = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          expect(norm(m, ord: 'fro').data[0], closeTo(math.sqrt(30.0), 1e-9));
          expect(norm(m, ord: 1).data[0], 6.0); // max of col sums ([4, 6])
          expect(
            norm(m, ord: double.infinity).data[0],
            7.0,
          ); // max of row sums ([3, 7])

          // Singular value matrix norms (ord=2)
          final sNorm = norm(m, ord: 2);
          expect(sNorm.data[0], greaterThan(0.0));

          // keepdims support
          final kd = norm(x, ord: 1, keepdims: true);
          expect(kd.shape, [1]);
          expect(kd.data[0], 10.0);

          // Axis reductions
          final m3 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final normAxis = norm(m3, ord: 1, axis: 0);
          expect(normAxis.shape, [2]);
          expect(normAxis.data[0], 4.0); // 1 + 3
          expect(normAxis.data[1], 6.0); // 2 + 4
        }),
      );

      test(
        'Kronecker product (kron) all dtypes coverage',
        () => NDArray.scope(() {
          final dtypes = [
            DType.float64,
            DType.float32,
            DType.int64,
            DType.int32,
            DType.uint8,
            DType.int16,
            DType.complex128,
            DType.complex64,
            DType.boolean,
          ];

          for (final dtype in dtypes) {
            final a = dtype == DType.boolean
                ? NDArray.fromList([true, false], [2], DType.boolean)
                : (dtype.isComplex
                      ? NDArray.fromList(
                          [Complex(1.0, 0.0), Complex(2.0, 0.0)],
                          [2],
                          dtype,
                        )
                      : (dtype.isFloating
                            ? NDArray.fromList([1.0, 2.0], [2], dtype)
                            : NDArray.fromList([1, 2], [2], dtype)));

            final b = dtype == DType.boolean
                ? NDArray.fromList([true, true], [2], DType.boolean)
                : (dtype.isComplex
                      ? NDArray.fromList(
                          [Complex(3.0, 0.0), Complex(4.0, 0.0)],
                          [2],
                          dtype,
                        )
                      : (dtype.isFloating
                            ? NDArray.fromList([3.0, 4.0], [2], dtype)
                            : NDArray.fromList([3, 4], [2], dtype)));

            final res = kron(a, b);
            expect(res.shape, [4]);
            final expectedDType = dtype == DType.boolean ? DType.uint8 : dtype;
            expect(res.dtype, expectedDType);

            if (dtype == DType.boolean) {
              expect(res.toList(), [1, 1, 0, 0]);
            } else if (dtype.isComplex) {
              expect(res.data[0].real, 3.0);
              expect(res.data[1].real, 4.0);
              expect(res.data[2].real, 6.0);
              expect(res.data[3].real, 8.0);
            } else {
              expect(
                res.toList(),
                [
                  3.0,
                  4.0,
                  6.0,
                  8.0,
                ].map((x) => dtype.isFloating ? x : x.toInt()).toList(),
              );
            }
          }
        }),
      );

      test(
        'Vector Outer Product (outer) all dtypes coverage',
        () => NDArray.scope(() {
          final dtypes = [
            DType.float64,
            DType.float32,
            DType.int64,
            DType.int32,
            DType.uint8,
            DType.int16,
            DType.complex128,
            DType.complex64,
            DType.boolean,
          ];

          for (final dtype in dtypes) {
            final a = dtype == DType.boolean
                ? NDArray.fromList([true, false], [2], DType.boolean)
                : (dtype.isComplex
                      ? NDArray.fromList(
                          [Complex(2.0, 0.0), Complex(3.0, 0.0)],
                          [2],
                          dtype,
                        )
                      : (dtype.isFloating
                            ? NDArray.fromList([2.0, 3.0], [2], dtype)
                            : NDArray.fromList([2, 3], [2], dtype)));

            final b = dtype == DType.boolean
                ? NDArray.fromList([true, true], [2], DType.boolean)
                : (dtype.isComplex
                      ? NDArray.fromList(
                          [Complex(4.0, 0.0), Complex(5.0, 0.0)],
                          [2],
                          dtype,
                        )
                      : (dtype.isFloating
                            ? NDArray.fromList([4.0, 5.0], [2], dtype)
                            : NDArray.fromList([4, 5], [2], dtype)));

            final res = outer(a, b);
            expect(res.shape, [2, 2]);
            final expectedDType = dtype == DType.boolean ? DType.uint8 : dtype;
            expect(res.dtype, expectedDType);

            if (dtype == DType.boolean) {
              expect(res.toList(), [1, 1, 0, 0]);
            } else if (dtype.isComplex) {
              expect(res.data[0].real, 8.0);
              expect(res.data[1].real, 10.0);
              expect(res.data[2].real, 12.0);
              expect(res.data[3].real, 15.0);
            } else {
              expect(
                res.toList(),
                [
                  8.0,
                  10.0,
                  12.0,
                  15.0,
                ].map((x) => dtype.isFloating ? x : x.toInt()).toList(),
              );
            }
          }
        }),
      );

      test(
        'Vector Cross Product (cross) all dtypes coverage',
        () => NDArray.scope(() {
          final dtypes = [
            DType.float64,
            DType.float32,
            DType.int64,
            DType.int32,
            DType.uint8,
            DType.int16,
            DType.complex128,
            DType.complex64,
            DType.boolean,
          ];

          for (final dtype in dtypes) {
            final a = dtype == DType.boolean
                ? NDArray.fromList([true, false, false], [3], DType.boolean)
                : (dtype.isComplex
                      ? NDArray.fromList(
                          [
                            Complex(1.0, 0.0),
                            Complex(0.0, 0.0),
                            Complex(0.0, 0.0),
                          ],
                          [3],
                          dtype,
                        )
                      : (dtype.isFloating
                            ? NDArray.fromList([1.0, 0.0, 0.0], [3], dtype)
                            : NDArray.fromList([1, 0, 0], [3], dtype)));

            final b = dtype == DType.boolean
                ? NDArray.fromList([false, true, false], [3], DType.boolean)
                : (dtype.isComplex
                      ? NDArray.fromList(
                          [
                            Complex(0.0, 0.0),
                            Complex(1.0, 0.0),
                            Complex(0.0, 0.0),
                          ],
                          [3],
                          dtype,
                        )
                      : (dtype.isFloating
                            ? NDArray.fromList([0.0, 1.0, 0.0], [3], dtype)
                            : NDArray.fromList([0, 1, 0], [3], dtype)));

            final res = cross(a, b);
            expect(res.shape, [3]);
            final expectedDType = dtype == DType.boolean ? DType.uint8 : dtype;
            expect(res.dtype, expectedDType);

            if (dtype == DType.boolean) {
              expect(res.toList(), [0, 0, 1]);
            } else if (dtype.isComplex) {
              expect(res.data[0].real, 0.0);
              expect(res.data[1].real, 0.0);
              expect(res.data[2].real, 1.0);
            } else {
              expect(
                res.toList(),
                [
                  0.0,
                  0.0,
                  1.0,
                ].map((x) => dtype.isFloating ? x : x.toInt()).toList(),
              );
            }
          }
        }),
      );

      test(
        'Vector Norm (norm) all dtypes coverage',
        () => NDArray.scope(() {
          final dtypes = [
            DType.float64,
            DType.float32,
            DType.int64,
            DType.int32,
            DType.uint8,
            DType.int16,
            DType.complex128,
            DType.complex64,
            DType.boolean,
          ];

          for (final dtype in dtypes) {
            final x = dtype == DType.boolean
                ? NDArray.fromList([true, false, true], [3], DType.boolean)
                : (dtype.isComplex
                      ? NDArray.fromList(
                          [
                            Complex(1.0, 0.0),
                            Complex(-2.0, 0.0),
                            Complex(3.0, 0.0),
                          ],
                          [3],
                          dtype,
                        )
                      : (dtype.isFloating
                            ? NDArray.fromList([1.0, -2.0, 3.0], [3], dtype)
                            : NDArray.fromList([1, -2, 3], [3], dtype)));

            final res = norm(x, ord: 1);
            expect(res.shape, []);
            final expectedDType =
                (dtype == DType.float32 || dtype == DType.complex64)
                ? DType.float32
                : DType.float64;
            expect(res.dtype, expectedDType);

            if (dtype == DType.boolean) {
              expect(res.data[0], 2.0);
            } else if (dtype == DType.uint8) {
              expect(res.data[0], 258.0);
            } else {
              expect(res.data[0], 6.0);
            }
          }
        }),
      );
    });
  });
}
