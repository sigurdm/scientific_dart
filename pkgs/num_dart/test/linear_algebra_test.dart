import 'package:num_dart/num_dart.dart';
import 'package:test/test.dart';
import 'dart:typed_data';

void main() {
  group('NDArray Advanced LAPACK Linear Algebra Decompositions Tests', () {
    group('LAPACK Inversion (inv) tests', () {
      test(
        'Invert Float64 square matrix and check identity reconstruction',
        () {
          final a = NDArray.fromList(
            Float64List.fromList([4.0, 7.0, 2.0, 6.0]),
            [2, 2],
            DType.float64,
          );
          addTearDown(a.dispose);

          final aInv = inv(a);
          addTearDown(aInv.dispose);
          expect(aInv.shape, [2, 2]);
          expect(aInv.dtype, DType.float64);

          // Check exact values (approx 0.6, -0.7, -0.2, 0.4)
          expect(aInv.data[0], closeTo(0.6, 1e-6));
          expect(aInv.data[1], closeTo(-0.7, 1e-6));
          expect(aInv.data[2], closeTo(-0.2, 1e-6));
          expect(aInv.data[3], closeTo(0.4, 1e-6));
        },
      );

      test('Invert Float32 matrix', () {
        final a = NDArray.fromList(Float32List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float32);
        addTearDown(a.dispose);

        final aInv = inv(a);
        addTearDown(aInv.dispose);
        expect(aInv.dtype, DType.float32);
        // expected inverse for [[1,2],[3,4]] is [[-2, 1], [1.5, -0.5]]
        expect(aInv.data[0], closeTo(-2.0, 1e-5));
        expect(aInv.data[1], closeTo(1.0, 1e-5));
        expect(aInv.data[2], closeTo(1.5, 1e-5));
        expect(aInv.data[3], closeTo(-0.5, 1e-5));
      });

      test('Singular matrix throws ArgumentError', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 2.0, 4.0]), [
          2,
          2,
        ], DType.float64); // dependent rows, det = 0
        addTearDown(a.dispose);

        expect(() => inv(a), throwsArgumentError);
      });

      test('Singular Float32 matrix throws ArgumentError', () {
        final a = NDArray.fromList(Float32List.fromList([1.0, 2.0, 2.0, 4.0]), [
          2,
          2,
        ], DType.float32); // dependent rows, det = 0
        addTearDown(a.dispose);

        expect(() => inv(a), throwsArgumentError);
      });
    });

    group('Cholesky Decomposition tests', () {
      test('Cholesky lower triangular factor and L * L^T reconstruction', () {
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
        addTearDown(a.dispose);

        final res = cholesky(a);
        addTearDown(() => res.values.forEach((v) => v.dispose()));
        final l = res['L']!;

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
      });

      test('Cholesky Float32 matrix', () {
        final a = NDArray.fromList(Float32List.fromList([4.0, 2.0, 2.0, 2.0]), [
          2,
          2,
        ], DType.float32);
        addTearDown(a.dispose);
        final res = cholesky(a);
        addTearDown(() => res.values.forEach((v) => v.dispose()));
        final l = res['L']!;
        expect(l.dtype, DType.float32);
        // L = [[2, 0], [1, 1]]
        expect(l.data[0], closeTo(2.0, 1e-5));
        expect(l.data[1], 0.0);
        expect(l.data[2], closeTo(1.0, 1e-5));
        expect(l.data[3], closeTo(1.0, 1e-5));
      });
    });

    group('QR Decomposition tests', () {
      test('QR orthogonal Q and upper triangular R verification', () {
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
        addTearDown(a.dispose);

        final res = qr(a);
        addTearDown(() => res.values.forEach((v) => v.dispose()));
        final q = res['Q']!;
        final r = res['R']!;

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
      });

      test('QR Float32 matrix', () {
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
        addTearDown(a.dispose);
        final res = qr(a);
        addTearDown(() => res.values.forEach((v) => v.dispose()));
        expect(res['Q']!.dtype, DType.float32);
        expect(res['R']!.dtype, DType.float32);
        expect(res['R']!.data[3], 0.0);
      });

      test('QR non-contiguous view matrix', () {
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
        addTearDown(parent.dispose);
        final view = parent.transpose();
        addTearDown(view.dispose);
        expect(view.isContiguous, false);
        final res = qr(view);
        addTearDown(() => res.values.forEach((v) => v.dispose()));
        expect(res['Q']!.shape, [3, 3]);
        expect(res['R']!.shape, [3, 3]);
        expect(res['R']!.data[3], 0.0);
      });
    });

    group('SVD Decomposition tests', () {
      test('SVD full matrices factorizations checks', () {
        final a = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
          [3, 2],
          DType.float64,
        );
        addTearDown(a.dispose);

        final res = svd(a);
        addTearDown(() => res.values.forEach((v) => v.dispose()));
        final u = res['U']!;
        final s = res['S']!;
        final vh = res['Vh']!;

        expect(u.shape, [3, 3]); // full matrix
        expect(s.shape, [2]); // min(3, 2) vector
        expect(vh.shape, [2, 2]); // full matrix

        // Singular values in S must be sorted in descending order by LAPACK rules!
        expect(s.data[0], greaterThan(s.data[1]));
        expect(s.data[0], greaterThan(0.0));
      });

      test('SVD Float32 matrix', () {
        final a = NDArray.fromList(
          Float32List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
          [3, 2],
          DType.float32,
        );
        addTearDown(a.dispose);
        final res = svd(a);
        addTearDown(() => res.values.forEach((v) => v.dispose()));
        expect(res['U']!.dtype, DType.float32);
        expect(res['S']!.dtype, DType.float32);
        expect(res['Vh']!.dtype, DType.float32);
        expect(res['U']!.shape, [3, 3]);
      });

      test('SVD non-contiguous view matrix', () {
        final parent = NDArray.fromList(
          Float64List.fromList([1.0, 3.0, 5.0, 2.0, 4.0, 6.0]),
          [2, 3],
          DType.float64,
        );
        addTearDown(parent.dispose);
        final view = parent.transpose();
        addTearDown(view.dispose);
        expect(view.isContiguous, false);
        final res = svd(view);
        addTearDown(() => res.values.forEach((v) => v.dispose()));
        expect(res['U']!.shape, [3, 3]);
        expect(res['S']!.shape, [2]);
      });
    });

    group('Determinant (det) tests', () {
      test('Determinant of Float64 2D square matrix', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        addTearDown(a.dispose);
        expect(det(a), closeTo(-2.0, 1e-9));
      });

      test('Determinant of Float32 2D square matrix', () {
        final a = NDArray.fromList(Float32List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float32);
        addTearDown(a.dispose);
        expect(det(a), closeTo(-2.0, 1e-5));
      });

      test('det() throws ArgumentError on non-square matrix', () {
        final a = NDArray<double>.zeros([2, 3], DType.float64);
        addTearDown(a.dispose);
        expect(() => det(a), throwsArgumentError);
      });
    });

    group('Linear Solver (solve) tests', () {
      test('Solve Float64 system of equations', () {
        final a = NDArray.fromList(Float64List.fromList([3.0, 1.0, 1.0, 2.0]), [
          2,
          2,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = NDArray.fromList(Float64List.fromList([9.0, 8.0]), [
          2,
          1,
        ], DType.float64);
        addTearDown(b.dispose);
        final x = solve(a, b);
        addTearDown(x.dispose);
        expect(x.shape, [2, 1]);
        expect(x.data[0], closeTo(2.0, 1e-9));
        expect(x.data[1], closeTo(3.0, 1e-9));
      });

      test('Solve Float32 system of equations', () {
        final a = NDArray.fromList(Float32List.fromList([3.0, 1.0, 1.0, 2.0]), [
          2,
          2,
        ], DType.float32);
        addTearDown(a.dispose);
        final b = NDArray.fromList(Float32List.fromList([9.0, 8.0]), [
          2,
          1,
        ], DType.float32);
        addTearDown(b.dispose);
        final x = solve(a, b);
        addTearDown(x.dispose);
        expect(x.shape, [2, 1]);
        expect(x.data[0], closeTo(2.0, 1e-5));
        expect(x.data[1], closeTo(3.0, 1e-5));
      });

      test('Solve Complex128 system of equations', () {
        final a = NDArray<Complex>.create([2, 2], DType.complex128);
        addTearDown(a.dispose);
        a.data[0] = Complex(3.0, 0.0);
        a.data[1] = Complex(1.0, 0.0);
        a.data[2] = Complex(1.0, 0.0);
        a.data[3] = Complex(2.0, 0.0);

        final b = NDArray<Complex>.create([2, 1], DType.complex128);
        addTearDown(b.dispose);
        b.data[0] = Complex(9.0, 0.0);
        b.data[1] = Complex(8.0, 0.0);

        final x = solve(a, b);
        addTearDown(x.dispose);
        expect(x.dtype, DType.complex128);
        expect(x.data[0], Complex(2.0, 0.0));
        expect(x.data[1], Complex(3.0, 0.0));
      });

      test('Singular matrix solve throws ArgumentError', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 2.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = NDArray.fromList(Float64List.fromList([9.0, 8.0]), [
          2,
          1,
        ], DType.float64);
        addTearDown(b.dispose);
        expect(() => solve(a, b), throwsArgumentError);
      });

      test('Solve integer matrices system (converts to Float64)', () {
        final a = NDArray.fromList(Int32List.fromList([3, 1, 1, 2]), [
          2,
          2,
        ], DType.int32);
        addTearDown(a.dispose);
        final b = NDArray.fromList(Int32List.fromList([9, 8]), [
          2,
          1,
        ], DType.int32);
        addTearDown(b.dispose);
        final x = solve(a, b);
        addTearDown(x.dispose);
        expect(x.dtype, DType.float64);
        expect(x.data[0], closeTo(2.0, 1e-9));
        expect(x.data[1], closeTo(3.0, 1e-9));
      });

      test('Solve Complex64 system of equations', () {
        final a = NDArray<Complex>.create([2, 2], DType.complex64);
        addTearDown(a.dispose);
        a.data[0] = Complex(3.0, 0.0);
        a.data[1] = Complex(1.0, 0.0);
        a.data[2] = Complex(1.0, 0.0);
        a.data[3] = Complex(2.0, 0.0);

        final b = NDArray<Complex>.create([2, 1], DType.complex64);
        addTearDown(b.dispose);
        b.data[0] = Complex(9.0, 0.0);
        b.data[1] = Complex(8.0, 0.0);

        final x = solve(a, b);
        addTearDown(x.dispose);
        expect(x.dtype, DType.complex64);
        expect(x.data[0].real, closeTo(2.0, 1e-5));
        expect(x.data[1].real, closeTo(3.0, 1e-5));
      });
    });

    group('Eigenvalues (eig) tests', () {
      test('Eigenvalues/eigenvectors of Float64 2D square matrix', () {
        final a = NDArray.fromList(
          Float64List.fromList([0.0, 1.0, -2.0, -3.0]),
          [2, 2],
          DType.float64,
        );
        addTearDown(a.dispose);
        final res = eig(a);
        addTearDown(() => res.values.forEach((v) => v.dispose()));
        final w = res['eigenvalues']!;
        final vr = res['eigenvectors']!;

        expect(w.shape, [2]);
        expect(vr.shape, [2, 2]);

        expect(w.data[0].real, closeTo(-1.0, 1e-9));
        expect(w.data[0].imag, closeTo(0.0, 1e-9));
        expect(w.data[1].real, closeTo(-2.0, 1e-9));
        expect(w.data[1].imag, closeTo(0.0, 1e-9));
      });

      test('Eigenvalues of Float32 2D square matrix', () {
        final a = NDArray.fromList(
          Float32List.fromList([0.0, 1.0, -2.0, -3.0]),
          [2, 2],
          DType.float32,
        );
        addTearDown(a.dispose);
        final res = eig(a);
        addTearDown(() => res.values.forEach((v) => v.dispose()));
        final w = res['eigenvalues']!;
        expect(w.dtype, DType.complex64);
        expect(w.data[0].real, closeTo(-1.0, 1e-5));
        expect(w.data[1].real, closeTo(-2.0, 1e-5));
      });

      test('Eigenvalues/eigenvectors of Complex128 matrix', () {
        final a = NDArray<Complex>.create([2, 2], DType.complex128);
        addTearDown(a.dispose);
        a.data[0] = Complex(0.0, 0.0);
        a.data[1] = Complex(1.0, 0.0);
        a.data[2] = Complex(-2.0, 0.0);
        a.data[3] = Complex(-3.0, 0.0);

        final res = eig(a);
        addTearDown(() => res.values.forEach((v) => v.dispose()));
        expect(res['eigenvalues']!.dtype, DType.complex128);
        expect(res['eigenvalues']!.data[0].real, closeTo(-1.0, 1e-9));
        expect(res['eigenvalues']!.data[1].real, closeTo(-2.0, 1e-9));
      });

      test('Eigenvalues/eigenvectors of Complex64 matrix', () {
        final a = NDArray<Complex>.create([2, 2], DType.complex64);
        addTearDown(a.dispose);
        a.data[0] = Complex(0.0, 0.0);
        a.data[1] = Complex(1.0, 0.0);
        a.data[2] = Complex(-2.0, 0.0);
        a.data[3] = Complex(-3.0, 0.0);

        final res = eig(a);
        addTearDown(() => res.values.forEach((v) => v.dispose()));
        expect(res['eigenvalues']!.dtype, DType.complex64);
        expect(res['eigenvalues']!.data[0].real, closeTo(-1.0, 1e-5));
        expect(res['eigenvalues']!.data[1].real, closeTo(-2.0, 1e-5));
      });
    });

    group('Linear Algebra Validation & Error Exceptions tests', () {
      test('cholesky() throws ArgumentError on non-square matrix', () {
        final a = NDArray.zeros([2, 3], DType.float64);
        addTearDown(a.dispose);
        expect(() => cholesky(a), throwsArgumentError);
      });

      test(
        'cholesky() throws ArgumentError on non positive-definite matrix',
        () {
          final a = NDArray.fromList(
            Float64List.fromList([-1.0, 0.0, 0.0, -1.0]),
            [2, 2],
            DType.float64,
          );
          addTearDown(a.dispose);
          expect(() => cholesky(a), throwsArgumentError);
        },
      );

      test('qr() throws ArgumentError on non-2D tensor', () {
        final a = NDArray.zeros([3], DType.float64);
        addTearDown(a.dispose);
        expect(() => qr(a), throwsArgumentError);
      });

      test('svd() throws ArgumentError on non-2D tensor', () {
        final a = NDArray.zeros([3], DType.float64);
        addTearDown(a.dispose);
        expect(() => svd(a), throwsArgumentError);
      });

      test('eig() throws ArgumentError on non-square matrix', () {
        final a = NDArray.zeros([2, 3], DType.float64);
        addTearDown(a.dispose);
        expect(() => eig(a), throwsArgumentError);
      });

      test('eig() throws UnimplementedError on boolean matrix', () {
        final a = NDArray.zeros([2, 2], DType.boolean);
        addTearDown(a.dispose);
        expect(() => eig(a), throwsUnimplementedError);
      });

      test('solve() throws ArgumentError on non-square matrix a', () {
        final a = NDArray.zeros([2, 3], DType.float64);
        final b = NDArray.zeros([2], DType.float64);
        addTearDown(a.dispose);
        addTearDown(b.dispose);
        expect(() => solve(a, b), throwsArgumentError);
      });

      test('solve() throws ArgumentError on mismatched b first dimension', () {
        final a = NDArray.zeros([2, 2], DType.float64);
        final b = NDArray.zeros([3], DType.float64);
        addTearDown(a.dispose);
        addTearDown(b.dispose);
        expect(() => solve(a, b), throwsArgumentError);
      });

      test('solve() throws ArgumentError on empty/scalar b array', () {
        final a = NDArray.zeros([2, 2], DType.float64);
        final b = NDArray.zeros([], DType.float64);
        addTearDown(a.dispose);
        addTearDown(b.dispose);
        expect(() => solve(a, b), throwsArgumentError);
      });
    });
  });
}
