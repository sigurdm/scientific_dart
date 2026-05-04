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

          final aInv = inv(a);
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

        final aInv = inv(a);
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

        final res = cholesky(a);
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
        final res = cholesky(a);
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

        final res = qr(a);
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
    });

    group('SVD Decomposition tests', () {
      test('SVD full matrices factorizations checks', () {
        final a = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
          [3, 2],
          DType.float64,
        );

        final res = svd(a);
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
    });
  });
}
