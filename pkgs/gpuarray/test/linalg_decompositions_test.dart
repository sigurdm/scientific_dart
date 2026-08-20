import 'package:test/test.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/linalg.dart' as linalg;
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Linear Algebra Decompositions (gpuarray.linalg)', () {
    test('QR Decomposition (qr)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0],
          [3, 3],
          DType.float64,
        );

        final qrRes = linalg.qr(a);
        final q = qrRes.q;
        final r = qrRes.r;

        expect(q.shape, equals([3, 3]));
        expect(r.shape, equals([3, 3]));

        // Q * R should equal A
        final recon = q.matmul(r);
        final reconList = recon.toList().cast<double>();
        final origList = a.toList().cast<double>();

        for (var i = 0; i < 9; i++) {
          expect(reconList[i], closeTo(origList[i], 1e-4));
        }

        // Q * Q^T should equal Identity
        final qqt = q.matmul(q.transpose([1, 0]));
        final qqtList = qqt.toList().cast<double>();
        for (var i = 0; i < 3; i++) {
          for (var j = 0; j < 3; j++) {
            final expected = (i == j) ? 1.0 : 0.0;
            expect(qqtList[i * 3 + j], closeTo(expected, 1e-4));
          }
        }
      });
    });

    test('Cholesky Decomposition (cholesky)', () {
      ResourceScope.scope(() {
        // Symmetric positive definite matrix
        final a = GpuArray.fromList(
          [4.0, 12.0, -16.0, 12.0, 37.0, -43.0, -16.0, -43.0, 98.0],
          [3, 3],
          DType.float64,
        );

        final l = linalg.cholesky(a);
        expect(l.shape, equals([3, 3]));

        // L * L^T should equal A
        final recon = l.matmul(l.transpose([1, 0]));
        final reconList = recon.toList().cast<double>();
        final origList = a.toList().cast<double>();

        for (var i = 0; i < 9; i++) {
          expect(reconList[i], closeTo(origList[i], 1e-4));
        }
      });
    });

    test('LU Decomposition (lu, lu_factor, lu_solve)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [2.0, 1.0, 1.0, 4.0, 3.0, 3.0, 8.0, 7.0, 9.0],
          [3, 3],
          DType.float64,
        );

        final luRes = linalg.lu(a);
        final p = luRes.p;
        final l = luRes.l;
        final u = luRes.u;

        // P^T * L * U should equal A
        final pT = p.transpose([1, 0]);
        final recon = pT.matmul(l.matmul(u));
        final reconList = recon.toList().cast<double>();
        final origList = a.toList().cast<double>();

        for (var i = 0; i < 9; i++) {
          expect(reconList[i], closeTo(origList[i], 1e-4));
        }

        // lu_factor and lu_solve
        final b = GpuArray.fromList([4.0, 10.0, 24.0], [3], DType.float64);
        final fact = linalg.lu_factor(a);
        final x = linalg.lu_solve(fact.lu, fact.piv, b);

        expect(x.shape, equals([3]));
        // A * x should equal b
        final ax = a.matmul(x.reshape([3, 1])).flatten();
        final axList = ax.toList().cast<double>();
        final bList = b.toList().cast<double>();
        for (var i = 0; i < 3; i++) {
          expect(axList[i], closeTo(bList[i], 1e-4));
        }
      });
    });

    test('Singular Value Decomposition (svd)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );

        final svdRes = linalg.svd(a, fullMatrices: false);
        final u = svdRes.u;
        final s = svdRes.s;
        final vt = svdRes.vt;

        expect(u.shape, equals([3, 2]));
        expect(s.shape, equals([2]));
        expect(vt.shape, equals([2, 2]));

        // Singular values are descending positive
        final sList = s.toList().cast<double>();
        expect(sList[0], greaterThan(sList[1]));
        expect(sList[1], greaterThan(0.0));

        // U * S * Vt should equal A
        final sDiag = diag(s);

        final recon = u.matmul(sDiag).matmul(vt);
        final reconList = recon.toList().cast<double>();
        final origList = a.toList().cast<double>();

        for (var i = 0; i < 6; i++) {
          expect(reconList[i], closeTo(origList[i], 1e-4));
        }
      });
    });

    test('Eigendecomposition of Symmetric Matrix (eigh, eigvalsh)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [2.0, 1.0, 1.0, 2.0],
          [2, 2],
          DType.float64,
        );

        final eigRes = linalg.eigh(a);
        final w = eigRes.eigenvalues;
        final v = eigRes.eigenvectors;

        expect(w.shape, equals([2]));
        expect(v.shape, equals([2, 2]));

        // Eigenvalues for [[2, 1], [1, 2]] are 1 and 3
        final wList = w.toList().cast<double>();
        expect(wList[0], closeTo(1.0, 1e-4));
        expect(wList[1], closeTo(3.0, 1e-4));

        // V * W * V^T should equal A
        final wDiag = diag(w);

        final recon = v.matmul(wDiag).matmul(v.transpose([1, 0]));
        final reconList = recon.toList().cast<double>();
        final origList = a.toList().cast<double>();

        for (var i = 0; i < 4; i++) {
          expect(reconList[i], closeTo(origList[i], 1e-4));
        }

        final vals = linalg.eigvalsh(a);
        expect(vals.toList(), equals(w.toList()));
      });
    });
  });
}
