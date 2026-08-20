import 'dart:math' as math;
import 'package:test/test.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/linalg.dart' as linalg;
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Linear Algebra Solvers & Metrics (gpuarray.linalg)', () {
    test('Linear system solver (solve) and matrix inverse (inv)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [3.0, 1.0, 1.0, 2.0],
          [2, 2],
          DType.float64,
        );

        final b = GpuArray.fromList([9.0, 8.0], [2], DType.float64);

        // Solve A x = b -> x = [2, 3]
        final x = linalg.solve(a, b);
        expect(x.shape, equals([2]));
        final xList = x.toList().cast<double>();
        expect(xList[0], closeTo(2.0, 1e-4));
        expect(xList[1], closeTo(3.0, 1e-4));

        // Matrix inverse A * A^-1 = I
        final aInv = linalg.inv(a);
        final identity = a.matmul(aInv);
        final idList = identity.toList().cast<double>();
        expect(idList[0], closeTo(1.0, 1e-4));
        expect(idList[1], closeTo(0.0, 1e-4));
        expect(idList[2], closeTo(0.0, 1e-4));
        expect(idList[3], closeTo(1.0, 1e-4));
      });
    });

    test('Pseudo-inverse (pinv)', () {
      ResourceScope.scope(() {
        // Non-square matrix (3x2)
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );

        final aPinvarr = linalg.pinv(a);
        expect(aPinvarr.shape, equals([2, 3]));

        // Penrose condition 1: A * A^+ * A = A
        final recon = a.matmul(aPinvarr).matmul(a);
        final reconList = recon.toList().cast<double>();
        final origList = a.toList().cast<double>();
        for (var i = 0; i < 6; i++) {
          expect(reconList[i], closeTo(origList[i], 1e-4));
        }
      });
    });

    test('Determinant and Slogdet (det, slogdet)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );

        // det(A) = 1*4 - 2*3 = -2
        final d = linalg.det(a) as double;
        expect(d, closeTo(-2.0, 1e-4));

        final slog = linalg.slogdet(a);
        expect(slog.sign.item(), closeTo(-1.0, 1e-4));
        expect(slog.logabsdet.item(), closeTo(math.log(2.0), 1e-4));
      });
    });

    test('Matrix power and matrix rank (matrix_power, matrix_rank)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );

        // A^2 = [[7, 10], [15, 22]]
        final a2 = linalg.matrix_power(a, 2);
        final a2List = a2.toList().cast<double>();
        expect(a2List[0], closeTo(7.0, 1e-4));
        expect(a2List[1], closeTo(10.0, 1e-4));
        expect(a2List[2], closeTo(15.0, 1e-4));
        expect(a2List[3], closeTo(22.0, 1e-4));

        // A^0 = Identity
        final a0 = linalg.matrix_power(a, 0);
        final a0List = a0.toList().cast<double>();
        expect(a0List, equals([1.0, 0.0, 0.0, 1.0]));

        // Matrix rank
        final rank = linalg.matrix_rank(a) as int;
        expect(rank, equals(2));

        // Rank deficient matrix
        final rankDef = GpuArray.fromList(
          [1.0, 2.0, 2.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final rank1 = linalg.matrix_rank(rankDef) as int;
        expect(rank1, equals(1));
      });
    });

    test('Matrix norm and condition number (norm, cond)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );

        // Frobenius norm: sqrt(1 + 4 + 9 + 16) = sqrt(30) = 5.4772
        final n = linalg.norm(a) as double;
        expect(n, closeTo(math.sqrt(30.0), 1e-4));

        // Condition number
        final c = linalg.cond(a) as double;
        expect(c, greaterThan(1.0));
      });
    });
  });
}
