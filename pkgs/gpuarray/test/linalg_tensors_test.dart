import 'package:test/test.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:gpuarray/linalg.dart' as linalg;
import 'package:resource_scope/resource_scope.dart';

void main() {
  group(
    'GpuArray Tensor Contractions & Einstein Summation (gpuarray.linalg)',
    () {
      test('Einstein summation (einsum)', () {
        ResourceScope.scope(() {
          final a = GpuArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );

          final b = GpuArray.fromList(
            [5.0, 6.0, 7.0, 8.0],
            [2, 2],
            DType.float64,
          );

          // 1. Matrix multiplication 'ij,jk->ik'
          final c = linalg.einsum('ij,jk->ik', [a, b]);
          expect(c.shape, equals([2, 2]));
          final cList = c.toList().cast<double>();
          expect(cList[0], closeTo(19.0, 1e-4));
          expect(cList[1], closeTo(22.0, 1e-4));
          expect(cList[2], closeTo(43.0, 1e-4));
          expect(cList[3], closeTo(50.0, 1e-4));

          // 2. Matrix transpose 'ij->ji'
          final aT = linalg.einsum('ij->ji', [a]);
          expect(aT.shape, equals([2, 2]));
          expect(aT.toList(), equals([1.0, 3.0, 2.0, 4.0]));

          // 3. Matrix trace 'ii->'
          final tr = linalg.einsum('ii->', [a]);
          expect(tr.shape, equals([]));
          expect(tr.item(), closeTo(5.0, 1e-4));

          // 4. Outer product 'i,j->ij'
          final v1 = GpuArray.fromList([1.0, 2.0], [2], DType.float64);
          final v2 = GpuArray.fromList([3.0, 4.0], [2], DType.float64);
          final outer = linalg.einsum('i,j->ij', [v1, v2]);
          expect(outer.shape, equals([2, 2]));
          expect(outer.toList(), equals([3.0, 4.0, 6.0, 8.0]));
        });
      });

      test(
        'Tensor contractions (tensordot, kron, inner, outer, cross, multi_dot)',
        () {
          ResourceScope.scope(() {
            // tensordot
            final a = GpuArray.fromList(
              [1.0, 2.0, 3.0, 4.0],
              [2, 2],
              DType.float64,
            );
            final b = GpuArray.fromList(
              [5.0, 6.0, 7.0, 8.0],
              [2, 2],
              DType.float64,
            );
            final td = linalg.tensordot(a, b, axes: 1);
            expect(td.shape, equals([2, 2]));
            expect(td.toList(), equals([19.0, 22.0, 43.0, 50.0]));

            // kron (Kronecker product)
            final k = linalg.kron(a, b);
            expect(k.shape, equals([4, 4]));
            expect(
              k.toList(),
              equals([
                5.0,
                6.0,
                10.0,
                12.0,
                7.0,
                8.0,
                14.0,
                16.0,
                15.0,
                18.0,
                20.0,
                24.0,
                21.0,
                24.0,
                28.0,
                32.0,
              ]),
            );

            // inner and outer
            final v1 = GpuArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
            final v2 = GpuArray.fromList([4.0, 5.0, 6.0], [3], DType.float64);

            final inn = linalg.inner(v1, v2);
            expect(inn.shape, equals([]));
            expect(inn.item(), equals(32.0));

            final out = linalg.outer(v1, v2);
            expect(out.shape, equals([3, 3]));
            expect(
              out.toList(),
              equals([4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0]),
            );

            // cross product
            final u = GpuArray.fromList([1.0, 0.0, 0.0], [3], DType.float64);
            final v = GpuArray.fromList([0.0, 1.0, 0.0], [3], DType.float64);
            final uxv = linalg.cross(u, v);
            expect(uxv.toList(), equals([0.0, 0.0, 1.0]));

            // multi_dot
            final md = linalg.multi_dot([a, b, a]);
            expect(md.shape, equals([2, 2]));
          });
        },
      );
    },
  );
}
