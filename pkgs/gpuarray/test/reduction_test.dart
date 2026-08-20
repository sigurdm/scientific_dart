import 'package:test/test.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Reductions', () {
    test('Full reductions (sum, mean, min, max, prod)', () {
      ResourceScope.scope(() {
        final a = GpuArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);

        expect(a.sum().scalar, equals(10.0));
        expect(a.mean().scalar, equals(2.5));
        expect(a.min().scalar, equals(1.0));
        expect(a.max().scalar, equals(4.0));
        expect(a.prod().scalar, equals(24.0));
      });
    });

    test('2D reductions along axis 0 and axis 1', () {
      ResourceScope.scope(() {
        final mat = GpuArray.fromList(
          [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
          ],
          [2, 3],
          DType.float64,
        );

        // Sum along columns (axis 0) -> [5.0, 7.0, 9.0]
        final sum0 = mat.sum(axis: 0);
        expect(sum0.shape, equals([3]));
        expect(sum0.toList(), equals([5.0, 7.0, 9.0]));

        // Sum along rows (axis 1) -> [6.0, 15.0]
        final sum1 = mat.sum(axis: 1);
        expect(sum1.shape, equals([2]));
        expect(sum1.toList(), equals([6.0, 15.0]));

        // Mean along axis 0
        final mean0 = mat.mean(axis: 0);
        expect(mean0.toList(), equals([2.5, 3.5, 4.5]));

        // Min and Max along axis 1
        expect(mat.min(axis: 1).toList(), equals([1.0, 4.0]));
        expect(mat.max(axis: 1).toList(), equals([3.0, 6.0]));
      });
    });

    test('Reductions with keepDims: true', () {
      ResourceScope.scope(() {
        final mat = GpuArray.fromList(
          [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
          ],
          [2, 3],
          DType.float64,
        );

        final sumKeep = mat.sum(axis: 1, keepDims: true);
        expect(sumKeep.shape, equals([2, 1]));
        expect(
          sumKeep.toNestedList(),
          equals([
            [6.0],
            [15.0],
          ]),
        );

        final fullSumKeep = mat.sum(keepDims: true);
        expect(fullSumKeep.shape, equals([1, 1]));
        expect(
          fullSumKeep.toNestedList(),
          equals([
            [21.0],
          ]),
        );
      });
    });
  });
}
