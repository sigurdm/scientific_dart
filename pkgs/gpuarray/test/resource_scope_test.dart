import 'package:test/test.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Resource Management & Scoping', () {
    test('Explicit dispose releases GPU buffer memory', () {
      final arr = GpuArray.zeros([100, 100], DType.float64);
      expect(arr.isDisposed, isFalse);
      expect(arr.buffer.isDisposed, isFalse);

      arr.dispose();
      expect(arr.isDisposed, isTrue);
      expect(arr.buffer.isDisposed, isTrue);
    });

    test(
      'ResourceScope.scope automatically cleans up intermediate GPU allocations',
      () {
        GpuArray? outerRef;

        ResourceScope.scope(() {
          final a = GpuArray.ones([50, 50], DType.float64);
          final b = GpuArray.ones([50, 50], DType.float64);
          final c = a + b;
          outerRef = c;
          expect(outerRef!.isDisposed, isFalse);
        });

        // Scope has exited, everything should be disposed
        expect(outerRef!.isDisposed, isTrue);
      },
    );

    test(
      'ResourceScope.returning promotes returned GPU array to outer scope',
      () {
        GpuArray? intermediateRef;

        final result = ResourceScope.returning(() {
          final a = GpuArray.ones([10, 10], DType.float64);
          final b = GpuArray.ones([10, 10], DType.float64);
          intermediateRef = a;
          final res = a.matmul(b);
          return res;
        });

        // Intermediate array should be disposed
        expect(intermediateRef!.isDisposed, isTrue);

        // Returned array should remain active
        expect(result.isDisposed, isFalse);
        expect(result.shape, equals([10, 10]));
        result.dispose();
        expect(result.isDisposed, isTrue);
      },
    );
  });
}
