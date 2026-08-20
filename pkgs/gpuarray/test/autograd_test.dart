import 'package:test/test.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';

void main() {
  group('GpuArray Automatic Differentiation (Autograd)', () {
    test('Scalar arithmetic and polynomial gradient', () {
      ResourceScope.scope(() {
        // f(x) = 3x^2 + 2x + 1, f'(x) = 6x + 2
        // at x = 4, f'(4) = 26
        final x = GpuArray.fromList([4.0], [1], DType.float64, requiresGrad: true);
        expect(x.isLeaf, isTrue);

        final y = (x * x * 3.0) + (x * 2.0) + 1.0;
        expect(y.requiresGrad, isTrue);
        expect(y.gradFn, isNotNull);

        y.backward();

        expect(x.grad, isNotNull);
        expect(x.grad!.toList().cast<double>().first, closeTo(26.0, 1e-4));
      });
    });

    test('Vector and matrix multi-variable gradient (z = x @ y)', () {
      ResourceScope.scope(() {
        // X: [2, 2], Y: [2, 2]
        // Z = X @ Y, L = sum(Z)
        // dL/dX = ones(2, 2) @ Y^T
        // dL/dY = X^T @ ones(2, 2)
        final x = GpuArray.fromList([
          1.0, 2.0,
          3.0, 4.0,
        ], [2, 2], DType.float64, requiresGrad: true);

        final y = GpuArray.fromList([
          5.0, 6.0,
          7.0, 8.0,
        ], [2, 2], DType.float64, requiresGrad: true);

        final z = x.matmul(y);
        final loss = z.sum();

        loss.backward();

        expect(x.grad, isNotNull);
        expect(y.grad, isNotNull);

        // dL/dX = [[1, 1], [1, 1]] @ [[5, 7], [6, 8]] = [[11, 15], [11, 15]]
        final xGrad = x.grad!.toList().cast<double>();
        expect(xGrad, equals([11.0, 15.0, 11.0, 15.0]));

        // dL/dY = [[1, 3], [2, 4]] @ [[1, 1], [1, 1]] = [[4, 4], [6, 6]]
        final yGrad = y.grad!.toList().cast<double>();
        expect(yGrad, equals([4.0, 4.0, 6.0, 6.0]));
      });
    });

    test('Broadcasting gradient accumulation (Linear layer: Y = XW + B)', () {
      ResourceScope.scope(() {
        // X: [3, 2], W: [2, 2], B: [2]
        final x = GpuArray.fromList([
          1.0, 2.0,
          3.0, 4.0,
          5.0, 6.0,
        ], [3, 2], DType.float64);

        final w = GpuArray.fromList([
          0.5, -0.5,
          1.0, 2.0,
        ], [2, 2], DType.float64, requiresGrad: true);

        final b = GpuArray.fromList([
          0.1, 0.2,
        ], [2], DType.float64, requiresGrad: true);

        final out = x.matmul(w) + b;
        final loss = out.mean();

        loss.backward();

        expect(w.grad, isNotNull);
        expect(b.grad, isNotNull);
        expect(b.grad!.shape, equals([2]));

        // Check gradient of bias (each element averaged over 3 batch samples * 2 output features = 1/6 * 3 = 0.5)
        final bGrad = b.grad!.toList().cast<double>();
        expect(bGrad[0], closeTo(0.5, 1e-4));
        expect(bGrad[1], closeTo(0.5, 1e-4));
      });
    });

    test('no_grad context disables gradient graph building', () {
      ResourceScope.scope(() {
        final x = GpuArray.fromList([2.0], [1], DType.float64, requiresGrad: true);

        final y = no_grad(() {
          return x * 3.0 + 4.0;
        });

        expect(y.requiresGrad, isFalse);
        expect(y.gradFn, isNull);
      });
    });

    test('zeroGrad and detach', () {
      ResourceScope.scope(() {
        final x = GpuArray.fromList([3.0], [1], DType.float64, requiresGrad: true);
        final y = x * 2.0;
        y.backward();

        expect(x.grad, isNotNull);
        x.zeroGrad();
        expect(x.grad, isNull);

        final detached = x.detach();
        expect(detached.requiresGrad, isFalse);
        expect(detached.toList(), equals([3.0]));
      });
    });
  });
}
