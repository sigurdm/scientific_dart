import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';

void main() {
  test('trapz and gradient with integer step spacing', () {
    NDArray.scope(() {
      final y = NDArray.fromList([1.0, 2.0, 4.0], [3], DType.float64);
      final result = trapz(y, spacing: const Spacing.step(1));
      expect(result.shape, []);
      expect(result.data[0], 4.5);
      final grad = gradient(y, spacing: const Spacing.step(2));
      expect(grad.shape, [3]);
      expect(grad.toList()[0], closeTo(0.5, 1e-9));
      expect(grad.toList()[1], closeTo(0.75, 1e-9));
      expect(grad.toList()[2], closeTo(1.0, 1e-9));
    });
  });
}
