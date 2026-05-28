import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';

void main() {
  test('sort on empty 1D array', () {
    NDArray.scope(() {
      final a = NDArray.create([0], DType.float64);
      final sorted = sort(a);
      expect(sorted.shape, [0]);
      expect(sorted.toList(), []);
      final indices = argsort(a);
      expect(indices.shape, [0]);
      expect(indices.toList(), []);
    });
  });
}
