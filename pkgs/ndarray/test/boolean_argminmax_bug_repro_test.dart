import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';

void main() {
  test('argmax/argmin on boolean array', () {
    NDArray.scope(() {
      final a = NDArray.fromList([false, true, false], [3], DType.boolean);
      final maxIdx = argmax(a);
      expect(maxIdx.scalar, 1);
      final minIdx = argmin(a);
      expect(minIdx.scalar, 0);
      final a2D = NDArray.fromList(
        [false, true, false, false],
        [2, 2],
        DType.boolean,
      );
      final max2D = argmax(a2D, axis: 1);
      expect(max2D.toList(), [1, 0]);
    });
  });
}
