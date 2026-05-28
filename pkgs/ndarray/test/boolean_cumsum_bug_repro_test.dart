import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';

void main() {
  test('cumsum on boolean array', () {
    NDArray.scope(() {
      final a = NDArray.fromList([true, false, true, true], [4], DType.boolean);
      final result = cumsum(a);
      expect(result.dtype, DType.int32);
      expect(result.shape, [4]);
      expect(result.toList(), [1, 1, 2, 3]);
      final a2D = NDArray.fromList(
        [true, false, true, true],
        [2, 2],
        DType.boolean,
      );
      final result2D = cumsum(a2D, axis: 0);
      expect(result2D.dtype, DType.int32);
      expect(result2D.shape, [2, 2]);
      expect(result2D.toList(), [1, 0, 2, 1]);
    });
  });
}
