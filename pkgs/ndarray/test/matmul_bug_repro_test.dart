import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';

void main() {
  test('matmul on integer matrices', () {
    NDArray.scope(() {
      final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
      final b = NDArray.fromList([5, 6, 7, 8], [2, 2], DType.int32);
      final result = matmul(a, b);
      expect(result.dtype, DType.int32);
      expect(result.shape, [2, 2]);
      expect(result.toList(), [19, 22, 43, 50]);
    });
  });
}
