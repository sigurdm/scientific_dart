import 'dart:math' as math;
import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';

void main() {
  test('sinh on non-contiguous int32 array', () {
    NDArray.scope(() {
      final a = NDArray.fromList([0, 1, 2, 3], [4], DType.int32);
      final sliced = a.slice([const Slice(start: 0, stop: 4, step: 2)]);
      expect(sliced.isContiguous, isFalse);
      final result = sinh(sliced);
      expect(result.shape, [2]);
      expect(result.toList()[0], 0.0);
      expect(
        result.toList()[1],
        closeTo((math.exp(2.0) - math.exp(-2.0)) / 2.0, 1e-9),
      );
    });
  });
}
