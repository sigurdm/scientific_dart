import 'package:num_dart/num_dart.dart';
import 'package:test/test.dart';

void main() {
  group('NDArray Logical Reductions (all, any) Tests', () {
    test('all() basic global reduction checks', () {
      final a = NDArray.fromList([true, true, true], [3], DType.boolean);
      final b = NDArray.fromList([true, false, true], [3], DType.boolean);
      final c = NDArray.fromList([1, 2, 3], [3], DType.int32);
      final d = NDArray.fromList([1, 0, 3], [3], DType.int32);

      expect(all(a), true);
      expect(all(b), false);
      expect(all(c), true);
      expect(all(d), false);
    });

    test('any() basic global reduction checks', () {
      final a = NDArray.fromList([false, false, false], [3], DType.boolean);
      final b = NDArray.fromList([false, true, false], [3], DType.boolean);
      final c = NDArray.fromList([0, 0, 0], [3], DType.int32);
      final d = NDArray.fromList([0, 5, 0], [3], DType.int32);

      expect(any(a), false);
      expect(any(b), true);
      expect(any(c), false);
      expect(any(d), true);
    });

    test('all() axis reductions on 2D matrices', () {
      final mat = NDArray.fromList(
        [true, true, false, true, false, false],
        [2, 3],
        DType.boolean,
      );

      // all along rows (axis: 0) -> shape [3]
      final res0 = all(mat, axis: 0);
      expect(res0.shape, [3]);
      expect(res0.toList(), [true, false, false]);

      // all along columns (axis: 1) -> shape [2]
      final res1 = all(mat, axis: 1);
      expect(res1.shape, [2]);
      expect(res1.toList(), [false, false]);

      // negative axis support
      final resNeg = all(mat, axis: -1);
      expect(resNeg.shape, [2]);
      expect(resNeg.toList(), [false, false]);
    });

    test('any() axis reductions on 2D matrices', () {
      final mat = NDArray.fromList(
        [true, false, false, false, false, false],
        [2, 3],
        DType.boolean,
      );

      // any along rows (axis: 0) -> shape [3]
      final res0 = any(mat, axis: 0);
      expect(res0.shape, [3]);
      expect(res0.toList(), [true, false, false]);

      // any along columns (axis: 1) -> shape [2]
      final res1 = any(mat, axis: 1);
      expect(res1.shape, [2]);
      expect(res1.toList(), [true, false]);
    });

    test('Disposed array checks throw StateError', () {
      final a = NDArray.fromList([true, false], [2], DType.boolean);
      a.dispose();

      expect(() => all(a), throwsStateError);
      expect(() => any(a), throwsStateError);
    });

    test('Axis out of bounds throws ArgumentError', () {
      final a = NDArray.fromList([true, false], [2], DType.boolean);
      expect(() => all(a, axis: 5), throwsArgumentError);
      expect(() => any(a, axis: -5), throwsArgumentError);
    });
  });
}
