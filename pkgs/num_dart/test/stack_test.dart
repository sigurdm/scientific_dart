import 'package:num_dart/num_dart.dart';
import 'package:test/test.dart';
import 'dart:typed_data';

void main() {
  group('NDArray New-Axis Stacking (stack) Tests', () {
    test('stack() basic 1D arrays to 2D along axis 0', () {
      final a = NDArray.fromList([1, 2], [2], DType.int32);
      final b = NDArray.fromList([3, 4], [2], DType.int32);

      final s0 = stack([a, b], axis: 0);
      expect(s0.shape, [2, 2]);
      expect(s0.dtype, DType.int32);
      expect(s0.toList(), [1, 2, 3, 4]);
    });

    test('stack() basic 1D arrays to 2D along axis 1', () {
      final a = NDArray.fromList([1, 2], [2], DType.int32);
      final b = NDArray.fromList([3, 4], [2], DType.int32);

      final s1 = stack([a, b], axis: 1);
      expect(s1.shape, [2, 2]);
      expect(s1.toList(), [1, 3, 2, 4]);
    });

    test('stack() multidimensional 2D matrices checks', () {
      // two matrices of shape [2, 3]
      final a = NDArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], DType.int32);
      final b = NDArray.fromList([10, 20, 30, 40, 50, 60], [2, 3], DType.int32);

      // Stacks into shape [2, 2, 3] along axis 0
      final s0 = stack([a, b], axis: 0);
      expect(s0.shape, [2, 2, 3]);
      expect(s0.toList(), [1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60]);

      // Stacks into shape [2, 2, 3] along axis 1
      final s1 = stack([a, b], axis: 1);
      expect(s1.shape, [2, 2, 3]);
      expect(s1.toList(), [1, 2, 3, 10, 20, 30, 4, 5, 6, 40, 50, 60]);
    });

    test('stack() resolved negative target axis', () {
      final a = NDArray.fromList([1, 2], [2], DType.int32);
      final b = NDArray.fromList([3, 4], [2], DType.int32);

      final sNeg = stack([a, b], axis: -1); // targets axis 1
      expect(sNeg.shape, [2, 2]);
      expect(sNeg.toList(), [1, 3, 2, 4]);
    });

    test('stack() validation errors throws exceptions', () {
      // Empty list throws ArgumentError
      expect(() => stack(<NDArray<Object>>[]), throwsArgumentError);

      final a = NDArray.fromList([1, 2], [2], DType.int32);
      final wrongShape = NDArray.fromList([1, 2, 3], [3], DType.int32);
      final wrongDType = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [2], DType.float64);

      // Mismatch shape/DType throws ArgumentError
      expect(() => stack([a, wrongShape]), throwsArgumentError);
      expect(() => stack([a, wrongDType]), throwsArgumentError);

      // Out of bounds axis throws RangeError
      expect(() => stack([a, a], axis: 5), throwsRangeError);
      expect(() => stack([a, a], axis: -5), throwsRangeError);
    });

    test('disposed arrays throw StateError', () {
      final a = NDArray.fromList([1, 2], [2], DType.int32);
      a.dispose();
      expect(() => stack([a, a]), throwsStateError);
    });
  });
}
