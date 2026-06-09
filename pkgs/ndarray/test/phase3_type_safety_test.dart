import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';
import 'dart:math' as math;

void main() {
  group('Phase 3 Type Safety Tests', () {
    test('floor_divide with uint8/int16', () {
      final a = NDArray.fromList([4, 5, 6], [3], DType.uint8);
      final b = NDArray.fromList([2, 2, 2], [3], DType.int16);

      // This should not crash
      final c = floor_divide(a, b);
      expect(c.data, [2, 2, 3]);
      expect(
        c.dtype,
        DType.int32,
      ); // resolved dtype of uint8 and int16 is int32
    });

    test('floor_divide with uint8/float64', () {
      final a = NDArray.fromList([5, 6, 7], [3], DType.uint8);
      final b = NDArray.fromList([2.0, 2.0, 2.0], [3], DType.float64);

      // This should not crash
      final c = floor_divide(a, b);
      expect(c.data, [2.0, 3.0, 3.0]);
      expect(c.dtype, DType.float64);
    });

    test('remainder with uint8/int16', () {
      final a = NDArray.fromList([5, 6, 7], [3], DType.uint8);
      final b = NDArray.fromList([3, 3, 3], [3], DType.int16);

      // This should not crash
      final c = remainder(a, b);
      expect(c.data, [2, 0, 1]);
      expect(c.dtype, DType.int32);
    });

    test('remainder with uint8/float64', () {
      final a = NDArray.fromList([5, 6, 7], [3], DType.uint8);
      final b = NDArray.fromList([3.0, 3.0, 3.0], [3], DType.float64);

      // This should not crash
      final c = remainder(a, b);
      expect(c.data, [2.0, 0.0, 1.0]);
      expect(c.dtype, DType.float64);
    });

    test('sin with uint8/int16', () {
      final a = NDArray.fromList([0, 30, 90], [3], DType.uint8);
      // This should not crash
      final c = sin(a);
      expect(c.dtype, DType.float64); // default float type for sin on int
    });

    test('abs with uint8/int16', () {
      final a = NDArray.fromList([-1, -2, 3], [3], DType.int16);
      final c = abs(a);
      expect(c.data, [1, 2, 3]);
      expect(c.dtype, DType.int16);
    });

    test('negative with uint8/int16', () {
      final a = NDArray.fromList([1, 2, 3], [3], DType.int16);
      final b = negative(a);
      expect(b.data, [-1, -2, -3]);
      expect(b.dtype, DType.int16);

      final c = NDArray.fromList([1, 2, 3], [3], DType.uint8);
      final d = negative(c);
      expect(d.data, [255, 254, 253]); // wrap around for uint8
      expect(d.dtype, DType.uint8);
    });

    test('det with float32 preserves type', () {
      final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float32);
      final d = det(a);
      expect(d.dtype, DType.float32);
      expect(d.data[0], closeTo(-2.0, 1e-5));
    });

    test('svd and qr throw ArgumentError for integer inputs', () {
      final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
      expect(() => svd(a), throwsArgumentError);
      expect(() => qr(a), throwsArgumentError);
    });

    test('complex SVD (complex128)', () {
      final a = NDArray.fromList(
        [
          Complex(2.0, 1.0),
          Complex(0.0, 0.0),
          Complex(0.0, 0.0),
          Complex(3.0, -1.0),
        ],
        [2, 2],
        DType.complex128,
      );

      final res = svd(a);
      expect(res.S.dtype, DType.float64);
      expect(res.S.data[0], closeTo(math.sqrt(10), 1e-5));
      expect(res.S.data[1], closeTo(math.sqrt(5), 1e-5));
    });

    test('complex pinv (complex128)', () {
      final a = NDArray.fromList(
        [
          Complex(2.0, 1.0),
          Complex(0.0, 0.0),
          Complex(0.0, 0.0),
          Complex(3.0, -1.0),
        ],
        [2, 2],
        DType.complex128,
      );

      final invA = pinv(a);
      expect(invA.dtype, DType.complex128);
      expect((invA.data[0] as Complex).real, closeTo(0.4, 1e-5));
      expect((invA.data[0] as Complex).imag, closeTo(-0.2, 1e-5));
      expect((invA.data[3] as Complex).real, closeTo(0.3, 1e-5));
      expect((invA.data[3] as Complex).imag, closeTo(0.1, 1e-5));
    });
  });
}
