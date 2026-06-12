import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';
import 'dart:math' as math;
import 'dart:typed_data';

void main() {
  group('hypot and power ufuncs', () {
    test('hypot basic', () {
      final a = NDArray.fromList(Float64List.fromList([3.0, 5.0]), [
        2,
      ], DType.float64);
      final b = NDArray.fromList(Float64List.fromList([4.0, 12.0]), [
        2,
      ], DType.float64);
      final h = hypot(a, b);
      expect(h.data[0], closeTo(5.0, 1e-10));
      expect(h.data[1], closeTo(13.0, 1e-10));
      expect(h.dtype, DType.float64);
    });

    test('hypot broadcasting', () {
      final a = NDArray.fromList(Float64List.fromList([3.0, 8.0]), [
        2,
      ], DType.float64);
      final b = NDArray.fromList(Float64List.fromList([4.0]), [
        1,
      ], DType.float64);
      final h = hypot(a, b);
      expect(h.data[0], closeTo(5.0, 1e-10));
      expect(h.data[1], closeTo(math.sqrt(80.0), 1e-10));
    });

    test('power basic', () {
      final a = NDArray.fromList(Float64List.fromList([2.0, 3.0]), [
        2,
      ], DType.float64);
      final b = NDArray.fromList(Float64List.fromList([3.0, 2.0]), [
        2,
      ], DType.float64);
      final p = power(a, b);
      expect(p.data[0], 8.0);
      expect(p.data[1], 9.0);
    });

    test('power broadcasting', () {
      final a = NDArray.fromList(Int32List.fromList([2, 3]), [2], DType.int32);
      final b = NDArray.fromList(Int32List.fromList([2]), [1], DType.int32);
      final p = power(a, b);
      expect(p.data[0], 4);
      expect(p.data[1], 9);
      expect(p.dtype, DType.int32);
    });
  });
}
