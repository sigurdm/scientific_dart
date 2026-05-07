import 'dart:math' as math;
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('NDArray Angle Converters (deg2rad, rad2deg) Tests', () {
    test('deg2rad basic float64 contiguous conversion checks',
        () => NDArray.scope(() {
      final a = NDArray.fromList([180.0, 90.0, 0.0, -45.0], [4], DType.float64);
      final res = deg2rad(a);

      expect(res.dtype, DType.float64);
      expect(res.data[0], closeTo(math.pi, 1e-10));
      expect(res.data[1], closeTo(math.pi / 2.0, 1e-10));
      expect(res.data[2], 0.0);
      expect(res.data[3], closeTo(-math.pi / 4.0, 1e-10));
    }));

    test('rad2deg basic float64 contiguous conversion checks',
        () => NDArray.scope(() {
      final a = NDArray.fromList(
        [math.pi, math.pi / 2.0, 0.0, -math.pi / 4.0],
        [4],
        DType.float64,
      );
      final res = rad2deg(a);

      expect(res.dtype, DType.float64);
      expect(res.data[0], closeTo(180.0, 1e-10));
      expect(res.data[1], closeTo(90.0, 1e-10));
      expect(res.data[2], 0.0);
      expect(res.data[3], closeTo(-45.0, 1e-10));
    }));

    test('deg2rad and rad2deg support Float32', () => NDArray.scope(() {
      final a = NDArray.fromList([180.0, 90.0], [2], DType.float32);
      final r = deg2rad(a);
      expect(r.dtype, DType.float32);
      expect(r.data[0], closeTo(math.pi, 1e-5));

      final b = NDArray.fromList([math.pi, math.pi / 2.0], [2], DType.float32);
      final d = rad2deg(b);
      expect(d.dtype, DType.float32);
      expect(d.data[0], closeTo(180.0, 1e-5));
    }));

    test('Complex arrays throw UnsupportedError', () => NDArray.scope(() {
      final a = NDArray<Complex>.create([2], DType.complex128);
      expect(() => deg2rad(a), throwsUnsupportedError);
      expect(() => rad2deg(a), throwsUnsupportedError);
    }));

    test('Disposed arrays throw StateError', () => NDArray.scope(() {
      final a = NDArray.fromList([180.0], [1], DType.float64);
      expect(() => deg2rad(a), throwsStateError);
      expect(() => rad2deg(a), throwsStateError);
    }));
  });
}
