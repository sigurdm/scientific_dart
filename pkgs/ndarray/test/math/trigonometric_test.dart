import 'package:ndarray/ndarray.dart';
import 'dart:typed_data';
import 'package:test/test.dart';
import 'dart:math' as math;

void main() {
  group('Trigonometric Tests', () {
    test('sin/cos/tan fallbacks (int types)', () {
      NDArray.scope(() {
        final a = NDArray.fromList([0, 1, 2], [3], DType.int32);

        final s = sin(a);
        expect(s.dtype, DType.float64);
        expect(s.toList()[0], closeTo(math.sin(0), 1e-10));
        expect(s.toList()[1], closeTo(math.sin(1), 1e-10));
        expect(s.toList()[2], closeTo(math.sin(2), 1e-10));

        final c = cos(a);
        expect(c.toList()[0], closeTo(math.cos(0), 1e-10));
        expect(c.toList()[1], closeTo(math.cos(1), 1e-10));
        expect(c.toList()[2], closeTo(math.cos(2), 1e-10));

        final t = tan(a);
        expect(t.toList()[0], closeTo(math.tan(0), 1e-10));
        expect(t.toList()[1], closeTo(math.tan(1), 1e-10));
        expect(t.toList()[2], closeTo(math.tan(2), 1e-10));
      });
    });

    test('asin/acos/atan fallbacks (int types)', () {
      NDArray.scope(() {
        final a = NDArray.fromList([0, 1], [2], DType.int32);

        final s = asin(a);
        expect(s.toList()[0], closeTo(math.asin(0), 1e-10));
        expect(s.toList()[1], closeTo(math.asin(1), 1e-10));

        final c = acos(a);
        expect(c.toList()[0], closeTo(math.acos(0), 1e-10));
        expect(c.toList()[1], closeTo(math.acos(1), 1e-10));

        final t = atan(a);
        expect(t.toList()[0], closeTo(math.atan(0), 1e-10));
        expect(t.toList()[1], closeTo(math.atan(1), 1e-10));

        // Specific check for asin/acos/atan being correct
        final a2 = NDArray.fromList([1], [1], DType.int32);
        expect(asin(a2).toList()[0], closeTo(math.asin(1), 1e-10));
        expect(acos(a2).toList()[0], closeTo(math.acos(1), 1e-10));
        expect(atan(a2).toList()[0], closeTo(math.atan(1), 1e-10));
      });
    });

    test('sin/cos strided fallbacks (int types)', () {
      NDArray.scope(() {
        // [0, 100, 1, 200, 2] -> slice [0, 1, 2]
        final a = NDArray.fromList([0, 100, 1, 200, 2], [5], DType.int32);
        final sliced = a.slice([
          const Slice(start: 0, stop: 5, step: 2),
        ]); // [0, 1, 2]
        expect(sliced.shape, [3]);
        expect(sliced.isContiguous, false);

        final s = sin(sliced);
        expect(s.shape, [3]);
        expect(s.toList()[0], closeTo(math.sin(0), 1e-10));
        expect(s.toList()[1], closeTo(math.sin(1), 1e-10));
        expect(s.toList()[2], closeTo(math.sin(2), 1e-10));

        final c = cos(sliced);
        expect(c.toList()[0], closeTo(math.cos(0), 1e-10));
        expect(c.toList()[1], closeTo(math.cos(1), 1e-10));
        expect(c.toList()[2], closeTo(math.cos(2), 1e-10));
      });
    });

    test('atan2 contiguous float64', () {
      NDArray.scope(() {
        final y = NDArray.fromList([1.0, 0.0], [2], DType.float64);
        final x = NDArray.fromList([1.0, 1.0], [2], DType.float64);
        final res = atan2(y, x);
        expect(res.toList()[0], closeTo(math.atan2(1, 1), 1e-10));
        expect(res.toList()[1], closeTo(math.atan2(0, 1), 1e-10));
      });
    });

    test('atan2 contiguous float32', () {
      NDArray.scope(() {
        final y = NDArray.fromList([1.0, 0.0], [2], DType.float32);
        final x = NDArray.fromList([1.0, 1.0], [2], DType.float32);
        final res = atan2(y, x);
        expect(res.dtype, DType.float32);
        expect(res.toList()[0], closeTo(math.atan2(1, 1), 1e-7));
        expect(res.toList()[1], closeTo(math.atan2(0, 1), 1e-7));
      });
    });

    test('atan2 strided broadcasting', () {
      NDArray.scope(() {
        // y: [1.0, 2.0] (2,)
        // x: [[1.0], [2.0]] (2, 1)
        // result: (2, 2)
        final y = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final x = NDArray.fromList([1.0, 2.0], [2, 1], DType.float64);
        final res = atan2(y, x);

        expect(res.shape, [2, 2]);
        // [atan2(1,1), atan2(2,1)]
        // [atan2(1,2), atan2(2,2)]
        expect(res[[0, 0]], closeTo(math.atan2(1, 1), 1e-10));
        expect(res[[0, 1]], closeTo(math.atan2(2, 1), 1e-10));
        expect(res[[1, 0]], closeTo(math.atan2(1, 2), 1e-10));
        expect(res[[1, 1]], closeTo(math.atan2(2, 2), 1e-10));
      });
    });

    test('atan2 int fallbacks', () {
      NDArray.scope(() {
        final y = NDArray.fromList([1, 0], [2], DType.int32);
        final x = NDArray.fromList([1, 1], [2], DType.int32);
        final res = atan2(y, x);
        expect(res.toList()[0], closeTo(math.atan2(1, 1), 1e-10));
        expect(res.toList()[1], closeTo(math.atan2(0, 1), 1e-10));
      });
    });
  });

  group('Sinh Bug Repro', () {
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
  });

  group('Angle Converters (deg2rad, rad2deg) Tests', () {
    test(
      'deg2rad basic float64 contiguous conversion checks',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [180.0, 90.0, 0.0, -45.0],
          [4],
          DType.float64,
        );
        final res = deg2rad(a);

        expect(res.dtype, DType.float64);
        expect(res.toList()[0], closeTo(math.pi, 1e-10));
        expect(res.toList()[1], closeTo(math.pi / 2.0, 1e-10));
        expect(res.toList()[2], 0.0);
        expect(res.toList()[3], closeTo(-math.pi / 4.0, 1e-10));
      }),
    );

    test(
      'rad2deg basic float64 contiguous conversion checks',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [math.pi, math.pi / 2.0, 0.0, -math.pi / 4.0],
          [4],
          DType.float64,
        );
        final res = rad2deg(a);

        expect(res.dtype, DType.float64);
        expect(res.toList()[0], closeTo(180.0, 1e-10));
        expect(res.toList()[1], closeTo(90.0, 1e-10));
        expect(res.toList()[2], 0.0);
        expect(res.toList()[3], closeTo(-45.0, 1e-10));
      }),
    );

    test(
      'deg2rad and rad2deg support Float32',
      () => NDArray.scope(() {
        final a = NDArray.fromList([180.0, 90.0], [2], DType.float32);
        final r = deg2rad(a);
        expect(r.dtype, DType.float32);
        expect(r.toList()[0], closeTo(math.pi, 1e-5));

        final b = NDArray.fromList(
          [math.pi, math.pi / 2.0],
          [2],
          DType.float32,
        );
        final d = rad2deg(b);
        expect(d.dtype, DType.float32);
        expect(d.toList()[0], closeTo(180.0, 1e-5));
      }),
    );

    test(
      'Complex arrays throw UnsupportedError',
      () => NDArray.scope(() {
        final a = NDArray<Complex>.create([2], DType.complex128);
        expect(() => deg2rad(a), throwsUnsupportedError);
        expect(() => rad2deg(a), throwsUnsupportedError);
      }),
    );

    test(
      'Disposed arrays throw StateError',
      () => NDArray.scope(() {
        final a = NDArray.fromList([180.0], [1], DType.float64);
        a.dispose();
        expect(() => deg2rad(a), throwsStateError);
        expect(() => rad2deg(a), throwsStateError);
      }),
    );
  });

  group('hypot and power ufuncs', () {
    test('hypot basic', () {
      final a = NDArray.fromList(Float64List.fromList([3.0, 5.0]), [
        2,
      ], DType.float64);
      final b = NDArray.fromList(Float64List.fromList([4.0, 12.0]), [
        2,
      ], DType.float64);
      final h = hypot(a, b);
      expect(h.toList()[0], closeTo(5.0, 1e-10));
      expect(h.toList()[1], closeTo(13.0, 1e-10));
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
      expect(h.toList()[0], closeTo(5.0, 1e-10));
      expect(h.toList()[1], closeTo(math.sqrt(80.0), 1e-10));
    });

    test('power basic', () {
      final a = NDArray.fromList(Float64List.fromList([2.0, 3.0]), [
        2,
      ], DType.float64);
      final b = NDArray.fromList(Float64List.fromList([3.0, 2.0]), [
        2,
      ], DType.float64);
      final p = power(a, b);
      expect(p.toList()[0], 8.0);
      expect(p.toList()[1], 9.0);
    });

    test('power broadcasting', () {
      final a = NDArray.fromList(Int32List.fromList([2, 3]), [2], DType.int32);
      final b = NDArray.fromList(Int32List.fromList([2]), [1], DType.int32);
      final p = power(a, b);
      expect(p.toList()[0], 4);
      expect(p.toList()[1], 9);
      expect(p.dtype, DType.int32);
    });
  });

  group('sign ufunc', () {
    test('int32 sign', () {
      final a = NDArray.fromList(Int32List.fromList([-5, 0, 10]), [
        3,
      ], DType.int32);
      final s = sign(a);
      expect(s.toList(), [-1, 0, 1]);
      expect(s.dtype, DType.int32);
    });

    test('float64 sign', () {
      final a = NDArray.fromList(
        Float64List.fromList([-5.5, 0.0, 10.2, double.nan]),
        [4],
        DType.float64,
      );
      final s = sign(a);
      expect(s.toList()[0], -1.0);
      expect(s.toList()[1], 0.0);
      expect(s.toList()[2], 1.0);
      expect(s.toList()[3].isNaN, isTrue);
      expect(s.dtype, DType.float64);
    });

    test('complex128 sign', () {
      final a = NDArray.fromList(
        [Complex(3.0, 4.0), Complex(0.0, 0.0)],
        [2],
        DType.complex128,
      );

      final s = sign(a);
      expect(s.toList()[0].real, closeTo(0.6, 1e-10));
      expect(s.toList()[0].imag, closeTo(0.8, 1e-10));
      expect(s.toList()[1].real, 0.0);
      expect(s.toList()[1].imag, 0.0);
      expect(s.dtype, DType.complex128);
    });
  });
}
