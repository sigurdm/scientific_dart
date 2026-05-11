import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';
import 'dart:math' as math;

void main() {
  group('Trigonometric Functions Coverage', () {
    test('sin/cos/tan fallbacks (int types)', () {
      NDArray.scope(() {
        final a = NDArray.fromList([0, 1, 2], [3], DType.int32);
        
        final s = sin(a);
        expect(s.dtype, DType.float64);
        expect(s.data[0], closeTo(math.sin(0), 1e-10));
        expect(s.data[1], closeTo(math.sin(1), 1e-10));
        expect(s.data[2], closeTo(math.sin(2), 1e-10));

        final c = cos(a);
        expect(c.data[0], closeTo(math.cos(0), 1e-10));
        expect(c.data[1], closeTo(math.cos(1), 1e-10));
        expect(c.data[2], closeTo(math.cos(2), 1e-10));

        final t = tan(a);
        expect(t.data[0], closeTo(math.tan(0), 1e-10));
        expect(t.data[1], closeTo(math.tan(1), 1e-10));
        expect(t.data[2], closeTo(math.tan(2), 1e-10));
      });
    });

    test('asin/acos/atan fallbacks (int types)', () {
      NDArray.scope(() {
        final a = NDArray.fromList([0, 1], [2], DType.int32);
        
        final s = asin(a);
        expect(s.data[0], closeTo(math.asin(0), 1e-10));
        expect(s.data[1], closeTo(math.asin(1), 1e-10));

        final c = acos(a);
        expect(c.data[0], closeTo(math.acos(0), 1e-10));
        expect(c.data[1], closeTo(math.acos(1), 1e-10));

        final t = atan(a);
        expect(t.data[0], closeTo(math.atan(0), 1e-10));
        expect(t.data[1], closeTo(math.atan(1), 1e-10));

        // Specific check for asin/acos/atan being correct
        final a2 = NDArray.fromList([1], [1], DType.int32);
        expect(asin(a2).data[0], closeTo(math.asin(1), 1e-10));
        expect(acos(a2).data[0], closeTo(math.acos(1), 1e-10));
        expect(atan(a2).data[0], closeTo(math.atan(1), 1e-10));
      });
    });

    test('sin/cos strided fallbacks (int types)', () {
      NDArray.scope(() {
        // [0, 100, 1, 200, 2] -> slice [0, 1, 2]
        final a = NDArray.fromList([0, 100, 1, 200, 2], [5], DType.int32);
        final sliced = a.slice([const Slice(start: 0, stop: 5, step: 2)]); // [0, 1, 2]
        expect(sliced.shape, [3]);
        expect(sliced.isContiguous, false);

        final s = sin(sliced);
        expect(s.shape, [3]);
        expect(s.data[0], closeTo(math.sin(0), 1e-10));
        expect(s.data[1], closeTo(math.sin(1), 1e-10));
        expect(s.data[2], closeTo(math.sin(2), 1e-10));

        final c = cos(sliced);
        expect(c.data[0], closeTo(math.cos(0), 1e-10));
        expect(c.data[1], closeTo(math.cos(1), 1e-10));
        expect(c.data[2], closeTo(math.cos(2), 1e-10));
      });
    });

    test('atan2 contiguous float64', () {
      NDArray.scope(() {
        final y = NDArray.fromList([1.0, 0.0], [2], DType.float64);
        final x = NDArray.fromList([1.0, 1.0], [2], DType.float64);
        final res = atan2(y, x);
        expect(res.data[0], closeTo(math.atan2(1, 1), 1e-10));
        expect(res.data[1], closeTo(math.atan2(0, 1), 1e-10));
      });
    });

    test('atan2 contiguous float32', () {
      NDArray.scope(() {
        final y = NDArray.fromList([1.0, 0.0], [2], DType.float32);
        final x = NDArray.fromList([1.0, 1.0], [2], DType.float32);
        final res = atan2(y, x);
        expect(res.dtype, DType.float32);
        expect(res.data[0], closeTo(math.atan2(1, 1), 1e-7));
        expect(res.data[1], closeTo(math.atan2(0, 1), 1e-7));
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
        expect(res.data[0], closeTo(math.atan2(1, 1), 1e-10));
        expect(res.data[1], closeTo(math.atan2(0, 1), 1e-10));
      });
    });
  });
}
