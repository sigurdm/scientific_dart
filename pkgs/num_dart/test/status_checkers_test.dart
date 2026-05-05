import 'package:num_dart/num_dart.dart';
import 'package:test/test.dart';
import 'dart:typed_data';

void main() {
  group('NDArray Status Checkers (isnan, isinf, isfinite) Tests', () {
    test('isnan, isinf, isfinite basic double (Float64)', () {
      final a = NDArray.fromList(
        Float64List.fromList([
          1.0,
          double.nan,
          double.infinity,
          double.negativeInfinity,
          5.0,
        ]),
        [5],
        DType.float64,
      );

      final nanMask = isnan(a);
      expect(nanMask.dtype, DType.boolean);
      expect(nanMask.toList(), [false, true, false, false, false]);

      final infMask = isinf(a);
      expect(infMask.dtype, DType.boolean);
      expect(infMask.toList(), [false, false, true, true, false]);

      final finiteMask = isfinite(a);
      expect(finiteMask.dtype, DType.boolean);
      expect(finiteMask.toList(), [true, false, false, false, true]);
    });

    test('isnan, isinf, isfinite float32', () {
      final a = NDArray.fromList(
        Float32List.fromList([1.0, double.nan, double.infinity, 5.0]),
        [4],
        DType.float32,
      );

      expect(isnan(a).toList(), [false, true, false, false]);
      expect(isinf(a).toList(), [false, false, true, false]);
      expect(isfinite(a).toList(), [true, false, false, true]);
    });

    test('isnan, isinf, isfinite integer arrays (Int32 and Int64)', () {
      final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
      final b = NDArray.fromList([10, 20, 30], [3], DType.int64);

      // Integers are never NaN or Infinite, and always finite
      expect(isnan(a).toList(), [false, false, false]);
      expect(isinf(a).toList(), [false, false, false]);
      expect(isfinite(a).toList(), [true, true, true]);

      expect(isnan(b).toList(), [false, false, false]);
      expect(isinf(b).toList(), [false, false, false]);
      expect(isfinite(b).toList(), [true, true, true]);
    });

    test(
      'isnan, isinf, isfinite complex arrays (Complex128 and Complex64)',
      () {
        final a = NDArray<Complex>.create([3], DType.complex128);
        a.data[0] = Complex(1.0, 0.0);
        a.data[1] = Complex(double.nan, 1.0);
        a.data[2] = Complex(1.0, double.infinity);

        expect(isnan(a).toList(), [false, true, false]);
        expect(isinf(a).toList(), [false, false, true]);
        expect(isfinite(a).toList(), [true, false, false]);

        final b = NDArray<Complex>.create([3], DType.complex64);
        b.data[0] = Complex(1.0, 0.0);
        b.data[1] = Complex(double.nan, 1.0);
        b.data[2] = Complex(1.0, double.infinity);

        expect(isnan(b).toList(), [false, true, false]);
        expect(isinf(b).toList(), [false, false, true]);
        expect(isfinite(b).toList(), [true, false, false]);
      },
    );

    test('isnan, isinf, isfinite on transposed/sliced views', () {
      final parent = NDArray.fromList(
        Float64List.fromList([1.0, double.nan, double.infinity, 4.0]),
        [2, 2],
        DType.float64,
      );

      final view = parent.transposed;

      expect(isnan(view).toList(), [
        false,
        false,
        true,
        false,
      ]); // transposed order: [1.0, inf, nan, 4.0]
      expect(isinf(view).toList(), [false, true, false, false]);
      expect(isfinite(view).toList(), [true, false, false, true]);
    });

    test('Disposed array checks throw StateError', () {
      final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
      a.dispose();

      expect(() => isnan(a), throwsStateError);
      expect(() => isinf(a), throwsStateError);
      expect(() => isfinite(a), throwsStateError);
    });
  });
}
