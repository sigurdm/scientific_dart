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
      addTearDown(a.dispose);

      final nanMask = isnan(a);
      addTearDown(nanMask.dispose);
      expect(nanMask.dtype, DType.boolean);
      expect(nanMask.toList(), [false, true, false, false, false]);

      final infMask = isinf(a);
      addTearDown(infMask.dispose);
      expect(infMask.dtype, DType.boolean);
      expect(infMask.toList(), [false, false, true, true, false]);

      final finiteMask = isfinite(a);
      addTearDown(finiteMask.dispose);
      expect(finiteMask.dtype, DType.boolean);
      expect(finiteMask.toList(), [true, false, false, false, true]);
    });

    test('isnan, isinf, isfinite float32', () {
      final a = NDArray.fromList(
        Float32List.fromList([1.0, double.nan, double.infinity, 5.0]),
        [4],
        DType.float32,
      );
      addTearDown(a.dispose);

      final resNan = isnan(a);
      addTearDown(resNan.dispose);
      expect(resNan.toList(), [false, true, false, false]);

      final resInf = isinf(a);
      addTearDown(resInf.dispose);
      expect(resInf.toList(), [false, false, true, false]);

      final resFinite = isfinite(a);
      addTearDown(resFinite.dispose);
      expect(resFinite.toList(), [true, false, false, true]);
    });

    test('isnan, isinf, isfinite integer arrays (Int32 and Int64)', () {
      final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
      addTearDown(a.dispose);
      final b = NDArray.fromList([10, 20, 30], [3], DType.int64);
      addTearDown(b.dispose);

      // Integers are never NaN or Infinite, and always finite
      final nanA = isnan(a);
      addTearDown(nanA.dispose);
      expect(nanA.toList(), [false, false, false]);

      final infA = isinf(a);
      addTearDown(infA.dispose);
      expect(infA.toList(), [false, false, false]);

      final finiteA = isfinite(a);
      addTearDown(finiteA.dispose);
      expect(finiteA.toList(), [true, true, true]);

      final nanB = isnan(b);
      addTearDown(nanB.dispose);
      expect(nanB.toList(), [false, false, false]);

      final infB = isinf(b);
      addTearDown(infB.dispose);
      expect(infB.toList(), [false, false, false]);

      final finiteB = isfinite(b);
      addTearDown(finiteB.dispose);
      expect(finiteB.toList(), [true, true, true]);
    });

    test(
      'isnan, isinf, isfinite complex arrays (Complex128 and Complex64)',
      () {
        final a = NDArray<Complex>.create([3], DType.complex128);
        addTearDown(a.dispose);
        a.data[0] = Complex(1.0, 0.0);
        a.data[1] = Complex(double.nan, 1.0);
        a.data[2] = Complex(1.0, double.infinity);

        final nanA = isnan(a);
        addTearDown(nanA.dispose);
        expect(nanA.toList(), [false, true, false]);

        final infA = isinf(a);
        addTearDown(infA.dispose);
        expect(infA.toList(), [false, false, true]);

        final finiteA = isfinite(a);
        addTearDown(finiteA.dispose);
        expect(finiteA.toList(), [true, false, false]);

        final b = NDArray<Complex>.create([3], DType.complex64);
        addTearDown(b.dispose);
        b.data[0] = Complex(1.0, 0.0);
        b.data[1] = Complex(double.nan, 1.0);
        b.data[2] = Complex(1.0, double.infinity);

        final nanB = isnan(b);
        addTearDown(nanB.dispose);
        expect(nanB.toList(), [false, true, false]);

        final infB = isinf(b);
        addTearDown(infB.dispose);
        expect(infB.toList(), [false, false, true]);

        final finiteB = isfinite(b);
        addTearDown(finiteB.dispose);
        expect(finiteB.toList(), [true, false, false]);
      },
    );

    test('isnan, isinf, isfinite on transposed/sliced views', () {
      final parent = NDArray.fromList(
        Float64List.fromList([1.0, double.nan, double.infinity, 4.0]),
        [2, 2],
        DType.float64,
      );
      addTearDown(parent.dispose);

      final view = parent.transposed;
      addTearDown(view.dispose);

      final resNan = isnan(view);
      addTearDown(resNan.dispose);
      expect(resNan.toList(), [
        false,
        false,
        true,
        false,
      ]); // transposed order: [1.0, inf, nan, 4.0]

      final resInf = isinf(view);
      addTearDown(resInf.dispose);
      expect(resInf.toList(), [false, true, false, false]);

      final resFinite = isfinite(view);
      addTearDown(resFinite.dispose);
      expect(resFinite.toList(), [true, false, false, true]);
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
