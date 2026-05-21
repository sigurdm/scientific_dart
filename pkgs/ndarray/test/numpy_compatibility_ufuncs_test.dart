import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';
import 'dart:typed_data';

void main() {
  group('NumPy Compatibility Universal Functions (ufuncs) tests', () {
    group('square ufunc', () {
      test('contiguous basic types', () {
        final a = NDArray.fromList(Float64List.fromList([2.0, -3.0, 4.0]), [
          3,
        ], DType.float64);
        final res = square(a);
        expect(res.data[0], 4.0);
        expect(res.data[1], 9.0);
        expect(res.data[2], 16.0);
        expect(res.dtype, DType.float64);

        final aInt = NDArray.fromList(Int32List.fromList([2, -5, 10]), [
          3,
        ], DType.int32);
        final resInt = square(aInt);
        expect(resInt.data[0], 4);
        expect(resInt.data[1], 25);
        expect(resInt.data[2], 100);
        expect(resInt.dtype, DType.int32);
      });

      test('complex types', () {
        final a = NDArray<Complex>.create([2], DType.complex128);
        a.data[0] = Complex(3.0, 4.0); // (3+4i)^2 = 9 - 16 + 24i = -7 + 24i
        a.data[1] = Complex(0.0, -2.0); // (0-2i)^2 = -4

        final res = square(a);
        expect(res.data[0].real, closeTo(-7.0, 1e-10));
        expect(res.data[0].imag, closeTo(24.0, 1e-10));
        expect(res.data[1].real, closeTo(-4.0, 1e-10));
        expect(res.data[1].imag, closeTo(0.0, 1e-10));
      });

      test('strided non-contiguous views', () {
        // Shape [3, 2]
        final parent = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
          [3, 2],
          DType.float64,
        );

        // Slice column 0: [1.0, 3.0, 5.0] (strided)
        final col0 = parent.slice([Slice(start: 0, stop: 3), Index(0)]);
        expect(col0.isContiguous, false);

        final res = square(col0);
        expect(res.data[0], 1.0);
        expect(res.data[1], 9.0);
        expect(res.data[2], 25.0);
      });

      test('boolean array and fallbacks', () {
        final aBool = NDArray.fromList([true, false], [2], DType.boolean);
        final resBool = square(aBool);
        expect(resBool.data[0], true);
        expect(resBool.data[1], false);

        // Fallback types (uint8)
        final aUint = NDArray.fromList(Uint8List.fromList([3, 5]), [
          2,
        ], DType.uint8);
        final resUint = square(aUint);
        expect(resUint.data[0], 9);
        expect(resUint.data[1], 25);
      });

      test('out recycler buffer', () {
        final a = NDArray.fromList(Float64List.fromList([2.0, 3.0]), [
          2,
        ], DType.float64);
        final outRecycler = NDArray<double>.zeros([2], DType.float64);
        final res = square(a, out: outRecycler);
        expect(identical(res, outRecycler), true);
        expect(outRecycler.data[0], 4.0);
        expect(outRecycler.data[1], 9.0);
      });
    });

    group('power float ufunc FFI speedups', () {
      test('basic float power', () {
        final x1 = NDArray.fromList(Float64List.fromList([2.0, 3.0]), [
          2,
        ], DType.float64);
        final x2 = NDArray.fromList(Float64List.fromList([3.0, 2.0]), [
          2,
        ], DType.float64);
        final res = power(x1, x2);
        expect(res.data[0], 8.0);
        expect(res.data[1], 9.0);
      });

      test('broadcasting strided power', () {
        final x1 = NDArray.fromList(Float64List.fromList([2.0, 3.0]), [
          2,
        ], DType.float64);
        final x2 = NDArray.fromList(Float64List.fromList([3.0]), [
          1,
        ], DType.float64);
        final res = power(x1, x2);
        expect(res.data[0], 8.0);
        expect(res.data[1], 27.0);
      });
    });

    group('floor_divide, remainder/mod, and divmod ufuncs', () {
      test(
        'positive and negative floor division combinations matching Python/NumPy',
        () {
          final x = NDArray.fromList(
            Float64List.fromList([5.0, -5.0, -5.0, 5.0]),
            [4],
            DType.float64,
          );
          final y = NDArray.fromList(
            Float64List.fromList([2.0, 2.0, -2.0, -2.0]),
            [4],
            DType.float64,
          );

          final q = floor_divide(x, y);
          // 5 // 2 = 2
          expect(q.data[0], 2.0);
          // -5 // 2 = -3
          expect(q.data[1], -3.0);
          // -5 // -2 = 2
          expect(q.data[2], 2.0);
          // 5 // -2 = -3
          expect(q.data[3], -3.0);
        },
      );

      test('remainder and mod matching Python/NumPy signed floor modulo', () {
        final x = NDArray.fromList(
          Float64List.fromList([5.0, -5.0, -5.0, 5.0]),
          [4],
          DType.float64,
        );
        final y = NDArray.fromList(
          Float64List.fromList([2.0, 2.0, -2.0, -2.0]),
          [4],
          DType.float64,
        );

        final r = remainder(x, y);
        final m = mod(x, y);
        // 5 % 2 = 1
        expect(r.data[0], 1.0);
        expect(m.data[0], 1.0);
        // -5 % 2 = 1 (since -5 = 2 * -3 + 1)
        expect(r.data[1], 1.0);
        expect(m.data[1], 1.0);
        // -5 % -2 = -1 (since -5 = -2 * 2 - 1)
        expect(r.data[2], -1.0);
        expect(m.data[2], -1.0);
        // 5 % -2 = -1 (since 5 = -2 * -3 - 1)
        expect(r.data[3], -1.0);
        expect(m.data[3], -1.0);
      });

      test('divmod combined tuple', () {
        final x = NDArray.fromList(Int32List.fromList([5, -5]), [
          2,
        ], DType.int32);
        final y = NDArray.fromList(Int32List.fromList([-2, -2]), [
          2,
        ], DType.int32);

        final res = divmod(x, y);
        final q = res.$1;
        final r = res.$2;

        expect(q.data[0], -3);
        expect(r.data[0], -1);

        expect(q.data[1], 2);
        expect(r.data[1], -1);
      });

      test('integer division by zero checks', () {
        final x = NDArray.fromList(Int32List.fromList([5]), [1], DType.int32);
        final y = NDArray.fromList(Int32List.fromList([0]), [1], DType.int32);

        expect(() => floor_divide(x, y), throwsA(isA<UnsupportedError>()));
        expect(() => remainder(x, y), throwsA(isA<UnsupportedError>()));
        expect(() => divmod(x, y), throwsA(isA<UnsupportedError>()));
      });
    });

    group('floating-point classifications (isnan, isinf, isfinite)', () {
      test('real types', () {
        final a = NDArray.fromList(
          Float64List.fromList([
            1.0,
            double.nan,
            double.infinity,
            double.negativeInfinity,
          ]),
          [4],
          DType.float64,
        );

        final nanMask = isnan(a);
        expect(nanMask.data[0], false);
        expect(nanMask.data[1], true);
        expect(nanMask.data[2], false);
        expect(nanMask.data[3], false);

        final infMask = isinf(a);
        expect(infMask.data[0], false);
        expect(infMask.data[1], false);
        expect(infMask.data[2], true);
        expect(infMask.data[3], true);

        final finMask = isfinite(a);
        expect(finMask.data[0], true);
        expect(finMask.data[1], false);
        expect(finMask.data[2], false);
        expect(finMask.data[3], false);
      });

      test('complex types', () {
        final a = NDArray<Complex>.create([3], DType.complex128);
        a.data[0] = Complex(double.nan, 1.0);
        a.data[1] = Complex(1.0, double.infinity);
        a.data[2] = Complex(1.0, 2.0);

        final nanMask = isnan(a);
        expect(nanMask.data[0], true);
        expect(nanMask.data[1], false);
        expect(nanMask.data[2], false);

        final infMask = isinf(a);
        expect(infMask.data[0], false);
        expect(infMask.data[1], true);
        expect(infMask.data[2], false);

        final finMask = isfinite(a);
        expect(finMask.data[0], false);
        expect(finMask.data[1], false);
        expect(finMask.data[2], true);
      });

      test('integer fallbacks always clean', () {
        final a = NDArray.fromList(Int32List.fromList([1, 2]), [
          2,
        ], DType.int32);
        expect(isnan(a).data.every((x) => !x), true);
        expect(isinf(a).data.every((x) => !x), true);
        expect(isfinite(a).data.every((x) => x), true);
      });
    });

    group('copysign ufunc', () {
      test('basic copysign', () {
        final x1 = NDArray.fromList(Float64List.fromList([1.0, -1.0, 2.0]), [
          3,
        ], DType.float64);
        final x2 = NDArray.fromList(Float64List.fromList([-3.0, 5.0, -0.0]), [
          3,
        ], DType.float64);

        final res = copysign(x1, x2);
        expect(res.data[0], -1.0);
        expect(res.data[1], 1.0);
        // sign of -0.0 should be negative!
        expect(res.data[2].isNegative, true);
        expect(res.data[2].abs(), 2.0);
      });

      test('broadcasting copysign', () {
        final x1 = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          2,
        ], DType.float64);
        final x2 = NDArray.fromList(Float64List.fromList([-5.0]), [
          1,
        ], DType.float64);

        final res = copysign(x1, x2);
        expect(res.data[0], -1.0);
        expect(res.data[1], -2.0);
      });
    });
  });
}
