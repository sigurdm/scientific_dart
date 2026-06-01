import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';
import 'dart:typed_data';

void main() {
  group('Mathematical ufuncs with out parameter', () {
    test('hypot with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([3.0, 5.0]), [
        2,
      ], DType.float64);
      final b = NDArray.fromList(Float64List.fromList([4.0, 12.0]), [
        2,
      ], DType.float64);
      final out = NDArray<double>.create([2], DType.float64);

      final res = hypot(a, b, out: out);
      expect(identical(res, out), isTrue);
      expect(out.data[0], closeTo(5.0, 1e-10));
      expect(out.data[1], closeTo(13.0, 1e-10));

      // Incompatible shape/dtype validation
      final badOut = NDArray<double>.create([3], DType.float64);
      expect(() => hypot(a, b, out: badOut), throwsArgumentError);
    });

    test('power with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([2.0, 3.0]), [
        2,
      ], DType.float64);
      final b = NDArray.fromList(Float64List.fromList([3.0, 2.0]), [
        2,
      ], DType.float64);
      final out = NDArray.create([2], DType.float64);

      final res = power(a, b, out: out);
      expect(identical(res, out), isTrue);
      expect(out.data[0], 8.0);
      expect(out.data[1], 9.0);

      final badOut = NDArray.create([2], DType.float32);
      expect(() => power(a, b, out: badOut), throwsArgumentError);
    });

    test('negative with out parameter', () {
      final a = NDArray.fromList(Int32List.fromList([2, -3]), [2], DType.int32);
      final out = NDArray.create([2], DType.int32);

      final res = negative(a, out: out);
      expect(identical(res, out), isTrue);
      expect(out.data[0], -2);
      expect(out.data[1], 3);

      final badOut = NDArray.create([3], DType.int32);
      expect(() => negative(a, out: badOut), throwsArgumentError);
    });

    test('floor_divide with out parameter', () {
      final a = NDArray.fromList(Int32List.fromList([10, 25]), [
        2,
      ], DType.int32);
      final b = NDArray.fromList(Int32List.fromList([3, 4]), [2], DType.int32);
      final out = NDArray.create([2], DType.int32);

      final res = floor_divide(a, b, out: out);
      expect(identical(res, out), isTrue);
      expect(out.data[0], 3);
      expect(out.data[1], 6);

      final badOut = NDArray.create([2], DType.float64);
      expect(() => floor_divide(a, b, out: badOut), throwsArgumentError);
    });

    test('remainder and mod with out parameter', () {
      final a = NDArray.fromList(Int32List.fromList([10, 25]), [
        2,
      ], DType.int32);
      final b = NDArray.fromList(Int32List.fromList([3, 4]), [2], DType.int32);
      final out = NDArray.create([2], DType.int32);

      final res = remainder(a, b, out: out);
      expect(identical(res, out), isTrue);
      expect(out.data[0], 1);
      expect(out.data[1], 1);

      final outMod = NDArray.create([2], DType.int32);
      final resMod = mod(a, b, out: outMod);
      expect(identical(resMod, outMod), isTrue);
      expect(outMod.data[0], 1);
      expect(outMod.data[1], 1);
    });

    test('abs with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([-1.5, 2.0]), [
        2,
      ], DType.float64);
      final out = NDArray.create([2], DType.float64);

      final res = abs(a, out: out);
      expect(identical(res, out), isTrue);
      expect(out.data[0], 1.5);
      expect(out.data[1], 2.0);
    });

    test('sign with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([-1.5, 0.0, 2.5]), [
        3,
      ], DType.float64);
      final out = NDArray.create([3], DType.float64);

      final res = sign(a, out: out);
      expect(identical(res, out), isTrue);
      expect(out.data[0], -1.0);
      expect(out.data[1], 0.0);
      expect(out.data[2], 1.0);
    });

    test('ceil with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([1.2, -1.7]), [
        2,
      ], DType.float64);
      final out = NDArray.create([2], DType.float64);

      final res = ceil(a, out: out);
      expect(identical(res, out), isTrue);
      expect(out.data[0], 2.0);
      expect(out.data[1], -1.0);
    });

    test('floor with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([1.2, -1.7]), [
        2,
      ], DType.float64);
      final out = NDArray.create([2], DType.float64);

      final res = floor(a, out: out);
      expect(identical(res, out), isTrue);
      expect(out.data[0], 1.0);
      expect(out.data[1], -2.0);
    });

    test('round with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([1.2, -1.7]), [
        2,
      ], DType.float64);
      final out = NDArray.create([2], DType.float64);

      final res = round(a, out: out);
      expect(identical(res, out), isTrue);
      expect(out.data[0], 1.0);
      expect(out.data[1], -2.0);
    });

    test('isnan with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, double.nan]), [
        2,
      ], DType.float64);
      final out = NDArray<bool>.create([2], DType.boolean);

      final res = isnan(a, out: out);
      expect(identical(res, out), isTrue);
      expect(out.data[0], isFalse);
      expect(out.data[1], isTrue);
    });

    test('isinf with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, double.infinity]), [
        2,
      ], DType.float64);
      final out = NDArray<bool>.create([2], DType.boolean);

      final res = isinf(a, out: out);
      expect(identical(res, out), isTrue);
      expect(out.data[0], isFalse);
      expect(out.data[1], isTrue);
    });

    test('isfinite with out parameter', () {
      final a = NDArray.fromList(Float64List.fromList([1.0, double.infinity]), [
        2,
      ], DType.float64);
      final out = NDArray<bool>.create([2], DType.boolean);

      final res = isfinite(a, out: out);
      expect(identical(res, out), isTrue);
      expect(out.data[0], isTrue);
      expect(out.data[1], isFalse);
    });

    test('copysign with out parameter', () {
      final x1 = NDArray.fromList(Float64List.fromList([2.0, -3.0]), [
        2,
      ], DType.float64);
      final x2 = NDArray.fromList(Float64List.fromList([-1.0, 1.0]), [
        2,
      ], DType.float64);
      final out = NDArray.create([2], DType.float64);

      final res = copysign(x1, x2, out: out);
      expect(identical(res, out), isTrue);
      expect(out.data[0], -2.0);
      expect(out.data[1], 3.0);
    });
  });
}
