import 'package:num_dart/num_dart.dart';
import 'package:test/test.dart';
import 'dart:math' as math;
import 'dart:typed_data';

void main() {
  group('Universal Functions (ufuncs) & Broadcasting Tests', () {
    group('Mixed DType Arithmetic Upcasting', () {
      test('Add Complex and Float64 arrays', () {
        final a = NDArray<Complex>.create([2], DType.complex128);
        addTearDown(a.dispose);
        a.data[0] = Complex(1.0, 2.0);
        a.data[1] = Complex(3.0, 4.0);

        final b = NDArray.fromList(Float64List.fromList([10.0, 20.0]), [
          2,
        ], DType.float64);
        addTearDown(b.dispose);

        // Should not crash with type-cast error, should upcast b to Complex
        final c = add(a, b);
        addTearDown(c.dispose);
        expect(c.dtype, DType.complex128);
        expect(c.shape, [2]);
        expect(c.data[0], Complex(11.0, 2.0));
        expect(c.data[1], Complex(23.0, 4.0));
      });

      test('Subtract Float64 and Complex arrays', () {
        final a = NDArray.fromList(Float64List.fromList([10.0, 20.0]), [
          2,
        ], DType.float64);
        addTearDown(a.dispose);

        final b = NDArray<Complex>.create([2], DType.complex128);
        addTearDown(b.dispose);
        b.data[0] = Complex(1.0, 2.0);
        b.data[1] = Complex(3.0, 4.0);

        final c = subtract(a, b);
        addTearDown(c.dispose);
        expect(c.dtype, DType.complex128);
        expect(c.data[0], Complex(9.0, -2.0));
        expect(c.data[1], Complex(17.0, -4.0));
      });

      test('Multiply Int32 and Complex arrays', () {
        final a = NDArray.fromList(Int32List.fromList([2, 3]), [
          2,
        ], DType.int32);
        addTearDown(a.dispose);

        final b = NDArray<Complex>.create([2], DType.complex128);
        addTearDown(b.dispose);
        b.data[0] = Complex(4.0, 5.0);
        b.data[1] = Complex(1.0, -2.0);

        final c = multiply(a, b);
        addTearDown(c.dispose);
        expect(c.dtype, DType.complex128);
        expect(c.data[0], Complex(8.0, 10.0));
        expect(c.data[1], Complex(3.0, -6.0));
      });

      test('True Division always returns floats or complex', () {
        final a = NDArray.fromList(Int32List.fromList([5, 10]), [
          2,
        ], DType.int32);
        addTearDown(a.dispose);
        final b = NDArray.fromList(Int32List.fromList([2, 4]), [
          2,
        ], DType.int32);
        addTearDown(b.dispose);

        // int / int -> float64
        final c = divide(a, b);
        addTearDown(c.dispose);
        expect(c.dtype, DType.float64);
        expect(c.data[0], 2.5);
        expect(c.data[1], 2.5);
      });
    });

    group('New Math Ufuncs', () {
      test('tan', () {
        final a = NDArray.fromList(Float64List.fromList([0.0, math.pi / 4]), [
          2,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = tan(a);
        addTearDown(b.dispose);
        expect(b.shape, [2]);
        expect(b.data[0], closeTo(0.0, 1e-10));
        expect(b.data[1], closeTo(1.0, 1e-10));

        final c = NDArray<Complex>.create([1], DType.complex128);
        addTearDown(c.dispose);
        expect(() => tan(c), throwsUnsupportedError);
      });

      test('atan2 with broadcasting', () {
        final y = NDArray.fromList(Float64List.fromList([1.0]), [
          1,
        ], DType.float64);
        addTearDown(y.dispose);
        final x = NDArray.fromList(Float64List.fromList([1.0, math.sqrt(3)]), [
          2,
        ], DType.float64);
        addTearDown(x.dispose);

        final b = atan2(y, x); // y broadcasts to [2]
        addTearDown(b.dispose);
        expect(b.shape, [2]);
        expect(b.data[0], closeTo(math.pi / 4, 1e-10)); // atan2(1, 1) = pi/4
        expect(
          b.data[1],
          closeTo(math.pi / 6, 1e-10),
        ); // atan2(1, sqrt(3)) = pi/6
      });

      test('Hyperbolic trig (sinh, cosh, tanh)', () {
        final a = NDArray.fromList(Float64List.fromList([0.0]), [
          1,
        ], DType.float64);
        addTearDown(a.dispose);
        final resSinh = sinh(a);
        addTearDown(resSinh.dispose);
        expect(resSinh.data[0], 0.0);

        final resCosh = cosh(a);
        addTearDown(resCosh.dispose);
        expect(resCosh.data[0], 1.0);

        final resTanh = tanh(a);
        addTearDown(resTanh.dispose);
        expect(resTanh.data[0], 0.0);
      });

      test('abs (Complex magnitude)', () {
        final a = NDArray<Complex>.create([2], DType.complex128);
        addTearDown(a.dispose);
        a.data[0] = Complex(3.0, 4.0); // mag = 5.0
        a.data[1] = Complex(-5.0, 12.0); // mag = 13.0

        final b = abs(a);
        addTearDown(b.dispose);
        expect(b.dtype, DType.float64); // should return double real array
        expect(b.shape, [2]);
        expect(b.data[0], 5.0);
        expect(b.data[1], 13.0);
      });

      test('Rounding (ceil, floor, round)', () {
        final a = NDArray.fromList(Float64List.fromList([-1.5, 1.2, 2.7]), [
          3,
        ], DType.float64);
        addTearDown(a.dispose);
        final resCeil = ceil(a);
        addTearDown(resCeil.dispose);
        expect(resCeil.data, [-1.0, 2.0, 3.0]);

        final resFloor = floor(a);
        addTearDown(resFloor.dispose);
        expect(resFloor.data, [-2.0, 1.0, 2.0]);

        final resRound = round(a);
        addTearDown(resRound.dispose);
        expect(resRound.data, [-2.0, 1.0, 3.0]); // Dart round() for -1.5 is -2
      });

      test('clip', () {
        final a = NDArray.fromList(Float64List.fromList([-5.0, 1.5, 10.0]), [
          3,
        ], DType.float64);
        addTearDown(a.dispose);
        final b = clip(a, min: -1.0, max: 5.0);
        addTearDown(b.dispose);
        expect(b.data, [-1.0, 1.5, 5.0]);

        // Verify Complex clip throws UnsupportedError
        final c = NDArray<Complex>.create([2], DType.complex128);
        addTearDown(c.dispose);
        expect(() => clip(c, min: -1.0, max: 5.0), throwsUnsupportedError);
      });
    });

    group('Comparison Operator Broadcasting', () {
      test('Array and Scalar comparison', () {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        addTearDown(a.dispose);
        final mask = a > 2.0;
        addTearDown(mask.dispose);
        expect(mask.dtype, DType.boolean);
        expect(mask.shape, [2, 2]);
        expect(mask.toList(), [false, false, true, true]);
      });

      test('Compatible shapes array comparison broadcasting', () {
        final mat = NDArray.fromList(
          Float64List.fromList([1.0, 10.0, 4.0, 2.0]),
          [2, 2],
          DType.float64,
        );
        addTearDown(mat.dispose);

        final vec = NDArray.fromList(Float64List.fromList([3.0]), [
          1,
        ], DType.float64); // shape [1] broadcasts to [2, 2]
        addTearDown(vec.dispose);

        final mask = mat < vec;
        addTearDown(mask.dispose);
        expect(mask.shape, [2, 2]);
        expect(mask.toList(), [
          true,
          false,
          false,
          true,
        ]); // [1<3(true), 10<3(false), 4<3(false), 2<3(true)]
      });

      test('Complex equality and inequality exceptions', () {
        final c1 = NDArray<Complex>.create([1], DType.complex128);
        addTearDown(c1.dispose);
        c1.data[0] = Complex(1, 2);

        final c2 = NDArray<Complex>.create([1], DType.complex128);
        addTearDown(c2.dispose);
        c2.data[0] = Complex(1, 2);

        // Equality is supported for complex!
        final eqRes = c1.eq(c2);
        addTearDown(eqRes.dispose);
        expect(eqRes.data[0], true);

        // Inequalities throw UnsupportedError
        expect(() => c1 > c2, throwsUnsupportedError);
        expect(() => c1 <= 2.0, throwsUnsupportedError);
      });
    });

    group('Logical Operations', () {
      test('logical_not on ints and doubles', () {
        final a = NDArray.fromList(Float64List.fromList([0.0, 2.5, -1.1]), [
          3,
        ], DType.float64);
        addTearDown(a.dispose);
        final resNot = logical_not(a);
        addTearDown(resNot.dispose);
        expect(resNot.data, [
          1,
          0,
          0,
        ]); // 0.0 is false (so not is 1), others are true (so not is 0)
      });

      test('logical_and, logical_or, logical_xor with broadcasting', () {
        // shape [2, 1]
        final m1 = NDArray.fromList(Int32List.fromList([0, 1]), [
          2,
          1,
        ], DType.int32);
        addTearDown(m1.dispose);
        // shape [1, 2]
        final m2 = NDArray.fromList(Int32List.fromList([0, 1]), [
          1,
          2,
        ], DType.int32);
        addTearDown(m2.dispose);

        // Broadcasted shape [2, 2]
        // m1 expanded: [[0, 0], [1, 1]]
        // m2 expanded: [[0, 1], [0, 1]]

        final andRes = logical_and(m1, m2);
        addTearDown(andRes.dispose);
        expect(andRes.shape, [2, 2]);
        expect(andRes.toList(), [
          0, 0, // 0&&0, 0&&1
          0, 1, // 1&&0, 1&&1
        ]);

        final orRes = logical_or(m1, m2);
        addTearDown(orRes.dispose);
        expect(orRes.toList(), [
          0, 1, // 0||0, 0||1
          1, 1, // 1||0, 1||1
        ]);

        final xorRes = logical_xor(m1, m2);
        addTearDown(xorRes.dispose);
        expect(xorRes.toList(), [
          0, 1, // 0^0, 0^1
          1, 0, // 1^0, 1^1
        ]);
      });
    });
  });
}
