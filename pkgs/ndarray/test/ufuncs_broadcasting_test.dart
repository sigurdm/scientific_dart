import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';
import 'dart:math' as math;
import 'dart:typed_data';

void main() {
  group('Universal Functions (ufuncs) & Broadcasting Tests', () {
    group('Mixed DType Arithmetic Upcasting', () {
      test(
        'Add Complex and Float64 arrays',
        () => NDArray.scope(() {
          final a = NDArray<Complex>.create([2], DType.complex128);
          a.data[0] = Complex(1.0, 2.0);
          a.data[1] = Complex(3.0, 4.0);

          final b = NDArray.fromList(Float64List.fromList([10.0, 20.0]), [
            2,
          ], DType.float64);

          // Should not crash with type-cast error, should upcast b to Complex
          final c = add(a, b);
          expect(c.dtype, DType.complex128);
          expect(c.shape, [2]);
          expect(c.data[0], Complex(11.0, 2.0));
          expect(c.data[1], Complex(23.0, 4.0));
        }),
      );

      test(
        'Subtract Float64 and Complex arrays',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([10.0, 20.0]), [
            2,
          ], DType.float64);

          final b = NDArray<Complex>.create([2], DType.complex128);
          b.data[0] = Complex(1.0, 2.0);
          b.data[1] = Complex(3.0, 4.0);

          final c = subtract(a, b);
          expect(c.dtype, DType.complex128);
          expect(c.data[0], Complex(9.0, -2.0));
          expect(c.data[1], Complex(17.0, -4.0));
        }),
      );

      test(
        'Multiply Int32 and Complex arrays',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([2, 3]), [
            2,
          ], DType.int32);

          final b = NDArray<Complex>.create([2], DType.complex128);
          b.data[0] = Complex(4.0, 5.0);
          b.data[1] = Complex(1.0, -2.0);

          final c = multiply(a, b);
          expect(c.dtype, DType.complex128);
          expect(c.data[0], Complex(8.0, 10.0));
          expect(c.data[1], Complex(3.0, -6.0));
        }),
      );

      test(
        'True Division always returns floats or complex',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([5, 10]), [
            2,
          ], DType.int32);
          final b = NDArray.fromList(Int32List.fromList([2, 4]), [
            2,
          ], DType.int32);

          // int / int -> float64
          final c = divide(a, b);
          expect(c.dtype, DType.float64);
          expect(c.data[0], 2.5);
          expect(c.data[1], 2.5);
        }),
      );
    });

    group('New Math Ufuncs', () {
      test(
        'tan',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([0.0, math.pi / 4]), [
            2,
          ], DType.float64);
          final b = tan(a);
          expect(b.shape, [2]);
          expect(b.data[0], closeTo(0.0, 1e-10));
          expect(b.data[1], closeTo(1.0, 1e-10));

          final c = NDArray<Complex>.create([1], DType.complex128);
          c.data[0] = Complex(0.0, 0.0);
          final resC = tan(c);
          expect(resC.dtype, DType.complex128);
          expect(resC.data[0], Complex(0.0, 0.0));
        }),
      );

      test(
        'atan2 with broadcasting',
        () => NDArray.scope(() {
          final y = NDArray.fromList(Float64List.fromList([1.0]), [
            1,
          ], DType.float64);
          final x = NDArray.fromList(
            Float64List.fromList([1.0, math.sqrt(3)]),
            [2],
            DType.float64,
          );

          final b = atan2(y, x); // y broadcasts to [2]
          expect(b.shape, [2]);
          expect(b.data[0], closeTo(math.pi / 4, 1e-10)); // atan2(1, 1) = pi/4
          expect(
            b.data[1],
            closeTo(math.pi / 6, 1e-10),
          ); // atan2(1, sqrt(3)) = pi/6
        }),
      );

      test(
        'Hyperbolic trig (sinh, cosh, tanh)',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([0.0]), [
            1,
          ], DType.float64);
          final resSinh = sinh(a);
          expect(resSinh.data[0], 0.0);

          final resCosh = cosh(a);
          expect(resCosh.data[0], 1.0);

          final resTanh = tanh(a);
          expect(resTanh.data[0], 0.0);
        }),
      );

      test(
        'abs (Complex magnitude)',
        () => NDArray.scope(() {
          final a = NDArray<Complex>.create([2], DType.complex128);
          a.data[0] = Complex(3.0, 4.0); // mag = 5.0
          a.data[1] = Complex(-5.0, 12.0); // mag = 13.0

          final b = abs(a);
          expect(b.dtype, DType.float64); // should return double real array
          expect(b.shape, [2]);
          expect(b.data[0], 5.0);
          expect(b.data[1], 13.0);
        }),
      );

      test(
        'Rounding (ceil, floor, round)',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([-1.5, 1.2, 2.7]), [
            3,
          ], DType.float64);
          final resCeil = ceil(a);
          expect(resCeil.data, [-1.0, 2.0, 3.0]);

          final resFloor = floor(a);
          expect(resFloor.data, [-2.0, 1.0, 2.0]);

          final resRound = round(a);
          expect(resRound.data, [
            -2.0,
            1.0,
            3.0,
          ]); // Dart round() for -1.5 is -2
        }),
      );

      test(
        'clip',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([-5.0, 1.5, 10.0]), [
            3,
          ], DType.float64);
          final b = clip(a, min: -1.0, max: 5.0);
          expect(b.data, [-1.0, 1.5, 5.0]);

          // Verify Complex clip throws UnsupportedError
          final c = NDArray<Complex>.create([2], DType.complex128);
          expect(() => clip(c, min: -1.0, max: 5.0), throwsUnsupportedError);
        }),
      );
    });

    group('Comparison Operator Broadcasting', () {
      test(
        'Array and Scalar comparison',
        () => NDArray.scope(() {
          final a = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float64,
          );
          final mask = a > 2.0;
          expect(mask.dtype, DType.boolean);
          expect(mask.shape, [2, 2]);
          expect(mask.toList(), [false, false, true, true]);
        }),
      );

      test(
        'Compatible shapes array comparison broadcasting',
        () => NDArray.scope(() {
          final mat = NDArray.fromList(
            Float64List.fromList([1.0, 10.0, 4.0, 2.0]),
            [2, 2],
            DType.float64,
          );

          final vec = NDArray.fromList(Float64List.fromList([3.0]), [
            1,
          ], DType.float64); // shape [1] broadcasts to [2, 2]

          final mask = mat < vec;
          expect(mask.shape, [2, 2]);
          expect(mask.toList(), [
            true,
            false,
            false,
            true,
          ]); // [1<3(true), 10<3(false), 4<3(false), 2<3(true)]
        }),
      );

      test(
        'Complex equality and inequality exceptions',
        () => NDArray.scope(() {
          final c1 = NDArray<Complex>.create([1], DType.complex128);
          c1.data[0] = Complex(1, 2);

          final c2 = NDArray<Complex>.create([1], DType.complex128);
          c2.data[0] = Complex(1, 2);

          // Equality is supported for complex!
          final eqRes = c1.eq(c2);
          expect(eqRes.data[0], true);

          // Inequalities throw UnsupportedError
          expect(() => c1 > c2, throwsUnsupportedError);
          expect(() => c1 <= 2.0, throwsUnsupportedError);
        }),
      );
    });

    group('Logical Operations', () {
      test(
        'logical_not on ints and doubles',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([0.0, 2.5, -1.1]), [
            3,
          ], DType.float64);
          final resNot = logical_not(a);
          expect(resNot.data, [
            1,
            0,
            0,
          ]); // 0.0 is false (so not is 1), others are true (so not is 0)
        }),
      );

      test(
        'logical_and, logical_or, logical_xor with broadcasting',
        () => NDArray.scope(() {
          // shape [2, 1]
          final m1 = NDArray.fromList(Int32List.fromList([0, 1]), [
            2,
            1,
          ], DType.int32);
          // shape [1, 2]
          final m2 = NDArray.fromList(Int32List.fromList([0, 1]), [
            1,
            2,
          ], DType.int32);

          // Broadcasted shape [2, 2]
          // m1 expanded: [[0, 0], [1, 1]]
          // m2 expanded: [[0, 1], [0, 1]]

          final andRes = logical_and(m1, m2);
          expect(andRes.shape, [2, 2]);
          expect(andRes.toList(), [
            0, 0, // 0&&0, 0&&1
            0, 1, // 1&&0, 1&&1
          ]);

          final orRes = logical_or(m1, m2);
          expect(orRes.toList(), [
            0, 1, // 0||0, 0||1
            1, 1, // 1||0, 1||1
          ]);

          final xorRes = logical_xor(m1, m2);
          expect(xorRes.toList(), [
            0, 1, // 0^0, 0^1
            1, 0, // 1^0, 1^1
          ]);
        }),
      );
    });
  });
}
