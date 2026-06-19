import 'dart:math' as math;
import 'dart:typed_data';
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Vectorized Bitwise Operations Tests', () {
    test(
      'bitwise_and, bitwise_or, bitwise_xor basic behavior (int32 and int64)',
      () {
        NDArray.scope(() {
          final a32 = NDArray.fromList([5, 12, 3], [3], DType.int32);
          final b32 = NDArray.fromList([3, 4, 5], [3], DType.int32);

          final a64 = NDArray.fromList([5, 12, 3], [3], DType.int64);
          final b64 = NDArray.fromList([3, 4, 5], [3], DType.int64);

          // bitwise_and: 5 & 3 = 1, 12 & 4 = 4, 3 & 5 = 1
          expect(bitwise_and(a32, b32).toList(), [1, 4, 1]);
          expect(bitwise_and(a64, b64).toList(), [1, 4, 1]);

          // bitwise_or: 5 | 3 = 7, 12 | 4 = 12, 3 | 5 = 7
          expect(bitwise_or(a32, b32).toList(), [7, 12, 7]);
          expect(bitwise_or(a64, b64).toList(), [7, 12, 7]);

          // bitwise_xor: 5 ^ 3 = 6, 12 ^ 4 = 8, 3 ^ 5 = 6
          expect(bitwise_xor(a32, b32).toList(), [6, 8, 6]);
          expect(bitwise_xor(a64, b64).toList(), [6, 8, 6]);
        });
      },
    );

    test('bitwise operations with uint8 and int16 dtypes', () {
      NDArray.scope(() {
        final a8 = NDArray.fromList([15, 240], [2], DType.uint8);
        final b8 = NDArray.fromList([240, 15], [2], DType.uint8);

        final a16 = NDArray.fromList([15, 240], [2], DType.int16);
        final b16 = NDArray.fromList([240, 15], [2], DType.int16);

        expect(bitwise_and(a8, b8).toList(), [0, 0]);
        expect(bitwise_and(a16, b16).toList(), [0, 0]);

        expect(bitwise_or(a8, b8).toList(), [255, 255]);
        expect(bitwise_or(a16, b16).toList(), [255, 255]);

        expect(bitwise_xor(a8, b8).toList(), [255, 255]);
        expect(bitwise_xor(a16, b16).toList(), [255, 255]);
      });
    });

    test('left_shift and right_shift basic and edge case safe shifting', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 4, 8], [4], DType.int32);
        final shift = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);

        // left_shift: 1<<1=2, 2<<2=8, 4<<3=32, 8<<4=128
        expect(left_shift(a, shift).toList(), [2, 8, 32, 128]);

        // right_shift: 2>>1=1, 8>>2=2, 32>>3=4, 128>>4=8
        final shifted = left_shift(a, shift);
        expect(right_shift(shifted, shift).toList(), [1, 2, 4, 8]);

        // Safe shifting behavior (prevents undefined C behavior)
        final badShift = NDArray.fromList([-1, 32, 100, 0], [4], DType.int32);
        expect(left_shift(a, badShift).toList(), [0, 0, 0, 8]);
        expect(right_shift(a, badShift).toList(), [0, 0, 0, 8]);
      });
    });

    test('invert (bitwise NOT) for all four integer dtypes', () {
      NDArray.scope(() {
        final a32 = NDArray.fromList([0, -1, 5], [3], DType.int32);
        final a64 = NDArray.fromList([0, -1, 5], [3], DType.int64);
        final a8 = NDArray.fromList([0, 255, 5], [3], DType.uint8);
        final a16 = NDArray.fromList([0, -1, 5], [3], DType.int16);

        expect(invert(a32).toList(), [-1, 0, -6]);
        expect(invert(a64).toList(), [-1, 0, -6]);
        expect(invert(a8).toList(), [255, 0, 250]);
        expect(invert(a16).toList(), [-1, 0, -6]);
      });
    });

    test('broadcasting with bitwise operations (contiguous & strided)', () {
      NDArray.scope(() {
        // Broadcast scalar-like array [2] (shape [1]) against [3, 4, 5] (shape [3])
        final a = NDArray.fromList([3, 4, 5], [3], DType.int32);
        final b = NDArray.fromList([2], [1], DType.int32);

        expect(bitwise_and(a, b).toList(), [2, 0, 0]); // 3&2=2, 4&2=0, 5&2=0
        expect(bitwise_or(a, b).toList(), [3, 6, 7]); // 3|2=3, 4|2=6, 5|2=7

        // Strided sliced broadcasting test
        final stridedA = a.slice([Slice(start: 0, stop: 3, step: 2)]); // [3, 5]
        expect(bitwise_or(stridedA, b).toList(), [3, 7]);
      });
    });

    test('mixed integer type upcasting/promotion', () {
      NDArray.scope(() {
        final a32 = NDArray.fromList([5, 12], [2], DType.int32);
        final b64 = NDArray.fromList([3, 4], [2], DType.int64);

        // bitwise_and should resolve to DType.int64
        final res = bitwise_and(a32, b64);
        expect(res.dtype, DType.int64);
        expect(res.toList(), [1, 4]);
      });
    });

    test('named recycler out parameter buffer reuse', () {
      NDArray.scope(() {
        final a = NDArray.fromList([5, 12], [2], DType.int32);
        final b = NDArray.fromList([3, 4], [2], DType.int32);
        final out = NDArray<int>.create([2], DType.int32);

        final res = bitwise_and(a, b, out: out);
        expect(identical(res, out), true);
        expect(out.toList(), [1, 4]);
      });
    });

    test('non-integer input throws ArgumentError', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final b = NDArray.fromList([3, 4], [2], DType.int32);

        expect(() => bitwise_and(a, b), throwsArgumentError);
        expect(() => invert(a), throwsArgumentError);
      });
    });

    test('recycler out buffer shape incompatibility throws ArgumentError', () {
      NDArray.scope(() {
        final a = NDArray.fromList([5, 12], [2], DType.int32);
        final b = NDArray.fromList([3, 4], [2], DType.int32);
        final wrongOut = NDArray<int>.create([3], DType.int32);

        expect(() => bitwise_and(a, b, out: wrongOut), throwsArgumentError);
      });
    });
  });

  group('Top-Level Comparison ufuncs with Recycling Tests', () {
    test(
      'equal() and notEqual() broadcasted comparison',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final b = NDArray.fromList([1.0, 9.9, 3.0], [3], DType.float64);

        // equal
        final eq = equal(a, b);
        expect(eq.shape, [3]);
        expect(eq.dtype, DType.boolean);
        expect(eq.toList(), [true, false, true]);

        // notEqual
        final neq = notEqual(a, b);
        expect(neq.shape, [3]);
        expect(neq.dtype, DType.boolean);
        expect(neq.toList(), [false, true, false]);
      }),
    );

    test(
      'equal() with named recycler out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList([10.0, 20.0], [2], DType.float64);
        final b = NDArray.fromList([10.0, 99.0], [2], DType.float64);
        final out = NDArray<bool>.create([2], DType.boolean);

        final res = equal(a, b, out: out);
        expect(identical(res, out), true);
        expect(out.toList(), [true, false]);
      }),
    );

    test(
      'inequalities (greater, greaterEqual, less, lessEqual) contiguous & strided',
      () => NDArray.scope(() {
        final a = NDArray.fromList([3.0, 1.0, 5.0], [3], DType.float64);
        final b = NDArray.fromList([2.0, 1.0, 9.0], [3], DType.float64);

        // greater
        final gt = greater(a, b);
        expect(gt.toList(), [true, false, false]);

        // greaterEqual
        final gte = greaterEqual(a, b);
        expect(gte.toList(), [true, true, false]);

        // less
        final lt = less(a, b);
        expect(lt.toList(), [false, false, true]);

        // lessEqual
        final lte = lessEqual(a, b);
        expect(lte.toList(), [false, true, true]);
      }),
    );

    test(
      'complex numbers inequality throws UnsupportedError',
      () => NDArray.scope(() {
        final a = NDArray<Complex>.create([2], DType.complex128);
        final b = NDArray<Complex>.create([2], DType.complex128);

        expect(() => greater(a, b), throwsUnsupportedError);
        expect(() => greaterEqual(a, b), throwsUnsupportedError);
        expect(() => less(a, b), throwsUnsupportedError);
        expect(() => lessEqual(a, b), throwsUnsupportedError);
      }),
    );

    test(
      'recycler out buffer shape incompatibility throws ArgumentError',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);
        final wrongShape = NDArray<bool>.create([3], DType.boolean);

        expect(() => equal(a, b, out: wrongShape), throwsArgumentError);
      }),
    );
  });

  group('NumPy Compatibility Universal Functions (ufuncs) tests', () {
    group('square ufunc', () {
      test('contiguous basic types', () {
        final a = NDArray.fromList(Float64List.fromList([2.0, -3.0, 4.0]), [
          3,
        ], DType.float64);
        final res = square(a);
        expect(res.toList(), [4.0, 9.0, 16.0]);
        expect(res.dtype, DType.float64);

        final aInt = NDArray.fromList(Int32List.fromList([2, -5, 10]), [
          3,
        ], DType.int32);
        final resInt = square(aInt);
        expect(resInt.toList(), [4, 25, 100]);
        expect(resInt.dtype, DType.int32);
      });

      test('complex types', () {
        final a = NDArray<Complex>.create([2], DType.complex128);
        a[[0]] = Complex(3.0, 4.0); // (3+4i)^2 = 9 - 16 + 24i = -7 + 24i
        a[[1]] = Complex(0.0, -2.0); // (0-2i)^2 = -4

        final res = square(a);
        final resList = res.toList();
        expect(resList[0].real, closeTo(-7.0, 1e-10));
        expect(resList[0].imag, closeTo(24.0, 1e-10));
        expect(resList[1].real, closeTo(-4.0, 1e-10));
        expect(resList[1].imag, closeTo(0.0, 1e-10));
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
        expect(res.toList(), [1.0, 9.0, 25.0]);
      });

      test('boolean array and fallbacks', () {
        final aBool = NDArray.fromList([true, false], [2], DType.boolean);
        final resBool = square(aBool);
        expect(resBool.toList(), [true, false]);

        // Fallback types (uint8)
        final aUint = NDArray.fromList(Uint8List.fromList([3, 5]), [
          2,
        ], DType.uint8);
        final resUint = square(aUint);
        expect(resUint.toList(), [9, 25]);
      });

      test('out recycler buffer', () {
        final a = NDArray.fromList(Float64List.fromList([2.0, 3.0]), [
          2,
        ], DType.float64);
        final outRecycler = NDArray<double>.zeros([2], DType.float64);
        final res = square(a, out: outRecycler);
        expect(identical(res, outRecycler), true);
        expect(outRecycler.toList(), [4.0, 9.0]);
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
        expect(res.toList(), [8.0, 9.0]);
      });

      test('broadcasting strided power', () {
        final x1 = NDArray.fromList(Float64List.fromList([2.0, 3.0]), [
          2,
        ], DType.float64);
        final x2 = NDArray.fromList(Float64List.fromList([3.0]), [
          1,
        ], DType.float64);
        final res = power(x1, x2);
        expect(res.toList(), [8.0, 27.0]);
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
          // -5 // 2 = -3
          // -5 // -2 = 2
          // 5 // -2 = -3
          expect(q.toList(), [2.0, -3.0, 2.0, -3.0]);
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
        // -5 % 2 = 1 (since -5 = 2 * -3 + 1)
        // -5 % -2 = -1 (since -5 = -2 * 2 - 1)
        // 5 % -2 = -1 (since 5 = -2 * -3 - 1)
        expect(r.toList(), [1.0, 1.0, -1.0, -1.0]);
        expect(m.toList(), [1.0, 1.0, -1.0, -1.0]);
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

        expect(q.toList(), [-3, 2]);
        expect(r.toList(), [-1, -1]);
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
        expect(nanMask.toList(), [false, true, false, false]);

        final infMask = isinf(a);
        expect(infMask.toList(), [false, false, true, true]);

        final finMask = isfinite(a);
        expect(finMask.toList(), [true, false, false, false]);
      });

      test('complex types', () {
        final a = NDArray<Complex>.create([3], DType.complex128);
        a[[0]] = Complex(double.nan, 1.0);
        a[[1]] = Complex(1.0, double.infinity);
        a[[2]] = Complex(1.0, 2.0);

        final nanMask = isnan(a);
        expect(nanMask.toList(), [true, false, false]);

        final infMask = isinf(a);
        expect(infMask.toList(), [false, true, false]);

        final finMask = isfinite(a);
        expect(finMask.toList(), [false, false, true]);
      });

      test('integer fallbacks always clean', () {
        final a = NDArray.fromList(Int32List.fromList([1, 2]), [
          2,
        ], DType.int32);
        expect(isnan(a).toList().every((x) => !x), true);
        expect(isinf(a).toList().every((x) => !x), true);
        expect(isfinite(a).toList().every((x) => x), true);
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
        final resList = res.toList();
        expect(resList[0], -1.0);
        expect(resList[1], 1.0);
        // sign of -0.0 should be negative!
        expect(resList[2].isNegative, true);
        expect(resList[2].abs(), 2.0);
      });

      test('broadcasting copysign', () {
        final x1 = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          2,
        ], DType.float64);
        final x2 = NDArray.fromList(Float64List.fromList([-5.0]), [
          1,
        ], DType.float64);

        final res = copysign(x1, x2);
        expect(res.toList(), [-1.0, -2.0]);
      });
    });
  });

  group('Universal Functions (ufuncs) & Broadcasting Tests', () {
    group('Mixed DType Arithmetic Upcasting', () {
      test(
        'Add Complex and Float64 arrays',
        () => NDArray.scope(() {
          final a = NDArray<Complex>.create([2], DType.complex128);
          a[[0]] = Complex(1.0, 2.0);
          a[[1]] = Complex(3.0, 4.0);

          final b = NDArray.fromList(Float64List.fromList([10.0, 20.0]), [
            2,
          ], DType.float64);

          // Should not crash with type-cast error, should upcast b to Complex
          final c = add(a, b);
          expect(c.dtype, DType.complex128);
          expect(c.shape, [2]);
          expect(c.toList(), [Complex(11.0, 2.0), Complex(23.0, 4.0)]);
        }),
      );

      test(
        'Subtract Float64 and Complex arrays',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([10.0, 20.0]), [
            2,
          ], DType.float64);

          final b = NDArray<Complex>.create([2], DType.complex128);
          b[[0]] = Complex(1.0, 2.0);
          b[[1]] = Complex(3.0, 4.0);

          final c = subtract(a, b);
          expect(c.dtype, DType.complex128);
          expect(c.toList(), [Complex(9.0, -2.0), Complex(17.0, -4.0)]);
        }),
      );

      test(
        'Multiply Int32 and Complex arrays',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([2, 3]), [
            2,
          ], DType.int32);

          final b = NDArray<Complex>.create([2], DType.complex128);
          b[[0]] = Complex(4.0, 5.0);
          b[[1]] = Complex(1.0, -2.0);

          final c = multiply(a, b);
          expect(c.dtype, DType.complex128);
          expect(c.toList(), [Complex(8.0, 10.0), Complex(3.0, -6.0)]);
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
          expect(c.toList(), [2.5, 2.5]);
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
          final bList = b.toList();
          expect(bList[0], closeTo(0.0, 1e-10));
          expect(bList[1], closeTo(1.0, 1e-10));

          final c = NDArray<Complex>.create([1], DType.complex128);
          c[[0]] = Complex(0.0, 0.0);
          final resC = tan(c);
          expect(resC.dtype, DType.complex128);
          expect(resC.toList(), [Complex(0.0, 0.0)]);
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
          final bList = b.toList();
          expect(bList[0], closeTo(math.pi / 4, 1e-10)); // atan2(1, 1) = pi/4
          expect(
            bList[1],
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
          expect(resSinh.toList(), [0.0]);

          final resCosh = cosh(a);
          expect(resCosh.toList(), [1.0]);

          final resTanh = tanh(a);
          expect(resTanh.toList(), [0.0]);
        }),
      );

      test(
        'abs (Complex magnitude)',
        () => NDArray.scope(() {
          final a = NDArray<Complex>.create([2], DType.complex128);
          a[[0]] = Complex(3.0, 4.0); // mag = 5.0
          a[[1]] = Complex(-5.0, 12.0); // mag = 13.0

          final b = abs(a);
          expect(b.dtype, DType.float64); // should return double real array
          expect(b.shape, [2]);
          expect(b.toList(), [5.0, 13.0]);
        }),
      );

      test(
        'Rounding (ceil, floor, round)',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([-1.5, 1.2, 2.7]), [
            3,
          ], DType.float64);
          final resCeil = ceil(a);
          expect(resCeil.toList(), [-1.0, 2.0, 3.0]);

          final resFloor = floor(a);
          expect(resFloor.toList(), [-2.0, 1.0, 2.0]);

          final resRound = round(a);
          expect(resRound.toList(), [
            -2.0,
            1.0,
            3.0,
          ]); // Dart round() for -1.5 is -2
        }),
      );

      test(
        'clip with scalar bounds',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([-5.0, 1.5, 10.0]), [
            3,
          ], DType.float64);
          final b = clip(a, min: -1.0, max: 5.0);
          expect(b.toList(), [-1.0, 1.5, 5.0]);

          // Verify Complex clip throws UnsupportedError
          final c = NDArray<Complex>.create([2], DType.complex128);
          expect(() => clip(c, min: -1.0, max: 5.0), throwsUnsupportedError);
        }),
      );

      test(
        'clip with array bounds (exact matching shape)',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([1.0, 5.0, 10.0]), [
            3,
          ], DType.float64);
          final minBounds = NDArray.fromList(
            Float64List.fromList([2.0, 2.0, 2.0]),
            [3],
            DType.float64,
          );
          final maxBounds = NDArray.fromList(
            Float64List.fromList([8.0, 8.0, 8.0]),
            [3],
            DType.float64,
          );

          final b = clipArray(a, min: minBounds, max: maxBounds);
          expect(b.toList(), [2.0, 5.0, 8.0]);
        }),
      );

      test(
        'clip with broadcasted array bounds',
        () => NDArray.scope(() {
          // Input is 2x3 matrix
          final a = NDArray.fromList(
            Float64List.fromList([
              1.0, 2.0, 3.0, // row 0
              4.0, 5.0, 6.0, // row 1
            ]),
            [2, 3],
            DType.float64,
          );

          // min bounds is a 1D row vector [2.5, 2.5, 2.5] (shape [3]) -> broadcasts to [2, 3]
          final minBounds = NDArray.fromList(
            Float64List.fromList([2.5, 2.5, 2.5]),
            [3],
            DType.float64,
          );

          // max bounds is a 2D column vector [[4.5], [4.5]] (shape [2, 1]) -> broadcasts to [2, 3]
          final maxBounds = NDArray.fromList(Float64List.fromList([4.5, 4.5]), [
            2,
            1,
          ], DType.float64);

          // Clip bounds for row 0: min=2.5, max=4.5
          // row 0 values: [1.0, 2.0, 3.0] -> clipped: [2.5, 2.5, 3.0]
          // Clip bounds for row 1: min=2.5, max=4.5
          // row 1 values: [4.0, 5.0, 6.0] -> clipped: [4.0, 4.5, 4.5]
          final b = clipArray(a, min: minBounds, max: maxBounds);
          expect(b.shape, [2, 3]);
          expect(b.toList(), [2.5, 2.5, 3.0, 4.0, 4.5, 4.5]);
        }),
      );

      test(
        'clip with mixed scalar and array bounds on integer DType',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Int32List.fromList([1, 5, 10]), [
            3,
          ], DType.int32);

          // min is scalar, max is array
          final maxBounds = NDArray.fromList(Int32List.fromList([3, 7, 12]), [
            3,
          ], DType.int32);
          final minBounds = NDArray.fromList(Int32List.fromList([2]), [
            1,
          ], DType.int32);

          final b = clipArray(a, min: minBounds, max: maxBounds);
          expect(b.dtype, DType.int32);
          expect(b.toList(), [2, 5, 10]);
        }),
      );

      test(
        'clip throws ArgumentError on incompatible shape broadcasting',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
            3,
          ], DType.float64);
          final incompatible = NDArray.fromList(
            Float64List.fromList([1.0, 2.0]),
            [2],
            DType.float64,
          );
          final maxBounds = NDArray.fromList(Float64List.fromList([5.0]), [
            1,
          ], DType.float64);

          expect(
            () => clipArray(a, min: incompatible, max: maxBounds),
            throwsArgumentError,
          );
        }),
      );
      test(
        'clip with one-sided (nullable) scalar bounds',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([-5.0, 1.5, 10.0]), [
            3,
          ], DType.float64);

          // Clip with min only
          final b1 = clip(a, min: -1.0);
          expect(b1.toList(), [-1.0, 1.5, 10.0]);

          // Clip with max only
          final b2 = clip(a, max: 5.0);
          expect(b2.toList(), [-5.0, 1.5, 5.0]);

          // Clip with neither (should return exact copy)
          final b3 = clip(a);
          expect(b3.toList(), [-5.0, 1.5, 10.0]);
        }),
      );

      test(
        'clipArray with one-sided (nullable) array bounds',
        () => NDArray.scope(() {
          final a = NDArray.fromList(Float64List.fromList([1.0, 5.0, 10.0]), [
            3,
          ], DType.float64);
          final minBounds = NDArray.fromList(
            Float64List.fromList([2.0, 2.0, 2.0]),
            [3],
            DType.float64,
          );
          final maxBounds = NDArray.fromList(
            Float64List.fromList([8.0, 8.0, 8.0]),
            [3],
            DType.float64,
          );

          // Clip with min only
          final b1 = clipArray(a, min: minBounds);
          expect(b1.toList(), [2.0, 5.0, 10.0]);

          // Clip with max only
          final b2 = clipArray(a, max: maxBounds);
          expect(b2.toList(), [1.0, 5.0, 8.0]);

          // Clip with neither (should return exact copy)
          final b3 = clipArray(a);
          expect(b3.toList(), [1.0, 5.0, 10.0]);
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
          c1[[0]] = Complex(1, 2);

          final c2 = NDArray<Complex>.create([1], DType.complex128);
          c2[[0]] = Complex(1, 2);

          // Equality is supported for complex!
          final eqRes = c1.eq(c2);
          expect(eqRes.toList(), [true]);

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
          expect(resNot.dtype, DType.boolean);
          expect(
            resNot.toList(),
            [true, false, false],
          ); // 0.0 is false (so not is true), others are true (so not is false)
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
          expect(andRes.dtype, DType.boolean);
          expect(andRes.shape, [2, 2]);
          expect(andRes.toList(), [
            false, false, // false&&false, false&&true
            false, true, // true&&false, true&&true
          ]);

          final orRes = logical_or(m1, m2);
          expect(orRes.dtype, DType.boolean);
          expect(orRes.toList(), [
            false, true, // false||false, false||true
            true, true, // true||false, true||true
          ]);

          final xorRes = logical_xor(m1, m2);
          expect(xorRes.dtype, DType.boolean);
          expect(xorRes.toList(), [
            false, true, // false^false, false^true
            true, false, // true^false, true^true
          ]);
        }),
      );
    });
  });

  group('NDArray Logical Reductions (all, any) Tests', () {
    test(
      'all() basic global reduction checks',
      () => NDArray.scope(() {
        final a = NDArray.fromList([true, true, true], [3], DType.boolean);
        final b = NDArray.fromList([true, false, true], [3], DType.boolean);
        final c = NDArray.fromList([1, 2, 3], [3], DType.int32);
        final d = NDArray.fromList([1, 0, 3], [3], DType.int32);

        expect(all(a).scalar, true);
        expect(all(b).scalar, false);
        expect(all(c).scalar, true);
        expect(all(d).scalar, false);
      }),
    );

    test(
      'any() basic global reduction checks',
      () => NDArray.scope(() {
        final a = NDArray.fromList([false, false, false], [3], DType.boolean);
        final b = NDArray.fromList([false, true, false], [3], DType.boolean);
        final c = NDArray.fromList([0, 0, 0], [3], DType.int32);
        final d = NDArray.fromList([0, 5, 0], [3], DType.int32);

        expect(any(a).scalar, false);
        expect(any(b).scalar, true);
        expect(any(c).scalar, false);
        expect(any(d).scalar, true);
      }),
    );

    test(
      'all() axis reductions on 2D matrices',
      () => NDArray.scope(() {
        final mat = NDArray.fromList(
          [true, true, false, true, false, false],
          [2, 3],
          DType.boolean,
        );

        // all along rows (axis: 0) -> shape [3]
        final res0 = all(mat, axis: 0);
        expect(res0.shape, [3]);
        expect(res0.toList(), [true, false, false]);

        // all along columns (axis: 1) -> shape [2]
        final res1 = all(mat, axis: 1);
        expect(res1.shape, [2]);
        expect(res1.toList(), [false, false]);

        // negative axis support
        final resNeg = all(mat, axis: -1);
        expect(resNeg.shape, [2]);
        expect(resNeg.toList(), [false, false]);
      }),
    );

    test(
      'any() axis reductions on 2D matrices',
      () => NDArray.scope(() {
        final mat = NDArray.fromList(
          [true, false, false, false, false, false],
          [2, 3],
          DType.boolean,
        );

        // any along rows (axis: 0) -> shape [3]
        final res0 = any(mat, axis: 0);
        expect(res0.shape, [3]);
        expect(res0.toList(), [true, false, false]);

        // any along columns (axis: 1) -> shape [2]
        final res1 = any(mat, axis: 1);
        expect(res1.shape, [2]);
        expect(res1.toList(), [true, false]);
      }),
    );

    test(
      'logical_and, logical_or, and logical_xor successfully combine boolean mask arrays',
      () => NDArray.scope(() {
        final mask1 = NDArray.fromList([true, false, true], [3], DType.boolean);
        final mask2 = NDArray.fromList([true, true, false], [3], DType.boolean);

        final resAnd = logical_and(mask1, mask2);
        expect(resAnd.dtype, DType.boolean);
        expect(resAnd.toList(), [true, false, false]);

        final resOr = logical_or(mask1, mask2);
        expect(resOr.dtype, DType.boolean);
        expect(resOr.toList(), [true, true, true]);

        final resXor = logical_xor(mask1, mask2);
        expect(resXor.dtype, DType.boolean);
        expect(resXor.toList(), [false, true, true]);
      }),
    );

    test(
      'logical_and cross-type promotions on float/boolean, complex/boolean, and int/boolean',
      () => NDArray.scope(() {
        final mask = NDArray.fromList([true, true, false], [3], DType.boolean);

        // Float64 / boolean
        final f64 = NDArray.fromList([0.0, 2.5, 0.0], [3], DType.float64);
        final resF1 = logical_and(f64, mask);
        expect(resF1.dtype, DType.boolean);
        expect(resF1.toList(), [false, true, false]);
        final resF2 = logical_and(mask, f64);
        expect(resF2.dtype, DType.boolean);
        expect(resF2.toList(), [false, true, false]);

        // Complex128 / boolean
        final c128 = NDArray<Complex>.create([3], DType.complex128);
        c128[[0]] = Complex(0.0, 0.0); // false
        c128[[1]] = Complex(1.0, -1.0); // true
        c128[[2]] = Complex(0.0, 0.0);
        final resC1 = logical_and(c128, mask);
        expect(resC1.dtype, DType.boolean);
        expect(resC1.toList(), [false, true, false]);
        final resC2 = logical_and(mask, c128);
        expect(resC2.dtype, DType.boolean);
        expect(resC2.toList(), [false, true, false]);

        // Int32 / boolean
        final i32 = NDArray.fromList([0, 5, 0], [3], DType.int32);
        final resI1 = logical_and(i32, mask);
        expect(resI1.dtype, DType.boolean);
        expect(resI1.toList(), [false, true, false]);
        final resI2 = logical_and(mask, i32);
        expect(resI2.dtype, DType.boolean);
        expect(resI2.toList(), [false, true, false]);
      }),
    );

    test(
      'Disposed array checks throw StateError',
      () => NDArray.scope(() {
        final a = NDArray.fromList([true, false], [2], DType.boolean);
        a.dispose();

        expect(() => all(a), throwsStateError);
        expect(() => any(a), throwsStateError);
      }),
    );

    test(
      'Axis out of bounds throws ArgumentError',
      () => NDArray.scope(() {
        final a = NDArray.fromList([true, false], [2], DType.boolean);
        expect(() => all(a, axis: 5), throwsArgumentError);
        expect(() => any(a, axis: -5), throwsArgumentError);
      }),
    );
  });
}
