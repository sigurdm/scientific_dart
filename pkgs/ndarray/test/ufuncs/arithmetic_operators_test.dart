import 'dart:typed_data';
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('NDArray Arithmetic Extension Methods Tests', () {
    test(
      'Generic add, subtract, multiply, divide on contiguous double arrays',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            [2.0, 2.0, 2.0, 2.0],
            [2, 2],
            DType.float64,
          );

          final addRes = add(a, b);
          expect(addRes.dtype, DType.float64);
          expect(addRes.toList(), [3.0, 4.0, 5.0, 6.0]);

          final subRes = subtract(a, b);
          expect(subRes.toList(), [-1.0, 0.0, 1.0, 2.0]);

          final mulRes = multiply(a, b);
          expect(mulRes.toList(), [2.0, 4.0, 6.0, 8.0]);

          final divRes = divide(a, b);
          expect(divRes.toList(), [0.5, 1.0, 1.5, 2.0]);
        });
      },
    );

    test('Contiguous Float32 SIMD lanes and C vector sweeps', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float32);
        final b = NDArray.fromList([1.0, 1.0, 1.0, 1.0], [4], DType.float32);

        final addRes = add(a, b);
        expect(addRes.dtype, DType.float32);
        expect(addRes.toList(), [2.0, 3.0, 4.0, 5.0]);
      });
    });

    test('Multidimensional strided broadcasting with FFI', () {
      NDArray.scope(() {
        // Shape [2, 2]
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        // Shape [2, 1] (broadcast column)
        final b = NDArray.fromList([10.0, 20.0], [2, 1], DType.float64);

        final res = add(a, b);
        expect(res.shape, [2, 2]);
        expect(res.toList(), [11.0, 12.0, 23.0, 24.0]);
      });
    });

    test('Complex128 and Float64 upcasting promotions', () {
      NDArray.scope(() {
        final a = NDArray.fromList(
          [Complex(1.0, 2.0), Complex(3.0, 4.0)],
          [2],
          DType.complex128,
        );
        final b = NDArray.fromList([10.0, 20.0], [2], DType.float64);

        // Explicit generic call to allow promotion
        final res = add<Complex, double, Complex>(a, b);
        expect(res.dtype, DType.complex128);
        final resList = res.toList();
        expect(resList[0].real, 11.0);
        expect(resList[0].imag, 2.0);
        expect(resList[1].real, 23.0);
        expect(resList[1].imag, 4.0);
      });
    });

    test('out parameter validation and success/error paths', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final b = NDArray.fromList([2.0, 2.0], [2], DType.float64);
        final outValid = NDArray<Float64>.create([2], DType.float64);

        final res = add<double, double, double>(a, b, out: outValid);
        expect(identical(res, outValid), true);
        expect(outValid.toList(), [3.0, 4.0]);

        final outInvalidShape = NDArray<Float64>.create([3], DType.float64);
        expect(
          () => add<double, double, double>(a, b, out: outInvalidShape),
          throwsArgumentError,
        );

        final outInvalidDType = NDArray<Int32>.create([2], DType.int32);
        expect(
          () => add<double, double, int>(a, b, out: outInvalidDType),
          throwsArgumentError,
        );
      });
    });
  });

  group('NDArray Operator Overloading Tests', () {
    test(
      'Basic Arithmetic Operators',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float64,
        );

        final sum = a + b;
        expect(sum.toList(), [11.0, 22.0, 33.0, 44.0]);

        final diff = b - a;
        expect(diff.toList(), [9.0, 18.0, 27.0, 36.0]);

        final prod = a * b;
        expect(prod.toList(), [10.0, 40.0, 90.0, 160.0]);

        final quot = b / a;
        expect(quot.toList(), [10.0, 10.0, 10.0, 10.0]);
      }),
    );

    test(
      'Scalar Broadcasting Operators',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);

        final b = a + 10.0;
        expect(b.toList(), [11.0, 12.0]);

        final c = a * 2;
        expect(c.toList(), [2.0, 4.0]);

        final d = a - 1;
        expect(d.toList(), [0.0, 1.0]);

        final e = a / 0.5;
        expect(e.toList(), [2.0, 4.0]);
      }),
    );

    test(
      'Unary Negation',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, -2.0, 0.0], [3], DType.float64);
        final b = -a;
        expect(b.toList(), [-1.0, 2.0, 0.0]);
      }),
    );

    test(
      'Integer Division and Remainder',
      () => NDArray.scope(() {
        final a = NDArray.fromList([7, 10], [2], DType.int32);
        final b = NDArray.fromList([3, 4], [2], DType.int32);

        final floorDiv = a ~/ b;
        expect(floorDiv.toList(), [2, 2]);

        final rem = a % b;
        expect(rem.toList(), [1, 2]);

        final floorDivScalar = a ~/ 2;
        expect(floorDivScalar.toList(), [3, 5]);
      }),
    );

    test(
      'Bitwise Operators',
      () => NDArray.scope(() {
        final a = NDArray.fromList([0x0F, 0xF0], [2], DType.int32);
        final b = NDArray.fromList([0xFF, 0x00], [2], DType.int32);

        expect((a & b).toList(), [0x0F, 0x00]);
        expect((a | b).toList(), [0xFF, 0xF0]);
        expect((a ^ b).toList(), [0xF0, 0xF0]);
        expect((~a).toList().map((x) => x & 0xFF).toList(), [0xF0, 0x0F]);

        final c = NDArray.fromList([1, 2], [2], DType.int32);
        expect((c << 1).toList(), [2, 4]);
        expect((c >> 1).toList(), [0, 1]);
      }),
    );

    test(
      'Complex Arithmetic',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [Complex(1, 2), Complex(3, 4)],
          [2],
          DType.complex128,
        );
        final b = NDArray.fromList(
          [Complex(10, 20), Complex(30, 40)],
          [2],
          DType.complex128,
        );

        final sum = a + b;
        final sumList = sum.toList();
        expect(sumList[0].real, 11.0);
        expect(sumList[0].imag, 22.0);

        final neg = -a;
        final negList = neg.toList();
        expect(negList[0].real, -1.0);
        expect(negList[0].imag, -2.0);

        final scalarSum = a + Complex(10, 10);
        final scalarSumList = scalarSum.toList();
        expect(scalarSumList[0].real, 11.0);
        expect(scalarSumList[0].imag, 12.0);
      }),
    );

    test(
      'Mixed Type Arithmetic',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 2], [2], DType.int32);
        final b = NDArray.fromList([0.5, 1.5], [2], DType.float64);

        final c = a + b;
        expect(c.dtype, DType.float64);
        expect(c.toList(), [1.5, 3.5]);
      }),
    );

    test(
      'Arithmetic Error Handling (Division by Zero & Overflow)',
      () => NDArray.scope(() {
        // 1. True Division (operator /) by zero
        final doubleArr = NDArray.fromList(
          [5.0, -5.0, 0.0],
          [3],
          DType.float64,
        );
        final zeroDoubleArr = NDArray.zeros([3], DType.float64);
        final divDouble = doubleArr / zeroDoubleArr;
        final divDoubleList = divDouble.toList();
        expect(divDoubleList[0], double.infinity);
        expect(divDoubleList[1], double.negativeInfinity);
        expect(divDoubleList[2].isNaN, true);

        // True Division on integer arrays upcasts to float64 and handles division by zero identically
        final intArr = NDArray.fromList([5, -5, 0], [3], DType.int32);
        final zeroIntArr = NDArray.zeros([3], DType.int32);
        final divInt = intArr / zeroIntArr;
        expect(divInt.dtype, DType.float64);
        final divIntList = divInt.toList();
        expect(divIntList[0], double.infinity);
        expect(divIntList[1], double.negativeInfinity);
        expect(divIntList[2].isNaN, true);

        // 2. Floor Division (operator ~/) and Remainder (operator %) by zero on integer throws UnsupportedError
        expect(() => intArr ~/ zeroIntArr, throwsA(isA<UnsupportedError>()));
        expect(() => intArr % zeroIntArr, throwsA(isA<UnsupportedError>()));

        final minIntArr = NDArray.fromList([-2147483648], [1], DType.int32);
        final minusOneArr = NDArray.fromList([-1], [1], DType.int32);
        expect(
          (minIntArr ~/ minusOneArr).toList()[0],
          -2147483648,
        ); // or wraps/throws? Let's see!

        // Floor Division on floats with zero divisor returns NaN
        final divFloorDouble = doubleArr ~/ zeroDoubleArr;
        expect(divFloorDouble.toList().every((x) => x.isNaN), true);

        // 3. Integer Multiplication Overflow wraps around (two's complement)
        // Max 32-bit signed int is 2147483647
        final overflowArr32 = NDArray.fromList([2147483647], [1], DType.int32);
        final scaleArr32 = NDArray.fromList([2], [1], DType.int32);
        final prod32 = overflowArr32 * scaleArr32;
        expect(prod32.dtype, DType.int32);
        expect(
          prod32.toList()[0],
          -2,
        ); // 2147483647 * 2 = 4294967294 -> wraps to -2

        // Max 64-bit signed int is 9223372036854775807
        final overflowArr64 = NDArray.fromList(
          [9223372036854775807],
          [1],
          DType.int64,
        );
        final scaleArr64 = NDArray.fromList([2], [1], DType.int64);
        final prod64 = overflowArr64 * scaleArr64;
        expect(prod64.dtype, DType.int64);
        expect(
          prod64.toList()[0],
          -2,
        ); // 9223372036854775807 * 2 wraps to -2 in 64-bit two's complement

        // 4. Float Multiplication Overflow yields infinity
        final overflowArrFloat = NDArray.fromList([1e308], [1], DType.float64);
        final prodFloat = overflowArrFloat * 10.0;
        expect(prodFloat.toList()[0], double.infinity);
      }),
    );
  });

  group('Type Safety Fixes (Phase 3)', () {
    test('floor_divide and remainder with uint8 and int16', () {
      NDArray.scope(() {
        final a = NDArray.fromList([10, 20, 30, 40], [4], DType.uint8);
        final b = NDArray.fromList([3, 3, 3, 3], [4], DType.uint8);

        // floor_divide
        final resDiv = floor_divide(a, b);
        expect(resDiv.dtype, DType.uint8);
        expect(resDiv.toList(), [3, 6, 10, 13]);

        // remainder
        final resRem = remainder(a, b);
        expect(resRem.dtype, DType.uint8);
        expect(resRem.toList(), [1, 2, 0, 1]);

        // int16
        final a16 = NDArray.fromList([10, -20, 30, -40], [4], DType.int16);
        final b16 = NDArray.fromList([3, 3, 3, 3], [4], DType.int16);

        final resDiv16 = floor_divide(a16, b16);
        expect(resDiv16.dtype, DType.int16);
        expect(resDiv16.toList(), [3, -7, 10, -14]);

        final resRem16 = remainder(a16, b16);
        expect(resRem16.dtype, DType.int16);
        expect(resRem16.toList(), [1, 1, 0, 2]);
      });
    });

    test('sin and abs with uint8 and int16', () {
      NDArray.scope(() {
        final a = NDArray.fromList([0, 30, 90], [3], DType.uint8);

        final resSin = sin(a);
        expect(resSin.dtype, DType.float64);
        expect(resSin.toList()[0], closeTo(0.0, 1e-5));

        // abs
        final a16 = NDArray.fromList([-10, 0, 20], [3], DType.int16);
        final resAbs = abs(a16);
        expect(resAbs.dtype, DType.int16);
        expect(resAbs.toList(), [10, 0, 20]);
      });
    });

    test('negative with uint8 and int16 (no silent fallthrough)', () {
      NDArray.scope(() {
        final a = NDArray.fromList([-10, 0, 20], [3], DType.int16);
        final resNeg = negative(a);
        expect(resNeg.dtype, DType.int16);
        expect(resNeg.toList(), [10, 0, -20]);

        final a8 = NDArray.fromList([10, 0, 20], [3], DType.uint8);
        final resNeg8 = negative(a8);
        expect(resNeg8.dtype, DType.uint8);
        expect(resNeg8.toList(), [246, 0, 236]);
      });
    });

    test('det type consistency for float32', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float32);
        final d = det(a);
        expect(d.dtype, DType.float32);
        expect(d.toList()[0], closeTo(-2.0, 1e-5));
      });
    });

    test('svd and qr throw ArgumentError for integer inputs', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        expect(() => svd(a), throwsArgumentError);
        expect(() => qr(a), throwsArgumentError);

        final a8 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.uint8);
        expect(() => svd(a8), throwsArgumentError);
        expect(() => qr(a8), throwsArgumentError);
      });
    });

    test('complex SVD and pinv', () {
      NDArray.scope(() {
        // complex128
        final a = NDArray.fromList(
          [Complex(1, 2), Complex(3, 4), Complex(5, 6), Complex(7, 8)],
          [2, 2],
          DType.complex128,
        );

        final svdRes = svd(a);
        expect(svdRes.U.dtype, DType.complex128);
        expect(svdRes.S.dtype, DType.float64);
        expect(svdRes.Vh.dtype, DType.complex128);

        final sDiag = diag(svdRes.S);
        final NDArray<Complex> uS = matmul(svdRes.U, sDiag);
        final reconstructed = matmul(uS, svdRes.Vh);

        expect(allClose(reconstructed, a, rtol: 1e-5, atol: 1e-5), isTrue);

        // complex64
        final a64 = NDArray.fromList(
          [Complex(1, 2), Complex(3, 4), Complex(5, 6), Complex(7, 8)],
          [2, 2],
          DType.complex64,
        );

        final svdRes64 = svd(a64);
        expect(svdRes64.U.dtype, DType.complex64);
        expect(svdRes64.S.dtype, DType.float32);
        expect(svdRes64.Vh.dtype, DType.complex64);

        final sDiag64 = diag(svdRes64.S);
        final NDArray<Complex> uS64 = matmul(svdRes64.U, sDiag64);
        final reconstructed64 = matmul(uS64, svdRes64.Vh);
        expect(
          allClose(reconstructed64, a64, rtol: 1e-3, atol: 1e-3),
          isTrue,
        ); // Lower tolerance for float32

        // complex pinv
        final aPinv = pinv(a);
        expect(aPinv.dtype, DType.complex128);

        final aPinvA = matmul(a, aPinv);
        final aPinvAA = matmul(aPinvA, a);
        expect(allClose(aPinvAA, a, rtol: 1e-5, atol: 1e-5), isTrue);

        final aPinv64 = pinv(a64);
        expect(aPinv64.dtype, DType.complex64);
      });
    });
  });

  group('Out Parameter Tests', () {
    test(
      'det with out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final outBuffer = NDArray<Float64>.zeros([], DType.float64);
        final d = det(a, out: outBuffer);
        expect(identical(d, outBuffer), true);
        expect(outBuffer.toList()[0], closeTo(-2.0, 1e-9));
      }),
    );

    test(
      'min with out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList([4.0, 2.0, 5.0, 1.0], [2, 2], DType.float64);
        final outBuffer = NDArray<Float64>.zeros([2], DType.float64);
        final res = min(a, axis: 1, out: outBuffer);
        expect(identical(res, outBuffer), true);
        expect(outBuffer.toList(), [2.0, 1.0]);
      }),
    );

    test(
      'max with out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList([4.0, 2.0, 5.0, 1.0], [2, 2], DType.float64);
        final outBuffer = NDArray<Float64>.zeros([2], DType.float64);
        final res = max(a, axis: 1, out: outBuffer);
        expect(identical(res, outBuffer), true);
        expect(outBuffer.toList(), [4.0, 5.0]);
      }),
    );

    test(
      'nanmin with out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [4.0, double.nan, 5.0, 1.0],
          [2, 2],
          DType.float64,
        );
        final outBuffer = NDArray<Float64>.zeros([2], DType.float64);
        final res = nanmin(a, axis: 1, out: outBuffer);
        expect(identical(res, outBuffer), true);
        expect(outBuffer.toList(), [4.0, 1.0]);
      }),
    );

    test(
      'nanmax with out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [4.0, double.nan, 5.0, 1.0],
          [2, 2],
          DType.float64,
        );
        final outBuffer = NDArray<Float64>.zeros([2], DType.float64);
        final res = nanmax(a, axis: 1, out: outBuffer);
        expect(identical(res, outBuffer), true);
        expect(outBuffer.toList(), [4.0, 5.0]);
      }),
    );

    test(
      'fft with out parameter (no transpose)',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final outBuffer = NDArray<Complex>.zeros([4], DType.complex128);
        final res = fft(a, out: outBuffer);
        expect(identical(res, outBuffer), true);
        // Expected FFT result: [10, -2+2i, -2, -2-2i]
        final outBufferList = outBuffer.toList();
        expect(outBufferList[0].real, closeTo(10.0, 1e-9));
        expect(outBufferList[0].imag, closeTo(0.0, 1e-9));
        expect(outBufferList[1].real, closeTo(-2.0, 1e-9));
        expect(outBufferList[1].imag, closeTo(2.0, 1e-9));
      }),
    );

    test(
      'fft with out parameter (with transpose)',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final outBuffer = NDArray<Complex>.zeros([2, 2], DType.complex128);
        // FFT along axis 0
        final res = fft(a, axis: 0, out: outBuffer);
        expect(identical(res, outBuffer), true);
        // a is:
        // [[1, 2],
        //  [3, 4]]
        // Along axis 0:
        // Col 0: [1, 3] -> FFT: [4, -2]
        // Col 1: [2, 4] -> FFT: [6, -2]
        // Result:
        // [[4, 6],
        //  [-2, -2]]
        final outBufferList = outBuffer.toList();
        expect(outBufferList[0].real, closeTo(4.0, 1e-9));
        expect(outBufferList[1].real, closeTo(6.0, 1e-9));
        expect(outBufferList[2].real, closeTo(-2.0, 1e-9));
        expect(outBufferList[3].real, closeTo(-2.0, 1e-9));
      }),
    );

    test(
      'ifft with out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [
            Complex(10.0, 0.0),
            Complex(-2.0, 2.0),
            Complex(-2.0, 0.0),
            Complex(-2.0, -2.0),
          ],
          [4],
          DType.complex128,
        );
        final outBuffer = NDArray<Complex>.zeros([4], DType.complex128);
        final res = ifft(a, out: outBuffer);
        expect(identical(res, outBuffer), true);
        final outBufferList = outBuffer.toList();
        expect(outBufferList[0].real, closeTo(1.0, 1e-9));
        expect(outBufferList[1].real, closeTo(2.0, 1e-9));
        expect(outBufferList[2].real, closeTo(3.0, 1e-9));
        expect(outBufferList[3].real, closeTo(4.0, 1e-9));
      }),
    );

    test(
      'gradientArray with out parameter',
      () => NDArray.scope(() {
        final f = NDArray.fromList([1.0, 2.0, 4.0, 8.0], [2, 2], DType.float64);
        final out1 = NDArray<Float64>.zeros([2, 2], DType.float64);
        final out2 = NDArray<Float64>.zeros([2, 2], DType.float64);
        final res = gradientArray(f, out: [out1, out2]);
        expect(identical(res[0], out1), true);
        expect(identical(res[1], out2), true);
        // f is:
        // [[1, 2],
        //  [4, 8]]
        // Grad axis 0 (rows):
        // Col 0: [1, 4] -> [3, 3]
        // Col 1: [2, 8] -> [6, 6]
        // Grad 0:
        // [[3, 6],
        //  [3, 6]]
        // Grad axis 1 (cols):
        // Row 0: [1, 2] -> [1, 1]
        // Row 1: [4, 8] -> [4, 4]
        // Grad 1:
        // [[1, 1],
        //  [4, 4]]
        expect(out1.toList(), [3.0, 6.0, 3.0, 6.0]);
        expect(out2.toList(), [1.0, 1.0, 4.0, 4.0]);
      }),
    );

    test(
      'svd with out parameter (m >= n)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );
        final outU = NDArray<Float64>.zeros([3, 3], DType.float64);
        final outS = NDArray<Float64>.zeros([2], DType.float64);
        final outVh = NDArray<Float64>.zeros([2, 2], DType.float64);

        final res = svd(a, out: (U: outU, S: outS, Vh: outVh));
        expect(identical(res.U, outU), true);
        expect(identical(res.S, outS), true);
        expect(identical(res.Vh, outVh), true);

        // SVD values should be populated
        expect(outS.toList()[0], greaterThan(0.0));
        expect(outS.toList()[1], greaterThan(0.0));
      }),
    );

    test(
      'svd with out parameter (m < n)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final outU = NDArray<Float64>.zeros([2, 2], DType.float64);
        final outS = NDArray<Float64>.zeros([2], DType.float64);
        final outVh = NDArray<Float64>.zeros([3, 3], DType.float64);

        final res = svd(a, out: (U: outU, S: outS, Vh: outVh));
        expect(identical(res.U, outU), true);
        expect(identical(res.S, outS), true);
        expect(identical(res.Vh, outVh), true);

        expect(outS.toList()[0], greaterThan(0.0));
        expect(outS.toList()[1], greaterThan(0.0));
      }),
    );
  });
}
