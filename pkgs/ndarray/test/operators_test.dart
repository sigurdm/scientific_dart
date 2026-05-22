import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
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
        expect(sum.toList()[0].real, 11.0);
        expect(sum.toList()[0].imag, 22.0);

        final neg = -a;
        expect(neg.toList()[0].real, -1.0);
        expect(neg.toList()[0].imag, -2.0);

        final scalarSum = a + Complex(10, 10);
        expect(scalarSum.toList()[0].real, 11.0);
        expect(scalarSum.toList()[0].imag, 12.0);
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
        expect(divDouble.data[0], double.infinity);
        expect(divDouble.data[1], double.negativeInfinity);
        expect(divDouble.data[2].isNaN, true);

        // True Division on integer arrays upcasts to float64 and handles division by zero identically
        final intArr = NDArray.fromList([5, -5, 0], [3], DType.int32);
        final zeroIntArr = NDArray.zeros([3], DType.int32);
        final divInt = intArr / zeroIntArr;
        expect(divInt.dtype, DType.float64);
        expect(divInt.data[0], double.infinity);
        expect(divInt.data[1], double.negativeInfinity);
        expect(divInt.data[2].isNaN, true);

        // 2. Floor Division (operator ~/) and Remainder (operator %) by zero on integer throws UnsupportedError
        expect(() => intArr ~/ zeroIntArr, throwsA(isA<UnsupportedError>()));
        expect(() => intArr % zeroIntArr, throwsA(isA<UnsupportedError>()));

        final minIntArr = NDArray.fromList([-2147483648], [1], DType.int32);
        final minusOneArr = NDArray.fromList([-1], [1], DType.int32);
        expect(
          (minIntArr ~/ minusOneArr).data[0],
          -2147483648,
        ); // or wraps/throws? Let's see!

        // Floor Division on floats with zero divisor returns NaN
        final divFloorDouble = doubleArr ~/ zeroDoubleArr;
        expect(divFloorDouble.data.every((x) => x.isNaN), true);

        // 3. Integer Multiplication Overflow wraps around (two's complement)
        // Max 32-bit signed int is 2147483647
        final overflowArr32 = NDArray.fromList([2147483647], [1], DType.int32);
        final scaleArr32 = NDArray.fromList([2], [1], DType.int32);
        final prod32 = overflowArr32 * scaleArr32;
        expect(prod32.dtype, DType.int32);
        expect(
          prod32.data[0],
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
          prod64.data[0],
          -2,
        ); // 9223372036854775807 * 2 wraps to -2 in 64-bit two's complement

        // 4. Float Multiplication Overflow yields infinity
        final overflowArrFloat = NDArray.fromList([1e308], [1], DType.float64);
        final prodFloat = overflowArrFloat * 10.0;
        expect(prodFloat.data[0], double.infinity);
      }),
    );
  });
}
