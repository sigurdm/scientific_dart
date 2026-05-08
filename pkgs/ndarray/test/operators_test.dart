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
  });
}
