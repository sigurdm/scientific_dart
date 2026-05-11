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

          final addRes = a.add(b);
          expect(addRes.dtype, DType.float64);
          expect(addRes.toList(), [3.0, 4.0, 5.0, 6.0]);

          final subRes = a.subtract(b);
          expect(subRes.toList(), [-1.0, 0.0, 1.0, 2.0]);

          final mulRes = a.multiply(b);
          expect(mulRes.toList(), [2.0, 4.0, 6.0, 8.0]);

          final divRes = a.divide(b);
          expect(divRes.toList(), [0.5, 1.0, 1.5, 2.0]);
        });
      },
    );

    test('Contiguous Float32 SIMD lanes and C vector sweeps', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float32);
        final b = NDArray.fromList([1.0, 1.0, 1.0, 1.0], [4], DType.float32);

        final addRes = a.add(b);
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

        final res = a.add(b);
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
        final res = a.add<Complex>(b);
        expect(res.dtype, DType.complex128);
        expect(res.data[0].real, 11.0);
        expect(res.data[0].imag, 2.0);
        expect(res.data[1].real, 23.0);
        expect(res.data[1].imag, 4.0);
      });
    });

    test('out parameter validation and success/error paths', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final b = NDArray.fromList([2.0, 2.0], [2], DType.float64);
        final outValid = NDArray<Float64>.create([2], DType.float64);

        final res = a.add<Float64>(b, out: outValid);
        expect(identical(res, outValid), true);
        expect(outValid.toList(), [3.0, 4.0]);

        final outInvalidShape = NDArray<Float64>.create([3], DType.float64);
        expect(
          () => a.add<Float64>(b, out: outInvalidShape),
          throwsArgumentError,
        );

        final outInvalidDType = NDArray<Int32>.create([2], DType.int32);
        expect(
          () => a.add<Int32>(b, out: outInvalidDType),
          throwsArgumentError,
        );
      });
    });
  });
}
