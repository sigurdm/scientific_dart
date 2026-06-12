import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';
import 'dart:typed_data';

void main() {
  group('sign ufunc', () {
    test('int32 sign', () {
      final a = NDArray.fromList(Int32List.fromList([-5, 0, 10]), [
        3,
      ], DType.int32);
      final s = sign(a);
      expect(s.data, [-1, 0, 1]);
      expect(s.dtype, DType.int32);
    });

    test('float64 sign', () {
      final a = NDArray.fromList(
        Float64List.fromList([-5.5, 0.0, 10.2, double.nan]),
        [4],
        DType.float64,
      );
      final s = sign(a);
      expect(s.data[0], -1.0);
      expect(s.data[1], 0.0);
      expect(s.data[2], 1.0);
      expect(s.data[3].isNaN, isTrue);
      expect(s.dtype, DType.float64);
    });

    test('complex128 sign', () {
      final a = NDArray.create([2], DType.complex128);
      a.data[0] = Complex(3.0, 4.0); // mag = 5.0
      a.data[1] = Complex(0.0, 0.0);

      final s = sign(a);
      expect(s.data[0].real, closeTo(0.6, 1e-10));
      expect(s.data[0].imag, closeTo(0.8, 1e-10));
      expect(s.data[1].real, 0.0);
      expect(s.data[1].imag, 0.0);
      expect(s.dtype, DType.complex128);
    });
  });
}
