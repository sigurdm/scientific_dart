import 'package:num_dart/num_dart.dart';
import 'package:test/test.dart';

void main() {
  group('NDArray Complex Components (real, imag) Tests', () {
    test('real() and imag() basic complex128 contiguous checks', () {
      final a = NDArray<Complex>.create([3], DType.complex128);
      a.data[0] = Complex(3.5, 4.5);
      a.data[1] = Complex(-1.0, 0.0);
      a.data[2] = Complex(0.0, -2.5);

      final r = real(a);
      expect(r.dtype, DType.float64);
      expect(r.shape, [3]);
      expect(r.toList(), [3.5, -1.0, 0.0]);

      final im = imag(a);
      expect(im.dtype, DType.float64);
      expect(im.shape, [3]);
      expect(im.toList(), [4.5, 0.0, -2.5]);
    });

    test('real() and imag() support complex64', () {
      final a = NDArray<Complex>.create([2], DType.complex64);
      a.data[0] = Complex(1.5, -2.5);
      a.data[1] = Complex(0.0, 3.0);

      final r = real(a);
      expect(r.dtype, DType.float32);
      expect(r.toList(), [1.5, 0.0]);

      final im = imag(a);
      expect(im.dtype, DType.float32);
      expect(im.toList(), [-2.5, 3.0]);
    });

    test('real() on already real arrays returns zero-copy views', () {
      final a = NDArray.fromList([10.0, 20.0], [2], DType.float64);
      final r = real(a);

      expect(r.dtype, DType.float64);
      expect(r.toList(), [10.0, 20.0]);

      // Mutating view updates parent zero-copy!
      r.data[0] = 99.0;
      expect(a.data[0], 99.0);
    });

    test('imag() on already real arrays returns zero-filled arrays', () {
      final a = NDArray.fromList([10.0, 20.0], [2], DType.float64);
      final im = imag(a);

      expect(im.dtype, DType.float64);
      expect(im.toList(), [0.0, 0.0]);
    });

    test('recycler out parameter checks', () {
      final a = NDArray<Complex>.create([2], DType.complex128);
      a.data[0] = Complex(1.0, 2.0);
      a.data[1] = Complex(3.0, 4.0);

      final out = NDArray.create([2], DType.float64);
      final res = real(a, out: out);

      expect(identical(res, out), true);
      expect(out.toList(), [1.0, 3.0]);
    });

    test('disposed arrays throw StateError', () {
      final a = NDArray<Complex>.create([2], DType.complex128);
      a.dispose();
      expect(() => real(a), throwsStateError);
      expect(() => imag(a), throwsStateError);
    });
  });
}
