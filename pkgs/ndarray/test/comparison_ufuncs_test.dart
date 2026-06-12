import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
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
}
