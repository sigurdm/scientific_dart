import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('NDArray Logical Reductions (all, any) Tests', () {
    test(
      'all() basic global reduction checks',
      () => NDArray.scope(() {
        final a = NDArray.fromList([true, true, true], [3], DType.boolean);
        final b = NDArray.fromList([true, false, true], [3], DType.boolean);
        final c = NDArray.fromList([1, 2, 3], [3], DType.int32);
        final d = NDArray.fromList([1, 0, 3], [3], DType.int32);

        expect(all(a), true);
        expect(all(b), false);
        expect(all(c), true);
        expect(all(d), false);
      }),
    );

    test(
      'any() basic global reduction checks',
      () => NDArray.scope(() {
        final a = NDArray.fromList([false, false, false], [3], DType.boolean);
        final b = NDArray.fromList([false, true, false], [3], DType.boolean);
        final c = NDArray.fromList([0, 0, 0], [3], DType.int32);
        final d = NDArray.fromList([0, 5, 0], [3], DType.int32);

        expect(any(a), false);
        expect(any(b), true);
        expect(any(c), false);
        expect(any(d), true);
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
        expect(resAnd.dtype, DType.int32);
        expect(resAnd.toList(), [1, 0, 0]); // 1 is true, 0 is false

        final resOr = logical_or(mask1, mask2);
        expect(resOr.dtype, DType.int32);
        expect(resOr.toList(), [1, 1, 1]);

        final resXor = logical_xor(mask1, mask2);
        expect(resXor.dtype, DType.int32);
        expect(resXor.toList(), [0, 1, 1]);
      }),
    );

    test(
      'logical_and cross-type promotions on float/boolean, complex/boolean, and int/boolean',
      () => NDArray.scope(() {
        final mask = NDArray.fromList([true, true, false], [3], DType.boolean);

        // Float64 / boolean
        final f64 = NDArray.fromList([0.0, 2.5, 0.0], [3], DType.float64);
        final resF1 = logical_and(f64, mask);
        expect(resF1.toList(), [0, 1, 0]);
        final resF2 = logical_and(mask, f64);
        expect(resF2.toList(), [0, 1, 0]);

        // Complex128 / boolean
        final c128 = NDArray<Complex>.create([3], DType.complex128);
        c128.data[0] = Complex(0.0, 0.0); // false
        c128.data[1] = Complex(1.0, -1.0); // true
        c128.data[2] = Complex(0.0, 0.0);
        final resC1 = logical_and(c128, mask);
        expect(resC1.toList(), [0, 1, 0]);
        final resC2 = logical_and(mask, c128);
        expect(resC2.toList(), [0, 1, 0]);

        // Int32 / boolean
        final i32 = NDArray.fromList([0, 5, 0], [3], DType.int32);
        final resI1 = logical_and(i32, mask);
        expect(resI1.toList(), [0, 1, 0]);
        final resI2 = logical_and(mask, i32);
        expect(resI2.toList(), [0, 1, 0]);
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
