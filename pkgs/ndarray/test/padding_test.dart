import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Padding Tests', () {
    test(
      'Constant Padding - 1D - Uniform',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
        final r = pad(
          a,
          PadWidth.all(2),
          mode: PaddingMode.constant,
          constantValues: PadValues.all(9),
        );
        expect(r.shape, [7]);
        expect(r.toList(), [9, 9, 1, 2, 3, 9, 9]);
      }),
    );

    test(
      'Constant Padding - 2D - Per Axis',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        final r = pad(
          a,
          PadWidth.axes([(1, 1), (2, 2)]),
          mode: PaddingMode.constant,
          constantValues: PadValues.axes([(8, 8), (9, 9)]),
        );
        expect(r.shape, [4, 6]);
        expect(r.toList(), [
          9,
          9,
          8,
          8,
          9,
          9,
          9,
          9,
          1,
          2,
          9,
          9,
          9,
          9,
          3,
          4,
          9,
          9,
          9,
          9,
          8,
          8,
          9,
          9,
        ]);
      }),
    );

    test(
      'Constant Mode - 2D Int with different before/after',
      () => NDArray.scope(() {
        final arr = NDArray<int>.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        final padded = pad(
          arr,
          PadWidth.axes([(1, 2), (2, 1)]),
          mode: PaddingMode.constant,
          constantValues: PadValues.axes([(10, 20), (30, 40)]),
        );
        expect(padded.shape, [5, 5]);
        expect(padded.toList(), [
          30,
          30,
          10,
          10,
          40,
          30,
          30,
          1,
          2,
          40,
          30,
          30,
          3,
          4,
          40,
          30,
          30,
          20,
          20,
          40,
          30,
          30,
          20,
          20,
          40,
        ]);
      }),
    );

    test(
      'Edge Padding - 1D',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
        final r = pad(a, PadWidth.axes([(2, 3)]), mode: PaddingMode.edge);
        expect(r.shape, [8]);
        expect(r.toList(), [1, 1, 1, 2, 3, 3, 3, 3]);
      }),
    );

    test(
      'Edge Padding - 2D',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        final r = pad(a, PadWidth.all(1), mode: PaddingMode.edge);
        expect(r.shape, [4, 4]);
        expect(r.toList(), [1, 1, 2, 2, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 4, 4]);
      }),
    );

    test(
      'Wrap Padding',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
        final r = pad(a, PadWidth.axes([(2, 2)]), mode: PaddingMode.wrap);
        expect(r.shape, [7]);
        expect(r.toList(), [2, 3, 1, 2, 3, 1, 2]);
      }),
    );

    test(
      'Reflect Padding - 1D',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
        final r = pad(a, PadWidth.all(2), mode: PaddingMode.reflect);
        expect(r.shape, [7]);
        expect(r.toList(), [3, 2, 1, 2, 3, 2, 1]);
      }),
    );

    test(
      'Reflect Padding - Large padding',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
        final r = pad(a, PadWidth.all(5), mode: PaddingMode.reflect);
        expect(r.shape, [13]);
        expect(r.toList(), [2, 1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2]);
      }),
    );

    test(
      'Symmetric Padding',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
        final r = pad(a, PadWidth.all(2), mode: PaddingMode.symmetric);
        expect(r.shape, [7]);
        expect(r.toList(), [2, 1, 1, 2, 3, 3, 2]);
      }),
    );

    test(
      'Symmetric Mode - 1D Large Pad',
      () => NDArray.scope(() {
        final arr = NDArray<double>.fromList(
          [1.0, 2.0, 3.0],
          [3],
          DType.float64,
        );
        final padded = pad(arr, PadWidth.all(5), mode: PaddingMode.symmetric);
        expect(padded.toList(), [
          2.0,
          3.0,
          3.0,
          2.0,
          1.0,
          1.0,
          2.0,
          3.0,
          3.0,
          2.0,
          1.0,
          1.0,
          2.0,
        ]);
      }),
    );

    test(
      'Linear Ramp Padding',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final r = pad(
          a,
          PadWidth.all(2),
          mode: PaddingMode.linearRamp,
          endValues: PadValues.all(5.0, 7.0),
        );
        expect(r.shape, [7]);
        expect(r.toList(), [5.0, 3.0, 1.0, 2.0, 3.0, 5.0, 7.0]);
      }),
    );

    test(
      'Linear Ramp Padding - Int',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
        final r = pad(
          a,
          PadWidth.all(2),
          mode: PaddingMode.linearRamp,
          endValues: PadValues.all(5, 7),
        );
        expect(r.shape, [7]);
        expect(r.toList(), [5, 3, 1, 2, 3, 5, 7]);
      }),
    );

    test(
      'Stats Padding - Maximum',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 5, 3, 9, 2], [5], DType.int32);
        final r = pad(a, PadWidth.all(2), mode: PaddingMode.maximum);
        expect(r.shape, [9]);
        expect(r.toList(), [9, 9, 1, 5, 3, 9, 2, 9, 9]);
      }),
    );

    test(
      'Stats Padding - Maximum with Window',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 5, 3, 9, 2], [5], DType.int32);
        final r = pad(
          a,
          PadWidth.all(2),
          mode: PaddingMode.maximum,
          statLength: StatLength.all(3),
        );
        expect(r.shape, [9]);
        expect(r.toList(), [5, 5, 1, 5, 3, 9, 2, 9, 9]);
      }),
    );

    test(
      'Stats Padding - Mean',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final r = pad(a, PadWidth.all(1), mode: PaddingMode.mean);
        expect(r.shape, [6]);
        expect(r.toList(), [2.5, 1.0, 2.0, 3.0, 4.0, 2.5]);
      }),
    );

    test(
      'Stats Padding - Median - Odd',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 5, 3], [3], DType.int32);
        final r = pad(a, PadWidth.all(1), mode: PaddingMode.median);
        expect(r.shape, [5]);
        expect(r.toList(), [3, 1, 5, 3, 3]);
      }),
    );

    test(
      'Stats Padding - Median - Even',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 5.0, 3.0, 9.0], [4], DType.float64);
        final r = pad(a, PadWidth.all(1), mode: PaddingMode.median);
        expect(r.shape, [6]);
        expect(r.toList(), [4.0, 1.0, 5.0, 3.0, 9.0, 4.0]);
      }),
    );

    test(
      'Complex Padding - Constant',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [Complex(1, 2), Complex(3, 4)],
          [2],
          DType.complex128,
        );
        final r = pad(
          a,
          PadWidth.all(1),
          mode: PaddingMode.constant,
          constantValues: PadValues.all(Complex(9, 9)),
        );
        expect(r.shape, [4]);
        expect(r.toList(), [
          Complex(9, 9),
          Complex(1, 2),
          Complex(3, 4),
          Complex(9, 9),
        ]);
      }),
    );

    test(
      'Complex Padding - Median (Independent)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [Complex(1, 6), Complex(5, 2), Complex(3, 4)],
          [3],
          DType.complex128,
        );
        final r = pad(a, PadWidth.all(1), mode: PaddingMode.median);
        expect(r.shape, [5]);
        expect(r.toList(), [
          Complex(3, 4),
          Complex(1, 6),
          Complex(5, 2),
          Complex(3, 4),
          Complex(3, 4),
        ]);
      }),
    );

    test(
      'Complex Padding - Min/Max (Lexicographical)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [Complex(1, 10), Complex(2, 1), Complex(2, 2)],
          [3],
          DType.complex128,
        );
        final rMin = pad(a, PadWidth.all(1), mode: PaddingMode.minimum);
        final rMax = pad(a, PadWidth.all(1), mode: PaddingMode.maximum);

        expect(rMin.toList(), [
          Complex(1, 10),
          Complex(1, 10),
          Complex(2, 1),
          Complex(2, 2),
          Complex(1, 10),
        ]);
        expect(rMax.toList(), [
          Complex(2, 2),
          Complex(1, 10),
          Complex(2, 1),
          Complex(2, 2),
          Complex(2, 2),
        ]);
      }),
    );

    test(
      'Boolean Padding',
      () => NDArray.scope(() {
        final a = NDArray.fromList([true, false, true], [3], DType.boolean);
        final r = pad(
          a,
          PadWidth.all(1),
          mode: PaddingMode.constant,
          constantValues: PadValues.all(true),
        );
        expect(r.shape, [5]);
        expect(r.toList(), [true, true, false, true, true]);
      }),
    );

    test(
      'Zero Padding',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
        final r = pad(a, PadWidth.all(0));
        expect(r.shape, [3]);
        expect(r.toList(), [1, 2, 3]);
        expect(r, isNot(same(a)));
      }),
    );

    test(
      'Out Parameter Reuse',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
        final out = NDArray<int>.zeros([5], DType.int32);
        final r = pad(
          a,
          PadWidth.all(1),
          mode: PaddingMode.constant,
          constantValues: PadValues.all(9),
          out: out,
        );
        expect(r, same(out));
        expect(out.toList(), [9, 1, 2, 3, 9]);
      }),
    );

    test(
      'Preconditions - Disposed',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
        a.dispose();
        expect(() => pad(a, PadWidth.all(1)), throwsStateError);
      }),
    );

    test(
      'Preconditions - 0-D Array',
      () => NDArray.scope(() {
        final a = NDArray<int>.fromList([1], [], DType.int32);
        expect(() => pad(a, PadWidth.all(1)), throwsArgumentError);
      }),
    );
  });
}
