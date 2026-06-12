import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';

void main() {
  group('unique', () {
    test('double with duplicates and NaNs', () {
      final a = NDArray.fromList(
        [3.0, 1.0, 2.0, 1.0, double.nan, 2.0, double.nan],
        [7],
        DType.float64,
      );
      final res = unique(a);
      expect(res.shape, [4]);
      // NaNs should be sorted to the end
      expectListEqualsWithNaNs(res.toList(), [1.0, 2.0, 3.0, double.nan]);
      a.dispose();
      res.dispose();
    });

    test('int32 flat', () {
      final a = NDArray.fromList([1, 2, 2, 3, 1, 4], [6], DType.int32);
      final res = unique(a);
      expect(res.shape, [4]);
      expect(res.toList(), [1, 2, 3, 4]);
      a.dispose();
      res.dispose();
    });

    test('int32 2D (flattened)', () {
      final a = NDArray.fromList([1, 2, 2, 3], [2, 2], DType.int32);
      final res = unique(a);
      expect(res.shape, [3]);
      expect(res.toList(), [1, 2, 3]);
      a.dispose();
      res.dispose();
    });

    test('uint8 empty', () {
      final a = NDArray.fromList([], [0], DType.uint8);
      final res = unique(a);
      expect(res.shape, [0]);
      expect(res.toList(), <int>[]);
      a.dispose();
      res.dispose();
    });

    test('uint8 non-empty unique', () {
      final a = NDArray.fromList([3, 1, 2, 1, 3], [5], DType.uint8);
      final res = unique(a);
      expect(res.dtype, DType.uint8);
      expect(res.toList(), [1, 2, 3]);
      a.dispose();
      res.dispose();
    });

    test('complex128', () {
      final a = NDArray.fromList(
        [Complex(1, 2), Complex(3, 4), Complex(1, 2), Complex(2, 3)],
        [4],
        DType.complex128,
      );
      final res = unique(a);
      expect(res.shape, [3]);
      // Sorted lexicographically: (1,2), (2,3), (3,4)
      expect(res.toList(), [Complex(1, 2), Complex(2, 3), Complex(3, 4)]);
      a.dispose();
      res.dispose();
    });

    test('complex128 with duplicates and NaNs', () {
      final a = NDArray.fromList(
        [
          Complex(3.0, 4.0),
          Complex(1.0, 2.0),
          Complex(double.nan, 2.0),
          Complex(1.0, double.nan),
          Complex(double.nan, double.nan),
          Complex(1.0, 2.0),
          Complex(double.nan, 2.0),
        ],
        [7],
        DType.complex128,
      );
      final res = unique(a);
      expect(res.shape, [5]);
      expectListEqualsWithNaNs(res.toList(), [
        Complex(1.0, 2.0),
        Complex(1.0, double.nan),
        Complex(3.0, 4.0),
        Complex(double.nan, 2.0),
        Complex(double.nan, double.nan),
      ]);
      a.dispose();
      res.dispose();
    });

    test('int32 non-contiguous 1D', () {
      final a = NDArray.fromList([1, 2, 3, 4, 5, 6], [6], DType.int32);
      // Slice with step 2: [1, 3, 5]
      final slice = a.slice([Slice(step: 2)]);
      expect(slice.isContiguous, false);
      expect(slice.toList(), [1, 3, 5]);

      final res = unique(slice);
      expect(res.shape, [3]);
      expect(res.toList(), [1, 3, 5]);

      a.dispose();
      res.dispose();
    });

    test('optional returns int32', () {
      final a = NDArray.fromList([1, 2, 2, 3, 1, 4], [6], DType.int32);

      final (u, index: idx, inverse: inv, counts: cnt) = unique(
        a,
        returnIndex: true,
        returnInverse: true,
        returnCounts: true,
      );

      expect(u.toList(), [1, 2, 3, 4]);
      expect(idx!.toList(), [0, 1, 3, 5]);
      expect(inv!.toList(), [0, 1, 1, 2, 0, 3]);
      expect(cnt!.toList(), [2, 2, 1, 1]);

      a.dispose();
      u.dispose();
      idx.dispose();
      inv.dispose();
      cnt.dispose();
    });

    test('optional returns double with NaNs', () {
      final a = NDArray.fromList(
        [3.0, 1.0, 2.0, 1.0, double.nan, 2.0, double.nan],
        [7],
        DType.float64,
      );

      final (u, index: idx, inverse: inv, counts: cnt) = unique(
        a,
        returnIndex: true,
        returnInverse: true,
        returnCounts: true,
      );

      expectListEqualsWithNaNs(u.toList(), [1.0, 2.0, 3.0, double.nan]);
      expect(idx!.toList(), [1, 2, 0, 4]);
      expect(inv!.toList(), [2, 0, 1, 0, 3, 1, 3]);
      expect(cnt!.toList(), [2, 2, 1, 2]);

      a.dispose();
      u.dispose();
      idx.dispose();
      inv.dispose();
      cnt.dispose();
    });
  });

  group('intersect1d', () {
    test('int32 basic', () {
      final a = NDArray.fromList([1, 3, 4, 3], [4], DType.int32);
      final b = NDArray.fromList([3, 1, 2, 1], [4], DType.int32);
      final res = intersect1d(a, b);
      expect(res.shape, [2]);
      expect(res.toList(), [1, 3]);
      a.dispose();
      b.dispose();
      res.dispose();
    });

    test('int32 assumeUnique', () {
      final a = NDArray.fromList([1, 3, 4], [3], DType.int32);
      final b = NDArray.fromList([1, 2, 3], [3], DType.int32);
      final res = intersect1d(a, b, assumeUnique: true);
      expect(res.shape, [2]);
      expect(res.toList(), [1, 3]);
      a.dispose();
      b.dispose();
      res.dispose();
    });

    test('int32 assumeUnique with unsorted inputs', () {
      final a = NDArray.fromList([3, 1, 4], [3], DType.int32);
      final b = NDArray.fromList([2, 1, 3], [3], DType.int32);
      final res = intersect1d(a, b, assumeUnique: true);
      expect(res.shape, [2]);
      expect(res.toList(), [1, 3]);
      a.dispose();
      b.dispose();
      res.dispose();
    });

    test('uint8 basic intersect1d', () {
      final a = NDArray.fromList([1, 2, 3, 2], [4], DType.uint8);
      final b = NDArray.fromList([2, 3, 4, 3], [4], DType.uint8);
      final res = intersect1d(a, b);
      expect(res.dtype, DType.uint8);
      expect(res.toList(), [2, 3]);
      a.dispose();
      b.dispose();
      res.dispose();
    });

    test('double with NaNs', () {
      final a = NDArray.fromList([double.nan, 1.0, 2.0], [3], DType.float64);
      final b = NDArray.fromList([2.0, double.nan, 3.0], [3], DType.float64);
      final res = intersect1d(a, b);
      // NaNs should match
      expect(res.shape, [2]);
      expectListEqualsWithNaNs(res.toList(), [2.0, double.nan]);
      a.dispose();
      b.dispose();
      res.dispose();
    });

    test('int32 non-contiguous 1D', () {
      final a = NDArray.fromList([1, 2, 3, 4, 5, 6], [6], DType.int32);
      final b = NDArray.fromList([3, 0, 5, 0], [4], DType.int32);
      final sliceA = a.slice([Slice(step: 2)]); // [1, 3, 5]
      final sliceB = b.slice([Slice(step: 2)]); // [3, 5]

      final res = intersect1d(sliceA, sliceB);
      expect(res.toList(), [3, 5]);

      a.dispose();
      b.dispose();
      res.dispose();
    });
  });

  group('setdiff1d', () {
    test('int32 basic', () {
      final a = NDArray.fromList([1, 2, 3, 2, 4], [5], DType.int32);
      final b = NDArray.fromList([2, 3, 5], [3], DType.int32);
      final res = setdiff1d(a, b);
      expect(res.shape, [2]);
      expect(res.toList(), [1, 4]);
      a.dispose();
      b.dispose();
      res.dispose();
    });

    test('int32 non-contiguous 1D', () {
      final a = NDArray.fromList([1, 2, 3, 4, 5, 6], [6], DType.int32);
      final b = NDArray.fromList([3, 0, 5, 0], [4], DType.int32);
      final sliceA = a.slice([Slice(step: 2)]); // [1, 3, 5]
      final sliceB = b.slice([Slice(step: 2)]); // [3, 5]

      final res = setdiff1d(sliceA, sliceB);
      expect(res.toList(), [1]);

      a.dispose();
      b.dispose();
      res.dispose();
    });

    test('uint8 setdiff1d', () {
      final a = NDArray.fromList([1, 2, 3, 2], [4], DType.uint8);
      final b = NDArray.fromList([2, 4], [2], DType.uint8);
      final res = setdiff1d(a, b);
      expect(res.dtype, DType.uint8);
      expect(res.toList(), [1, 3]);
      a.dispose();
      b.dispose();
      res.dispose();
    });
  });

  group('setxor1d', () {
    test('int32 basic', () {
      final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
      final b = NDArray.fromList([2, 3, 4], [3], DType.int32);
      final res = setxor1d(a, b);
      expect(res.shape, [2]);
      expect(res.toList(), [1, 4]);
      a.dispose();
      b.dispose();
      res.dispose();
    });

    test('uint8 setxor1d', () {
      final a = NDArray.fromList([1, 2, 3], [3], DType.uint8);
      final b = NDArray.fromList([2, 3, 4], [3], DType.uint8);
      final res = setxor1d(a, b);
      expect(res.dtype, DType.uint8);
      expect(res.toList(), [1, 4]);
      a.dispose();
      b.dispose();
      res.dispose();
    });
  });

  group('union1d', () {
    test('int32 basic', () {
      final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
      final b = NDArray.fromList([2, 3, 4, 5], [4], DType.int32);
      final res = union1d(a, b);
      expect(res.shape, [5]);
      expect(res.toList(), [1, 2, 3, 4, 5]);
      a.dispose();
      b.dispose();
      res.dispose();
    });

    test('uint8 union1d', () {
      final a = NDArray.fromList([1, 2, 3], [3], DType.uint8);
      final b = NDArray.fromList([2, 3, 4, 5], [4], DType.uint8);
      final res = union1d(a, b);
      expect(res.dtype, DType.uint8);
      expect(res.toList(), [1, 2, 3, 4, 5]);
      a.dispose();
      b.dispose();
      res.dispose();
    });
  });

  group('isin', () {
    test('int32 basic', () {
      final element = NDArray.fromList([1, 2, 3, 4, 2, 1], [6], DType.int32);
      final testElements = NDArray.fromList([2, 4], [2], DType.int32);
      final res = isin(element, testElements);
      expect(res.shape, [6]);
      expect(res.dtype, DType.boolean);
      expect(res.toList(), [false, true, false, true, true, false]);
      element.dispose();
      testElements.dispose();
      res.dispose();
    });

    test('int32 2D element shape preserved', () {
      final element = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
      final testElements = NDArray.fromList([2, 3], [2], DType.int32);
      final res = isin(element, testElements);
      expect(res.shape, [2, 2]);
      expect(res.dtype, DType.boolean);
      expect(res.toList(), [false, true, true, false]);
      element.dispose();
      testElements.dispose();
      res.dispose();
    });

    test('invert', () {
      final element = NDArray.fromList([1, 2, 3], [3], DType.int32);
      final testElements = NDArray.fromList([2], [1], DType.int32);
      final res = isin(element, testElements, invert: true);
      expect(res.toList(), [true, false, true]);
      element.dispose();
      testElements.dispose();
      res.dispose();
    });

    test('uint8 isin', () {
      final element = NDArray.fromList([1, 2, 3, 2], [4], DType.uint8);
      final testElements = NDArray.fromList([2, 4], [2], DType.uint8);
      final res = isin(element, testElements);
      expect(res.toList(), [false, true, false, true]);
      element.dispose();
      testElements.dispose();
      res.dispose();
    });

    test('isin assumeUnique with unsorted testElements', () {
      final element = NDArray.fromList([1, 2, 3], [3], DType.int32);
      final testElements = NDArray.fromList([3, 1], [2], DType.int32);
      final res = isin(element, testElements, assumeUnique: true);
      expect(res.toList(), [true, false, true]);
      element.dispose();
      testElements.dispose();
      res.dispose();
    });

    test('non-contiguous elements and testElements', () {
      final element = NDArray.fromList([1, 2, 3, 4, 5, 6], [6], DType.int32);
      final testElements = NDArray.fromList([3, 0, 5, 0], [4], DType.int32);

      final sliceElement = element.slice([Slice(step: 2)]); // [1, 3, 5]
      final sliceTest = testElements.slice([Slice(step: 2)]); // [3, 5]

      final res = isin(sliceElement, sliceTest);
      expect(res.toList(), [false, true, true]);

      element.dispose();
      testElements.dispose();
      res.dispose();
    });
  });
}

void expectListEqualsWithNaNs(List actual, List expected) {
  expect(actual.length, expected.length);
  for (var i = 0; i < actual.length; i++) {
    final a = actual[i];
    final e = expected[i];
    if (a is double && e is double) {
      if (a.isNaN && e.isNaN) continue;
    }
    if (a is Complex && e is Complex) {
      if ((a.real.isNaN && e.real.isNaN || a.real == e.real) &&
          (a.imag.isNaN && e.imag.isNaN || a.imag == e.imag)) {
        continue;
      }
    }
    expect(a, e);
  }
}
