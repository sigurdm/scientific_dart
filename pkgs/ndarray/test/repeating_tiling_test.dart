import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

List<int> rep(dynamic v) {
  if (v is int) return [v];
  if (v is List<int>) return v;
  throw ArgumentError();
}

void main() {
  group('Repeating & Tiling Tests', () {
    group('repeat Tests', () {
      test('1D array, scalar repeats', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = repeat(a, rep(2));
          expect(r.shape, [6]);
          expect(r.toList(), [1, 1, 2, 2, 3, 3]);
        });
      });

      test('1D array, list repeats', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = repeat(a, rep([1, 2, 0]));
          expect(r.shape, [3]);
          expect(r.toList(), [1, 2, 2]);
        });
      });

      test('2D array, axis = 0', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final r = repeat(a, rep(2), axis: 0);
          expect(r.shape, [4, 2]);
          expect(r.toList(), [1, 2, 1, 2, 3, 4, 3, 4]);
        });
      });

      test('2D array, axis = 1', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final r = repeat(a, rep([2, 1]), axis: 1);
          expect(r.shape, [2, 3]);
          expect(r.toList(), [1, 1, 2, 3, 3, 4]);
        });
      });

      test('2D array, negative axis', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final r = repeat(a, rep(2), axis: -1);
          expect(r.shape, [2, 4]);
          expect(r.toList(), [1, 1, 2, 2, 3, 3, 4, 4]);
        });
      });

      test('2D array, axis = null (flatten)', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final r = repeat(a, rep(2));
          expect(r.shape, [8]);
          expect(r.toList(), [1, 1, 2, 2, 3, 3, 4, 4]);
        });
      });

      test('3D array, repeat along axis 1', () {
        NDArray.scope(() {
          final a = NDArray.fromList(List.generate(8, (i) => i + 1), [
            2,
            2,
            2,
          ], DType.int32);
          final r = repeat(a, rep([2, 1]), axis: 1);
          expect(r.shape, [2, 3, 2]);
          expect(r.toList(), [
            1, 2, 1, 2, 3, 4, // block 0
            5, 6, 5, 6, 7, 8, // block 1
          ]);
        });
      });

      test('Edge case: repeats with 0 (deletes elements)', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = repeat(a, rep(0));
          expect(r.shape, [0]);
          expect(r.toList(), <int>[]);
        });
      });

      test('Precondition checks', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          expect(() => repeat(a, rep([-1, 1, 1])), throwsArgumentError);
          expect(() => repeat(a, rep([1, -1, 1])), throwsArgumentError);
          expect(() => repeat(a, rep([1, 2])), throwsArgumentError);
          expect(() => repeat(a, rep(2), axis: 2), throwsRangeError);
          expect(() => repeat(a, rep(2), axis: -2), throwsRangeError);
        });
      });

      test('out parameter: correct shape/dtype', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final out = NDArray.create([6], DType.int32);
          final r = repeat(a, rep(2), out: out);
          expect(identical(r, out), true);
          expect(r.toList(), [1, 1, 2, 2, 3, 3]);
        });
      });

      test('out parameter: incorrect shape/dtype', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final outWrongShape = NDArray.create([5], DType.int32);
          final outWrongDType = NDArray.create([6], DType.float64);

          expect(
            () => repeat(a, rep(2), out: outWrongShape),
            throwsArgumentError,
          );
          expect(
            () => repeat(a, rep(2), out: outWrongDType),
            throwsArgumentError,
          );
        });
      });

      test('Different dtypes (float64, complex128, boolean)', () {
        NDArray.scope(() {
          final f = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final rF = repeat(f, rep(2));
          expect(rF.dtype, DType.float64);
          expect(rF.toList(), [1.0, 1.0, 2.0, 2.0]);

          final c = NDArray.fromList(
            [Complex(1, 2), Complex(3, 4)],
            [2],
            DType.complex128,
          );
          final rC = repeat(c, rep(2));
          expect(rC.dtype, DType.complex128);
          expect(rC.toList(), [
            Complex(1, 2),
            Complex(1, 2),
            Complex(3, 4),
            Complex(3, 4),
          ]);

          final b = NDArray.fromList([true, false], [2], DType.boolean);
          final rB = repeat(b, rep(2));
          expect(rB.dtype, DType.boolean);
          expect(rB.toList(), [true, true, false, false]);
        });
      });

      test('Non-contiguous strided view input', () {
        NDArray.scope(() {
          final parent = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final view = parent.transposed; // [[1, 3], [2, 4]]
          expect(view.isContiguous, false);

          final r = repeat(view, rep(2), axis: 0);
          expect(r.shape, [4, 2]);
          expect(r.toList(), [1, 3, 1, 3, 2, 4, 2, 4]);
        });
      });
    });

    group('tile Tests', () {
      test('1D array, scalar reps', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          final t = tile(a, rep(3));
          expect(t.shape, [6]);
          expect(t.toList(), [1, 2, 1, 2, 1, 2]);
        });
      });

      test('1D array, list reps', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          final t = tile(a, rep([3]));
          expect(t.shape, [6]);
          expect(t.toList(), [1, 2, 1, 2, 1, 2]);
        });
      });

      test('2D array, reps of same rank', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final t = tile(a, rep([2, 3]));
          expect(t.shape, [4, 6]);
          expect(t.toList(), [
            1,
            2,
            1,
            2,
            1,
            2,
            3,
            4,
            3,
            4,
            3,
            4,
            1,
            2,
            1,
            2,
            1,
            2,
            3,
            4,
            3,
            4,
            3,
            4,
          ]);
        });
      });

      test('2D array, reps with prepended 1s (reps.length < rank)', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final t = tile(a, rep([2])); // treated as [1, 2]
          expect(t.shape, [2, 4]);
          expect(t.toList(), [1, 2, 1, 2, 3, 4, 3, 4]);
        });
      });

      test('2D array, reps with expanded rank (reps.length > rank)', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final t = tile(a, rep([2, 1, 2])); // a is promoted to [1, 2, 2]
          expect(t.shape, [2, 2, 4]);
          expect(t.toList(), [1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4]);
        });
      });

      test('Edge case: reps with 0 (returns empty array)', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          final t = tile(a, rep(0));
          expect(t.shape, [0]);
          expect(t.toList(), <int>[]);
        });
      });

      test('Precondition checks', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          expect(() => tile(a, rep([-1])), throwsArgumentError);
          expect(() => tile(a, rep([1, -1])), throwsArgumentError);
        });
      });

      test('out parameter: correct shape/dtype', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          final out = NDArray.create([4], DType.int32);
          final t = tile(a, rep(2), out: out);
          expect(identical(t, out), true);
          expect(t.toList(), [1, 2, 1, 2]);
        });
      });

      test('out parameter: incorrect shape/dtype', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          final outWrongShape = NDArray.create([5], DType.int32);
          final outWrongDType = NDArray.create([4], DType.float64);

          expect(
            () => tile(a, rep(2), out: outWrongShape),
            throwsArgumentError,
          );
          expect(
            () => tile(a, rep(2), out: outWrongDType),
            throwsArgumentError,
          );
        });
      });

      test('Different dtypes (float64, complex128, boolean)', () {
        NDArray.scope(() {
          final f = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final tF = tile(f, rep(2));
          expect(tF.dtype, DType.float64);
          expect(tF.toList(), [1.0, 2.0, 1.0, 2.0]);

          final c = NDArray.fromList(
            [Complex(1, 2), Complex(3, 4)],
            [2],
            DType.complex128,
          );
          final tC = tile(c, rep(2));
          expect(tC.dtype, DType.complex128);
          expect(tC.toList(), [
            Complex(1, 2),
            Complex(3, 4),
            Complex(1, 2),
            Complex(3, 4),
          ]);

          final b = NDArray.fromList([true, false], [2], DType.boolean);
          final tB = tile(b, rep(2));
          expect(tB.dtype, DType.boolean);
          expect(tB.toList(), [true, false, true, false]);
        });
      });

      test('Non-contiguous strided view input', () {
        NDArray.scope(() {
          final parent = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final view = parent.transposed; // [[1, 3], [2, 4]]
          expect(view.isContiguous, false);

          final t = tile(view, rep([2, 1])); // tile rows 2x, cols 1x
          expect(t.shape, [4, 2]);
          expect(t.toList(), [1, 3, 2, 4, 1, 3, 2, 4]);
        });
      });
    });
  });
}
