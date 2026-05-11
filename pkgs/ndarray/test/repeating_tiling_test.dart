import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Repeating & Tiling Tests', () {
    group('repeat Tests', () {
      test('1D array, scalar repeats', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = repeat(a, 2);
          expect(r.shape, [6]);
          expect(r.toList(), [1, 1, 2, 2, 3, 3]);
        });
      });

      test('1D array, list repeats', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final r = repeat(a, [1, 2, 0]);
          expect(r.shape, [3]);
          expect(r.toList(), [1, 2, 2]);
        });
      });

      test('2D array, axis = 0', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final r = repeat(a, 2, axis: 0);
          expect(r.shape, [4, 2]);
          expect(r.toList(), [1, 2, 1, 2, 3, 4, 3, 4]);
        });
      });

      test('2D array, axis = 1', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final r = repeat(a, [2, 1], axis: 1);
          expect(r.shape, [2, 3]);
          expect(r.toList(), [1, 1, 2, 3, 3, 4]);
        });
      });

      test('2D array, negative axis', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final r = repeat(a, 2, axis: -1);
          expect(r.shape, [2, 4]);
          expect(r.toList(), [1, 1, 2, 2, 3, 3, 4, 4]);
        });
      });

      test('2D array, axis = null (flatten)', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final r = repeat(a, 2);
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
          final r = repeat(a, [2, 1], axis: 1);
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
          final r = repeat(a, 0);
          expect(r.shape, [0]);
          expect(r.toList(), <int>[]);
        });
      });

      test('Precondition checks', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          expect(() => repeat(a, -1), throwsArgumentError);
          expect(() => repeat(a, [1, -1, 1]), throwsArgumentError);
          expect(() => repeat(a, [1, 2]), throwsArgumentError);
          expect(() => repeat(a, 2, axis: 2), throwsRangeError);
          expect(() => repeat(a, 2, axis: -2), throwsRangeError);
          expect(() => repeat(a, 'invalid'), throwsArgumentError);
        });
      });

      test('out parameter: correct shape/dtype', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final out = NDArray<int>.create([6], DType.int32);
          final r = repeat(a, 2, out: out);
          expect(identical(r, out), true);
          expect(r.toList(), [1, 1, 2, 2, 3, 3]);
        });
      });

      test('out parameter: incorrect shape/dtype', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3], [3], DType.int32);
          final outWrongShape = NDArray<int>.create([5], DType.int32);
          final outWrongDType = NDArray<double>.create([6], DType.float64);

          expect(() => repeat(a, 2, out: outWrongShape), throwsArgumentError);
          expect(() => repeat(a, 2, out: outWrongDType), throwsArgumentError);
        });
      });
    });

    group('tile Tests', () {
      test('1D array, scalar reps', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          final t = tile(a, 3);
          expect(t.shape, [6]);
          expect(t.toList(), [1, 2, 1, 2, 1, 2]);
        });
      });

      test('1D array, list reps', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          final t = tile(a, [3]);
          expect(t.shape, [6]);
          expect(t.toList(), [1, 2, 1, 2, 1, 2]);
        });
      });

      test('2D array, reps of same rank', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final t = tile(a, [2, 3]);
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
          final t = tile(a, [2]); // treated as [1, 2]
          expect(t.shape, [2, 4]);
          expect(t.toList(), [1, 2, 1, 2, 3, 4, 3, 4]);
        });
      });

      test('2D array, reps with expanded rank (reps.length > rank)', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final t = tile(a, [2, 1, 2]); // a is promoted to [1, 2, 2]
          expect(t.shape, [2, 2, 4]);
          expect(t.toList(), [1, 2, 1, 2, 3, 4, 3, 4, 1, 2, 1, 2, 3, 4, 3, 4]);
        });
      });

      test('Edge case: reps with 0 (returns empty array)', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          final t = tile(a, 0);
          expect(t.shape, [0]);
          expect(t.toList(), <int>[]);
        });
      });

      test('Precondition checks', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          expect(() => tile(a, -1), throwsArgumentError);
          expect(() => tile(a, [1, -1]), throwsArgumentError);
          expect(() => tile(a, 'invalid'), throwsArgumentError);
        });
      });

      test('out parameter: correct shape/dtype', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          final out = NDArray<int>.create([4], DType.int32);
          final t = tile(a, 2, out: out);
          expect(identical(t, out), true);
          expect(t.toList(), [1, 2, 1, 2]);
        });
      });

      test('out parameter: incorrect shape/dtype', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2], [2], DType.int32);
          final outWrongShape = NDArray<int>.create([5], DType.int32);
          final outWrongDType = NDArray<double>.create([4], DType.float64);

          expect(() => tile(a, 2, out: outWrongShape), throwsArgumentError);
          expect(() => tile(a, 2, out: outWrongDType), throwsArgumentError);
        });
      });
    });
  });
}
