import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Shaping & Meshes Tests', () {
    group('asStrided Tests', () {
      test('basic 1D to 2D asStrided', () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);
          final view = asStrided(a, shape: [2, 2], strides: [2, 1]);

          expect(view.shape, [2, 2]);
          expect(view.toList(), [1, 2, 3, 4]);

          // Mutating view must affect original
          view.setCell([0, 1], 99);
          expect(a.getCell([1]), 99);
        });
      });

      test('asStrided keeps strides if null', () {
        NDArray.scope(() {
          final a = NDArray.fromList([10, 20, 30], [3], DType.int32);
          final view = asStrided(a);
          expect(view.shape, [3]);
          expect(view.strides, [1]);
          expect(view.toList(), [10, 20, 30]);
        });
      });

      test('asStrided invalid shape/strides throws', () {
        NDArray.scope(() {
          final a = NDArray.create([4], DType.int32);
          expect(
            () => asStrided(a, shape: [2, 2], strides: [1]),
            throwsArgumentError,
          );
        });
      });
    });

    group('GridRange Tests', () {
      test('GridRange standard step', () {
        final r = GridRange(0, 5, step: 2);
        expect(r.start, 0.0);
        expect(r.stop, 5.0);
        expect(r.step, 2.0);
        expect(r.numPoints, isNull);
      });

      test('GridRange numPoints', () {
        final r = GridRange(0, 5, numPoints: 5);
        expect(r.start, 0.0);
        expect(r.stop, 5.0);
        expect(r.numPoints, 5);
      });

      test('GridRange.numpy standard step', () {
        final r = GridRange.numpy(0, 5, 2);
        expect(r.start, 0.0);
        expect(r.stop, 5.0);
        expect(r.step, 2.0);
        expect(r.numPoints, isNull);
      });

      test('GridRange.numpy complex step', () {
        final r = GridRange.numpy(0, 5, Complex(0, 5));
        expect(r.start, 0.0);
        expect(r.stop, 5.0);
        expect(r.numPoints, 5);
      });
    });

    group('ogrid Tests', () {
      test('basic 2D ogrid', () {
        NDArray.scope(() {
          final grid = ogrid([
            GridRange(0, 3), // 0, 1, 2
            GridRange(0, 2), // 0, 1
          ]);

          expect(grid.length, 2);
          expect(grid[0].shape, [3, 1]);
          expect(grid[0].toList(), [0.0, 1.0, 2.0]);

          expect(grid[1].shape, [1, 2]);
          expect(grid[1].toList(), [0.0, 1.0]);
        });
      });

      test('3D ogrid with numPoints and step', () {
        NDArray.scope(() {
          final grid = ogrid([
            GridRange(0, 2, numPoints: 3), // 0, 1, 2 (inclusive)
            GridRange(0, 2, step: 1), // 0, 1 (exclusive)
            GridRange.numpy(0, 1, Complex(0, 2)), // 0, 1 (inclusive, 2 points)
          ]);

          expect(grid.length, 3);
          expect(grid[0].shape, [3, 1, 1]);
          expect(grid[0].toList(), [0.0, 1.0, 2.0]);

          expect(grid[1].shape, [1, 2, 1]);
          expect(grid[1].toList(), [0.0, 1.0]);

          expect(grid[2].shape, [1, 1, 2]);
          expect(grid[2].toList(), [0.0, 1.0]);
        });
      });

      test('empty ogrid throws', () {
        expect(() => ogrid([]), throwsArgumentError);
      });
    });

    group('mgrid Tests', () {
      test('basic 2D mgrid', () {
        NDArray.scope(() {
          final grid = mgrid([GridRange(0, 3), GridRange(0, 2)]);

          expect(grid.shape, [2, 3, 2]);
          expect(grid.toList(), [
            0.0, 0.0, 1.0, 1.0, 2.0, 2.0, // grid 0
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, // grid 1
          ]);
        });
      });

      test('3D mgrid', () {
        NDArray.scope(() {
          final grid = mgrid([
            GridRange(0, 2, step: 1),
            GridRange(0, 2, step: 1),
            GridRange(0, 2, step: 1),
          ]);

          expect(grid.shape, [3, 2, 2, 2]);
          expect(grid.slice([Index(0)]).toList(), [
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
          ]);
        });
      });

      test('empty mgrid throws', () {
        expect(() => mgrid([]), throwsArgumentError);
      });
    });
  });
}
