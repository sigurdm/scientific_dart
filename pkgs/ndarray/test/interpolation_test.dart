import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';

void main() {
  group('1D Linear Interpolation (interp)', () {
    test('Basic interpolation', () {
      final xp = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final fp = NDArray.fromList([10.0, 20.0, 40.0], [3], DType.float64);
      final x = NDArray.fromList([1.5, 2.5], [2], DType.float64);

      final res = interp(x, xp, fp);

      expect(res.shape, [2]);
      expect(res.dtype, DType.float64);
      expect(res.getCell([0]), closeTo(15.0, 1e-9));
      expect(res.getCell([1]), closeTo(30.0, 1e-9));

      xp.dispose();
      fp.dispose();
      x.dispose();
      res.dispose();
    });

    test('Boundary values (default left/right)', () {
      final xp = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final fp = NDArray.fromList([10.0, 20.0, 40.0], [3], DType.float64);
      final x = NDArray.fromList([0.0, 4.0], [2], DType.float64);

      final res = interp(x, xp, fp);

      expect(res.getCell([0]), closeTo(10.0, 1e-9)); // fp[0]
      expect(res.getCell([1]), closeTo(40.0, 1e-9)); // fp[last]

      xp.dispose();
      fp.dispose();
      x.dispose();
      res.dispose();
    });

    test('Boundary values (custom left/right)', () {
      final xp = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final fp = NDArray.fromList([10.0, 20.0, 40.0], [3], DType.float64);
      final x = NDArray.fromList([0.0, 4.0], [2], DType.float64);

      final res = interp(x, xp, fp, left: -1.0, right: -2.0);

      expect(res.getCell([0]), closeTo(-1.0, 1e-9));
      expect(res.getCell([1]), closeTo(-2.0, 1e-9));

      xp.dispose();
      fp.dispose();
      x.dispose();
      res.dispose();
    });

    test('Exact data points', () {
      final xp = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final fp = NDArray.fromList([10.0, 20.0, 40.0], [3], DType.float64);
      final x = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);

      final res = interp(x, xp, fp);

      expect(res.getCell([0]), closeTo(10.0, 1e-9));
      expect(res.getCell([1]), closeTo(20.0, 1e-9));
      expect(res.getCell([2]), closeTo(40.0, 1e-9));

      xp.dispose();
      fp.dispose();
      x.dispose();
      res.dispose();
    });

    test('Multi-dimensional x', () {
      final xp = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final fp = NDArray.fromList([10.0, 20.0, 40.0], [3], DType.float64);
      final x = NDArray.fromList([1.5, 2.5, 0.0, 4.0], [2, 2], DType.float64);

      final res = interp(x, xp, fp, left: -1.0, right: -2.0);

      expect(res.shape, [2, 2]);
      expect(res.getCell([0, 0]), closeTo(15.0, 1e-9));
      expect(res.getCell([0, 1]), closeTo(30.0, 1e-9));
      expect(res.getCell([1, 0]), closeTo(-1.0, 1e-9));
      expect(res.getCell([1, 1]), closeTo(-2.0, 1e-9));

      xp.dispose();
      fp.dispose();
      x.dispose();
      res.dispose();
    });

    test('Integer input promotion', () {
      final xp = NDArray.fromList([1, 2, 3], [3], DType.int32);
      final fp = NDArray.fromList([10, 20, 40], [3], DType.int32);
      final x = NDArray.fromList([1.5, 2.5], [2], DType.float64);

      final res = interp(x, xp, fp);

      expect(res.dtype, DType.float64);
      expect(res.getCell([0]), closeTo(15.0, 1e-9));
      expect(res.getCell([1]), closeTo(30.0, 1e-9));

      xp.dispose();
      fp.dispose();
      x.dispose();
      res.dispose();
    });

    test('Single point xp/fp', () {
      final xp = NDArray.fromList([2.0], [1], DType.float64);
      final fp = NDArray.fromList([20.0], [1], DType.float64);
      final x = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);

      final res = interp(x, xp, fp);

      expect(res.getCell([0]), closeTo(20.0, 1e-9)); // left fallback to fp[0]
      expect(res.getCell([1]), closeTo(20.0, 1e-9)); // exact
      expect(res.getCell([2]), closeTo(20.0, 1e-9)); // right fallback to fp[0]

      final resCustom = interp(x, xp, fp, left: -1.0, right: -2.0);
      expect(resCustom.getCell([0]), closeTo(-1.0, 1e-9));
      expect(resCustom.getCell([1]), closeTo(20.0, 1e-9));
      expect(resCustom.getCell([2]), closeTo(-2.0, 1e-9));

      xp.dispose();
      fp.dispose();
      x.dispose();
      res.dispose();
      resCustom.dispose();
    });

    test('Unsorted xp throws ArgumentError', () {
      final xp = NDArray.fromList([2.0, 1.0, 3.0], [3], DType.float64);
      final fp = NDArray.fromList([20.0, 10.0, 40.0], [3], DType.float64);
      final x = NDArray.fromList([1.5], [1], DType.float64);

      expect(() => interp(x, xp, fp), throwsArgumentError);

      xp.dispose();
      fp.dispose();
      x.dispose();
    });

    test('Duplicate xp throws ArgumentError (strictly increasing)', () {
      final xp = NDArray.fromList([1.0, 2.0, 2.0, 3.0], [4], DType.float64);
      final fp = NDArray.fromList([10.0, 20.0, 30.0, 40.0], [4], DType.float64);
      final x = NDArray.fromList([1.5], [1], DType.float64);

      expect(() => interp(x, xp, fp), throwsArgumentError);

      xp.dispose();
      fp.dispose();
      x.dispose();
    });

    test('Mismatched xp and fp lengths throws ArgumentError', () {
      final xp = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final fp = NDArray.fromList([10.0, 20.0], [2], DType.float64);
      final x = NDArray.fromList([1.5], [1], DType.float64);

      expect(() => interp(x, xp, fp), throwsArgumentError);

      xp.dispose();
      fp.dispose();
      x.dispose();
    });

    test('Non-1D xp or fp throws ArgumentError', () {
      final xp = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
      final fp = NDArray.fromList([10.0, 20.0, 30.0, 40.0], [4], DType.float64);
      final x = NDArray.fromList([1.5], [1], DType.float64);

      expect(() => interp(x, xp, fp), throwsArgumentError);

      final xp2 = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
      final fp2 = NDArray.fromList(
        [10.0, 20.0, 30.0, 40.0],
        [2, 2],
        DType.float64,
      );

      expect(() => interp(x, xp2, fp2), throwsArgumentError);

      xp.dispose();
      fp.dispose();
      x.dispose();
      xp2.dispose();
      fp2.dispose();
    });

    test('Empty xp throws ArgumentError', () {
      final xp = NDArray.fromList(<double>[], [0], DType.float64);
      final fp = NDArray.fromList(<double>[], [0], DType.float64);
      final x = NDArray.fromList([1.5], [1], DType.float64);

      expect(() => interp(x, xp, fp), throwsArgumentError);

      xp.dispose();
      fp.dispose();
      x.dispose();
    });

    test('Strided inputs (views)', () {
      final xpFull = NDArray.fromList(
        [1.0, 99.0, 2.0, 99.0, 3.0],
        [5],
        DType.float64,
      );
      final xp = xpFull.slice([
        Slice(start: 0, stop: 5, step: 2),
      ]); // [1.0, 2.0, 3.0]
      expect(xp.isContiguous, false);
      expect(xp.shape, [3]);

      final fpFull = NDArray.fromList(
        [10.0, 99.0, 20.0, 99.0, 40.0],
        [5],
        DType.float64,
      );
      final fp = fpFull.slice([
        Slice(start: 0, stop: 5, step: 2),
      ]); // [10.0, 20.0, 40.0]
      expect(fp.isContiguous, false);
      expect(fp.shape, [3]);

      final xFull = NDArray.fromList(
        [1.5, 99.0, 2.5, 99.0],
        [4],
        DType.float64,
      );
      final x = xFull.slice([Slice(start: 0, stop: 4, step: 2)]); // [1.5, 2.5]
      expect(x.isContiguous, false);
      expect(x.shape, [2]);

      final res = interp(x, xp, fp);

      expect(res.shape, [2]);
      expect(res.getCell([0]), closeTo(15.0, 1e-9));
      expect(res.getCell([1]), closeTo(30.0, 1e-9));

      xpFull.dispose();
      fpFull.dispose();
      xFull.dispose();
      res.dispose();
    });

    test('0D x (scalar) contiguous', () {
      final xp = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final fp = NDArray.fromList([10.0, 20.0, 40.0], [3], DType.float64);
      final x = NDArray.fromList([1.5], [], DType.float64);

      final res = interp(x, xp, fp);

      expect(res.shape, []);
      expect(res.getCell([]), closeTo(15.0, 1e-9));

      xp.dispose();
      fp.dispose();
      x.dispose();
      res.dispose();
    });

    test('0D x (scalar) strided xp/fp', () {
      final xpFull = NDArray.fromList(
        [1.0, 99.0, 2.0, 99.0, 3.0],
        [5],
        DType.float64,
      );
      final xp = xpFull.slice([
        Slice(start: 0, stop: 5, step: 2),
      ]); // [1.0, 2.0, 3.0]
      final fpFull = NDArray.fromList(
        [10.0, 99.0, 20.0, 99.0, 40.0],
        [5],
        DType.float64,
      );
      final fp = fpFull.slice([
        Slice(start: 0, stop: 5, step: 2),
      ]); // [10.0, 20.0, 40.0]

      final x = NDArray.fromList([1.5], [], DType.float64);

      final res = interp(x, xp, fp);

      expect(res.shape, []);
      expect(res.getCell([]), closeTo(15.0, 1e-9));

      xpFull.dispose();
      fpFull.dispose();
      x.dispose();
      res.dispose();
    });
  });
}
