import "package:ndarray/ndarray.dart";
import "package:test/test.dart";

void main() {
  group("Standard Polynomial Operations - polyval, polyfit, roots", () {
    test("polyval evaluates 1D polynomials correctly", () {
      NDArray.scope(() {
        // p(x) = 3x^2 + 2x + 1
        final c = NDArray.fromList([3.0, 2.0, 1.0], [3], DType.float64);
        final x = NDArray.fromList([0.0, 0.5, 1.0], [3], DType.float64);
        final y = polyval(c, x);
        expect(y.shape, equals([3]));
        expect(y.getCell([0]), closeTo(1.0, 1e-6));
        expect(y.getCell([1]), closeTo(2.75, 1e-6));
        expect(y.getCell([2]), closeTo(6.0, 1e-6));
      });
    });

    test("polyval with single coefficient and mixed dtypes", () {
      NDArray.scope(() {
        final c = NDArray.fromList([5.0], [1], DType.float64);
        final x = NDArray.fromList([1, 2, 3], [3], DType.int64);
        final y = polyval(c, x);
        expect(y.shape, equals([3]));
        expect(y.getCell([0]), equals(5.0));
        expect(y.getCell([1]), equals(5.0));
        expect(y.getCell([2]), equals(5.0));
      });
    });

    test("polyval with out parameter", () {
      NDArray.scope(() {
        final c = NDArray.fromList(
          [1.0, -3.0, 2.0],
          [3],
          DType.float64,
        ); // x^2 - 3x + 2
        final x = NDArray.fromList([0.0, 1.0, 2.0], [3], DType.float64);
        final out = NDArray<double>.zeros([3], DType.float64);
        final res = polyval(c, x, out: out);
        expect(identical(res, out), isTrue);
        expect(out.getCell([0]), closeTo(2.0, 1e-6));
        expect(out.getCell([1]), closeTo(0.0, 1e-6));
        expect(out.getCell([2]), closeTo(0.0, 1e-6));
      });
    });

    test("polyfit fits linear and quadratic data accurately", () {
      NDArray.scope(() {
        final x = NDArray.fromList([0.0, 1.0, 2.0, 3.0], [4], DType.float64);
        final y = NDArray.fromList(
          [1.0, 3.0, 5.0, 7.0],
          [4],
          DType.float64,
        ); // y = 2x + 1
        final p = polyfit(x, y, 1);
        expect(p.shape, equals([2]));
        expect(p.getCell([0]), closeTo(2.0, 1e-5));
        expect(p.getCell([1]), closeTo(1.0, 1e-5));
      });
    });

    test("polyfit with weights", () {
      NDArray.scope(() {
        final x = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final y = NDArray.fromList([3.0, 5.0, 7.0, 9.0], [4], DType.float64);
        final w = NDArray.fromList([1.0, 1.0, 1.0, 1.0], [4], DType.float64);
        final p = polyfit(x, y, 1, w: w);
        expect(p.getCell([0]), closeTo(2.0, 1e-5));
        expect(p.getCell([1]), closeTo(1.0, 1e-5));
      });
    });

    test("quadratic polyfit", () {
      NDArray.scope(() {
        final x = NDArray.fromList([-1.0, 0.0, 1.0, 2.0], [4], DType.float64);
        final y = NDArray.fromList([6.0, 2.0, 0.0, 0.0], [4], DType.float64);
        final p = polyfit(x, y, 2);
        expect(p.shape, equals([3]));
        expect(p.getCell([0]), closeTo(1.0, 1e-5));
        expect(p.getCell([1]), closeTo(-3.0, 1e-5));
        expect(p.getCell([2]), closeTo(2.0, 1e-5));
      });
    });

    test("roots finds real polynomial roots", () {
      NDArray.scope(() {
        // 2x^2 - 4x - 6 = 0 => roots -1 and 3
        final p = NDArray.fromList([2.0, -4.0, -6.0], [3], DType.float64);
        final r = roots(p);
        expect(r.shape, equals([2]));
        final r1 = r.getCell([0]).real;
        final r2 = r.getCell([1]).real;
        final vals = [r1, r2]..sort();
        expect(vals[0], closeTo(-1.0, 1e-5));
        expect(vals[1], closeTo(3.0, 1e-5));
      });
    });

    test("roots finds linear root", () {
      NDArray.scope(() {
        final p = NDArray.fromList([3.0, -6.0], [2], DType.float64);
        final r = roots(p);
        expect(r.shape, equals([1]));
        expect(r.getCell([0]).real, closeTo(2.0, 1e-5));
      });
    });

    test("roots finds complex roots for x^2 + 1", () {
      NDArray.scope(() {
        final p = NDArray.fromList([1.0, 0.0, 1.0], [3], DType.float64);
        final r = roots(p);
        expect(r.shape, equals([2]));
        final i1 = r.getCell([0]).imag.abs();
        final i2 = r.getCell([1]).imag.abs();
        expect(i1, closeTo(1.0, 1e-5));
        expect(i2, closeTo(1.0, 1e-5));
      });
    });

    test("error conditions for polyval, polyfit, roots", () {
      NDArray.scope(() {
        final empty = NDArray.zeros([0], DType.float64);
        final mat = NDArray.zeros([2, 2], DType.float64);
        final vec = NDArray.fromList([1.0, 2.0], [2], DType.float64);

        expect(() => polyval(empty, vec), throwsArgumentError);
        expect(() => polyval(mat, vec), throwsArgumentError);
        expect(() => polyfit(vec, vec, -1), throwsArgumentError);
        expect(
          () => polyfit(vec, NDArray.fromList([1.0], [1], DType.float64), 1),
          throwsArgumentError,
        );
        expect(() => roots(mat), throwsArgumentError);
      });
    });
  });

  group("Orthogonal Polynomial Bases - Chebyshev, Legendre, Hermite", () {
    test("chebval evaluation and chebroots", () {
      NDArray.scope(() {
        final c = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final x = NDArray.fromList([0.0, 0.5, 1.0], [3], DType.float64);
        final y = chebval(c, x);
        expect(y.getCell([0]), closeTo(-2.0, 1e-5));
        expect(y.getCell([1]), closeTo(0.5, 1e-5));
        expect(y.getCell([2]), closeTo(6.0, 1e-5));

        final c2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
        final r = chebroots(c2);
        final r1 = r.getCell([0]).real.abs();
        final r2 = r.getCell([1]).real.abs();
        final rVals = [r1, r2]..sort();
        expect(rVals[0], closeTo(0.707106, 1e-4));
      });
    });

    test("legval evaluation and legroots", () {
      NDArray.scope(() {
        final c = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final x = NDArray.fromList([0.0, 0.5, 1.0], [3], DType.float64);
        final y = legval(c, x);
        expect(y.getCell([0]), closeTo(-0.5, 1e-5));
        expect(y.getCell([1]), closeTo(1.625, 1e-5));
        expect(y.getCell([2]), closeTo(6.0, 1e-5));

        final c2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
        final r = legroots(c2);
        final r1 = r.getCell([0]).real.abs();
        final r2 = r.getCell([1]).real.abs();
        final rVals = [r1, r2]..sort();
        expect(rVals[0], closeTo(0.57735, 1e-4));
      });
    });

    test("hermval evaluation and hermroots", () {
      NDArray.scope(() {
        final c = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final x = NDArray.fromList([0.0, 0.5, 1.0], [3], DType.float64);
        final y = hermval(c, x);
        expect(y.getCell([0]), closeTo(-5.0, 1e-5));
        expect(y.getCell([1]), closeTo(0.0, 1e-5));
        expect(y.getCell([2]), closeTo(11.0, 1e-5));

        final c2 = NDArray.fromList([0.0, 0.0, 1.0], [3], DType.float64);
        final r = hermroots(c2);
        final r1 = r.getCell([0]).real.abs();
        final r2 = r.getCell([1]).real.abs();
        final rVals = [r1, r2]..sort();
        expect(rVals[0], closeTo(0.707106, 1e-4));
      });
    });
  });
}
