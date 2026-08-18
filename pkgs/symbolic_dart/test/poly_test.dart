import 'package:test/test.dart';
import 'package:symbolic_dart/symbolic_dart.dart';

void main() {
  group('FlintRationalPoly exact polynomial operations', () {
    test('degree and length', () {
      final p = FlintRationalPoly.fromIntCoefficients([
        1,
        2,
        3,
      ]); // 3x^2 + 2x + 1
      expect(p.degree, 2);
      expect(p.length, 3);
    });

    test('addition, subtraction, multiplication', () {
      final p1 = FlintRationalPoly.fromIntCoefficients([1, 2]); // 2x + 1
      final p2 = FlintRationalPoly.fromIntCoefficients([-1, 3]); // 3x - 1

      final sum = p1 + p2; // 5x
      expect(sum.degree, 1);

      final prod = p1 * p2; // (2x+1)(3x-1) = 6x^2 + x - 1
      expect(prod.degree, 2);
    });

    test('exact GCD over Q[x]', () {
      // P1 = (x - 2) * (x + 3) = x^2 + x - 6
      final p1 = FlintRationalPoly.fromIntCoefficients([-6, 1, 1]);
      // P2 = (x - 2) * (x - 5) = x^2 - 7x + 10
      final p2 = FlintRationalPoly.fromIntCoefficients([10, -7, 1]);

      final gcd = p1.gcd(p2);
      // GCD monic should be x - 2 = [-2, 1]
      expect(gcd.degree, 1);
    });

    test('exact polynomial factorization over Q[x]', () {
      // P(x) = (x^2 - 4) * (2x + 3)^2
      final p1 = FlintRationalPoly.fromIntCoefficients([-4, 0, 1]); // x^2 - 4
      final p2 = FlintRationalPoly.fromIntCoefficients([3, 2]); // 2x + 3
      final p = p1 * p2 * p2;

      final fac = p.factor();
      expect(fac.factors.length, greaterThanOrEqualTo(2));
    });

    test('derivative and integral', () {
      final p = FlintRationalPoly.fromIntCoefficients([
        5,
        2,
        3,
      ]); // 3x^2 + 2x + 5
      final dp = p.derivative(); // 6x + 2
      expect(dp.degree, 1);
    });

    test('toExpr conversion', () {
      final x = Symbol('x');
      final p = FlintRationalPoly.fromIntCoefficients([1, 2, 3]);
      final e = p.toExpr(x);
      expect(
        e.subs({x: 2.0}).asDouble,
        closeTo(17.0, 1e-12),
      ); // 3(4) + 2(2) + 1 = 17
    });

    test('pow exact polynomial exponentiation', () {
      // P(x) = x + 1
      final p = FlintRationalPoly.fromIntCoefficients([1, 1]);
      final squared = p ^ 2; // (x + 1)^2 = x^2 + 2x + 1
      expect(squared.degree, equals(2));
      expect(squared.evaluate(2.0), closeTo(9.0, 1e-12));
    });

    test('compose exact polynomial composition', () {
      // P(x) = x^2
      final p = FlintRationalPoly.fromIntCoefficients([0, 0, 1]);
      // Q(x) = 2x + 1
      final q = FlintRationalPoly.fromIntCoefficients([1, 2]);
      // P(Q(x)) = (2x + 1)^2 = 4x^2 + 4x + 1
      final comp = p.compose(q);
      expect(comp.degree, equals(2));
      expect(comp.evaluate(1.0), closeTo(9.0, 1e-12));
    });

    test('evaluate and evaluateRational', () {
      // P(x) = 3x^2 + 2x + 1
      final p = FlintRationalPoly.fromIntCoefficients([1, 2, 3]);
      expect(p.evaluate(2.0), closeTo(17.0, 1e-12));

      // Evaluate at 1/2: 3*(1/4) + 2*(1/2) + 1 = 3/4 + 1 + 1 = 11/4
      final ratRes = p.evaluateRational(1, 2);
      expect(ratRes.numerator, equals(BigInt.from(11)));
      expect(ratRes.denominator, equals(BigInt.from(4)));
    });

    test('orthogonal polynomials legendre and laguerre', () {
      // P_2(x) = (3x^2 - 1) / 2 = -1/2 + 3/2 x^2
      final p2 = FlintRationalPoly.legendre(2);
      expect(p2.degree, equals(2));
      expect(p2.evaluate(1.0), closeTo(1.0, 1e-12));

      // L_2(x)
      final l2 = FlintRationalPoly.laguerre(2);
      expect(l2.degree, equals(2));
    });

    test('resultant, isMonic, isSquareFree', () {
      // P(x) = x^2 - 1, Q(x) = x - 1. Common root x=1 => Resultant is 0.
      final p = FlintRationalPoly.fromIntCoefficients([-1, 0, 1]);
      final q = FlintRationalPoly.fromIntCoefficients([-1, 1]);
      expect(p.isMonic, isTrue);
      expect(p.isSquareFree, isTrue);
      final res = p.resultant(q);
      expect(res.numerator, equals(BigInt.zero));

      final nonMonic = FlintRationalPoly.fromIntCoefficients([2, 4]);
      expect(nonMonic.isMonic, isFalse);
      final monic = nonMonic.makeMonic();
      expect(monic.isMonic, isTrue);
    });

    test('formal power series expSeries and sinSeries', () {
      // P(x) = x
      final p = FlintRationalPoly.fromIntCoefficients([0, 1]);
      // exp(x) mod x^4 = 1 + x + x^2/2 + x^3/6
      final expS = p.expSeries(4);
      expect(expS.degree, equals(3));
      // Coeff of x^3 is 1/6
      final ratCoeff = expS.evaluateRational(1); // 1 + 1 + 1/2 + 1/6 = 8/3
      expect(ratCoeff.numerator, equals(BigInt.from(8)));
      expect(ratCoeff.denominator, equals(BigInt.from(3)));

      final pOne = FlintRationalPoly.fromIntCoefficients([1, 1]);
      final logS = pOne.logSeries(3); // log(1+x) = x - x^2/2
      expect(logS.degree, equals(2));

      final cosS = p.cosSeries(3);
      expect(cosS.degree, equals(2));

      final sinS = p.sinSeries(3);
      expect(sinS.degree, equals(1)); // sin(x) mod x^3 = x
    });

    test('constructors zero, one, monomial, and coefficients', () {
      final z = FlintRationalPoly.zero();
      expect(z.degree, equals(-1));
      expect(z.length, equals(0));

      final o = FlintRationalPoly.one();
      expect(o.degree, equals(0));
      expect(o.evaluate(5.0), equals(1.0));

      final m = FlintRationalPoly.monomial(3, coefficient: 4);
      expect(m.degree, equals(3));
      expect(m.getCoefficientAsDouble(3), equals(4.0));
      expect(m.getCoefficientAsDouble(100), equals(0.0));

      expect(() => FlintRationalPoly.monomial(-1), throwsArgumentError);
      expect(() => FlintRationalPoly.legendre(-1), throwsArgumentError);
      expect(() => FlintRationalPoly.laguerre(-1), throwsArgumentError);
    });

    test('poly addition, subtraction, division, remainder, divmod', () {
      // P(x) = x^2 - 1 = (x - 1)(x + 1)
      final p = FlintRationalPoly.fromIntCoefficients([-1, 0, 1]);
      final q = FlintRationalPoly.fromIntCoefficients([1, 1]); // x + 1

      final sum = p + q; // x^2 + x
      expect(sum.evaluate(2.0), equals(6.0));

      final diff = p - q; // x^2 - x - 2
      expect(diff.evaluate(2.0), equals(0.0));

      final div = p / q; // x - 1
      expect(div.evaluate(3.0), equals(2.0));

      final rem = p % q; // 0
      expect(rem.degree, equals(-1));

      final dm = p.divmod(q);
      expect(dm.quotient.evaluate(3.0), equals(2.0));
      expect(dm.remainder.degree, equals(-1));

      expect(() => p / FlintRationalPoly.zero(), throwsStateError);
      expect(() => p % FlintRationalPoly.zero(), throwsStateError);
      expect(() => p.divmod(FlintRationalPoly.zero()), throwsStateError);
    });

    test('poly derivative, integral, and gcd', () {
      // P(x) = x^3
      final p = FlintRationalPoly.monomial(3);
      final der = p.derivative(); // 3x^2
      expect(der.getCoefficientAsDouble(2), equals(3.0));

      final integ = der.integral(); // x^3
      expect(integ.getCoefficientAsDouble(3), equals(1.0));

      // GCD of x^2 - 1 and x - 1 is x - 1 (monic)
      final p1 = FlintRationalPoly.fromIntCoefficients([-1, 0, 1]);
      final p2 = FlintRationalPoly.fromIntCoefficients([-1, 1]);
      final g = p1.gcd(p2);
      expect(g.evaluate(2.0), equals(1.0)); // 2 - 1 = 1
    });
  });
}
