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
  });
}
