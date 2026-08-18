import 'package:test/test.dart';
import 'package:symbolic_dart/symbolic_dart.dart';

void main() {
  group('Expr constants and elementary constructors', () {
    test('zero, one, minusOne, i, pi, e', () {
      expect(Expr.zero.isZero, isTrue);
      expect(Expr.one.asDouble, closeTo(1.0, 1e-12));
      expect(Expr.minusOne.isNegative, isTrue);
      expect(Expr.i.isComplex, isTrue);
      expect(Expr.pi.asDouble, closeTo(3.141592653589793, 1e-12));
      expect(Expr.e.asDouble, closeTo(2.718281828459045, 1e-12));
      expect(Expr.eulerGamma.asDouble, closeTo(0.5772156649, 1e-6));
      expect(Expr.catalan.asDouble, closeTo(0.9159655941, 1e-6));
      expect(Expr.goldenRatio.asDouble, closeTo(1.6180339887, 1e-6));
      expect(Expr.infinity.toString(), contains('oo'));
      expect(
        Expr.modInverse(3, 7).asDouble,
        equals(5.0),
      ); // 3 * 5 = 15 = 1 mod 7
    });

    test('rational constructors', () {
      final half = Rational(1, 2);
      expect(half.asDouble, closeTo(0.5, 1e-12));

      expect(() => Rational(1, 0), throwsArgumentError);
    });

    test('symbol constructors and parsing', () {
      final x = Symbol('x');
      expect(x.toString(), 'x');

      final parsed = Expr.parse('sin(x) + 2*x');
      expect(parsed.hasSymbol(x), isTrue);
    });
  });

  group('Expr arithmetic operators', () {
    test('addition, subtraction, multiplication, division', () {
      final x = Symbol('x');
      final expr = (x * 2) + 5 - x;
      final evaluated = expr.subs({x: 3.0}).asDouble;
      // 3*2 + 5 - 3 = 8
      expect(evaluated, closeTo(8.0, 1e-12));
    });

    test('exponentiation using ^ operator', () {
      final x = Symbol('x');
      final square = x ^ 2;
      expect(square.subs({x: 4.0}).asDouble, closeTo(16.0, 1e-12));
    });
  });

  group('Calculus and symbolic manipulation', () {
    test('differentiation (diff)', () {
      final x = Symbol('x');
      // f(x) = x^3 + 2*x^2 + 5*x
      final f = (x ^ 3) + ((x ^ 2) * 2) + (x * 5);
      final df = f.diff(x);
      // df/dx at x=2: 3(4) + 4(2) + 5 = 12 + 8 + 5 = 25
      expect(df.subs({x: 2.0}).asDouble, closeTo(25.0, 1e-12));
    });

    test('differentiation of elementary functions', () {
      final x = Symbol('x');
      final f = sin(x);
      final df = f.diff(x);
      // derivative of sin(x) is cos(x)
      expect(df.subs({x: 0.0}).asDouble, closeTo(1.0, 1e-12));
    });

    test('polynomial expansion', () {
      final x = Symbol('x');
      final f = (x + 2) ^ 2;
      final expanded = f.expand();
      // (x+2)^2 = x^2 + 4x + 4
      expect(expanded.subs({x: 10.0}).asDouble, closeTo(144.0, 1e-12));
    });

    test('args and freeSymbols properties', () {
      final x = Symbol('x');
      final y = Symbol('y');
      final f = x + y + 5;
      expect(f.freeSymbols.length, 2);
    });

    test('toLatex, toCCode, toJSCode, toMathML format printers', () {
      final x = Symbol('x');
      final f = sin(x ^ 2);
      expect(f.toLatex(), isNotEmpty);
      expect(f.toCCode(), isNotEmpty);
      expect(f.toJSCode(), isNotEmpty);
      expect(f.toMathML(), isNotEmpty);
    });

    test('trig, hyperbolic, and special functions chaining', () {
      expect(Integer(9).sqrt().asDouble, closeTo(3.0, 1e-12));
      expect(Integer(27).cbrt().asDouble, closeTo(3.0, 1e-12));
      expect(kroneckerDelta(1, 1).asDouble, equals(1.0));
      expect(kroneckerDelta(1, 2).asDouble, equals(0.0));
      expect(gcd(12, 18).asDouble, equals(6.0));
      expect(lcm(12, 18).asDouble, equals(36.0));
      expect(Real(3.7).floor().asDouble, equals(3.0));
      expect(Real(3.7).ceil().asDouble, equals(4.0));
    });

    test('asNumerDenom rational separation', () {
      final x = Symbol('x');
      final frac = (x ^ 2) / (x + 1);
      final parts = frac.asNumerDenom();
      expect(parts.numerator.eq(x ^ 2), isTrue);
      expect(parts.denominator.eq(x + 1), isTrue);
    });

    test('exact polynomial solver solvePoly', () {
      final x = Symbol('x');
      // x^2 - 9 = 0
      final roots = Expr.solvePoly((x ^ 2) - 9, x);
      expect(roots.length, equals(2));
      final rootVals = roots.map((r) => r.asDouble).toSet();
      expect(rootVals, equals({-3.0, 3.0}));
    });

    test('exact linear system solver solveLinearSystem', () {
      final x = Symbol('x');
      final y = Symbol('y');
      // 2*x + y - 5 = 0
      // x + 3*y - 5 = 0
      // Solution: x = 2, y = 1
      final sol = Expr.solveLinearSystem(
        [(Integer(2) * x) + y - 5, x + (Integer(3) * y) - 5],
        [x, y],
      );
      expect(sol.length, equals(2));
      expect(sol[0].asDouble, closeTo(2.0, 1e-12));
      expect(sol[1].asDouble, closeTo(1.0, 1e-12));
    });

    test('all remaining top-level trig, hyperbolic, and special functions', () {
      final x = Symbol('x');
      expect(tan(0).asDouble, closeTo(0.0, 1e-12));
      expect(asin(0).asDouble, closeTo(0.0, 1e-12));
      expect(acos(1).asDouble, closeTo(0.0, 1e-12));
      expect(atan(0).asDouble, closeTo(0.0, 1e-12));
      expect(atan2(0, 1).asDouble, closeTo(0.0, 1e-12));
      expect(csc(Expr.pi / 2).asDouble, closeTo(1.0, 1e-12));
      expect(sec(0).asDouble, closeTo(1.0, 1e-12));
      expect(cot(Expr.pi / 2).asDouble, closeTo(0.0, 1e-12));

      expect(sinh(0).asDouble, closeTo(0.0, 1e-12));
      expect(cosh(0).asDouble, closeTo(1.0, 1e-12));
      expect(tanh(0).asDouble, closeTo(0.0, 1e-12));
      expect(asinh(0).asDouble, closeTo(0.0, 1e-12));
      expect(acosh(1).asDouble, closeTo(0.0, 1e-12));
      expect(atanh(0).asDouble, closeTo(0.0, 1e-12));

      expect(floor(3.8).asDouble, equals(3.0));
      expect(ceil(3.2).asDouble, equals(4.0));
      expect(erf(0).asDouble, closeTo(0.0, 1e-12));
      expect(erfc(0).asDouble, closeTo(1.0, 1e-12));
      expect(gamma(1).asDouble, closeTo(1.0, 1e-12));
      expect(lambertw(0).asDouble, closeTo(0.0, 1e-12));
      expect(zeta(2).asDouble, closeTo(1.644934, 1e-5)); // pi^2/6
    });

    test('special constants and BigInt / String constructors', () {
      expect(Expr.negInfinity.toString(), contains('-oo'));
      expect(Expr.complexInfinity.toString(), contains('zoo'));
      expect(Expr.nan.toString(), contains('nan'));

      final bigIntExpr = Expr.fromObject(BigInt.from(123456789));
      expect(bigIntExpr.asDouble, closeTo(123456789.0, 1e-6));

      final strExpr = Expr.fromObject('y');
      expect(strExpr.toString(), equals('y'));

      expect(() => Expr.parse('invalid ((( formula'), throwsFormatException);
      expect(() => Expr.fromObject(Object()), throwsArgumentError);
    });

    test('unary negation and binary arithmetic operators', () {
      final x = Symbol('x');
      final neg = -x;
      expect(neg.subs({x: 5.0}).asDouble, closeTo(-5.0, 1e-12));

      final a = Integer(10);
      final b = Integer(2);
      expect((a + b).asDouble, equals(12.0));
      expect((a - b).asDouble, equals(8.0));
      expect((a * b).asDouble, equals(20.0));
      expect((a / b).asDouble, equals(5.0));
    });
  });
}
