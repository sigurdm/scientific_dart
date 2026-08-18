import 'package:symbolic_dart/symbolic_dart.dart';

void main() {
  // P(x) = (x^2 - 4) * (2x + 3)^2
  // Let's create P1 = x^2 - 4 and P2 = 2x + 3
  final p1 = FlintRationalPoly.fromIntCoefficients([-4, 0, 1]);
  final p2 = FlintRationalPoly.fromIntCoefficients([3, 2]);

  final p = p1 * (p2 * p2);
  print('Expanded polynomial P(x) = $p');
  print('Degree                   = ${p.degree}');

  // Exact polynomial derivative dP(x)/dx
  print('Derivative dP/dx         = ${p.derivative()}');

  // Exact polynomial GCD with (x^2 - 4)
  final gcd = p.gcd(p1);
  print('GCD(P, x^2 - 4)          = $gcd');

  // Exact polynomial factorization over Q[x]
  final fac = p.factor();
  print(
    'Content                  = ${fac.content.numerator}/${fac.content.denominator}',
  );
  for (final item in fac.factors) {
    print('  Factor (${item.factor}) ^ ${item.exponent}');
  }
}
