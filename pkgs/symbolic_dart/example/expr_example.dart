import 'package:symbolic_dart/symbolic_dart.dart';

void main() {
  final x = Symbol('x');
  final y = Symbol('y');

  // Build an expression f(x, y) = sin(x^2) + 2*y
  final f = sin(x ^ 2) + (Integer(2) * y);
  print('Expression f(x, y) = $f');
  print('LaTeX format       = ${f.toLatex()}');

  // Compute symbolic derivative with respect to x: df/dx = 2*x*cos(x^2)
  final dfDx = f.diff(x);
  print('Derivative df/dx   = $dfDx');

  // Substitute values: x = 2.0, y = 3.0
  final eval = f.subs({x: 2.0, y: 3.0});
  print('f(2.0, 3.0)        = ${eval.asDouble}');

  // Algebraic expansion of (x + 3)^3
  final polyExpr = (x + Integer(3)) ^ 3;
  print('Expanded (x+3)^3   = ${polyExpr.expand()}');
}
