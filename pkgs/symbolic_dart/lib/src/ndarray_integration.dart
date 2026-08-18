import 'package:ndarray/ndarray.dart';
import 'expr.dart';
import 'lambdify.dart';

/// Extension on [NDArray<Float64>] to evaluate symbolic formulas element-wise
/// without manual variable ordering boilerplate.
extension NDArraySymbolicExtension on NDArray<Float64> {
  /// Evaluates a univariate symbolic [expr] element-wise across this array
  /// treating this array as [variable].
  ///
  /// If [out] is provided, stores the results in [out] in-place.
  NDArray<Float64> mapSymbolic(
    Expr expr,
    Expr variable, {
    NDArray<Float64>? out,
  }) {
    final lambda = expr.lambdify([variable]);
    return lambda.callArray([this], out: out);
  }
}

/// Evaluates a multi-variable symbolic [expr] over a named map of input
/// [NDArray<Float64>]s with automatic multi-dimensional shape broadcasting.
///
/// Example:
/// ```dart
/// final zArr = evaluateSymbolic(
///   sin(x) * exp(-y),
///   inputs: {x: xArr, y: yArr},
/// );
/// ```
NDArray<Float64> evaluateSymbolic(
  Expr expr, {
  required Map<Expr, NDArray<Float64>> inputs,
  NDArray<Float64>? out,
}) {
  if (inputs.isEmpty) {
    final val = expr.asDouble;
    if (out != null) {
      out.setCell([], Float64(val));
      return out;
    }
    return NDArray.scalar(Float64(val), dtype: DType.float64);
  }

  final variables = inputs.keys.toList(growable: false);
  final arrayList = variables.map((v) => inputs[v]!).toList(growable: false);

  final lambda = expr.lambdify(variables);
  return lambda.callArray(arrayList, out: out);
}
