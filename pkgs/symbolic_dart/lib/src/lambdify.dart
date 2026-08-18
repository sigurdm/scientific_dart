import 'package:ndarray/ndarray.dart';
import 'expr.dart';

/// A callable vectorized mathematical evaluator compiled from a symbolic [Expr].
///
/// Supports evaluating at scalar numbers via [callScalar] or vectorizing
/// over N-dimensional numerical arrays via [callArray] with optional broadcasting
/// and output destination buffer (`out`).
final class SymbolicLambda {
  final Expr _expr;
  final List<Expr> _variables;

  SymbolicLambda._(this._expr, this._variables);

  /// Evaluates the symbolic function at scalar points [values].
  ///
  /// It is an error if `values.length` does not match the number of parameters.
  double callScalar(List<num> values) {
    if (values.length != _variables.length) {
      throw ArgumentError(
        'Expected ${_variables.length} input values, got ${values.length}',
      );
    }
    final subsMap = <Object, Object>{};
    for (var i = 0; i < _variables.length; i++) {
      subsMap[_variables[i]] = values[i].toDouble();
    }
    return _expr.subs(subsMap).asDouble;
  }

  /// Evaluates the symbolic function vectorized over N-dimensional arrays [inputs].
  ///
  /// If [out] is provided, writes results into [out] in-place. Otherwise allocates
  /// a new [NDArray<Float64>] matching the broadcast shape of [inputs].
  ///
  /// **Memory & Performance:**
  /// - Supports broadcasting multiple input arrays of compatible shapes.
  /// - Avoids accessing internal raw buffer pointers directly.
  NDArray<Float64> callArray(
    List<NDArray<Float64>> inputs, {
    NDArray<Float64>? out,
  }) {
    if (inputs.length != _variables.length) {
      throw ArgumentError(
        'Expected ${_variables.length} array inputs, got ${inputs.length}',
      );
    }
    if (inputs.isEmpty) {
      final resScalar = _expr.asDouble;
      if (out != null) {
        out.setCell([], Float64(resScalar));
        return out;
      }
      return NDArray.scalar(Float64(resScalar), dtype: DType.float64);
    }

    // Determine broadcast shape
    final firstShape = inputs.first.shape;
    var broadcastShape = List<int>.from(firstShape);
    for (var i = 1; i < inputs.length; i++) {
      broadcastShape = _broadcastShapes(broadcastShape, inputs[i].shape);
    }

    final destination = out ?? NDArray.zeros(broadcastShape, DType.float64);
    if (!destination.hasSameShape(
      NDArray.zeros(broadcastShape, DType.float64),
    )) {
      throw ArgumentError(
        'out array shape ${destination.shape} does not match broadcast shape $broadcastShape',
      );
    }

    final iter = NDIter.broadcast([...inputs, destination]);
    final scalarBuf = List<double>.filled(_variables.length, 0.0);

    while (iter.moveNext()) {
      final coords = iter.coords;
      for (var i = 0; i < inputs.length; i++) {
        final inShape = inputs[i].shape;
        final inCoords = List<int>.generate(inShape.length, (dimIdx) {
          final axisOffset = broadcastShape.length - inShape.length;
          final mappedDim = dimIdx + axisOffset;
          if (mappedDim < 0) return 0;
          return inShape[dimIdx] == 1 ? 0 : coords[mappedDim];
        });
        scalarBuf[i] = inputs[i].getCell(inCoords).toDouble();
      }
      final resultVal = callScalar(scalarBuf);
      destination.setCell(coords, Float64(resultVal));
    }

    return destination;
  }

  static List<int> _broadcastShapes(List<int> a, List<int> b) {
    final maxLen = a.length > b.length ? a.length : b.length;
    final aPadded = List<int>.filled(maxLen - a.length, 1, growable: true)
      ..addAll(a);
    final bPadded = List<int>.filled(maxLen - b.length, 1, growable: true)
      ..addAll(b);
    final result = <int>[];
    for (var i = 0; i < maxLen; i++) {
      if (aPadded[i] == bPadded[i]) {
        result.add(aPadded[i]);
      } else if (aPadded[i] == 1) {
        result.add(bPadded[i]);
      } else if (bPadded[i] == 1) {
        result.add(aPadded[i]);
      } else {
        throw ArgumentError(
          'Incompatible broadcast shapes: $a and $b at dimension $i',
        );
      }
    }
    return result;
  }
}

extension LambdifyExtension on Expr {
  /// Compiles this symbolic expression into a [SymbolicLambda] evaluator
  /// taking the specified [variables] as parameters.
  SymbolicLambda lambdify(List<Expr> variables) {
    return SymbolicLambda._(this, List<Expr>.unmodifiable(variables));
  }
}
