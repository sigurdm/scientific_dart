import 'ndarray.dart';

class BroadcastResult {
  final List<int> shape;
  final List<int> stridesA;
  final List<int> stridesB;
  BroadcastResult(this.shape, this.stridesA, this.stridesB);
}

/// Calculates the broadcasted shape and strides for two arrays.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0], [2, 1], DType.float64);
/// final b = NDArray.fromList([10.0, 20.0, 30.0], [1, 3], DType.float64);
/// final result = broadcast(a, b);
/// print(result.shape); // [2, 3]
/// ```
///
/// Throws [ArgumentError] if the shapes are not compatible for broadcasting.
///
/// Two dimensions are compatible when:
/// 1. They are equal, or
/// 2. One of them is 1.
///
/// Broadcasting compares dimensions starting from the trailing dimensions and working forward.
BroadcastResult broadcast(NDArray a, NDArray b) {
  final shapeA = a.shape;
  final shapeB = b.shape;
  final stridesA = a.strides;
  final stridesB = b.strides;

  final maxLen = shapeA.length > shapeB.length ? shapeA.length : shapeB.length;
  final commonShape = List<int>.filled(maxLen, 1);
  final newStridesA = List<int>.filled(maxLen, 0);
  final newStridesB = List<int>.filled(maxLen, 0);

  for (var i = 0; i < maxLen; i++) {
    // Compare from right to left
    final dimA = i < shapeA.length ? shapeA[shapeA.length - 1 - i] : 1;
    final dimB = i < shapeB.length ? shapeB[shapeB.length - 1 - i] : 1;

    if (dimA == dimB) {
      commonShape[maxLen - 1 - i] = dimA;
      if (i < shapeA.length)
        newStridesA[maxLen - 1 - i] = stridesA[shapeA.length - 1 - i];
      if (i < shapeB.length)
        newStridesB[maxLen - 1 - i] = stridesB[shapeB.length - 1 - i];
    } else if (dimA == 1) {
      commonShape[maxLen - 1 - i] = dimB;
      // newStridesA remains 0 for this dimension, effectively stretching it
      if (i < shapeB.length)
        newStridesB[maxLen - 1 - i] = stridesB[shapeB.length - 1 - i];
    } else if (dimB == 1) {
      commonShape[maxLen - 1 - i] = dimA;
      if (i < shapeA.length)
        newStridesA[maxLen - 1 - i] = stridesA[shapeA.length - 1 - i];
      // newStridesB remains 0 for this dimension, effectively stretching it
    } else {
      throw ArgumentError(
        'Shapes $shapeA and $shapeB are not compatible for broadcasting',
      );
    }
  }

  return BroadcastResult(commonShape, newStridesA, newStridesB);
}
