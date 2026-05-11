import 'ndarray.dart';

/// Encapsulates the results of a shape broadcasting operation.
///
/// Contains the combined [shape], and the adjusted strides ([stridesA] and [stridesB])
/// for both operand arrays to enable aligned memory walks.
final class BroadcastResult {
  /// The combined broadcasted shape list representing the final target shape
  /// resulting from broadcasting matricial dimensions.
  final List<int> shape;

  /// The adjusted memory stride offsets for the first operand array,
  /// containing `0` on stretched dimensions.
  final List<int> stridesA;

  /// The adjusted memory stride offsets for the second operand array,
  /// containing `0` on stretched dimensions.
  final List<int> stridesB;

  /// Creates a new [BroadcastResult] representing aligned shape and strides.
  BroadcastResult(this.shape, this.stridesA, this.stridesB);
}

/// Calculates the broadcasted shape and strides for two matrices.
///
/// Compares dimensions starting from the trailing dimensions and working forward
/// according to standard NumPy shape broadcasting guidelines.
///
/// **Preconditions:**
/// - Trailing dimensions comparing from right-to-left must either be equal or one of them must be 1.
///
/// **Throws:**
/// - [ArgumentError] if matrix shapes are not compatible for broadcasting.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(D)$ where $D$ is the maximum rank dimension length, executing
///   in zero unmanaged heap allocations.
///
/// **Example:**
/// ```dart
/// final a = NDArray<Float64>.fromList([1.0, 2.0], [2, 1], DType.float64);
/// final b = NDArray<Float64>.fromList([10.0, 20.0, 30.0], [1, 3], DType.float64);
/// final result = broadcast(a, b);
/// print(result.shape); // [2, 3]
/// ```
BroadcastResult broadcast(NDArray a, NDArray b) {
  final shapeA = a.shape;
  final shapeB = b.shape;
  final stridesA = a.strides;
  final stridesB = b.strides;

  if (_listEquals(shapeA, shapeB)) {
    return BroadcastResult(shapeA, stridesA, stridesB);
  }

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
      if (i < shapeA.length) {
        newStridesA[maxLen - 1 - i] = stridesA[shapeA.length - 1 - i];
      }
      if (i < shapeB.length) {
        newStridesB[maxLen - 1 - i] = stridesB[shapeB.length - 1 - i];
      }
    } else if (dimA == 1) {
      commonShape[maxLen - 1 - i] = dimB;
      // newStridesA remains 0 for this dimension, effectively stretching it
      if (i < shapeB.length) {
        newStridesB[maxLen - 1 - i] = stridesB[shapeB.length - 1 - i];
      }
    } else if (dimB == 1) {
      commonShape[maxLen - 1 - i] = dimA;
      if (i < shapeA.length) {
        newStridesA[maxLen - 1 - i] = stridesA[shapeA.length - 1 - i];
      }
      // newStridesB remains 0 for this dimension, effectively stretching it
    } else {
      throw ArgumentError(
        'Shapes $shapeA and $shapeB are not compatible for broadcasting',
      );
    }
  }

  return BroadcastResult(commonShape, newStridesA, newStridesB);
}

/// Helper to compare two lists structurally for elements equality.
bool _listEquals<E>(List<E>? list1, List<E>? list2) {
  if (identical(list1, list2)) return true;
  if (list1 == null || list2 == null) return false;
  if (list1.length != list2.length) return false;
  for (var i = 0; i < list1.length; i++) {
    if (list1[i] != list2[i]) return false;
  }
  return true;
}
