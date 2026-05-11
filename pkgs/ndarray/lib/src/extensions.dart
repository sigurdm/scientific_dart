import 'broadcasting.dart';
import 'ndarray.dart';
import 'operations.dart' as ops;

// =============================================================================
// RearrangingNDArrayOperations (NDArray<T extends Object>)
// =============================================================================

/// Rearranging and axis manipulation operations on [NDArray] instances.
extension RearrangingNDArrayOperations<T extends Object> on NDArray<T> {
  /// Reverses the order of elements along the given [axis].
  ///
  /// If [axis] is null, all axes are flipped. If [axis] is an int or a list of ints,
  /// only the specified axes are flipped.
  ///
  /// **Preconditions:**
  /// - [axis] must be an [int], a [List<int>], or `null`.
  /// - Every axis in [axis] must be a valid axis index for the array.
  ///
  /// **Throws:**
  /// - [ArgumentError] if [axis] type is invalid.
  /// - [RangeError] if any axis index is out of bounds.
  ///
  /// {@example /example/rearranging_example.dart lang=dart}
  ///
  /// Refer to [NumPy's flip](https://numpy.org/doc/stable/reference/generated/numpy.flip.html)
  /// for more information on behavioral specifications.
  NDArray<T> flip({dynamic axis}) => ops.flip(this, axis: axis);

  /// Reverses the order of elements along axis 1 (left/right direction).
  ///
  /// **Preconditions:**
  /// - Array rank must be at least 2.
  ///
  /// **Throws:**
  /// - [ArgumentError] if the array rank is less than 2.
  ///
  /// {@example /example/rearranging_example.dart lang=dart}
  ///
  /// Refer to [NumPy's fliplr](https://numpy.org/doc/stable/reference/generated/numpy.fliplr.html)
  /// for more information on behavioral specifications.
  NDArray<T> fliplr() => ops.fliplr(this);

  /// Reverses the order of elements along axis 0 (up/down direction).
  ///
  /// **Preconditions:**
  /// - Array rank must be at least 1.
  ///
  /// **Throws:**
  /// - [ArgumentError] if the array rank is less than 1.
  ///
  /// {@example /example/rearranging_example.dart lang=dart}
  ///
  /// Refer to [NumPy's flipud](https://numpy.org/doc/stable/reference/generated/numpy.flipud.html)
  /// for more information on behavioral specifications.
  NDArray<T> flipud() => ops.flipud(this);

  /// Rolls array elements along a given axis.
  ///
  /// Elements that roll beyond the last position are re-introduced at the first.
  ///
  /// If [axis] is null, the array is flattened before rolling, then restored to the
  /// original shape.
  ///
  /// **Preconditions:**
  /// - [shift] must be an [int] or a [List<int>].
  /// - [axis] must be an [int], a [List<int>], or `null`.
  /// - Every axis in [axis] must be a valid axis index for the array.
  ///
  /// **Throws:**
  /// - [ArgumentError] if [shift] or [axis] types are invalid, or if their lengths mismatch.
  /// - [RangeError] if any axis index is out of bounds.
  ///
  /// {@example /example/rearranging_example.dart lang=dart}
  ///
  /// Refer to [NumPy's roll](https://numpy.org/doc/stable/reference/generated/numpy.roll.html)
  /// for more information on behavioral specifications.
  NDArray<T> roll(dynamic shift, {dynamic axis}) =>
      ops.roll(this, shift, axis: axis);
}
