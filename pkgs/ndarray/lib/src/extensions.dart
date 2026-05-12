import 'ndarray.dart';
import 'ndarray_bindings.dart';
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'operations.dart' as ops;
import 'broadcasting.dart';

// =============================================================================
// RearrangingNDArrayOperations (NDArray<T extends Object>)
// =============================================================================

/// Rearranging and axis manipulation operations on [NDArray] instances.
extension RearrangingNDArrayOperations<T extends Object> on NDArray<T> {
  /// Reverses the order of elements along the given [axis].
  NDArray<T> flip({dynamic axis}) => ops.flip(this, axis: axis);

  /// Reverses the order of elements along axis 1 (left/right direction).
  NDArray<T> fliplr() => ops.fliplr(this);

  /// Reverses the order of elements along axis 0 (up/down direction).
  NDArray<T> flipud() => ops.flipud(this);

  /// Rolls array elements along a given axis.
  NDArray<T> roll(dynamic shift, {dynamic axis}) =>
      ops.roll(this, shift, axis: axis);
}

// =============================================================================
// Private Internal Helpers
// =============================================================================

