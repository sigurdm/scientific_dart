import 'ndarray.dart';
import 'operations.dart' as ops;
import 'fft.dart' as fft_ops;

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

  /// Shifts the zero-frequency component to the center of the spectrum.
  NDArray<T> fftshift({dynamic axes}) => fft_ops.fftshift(this, axes: axes);

  /// Inverse of [fftshift].
  NDArray<T> ifftshift({dynamic axes}) => fft_ops.ifftshift(this, axes: axes);
}

// =============================================================================
// Private Internal Helpers
// =============================================================================
