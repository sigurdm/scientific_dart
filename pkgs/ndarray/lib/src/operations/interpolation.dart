// ignore_for_file: non_constant_identifier_names
import 'dart:ffi' as ffi;
import '../ndarray.dart';
import '../ndarray_bindings.dart';
import '../scratch_arena.dart';
import 'helpers.dart';

/// Validates that [xp] is strictly increasing.
///
/// Throws [ArgumentError] if [xp] is not strictly increasing.
void _validateSorted(NDArray<double> xp) {
  final size = xp.shape[0];
  if (size <= 1) return;

  final res = is_strictly_increasing_double(
    xp.pointer.cast(),
    size,
    xp.strides[0],
  );
  if (res == 0) {
    throw ArgumentError('xp must be strictly increasing.');
  }
}

/// Computes one-dimensional linear interpolation.
///
/// Returns the one-dimensional piecewise linear interpolant to a function with
/// given discrete data points ([xp], [fp]), evaluated at [x].
/// The [xp] array must be strictly increasing and have the same length as [fp].
/// Optional [left] and [right] specify values to return for `x < xp[0]` and `x > xp[xp.length-1]` respectively, defaulting to `fp[0]` and `fp[fp.length-1]`.
///
/// **Preconditions:**
/// - [x], [xp], [fp] must not be disposed.
/// - [xp] and [fp] must be 1D arrays.
/// - [xp] and [fp] must have the same length.
/// - [xp] must be strictly increasing.
///
/// **Throws:**
/// - [StateError] if any input array is disposed.
/// - [ArgumentError] if [xp] or [fp] is not 1D, or if their lengths mismatch.
/// - [ArgumentError] if [xp] is empty.
/// - [ArgumentError] if [xp] is not strictly increasing.
///
/// {@example /example/interpolation_example.dart}
NDArray<double> interp(
  NDArray<num> x,
  NDArray<num> xp,
  NDArray<num> fp, {
  double? left,
  double? right,
}) {
  if (x.isDisposed || xp.isDisposed || fp.isDisposed) {
    throw StateError('Cannot execute interp() with disposed arrays.');
  }

  if (xp.shape.length != 1 || fp.shape.length != 1) {
    throw ArgumentError('xp and fp must be 1-dimensional arrays.');
  }

  if (xp.shape[0] != fp.shape[0]) {
    throw ArgumentError('xp and fp must have the same length.');
  }

  if (xp.shape[0] == 0) {
    throw ArgumentError('xp must not be empty.');
  }

  final xDouble = x.dtype == DType.float64
      ? x as NDArray<double>
      : promoteToDouble(x);
  final xpDouble = xp.dtype == DType.float64
      ? xp as NDArray<double>
      : promoteToDouble(xp);
  final fpDouble = fp.dtype == DType.float64
      ? fp as NDArray<double>
      : promoteToDouble(fp);

  try {
    _validateSorted(xpDouble);
  } catch (e) {
    if (!identical(xDouble, x)) xDouble.dispose();
    if (!identical(xpDouble, xp)) xpDouble.dispose();
    if (!identical(fpDouble, fp)) fpDouble.dispose();
    rethrow;
  }

  final res = NDArray<double>.create(x.shape, DType.float64);

  final marker = ScratchArena.marker;
  try {
    // Prepare left/right pointers.
    ffi.Pointer<ffi.Double> pLeft = ffi.nullptr;
    if (left != null) {
      pLeft = ScratchArena.allocate<ffi.Double>(ffi.sizeOf<ffi.Double>());
      pLeft.value = left;
    }
    ffi.Pointer<ffi.Double> pRight = ffi.nullptr;
    if (right != null) {
      pRight = ScratchArena.allocate<ffi.Double>(ffi.sizeOf<ffi.Double>());
      pRight.value = right;
    }

    final isContiguous =
        xDouble.isContiguous &&
        xpDouble.isContiguous &&
        fpDouble.isContiguous &&
        res.isContiguous;

    if (isContiguous) {
      v_interp_double(
        xDouble.pointer.cast(),
        xDouble.shape.isEmpty ? 1 : xDouble.shape.reduce((a, b) => a * b),
        xpDouble.pointer.cast(),
        xpDouble.shape[0],
        fpDouble.pointer.cast(),
        res.pointer.cast(),
        pLeft,
        pRight,
      );
    } else {
      // Strided version.
      var ndim = xDouble.shape.length;
      final cBuffer = ScratchArena.getStridedBuffer(ndim == 0 ? 1 : ndim);
      final cShape = cBuffer;
      final cStridesX = ScratchArena.copyInts(
        ndim == 0 ? [0] : xDouble.strides,
      );
      final cStridesRes = ScratchArena.copyInts(ndim == 0 ? [0] : res.strides);

      if (ndim == 0) {
        cShape[0] = 1;
        ndim = 1;
      } else {
        for (var i = 0; i < ndim; i++) {
          cShape[i] = xDouble.shape[i];
        }
      }

      s_interp_double(
        xDouble.pointer.cast(),
        cStridesX,
        xpDouble.pointer.cast(),
        xpDouble.strides[0],
        xpDouble.shape[0],
        fpDouble.pointer.cast(),
        fpDouble.strides[0],
        res.pointer.cast(),
        cStridesRes,
        cShape,
        ndim,
        pLeft,
        pRight,
      );
    }
  } finally {
    ScratchArena.reset(marker);
    // Dispose promoted arrays if they were created.
    if (!identical(xDouble, x)) xDouble.dispose();
    if (!identical(xpDouble, xp)) xpDouble.dispose();
    if (!identical(fpDouble, fp)) fpDouble.dispose();
  }

  return res;
}
