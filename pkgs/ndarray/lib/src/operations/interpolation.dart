// ignore_for_file: non_constant_identifier_names
import 'dart:ffi' as ffi;
import '../ndarray.dart';
import '../ndarray_bindings.dart';
import '../scratch_arena.dart';
import 'helpers.dart';

/// Supported interpolation methods for one-dimensional interpolation.
enum InterpolationMethod {
  /// Piecewise linear interpolation.
  linear,

  /// Nearest-neighbor interpolation.
  nearest,
}

/// Validates that [xp] is strictly increasing.
///
/// It is an error if [xp] is not strictly increasing.
void _validateSorted(NDArray<Float64> xp) {
  final size = xp.shape[0];
  if (size <= 1) return;

  final marker = ScratchArena.marker;
  try {
    final ptr = xp.pointer.cast<ffi.Double>();
    final stride = xp.strides[0];
    for (var i = 1; i < size; i++) {
      final prev = xp.isContiguous ? ptr[i - 1] : ptr[(i - 1) * stride];
      final curr = xp.isContiguous ? ptr[i] : ptr[i * stride];
      if (curr <= prev) {
        throw ArgumentError('xp must be strictly increasing.');
      }
    }
  } finally {
    ScratchArena.reset(marker);
  }
}

/// One-dimensional linear interpolation for monotonically increasing sample points.
///
/// **Preconditions:**
/// - Input arrays [x], [xp], and [fp] must not be disposed.
/// - [xp] and [fp] must be 1D arrays of equal length.
/// - [xp] must be monotonically increasing.
/// - If provided, [out] must have shape matching [x] and float64 dtype.
///
/// It is an error if [x], [xp], or [fp] is disposed, if [xp] or [fp] is not 1D,
/// if [xp] and [fp] lengths mismatch, if [xp] is not strictly increasing, or if [out] has an incompatible shape or dtype.
///
/// **Example:**
/// {@example /example/interpolation_example.dart}
NDArray<Float64> interp(
  NDArray<num> x,
  NDArray<num> xp,
  NDArray<num> fp, {
  double? left,
  double? right,
  InterpolationMethod method = InterpolationMethod.linear,
  NDArray<Float64>? out,
}) {
  if (x.isDisposed ||
      xp.isDisposed ||
      fp.isDisposed ||
      (out != null && out.isDisposed)) {
    throw StateError('Cannot execute interp() with disposed arrays.');
  }

  if (out != null) {
    if (!listEquals(out.shape, x.shape) || out.dtype != DType.float64) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
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
      ? x as NDArray<Float64>
      : promoteToDouble(x);
  final xpDouble = xp.dtype == DType.float64
      ? xp as NDArray<Float64>
      : promoteToDouble(xp);
  final fpDouble = fp.dtype == DType.float64
      ? fp as NDArray<Float64>
      : promoteToDouble(fp);

  try {
    _validateSorted(xpDouble);
  } catch (e) {
    if (!identical(xDouble, x)) xDouble.dispose();
    if (!identical(xpDouble, xp)) xpDouble.dispose();
    if (!identical(fpDouble, fp)) fpDouble.dispose();
    rethrow;
  }

  final res = out ?? NDArray<Float64>.create(x.shape, DType.float64);

  final marker = ScratchArena.marker;
  try {
    if (method == InterpolationMethod.nearest) {
      final size = xDouble.size;
      final xpSize = xpDouble.shape[0];
      final xpPtr = xpDouble.pointer.cast<ffi.Double>();
      final fpPtr = fpDouble.pointer.cast<ffi.Double>();
      final xpStride = xpDouble.strides[0];
      final fpStride = fpDouble.strides[0];
      final xpIsContiguous = xpDouble.isContiguous;
      final fpIsContiguous = fpDouble.isContiguous;

      double getXp(int idx) =>
          xpIsContiguous ? xpPtr[idx] : xpPtr[idx * xpStride];
      double getFp(int idx) =>
          fpIsContiguous ? fpPtr[idx] : fpPtr[idx * fpStride];

      final xpMin = getXp(0);
      final xpMax = getXp(xpSize - 1);
      final defaultLeft = left ?? getFp(0);
      final defaultRight = right ?? getFp(xpSize - 1);

      final tempRes = res.isContiguous
          ? res
          : NDArray<double>.create(x.shape, DType.float64);
      final tempResPtr = tempRes.pointer.cast<ffi.Double>();

      final xDoubleContig = xDouble.isContiguous ? xDouble : xDouble.copy();
      final xPtr = xDoubleContig.pointer.cast<ffi.Double>();

      try {
        for (var i = 0; i < size; i++) {
          final xv = xPtr[i];
          if (xv < xpMin) {
            tempResPtr[i] = defaultLeft;
          } else if (xv > xpMax) {
            tempResPtr[i] = defaultRight;
          } else if (xpSize == 1) {
            tempResPtr[i] = getFp(0);
          } else {
            var low = 0;
            var high = xpSize - 1;
            while (low < high - 1) {
              final mid = (low + high) ~/ 2;
              if (getXp(mid) <= xv) {
                low = mid;
              } else {
                high = mid;
              }
            }
            final x0 = getXp(low);
            final x1 = getXp(low + 1);
            final y0 = getFp(low);
            final y1 = getFp(low + 1);
            if ((xv - x0).abs() <= (x1 - xv).abs()) {
              tempResPtr[i] = y0;
            } else {
              tempResPtr[i] = y1;
            }
          }
        }
      } finally {
        if (!identical(xDoubleContig, xDouble)) {
          xDoubleContig.dispose();
        }
      }
      if (!identical(tempRes, res)) {
        tempRes.copy(out: res);
        tempRes.dispose();
      }
    } else {
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
        final cStridesRes = ScratchArena.copyInts(
          ndim == 0 ? [0] : res.strides,
        );

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

/// Computes one-dimensional interpolation.
///
/// Alias for [interp].
NDArray<Float64> interpolate(
  NDArray<num> x,
  NDArray<num> xp,
  NDArray<num> fp, {
  double? left,
  double? right,
  InterpolationMethod method = InterpolationMethod.linear,
  NDArray<Float64>? out,
}) => interp(x, xp, fp, left: left, right: right, method: method, out: out);
