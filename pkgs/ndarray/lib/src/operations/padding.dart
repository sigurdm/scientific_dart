import 'dart:ffi' as ffi;
import '../ndarray.dart';
import '../scratch_arena.dart';
import '../ndarray_bindings.dart' as bindings;

/// Supported padding modes.
enum PaddingMode {
  constant,
  edge,
  reflect,
  symmetric,
  wrap,
  linearRamp,
  maximum,
  mean,
  median,
  minimum,
}

/// Represents padding widths before and after for each axis.
final class PadWidth {
  final (int before, int after)? _uniform;
  final List<(int before, int after)>? _axes;

  const PadWidth._(this._uniform, this._axes);

  /// Creates a uniform [PadWidth] for all axes.
  ///
  /// If [after] is not specified, it defaults to [before].
  ///
  /// Preconditions:
  /// - [before] must be non-negative.
  /// - [after] must be non-negative if specified.
  factory PadWidth.all(int before, [int? after]) {
    RangeError.checkNotNegative(before, 'before');
    if (after != null) {
      RangeError.checkNotNegative(after, 'after');
    }
    return PadWidth._((before, after ?? before), null);
  }

  /// Creates a [PadWidth] with specific widths for each axis.
  ///
  /// Preconditions:
  /// - All before and after widths must be non-negative.
  factory PadWidth.axes(List<(int before, int after)> widths) {
    for (final (before, after) in widths) {
      RangeError.checkNotNegative(before, 'before');
      RangeError.checkNotNegative(after, 'after');
    }
    return PadWidth._(null, List.unmodifiable(widths));
  }

  /// Normalizes the pad widths to a list of length [rank].
  List<(int before, int after)> normalize(int rank) {
    final uniform = _uniform;
    if (uniform != null) {
      return List.filled(rank, uniform);
    }
    final axes = _axes;
    if (axes != null) {
      if (axes.length != rank) {
        throw ArgumentError(
          'Length of padding widths (${axes.length}) must match array rank ($rank)',
        );
      }
      return axes;
    }
    throw StateError('Invalid PadWidth state');
  }
}

/// Represents constant padding values for each axis.
final class PadValues<T> {
  final (T before, T after)? _uniform;
  final List<(T before, T after)>? _axes;

  const PadValues._(this._uniform, this._axes);

  /// Creates a uniform [PadValues] for all axes.
  ///
  /// If [after] is not specified, it defaults to [before].
  factory PadValues.all(T before, [T? after]) {
    return PadValues._((before, after ?? before), null);
  }

  /// Creates a [PadValues] with specific values for each axis.
  factory PadValues.axes(List<(T before, T after)> values) {
    return PadValues._(null, List.unmodifiable(values));
  }

  /// Normalizes the pad values to a list of length [rank].
  List<(T before, T after)> normalize(int rank, T defaultValue) {
    final uniform = _uniform;
    if (uniform != null) {
      return List.filled(rank, uniform);
    }
    final axes = _axes;
    if (axes != null) {
      if (axes.length != rank) {
        throw ArgumentError(
          'Length of padding values (${axes.length}) must match array rank ($rank)',
        );
      }
      return axes;
    }
    return List.filled(rank, (defaultValue, defaultValue));
  }
}

/// Represents the window length for statistical padding modes.
final class StatLength {
  final (int before, int after)? _uniform;
  final List<(int before, int after)>? _axes;

  const StatLength._(this._uniform, this._axes);

  /// Creates a uniform [StatLength] for all axes.
  ///
  /// If [after] is not specified, it defaults to [before].
  ///
  /// Preconditions:
  /// - [before] must be positive.
  /// - [after] must be positive if specified.
  factory StatLength.all(int before, [int? after]) {
    if (before <= 0) {
      throw ArgumentError.value(before, 'before', 'Must be positive');
    }
    if (after != null && after <= 0) {
      throw ArgumentError.value(after, 'after', 'Must be positive');
    }
    return StatLength._((before, after ?? before), null);
  }

  /// Creates a [StatLength] with specific lengths for each axis.
  ///
  /// Preconditions:
  /// - All before and after lengths must be positive.
  factory StatLength.axes(List<(int before, int after)> lengths) {
    for (final (before, after) in lengths) {
      if (before <= 0) {
        throw ArgumentError.value(before, 'before', 'Must be positive');
      }
      if (after <= 0) {
        throw ArgumentError.value(after, 'after', 'Must be positive');
      }
    }
    return StatLength._(null, List.unmodifiable(lengths));
  }

  /// Normalizes the stat lengths to a list of length [rank], capping them by the [shape].
  List<(int before, int after)> normalize(List<int> shape) {
    final rank = shape.length;
    final uniform = _uniform;
    if (uniform != null) {
      return List.generate(rank, (i) {
        final b = uniform.$1.clamp(1, shape[i]);
        final a = uniform.$2.clamp(1, shape[i]);
        return (b, a);
      });
    }
    final axes = _axes;
    if (axes != null) {
      if (axes.length != rank) {
        throw ArgumentError(
          'Length of stat lengths (${axes.length}) must match array rank ($rank)',
        );
      }
      return List.generate(rank, (i) {
        final b = axes[i].$1.clamp(1, shape[i]);
        final a = axes[i].$2.clamp(1, shape[i]);
        return (b, a);
      });
    }
    return List.generate(rank, (i) => (shape[i], shape[i]));
  }
}

/// Helper to get default constant value for a DType.
Object _getDefaultValue(DType dtype) {
  if (dtype == DType.float64 || dtype == DType.float32) {
    return 0.0;
  } else if (dtype == DType.complex128 || dtype == DType.complex64) {
    return Complex(0.0, 0.0);
  } else if (dtype == DType.boolean) {
    return false;
  } else {
    return 0;
  }
}

/// Pads an N-dimensional array according to the specified [padWidth] and [mode].
///
/// This operation returns a new array with padded values along each axis, or
/// writes to [out] if provided.
///
/// ### Preconditions
/// - The [array] must not be disposed.
/// - If [out] is provided, it must not be disposed, and its shape must match the
///   calculated padded shape.
/// - The [array] must have rank >= 1 (cannot pad 0-dimensional arrays).
/// - The length of [padWidth], [constantValues], [endValues], and [statLength]
///   must match the rank of the [array] when specified per-axis.
///
/// ### Exceptions
/// - Throws [StateError] if [array] or [out] is disposed.
/// - Throws [ArgumentError] if [array] is 0-dimensional, or if parameter
///   dimensions mismatch.
///
/// ### Performance Considerations
/// - The operation performs padding sequentially axis-by-axis.
/// - If an axis has 0 padding (both before and after), it is skipped to save
///   memory and copy overhead, unless it is the only axis or `out` is provided
///   (in which case at least one copy is performed).
/// - Statistical modes (`median`, `mean`, etc.) cache the statistics per slice
///   to avoid redundant calculations (like sorting for median) for every padded
///   element.
///
/// ### Example
/// {@example /example/padding_example.dart}
NDArray<T> pad<T extends Object>(
  NDArray<T> array,
  PadWidth padWidth, {
  PaddingMode mode = PaddingMode.constant,
  PadValues<T>? constantValues,
  PadValues<T>? endValues,
  StatLength? statLength,
  NDArray<T>? out,
}) {
  if (array.isDisposed) {
    throw StateError('Source array is disposed.');
  }
  if (out != null && out.isDisposed) {
    throw StateError('Output array is disposed.');
  }

  final rank = array.rank;
  if (rank == 0) {
    throw ArgumentError('Cannot pad a 0-dimensional array.');
  }

  // Normalize parameters
  final normPadWidths = padWidth.normalize(rank);
  final defaultValue = _getDefaultValue(array.dtype) as T;
  final normConstantValues = (constantValues ?? PadValues.all(defaultValue))
      .normalize(rank, defaultValue);
  final normEndValues = (endValues ?? PadValues.all(defaultValue)).normalize(
    rank,
    defaultValue,
  );
  final normStatLengths =
      (statLength ??
              StatLength.axes(
                List.generate(rank, (i) => (array.shape[i], array.shape[i])),
              ))
          .normalize(array.shape);

  // Calculate expected output shape
  final finalShape = List<int>.generate(rank, (i) {
    final (before, after) = normPadWidths[i];
    return array.shape[i] + before + after;
  });

  if (out != null) {
    if (out.dtype != array.dtype) {
      throw ArgumentError(
        'Output array dtype (${out.dtype}) must match source array dtype (${array.dtype}).',
      );
    }
    if (out.rank != rank) {
      throw ArgumentError(
        'Output array rank (${out.rank}) must match expected rank ($rank).',
      );
    }
    for (var i = 0; i < rank; i++) {
      if (out.shape[i] != finalShape[i]) {
        throw ArgumentError(
          'Output array shape at dim $i (${out.shape[i]}) must match expected shape (${finalShape[i]}).',
        );
      }
    }
  }

  var needsPadding = false;
  for (final (before, after) in normPadWidths) {
    if (before > 0 || after > 0) {
      needsPadding = true;
      break;
    }
  }

  if (!needsPadding) {
    if (out != null) {
      // If no padding is needed, we still copy to out if provided
      // We can just run the loop below, it will just copy.
    } else {
      return array.copy();
    }
  }

  return NDArray.scope(() {
    var currentSrc = array;

    for (int axis = 0; axis < rank; axis++) {
      final padBefore = normPadWidths[axis].$1;
      final padAfter = normPadWidths[axis].$2;

      final isLastAxis = axis == rank - 1;
      final skipThisAxis =
          padBefore == 0 && padAfter == 0 && (!isLastAxis || out == null);

      if (skipThisAxis) {
        continue;
      }

      final nextShape = List<int>.from(currentSrc.shape);
      nextShape[axis] += padBefore + padAfter;

      final NDArray<T> currentDest;
      if (isLastAxis && out != null) {
        currentDest = out;
      } else {
        currentDest = NDArray<T>.create(
          nextShape,
          array.dtype,
          zeroInit: false,
        );
      }

      _padAxis(
        currentSrc,
        currentDest,
        axis,
        padBefore,
        padAfter,
        mode,
        normConstantValues[axis].$1,
        normConstantValues[axis].$2,
        normEndValues[axis].$1,
        normEndValues[axis].$2,
        normStatLengths[axis].$1,
        normStatLengths[axis].$2,
      );

      if (currentSrc != array) {
        currentSrc.dispose();
      }
      currentSrc = currentDest;
    }

    if (out == null) {
      currentSrc.detachToParentScope();
    }
    return currentSrc;
  });
}

void _padAxis<T extends Object>(
  NDArray<T> src,
  NDArray<T> dest,
  int axis,
  int padBefore,
  int padAfter,
  PaddingMode mode,
  T constantBefore,
  T constantAfter,
  T endBefore,
  T endAfter,
  int statLengthBefore,
  int statLengthAfter,
) {
  final marker = ScratchArena.marker;
  try {
    final shapeSrcPtr = ScratchArena.copyInts(src.shape);
    final stridesSrcPtr = ScratchArena.copyInts(src.strides);
    final shapeDestPtr = ScratchArena.copyInts(dest.shape);
    final rank = src.rank;
    final modeInt = mode.index;

    switch (src.dtype) {
      case DType.float64:
        bindings.pad_axis_double(
          src.pointer.cast(),
          shapeSrcPtr,
          stridesSrcPtr,
          dest.pointer.cast(),
          shapeDestPtr,
          rank,
          axis,
          padBefore,
          padAfter,
          modeInt,
          (constantBefore as num).toDouble(),
          (constantAfter as num).toDouble(),
          (endBefore as num).toDouble(),
          (endAfter as num).toDouble(),
          statLengthBefore,
          statLengthAfter,
        );
      case DType.float32:
        bindings.pad_axis_float(
          src.pointer.cast(),
          shapeSrcPtr,
          stridesSrcPtr,
          dest.pointer.cast(),
          shapeDestPtr,
          rank,
          axis,
          padBefore,
          padAfter,
          modeInt,
          (constantBefore as num).toDouble(),
          (constantAfter as num).toDouble(),
          (endBefore as num).toDouble(),
          (endAfter as num).toDouble(),
          statLengthBefore,
          statLengthAfter,
        );
      case DType.int64:
        bindings.pad_axis_int64(
          src.pointer.cast(),
          shapeSrcPtr,
          stridesSrcPtr,
          dest.pointer.cast(),
          shapeDestPtr,
          rank,
          axis,
          padBefore,
          padAfter,
          modeInt,
          constantBefore as int,
          constantAfter as int,
          endBefore as int,
          endAfter as int,
          statLengthBefore,
          statLengthAfter,
        );
      case DType.int32:
        bindings.pad_axis_int32(
          src.pointer.cast(),
          shapeSrcPtr,
          stridesSrcPtr,
          dest.pointer.cast(),
          shapeDestPtr,
          rank,
          axis,
          padBefore,
          padAfter,
          modeInt,
          constantBefore as int,
          constantAfter as int,
          endBefore as int,
          endAfter as int,
          statLengthBefore,
          statLengthAfter,
        );
      case DType.uint8:
      case DType.boolean:
        int cb = 0;
        int ca = 0;
        int eb = 0;
        int ea = 0;
        if (src.dtype == DType.boolean) {
          cb = (constantBefore as bool) ? 1 : 0;
          ca = (constantAfter as bool) ? 1 : 0;
          eb = (endBefore as bool) ? 1 : 0;
          ea = (endAfter as bool) ? 1 : 0;
        } else {
          cb = constantBefore as int;
          ca = constantAfter as int;
          eb = endBefore as int;
          ea = endAfter as int;
        }
        bindings.pad_axis_uint8(
          src.pointer.cast(),
          shapeSrcPtr,
          stridesSrcPtr,
          dest.pointer.cast(),
          shapeDestPtr,
          rank,
          axis,
          padBefore,
          padAfter,
          modeInt,
          cb,
          ca,
          eb,
          ea,
          statLengthBefore,
          statLengthAfter,
        );
      case DType.complex128:
        final cbPtr = ScratchArena.allocate<bindings.cpx_t>(
          ffi.sizeOf<bindings.cpx_t>(),
        );
        final caPtr = ScratchArena.allocate<bindings.cpx_t>(
          ffi.sizeOf<bindings.cpx_t>(),
        );
        final ebPtr = ScratchArena.allocate<bindings.cpx_t>(
          ffi.sizeOf<bindings.cpx_t>(),
        );
        final eaPtr = ScratchArena.allocate<bindings.cpx_t>(
          ffi.sizeOf<bindings.cpx_t>(),
        );

        final cb = constantBefore as Complex;
        final ca = constantAfter as Complex;
        final eb = endBefore as Complex;
        final ea = endAfter as Complex;

        cbPtr.ref.r = cb.real;
        cbPtr.ref.i = cb.imag;
        caPtr.ref.r = ca.real;
        caPtr.ref.i = ca.imag;
        ebPtr.ref.r = eb.real;
        ebPtr.ref.i = eb.imag;
        eaPtr.ref.r = ea.real;
        eaPtr.ref.i = ea.imag;

        bindings.pad_axis_complex128(
          src.pointer.cast(),
          shapeSrcPtr,
          stridesSrcPtr,
          dest.pointer.cast(),
          shapeDestPtr,
          rank,
          axis,
          padBefore,
          padAfter,
          modeInt,
          cbPtr.ref,
          caPtr.ref,
          ebPtr.ref,
          eaPtr.ref,
          statLengthBefore,
          statLengthAfter,
        );
      case DType.complex64:
        final cbPtr = ScratchArena.allocate<bindings.cpx_f_t>(
          ffi.sizeOf<bindings.cpx_f_t>(),
        );
        final caPtr = ScratchArena.allocate<bindings.cpx_f_t>(
          ffi.sizeOf<bindings.cpx_f_t>(),
        );
        final ebPtr = ScratchArena.allocate<bindings.cpx_f_t>(
          ffi.sizeOf<bindings.cpx_f_t>(),
        );
        final eaPtr = ScratchArena.allocate<bindings.cpx_f_t>(
          ffi.sizeOf<bindings.cpx_f_t>(),
        );

        final cb = constantBefore as Complex;
        final ca = constantAfter as Complex;
        final eb = endBefore as Complex;
        final ea = endAfter as Complex;

        cbPtr.ref.r = cb.real;
        cbPtr.ref.i = cb.imag;
        caPtr.ref.r = ca.real;
        caPtr.ref.i = ca.imag;
        ebPtr.ref.r = eb.real;
        ebPtr.ref.i = eb.imag;
        eaPtr.ref.r = ea.real;
        eaPtr.ref.i = ea.imag;

        bindings.pad_axis_complex64(
          src.pointer.cast(),
          shapeSrcPtr,
          stridesSrcPtr,
          dest.pointer.cast(),
          shapeDestPtr,
          rank,
          axis,
          padBefore,
          padAfter,
          modeInt,
          cbPtr.ref,
          caPtr.ref,
          ebPtr.ref,
          eaPtr.ref,
          statLengthBefore,
          statLengthAfter,
        );
      default:
        throw UnsupportedError('Unsupported dtype for padding: ${src.dtype}');
    }
  } finally {
    ScratchArena.reset(marker);
  }
}
