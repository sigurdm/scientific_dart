import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'ndarray.dart';
import 'ndarray_bindings.dart';

/// Represents the spacing between points for calculus operations.
///
/// Spacing can be a constant scalar (step) or a list of coordinates
/// (variable spacing). Both real and complex types are supported.
sealed class Spacing {
  const Spacing();

  /// Constant real spacing.
  const factory Spacing.step(double dx) = StepSpacing;

  /// Variable real spacing using a coordinate array.
  const factory Spacing.coordinates(List<double> x) = CoordinateSpacing;

  /// Constant complex spacing.
  const factory Spacing.complexStep(Complex dz) = ComplexStepSpacing;

  /// Variable complex spacing using complex coordinate arrays.
  const factory Spacing.complexCoordinates(List<Complex> z) =
      ComplexCoordinateSpacing;
}

/// Constant real spacing implementation.
final class StepSpacing extends Spacing {
  final double dx;
  const StepSpacing(this.dx);
}

/// Variable real spacing implementation.
final class CoordinateSpacing extends Spacing {
  final List<double> x;
  const CoordinateSpacing(this.x);
}

/// Constant complex spacing implementation.
final class ComplexStepSpacing extends Spacing {
  final Complex dz;
  const ComplexStepSpacing(this.dz);
}

/// Variable complex spacing implementation.
final class ComplexCoordinateSpacing extends Spacing {
  final List<Complex> z;
  const ComplexCoordinateSpacing(this.z);
}

// Helper for list equality comparison
bool _listEquals(List a, List b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

/// Integrate along the given axis using the composite trapezoidal rule.
///
/// The composite trapezoidal rule approximates the integral of a function by
/// dividing the area under the curve into trapezoids, which is much more accurate
/// than simple rectangular integration.
///
/// Spacing along the axis is specified by [spacing].
///
/// **Preconditions:**
/// - Input [y] must not be disposed.
/// - Input [y] must be a floating-point or complex type.
/// - [axis] must be within bounds `[-y.rank, y.rank - 1]`.
/// - If [spacing] is [CoordinateSpacing] or [ComplexCoordinateSpacing], its
///   length must match `y.shape[axis]`.
/// - If [spacing] is complex, input [y] must also be complex.
/// - If [out] is provided, it must match the resolved shape and dtype.
///
/// **Throws:**
/// - [StateError] if [y] is disposed.
/// - [ArgumentError] if [y] has an integer or boolean dtype.
/// - [ArgumentError] if complex spacing is used with a real input array.
/// - [ArgumentError] if [axis] is out of bounds or coordinate spacing length is mismatched.
///
/// **Example:**
/// ```dart
/// final y = NDArray.fromList([1.0, 2.0, 4.0], [3], DType.float64);
/// final res = trapz(y, spacing: Spacing.step(1.0)); // 4.5
/// ```
NDArray<T> trapz<T>(
  NDArray<T> y, {
  Spacing spacing = const Spacing.step(1.0),
  int axis = -1,
  NDArray<T>? out,
}) {
  if (y.isDisposed) {
    throw StateError('Cannot execute trapz() on a disposed array.');
  }

  if (y.dtype.isInteger || y.dtype == DType.boolean) {
    throw ArgumentError(
      'Calculus operations are not supported on integer or boolean arrays. '
      'Cast to a floating-point or complex type first.',
    );
  }

  final bool isComplexSpacing =
      spacing is ComplexStepSpacing || spacing is ComplexCoordinateSpacing;
  if (isComplexSpacing && !y.dtype.isComplex) {
    throw ArgumentError(
      'Complex spacing requires a complex input array. '
      'Cast the array to complex first.',
    );
  }

  var targetAxis = axis;
  if (targetAxis < 0) {
    targetAxis = y.shape.length + targetAxis;
  }
  if (targetAxis < 0 || targetAxis >= y.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${y.shape}');
  }

  final N = y.shape[targetAxis];
  switch (spacing) {
    case CoordinateSpacing(x: final x):
      if (x.length != N) {
        throw ArgumentError(
          'Coordinate spacing length (${x.length}) must match dimension size ($N).',
        );
      }
    case ComplexCoordinateSpacing(z: final z):
      if (z.length != N) {
        throw ArgumentError(
          'Complex coordinate spacing length (${z.length}) must match dimension size ($N).',
        );
      }
    default:
      break;
  }

  final targetShape = List<int>.from(y.shape)..removeAt(targetAxis);
  final result = out ?? NDArray<T>.zeros(targetShape, y.dtype);
  if (out != null) {
    if (!_listEquals(out.shape, targetShape) || out.dtype != y.dtype) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  final rank = y.shape.length;
  final cShape = malloc<ffi.Int>(rank);
  final cStridesY = malloc<ffi.Int>(rank);
  final cStridesRes = malloc<ffi.Int>(rank.clamp(1, 8));

  for (var i = 0; i < rank; i++) {
    cShape[i] = y.shape[i];
    cStridesY[i] = y.strides[i];
  }
  for (var i = 0; i < result.strides.length; i++) {
    cStridesRes[i] = result.strides[i];
  }

  try {
    switch (spacing) {
      case ComplexStepSpacing():
      case ComplexCoordinateSpacing():
        NDArray<T>? spacingArray;
        if (y.dtype == DType.complex128) {
          final dxStruct = malloc<cpx_t>();
          try {
            if (spacing is ComplexStepSpacing) {
              dxStruct.ref.r = spacing.dz.real;
              dxStruct.ref.i = spacing.dz.imag;
            } else {
              dxStruct.ref.r = 1.0;
              dxStruct.ref.i = 0.0;
              spacingArray =
                  NDArray<Complex>.fromList(
                        (spacing as ComplexCoordinateSpacing).z,
                        [N],
                        DType.complex128,
                      )
                      as NDArray<T>;
            }
            s_trapz_complex128_all(
              y.pointer.cast(),
              cStridesY,
              spacingArray?.pointer.cast() ?? ffi.nullptr,
              spacingArray?.strides[0] ?? 0,
              dxStruct.ref,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              targetAxis,
            );
          } finally {
            malloc.free(dxStruct);
          }
        } else if (y.dtype == DType.complex64) {
          final dxStruct = malloc<cpx_f_t>();
          try {
            if (spacing is ComplexStepSpacing) {
              dxStruct.ref.r = spacing.dz.real;
              dxStruct.ref.i = spacing.dz.imag;
            } else {
              dxStruct.ref.r = 1.0;
              dxStruct.ref.i = 0.0;
              spacingArray =
                  NDArray<Complex>.fromList(
                        (spacing as ComplexCoordinateSpacing).z,
                        [N],
                        DType.complex64,
                      )
                      as NDArray<T>;
            }
            s_trapz_complex64_all(
              y.pointer.cast(),
              cStridesY,
              spacingArray?.pointer.cast() ?? ffi.nullptr,
              spacingArray?.strides[0] ?? 0,
              dxStruct.ref,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              targetAxis,
            );
          } finally {
            malloc.free(dxStruct);
          }
        }
        spacingArray?.dispose();

      case StepSpacing():
      case CoordinateSpacing():
        double dxVal = 1.0;
        NDArray? spacingArray;
        if (spacing is StepSpacing) {
          dxVal = spacing.dx;
        } else {
          spacingArray = NDArray<double>.fromList(
            (spacing as CoordinateSpacing).x,
            [N],
            y.dtype.isFloating ? y.dtype as DType<double> : DType.float64,
          );
        }

        final dtype = y.dtype;
        if (dtype == DType.float64) {
          s_trapz_double(
            y.pointer.cast(),
            cStridesY,
            spacingArray?.pointer.cast() ?? ffi.nullptr,
            spacingArray?.strides[0] ?? 0,
            dxVal,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            targetAxis,
          );
        } else if (dtype == DType.float32) {
          s_trapz_float(
            y.pointer.cast(),
            cStridesY,
            spacingArray?.pointer.cast() ?? ffi.nullptr,
            spacingArray?.strides[0] ?? 0,
            dxVal.toDouble(),
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            targetAxis,
          );
        } else if (dtype == DType.complex128) {
          s_trapz_complex128(
            y.pointer.cast(),
            cStridesY,
            spacingArray?.pointer.cast() ?? ffi.nullptr,
            spacingArray?.strides[0] ?? 0,
            dxVal,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            targetAxis,
          );
        } else if (dtype == DType.complex64) {
          s_trapz_complex64(
            y.pointer.cast(),
            cStridesY,
            spacingArray?.pointer.cast() ?? ffi.nullptr,
            spacingArray?.strides[0] ?? 0,
            dxVal.toDouble(),
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            targetAxis,
          );
        }
        spacingArray?.dispose();
    }
  } finally {
    malloc.free(cShape);
    malloc.free(cStridesY);
    malloc.free(cStridesRes);
  }

  return result;
}

/// Calculate the N-Dimensional gradient along a single axis.
///
/// Returns a single [NDArray] representing the derivative along [axis].
/// For a 1D array, this is equivalent to `gradientArray(f)[0]`.
/// To calculate gradients along multiple axes at once, use [gradientArray].
///
/// The gradient is calculated using second-order accurate central differences
/// for interior points.
///
/// **Boundary accuracy ([edgeOrder]):**
/// At the edges of the array, central differences cannot be used.
/// - `edgeOrder = 1`: Uses first-order accurate one-sided differences (forward
///   difference at the start, backward difference at the end).
/// - `edgeOrder = 2`: Uses second-order accurate one-sided differences,
///   providing higher precision at the boundaries by using more neighboring
///   points.
///
/// Spacing along the axis is specified by [spacing].
///
/// **Preconditions:**
/// - Input [f] must not be disposed.
/// - Input [f] must be a floating-point or complex type.
/// - [axis] must be within bounds `[-f.rank, f.rank - 1]`.
/// - If [spacing] is [CoordinateSpacing] or [ComplexCoordinateSpacing], its
///   length must match `f.shape[axis]`.
/// - If [spacing] is complex, input [f] must also be complex.
/// - If [out] is provided, it must match the resolved shape and dtype.
///
/// **Throws:**
/// - [StateError] if [f] is disposed.
/// - [ArgumentError] if [f] has an integer or boolean dtype.
/// - [ArgumentError] if complex spacing is used with a real input array.
/// - [ArgumentError] if [axis] is out of bounds or spacing is invalid.
/// - [ArgumentError] if [edgeOrder] is not 1 or 2.
///
/// **Example:**
/// ```dart
/// final f = NDArray.fromList([1.0, 2.0, 4.0, 7.0], [4], DType.float64);
/// final res = gradient(f, spacing: Spacing.step(1.0)); // [1.0, 1.5, 2.5, 3.0]
/// ```
NDArray<T> gradient<T extends Object>(
  NDArray<T> f, {
  Spacing spacing = const Spacing.step(1.0),
  int axis = 0,
  int edgeOrder = 1,
  NDArray<T>? out,
}) {
  if (f.isDisposed) {
    throw StateError('Cannot execute gradient() on a disposed array.');
  }
  if (edgeOrder != 1 && edgeOrder != 2) {
    throw ArgumentError('edgeOrder must be 1 or 2 (was $edgeOrder).');
  }

  if (f.dtype.isInteger || f.dtype == DType.boolean) {
    throw ArgumentError(
      'Calculus operations are not supported on integer or boolean arrays. '
      'Cast to a floating-point or complex type first.',
    );
  }

  final bool isComplexSpacing =
      spacing is ComplexStepSpacing || spacing is ComplexCoordinateSpacing;
  if (isComplexSpacing && !f.dtype.isComplex) {
    throw ArgumentError(
      'Complex spacing requires a complex input array. '
      'Cast the array to complex first.',
    );
  }

  var targetAxis = axis;
  if (targetAxis < 0) {
    targetAxis = f.shape.length + targetAxis;
  }
  if (targetAxis < 0 || targetAxis >= f.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${f.shape}');
  }

  final N = f.shape[targetAxis];
  switch (spacing) {
    case CoordinateSpacing(x: final x):
      if (x.length != N) {
        throw ArgumentError(
          'Coordinate spacing length (${x.length}) must match dimension size ($N).',
        );
      }
    case ComplexCoordinateSpacing(z: final z):
      if (z.length != N) {
        throw ArgumentError(
          'Complex coordinate spacing length (${z.length}) must match dimension size ($N).',
        );
      }
    default:
      break;
  }

  final result = out ?? NDArray<T>.zeros(f.shape, f.dtype);
  if (out != null) {
    if (!_listEquals(out.shape, f.shape) || out.dtype != f.dtype) {
      throw ArgumentError('Incompatible out buffer shape or dtype.');
    }
  }

  final rank = f.shape.length;
  final cShape = malloc<ffi.Int>(rank);
  final cStridesF = malloc<ffi.Int>(rank);
  final cStridesRes = malloc<ffi.Int>(rank);

  for (var i = 0; i < rank; i++) {
    cShape[i] = f.shape[i];
    cStridesF[i] = f.strides[i];
    cStridesRes[i] = result.strides[i];
  }

  try {
    switch (spacing) {
      case ComplexStepSpacing():
      case ComplexCoordinateSpacing():
        NDArray<T>? spacingArray;
        if (f.dtype == DType.complex128) {
          final dxStruct = malloc<cpx_t>();
          try {
            if (spacing is ComplexStepSpacing) {
              dxStruct.ref.r = spacing.dz.real;
              dxStruct.ref.i = spacing.dz.imag;
            } else {
              dxStruct.ref.r = 1.0;
              dxStruct.ref.i = 0.0;
              spacingArray =
                  NDArray<Complex>.fromList(
                        (spacing as ComplexCoordinateSpacing).z,
                        [N],
                        DType.complex128,
                      )
                      as NDArray<T>;
            }
            s_gradient_complex128_all(
              f.pointer.cast(),
              cStridesF,
              spacingArray?.pointer.cast() ?? ffi.nullptr,
              spacingArray?.strides[0] ?? 0,
              dxStruct.ref,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              targetAxis,
              edgeOrder,
            );
          } finally {
            malloc.free(dxStruct);
          }
        } else if (f.dtype == DType.complex64) {
          final dxStruct = malloc<cpx_f_t>();
          try {
            if (spacing is ComplexStepSpacing) {
              dxStruct.ref.r = spacing.dz.real;
              dxStruct.ref.i = spacing.dz.imag;
            } else {
              dxStruct.ref.r = 1.0;
              dxStruct.ref.i = 0.0;
              spacingArray =
                  NDArray<Complex>.fromList(
                        (spacing as ComplexCoordinateSpacing).z,
                        [N],
                        DType.complex64,
                      )
                      as NDArray<T>;
            }
            s_gradient_complex64_all(
              f.pointer.cast(),
              cStridesF,
              spacingArray?.pointer.cast() ?? ffi.nullptr,
              spacingArray?.strides[0] ?? 0,
              dxStruct.ref,
              result.pointer.cast(),
              cStridesRes,
              cShape,
              rank,
              targetAxis,
              edgeOrder,
            );
          } finally {
            malloc.free(dxStruct);
          }
        }
        spacingArray?.dispose();

      case StepSpacing():
      case CoordinateSpacing():
        double dxVal = 1.0;
        NDArray? spacingArray;
        if (spacing is StepSpacing) {
          dxVal = spacing.dx;
        } else {
          spacingArray = NDArray<double>.fromList(
            (spacing as CoordinateSpacing).x,
            [N],
            f.dtype.isFloating ? f.dtype as DType<double> : DType.float64,
          );
        }

        final dtype = f.dtype;
        if (dtype == DType.float64) {
          s_gradient_double(
            f.pointer.cast(),
            cStridesF,
            spacingArray?.pointer.cast() ?? ffi.nullptr,
            spacingArray?.strides[0] ?? 0,
            dxVal,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            targetAxis,
            edgeOrder,
          );
        } else if (dtype == DType.float32) {
          s_gradient_float(
            f.pointer.cast(),
            cStridesF,
            spacingArray?.pointer.cast() ?? ffi.nullptr,
            spacingArray?.strides[0] ?? 0,
            dxVal.toDouble(),
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            targetAxis,
            edgeOrder,
          );
        } else if (dtype == DType.complex128) {
          s_gradient_complex128(
            f.pointer.cast(),
            cStridesF,
            spacingArray?.pointer.cast() ?? ffi.nullptr,
            spacingArray?.strides[0] ?? 0,
            dxVal,
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            targetAxis,
            edgeOrder,
          );
        } else if (dtype == DType.complex64) {
          s_gradient_complex64(
            f.pointer.cast(),
            cStridesF,
            spacingArray?.pointer.cast() ?? ffi.nullptr,
            spacingArray?.strides[0] ?? 0,
            dxVal.toDouble(),
            result.pointer.cast(),
            cStridesRes,
            cShape,
            rank,
            targetAxis,
            edgeOrder,
          );
        }
        spacingArray?.dispose();
    }
  } finally {
    malloc.free(cShape);
    malloc.free(cStridesF);
    malloc.free(cStridesRes);
  }

  return result;
}

/// Calculate the N-Dimensional gradient along multiple axes.
///
/// Returns a [List<NDArray>] containing the partial derivatives along each
/// specified [axis]. For a 1D array, this returns a list with a single element
/// equivalent to `gradient(f)`.
///
/// To calculate the gradient along a single specific axis, use [gradient].
///
/// Differentiates along the axes specified by [axis] (defaulting to all axes).
/// Returns a list of single-axis gradients.
///
/// Spacing along the axes can be specified in two ways:
/// - [spacing]: A single [Spacing] object applied to all axes (shortcut).
/// - [spacings]: A list of [Spacing] objects, one for each axis being differentiated.
/// - [edgeOrder]: Accuracy of the calculation at the boundaries (1 or 2).
///   See [gradient] for details.
///
/// If neither is provided, constant spacing [Spacing.step(1.0)] is used for all axes.
///
/// **Preconditions:**
/// - Input [f] must not be disposed.
/// - Input [f] must be a floating-point or complex type.
/// - If provided, [axis] elements must be unique and within bounds `[-f.rank, f.rank - 1]`.
/// - [spacing] and [spacings] are mutually exclusive.
/// - If provided, [spacings] length must match the number of axes being differentiated.
///
/// **Throws:**
/// - [StateError] if [f] is disposed.
/// - [ArgumentError] if [f] has an integer or boolean dtype.
/// - [ArgumentError] if [axis] contains out of bounds or duplicate indices.
/// - [ArgumentError] if both [spacing] and [spacings] are provided.
/// - [ArgumentError] if [spacings] length does not match the number of axes.
/// - [ArgumentError] if [edgeOrder] is not 1 or 2.
///
/// **Example:**
/// ```dart
/// final f = NDArray.fromList([1.0, 2.0, 4.0, 8.0], [2, 2], DType.float64);
/// // Shortcut for all axes:
/// final grads = gradientArray(f, spacing: Spacing.step(1.0));
/// // Specific per axis:
/// final grads2 = gradientArray(f, spacings: [Spacing.step(1.0), Spacing.step(2.0)]);
/// ```
List<NDArray<T>> gradientArray<T extends Object>(
  NDArray<T> f, {
  Spacing? spacing,
  List<Spacing>? spacings,
  List<int>? axis,
  int edgeOrder = 1,
}) {
  if (f.isDisposed) {
    throw StateError('Cannot execute gradientArray() on a disposed array.');
  }

  if (f.dtype.isInteger || f.dtype == DType.boolean) {
    throw ArgumentError(
      'Calculus operations are not supported on integer or boolean arrays. '
      'Cast to a floating-point or complex type first.',
    );
  }

  if (spacing != null && spacings != null) {
    throw ArgumentError('spacing and spacings are mutually exclusive.');
  }

  // Resolve axes
  final List<int> targetAxes;
  if (axis == null) {
    targetAxes = List<int>.generate(f.shape.length, (i) => i);
  } else {
    targetAxes = [];
    for (var ax in axis) {
      var resolvedAx = ax;
      if (resolvedAx < 0) {
        resolvedAx = f.shape.length + resolvedAx;
      }
      if (resolvedAx < 0 || resolvedAx >= f.shape.length) {
        throw ArgumentError(
          'axis index $ax out of bounds for shape ${f.shape}',
        );
      }
      if (targetAxes.contains(resolvedAx)) {
        throw ArgumentError('axis index $ax specified multiple times.');
      }
      targetAxes.add(resolvedAx);
    }
  }

  if (spacings != null && spacings.length != targetAxes.length) {
    throw ArgumentError(
      'spacings list length (${spacings.length}) must match the number of axes (${targetAxes.length}).',
    );
  }

  final List<NDArray<T>> results = [];
  try {
    for (var i = 0; i < targetAxes.length; i++) {
      final singleGrad = gradient<T>(
        f,
        spacing: spacings?[i] ?? spacing ?? const Spacing.step(1.0),
        axis: targetAxes[i],
        edgeOrder: edgeOrder,
      );
      results.add(singleGrad);
    }
  } catch (e) {
    // Clean up any successful allocations if one fails
    for (var res in results) {
      res.dispose();
    }
    rethrow;
  }

  return results;
}
