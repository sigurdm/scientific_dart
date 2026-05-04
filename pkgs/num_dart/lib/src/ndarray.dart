import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'dart:collection';
import 'broadcasting.dart';

/// Supported data types for the elements of an [NDArray].
enum DType { float32, float64, int32, int64, complex64, complex128 }

/// An n-dimensional array with memory allocated on the C heap.
///
/// **Memory Management Guidelines:**
/// - **Explicit Disposal**: Always call [dispose] explicitly as soon as an array is no
///   longer needed. While the garbage collector will eventually free C memory to prevent hard
///   leaks, it is blind to large native allocations, and garbage collection might not be
///   triggered early enough.
/// - **Views & Shared Memory**: Views (slices, reshapes, transposes, etc.) share the exact same
///   C memory as their parent; modifying a view mutates the parent and vice versa. Calling [dispose]
///   on a parent array immediately **invalidates all views** derived from it; accessing an
///   invalidated view causes crashes or undefined behavior.
///
/// **Example Usage:**
/// ```dart
/// // Create a 2x3 array filled with ones
/// final a = NDArray<double>.ones([2, 3], DType.float64);
/// print(a.data); // [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
///
/// // Explicitly free memory when done
/// a.dispose();
/// ```
class NDArray<T> implements ffi.Finalizable {
  /// Pointer to the raw C memory allocated for this array.
  final ffi.Pointer<ffi.Void> _pointer;

  /// A Dart list view of the raw C memory.
  ///
  /// **Restrictions:**
  /// - This list has a fixed length and cannot be resized.
  /// - This list becomes invalid as soon as the underlying C memory is freed (via `dispose()` or garbage collection). Accessing it afterwards leads to undefined behavior or crashes.
  final List<T> data;

  /// The dimensions of the n-dimensional array.
  final List<int> shape;

  /// The number of elements to skip in memory to move to the next position along each dimension.
  final List<int> strides;

  /// The data type of the elements in the array.
  final DType dtype;

  /// The parent array if this is a view, to prevent it from being garbage collected.
  final NDArray? _parent;

  static final _finalizer = ffi.NativeFinalizer(malloc.nativeFree);

  /// Private constructor for internal use and factories.
  NDArray._(
    this._pointer,
    this.data,
    this._parent, {
    required this.shape,
    required this.strides,
    required this.dtype,
  }) {
    if (_parent == null) {
      _finalizer.attach(this, _pointer, detach: this);
    }
  }

  /// Returns true if the array is C-contiguous in memory.
  bool get isContiguous {
    final cStrides = computeCStrides(shape);
    if (strides.length != cStrides.length) return false;
    for (var i = 0; i < strides.length; i++) {
      if (strides[i] != cStrides[i]) return false;
    }
    return true;
  }

  /// Factory to create a new array with allocated C memory.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray<double>.create([2, 2], DType.float64);
  /// ```
  factory NDArray.create(List<int> shape, DType dtype) {
    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    final strides = computeCStrides(shape);

    ffi.Pointer<ffi.Void> pointer;
    List<T> data;

    if (dtype == DType.float64) {
      final p = malloc<ffi.Double>(totalSize);
      pointer = p.cast();
      data = p.asTypedList(totalSize) as List<T>;
    } else if (dtype == DType.float32) {
      final p = malloc<ffi.Float>(totalSize);
      pointer = p.cast();
      data = p.asTypedList(totalSize) as List<T>;
    } else if (dtype == DType.int32) {
      final p = malloc<ffi.Int32>(totalSize);
      pointer = p.cast();
      data = p.asTypedList(totalSize) as List<T>;
    } else if (dtype == DType.int64) {
      final p = malloc<ffi.Int64>(totalSize);
      pointer = p.cast();
      data = p.asTypedList(totalSize) as List<T>;
    } else if (dtype == DType.complex128) {
      final p = malloc<ffi.Double>(totalSize * 2);
      pointer = p.cast();
      final doubleList = p.asTypedList(totalSize * 2);
      data = ComplexList(doubleList) as List<T>;
    } else if (dtype == DType.complex64) {
      final p = malloc<ffi.Float>(totalSize * 2);
      pointer = p.cast();
      final floatList = p.asTypedList(totalSize * 2);
      data = ComplexList(floatList) as List<T>;
    } else {
      throw UnimplementedError('Type $dtype not supported yet');
    }

    return NDArray._(
      pointer,
      data,
      null,
      shape: shape,
      strides: strides,
      dtype: dtype,
    );
  }

  /// Factory to create a C-contiguous array from a Dart list (copies data).
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
  /// ```
  factory NDArray.fromList(List<T> list, List<int> shape, DType dtype) {
    final arr = NDArray<T>.create(shape, dtype);
    arr.data.setAll(0, list);
    return arr;
  }

  /// Factory to create an array filled with zeros.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray<double>.zeros([2, 2], DType.float64);
  /// print(a.data); // [0.0, 0.0, 0.0, 0.0]
  /// ```
  factory NDArray.zeros(List<int> shape, DType dtype) {
    final arr = NDArray<T>.create(shape, dtype);
    if (dtype == DType.float32 || dtype == DType.float64) {
      arr.data.fillRange(0, arr.data.length, 0.0 as T);
    } else if (dtype == DType.complex128 || dtype == DType.complex64) {
      arr.data.fillRange(0, arr.data.length, Complex(0.0, 0.0) as T);
    } else {
      arr.data.fillRange(0, arr.data.length, 0 as T);
    }
    return arr;
  }

  /// Factory to create an array filled with ones.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray<double>.ones([2, 2], DType.float64);
  /// print(a.data); // [1.0, 1.0, 1.0, 1.0]
  /// ```
  factory NDArray.ones(List<int> shape, DType dtype) {
    final arr = NDArray<T>.create(shape, dtype);
    if (dtype == DType.float32 || dtype == DType.float64) {
      arr.data.fillRange(0, arr.data.length, 1.0 as T);
    } else if (dtype == DType.complex128 || dtype == DType.complex64) {
      arr.data.fillRange(0, arr.data.length, Complex(1.0, 0.0) as T);
    } else {
      arr.data.fillRange(0, arr.data.length, 1 as T);
    }
    return arr;
  }

  /// Factory to create an array with a range of values.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray<double>.arange(0.0, 5.0, step: 1.0, dtype: DType.float64);
  /// print(a.data); // [0.0, 1.0, 2.0, 3.0, 4.0]
  /// ```
  factory NDArray.arange(
    double start,
    double stop, {
    double step = 1.0,
    DType dtype = DType.float64,
  }) {
    final length = ((stop - start) / step).ceil();
    final arr = NDArray<T>.create([length], dtype);
    for (var i = 0; i < length; i++) {
      arr.data[i] = (start + i * step) as T;
    }
    return arr;
  }

  /// Factory to create an array with evenly spaced values.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray<double>.linspace(0.0, 1.0, 5, dtype: DType.float64);
  /// print(a.data); // [0.0, 0.25, 0.5, 0.75, 1.0]
  /// ```
  factory NDArray.linspace(
    double start,
    double stop,
    int num, {
    DType dtype = DType.float64,
  }) {
    if (num <= 0) throw ArgumentError('num must be positive');
    final arr = NDArray<T>.create([num], dtype);
    if (num == 1) {
      arr.data[0] = start as T;
      return arr;
    }
    final step = (stop - start) / (num - 1);
    for (var i = 0; i < num; i++) {
      arr.data[i] = (start + i * step) as T;
    }
    return arr;
  }

  /// Factory to create a 2D identity matrix.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray<double>.eye(3, DType.float64);
  /// print(a.data); // [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
  /// ```
  ///
  /// **Gotchas:**
  /// - This only creates 2D square matrices.
  factory NDArray.eye(int n, DType dtype) {
    final arr = NDArray<T>.zeros([n, n], dtype);
    for (var i = 0; i < n; i++) {
      if (dtype == DType.float32 || dtype == DType.float64) {
        arr.data[i * n + i] = 1.0 as T;
      } else {
        arr.data[i * n + i] = 1 as T;
      }
    }
    return arr;
  }

  /// Factory to create a view sharing the same memory.
  ///
  /// **Example:**
  /// ```dart
  /// final view = NDArray.view(parent, [2], [1], offsetElements: 1);
  /// ```
  ///
  /// **Restrictions:**
  /// - **Lifetime Dependency**: The view is only valid as long as the parent's memory is not freed. If you call `parent.dispose()`, this view becomes invalid.
  /// - **Shared Mutations**: Modifications to the view affect the parent and vice versa.
  /// - **No Ownership**: Calling `dispose()` on a view does nothing.
  factory NDArray.view(
    NDArray parent,
    List<int> shape,
    List<int> strides, {
    int offsetElements = 0,
  }) {
    ffi.Pointer<ffi.Void> pointer;
    List<T> data;

    // Calculate the offset pointer
    if (parent.dtype == DType.float64) {
      final p = parent._pointer.cast<ffi.Double>() + offsetElements;
      pointer = p.cast();
      data = p.asTypedList(parent.data.length - offsetElements) as List<T>;
    } else if (parent.dtype == DType.float32) {
      final p = parent._pointer.cast<ffi.Float>() + offsetElements;
      pointer = p.cast();
      data = p.asTypedList(parent.data.length - offsetElements) as List<T>;
    } else if (parent.dtype == DType.int32) {
      final p = parent._pointer.cast<ffi.Int32>() + offsetElements;
      pointer = p.cast();
      data = p.asTypedList(parent.data.length - offsetElements) as List<T>;
    } else if (parent.dtype == DType.int64) {
      final p = parent._pointer.cast<ffi.Int64>() + offsetElements;
      pointer = p.cast();
      data = p.asTypedList(parent.data.length - offsetElements) as List<T>;
    } else if (parent.dtype == DType.complex128) {
      final p = parent._pointer.cast<ffi.Double>() + (offsetElements * 2);
      pointer = p.cast();
      final doubleList = p.asTypedList(
        parent.data.length * 2 - offsetElements * 2,
      );
      data = ComplexList(doubleList) as List<T>;
    } else if (parent.dtype == DType.complex64) {
      final p = parent._pointer.cast<ffi.Float>() + (offsetElements * 2);
      pointer = p.cast();
      final floatList = p.asTypedList(
        parent.data.length * 2 - offsetElements * 2,
      );
      data = ComplexList(floatList) as List<T>;
    } else {
      throw UnimplementedError('Type ${parent.dtype} not supported yet');
    }

    return NDArray._(
      pointer,
      data,
      parent,
      shape: shape,
      strides: strides,
      dtype: parent.dtype,
    );
  }

  /// Helper to calculate default strides for a C-contiguous array (in elements).
  static List<int> computeCStrides(List<int> shape) {
    if (shape.isEmpty) return [];
    final strides = List<int>.filled(shape.length, 1);
    for (var i = shape.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  }

  /// Expose the raw pointer for FFI use.
  ffi.Pointer<ffi.Void> get pointer => _pointer;

  /// Returns a new view of this array with a new shape.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
  /// final b = a.reshape([2, 2]);
  /// print(b.shape); // [2, 2]
  /// ```
  ///
  /// **Gotchas:**
  /// - Returns a view sharing the same memory. Modifications reflect in both.
  /// - The total size of the new shape must match the old one.
  NDArray<T> reshape(List<int> newShape) {
    final oldSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    final newSize = newShape.isEmpty ? 1 : newShape.reduce((a, b) => a * b);
    if (oldSize != newSize) {
      throw ArgumentError(
        'Total size must not change during reshape (was $oldSize, new is $newSize)',
      );
    }

    final newStrides = computeCStrides(newShape);
    return NDArray._(
      _pointer,
      data,
      _parent ?? this,
      shape: newShape,
      strides: newStrides,
      dtype: dtype,
    );
  }

  /// Returns a new 1D array containing a copy of the elements.
  NDArray<T> flatten() {
    final flatList = toList();
    return NDArray.fromList(flatList, [flatList.length], dtype);
  }

  /// Returns a 1D array containing the elements, as a view if contiguous, or a copy.
  NDArray<T> ravel() {
    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    if (isContiguous) {
      return NDArray._(
        _pointer,
        data,
        _parent ?? this,
        shape: [totalSize],
        strides: [1],
        dtype: dtype,
      );
    } else {
      return flatten();
    }
  }

  /// Returns a view of the array with dimensions permuted.
  ///
  /// If [axes] is null, it reverses the order of the dimensions.
  NDArray<T> transpose([List<int>? axes]) {
    List<int> permutedAxes;
    if (axes == null) {
      permutedAxes = List.generate(shape.length, (i) => shape.length - 1 - i);
    } else {
      if (axes.length != shape.length) {
        throw ArgumentError('Axes must match the rank of the array');
      }
      final seen = <int>{};
      for (final axis in axes) {
        if (axis < 0 || axis >= shape.length) {
          throw RangeError.index(axis, shape, 'axis out of range');
        }
        if (seen.contains(axis)) {
          throw ArgumentError('Axes must be a permutation without duplicates');
        }
        seen.add(axis);
      }
      permutedAxes = axes;
    }

    final newShape = List<int>.filled(shape.length, 0);
    final newStrides = List<int>.filled(shape.length, 0);

    for (var i = 0; i < shape.length; i++) {
      newShape[i] = shape[permutedAxes[i]];
      newStrides[i] = strides[permutedAxes[i]];
    }

    return NDArray._(
      _pointer,
      data,
      _parent ?? this,
      shape: newShape,
      strides: newStrides,
      dtype: dtype,
    );
  }

  /// Returns a view of the array with dimensions reversed.
  NDArray<T> get transposed => transpose();

  /// Fetches the single scalar element at the specified multi-dimensional [coords].
  ///
  /// **Polymorphic Equivalence:**
  /// Equivalent to calling `this[coords]` via a flat list parameter.
  ///
  /// **Preconditions:**
  /// - [coords] length must match the rank of the array.
  ///
  /// **Throws:**
  /// - [ArgumentError] if `coords.length` doesn't match array rank.
  /// - [RangeError] if any coordinate is out of bounds for its dimension.
  T getCell(List<int> coords) {
    if (coords.length != shape.length) {
      throw ArgumentError(
        'Number of coordinates (${coords.length}) must match array rank (${shape.length})',
      );
    }
    var offset = 0;
    for (var i = 0; i < coords.length; i++) {
      final idx = coords[i];
      if (idx < 0 || idx >= shape[i]) {
        throw RangeError.range(
          idx,
          0,
          shape[i] - 1,
          'coordinate at dimension $i',
        );
      }
      offset += idx * strides[i];
    }
    return data[offset];
  }

  /// Sets the single scalar element at the specified multi-dimensional [coords] to [value].
  ///
  /// **Polymorphic Equivalence:**
  /// Equivalent to calling `this[coords] = value` via a flat list parameter.
  void setCell(List<int> coords, T value) {
    if (coords.length != shape.length) {
      throw ArgumentError(
        'Number of coordinates (${coords.length}) must match array rank (${shape.length})',
      );
    }
    var offset = 0;
    for (var i = 0; i < coords.length; i++) {
      final idx = coords[i];
      if (idx < 0 || idx >= shape[i]) {
        throw RangeError.range(
          idx,
          0,
          shape[i] - 1,
          'coordinate at dimension $i',
        );
      }
      offset += idx * strides[i];
    }
    data[offset] = value;
  }

  /// Modifies elements where the provided boolean binary [mask] contains `1`.
  ///
  /// **Polymorphic Equivalence:**
  /// Equivalent to calling `this[mask] = value`.
  ///
  /// **Preconditions:**
  /// - [mask] must share identical dimensions ([shape]) with this array.
  /// - [mask] values must be binary (`0` or `1`).
  ///
  /// **Arguments:**
  /// - [value]: Can be a single scalar of matching type (performs uniform clipping)
  ///   or an [NDArray] containing sequential values to write into the masked positions.
  void setByMask(NDArray<int> mask, dynamic value) {
    if (mask.shape.length != shape.length) {
      throw ArgumentError(
        'Mask shape length (${mask.shape.length}) must match array rank (${shape.length})',
      );
    }
    for (var i = 0; i < shape.length; i++) {
      if (mask.shape[i] != shape[i]) {
        throw ArgumentError(
          'Mask dimensions (${mask.shape}) must exactly match array shape ($shape)',
        );
      }
    }

    var valueIndex = 0;
    List? valData;
    if (value is NDArray) {
      valData = value.data;
    }

    void walk(int dim, int currentOffset, int maskOffset) {
      if (dim == shape.length) {
        if (mask.data[maskOffset] == 1) {
          if (valData != null) {
            if (valueIndex >= valData.length) {
              throw ArgumentError(
                'Source values array contains fewer elements than the mask targets',
              );
            }
            data[currentOffset] = valData[valueIndex++] as T;
          } else {
            data[currentOffset] = value as T;
          }
        }
        return;
      }

      for (var i = 0; i < shape[dim]; i++) {
        walk(
          dim + 1,
          currentOffset + i * strides[dim],
          maskOffset + i * mask.strides[dim],
        );
      }
    }

    walk(0, 0, 0);
  }

  /// Modifies entire sub-matrix rows or slices along the specified [axis] targeted by a 1D list of [indices], setting them all to a single [value].
  ///
  /// **Polymorphic Equivalence:**
  /// When [axis] is `0`, equivalent to calling `this[ [indices.data] ] = value` (fancy row stack scalar mutation).
  ///
  void setIndicesScalar(NDArray<int> indices, T value, {int axis = 0}) {
    if (axis < 0 || axis >= shape.length) {
      throw RangeError.range(axis, 0, shape.length - 1, 'axis');
    }

    final idxData = indices.data;
    final sliceShape = List<int>.from(shape)..removeAt(axis);
    final sliceStrides = List<int>.from(strides)..removeAt(axis);

    for (var idx = 0; idx < idxData.length; idx++) {
      final targetIdx = idxData[idx];
      if (targetIdx < 0 || targetIdx >= shape[axis]) {
        throw RangeError.range(
          targetIdx,
          0,
          shape[axis] - 1,
          'index entry at position $idx',
        );
      }

      void overwriteSlice(int dim, int currentOffset) {
        if (dim == sliceShape.length) {
          data[currentOffset] = value;
          return;
        }
        for (var i = 0; i < sliceShape[dim]; i++) {
          overwriteSlice(dim + 1, currentOffset + i * sliceStrides[dim]);
        }
      }

      overwriteSlice(0, targetIdx * strides[axis]);
    }
  }

  /// Modifies entire sub-matrix rows or slices along the specified [axis] targeted by a 1D list of [indices], overwriting them with sequential values from [values].
  ///
  /// **Polymorphic Equivalence:**
  /// When [axis] is `0`, equivalent to calling `this[ [indices.data] ] = values` (fancy row stack array assignment).
  ///
  void setIndices(NDArray<int> indices, NDArray values, {int axis = 0}) {
    if (axis < 0 || axis >= shape.length) {
      throw RangeError.range(axis, 0, shape.length - 1, 'axis');
    }

    final idxData = indices.data;
    final sliceShape = List<int>.from(shape)..removeAt(axis);
    final sliceStrides = List<int>.from(strides)..removeAt(axis);

    final valData = values.data;
    var valOffset = 0;

    for (var idx = 0; idx < idxData.length; idx++) {
      final targetIdx = idxData[idx];
      if (targetIdx < 0 || targetIdx >= shape[axis]) {
        throw RangeError.range(
          targetIdx,
          0,
          shape[axis] - 1,
          'index entry at position $idx',
        );
      }

      void writeSlice(int dim, int currentOffset) {
        if (dim == sliceShape.length) {
          if (valOffset >= valData.length) {
            throw ArgumentError(
              'Source values array contains fewer elements than required for the fancy index allocation',
            );
          }
          data[currentOffset] = valData[valOffset++] as T;
          return;
        }
        for (var i = 0; i < sliceShape[dim]; i++) {
          writeSlice(dim + 1, currentOffset + i * sliceStrides[dim]);
        }
      }

      writeSlice(0, targetIdx * strides[axis]);
    }
  }

  /// Accesses elements of the array polymorphically based on the runtime type of [spec].
  ///
  /// **Behavior by Parameter Type:**
  /// - **[int]**: Equivalent to calling `slice([Index(spec)])`. Extracts a view along the first axis.
  /// - **`List<int>`**: Equivalent to calling `getCell(spec)`. Fetches a single coordinate cell scalar.
  /// - **`List<List<int>>`**: Equivalent to calling `take(spec[0], axis: 0)`. Fetches sub-matrix row slices.
  /// - **`NDArray<int>`**:
  ///   - If shapes match exactly (`spec.shape == shape`), equivalent to calling `applyMask(spec)`.
  ///   - If shapes differ, equivalent to calling `take(spec.data, axis: 0)`.
  ///
  /// Throws an [ArgumentError] if the type of [spec] is unsupported.
  dynamic operator [](dynamic spec) {
    if (spec is int) {
      return slice([Index(spec)]);
    } else if (spec is List) {
      if (spec.isNotEmpty && spec.first is List) {
        final subList = spec.first as List;
        final intIndices = subList.map((e) => e as int).toList();
        return take(intIndices);
      } else {
        final intCoords = spec.map((e) => e as int).toList();
        if (intCoords.length != shape.length) {
          throw ArgumentError(
            'Number of coordinate indices (${intCoords.length}) must match array rank (${shape.length})',
          );
        }
        return getCell(intCoords);
      }
    } else if (spec is NDArray<int>) {
      var shapesMatch = spec.shape.length == shape.length;
      if (shapesMatch) {
        for (var i = 0; i < shape.length; i++) {
          if (spec.shape[i] != shape[i]) {
            shapesMatch = false;
            break;
          }
        }
      }
      if (shapesMatch) {
        return applyMask(spec);
      } else {
        return take(spec.data);
      }
    } else {
      throw ArgumentError(
        'Unsupported selector type for operator []: ${spec.runtimeType}',
      );
    }
  }

  /// Mutates elements of the array polymorphically based on the runtime type of [spec].
  ///
  /// **Behavior by Parameter Type:**
  /// - **[int]**: Modifies an entire row or slice along the first axis via [setIndices] / [setIndicesScalar].
  /// - **`List<int>`**: Modifies a single coordinate cell scalar via [setCell].
  /// - **`List<List<int>>`**: Modifies targeted row slices along the first axis via [setIndices] / [setIndicesScalar].
  /// - **`NDArray<int>`**:
  ///   - If shapes match exactly, equivalent to calling `setByMask(spec, value)`.
  ///   - If shapes differ, equivalent to calling `setIndices(spec, value)` or `setIndicesScalar(spec, value)`.
  ///
  /// Throws an [ArgumentError] if the type of [spec] is unsupported.
  void operator []=(dynamic spec, dynamic value) {
    if (spec is int) {
      final indices = NDArray<int>.fromList([spec], [1], DType.int32);
      if (value is NDArray) {
        setIndices(indices, value);
      } else {
        setIndicesScalar(indices, value as T);
      }
    } else if (spec is List) {
      if (spec.isNotEmpty && spec.first is List) {
        final subList = spec.first as List;
        final intIndices = subList.map((e) => e as int).toList();
        final indices = NDArray<int>.fromList(intIndices, [
          intIndices.length,
        ], DType.int32);
        if (value is NDArray) {
          setIndices(indices, value);
        } else {
          setIndicesScalar(indices, value as T);
        }
      } else {
        final intCoords = spec.map((e) => e as int).toList();
        if (intCoords.length != shape.length) {
          throw ArgumentError(
            'Number of coordinate indices (${intCoords.length}) must match array rank (${shape.length})',
          );
        }
        setCell(intCoords, value as T);
      }
    } else if (spec is NDArray<int>) {
      var shapesMatch = spec.shape.length == shape.length;
      if (shapesMatch) {
        for (var i = 0; i < shape.length; i++) {
          if (spec.shape[i] != shape[i]) {
            shapesMatch = false;
            break;
          }
        }
      }
      if (shapesMatch) {
        setByMask(spec, value);
      } else {
        if (value is NDArray) {
          setIndices(spec, value);
        } else {
          setIndicesScalar(spec, value as T);
        }
      }
    } else {
      throw ArgumentError(
        'Unsupported selector type for operator []=: ${spec.runtimeType}',
      );
    }
  }

  void _compareOpRec<Ta, Tb>(
    List<int> result,
    List<Ta> a,
    List<Tb> b,
    List<int> shape,
    List<int> stridesA,
    List<int> stridesB,
    List<int> stridesResult,
    int dim,
    int offsetA,
    int offsetB,
    int offsetResult,
    bool Function(dynamic, dynamic) predicate,
  ) {
    if (dim == shape.length) {
      result[offsetResult] = predicate(a[offsetA], b[offsetB]) ? 1 : 0;
      return;
    }

    for (var i = 0; i < shape[dim]; i++) {
      _compareOpRec<Ta, Tb>(
        result,
        a,
        b,
        shape,
        stridesA,
        stridesB,
        stridesResult,
        dim + 1,
        offsetA + i * stridesA[dim],
        offsetB + i * stridesB[dim],
        offsetResult + i * stridesResult[dim],
        predicate,
      );
    }
  }

  void _dispatchCompare(
    List<int> rData,
    NDArray a,
    NDArray b,
    List<int> shape,
    List<int> sA,
    List<int> sB,
    List<int> sR,
    bool Function(dynamic, dynamic) predicate,
  ) {
    if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
      final aData = a.data as List<Complex>;
      if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
        _compareOpRec<Complex, Complex>(
          rData,
          aData,
          b.data as List<Complex>,
          shape,
          sA,
          sB,
          sR,
          0,
          0,
          0,
          0,
          predicate,
        );
      } else if (b.dtype == DType.float64 || b.dtype == DType.float32) {
        _compareOpRec<Complex, double>(
          rData,
          aData,
          b.data as List<double>,
          shape,
          sA,
          sB,
          sR,
          0,
          0,
          0,
          0,
          predicate,
        );
      } else {
        _compareOpRec<Complex, int>(
          rData,
          aData,
          b.data as List<int>,
          shape,
          sA,
          sB,
          sR,
          0,
          0,
          0,
          0,
          predicate,
        );
      }
    } else if (a.dtype == DType.float64 || a.dtype == DType.float32) {
      final aData = a.data as List<double>;
      if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
        _compareOpRec<double, Complex>(
          rData,
          aData,
          b.data as List<Complex>,
          shape,
          sA,
          sB,
          sR,
          0,
          0,
          0,
          0,
          predicate,
        );
      } else if (b.dtype == DType.float64 || b.dtype == DType.float32) {
        _compareOpRec<double, double>(
          rData,
          aData,
          b.data as List<double>,
          shape,
          sA,
          sB,
          sR,
          0,
          0,
          0,
          0,
          predicate,
        );
      } else {
        _compareOpRec<double, int>(
          rData,
          aData,
          b.data as List<int>,
          shape,
          sA,
          sB,
          sR,
          0,
          0,
          0,
          0,
          predicate,
        );
      }
    } else {
      final aData = a.data as List<int>;
      if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
        _compareOpRec<int, Complex>(
          rData,
          aData,
          b.data as List<Complex>,
          shape,
          sA,
          sB,
          sR,
          0,
          0,
          0,
          0,
          predicate,
        );
      } else if (b.dtype == DType.float64 || b.dtype == DType.float32) {
        _compareOpRec<int, double>(
          rData,
          aData,
          b.data as List<double>,
          shape,
          sA,
          sB,
          sR,
          0,
          0,
          0,
          0,
          predicate,
        );
      } else {
        _compareOpRec<int, int>(
          rData,
          aData,
          b.data as List<int>,
          shape,
          sA,
          sB,
          sR,
          0,
          0,
          0,
          0,
          predicate,
        );
      }
    }
  }

  NDArray _wrapScalar(dynamic value, List<int> targetShape) {
    if (value is Complex) {
      return NDArray.fromList(
        <Complex>[value],
        List.filled(targetShape.length, 1),
        DType.complex128,
      );
    } else if (value is int) {
      return NDArray.fromList(
        <int>[value],
        List.filled(targetShape.length, 1),
        DType.int64,
      );
    } else if (value is double) {
      return NDArray.fromList(
        <double>[value],
        List.filled(targetShape.length, 1),
        DType.float64,
      );
    } else {
      throw ArgumentError('Unsupported scalar type: ${value.runtimeType}');
    }
  }

  /// Element-wise greater than comparison with full broadcasting support.
  ///
  /// **Example:**
  /// {@example /example/ufuncs_example.dart lang=dart}
  NDArray<int> operator >(dynamic other) {
    NDArray otherArr;
    if (other is NDArray) {
      otherArr = other;
    } else {
      otherArr = _wrapScalar(other, shape);
    }

    if (dtype == DType.complex128 ||
        dtype == DType.complex64 ||
        otherArr.dtype == DType.complex128 ||
        otherArr.dtype == DType.complex64) {
      throw UnsupportedError(
        'Complex numbers do not support inequality comparisons',
      );
    }

    final broadcastResult = broadcast(this, otherArr);
    final commonShape = broadcastResult.shape;
    final result = NDArray<int>.create(commonShape, DType.int32);
    final resultStrides = computeCStrides(commonShape);

    _dispatchCompare(
      result.data,
      this,
      otherArr,
      commonShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      resultStrides,
      (x, y) => (x as num) > (y as num),
    );
    return result;
  }

  /// Element-wise less than comparison with full broadcasting support.
  ///
  /// **Example:**
  /// {@example /example/ufuncs_example.dart lang=dart}
  NDArray<int> operator <(dynamic other) {
    NDArray otherArr;
    if (other is NDArray) {
      otherArr = other;
    } else {
      otherArr = _wrapScalar(other, shape);
    }

    if (dtype == DType.complex128 ||
        dtype == DType.complex64 ||
        otherArr.dtype == DType.complex128 ||
        otherArr.dtype == DType.complex64) {
      throw UnsupportedError(
        'Complex numbers do not support inequality comparisons',
      );
    }

    final broadcastResult = broadcast(this, otherArr);
    final commonShape = broadcastResult.shape;
    final result = NDArray<int>.create(commonShape, DType.int32);
    final resultStrides = computeCStrides(commonShape);

    _dispatchCompare(
      result.data,
      this,
      otherArr,
      commonShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      resultStrides,
      (x, y) => (x as num) < (y as num),
    );
    return result;
  }

  /// Element-wise greater-or-equal comparison with full broadcasting support.
  ///
  /// **Example:**
  /// {@example /example/ufuncs_example.dart lang=dart}
  NDArray<int> operator >=(dynamic other) {
    NDArray otherArr;
    if (other is NDArray) {
      otherArr = other;
    } else {
      otherArr = _wrapScalar(other, shape);
    }

    if (dtype == DType.complex128 ||
        dtype == DType.complex64 ||
        otherArr.dtype == DType.complex128 ||
        otherArr.dtype == DType.complex64) {
      throw UnsupportedError(
        'Complex numbers do not support inequality comparisons',
      );
    }

    final broadcastResult = broadcast(this, otherArr);
    final commonShape = broadcastResult.shape;
    final result = NDArray<int>.create(commonShape, DType.int32);
    final resultStrides = computeCStrides(commonShape);

    _dispatchCompare(
      result.data,
      this,
      otherArr,
      commonShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      resultStrides,
      (x, y) => (x as num) >= (y as num),
    );
    return result;
  }

  /// Element-wise less-or-equal comparison with full broadcasting support.
  ///
  /// **Example:**
  /// {@example /example/ufuncs_example.dart lang=dart}
  NDArray<int> operator <=(dynamic other) {
    NDArray otherArr;
    if (other is NDArray) {
      otherArr = other;
    } else {
      otherArr = _wrapScalar(other, shape);
    }

    if (dtype == DType.complex128 ||
        dtype == DType.complex64 ||
        otherArr.dtype == DType.complex128 ||
        otherArr.dtype == DType.complex64) {
      throw UnsupportedError(
        'Complex numbers do not support inequality comparisons',
      );
    }

    final broadcastResult = broadcast(this, otherArr);
    final commonShape = broadcastResult.shape;
    final result = NDArray<int>.create(commonShape, DType.int32);
    final resultStrides = computeCStrides(commonShape);

    _dispatchCompare(
      result.data,
      this,
      otherArr,
      commonShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      resultStrides,
      (x, y) => (x as num) <= (y as num),
    );
    return result;
  }

  /// Element-wise equality comparison with full broadcasting support.
  ///
  /// **Example:**
  /// {@example /example/ufuncs_example.dart lang=dart}
  NDArray<int> eq(dynamic other) {
    NDArray otherArr;
    if (other is NDArray) {
      otherArr = other;
    } else {
      otherArr = _wrapScalar(other, shape);
    }

    final broadcastResult = broadcast(this, otherArr);
    final commonShape = broadcastResult.shape;
    final result = NDArray<int>.create(commonShape, DType.int32);
    final resultStrides = computeCStrides(commonShape);

    _dispatchCompare(
      result.data,
      this,
      otherArr,
      commonShape,
      broadcastResult.stridesA,
      broadcastResult.stridesB,
      resultStrides,
      (x, y) => x == y,
    );
    return result;
  }

  /// Returns a view of this array with elements sliced based on [selectors].
  ///
  /// [selectors] can contain integers (to select a single index and reduce rank)
  /// or [Slice] objects (to select a range and keep rank).
  ///
  /// **Example:**
  /// ```dart
  /// final view = arr.slice([Slice(1, 3), 2]);
  /// ```
  /// Returns a view or copy of the array with elements sliced based on [selectors].
  ///
  /// [selectors] must contain instances of [Selector] subclasses: [Index], [Slice], [Indices], [Mask].
  NDArray<T> slice(List<Selector> selectors) {
    if (selectors.length > shape.length) {
      throw ArgumentError('Too many selectors for array rank');
    }

    final newShape = <int>[];
    final newStrides = <int>[];
    var offsetElements = 0;
    var isAdvanced = false;

    final processedSelectors = List<Selector>.from(selectors);
    for (var i = 0; i < processedSelectors.length; i++) {
      final sel = processedSelectors[i];
      if (sel is Mask) {
        final mask = sel.mask;
        if (mask.mask.shape.length != 1 || mask.mask.shape[0] != shape[i]) {
          throw ArgumentError(
            'Boolean mask shape must match the size of dimension $i',
          );
        }
        final indices = <int>[];
        final maskData = mask.mask.toList();
        for (var j = 0; j < maskData.length; j++) {
          if (maskData[j] == 1) indices.add(j);
        }
        processedSelectors[i] = Indices(indices);
      }
    }

    for (var i = 0; i < shape.length; i++) {
      final selector = i < processedSelectors.length
          ? processedSelectors[i]
          : Slice.all();

      if (selector is Index) {
        final idx = selector.value < 0
            ? shape[i] + selector.value
            : selector.value;
        if (idx < 0 || idx >= shape[i]) {
          throw RangeError.index(
            idx,
            shape,
            'index out of range for dimension $i',
          );
        }
        offsetElements += idx * strides[i];
        // Rank reduction: don't add to newShape or newStrides
      } else if (selector is Slice) {
        final start = selector.start ?? (selector.step > 0 ? 0 : shape[i] - 1);
        final stop = selector.stop ?? (selector.step > 0 ? shape[i] : -1);
        final step = selector.step;

        final startIdx = start < 0 ? shape[i] + start : start;
        final stopIdx = stop < 0 ? shape[i] + stop : stop;

        // Bound checks
        final realStart = startIdx.clamp(0, shape[i] - 1);
        final realStop = stopIdx.clamp(-1, shape[i]);

        final dimSize = ((realStop - realStart) / step).ceil();
        if (dimSize <= 0) {
          newShape.add(0);
          newStrides.add(0);
        } else {
          newShape.add(dimSize);
          newStrides.add(strides[i] * step);
          offsetElements += realStart * strides[i];
        }
      } else if (selector is Indices) {
        isAdvanced = true;
        newShape.add(selector.values.length);
        newStrides.add(0); // Dummy value for now
      }
    }

    if (isAdvanced) {
      final result = NDArray<T>.create(newShape, dtype);
      _copyAdvancedRecursive(
        this,
        result,
        processedSelectors,
        List<int>.filled(shape.length, 0),
        List<int>.filled(newShape.length, 0),
        0,
        0,
      );
      return result;
    }

    return NDArray.view(
      this,
      newShape,
      newStrides,
      offsetElements: offsetElements,
    );
  }

  /// Selects elements along an [axis] using a list of [indices].
  ///
  /// This method corresponds to NumPy's `take` function.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
  /// final b = a.take([0, 1], axis: 1); // Select columns 0 and 1
  /// ```
  NDArray<T> take(List<int> indices, {int axis = 0}) {
    if (axis < 0 || axis >= shape.length) {
      throw RangeError.index(axis, shape, 'axis out of range');
    }
    final selectors = List<Selector>.filled(shape.length, Slice.all());
    selectors[axis] = Indices(indices);
    return slice(selectors);
  }

  /// Selects elements matching a boolean [mask].
  ///
  /// The [mask] array must have elements with value 0 or 1.
  /// Returns a 1D array containing the elements where the mask is 1.
  NDArray<T> applyMask(NDArray<int> mask) {
    return slice([Mask(BooleanMask(mask))]);
  }

  void _copyAdvancedRecursive(
    NDArray<T> src,
    NDArray<T> dest,
    List<Selector> selectors,
    List<int> srcIndices,
    List<int> destIndices,
    int srcDim,
    int destDim,
  ) {
    if (srcDim == src.shape.length) {
      dest.setCell(destIndices, src.getCell(srcIndices));
      return;
    }

    final selector = srcDim < selectors.length
        ? selectors[srcDim]
        : Slice.all();

    if (selector is Index) {
      final idx = selector.value < 0
          ? src.shape[srcDim] + selector.value
          : selector.value;
      srcIndices[srcDim] = idx;
      _copyAdvancedRecursive(
        src,
        dest,
        selectors,
        srcIndices,
        destIndices,
        srcDim + 1,
        destDim,
      );
    } else if (selector is Slice) {
      final start =
          selector.start ?? (selector.step > 0 ? 0 : src.shape[srcDim] - 1);
      final stop =
          selector.stop ?? (selector.step > 0 ? src.shape[srcDim] : -1);
      final step = selector.step;
      final startIdx = start < 0 ? src.shape[srcDim] + start : start;
      final realStart = startIdx.clamp(0, src.shape[srcDim] - 1);

      var destIdx = 0;
      for (
        var idx = realStart;
        selector.step > 0 ? idx < stop : idx > stop;
        idx += step
      ) {
        srcIndices[srcDim] = idx;
        destIndices[destDim] = destIdx;
        _copyAdvancedRecursive(
          src,
          dest,
          selectors,
          srcIndices,
          destIndices,
          srcDim + 1,
          destDim + 1,
        );
        destIdx++;
      }
    } else if (selector is Indices) {
      for (var i = 0; i < selector.values.length; i++) {
        final idx = selector.values[i];
        final realIdx = idx < 0 ? src.shape[srcDim] + idx : idx;
        if (realIdx < 0 || realIdx >= src.shape[srcDim]) {
          throw RangeError.index(
            realIdx,
            src.shape,
            'index out of range for dimension $srcDim',
          );
        }
        srcIndices[srcDim] = realIdx;
        destIndices[destDim] = i;
        _copyAdvancedRecursive(
          src,
          dest,
          selectors,
          srcIndices,
          destIndices,
          srcDim + 1,
          destDim + 1,
        );
      }
    }
  }

  /// Returns a flat list copy of the elements in this array, respecting shape and strides.
  List<T> toList() {
    final result = <T>[];
    _fillListRecursive(this, List<int>.filled(shape.length, 0), 0, result);
    return result;
  }

  void _fillListRecursive(
    NDArray<T> arr,
    List<int> indices,
    int dim,
    List<T> result,
  ) {
    if (dim == arr.shape.length) {
      result.add(arr.getCell(indices));
      return;
    }
    for (var i = 0; i < arr.shape[dim]; i++) {
      indices[dim] = i;
      _fillListRecursive(arr, indices, dim + 1, result);
    }
  }

  /// Returns a new view of this array with a new dimension of size 1 inserted at [axis].
  ///
  /// This method corresponds to NumPy's `expand_dims` function. It does not copy the
  /// underlying memory; it returns a lightweight view of the same array with updated
  /// shape and strides.
  ///
  /// **Preconditions:**
  /// - [axis] must be within the range `[-rank - 1, rank]`, where `rank` is the rank
  ///   (number of dimensions) of this array.
  ///
  /// **Throws:**
  /// - [RangeError] if [axis] is out of bounds.
  ///
  /// **Example:**
  /// {@example /example/shape_examples.dart lang=dart}
  NDArray<T> expandDims(int axis) {
    final rank = shape.length;
    if (axis < -rank - 1 || axis > rank) {
      throw RangeError.range(
        axis,
        -rank - 1,
        rank,
        'axis',
        'Axis out of range for expandDims',
      );
    }

    final normAxis = axis < 0 ? rank + 1 + axis : axis;

    final newShape = List<int>.from(shape);
    final newStrides = List<int>.from(strides);

    newShape.insert(normAxis, 1);

    if (normAxis == rank) {
      newStrides.insert(normAxis, 1);
    } else {
      newStrides.insert(normAxis, strides[normAxis]);
    }

    return NDArray.view(this, newShape, newStrides);
  }

  /// Returns a new view of this array with single-dimensional entries removed from the shape.
  ///
  /// This method corresponds to NumPy's `squeeze` function. It returns a view sharing the
  /// same memory.
  ///
  /// Squeezes either all dimensions of size 1 (if [axis] is omitted/null), or only specific
  /// axes (if [axis] is an `int` or `List<int>`).
  ///
  /// **Preconditions:**
  /// - If an [axis] is specified, the target dimension(s) must have size equal to 1.
  /// - [axis] (or components of it) must be within `[-rank, rank - 1]`.
  ///
  /// **Throws:**
  /// - [RangeError] if any specified axis is out of range.
  /// - [ArgumentError] if a specified axis has a size greater than 1.
  ///
  /// **Example:**
  /// {@example /example/shape_examples.dart lang=dart}
  NDArray<T> squeeze({dynamic axis}) {
    final rank = shape.length;
    final axesToRemove = <int>{};

    if (axis == null) {
      for (var i = 0; i < rank; i++) {
        if (shape[i] == 1) {
          axesToRemove.add(i);
        }
      }
    } else if (axis is int) {
      if (axis < -rank || axis >= rank) {
        throw RangeError.range(axis, -rank, rank - 1, 'axis');
      }
      final normAxis = axis < 0 ? rank + axis : axis;
      if (shape[normAxis] != 1) {
        throw ArgumentError(
          'Cannot squeeze axis $axis: size is ${shape[normAxis]}, must be 1',
        );
      }
      axesToRemove.add(normAxis);
    } else if (axis is List<int>) {
      for (final ax in axis) {
        if (ax < -rank || ax >= rank) {
          throw RangeError.range(ax, -rank, rank - 1, 'axis');
        }
        final normAxis = ax < 0 ? rank + ax : ax;
        if (shape[normAxis] != 1) {
          throw ArgumentError(
            'Cannot squeeze axis $ax: size is ${shape[normAxis]}, must be 1',
          );
        }
        axesToRemove.add(normAxis);
      }
    } else {
      throw ArgumentError('axis must be null, int, or List<int>');
    }

    final newShape = <int>[];
    final newStrides = <int>[];

    for (var i = 0; i < rank; i++) {
      if (!axesToRemove.contains(i)) {
        newShape.add(shape[i]);
        newStrides.add(strides[i]);
      }
    }

    return NDArray.view(this, newShape, newStrides);
  }

  /// Returns a new view of this array with [axis1] and [axis2] interchanged.
  ///
  /// This method corresponds to NumPy's `swapaxes` function.
  ///
  /// **Preconditions:**
  /// - Both [axis1] and [axis2] must be within `[-rank, rank - 1]`.
  ///
  /// **Throws:**
  /// - [RangeError] if either axis is out of bounds.
  ///
  /// **Example:**
  /// {@example /example/shape_examples.dart lang=dart}
  NDArray<T> swapaxes(int axis1, int axis2) {
    final rank = shape.length;
    if (axis1 < -rank || axis1 >= rank) {
      throw RangeError.range(axis1, -rank, rank - 1, 'axis1');
    }
    if (axis2 < -rank || axis2 >= rank) {
      throw RangeError.range(axis2, -rank, rank - 1, 'axis2');
    }

    final norm1 = axis1 < 0 ? rank + axis1 : axis1;
    final norm2 = axis2 < 0 ? rank + axis2 : axis2;

    if (norm1 == norm2) return this;

    final newShape = List<int>.from(shape);
    final newStrides = List<int>.from(strides);

    final tempShape = newShape[norm1];
    newShape[norm1] = newShape[norm2];
    newShape[norm2] = tempShape;

    final tempStride = newStrides[norm1];
    newStrides[norm1] = newStrides[norm2];
    newStrides[norm2] = tempStride;

    return NDArray.view(this, newShape, newStrides);
  }

  /// Returns a new view of this array with axes moved from [source] positions to [destination] positions.
  ///
  /// This method corresponds to NumPy's `moveaxis` function. Other axes remain in their original
  /// relative order.
  ///
  /// **Preconditions:**
  /// - [source] and [destination] can be `int` or `List<int>`. If lists, they must have the same length.
  /// - All axis indices must be within `[-rank, rank - 1]`.
  /// - No duplicate axes can be specified in [source] or [destination].
  ///
  /// **Throws:**
  /// - [RangeError] if an axis index is out of range.
  /// - [ArgumentError] if inputs have mismatched lengths or contain duplicates.
  ///
  /// **Example:**
  /// {@example /example/shape_examples.dart lang=dart}
  NDArray<T> moveaxis(dynamic source, dynamic destination) {
    final rank = shape.length;

    List<int> srcList;
    List<int> destList;

    if (source is int && destination is int) {
      srcList = [source];
      destList = [destination];
    } else if (source is List<int> && destination is List<int>) {
      if (source.length != destination.length) {
        throw ArgumentError(
          'source and destination lists must have the same length',
        );
      }
      srcList = List<int>.from(source);
      destList = List<int>.from(destination);
    } else {
      throw ArgumentError(
        'source and destination must be both ints or both List<int>',
      );
    }

    final normSrc = <int>[];
    final normDest = <int>[];

    for (var i = 0; i < srcList.length; i++) {
      final s = srcList[i];
      final d = destList[i];

      if (s < -rank || s >= rank)
        throw RangeError.range(s, -rank, rank - 1, 'source');
      if (d < -rank || d >= rank)
        throw RangeError.range(d, -rank, rank - 1, 'destination');

      normSrc.add(s < 0 ? rank + s : s);
      normDest.add(d < 0 ? rank + d : d);
    }

    if (normSrc.toSet().length != normSrc.length) {
      throw ArgumentError('Duplicate axes in source are not allowed');
    }
    if (normDest.toSet().length != normDest.length) {
      throw ArgumentError('Duplicate axes in destination are not allowed');
    }

    final remaining = <int>[];
    for (var i = 0; i < rank; i++) {
      if (!normSrc.contains(i)) {
        remaining.add(i);
      }
    }

    final newOrder = List<int>.filled(rank, -1);

    for (var i = 0; i < normDest.length; i++) {
      newOrder[normDest[i]] = normSrc[i];
    }

    var remIdx = 0;
    for (var i = 0; i < rank; i++) {
      if (newOrder[i] == -1) {
        newOrder[i] = remaining[remIdx++];
      }
    }

    final newShape = List<int>.filled(rank, 0);
    final newStrides = List<int>.filled(rank, 0);

    for (var i = 0; i < rank; i++) {
      newShape[i] = shape[newOrder[i]];
      newStrides[i] = strides[newOrder[i]];
    }

    return NDArray.view(this, newShape, newStrides);
  }

  /// Manually free the allocated C memory.
  ///
  /// This method detaches the finalizer to prevent double-freeing.
  /// Calling this on a view does nothing, as the memory is owned by the parent.
  void dispose() {
    if (_parent != null) return; // Views don't own memory
    _finalizer.detach(this);
    malloc.free(_pointer);
  }
}

/// A wrapper class for boolean masks used in advanced indexing.
final class BooleanMask {
  /// The underlying boolean array represented as 0s and 1s.
  final NDArray<int> mask;

  /// Creates a new boolean mask. Precondition: elements must be 0 or 1.
  BooleanMask(this.mask) {
    if (mask.dtype != DType.int32) {
      throw ArgumentError('Boolean mask must have DType.int32');
    }
    // Precondition check: elements must be 0 or 1!
    // This is slow but ensures correctness!
    final data = mask.toList();
    for (final val in data) {
      if (val != 0 && val != 1) {
        throw ArgumentError('Boolean mask elements must be 0 or 1');
      }
    }
  }
}

/// Represents a complex number with double precision real and imaginary parts.
final class Complex {
  final double real;
  final double imag;

  Complex(this.real, this.imag);

  Complex operator +(dynamic other) {
    if (other is Complex) {
      return Complex(real + other.real, imag + other.imag);
    } else if (other is num) {
      return Complex(real + other.toDouble(), imag);
    } else {
      throw ArgumentError(
        'Unsupported operand type for +: ${other.runtimeType}',
      );
    }
  }

  Complex operator -(dynamic other) {
    if (other is Complex) {
      return Complex(real - other.real, imag - other.imag);
    } else if (other is num) {
      return Complex(real - other.toDouble(), imag);
    } else {
      throw ArgumentError(
        'Unsupported operand type for -: ${other.runtimeType}',
      );
    }
  }

  Complex operator *(dynamic other) {
    if (other is Complex) {
      return Complex(
        real * other.real - imag * other.imag,
        real * other.imag + imag * other.real,
      );
    } else if (other is num) {
      final val = other.toDouble();
      return Complex(real * val, imag * val);
    } else {
      throw ArgumentError(
        'Unsupported operand type for *: ${other.runtimeType}',
      );
    }
  }

  Complex operator /(dynamic other) {
    if (other is Complex) {
      final div = other.real * other.real + other.imag * other.imag;
      if (div == 0) throw ArgumentError('Division by zero');
      return Complex(
        (real * other.real + imag * other.imag) / div,
        (imag * other.real - real * other.imag) / div,
      );
    } else if (other is num) {
      final val = other.toDouble();
      if (val == 0) throw ArgumentError('Division by zero');
      return Complex(real / val, imag / val);
    } else {
      throw ArgumentError(
        'Unsupported operand type for /: ${other.runtimeType}',
      );
    }
  }

  @override
  String toString() => '$real + ${imag}i';

  @override
  bool operator ==(Object other) =>
      other is Complex && real == other.real && imag == other.imag;

  @override
  int get hashCode => Object.hash(real, imag);
}

/// A list view of complex numbers backed by a flat list of doubles.
class ComplexList extends ListBase<Complex> {
  final List<double> _list;
  ComplexList(this._list);

  /// Returns the backing list of doubles.
  List<double> get backingList => _list;

  @override
  int get length => _list.length ~/ 2;

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot resize ComplexList');
  }

  @override
  Complex operator [](int index) {
    return Complex(_list[index * 2], _list[index * 2 + 1]);
  }

  @override
  void operator []=(int index, Complex value) {
    _list[index * 2] = value.real;
    _list[index * 2 + 1] = value.imag;
  }
}

/// Base class for selectors used in slicing and advanced indexing.
sealed class Selector {
  const Selector();
}

/// Selects a single index along a dimension, reducing rank.
final class Index extends Selector {
  final int value;
  Index(this.value);
}

/// Represents a slice of an array dimension.
final class Slice extends Selector {
  /// The starting index of the slice.
  final int? start;

  /// The ending index of the slice (exclusive).
  final int? stop;

  /// The step size for the slice.
  final int step;

  /// Creates a slice from [start] to [stop] with [step].
  const Slice({this.start, this.stop, this.step = 1})
    : assert(step != 0, 'Step cannot be zero');

  /// Creates a slice representing all elements along a dimension.
  const Slice.all({int step = 1}) : this(start: null, stop: null, step: step);
}

/// Selects specific indices along a dimension (advanced indexing).
final class Indices extends Selector {
  final List<int> values;
  Indices(this.values);
}

/// Selects elements matching a boolean mask.
final class Mask extends Selector {
  final BooleanMask mask;
  Mask(this.mask);
}
