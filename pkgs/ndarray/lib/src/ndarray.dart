import 'package:meta/meta.dart';
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'dart:async';
import 'package:ffi/ffi.dart';
import 'dart:collection';
import 'broadcasting.dart';
import 'ndarray_bindings.dart';
import 'operations.dart' as ops;

/// Supported data types for the elements of an [NDArray].
extension type const Float64(double value) implements double {}
extension type const Float32(double value) implements double {}

extension type const Int64(int value) implements int {}
extension type const Int32(int value) implements int {}
extension type const Uint8(int value) implements int {}
extension type const Int16(int value) implements int {}

/// Supported data types for the elements of an [NDArray].
sealed class DType<T> {
  final String name;
  final int byteWidth;
  final String npyDescriptor;

  const DType._(this.name, this.byteWidth, this.npyDescriptor);

  static const float32 = Float32DType();
  static const float64 = Float64DType();
  static const int32 = Int32DType();
  static const int64 = Int64DType();
  static const uint8 = Uint8DType();
  static const int16 = Int16DType();
  static const complex64 = Complex64DType();
  static const complex128 = Complex128DType();
  static const boolean = BooleanDType();

  static const values = [
    float32,
    float64,
    int32,
    int64,
    uint8,
    int16,
    complex64,
    complex128,
    boolean,
  ];

  bool get isComplex => this is Complex64DType || this is Complex128DType;
  bool get isFloating => this is Float32DType || this is Float64DType;
  bool get isInteger =>
      this is Int32DType ||
      this is Int64DType ||
      this is Uint8DType ||
      this is Int16DType;
}

final class Float32DType extends DType<Float32> {
  const Float32DType() : super._('float32', 4, '<f4');
}

final class Float64DType extends DType<Float64> {
  const Float64DType() : super._('float64', 8, '<f8');
}

final class Int32DType extends DType<Int32> {
  const Int32DType() : super._('int32', 4, '<i4');
}

final class Int64DType extends DType<Int64> {
  const Int64DType() : super._('int64', 8, '<i8');
}

final class Uint8DType extends DType<Uint8> {
  const Uint8DType() : super._('uint8', 1, '|u1');
}

final class Int16DType extends DType<Int16> {
  const Int16DType() : super._('int16', 2, '<i2');
}

final class Complex64DType extends DType<Complex> {
  const Complex64DType() : super._('complex64', 8, '<c8');
}

final class Complex128DType extends DType<Complex> {
  const Complex128DType() : super._('complex128', 16, '<c16');
}

final class BooleanDType extends DType<bool> {
  const BooleanDType() : super._('boolean', 1, '|b1');
}

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
/// final a = `NDArray<double>`.ones([2, 3], DType.float64);
/// print(a.data); // [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
///
/// // Explicitly free memory when done
/// a.dispose();
/// ```
final class NDArray<T> implements ffi.Finalizable {
  /// Pointer to the raw C memory allocated for this array.
  final ffi.Pointer<ffi.Void> _pointer;

  /// A Dart list view of the raw C memory.
  ///
  /// **Restrictions:**
  /// - This list has a fixed length and cannot be resized.
  /// - This list becomes invalid as soon as the underlying C memory is freed (via `dispose()` or garbage collection). Accessing it afterwards leads to undefined behavior or crashes.
  @internal
  final List<T> _data;

  /// A Dart list view of the raw C memory.
  @internal
  List<T> get data {
    if (isDisposed) {
      throw StateError('Cannot access a disposed NDArray.');
    }
    return _data;
  }

  /// The dimensions of the n-dimensional array.
  final List<int> shape;

  /// The number of elements to skip in memory to move to the next position along each dimension.
  @internal
  final List<int> strides;

  /// The logical start of the array in the [data] list.
  final int offsetElements;

  /// The data type of the elements in the array.
  final DType<T> dtype;

  /// Returns true if the array is C-contiguous in memory.
  final bool isContiguous;

  /// The parent array if this is a view, to prevent it from being garbage collected.
  final NDArray? _parent;

  /// The total number of elements in the n-dimensional array.
  int get size => shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);

  /// The number of dimensions of the n-dimensional array.
  int get rank => shape.length;

  static final _finalizer = ffi.NativeFinalizer(malloc.nativeFree);

  static const _scopeKey = #ndarray.NDArrayScope;

  /// Executes [callback] within an automatic resource management scope.
  ///
  /// Any [NDArray] created during the execution of [callback] (including
  /// intermediate results from mathematical operations) will be automatically
  /// disposed of when the callback returns (or throws).
  ///
  /// If you want an array to survive beyond the scope (e.g., if it's the result
  /// of a computation), call [detachFromScope] on it before returning.
  ///
  /// **Example:**
  /// ```dart
  /// final result = NDArray.scope(() {
  ///   final a = NDArray.zeros([100], DType.float64);
  ///   final b = NDArray.ones([100], DType.float64);
  ///   final c = add(a, b);
  ///   return c.detachFromScope(); // 'a' and 'b' are freed, 'c' survives.
  /// });
  /// ```
  static R scope<R>(R Function() callback) {
    final parentScope = Zone.current[_scopeKey] as _NDArrayScope?;
    final scope = _NDArrayScope(parentScope);
    return runZoned(() {
      R result;
      try {
        result = callback();
      } catch (e) {
        scope.dispose();
        rethrow;
      }

      if (result is Future) {
        return result.whenComplete(scope.dispose) as R;
      }
      scope.dispose();
      return result;
    }, zoneValues: {_scopeKey: scope});
  }

  /// Executes [callback] within an unmanaged context, preventing any created
  /// [NDArray]s from being registered in or disposed of by any active outer scopes.
  ///
  /// **Example:**
  /// ```dart
  /// NDArray.scope(() {
  ///   final a = NDArray.zeros([10]); // Automatically disposed by scope
  ///   final b = NDArray.unmanaged(() {
  ///     return NDArray.ones([10]); // 100% unmanaged, survives the scope block!
  ///   });
  /// });
  /// ```
  static R unmanaged<R>(R Function() callback) {
    return runZoned(callback, zoneValues: {_scopeKey: null});
  }

  static bool _checkContiguous(List<int> shape, List<int> strides) {
    final cStrides = computeCStrides(shape);
    if (strides.length != cStrides.length) return false;
    for (var i = 0; i < strides.length; i++) {
      if (strides[i] != cStrides[i]) return false;
    }
    return true;
  }

  /// Private constructor for internal use and factories.
  NDArray._(
    this._pointer,
    this._data,
    this._parent, {
    required List<int> shape,
    required List<int> strides,
    required this.dtype,
    this.offsetElements = 0,
  }) : shape = List<int>.unmodifiable(shape),
       strides = List<int>.unmodifiable(strides),
       isContiguous = _checkContiguous(shape, strides) {
    assert(
      identical(T, dynamic) ||
          identical(T, Object) ||
          identical(T, num) ||
          identical(T, Float64) ||
          identical(T, Float32) ||
          identical(T, Int64) ||
          identical(T, Int32) ||
          identical(T, Uint8) ||
          identical(T, Int16) ||
          identical(T, double) ||
          identical(T, int) ||
          identical(T, Complex) ||
          identical(T, bool),
      'NDArray cannot be created with type parameter $T. '
      'Only the following allowed types are supported: Float64, Float32, Int64, Int32, Uint8, Int16, double, int, Complex, bool.',
    );
    if (_parent == null) {
      _finalizer.attach(this, _pointer, detach: this);
      final scope = Zone.current[_scopeKey] as _NDArrayScope?;
      scope?._track(this);
    }
  }

  /// Recursively locates the root memory-allocating parent array.
  @internal
  NDArray get _rootParent {
    var current = this as NDArray;
    while (current._parent != null) {
      current = current._parent!;
    }
    return current;
  }

  /// Removes this array (or its allocating parent) from its automatic disposal scope.
  ///
  /// Use this when you want an array or view created inside an [NDArray.scope] to
  /// survive after the scope finishes (e.g. when returning it as a result).
  ///
  /// Returns this array to allow for method chaining.
  NDArray<T> detachFromScope() {
    final root = _rootParent;
    final scope = Zone.current[_scopeKey] as _NDArrayScope?;
    scope?._untrack(root);
    return this;
  }

  /// Detaches this array (or its allocating parent) from the current automatic disposal scope and
  /// promotes/reattaches it to the parent (outer) scope (if any).
  ///
  /// Use this when returning an array or view from a helper function that uses an internal
  /// [NDArray.scope] to clean up its own transients, but you want the returned array
  /// to remain managed by the caller's outer scope.
  ///
  /// Returns this array to allow for method chaining.
  NDArray<T> detachToParentScope() {
    final root = _rootParent;
    final scope = Zone.current[_scopeKey] as _NDArrayScope?;
    if (scope != null) {
      scope._untrack(root);
      if (scope._parentScope != null) {
        scope._parentScope._track(root);
      }
    }
    return this;
  }

  bool _isDisposed = false;

  /// Returns true if this array or parent array's memory has been explicitly freed.
  bool get isDisposed => _isDisposed || (_parent != null && _parent.isDisposed);

  /// Returns true if this is a zero-copy view sharing memory with another array.
  bool get isView => _parent != null;

  /// Factory to create a new multi-dimensional array with backing unmanaged C heap memory.
  ///
  /// This allocates raw, stable memory directly on the unmanaged C heap using `malloc` or `calloc`.
  /// The resulting array is backed by standard Dart TypedLists mapping directly to the unmanaged pages.
  ///
  /// **Preconditions:**
  /// - All dimensions in [shape] must be strictly non-negative ($\ge 0$).
  ///
  /// **Throws:**
  /// - [ArgumentError] if any dimension in [shape] is negative.
  /// - [UnimplementedError] if the provided [dtype] is unsupported.
  ///
  /// **Performance considerations:**
  /// - Algorithmic time complexity is $O(N)$ and space complexity is $O(N)$ where $N$ is the total
  ///   number of elements (product of all dimensions in [shape]).
  /// - Offloads allocation directly to raw OS heap memory management (virtual memory page mappings),
  ///   bypassing Dart isolate VM GC pressure.
  ///
  /// **Example:**
  /// ```dart
  /// final a = `NDArray<double>`.create([2, 2], DType.float64, zeroInit: true);
  /// print(a.toList()); // [0.0, 0.0, 0.0, 0.0]
  /// ```
  ///
  /// Refer to the [NumPy Array Creation Guidelines](https://numpy.org/doc/stable/reference/routines.array-creation.html)
  /// and [Dart FFI Memory Management](https://dart.dev/guides/libraries/c-interop) for additional details.  /// Factory to create a new multi-dimensional array with backing unmanaged C heap memory.
  ///
  /// This allocates raw, stable memory directly on the unmanaged C heap using `malloc` or `calloc`.
  ///
  /// **Gotchas:**
  /// - backing heap pages are not managed by isolate garbage collection. Call `dispose()` explicitly to prevent leaks.
  ///
  /// Refer to the [NumPy Array Creation Guidelines](https://numpy.org/doc/stable/reference/routines.array-creation.html)
  /// and [Dart FFI Memory Management](https://dart.dev/guides/libraries/c-interop) for additional details.
  factory NDArray.create(
    List<int> shape,
    DType<T> dtype, {
    bool zeroInit = false,
    @internal List<int>? strides,
  }) {
    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    final finalStrides = strides ?? computeCStrides(shape);

    final allocator = zeroInit ? calloc : malloc;
    ffi.Pointer<ffi.Void> pointer;
    List<T> data;

    switch (dtype) {
      case Float64DType():
        final p = allocator<ffi.Double>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case Float32DType():
        final p = allocator<ffi.Float>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case Int32DType():
        final p = allocator<ffi.Int32>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case Int64DType():
        final p = allocator<ffi.Int64>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case Uint8DType():
        final p = allocator<ffi.Uint8>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case Int16DType():
        final p = allocator<ffi.Int16>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case Complex128DType():
        final p = allocator<ffi.Double>(totalSize * 2);
        pointer = p.cast();
        final doubleList = p.asTypedList(totalSize * 2);
        data = ComplexList(doubleList) as List<T>;
      case Complex64DType():
        final p = allocator<ffi.Float>(totalSize * 2);
        pointer = p.cast();
        final floatList = p.asTypedList(totalSize * 2);
        data = ComplexList(floatList) as List<T>;
      case BooleanDType():
        final p = allocator<ffi.Uint8>(totalSize);
        pointer = p.cast();
        final uint8List = p.asTypedList(totalSize);
        data = BoolList(uint8List) as List<T>;
    }

    return NDArray._(
      pointer,
      data,
      null,
      shape: shape,
      strides: finalStrides,
      dtype: dtype,
    );
  }

  /// Factory to create a C-contiguous array from a Dart list (copies data).
  ///
  /// The [list] is flattened and copied into the newly allocated array memory.
  /// The total size of the [shape] must match the number of elements in [list].
  ///
  /// **Throws:**
  /// - [ArgumentError] if the total size of [shape] does not match the length of [list].
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
  /// ```
  factory NDArray.fromList(List list, List<int> shape, DType<T> dtype) {
    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    if (totalSize != list.length) {
      throw ArgumentError(
        'Total size of shape $shape ($totalSize) must match list length (${list.length})',
      );
    }
    final arr = NDArray<T>.create(shape, dtype);
    final List eagerList = switch (dtype) {
      Float64DType() => Float64List.fromList(list.cast<double>()),
      Float32DType() => Float32List.fromList(list.cast<double>()),
      Int64DType() => Int64List.fromList(
        list.map((e) => (e as num).toInt()).toList(),
      ),
      Int32DType() => Int32List.fromList(
        list.map((e) => (e as num).toInt()).toList(),
      ),
      Uint8DType() => Uint8List.fromList(
        list.map((e) => (e as num).toInt()).toList(),
      ),
      Int16DType() => Int16List.fromList(
        list.map((e) => (e as num).toInt()).toList(),
      ),
      BooleanDType() => List<bool>.from(list),
      Complex128DType() || Complex64DType() => List<Complex>.from(list),
    };
    arr.data.setRange(0, eagerList.length, eagerList as dynamic);
    return arr;
  }

  /// Factory to create a new C-contiguous array filled with zeros.
  ///
  /// Backed directly by unmanaged C heap memory pages allocated via `calloc`.
  ///
  /// **Preconditions:**
  /// - All dimensions in [shape] must be strictly non-negative ($\ge 0$).
  ///
  /// **Throws:**
  /// - [ArgumentError] if any dimension in [shape] is negative.
  /// - [UnimplementedError] if the provided [dtype] is unsupported.
  ///
  /// **Performance considerations:**
  /// - Algorithmic time complexity is $O(N)$ and space complexity is $O(N)$ where $N$ is the total
  ///   number of elements (product of all dimensions in [shape]).
  /// - Bypasses isolate VM GC pressure entirely by allocating memory via unmanaged `calloc` pages.
  ///
  /// **Example:**
  /// ```dart
  /// final a = `NDArray<double>`.zeros([2, 2], DType.float64);
  /// print(a.toList()); // [0.0, 0.0, 0.0, 0.0]
  /// ```
  ///
  /// Refer to the [NumPy zeros reference](https://numpy.org/doc/stable/reference/generated/numpy.zeros.html)
  /// and [Dart FFI calloc allocator](https://pub.dev/documentation/ffi/latest/ffi/calloc-constant.html) for additional details.
  factory NDArray.zeros(List<int> shape, DType<T> dtype) {
    return NDArray<T>.create(shape, dtype, zeroInit: true);
  }

  /// Factory to create an array filled with ones.
  ///
  /// **Example:**
  /// ```dart
  /// final a = `NDArray<double>`.ones([2, 2], DType.float64);
  /// print(a.data); // [1.0, 1.0, 1.0, 1.0]
  /// ```
  factory NDArray.ones(List<int> shape, DType<T> dtype) {
    final arr = NDArray<T>.create(shape, dtype);
    if (dtype == DType.complex128 || dtype == DType.complex64) {
      arr.fill(Complex(1.0, 0.0));
    } else if (dtype == DType.boolean) {
      arr.fill(true);
    } else if (dtype == DType.float32 || dtype == DType.float64) {
      arr.fill(1.0);
    } else {
      arr.fill(1);
    }
    return arr;
  }

  /// Factory to create an array with a range of values.
  ///
  /// **Example:**
  /// ```dart
  /// final a = `NDArray<double>`.arange(0.0, 5.0, step: 1.0, dtype: DType.float64);
  /// print(a.data); // [0.0, 1.0, 2.0, 3.0, 4.0]
  /// ```
  factory NDArray.arange(
    double start,
    double stop, {
    double step = 1.0,
    DType<T>? dtype,
  }) {
    final DType<T> resolvedDType = dtype ?? (DType.float64 as DType<T>);
    if (step == 0.0) {
      throw ArgumentError('Step size cannot be zero.');
    }
    if ((stop > start && step < 0.0) || (stop < start && step > 0.0)) {
      throw ArgumentError('Step size direction must match start/stop range.');
    }
    final length = ((stop - start) / step).ceil();
    final arr = NDArray<T>.create([length], resolvedDType);
    for (var i = 0; i < length; i++) {
      final val = start + i * step;
      if (resolvedDType.isComplex) {
        arr.data[i] = Complex(val, 0.0) as T;
      } else {
        arr.data[i] = val as T;
      }
    }
    return arr;
  }

  /// Factory to create an array with evenly spaced values.
  ///
  /// **Example:**
  /// ```dart
  /// final a = `NDArray<double>`.linspace(0.0, 1.0, 5, dtype: DType.float64);
  /// print(a.data); // [0.0, 0.25, 0.5, 0.75, 1.0]
  /// ```
  factory NDArray.linspace(
    double start,
    double stop,
    int num, {
    DType<T>? dtype,
  }) {
    final DType<T> resolvedDType = dtype ?? (DType.float64 as DType<T>);
    if (num <= 0) throw ArgumentError('num must be positive');
    final arr = NDArray<T>.create([num], resolvedDType);
    if (num == 1) {
      if (resolvedDType.isComplex) {
        arr.data[0] = Complex(start, 0.0) as T;
      } else {
        arr.data[0] = start as T;
      }
      return arr;
    }
    final step = (stop - start) / (num - 1);
    for (var i = 0; i < num; i++) {
      final val = start + i * step;
      if (resolvedDType.isComplex) {
        arr.data[i] = Complex(val, 0.0) as T;
      } else {
        arr.data[i] = val as T;
      }
    }
    return arr;
  }

  /// Factory to create a 2D identity matrix.
  ///
  /// **Example:**
  /// ```dart
  /// final a = `NDArray<double>`.eye(3, DType.float64);
  /// print(a.data); // [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
  /// ```
  ///
  /// **Gotchas:**
  /// - This only creates 2D square matrices.
  factory NDArray.eye(int n, DType<T> dtype) {
    final arr = NDArray<T>.zeros([n, n], dtype);
    for (var i = 0; i < n; i++) {
      if (dtype == DType.float32 || dtype == DType.float64) {
        arr.data[i * n + i] = 1.0 as T;
      } else if (dtype.isComplex) {
        arr.data[i * n + i] = Complex(1.0, 0.0) as T;
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
  /// final view = NDArray.view(parent, shape: [2], strides: [1], offsetElements: 1);
  /// ```
  ///
  /// **Restrictions:**
  /// - **Lifetime Dependency**: The view is only valid as long as the parent's memory is not freed. If you call `parent.dispose()`, this view becomes invalid.
  /// - **Shared Mutations**: Modifications to the view affect the parent and vice versa.
  /// - **No Ownership**: Calling `dispose()` on a view does nothing.
  factory NDArray.view(
    NDArray parent, {
    required List<int> shape,
    required List<int> strides,
    int offsetElements = 0,
  }) {
    if (identical(T, dynamic) || identical(T, Object)) {
      return switch (parent.dtype) {
        Float64DType() || Float32DType() =>
          NDArray<double>.view(
                parent,
                shape: shape,
                strides: strides,
                offsetElements: offsetElements,
              )
              as NDArray<T>,
        Int64DType() || Int32DType() || Uint8DType() || Int16DType() =>
          NDArray<int>.view(
                parent,
                shape: shape,
                strides: strides,
                offsetElements: offsetElements,
              )
              as NDArray<T>,
        Complex128DType() || Complex64DType() =>
          NDArray<Complex>.view(
                parent,
                shape: shape,
                strides: strides,
                offsetElements: offsetElements,
              )
              as NDArray<T>,
        BooleanDType() =>
          NDArray<bool>.view(
                parent,
                shape: shape,
                strides: strides,
                offsetElements: offsetElements,
              )
              as NDArray<T>,
      };
    }

    final hasNegativeStrides = strides.any((s) => s < 0);

    ffi.Pointer<ffi.Void> pointer;
    List<T> data;
    final int viewOffsetElements;

    if (hasNegativeStrides) {
      pointer = parent.pointer;
      data = parent.data as List<T>;
      viewOffsetElements = offsetElements;
    } else {
      viewOffsetElements = 0;
      // Calculate the offset pointer
      switch (parent.dtype) {
        case Float64DType():
          final p = parent._pointer.cast<ffi.Double>() + offsetElements;
          pointer = p.cast();
          data = p.asTypedList(parent.data.length - offsetElements) as List<T>;
        case Float32DType():
          final p = parent._pointer.cast<ffi.Float>() + offsetElements;
          pointer = p.cast();
          data = p.asTypedList(parent.data.length - offsetElements) as List<T>;
        case Int32DType():
          final p = parent._pointer.cast<ffi.Int32>() + offsetElements;
          pointer = p.cast();
          data = p.asTypedList(parent.data.length - offsetElements) as List<T>;
        case Int64DType():
          final p = parent._pointer.cast<ffi.Int64>() + offsetElements;
          pointer = p.cast();
          data = p.asTypedList(parent.data.length - offsetElements) as List<T>;
        case Uint8DType():
          final p = parent._pointer.cast<ffi.Uint8>() + offsetElements;
          pointer = p.cast();
          data = p.asTypedList(parent.data.length - offsetElements) as List<T>;
        case Int16DType():
          final p = parent._pointer.cast<ffi.Int16>() + offsetElements;
          pointer = p.cast();
          data = p.asTypedList(parent.data.length - offsetElements) as List<T>;
        case Complex128DType():
          final p = parent._pointer.cast<ffi.Double>() + (offsetElements * 2);
          pointer = p.cast();
          final doubleList = p.asTypedList(
            parent.data.length * 2 - offsetElements * 2,
          );
          data = ComplexList(doubleList) as List<T>;
        case Complex64DType():
          final p = parent._pointer.cast<ffi.Float>() + (offsetElements * 2);
          pointer = p.cast();
          final floatList = p.asTypedList(
            parent.data.length * 2 - offsetElements * 2,
          );
          data = ComplexList(floatList) as List<T>;
        case BooleanDType():
          final p = parent._pointer.cast<ffi.Uint8>() + offsetElements;
          pointer = p.cast();
          final uint8List = p.asTypedList(parent.data.length - offsetElements);
          data = BoolList(uint8List) as List<T>;
      }
    }

    return NDArray._(
      pointer,
      data,
      parent,
      shape: shape,
      strides: strides,
      dtype: parent.dtype as DType<T>,
      offsetElements: viewOffsetElements,
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
  ffi.Pointer<ffi.Void> get pointer {
    if (isDisposed) {
      throw StateError(
        'Cannot access an array or view whose memory has been explicitly freed/disposed!',
      );
    }
    return _pointer;
  }

  /// Returns a new view of this array with a new shape.
  ///
  /// **Preconditions:**
  /// - The total size (product of dimensions) of the [newShape] must exactly match the current size.
  ///
  /// **Throws:**
  /// - [StateError] if the array has been disposed.
  /// - [ArgumentError] if the total size of [newShape] does not match the original size.
  ///
  /// **Performance considerations:**
  /// - If the array [isContiguous], this operation is extremely fast ($O(1)$), returning a zero-allocation view sharing backing memory.
  /// - If the array is a non-contiguous view, this flattens it first, performing a copy and allocating a new contiguous array ($O(N)$ complexity).
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
  /// final b = a.reshape([2, 2]);
  /// print(b.shape); // [2, 2]
  /// ```
  NDArray<T> reshape(List<int> newShape) {
    if (isDisposed) {
      throw StateError(
        'Cannot access an array or view whose memory has been explicitly freed/disposed!',
      );
    }
    final oldSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    final newSize = newShape.isEmpty ? 1 : newShape.reduce((a, b) => a * b);
    if (oldSize != newSize) {
      throw ArgumentError(
        'Total size must not change during reshape (was $oldSize, new is $newSize)',
      );
    }

    if (!isContiguous) {
      return NDArray<T>.fromList(toList(), newShape, dtype);
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

  /// Returns a copy of the array collapsed into a one-dimensional tensor list.
  ///
  /// **Performance considerations:**
  /// - For C-contiguous layouts, offloads copy directly to raw unmanaged hardware FFI
  ///   pointer `setRange` copies, achieving maximum sequential throughput.
  /// - For strided non-contiguous views, performs dynamic coordinate walk copy.
  /// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
  ///
  /// **Example:**
  /// ```dart
  /// final a = `NDArray<double>`.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
  /// final flat = a.flatten();
  /// print(flat.shape); // [4]
  /// print(flat.toList()); // [1.0, 2.0, 3.0, 4.0]
  /// ```
  NDArray<T> flatten() {
    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    final result = NDArray<T>.create([totalSize], dtype);

    if (isContiguous) {
      _copyContiguousNDArray(this, result, totalSize);
    } else {
      _copyStridedToContiguous(result);
    }
    return result;
  }

  /// Returns a deep copy of this array, respecting shape, strides, and `DType.`
  ///
  /// **Performance considerations:**
  /// - For C-contiguous layouts, offloads elements copy directly to raw unmanaged FFI
  ///   memmove/memcpy sweeps, achieving optimal performance.
  /// - For strided non-contiguous views, allocates a single contiguous NDArray of
  ///   identical shape and walks coordinates recursively to duplicate elements in-place
  ///   without allocating intermediate JIT Lists.
  ///
  /// **Throws:**
  /// - [StateError] if the array is already disposed.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
  /// final b = a.copy();
  /// b.data[0] = 99;
  /// print(a.data[0]); // 1 (decoupled memory!)
  /// ```
  NDArray<T> copy() {
    if (isDisposed) {
      throw StateError('Cannot copy a disposed array.');
    }

    final result = NDArray<T>.create(shape, dtype);

    if (isContiguous) {
      final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
      _copyContiguousNDArray(this, result, totalSize);
    } else {
      _copyStridedToContiguous(result);
    }

    return result;
  }

  void _copyStridedToContiguous(NDArray<T> dest) {
    final cShape = malloc<ffi.Int>(shape.length);
    final cStridesSrc = malloc<ffi.Int>(strides.length);
    for (var i = 0; i < shape.length; i++) {
      cShape[i] = shape[i];
      cStridesSrc[i] = strides[i];
    }
    try {
      switch (dtype) {
        case Float64DType():
          s_flatten_double(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case Float32DType():
          s_flatten_float(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case Int64DType():
          s_flatten_int64(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case Int32DType():
          s_flatten_int32(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case Uint8DType():
          s_flatten_boolean(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case Int16DType():
          throw UnimplementedError('Type $dtype not supported yet');
        case Complex128DType():
          s_flatten_complex128(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case Complex64DType():
          s_flatten_complex64(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case BooleanDType():
          s_flatten_boolean(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesSrc);
    }
  }

  /// Returns a flattened one-dimensional view or copy of this array.
  ///
  /// **Preconditions:**
  /// - The array must not be disposed.
  ///
  /// **Throws:**
  /// - [StateError] if the array has been disposed.
  ///
  /// **View vs. Copy Behavior:**
  /// - **Returns a VIEW** when the array is **C-contiguous** (`isContiguous` is `true`).
  ///   Shares the exact same backing memory and raw pointer (`_pointer`). Mutations made to the
  ///   returned raveled array will directly affect the original array (and vice versa).
  /// - **Returns a COPY** when the array is **non-contiguous / strided** (e.g. sliced views or
  ///   transposed matrices). Allocates a brand-new contiguous C heap array and duplicates elements.
  ///   Mutations made to the returned raveled array are completely decoupled and will **not** affect
  ///   the original array.
  ///
  /// **Performance considerations:**
  /// - If the array [isContiguous], this returns a zero-allocation, zero-copy 1D view sharing backing memory ($O(1)$ complexity).
  /// - Otherwise, falls back to returning a deep flattened copy ($O(N)$ complexity).
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
  /// final r = a.ravel();
  /// print(r.shape); // [4]
  /// ```
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

  /// Fills the array with [value] in-place.
  ///
  /// **Performance considerations:**
  /// - For contiguous same-type arrays, utilizes blazing fast native C register filling kernels
  ///   (`v_fill_double`, `v_fill_float`, `v_fill_int64`, `v_fill_int32`), bypassing Dart VM loops entirely.
  /// - For strided or non-contiguous views, falls back to sequential element walk mutations.
  /// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
  ///
  /// **Example:**
  /// ```dart
  /// final a = `NDArray<double>`.create([100], DType.float64);
  /// a.fill(42.0);
  /// ```
  void fill(dynamic value) {
    if (isDisposed) {
      throw StateError('Cannot fill an array whose memory has been freed.');
    }
    final size = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);

    if (isContiguous) {
      switch (dtype) {
        case Float64DType():
          if (value is num) {
            v_fill_double(_pointer.cast(), value.toDouble(), size);
            return;
          }
        case Float32DType():
          if (value is num) {
            v_fill_float(_pointer.cast(), value.toDouble(), size);
            return;
          }
        case Int64DType():
          if (value is int) {
            v_fill_int64(_pointer.cast(), value, size);
            return;
          }
        case Int32DType():
          if (value is int) {
            v_fill_int32(_pointer.cast(), value, size);
            return;
          }
        default:
          break;
      }
    }

    // Fallback JIT loop for complex, boolean, or non-contiguous views
    final targetValue = value as T;

    void fillWalk(int dim, int currentOffset) {
      if (dim == shape.length) {
        data[currentOffset] = targetValue;
        return;
      }
      for (var i = 0; i < shape[dim]; i++) {
        fillWalk(dim + 1, currentOffset + i * strides[dim]);
      }
    }

    fillWalk(0, offsetElements);
  }

  /// Transposes the dimensions of this array.
  ///
  /// By default, reverses the order of dimensions. If [axes] is provided, permutes the
  /// dimensions according to the specified permutation list.
  ///
  /// **Preconditions:**
  /// - If provided, the length of [axes] must exactly match the array rank.
  /// - Every axis value must be a valid dimension index (within `[-rank, rank - 1]`).
  /// - [axes] must contain unique, non-duplicate indices.
  ///
  /// **Throws:**
  /// - [StateError] if the array has been disposed.
  /// - [ArgumentError] if [axes] length does not match the rank of the array.
  /// - [RangeError] if any axis index is out of bounds.
  /// - [ArgumentError] if [axes] contains duplicate indices.
  ///
  /// **Performance considerations:**
  /// - This is a zero-allocation, copy-free view manipulation ($O(1)$ complexity). Strides are
  ///   re-arranged internally without copying any underlying elements.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], DType.float64);
  /// final b = a.transpose(); // b has shape [2, 3] view
  /// ```
  NDArray<T> transpose([List<int>? axes]) {
    if (isDisposed) {
      throw StateError(
        'Cannot access an array or view whose memory has been explicitly freed/disposed!',
      );
    }
    List<int> permutedAxes;
    if (axes == null) {
      permutedAxes = List.generate(shape.length, (i) => shape.length - 1 - i);
    } else {
      if (axes.length != shape.length) {
        throw ArgumentError('Axes must match the rank of the array');
      }
      final seen = <int>{};
      final normAxes = <int>[];
      for (var i = 0; i < axes.length; i++) {
        var axis = axes[i];
        if (axis < -shape.length || axis >= shape.length) {
          throw RangeError.range(axis, -shape.length, shape.length - 1, 'axis');
        }
        final normAxis = axis < 0 ? shape.length + axis : axis;
        if (seen.contains(normAxis)) {
          throw ArgumentError('Axes must be a permutation without duplicates');
        }
        seen.add(normAxis);
        normAxes.add(normAxis);
      }
      permutedAxes = normAxes;
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
  ///
  /// **Preconditions:**
  /// - The array must not be disposed.
  ///
  /// **Performance considerations:**
  /// - This is a zero-allocation, copy-free view manipulation ($O(1)$ complexity).
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
  /// final t = a.transposed; // shape [2, 2]
  /// ```
  NDArray<T> get transposed => transpose();

  /// Returns the single scalar value of a 0-dimensional array.
  ///
  /// **Preconditions:**
  /// - The array must be 0-dimensional (empty [shape]).
  ///
  /// **Throws:**
  /// - [StateError] if the array has dimensions.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.fromList([42], [], DType.int32);
  /// print(a.scalar); // 42
  /// ```
  T get scalar {
    if (shape.isNotEmpty) {
      throw StateError(
        'scalar can only be called on 0-dimensional arrays (has shape $shape)',
      );
    }
    return data[0];
  }

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
    return data[offsetElements + offset];
  }

  /// Sets the single scalar element at the specified multi-dimensional [coords] to [value].
  ///
  /// **Polymorphic Equivalence:**
  /// Equivalent to calling `this[coords] = value` via a flat list parameter.
  ///
  /// **Preconditions:**
  /// - [coords] length must match the rank of the array.
  ///
  /// **Throws:**
  /// - [ArgumentError] if `coords.length` doesn't match array rank.
  /// - [RangeError] if any coordinate is out of bounds for its dimension.
  ///
  /// **Example:**
  /// ```dart
  /// final a = NDArray.zeros([2, 2], DType.int32);
  /// a.setCell([0, 1], 42);
  /// ```
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
    data[offsetElements + offset] = value;
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
  /// Modifies elements where the provided boolean binary [mask] contains `true`
  /// drawing sequential values from another [NDArray].
  ///
  /// **Preconditions:**
  /// - [mask] must share identical dimensions ([shape]) with this array.
  void setByMask(NDArray<bool> mask, NDArray<T> values) {
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
    final valData = values.data;

    void walk(int dim, int currentOffset, int maskOffset) {
      if (dim == shape.length) {
        if (mask.data[maskOffset]) {
          if (valueIndex >= valData.length) {
            throw ArgumentError(
              'Source values array contains fewer elements than the mask targets',
            );
          }
          data[currentOffset] = valData[valueIndex++];
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

    walk(0, offsetElements, mask.offsetElements);
  }

  /// Modifies elements where the provided boolean binary [mask] contains `true`,
  /// setting them all uniformly to the single scalar [value].
  ///
  /// **Preconditions:**
  /// - [mask] must share identical dimensions ([shape]) with this array.
  void setByMaskScalar(NDArray<bool> mask, T value) {
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

    void walk(int dim, int currentOffset, int maskOffset) {
      if (dim == shape.length) {
        if (mask.data[maskOffset]) {
          data[currentOffset] = value;
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

    walk(0, offsetElements, mask.offsetElements);
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

      overwriteSlice(0, offsetElements + targetIdx * strides[axis]);
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

      writeSlice(0, offsetElements + targetIdx * strides[axis]);
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
    if (isDisposed) {
      throw StateError(
        'Cannot access an array or view whose memory has been explicitly freed/disposed!',
      );
    }
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
    } else if (spec is NDArray && spec.dtype == DType.boolean) {
      final boolMask = spec as NDArray<bool>;
      var shapesMatch = boolMask.shape.length == shape.length;
      if (shapesMatch) {
        for (var i = 0; i < shape.length; i++) {
          if (boolMask.shape[i] != shape[i]) {
            shapesMatch = false;
            break;
          }
        }
      }
      if (shapesMatch) {
        return applyMask(boolMask);
      } else {
        throw ArgumentError(
          'Boolean mask shape must exactly match array shape',
        );
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
        // Handle legacy or error cases or convert to true bool if user accidentally used int array masks
        throw ArgumentError(
          'Masking requires an NDArray of DType.boolean, not integers.',
        );
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
    if (isDisposed) {
      throw StateError(
        'Cannot access an array or view whose memory has been explicitly freed/disposed!',
      );
    }
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
    } else if (spec is NDArray && spec.dtype == DType.boolean) {
      final boolMask = spec as NDArray<bool>;
      var shapesMatch = boolMask.shape.length == shape.length;
      if (shapesMatch) {
        for (var i = 0; i < shape.length; i++) {
          if (boolMask.shape[i] != shape[i]) {
            shapesMatch = false;
            break;
          }
        }
      }
      if (shapesMatch) {
        if (value is NDArray<T>) {
          setByMask(boolMask, value);
        } else if (value is T) {
          setByMaskScalar(boolMask, value);
        } else {
          throw ArgumentError(
            'Value type (${value.runtimeType}) must be either NDArray<$T> or a scalar $T for mask assignment',
          );
        }
      } else {
        throw ArgumentError(
          'Boolean mask shape must exactly match array shape',
        );
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
        throw ArgumentError(
          'Masking requires an NDArray of DType.boolean, not integers.',
        );
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
    List<bool> result,
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
      result[offsetResult] = predicate(a[offsetA], b[offsetB]);
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

  /// Internal package helper for broadcasting comparison walks.
  void dispatchCompare(
    List<bool> rData,
    NDArray a,
    NDArray b,
    List<int> shape,
    List<int> sA,
    List<int> sB,
    List<int> sR,
    bool Function(dynamic, dynamic) predicate,
  ) {
    _dispatchCompare(rData, a, b, shape, sA, sB, sR, predicate);
  }

  void _dispatchCompare(
    List<bool> rData,
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
    } else if (value is bool) {
      return NDArray.fromList(
        <bool>[value],
        List.filled(targetShape.length, 1),
        DType.boolean,
      );
    } else {
      throw ArgumentError('Unsupported scalar type: ${value.runtimeType}');
    }
  }

  /// Element-wise addition with full broadcasting support.
  NDArray operator +(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return add(otherArr);
  }

  /// Element-wise subtraction with full broadcasting support.
  NDArray operator -(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return subtract(otherArr);
  }

  /// Element-wise multiplication with full broadcasting support.
  NDArray operator *(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return multiply(otherArr);
  }

  /// Element-wise division with full broadcasting support.
  NDArray operator /(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return divide(otherArr);
  }

  /// Element-wise floor division with full broadcasting support.
  NDArray operator ~/(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.floor_divide(this, otherArr);
  }

  /// Element-wise remainder with full broadcasting support.
  NDArray operator %(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.remainder(this, otherArr);
  }

  /// Numerical negative, element-wise.
  NDArray operator -() {
    return ops.negative(this);
  }

  /// Element-wise bitwise AND with full broadcasting support.
  NDArray operator &(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.bitwise_and(this, otherArr);
  }

  /// Element-wise bitwise OR with full broadcasting support.
  NDArray operator |(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.bitwise_or(this, otherArr);
  }

  /// Element-wise bitwise XOR with full broadcasting support.
  NDArray operator ^(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.bitwise_xor(this, otherArr);
  }

  /// Element-wise bitwise NOT.
  NDArray operator ~() {
    return ops.bitwise_not(this);
  }

  /// Element-wise left shift with full broadcasting support.
  NDArray operator <<(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.left_shift(this, otherArr);
  }

  /// Element-wise right shift with full broadcasting support.
  NDArray operator >>(dynamic other) {
    final otherArr = (other is NDArray) ? other : _wrapScalar(other, shape);
    return ops.right_shift(this, otherArr);
  }

  /// Element-wise greater than comparison with full broadcasting support.
  ///
  /// **Example:**
  /// {@example /example/ufuncs_example.dart lang=dart}
  NDArray<bool> operator >(dynamic other) {
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
    final result = NDArray<bool>.create(commonShape, DType.boolean);
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
  NDArray<bool> operator <(dynamic other) {
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
    final result = NDArray<bool>.create(commonShape, DType.boolean);
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
  NDArray<bool> operator >=(dynamic other) {
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
    final result = NDArray<bool>.create(commonShape, DType.boolean);
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
  NDArray<bool> operator <=(dynamic other) {
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
    final result = NDArray<bool>.create(commonShape, DType.boolean);
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
  NDArray<bool> eq(dynamic other) {
    NDArray otherArr;
    if (other is NDArray) {
      otherArr = other;
    } else {
      otherArr = _wrapScalar(other, shape);
    }

    final broadcastResult = broadcast(this, otherArr);
    final commonShape = broadcastResult.shape;
    final result = NDArray<bool>.create(commonShape, DType.boolean);
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
          if (maskData[j]) indices.add(j);
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
        final step = selector.step;
        final int startIdx;
        if (selector.start == null) {
          startIdx = step > 0 ? 0 : shape[i] - 1;
        } else {
          final s = selector.start!;
          startIdx = s < 0 ? shape[i] + s : s;
        }

        final int stopIdx;
        if (selector.stop == null) {
          stopIdx = step > 0 ? shape[i] : -1;
        } else {
          final s = selector.stop!;
          stopIdx = s < 0 ? shape[i] + s : s;
        }

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
      shape: newShape,
      strides: newStrides,
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
  NDArray<T> applyMask(NDArray<bool> mask) {
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
      final step = selector.step;
      final int startIdx;
      if (selector.start == null) {
        startIdx = step > 0 ? 0 : src.shape[srcDim] - 1;
      } else {
        final s = selector.start!;
        startIdx = s < 0 ? src.shape[srcDim] + s : s;
      }

      final int stopIdx;
      if (selector.stop == null) {
        stopIdx = step > 0 ? src.shape[srcDim] : -1;
      } else {
        final s = selector.stop!;
        stopIdx = s < 0 ? src.shape[srcDim] + s : s;
      }

      final realStart = startIdx.clamp(0, src.shape[srcDim] - 1);

      var destIdx = 0;
      for (
        var idx = realStart;
        selector.step > 0 ? idx < stopIdx : idx > stopIdx;
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
    if (isDisposed) {
      throw StateError(
        'Cannot access an array or view whose memory has been explicitly freed/disposed!',
      );
    }
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

    return NDArray.view(this, shape: newShape, strides: newStrides);
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

    return NDArray.view(this, shape: newShape, strides: newStrides);
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

    return NDArray.view(this, shape: newShape, strides: newStrides);
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

      if (s < -rank || s >= rank) {
        throw RangeError.range(s, -rank, rank - 1, 'source');
      }
      if (d < -rank || d >= rank) {
        throw RangeError.range(d, -rank, rank - 1, 'destination');
      }

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

    return NDArray.view(this, shape: newShape, strides: newStrides);
  }

  /// Manually free the allocated C memory.
  ///
  /// This method detaches the finalizer to prevent double-freeing.
  /// Calling this on a view does nothing, as the memory is owned by the parent.
  void dispose() {
    if (_parent != null) return; // Views don't own memory
    if (_isDisposed) return; // Guard against double-free!
    _isDisposed = true;
    _finalizer.detach(this);
    malloc.free(_pointer);
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    if (other is! NDArray) return false;
    if (dtype != other.dtype) return false;
    if (!_listEquals(shape, other.shape)) return false;

    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);

    // 1. High-speed direct C memcmp block byte check for C-contiguous same-layout arrays
    if (isContiguous && other.isContiguous) {
      final byteSize = totalSize * dtype.byteWidth;
      return custom_memcmp(pointer, other.pointer, byteSize) == 0;
    }

    // 2. Zero-allocation recursive in-place coordinate walking comparison
    return _equalsRecursive(this, other, List<int>.filled(shape.length, 0), 0);
  }

  static bool _equalsRecursive(
    NDArray a,
    NDArray b,
    List<int> indices,
    int dim,
  ) {
    if (dim == a.shape.length) {
      return a.getCell(indices) == b.getCell(indices);
    }
    for (var i = 0; i < a.shape[dim]; i++) {
      indices[dim] = i;
      if (!_equalsRecursive(a, b, indices, dim + 1)) return false;
    }
    return true;
  }

  @override
  int get hashCode {
    var baseHash = Object.hash(dtype, Object.hashAll(shape));

    final int elementsHash;
    final cShape = malloc<ffi.Int>(shape.length);
    final cStrides = malloc<ffi.Int>(strides.length);
    for (var i = 0; i < shape.length; i++) {
      cShape[i] = shape[i];
      cStrides[i] = strides[i];
    }
    try {
      if (dtype == DType.float64) {
        elementsHash = s_hash_double(
          pointer.cast(),
          cStrides,
          cShape,
          shape.length,
          isContiguous ? 1 : 0,
        );
      } else if (dtype == DType.float32) {
        elementsHash = s_hash_float(
          pointer.cast(),
          cStrides,
          cShape,
          shape.length,
          isContiguous ? 1 : 0,
        );
      } else if (dtype == DType.int64) {
        elementsHash = s_hash_int64(
          pointer.cast(),
          cStrides,
          cShape,
          shape.length,
          isContiguous ? 1 : 0,
        );
      } else if (dtype == DType.int32) {
        elementsHash = s_hash_int32(
          pointer.cast(),
          cStrides,
          cShape,
          shape.length,
          isContiguous ? 1 : 0,
        );
      } else if (dtype == DType.complex128) {
        elementsHash = s_hash_complex128(
          pointer.cast(),
          cStrides,
          cShape,
          shape.length,
          isContiguous ? 1 : 0,
        );
      } else if (dtype == DType.complex64) {
        elementsHash = s_hash_complex64(
          pointer.cast(),
          cStrides,
          cShape,
          shape.length,
          isContiguous ? 1 : 0,
        );
      } else if (dtype == DType.boolean) {
        elementsHash = s_hash_boolean(
          pointer.cast(),
          cStrides,
          cShape,
          shape.length,
          isContiguous ? 1 : 0,
        );
      } else {
        throw UnimplementedError('Type $dtype not supported yet');
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStrides);
    }

    return Object.hash(baseHash, elementsHash);
  }
}

bool _listEquals<E>(List<E> a, List<E> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

/// A wrapper class for boolean masks used in advanced indexing.
final class BooleanMask {
  /// The underlying boolean array.
  final NDArray<bool> mask;

  /// Creates a new boolean mask. Precondition: mask dtype must be `DType.boolean.`
  BooleanMask(this.mask) {
    if (mask.dtype != DType.boolean) {
      throw ArgumentError('Boolean mask must have DType.boolean');
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

  Complex operator -() => Complex(-real, -imag);

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
final class ComplexList extends ListBase<Complex> {
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

  /// Returns the real part of the complex number at [index] without allocating a [Complex] object.
  double getReal(int index) => _list[index * 2];

  /// Returns the imaginary part of the complex number at [index] without allocating a [Complex] object.
  double getImag(int index) => _list[index * 2 + 1];

  /// Sets the real and imaginary parts of the complex number at [index] without allocating a [Complex] object.
  void setRealImag(int index, double real, double imag) {
    _list[index * 2] = real;
    _list[index * 2 + 1] = imag;
  }
}

/// A list view of boolean values backed by a flat list of uint8 bytes on the FFI heap.
final class BoolList extends ListBase<bool> {
  final Uint8List _list;
  BoolList(this._list);

  /// Returns the backing list of raw bytes.
  Uint8List get backingList => _list;

  @override
  int get length => _list.length;

  @override
  set length(int newLength) {
    throw UnsupportedError('Cannot resize BoolList');
  }

  @override
  bool operator [](int index) {
    return _list[index] != 0;
  }

  @override
  void operator []=(int index, bool value) {
    _list[index] = value ? 1 : 0;
  }
}

/// Base class for selectors used in slicing and advanced indexing on [NDArray].
///
/// Subclasses represent different indexing modes:
/// - [Index] to extract a single scalar index along a dimension and reduce rank.
/// - [Slice] to extract a continuous range of values along a dimension, keeping rank.
/// - [Indices] to extract specific coordinates along a dimension (advanced indexing).
/// - [Mask] to filter elements based on a boolean mask array.
///
/// Refer to the [Advanced Slicing & Indexing Guide](https://numpy.org/doc/stable/user/basics.indexing.html)
/// for standard concepts of array slicing.
///
/// {@example /example/indexing_example.dart lang=dart}
sealed class Selector {
  const Selector();
}

/// Selects a single index along a dimension of an [NDArray], reducing the rank of the resulting array by 1.
///
/// **Preconditions:**
/// - The [value] index must be within `[-dimSize, dimSize - 1]` where `dimSize` is the size of the targeted dimension.
///
/// **Throws:**
/// - [RangeError] during slicing if [value] is out of bounds.
///
/// **Example:**
/// ```dart
/// // Select the element at index 1 along the first dimension
/// final rowView = arr.slice([Index(1)]);
/// ```
final class Index extends Selector {
  /// The coordinate index to select. Can be negative to index from the end.
  final int value;

  /// Creates a single index selector with the specified [value].
  Index(this.value);
}

/// Represents a continuous or strided slice of an [NDArray] dimension.
///
/// Similar to Python's `start:stop:step` slice notation. Keeps the rank of the dimension intact.
///
/// **Preconditions:**
/// - [step] must be strictly non-zero.
/// - [start] and [stop], if provided, represent inclusive start and exclusive stop bounds.
///
/// **Throws:**
/// - [AssertionError] if [step] is zero.
///
/// **Example:**
/// ```dart
/// // Select elements from index 1 to 5 with step size of 2
/// final sliceView = arr.slice([Slice(start: 1, stop: 5, step: 2)]);
/// ```
final class Slice extends Selector {
  /// The starting index of the slice (inclusive).
  /// If null, defaults to the beginning of the dimension.
  final int? start;

  /// The ending index of the slice (exclusive).
  /// If null, defaults to the end of the dimension.
  final int? stop;

  /// The step size for the slice. Defaults to 1.
  final int step;

  /// Creates a slice from [start] to [stop] with [step].
  ///
  /// Precondition: [step] must be non-zero.
  const Slice({this.start, this.stop, this.step = 1})
    : assert(step != 0, 'Step cannot be zero');

  /// Creates a slice representing all elements along a dimension.
  const Slice.all({int step = 1}) : this(start: null, stop: null, step: step);
}

/// Selects specific coordinate indices along an [NDArray] dimension (advanced indexing).
///
/// Useful for extracting irregular intervals or custom lists of indices.
///
/// **Preconditions:**
/// - Every index in [values] must be within `[-dimSize, dimSize - 1]` where `dimSize` is the size of the dimension.
///
/// **Example:**
/// ```dart
/// // Extract rows at index 0 and 2 from a 2D matrix
/// final subMatrix = arr.slice([Indices([0, 2])]);
/// ```
final class Indices extends Selector {
  /// The list of specific indices to select.
  final List<int> values;

  /// Creates an indices selector with the specified coordinate [values].
  Indices(this.values);
}

/// Selects elements of an [NDArray] matching a boolean mask array.
///
/// Triggers boolean indexing/masking.
///
/// **Preconditions:**
/// - The [mask] must share identical shape and dimensions with the targeted dimension array.
///
/// **Example:**
/// ```dart
/// // Filter elements matching a boolean condition
/// final maskCondition = arr > 0.5;
/// final positiveValues = arr.slice([Mask(BooleanMask(maskCondition))]);
/// ```
final class Mask extends Selector {
  /// The boolean mask wrapper.
  final BooleanMask mask;

  /// Creates a mask selector wrapping the specified boolean [mask].
  Mask(this.mask);
}

void _copyContiguousNDArray(NDArray src, NDArray dest, int size) {
  final dtype = src.dtype;
  if (dtype == DType.float64) {
    final srcList = src._pointer.cast<ffi.Double>().asTypedList(size);
    final destList = dest._pointer.cast<ffi.Double>().asTypedList(size);
    destList.setRange(0, size, srcList);
  } else if (dtype == DType.float32) {
    final srcList = src._pointer.cast<ffi.Float>().asTypedList(size);
    final destList = dest._pointer.cast<ffi.Float>().asTypedList(size);
    destList.setRange(0, size, srcList);
  } else if (dtype == DType.int32) {
    final srcList = src._pointer.cast<ffi.Int32>().asTypedList(size);
    final destList = dest._pointer.cast<ffi.Int32>().asTypedList(size);
    destList.setRange(0, size, srcList);
  } else if (dtype == DType.int64) {
    final srcList = src._pointer.cast<ffi.Int64>().asTypedList(size);
    final destList = dest._pointer.cast<ffi.Int64>().asTypedList(size);
    destList.setRange(0, size, srcList);
  } else if (dtype == DType.complex128) {
    final srcList = src._pointer.cast<ffi.Double>().asTypedList(size * 2);
    final destList = dest._pointer.cast<ffi.Double>().asTypedList(size * 2);
    destList.setRange(0, size * 2, srcList);
  } else if (dtype == DType.complex64) {
    final srcList = src._pointer.cast<ffi.Float>().asTypedList(size * 2);
    final destList = dest._pointer.cast<ffi.Float>().asTypedList(size * 2);
    destList.setRange(0, size * 2, srcList);
  } else if (dtype == DType.boolean) {
    final srcList = src._pointer.cast<ffi.Uint8>().asTypedList(size);
    final destList = dest._pointer.cast<ffi.Uint8>().asTypedList(size);
    destList.setRange(0, size, srcList);
  } else if (dtype == DType.uint8) {
    final srcList = src._pointer.cast<ffi.Uint8>().asTypedList(size);
    final destList = dest._pointer.cast<ffi.Uint8>().asTypedList(size);
    destList.setRange(0, size, srcList);
  } else if (dtype == DType.int16) {
    final srcList = src._pointer.cast<ffi.Int16>().asTypedList(size);
    final destList = dest._pointer.cast<ffi.Int16>().asTypedList(size);
    destList.setRange(0, size, srcList);
  } else {
    throw UnimplementedError('Type $dtype not supported for fast flatten');
  }
}

/// Private class to manage a collection of [NDArray]s within a [Zone]
/// using a high-performance hybrid scaling design: a flat fast-path List
/// for collections up to 100 elements (yielding zero GC and blazingly fast sweeps),
/// promoting seamlessly to a HashSet for larger collections to guarantee O(1) scaling.
final class _NDArrayScope {
  // Parent outer scope context (if nested)
  final _NDArrayScope? _parentScope;

  // Standard fast-path: flat List of tracked arrays
  final List<NDArray> _list = [];

  // Fallback slow-path: Set for large-scale scopes (> 100 arrays)
  Set<NDArray>? _set;

  _NDArrayScope(this._parentScope);

  void _track(NDArray array) {
    if (_set != null) {
      _set!.add(array);
      return;
    }

    _list.add(array);

    // Promotion trigger: promote to HashSet once we cross 100 elements
    if (_list.length > 100) {
      _set = HashSet(equals: identical, hashCode: identityHashCode);
      _set!.addAll(_list);
      _list.clear();
    }
  }

  void _untrack(NDArray array) {
    if (_set != null) {
      _set!.remove(array);
      return;
    }

    // Swap-and-Pop optimization to avoid shifting subsequent elements in the list!
    final len = _list.length;
    for (var i = 0; i < len; i++) {
      if (identical(_list[i], array)) {
        if (i < len - 1) {
          _list[i] = _list.last;
        }
        _list.removeLast();
        break;
      }
    }
  }

  void dispose() {
    if (_set != null) {
      for (final array in _set!) {
        if (!array.isDisposed) {
          array.dispose();
        }
      }
      _set!.clear();
    } else {
      for (final array in _list) {
        if (!array.isDisposed) {
          array.dispose();
        }
      }
      _list.clear();
    }
  }
}
