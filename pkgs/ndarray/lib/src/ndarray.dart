import 'package:meta/meta.dart';
import 'dart:math' as math;
import 'dart:ffi' as ffi;
import 'dart:typed_data';
import 'dart:async';
import 'package:ffi/ffi.dart';
import 'dart:collection';

import 'ndarray_bindings.dart';
import 'scratch_arena.dart';
import 'package:openblas/openblas.dart' show openblas_set_num_threads;

/// Supported data types for the elements of an [NDArray].
extension type const Float64(double value) implements double {}
extension type const Float32(double value) implements double {}

extension type const Int64(int value) implements int {}
extension type const Int32(int value) implements int {}
extension type const Uint8(int value) implements int {}
extension type const Int16(int value) implements int {}

/// Base marker interface for all types that can be stored in an [NDArray].
abstract interface class Marker {}

/// Marker interface for numeric elements (integers, floats, complex).
abstract interface class NumericMarker implements Marker {}

/// Marker interface for floating-point elements.
abstract interface class FloatingMarker implements NumericMarker {}

/// Marker interface for integer elements.
abstract interface class IntegerMarker implements NumericMarker {}

/// Marker interface for complex number elements.
abstract interface class ComplexMarker implements NumericMarker {}

/// Represents a boolean element in an [NDArray].
abstract interface class BoolMarker implements Marker {}

/// Marker for double-precision float (64-bit).
final class Float64Marker implements FloatingMarker {}

/// Marker for single-precision float (32-bit).
final class Float32Marker implements FloatingMarker {}

/// Marker for 64-bit signed integer.
final class Int64Marker implements IntegerMarker {}

/// Marker for 32-bit signed integer.
final class Int32Marker implements IntegerMarker {}

/// Marker for 8-bit unsigned integer.
final class Uint8Marker implements IntegerMarker {}

/// Marker for 16-bit signed integer.
final class Int16Marker implements IntegerMarker {}

/// Marker for double-precision complex (128-bit).
final class Complex128Marker implements ComplexMarker {}

/// Marker for single-precision complex (64-bit).
final class Complex64Marker implements ComplexMarker {}

/// Marker for boolean.
final class BooleanMarker implements BoolMarker {}

/// Supported data types for the elements of an [NDArray].
enum DType<T, M extends Marker> {
  float32<double, Float32Marker>('float32', 4, '<f4'),
  float64<double, Float64Marker>('float64', 8, '<f8'),
  // 64 bits total: 32-bit float real + 32-bit float imaginary parts (8 bytes)
  complex64<Complex, Complex64Marker>('complex64', 8, '<c8'),
  // 128 bits total: 64-bit double real + 64-bit double imaginary parts (16 bytes)
  complex128<Complex, Complex128Marker>('complex128', 16, '<c16'),
  uint8<int, Uint8Marker>('uint8', 1, '|u1'),
  int16<int, Int16Marker>('int16', 2, '<i2'),
  int32<int, Int32Marker>('int32', 4, '<i4'),
  int64<int, Int64Marker>('int64', 8, '<i8'),
  // Uses 8 bits (1 byte) per boolean value (backed by ffi.Uint8)
  boolean<bool, BooleanMarker>('boolean', 1, '|b1');

  final String name;
  final int byteWidth;
  final String npyDescriptor;

  const DType(this.name, this.byteWidth, this.npyDescriptor);

  bool get isComplex => this == DType.complex64 || this == DType.complex128;
  bool get isFloating => this == DType.float32 || this == DType.float64;
  bool get isInteger =>
      this == DType.int32 ||
      this == DType.int64 ||
      this == DType.uint8 ||
      this == DType.int16;
}

/// An n-dimensional array with memory allocated on the C heap.
///
/// **Memory Management Guidelines:**
/// - **Explicit Disposal & Scopes**: Always call [dispose] explicitly as soon as an array is no
///   longer needed, or wrap your computations in [NDArray.scope] for automated scope-level resource
///   management. While the garbage collector will eventually free C memory to prevent hard
///   leaks, it is blind to large native allocations, and garbage collection might not be
///   triggered early enough.
/// - **Views & Shared Memory**: Views (slices, reshapes, transposes, etc.) share the exact same
///   C memory as their parent; modifying a view mutates the parent and vice versa. Calling [dispose]
///   on a parent array immediately **invalidates all views** derived from it; accessing an
///   invalidated view causes crashes or undefined behavior.
///
/// **Arithmetic Error & Overflow Handling:**
/// Mathematical operations on [NDArray] are backed by highly-optimized native C ufuncs,
/// and adhere to the following rules:
/// - **Division by Zero**:
///   - **True Division (`/`)**: Performs floating-point division under IEEE 754 standards.
///     If one or both operands are integers, they are promoted to [DType.float64] (matching NumPy).
///     Division of non-zero values by zero results in `double.infinity` or `double.negativeInfinity`.
///     Division of zero by zero results in `double.nan`. No exceptions are thrown.
///   - **Floor Division (`~/` or `floor_divide`) & Remainder (`%` or `remainder`)**:
///     - **For Floating-Point Types**: Behaves identically to true division, returning `double.nan`
///       on division by zero without throwing exceptions.
///     - **For Integer Types**: Throws an [UnsupportedError] if any element of the divisor is `0`.
///       This upfront safety check in Dart prevents native C integer division by zero, which is
///       undefined behavior in C and would crash the entire Dart VM process with a `SIGFPE` signal.
/// - **Overflow**:
///   - **Integer Overflow**: Integer operations (such as addition, subtraction, and multiplication)
///     performed on arrays of type `int32`, `int64`, `int16`, or `uint8` wrap around silently
///     using two's complement representation matching NumPy's wrapping behavior (unless operands
///     are upcasted, e.g., mixing `int32` and `int64` upcasts to `int64`).
///   - **Floating-Point Overflow**: Floating-point operations that exceed the representable bounds
///     of `float32` or `float64` overflow silently to `double.infinity` or `double.negativeInfinity`
///     per IEEE 754 rules (matching NumPy).
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
final class NDArray<T, M extends Marker> implements ffi.Finalizable {
  /// Pointer to the raw C memory allocated for this array.
  final ffi.Pointer<ffi.Void> _pointer;

  /// A Dart list view of the raw C memory.
  ///
  /// **Restrictions:**
  /// - This list has a fixed length and cannot be resized.
  /// - This list becomes invalid as soon as the underlying C memory is freed (via `dispose()` or garbage collection). Accessing it afterwards leads to undefined behavior or crashes.
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
  final DType<T, M> dtype;

  /// Returns true if the array is C-contiguous in memory.
  ///
  /// An array is C-contiguous if its elements are stored sequentially in memory
  /// such that the last dimension varies the fastest.
  ///
  /// **When does an array become non-contiguous?**
  /// - When creating views via slicing with a step greater than 1.
  /// - When transposing or permuting axes.
  /// - When reshaping a non-contiguous view.
  ///
  /// **Why is it relevant?**
  /// - **Performance**: Contiguous arrays allow for optimized, vectorized operations
  ///   (like BLAS calls) that achieve maximum memory throughput. Non-contiguous
  ///   arrays often fall back to slower element-by-element iteration.
  ///
  /// **How to make an array contiguous?**
  /// - Call [copy] to allocate a new contiguous array with the same elements.
  final bool isContiguous;

  /// The parent array if this is a view, to prevent it from being garbage collected.
  final NDArray? _parent;

  /// Whether the backing native memory is externally/user-allocated.
  final bool _isExternallyOwned;

  /// Optional user-provided native finalizer to deallocate external memory.
  final ffi.Pointer<
    ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>
  >?
  _customNativeFinalizer;

  /// The custom native finalizer instance for this array if it has one.
  final ffi.NativeFinalizer? _customFinalizerInstance;

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
    bool isExternallyOwned = false,
    ffi.Pointer<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>?
    customNativeFinalizer,
  }) : _isExternallyOwned = isExternallyOwned,
       _customNativeFinalizer = customNativeFinalizer,
       _customFinalizerInstance =
           (_parent == null &&
               isExternallyOwned &&
               customNativeFinalizer != null)
           ? ffi.NativeFinalizer(customNativeFinalizer)
           : null,
       shape = List<int>.unmodifiable(shape),
       strides = List<int>.unmodifiable(strides),
       isContiguous = _checkContiguous(shape, strides) {
    _initializeOpenBLASOnce();
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
      if (!_isExternallyOwned) {
        _finalizer.attach(this, _pointer, detach: this);
      } else if (_customFinalizerInstance != null) {
        _customFinalizerInstance.attach(this, _pointer, detach: this);
      }
      final scope = Zone.current[_scopeKey] as _NDArrayScope?;
      scope?._track(this);
    }
  }

  /// Recursively locates the root memory-allocating parent array.
  NDArray get _rootParent {
    var current = this as NDArray;
    while (current._parent != null) {
      current = current._parent;
    }
    return current;
  }

  /// Removes this array (or its allocating parent) from its automatic disposal scope.
  ///
  /// Use this when you want an array or view created inside an [NDArray.scope] to
  /// survive after the scope finishes (e.g. when returning it as a result).
  ///
  /// Returns this array to allow for method chaining.
  NDArray<T, M> detachFromScope() {
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
  NDArray<T, M> detachToParentScope() {
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
  /// **Gotchas:**
  /// - Backing heap pages are not managed by isolate garbage collection. Call `dispose()` explicitly to prevent leaks.
  ///
  /// Refer to the [NumPy Array Creation Guidelines](https://numpy.org/doc/stable/reference/routines.array-creation.html)
  /// and [Dart FFI Memory Management](https://dart.dev/guides/libraries/c-interop) for additional details.
  factory NDArray.create(
    List<int> shape,
    DType<T, M> dtype, {
    bool zeroInit = false,
    @internal List<int>? strides,
  }) {
    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    final finalStrides = strides ?? computeCStrides(shape);

    final allocator = zeroInit ? calloc : malloc;
    ffi.Pointer<ffi.Void> pointer;
    List<T> data;

    switch (dtype) {
      case DType.float64:
        final p = allocator<ffi.Double>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case DType.float32:
        final p = allocator<ffi.Float>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case DType.int32:
        final p = allocator<ffi.Int32>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case DType.int64:
        final p = allocator<ffi.Int64>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case DType.uint8:
        final p = allocator<ffi.Uint8>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case DType.int16:
        final p = allocator<ffi.Int16>(totalSize);
        pointer = p.cast();
        data = p.asTypedList(totalSize) as List<T>;
      case DType.complex128:
        final p = allocator<ffi.Double>(totalSize * 2);
        pointer = p.cast();
        final doubleList = p.asTypedList(totalSize * 2);
        data = ComplexList(doubleList) as List<T>;
      case DType.complex64:
        final p = allocator<ffi.Float>(totalSize * 2);
        pointer = p.cast();
        final floatList = p.asTypedList(totalSize * 2);
        data = ComplexList(floatList) as List<T>;
      case DType.boolean:
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
  factory NDArray.fromList(List list, List<int> shape, DType<T, M> dtype) {
    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    if (totalSize != list.length) {
      throw ArgumentError(
        'Total size of shape $shape ($totalSize) must match list length (${list.length})',
      );
    }
    final arr = NDArray<T, M>.create(shape, dtype);
    final List eagerList = switch (dtype) {
      DType.float64 => Float64List.fromList(list.cast<double>()),
      DType.float32 => Float32List.fromList(list.cast<double>()),
      DType.int64 => Int64List.fromList(
        list.map((e) => (e as num).toInt()).toList(),
      ),
      DType.int32 => Int32List.fromList(
        list.map((e) => (e as num).toInt()).toList(),
      ),
      DType.uint8 => Uint8List.fromList(
        list.map((e) => (e as num).toInt()).toList(),
      ),
      DType.int16 => Int16List.fromList(
        list.map((e) => (e as num).toInt()).toList(),
      ),
      DType.boolean => List<bool>.from(list),
      DType.complex128 || DType.complex64 => List<Complex>.from(list),
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
  factory NDArray.zeros(List<int> shape, DType<T, M> dtype) {
    return NDArray<T, M>.create(shape, dtype, zeroInit: true);
  }

  /// Factory to create an array filled with ones.
  ///
  /// **Example:**
  /// ```dart
  /// final a = `NDArray<double>`.ones([2, 2], DType.float64);
  /// print(a.data); // [1.0, 1.0, 1.0, 1.0]
  /// ```
  factory NDArray.ones(List<int> shape, DType<T, M> dtype) {
    final arr = NDArray<T, M>.create(shape, dtype);
    if (dtype == DType.complex128 || dtype == DType.complex64) {
      arr.fill(Complex(1.0, 0.0) as T);
    } else if (dtype == DType.boolean) {
      arr.fill(true as T);
    } else if (dtype == DType.float32 || dtype == DType.float64) {
      arr.fill(1.0 as T);
    } else {
      arr.fill(1 as T);
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
    DType<T, M>? dtype,
  }) {
    final DType<T, M> resolvedDType = dtype ?? (DType.float64 as DType<T, M>);
    if (step == 0.0) {
      throw ArgumentError('Step size cannot be zero.');
    }
    if ((stop > start && step < 0.0) || (stop < start && step > 0.0)) {
      throw ArgumentError('Step size direction must match start/stop range.');
    }
    final length = ((stop - start) / step).ceil();
    final arr = NDArray<T, M>.create([length], resolvedDType);
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
  factory NDArray.eye(int n, DType<T, M> dtype) {
    final arr = NDArray<T, M>.zeros([n, n], dtype);
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
    NDArray<T, M> parent, {
    required List<int> shape,
    required List<int> strides,
    int offsetElements = 0,
  }) {
    final root = parent._rootParent;
    final childLogicalPointer = _offsetPointer(
      parent.pointer,
      offsetElements,
      parent.dtype,
    );
    final cumulativeOffset =
        (childLogicalPointer.address - root._pointer.address) ~/
        parent.dtype.byteWidth;

    // Calculate min and max relative offsets
    var minRelativeOffset = 0;
    var maxRelativeOffset = 0;
    for (var d = 0; d < shape.length; d++) {
      final stride = strides[d];
      final size = shape[d];
      if (stride > 0) {
        maxRelativeOffset += (size - 1) * stride;
      } else {
        minRelativeOffset += (size - 1) * stride;
      }
    }

    final minPhysicalOffset = cumulativeOffset + minRelativeOffset;
    final maxPhysicalOffset = cumulativeOffset + maxRelativeOffset;

    final physicalPointer = _offsetPointer(
      root._pointer,
      minPhysicalOffset,
      parent.dtype,
    );
    final int viewSize = maxPhysicalOffset - minPhysicalOffset + 1;
    final List<T> data;

    switch (parent.dtype) {
      case DType.float64:
        data =
            physicalPointer.cast<ffi.Double>().asTypedList(viewSize) as List<T>;
      case DType.float32:
        data =
            physicalPointer.cast<ffi.Float>().asTypedList(viewSize) as List<T>;
      case DType.int32:
        data =
            physicalPointer.cast<ffi.Int32>().asTypedList(viewSize) as List<T>;
      case DType.int64:
        data =
            physicalPointer.cast<ffi.Int64>().asTypedList(viewSize) as List<T>;
      case DType.uint8:
        data =
            physicalPointer.cast<ffi.Uint8>().asTypedList(viewSize) as List<T>;
      case DType.int16:
        data =
            physicalPointer.cast<ffi.Int16>().asTypedList(viewSize) as List<T>;
      case DType.complex128:
        final p = _offsetPointer(
          root._pointer,
          minPhysicalOffset * 2,
          DType.float64,
        );
        final doubleList = p.cast<ffi.Double>().asTypedList(viewSize * 2);
        data = ComplexList(doubleList) as List<T>;
      case DType.complex64:
        final p = _offsetPointer(
          root._pointer,
          minPhysicalOffset * 2,
          DType.float32,
        );
        final floatList = p.cast<ffi.Float>().asTypedList(viewSize * 2);
        data = ComplexList(floatList) as List<T>;
      case DType.boolean:
        data =
            BoolList(physicalPointer.cast<ffi.Uint8>().asTypedList(viewSize))
                as List<T>;
    }

    final viewOffsetElements = -minRelativeOffset;

    return NDArray._(
      childLogicalPointer,
      data,
      parent,
      shape: shape,
      strides: strides,
      dtype: parent.dtype,
      offsetElements: viewOffsetElements,
    );
  }

  /// Factory to create a new [NDArray] view backed by a user-allocated external C memory pointer.
  ///
  /// The user must ensure that [pointer] points to a valid block of contiguous memory
  /// of at least `size * dtype.byteWidth` bytes, where `size` is the product of all dimensions in [shape].
  ///
  /// **Preconditions:**
  /// - [pointer] must not be null or point to an invalid memory location.
  /// - All dimensions in [shape] must be strictly non-negative ($\ge 0$).
  ///
  /// **Lifetime Management Options:**
  /// - **Externally Managed (Default):** If [nativeFinalizer] is omitted or `null`, the array does not
  ///   own the memory. Calling [dispose] will invalidate the array and any of its views, but will **not**
  ///   free the raw C memory pointer. The user is fully responsible for freeing the memory.
  /// - **Custom Finalization:** If [nativeFinalizer] is provided, it will be registered with a Dart
  ///   [NativeFinalizer] to automatically deallocate the backing C pointer when this array is garbage collected,
  ///   or when [dispose] is called.
  ///
  /// **Throws:**
  /// - [ArgumentError] if any dimension in [shape] is negative.
  ///
  /// **Performance Considerations:**
  /// - This is an $O(1)$ operation that performs zero copies, constructing a direct list view over
  ///   the provided raw C memory address.
  ///
  /// **Example:**
  /// {@example /example/external_memory_example.dart}
  factory NDArray.fromPointer(
    ffi.Pointer<ffi.Void> pointer,
    List<int> shape,
    DType<T, M> dtype, {
    ffi.Pointer<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>?
    nativeFinalizer,
    List<int>? strides,
  }) {
    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    final finalStrides = strides ?? computeCStrides(shape);

    List<T> data;
    switch (dtype) {
      case DType.float64:
        data = pointer.cast<ffi.Double>().asTypedList(totalSize) as List<T>;
      case DType.float32:
        data = pointer.cast<ffi.Float>().asTypedList(totalSize) as List<T>;
      case DType.int32:
        data = pointer.cast<ffi.Int32>().asTypedList(totalSize) as List<T>;
      case DType.int64:
        data = pointer.cast<ffi.Int64>().asTypedList(totalSize) as List<T>;
      case DType.uint8:
        data = pointer.cast<ffi.Uint8>().asTypedList(totalSize) as List<T>;
      case DType.int16:
        data = pointer.cast<ffi.Int16>().asTypedList(totalSize) as List<T>;
      case DType.complex128:
        data =
            ComplexList(pointer.cast<ffi.Double>().asTypedList(totalSize * 2))
                as List<T>;
      case DType.complex64:
        data =
            ComplexList(pointer.cast<ffi.Float>().asTypedList(totalSize * 2))
                as List<T>;
      case DType.boolean:
        data =
            BoolList(pointer.cast<ffi.Uint8>().asTypedList(totalSize))
                as List<T>;
    }

    return NDArray._(
      pointer,
      data,
      null,
      shape: shape,
      strides: finalStrides,
      dtype: dtype,
      isExternallyOwned: true,
      customNativeFinalizer: nativeFinalizer,
    );
  }

  /// Helper to calculate default strides for a C-contiguous array (in elements).
  @internal
  static List<int> computeCStrides(List<int> shape) {
    if (shape.isEmpty) return [];
    final strides = List<int>.filled(shape.length, 1);
    for (var i = shape.length - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
  }

  static ffi.Pointer<ffi.Void> _offsetPointer(
    ffi.Pointer<ffi.Void> ptr,
    int offsetElements,
    DType dtype,
  ) {
    switch (dtype) {
      case DType.float64:
        return (ptr.cast<ffi.Double>() + offsetElements).cast();
      case DType.float32:
        return (ptr.cast<ffi.Float>() + offsetElements).cast();
      case DType.int32:
        return (ptr.cast<ffi.Int32>() + offsetElements).cast();
      case DType.int64:
        return (ptr.cast<ffi.Int64>() + offsetElements).cast();
      case DType.uint8:
        return (ptr.cast<ffi.Uint8>() + offsetElements).cast();
      case DType.int16:
        return (ptr.cast<ffi.Int16>() + offsetElements).cast();
      case DType.complex128:
        return (ptr.cast<ffi.Double>() + (offsetElements * 2)).cast();
      case DType.complex64:
        return (ptr.cast<ffi.Float>() + (offsetElements * 2)).cast();
      case DType.boolean:
        return (ptr.cast<ffi.Uint8>() + offsetElements).cast();
    }
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
  NDArray<T, M> reshape(List<int> newShape) {
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
      final result = NDArray<T, M>.create(newShape, dtype);
      _copyStridedToContiguous(result);
      return result;
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
  NDArray<T, M> flatten() {
    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    final result = NDArray<T, M>.create([totalSize], dtype);

    if (isContiguous) {
      _copyContiguousNDArray(this, result, totalSize);
    } else {
      _copyStridedToContiguous(result);
    }
    return result;
  }

  /// Returns a deep, C-contiguous copy of this array.
  ///
  /// The copy preserves the logical order and values of the elements defined by
  /// this array's shape and strides. However, the physical memory layout of the
  /// returned array is always contiguous (and its strides are reset to standard
  /// C-contiguous strides).
  ///
  /// **Performance considerations:**
  /// - For C-contiguous layouts, offloads elements copy directly to raw unmanaged FFI
  ///   memmove/memcpy sweeps, achieving optimal performance.
  /// - For strided non-contiguous views, delegates the copy/flatten operation to optimized
  ///   native FFI/C functions (intrinsics) to write into the contiguous destination array
  ///   without allocating intermediate structures in Dart.
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
  NDArray<T, M> copy({NDArray<T, M>? out}) {
    if (isDisposed) {
      throw StateError('Cannot copy a disposed array.');
    }

    final NDArray<T, M> result;
    if (out != null) {
      if (out.isDisposed) {
        throw StateError('Cannot copy to a disposed array.');
      }
      if (!listEquals(shape, out.shape) || dtype != out.dtype) {
        throw ArgumentError(
          'Destination array must have matching shape and dtype (expected shape $shape, dtype $dtype; got shape ${out.shape}, dtype ${out.dtype}).',
        );
      }
      if (!out.isContiguous) {
        throw ArgumentError('Destination array must be contiguous.');
      }
      result = out;
    } else {
      result = NDArray<T, M>.create(shape, dtype);
    }

    if (isContiguous) {
      final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
      _copyContiguousNDArray(this, result, totalSize);
    } else {
      _copyStridedToContiguous(result);
    }

    return result;
  }

  /// Internal helper to copy contiguous array elements to another contiguous array,
  /// bypassing generic type constraints.
  @internal
  void copyToContiguous(NDArray dest) {
    if (isDisposed || dest.isDisposed) {
      throw StateError('Cannot copy to or from a disposed array.');
    }
    if (!listEquals(shape, dest.shape) || dtype != dest.dtype) {
      throw ArgumentError('Mismatched shape or dtype in copyToContiguous.');
    }
    if (!isContiguous || !dest.isContiguous) {
      throw ArgumentError(
        'Both arrays must be contiguous in copyToContiguous.',
      );
    }
    final totalSize = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);
    _copyContiguousNDArray(this, dest, totalSize);
  }

  void _copyStridedToContiguous(NDArray<T, M> dest) {
    final marker = ScratchArena.marker;
    try {
      final cShape = ScratchArena.copyInts(shape);
      final cStridesSrc = ScratchArena.copyInts(strides);
      switch (dtype) {
        case DType.float64:
          s_flatten_double(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case DType.float32:
          s_flatten_float(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case DType.int64:
          s_flatten_int64(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case DType.int32:
          s_flatten_int32(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case DType.uint8:
          s_flatten_uint8(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case DType.int16:
          s_flatten_int16(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case DType.complex128:
          s_flatten_complex128(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case DType.complex64:
          s_flatten_complex64(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
        case DType.boolean:
          s_flatten_uint8(
            pointer.cast(),
            cStridesSrc,
            dest.pointer.cast(),
            cShape,
            shape.length,
          );
      }
    } finally {
      ScratchArena.reset(marker);
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
  NDArray<T, M> ravel() {
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
  void fill(T value) {
    if (isDisposed) {
      throw StateError('Cannot fill an array whose memory has been freed.');
    }
    final size = shape.isEmpty ? 1 : shape.reduce((a, b) => a * b);

    if (isContiguous) {
      switch (dtype) {
        case DType.float64:
          if (value is num) {
            v_fill_double(_pointer.cast(), value.toDouble(), size);
            return;
          }
        case DType.float32:
          if (value is num) {
            v_fill_float(_pointer.cast(), value.toDouble(), size);
            return;
          }
        case DType.int64:
          if (value is int) {
            v_fill_int64(_pointer.cast(), value, size);
            return;
          }
        case DType.int32:
          if (value is int) {
            v_fill_int32(_pointer.cast(), value, size);
            return;
          }
        default:
          break;
      }
    }

    // Fallback JIT loop for complex, boolean, or non-contiguous views
    final targetValue = value;

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
  /// By default, reverses the order of dimensions (equivalent to calling `transposed`).
  /// If [axes] is provided, permutes the dimensions according to the specified
  /// permutation list.
  ///
  /// **Axes Interpretation:**
  /// - The length of [axes] must equal the rank of the array.
  /// - The value `axes[i]` specifies the index of the dimension in the original array
  ///   that will map to the `i`-th dimension of the transposed array.
  /// - Negative indices in [axes] are resolved relative to the end of the dimensions,
  ///   where `-1` represents the last dimension, `-2` represents the second-to-last,
  ///   and so on.
  /// - For example, on a 3-dimensional array with shape `[A, B, C]`:
  ///   - `transpose()` (or `transpose(null)`) results in a shape of `[C, B, A]`.
  ///   - `transpose([1, 0, 2])` results in a shape of `[B, A, C]`.
  ///   - `transpose([-1, -2, -3])` is equivalent to `transpose([2, 1, 0])`, which
  ///     results in a shape of `[C, B, A]`.
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
  NDArray<T, M> transpose([List<int>? axes]) {
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
  /// Equivalent to calling `transpose()` with no arguments. Reverses the order
  /// of dimensions (e.g., a 3-dimensional array of shape `[A, B, C]` becomes a
  /// view with shape `[C, B, A]`).
  ///
  /// To permute the dimensions in a custom order, use [transpose].
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
  NDArray<T, M> get transposed => transpose();

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
  void setByMask(NDArray<bool, BooleanMarker> mask, NDArray<T, M> values) {
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
  void setByMaskScalar(NDArray<bool, BooleanMarker> mask, T value) {
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
  void setIndicesScalar(
    NDArray<int, IntegerMarker> indices,
    T value, {
    int axis = 0,
  }) {
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
  void setIndices(
    NDArray<int, IntegerMarker> indices,
    NDArray values, {
    int axis = 0,
  }) {
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
      final boolMask = spec as NDArray<bool, BooleanMarker>;
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
    } else if (spec is NDArray<int, IntegerMarker>) {
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
      final indices = NDArray<int, Int32Marker>.fromList(
        [spec],
        [1],
        DType.int32,
      );
      if (value is NDArray) {
        setIndices(indices, value);
      } else {
        setIndicesScalar(indices, value as T);
      }
    } else if (spec is List) {
      if (spec.isNotEmpty && spec.first is List) {
        final subList = spec.first as List;
        final intIndices = subList.map((e) => e as int).toList();
        final indices = NDArray<int, Int32Marker>.fromList(intIndices, [
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
      final boolMask = spec as NDArray<bool, BooleanMarker>;
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
        if (value is NDArray<T, M>) {
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
    } else if (spec is NDArray<int, IntegerMarker>) {
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
  NDArray<T, M> slice(List<Selector> selectors) {
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
        final size = shape[i];
        final maskMarker = ScratchArena.marker;
        final List<int> indices;
        try {
          final pIndices = ScratchArena.allocate<ffi.Int>(
            size * ffi.sizeOf<ffi.Int>(),
          );
          final count = unpack_mask_c(
            mask.mask.pointer.cast(),
            size,
            mask.mask.strides[0],
            pIndices,
          );
          indices = pIndices.cast<ffi.Int32>().asTypedList(count).toList();
        } finally {
          ScratchArena.reset(maskMarker);
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
      final result = NDArray<T, M>.create(newShape, dtype);
      final rank = shape.length;

      final sliceMarker = ScratchArena.marker;
      try {
        final pTypes = ScratchArena.allocate<ffi.Int>(
          rank * ffi.sizeOf<ffi.Int>(),
        );
        final pIndexVals = ScratchArena.allocate<ffi.Int>(
          rank * ffi.sizeOf<ffi.Int>(),
        );
        final pSliceStarts = ScratchArena.allocate<ffi.Int>(
          rank * ffi.sizeOf<ffi.Int>(),
        );
        final pSliceStops = ScratchArena.allocate<ffi.Int>(
          rank * ffi.sizeOf<ffi.Int>(),
        );
        final pSliceSteps = ScratchArena.allocate<ffi.Int>(
          rank * ffi.sizeOf<ffi.Int>(),
        );
        final pIndicesPtrs = ScratchArena.allocate<ffi.Pointer<ffi.Int>>(
          rank * ffi.sizeOf<ffi.Pointer<ffi.Int>>(),
        );
        final pIndicesLens = ScratchArena.allocate<ffi.Int>(
          rank * ffi.sizeOf<ffi.Int>(),
        );

        for (var i = 0; i < rank; i++) {
          final selector = i < processedSelectors.length
              ? processedSelectors[i]
              : Slice.all();

          if (selector is Index) {
            pTypes[i] = 0;
            final idx = selector.value < 0
                ? shape[i] + selector.value
                : selector.value;
            pIndexVals[i] = idx;
            pSliceStarts[i] = 0;
            pSliceStops[i] = 0;
            pSliceSteps[i] = 0;
            pIndicesPtrs[i] = ffi.Pointer.fromAddress(0);
            pIndicesLens[i] = 0;
          } else if (selector is Slice) {
            pTypes[i] = 1;
            pIndexVals[i] = 0;

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

            pSliceStarts[i] = startIdx.clamp(0, shape[i] - 1);
            pSliceStops[i] = stopIdx.clamp(-1, shape[i]);
            pSliceSteps[i] = step;
            pIndicesPtrs[i] = ffi.Pointer.fromAddress(0);
            pIndicesLens[i] = 0;
          } else if (selector is Indices) {
            pTypes[i] = 2;
            pIndexVals[i] = 0;
            pSliceStarts[i] = 0;
            pSliceStops[i] = 0;
            pSliceSteps[i] = 0;

            final values = selector.values;
            final pIndices = ScratchArena.allocate<ffi.Int>(
              values.length * ffi.sizeOf<ffi.Int>(),
            );
            for (var j = 0; j < values.length; j++) {
              final idx = values[j];
              final realIdx = idx < 0 ? shape[i] + idx : idx;
              if (realIdx < 0 || realIdx >= shape[i]) {
                throw RangeError.index(
                  realIdx,
                  shape,
                  'index out of range for dimension $i',
                );
              }
              pIndices[j] = realIdx;
            }
            pIndicesPtrs[i] = pIndices;
            pIndicesLens[i] = values.length;
          }
        }

        final pSrcStrides = ScratchArena.copyInts(strides);
        final pSrcShape = ScratchArena.copyInts(shape);

        copy_advanced_c(
          pointer.cast(),
          result.pointer.cast(),
          pSrcStrides,
          pSrcShape,
          rank,
          dtype.byteWidth,
          pTypes,
          pIndexVals,
          pSliceStarts,
          pSliceStops,
          pSliceSteps,
          pIndicesPtrs,
          pIndicesLens,
        );
      } finally {
        ScratchArena.reset(sliceMarker);
      }

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
  NDArray<T, M> take(List<int> indices, {int axis = 0}) {
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
  NDArray<T, M> applyMask(NDArray<bool, BooleanMarker> mask) {
    if (shape.length == 1 &&
        mask.shape.length == 1 &&
        isContiguous &&
        mask.isContiguous) {
      if (mask.shape[0] != shape[0]) {
        throw ArgumentError(
          'Boolean mask shape ${mask.shape} must match target shape $shape',
        );
      }
      final size = shape[0];
      final count = native_count_mask(mask.pointer.cast(), size);
      final result = NDArray<T, M>.create([count], dtype);
      native_apply_mask(
        dtype.index,
        pointer.cast(),
        mask.pointer.cast(),
        result.pointer.cast(),
        size,
      );
      return result;
    }
    return slice([Mask(BooleanMask(mask))]);
  }

  void _copyAdvancedRecursive(
    NDArray<T, M> src,
    NDArray<T, M> dest,
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

  /// Returns a flat Dart list containing a copy of the elements in this array,
  /// traversed in the logical order defined by its shape and strides.
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
    NDArray<T, M> arr,
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
  NDArray<T, M> expandDims(int axis) {
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
  NDArray<T, M> squeeze({dynamic axis}) {
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
  NDArray<T, M> swapaxes(int axis1, int axis2) {
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
  NDArray<T, M> moveaxis(dynamic source, dynamic destination) {
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

    if (!_isExternallyOwned) {
      _finalizer.detach(this);
      malloc.free(_pointer);
    } else {
      if (_customFinalizerInstance != null) {
        _customFinalizerInstance.detach(this);
      }
      if (_customNativeFinalizer != null) {
        final freeFunc = _customNativeFinalizer
            .asFunction<void Function(ffi.Pointer<ffi.Void>)>();
        freeFunc(_pointer);
      }
    }
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    if (other is! NDArray) return false;
    if (dtype != other.dtype) return false;
    if (!listEquals(shape, other.shape)) return false;

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
    final marker = ScratchArena.marker;
    try {
      final cShape = ScratchArena.copyInts(shape);
      final cStrides = ScratchArena.copyInts(strides);
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
      ScratchArena.reset(marker);
    }

    return Object.hash(baseHash, elementsHash);
  }
}

/// Structural elements equality check between two lists.
bool listEquals<E>(List<E>? a, List<E>? b) {
  if (identical(a, b)) return true;
  if (a == null || b == null) return false;
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

/// A wrapper class for boolean masks used in advanced indexing.
final class BooleanMask {
  /// The underlying boolean array.
  final NDArray<bool, BooleanMarker> mask;

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

  /// Returns the absolute value (magnitude) of this complex number.
  double get abs => math.sqrt(real * real + imag * imag);

  /// Returns the argument (phase) of this complex number in radians.
  double get arg => math.atan2(imag, real);

  /// Returns the natural logarithm of this complex number.
  Complex log() => Complex(math.log(abs), arg);

  /// Returns this complex number raised to the power of [exponent].
  ///
  /// Supports [num] and [Complex] exponents.
  Complex pow(dynamic exponent) {
    if (exponent is num) {
      final r = abs;
      final theta = arg;
      final newR = math.pow(r, exponent);
      final newTheta = theta * exponent;
      return Complex(newR * math.cos(newTheta), newR * math.sin(newTheta));
    } else if (exponent is Complex) {
      // z^w = exp(w * log(z))
      final lz = log();
      final prod = exponent * lz;
      final r = math.exp(prod.real);
      return Complex(r * math.cos(prod.imag), r * math.sin(prod.imag));
    } else {
      throw ArgumentError('Unsupported exponent type: ${exponent.runtimeType}');
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
  custom_memcpy(dest._pointer, src._pointer, size * src.dtype.byteWidth);
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

bool _openblasInitialized = false;

void _initializeOpenBLASOnce() {
  if (_openblasInitialized) return;
  _openblasInitialized = true;
  try {
    openblas_set_num_threads(1);
  } catch (_) {
    // Silently ignore library load/init errors in non-OpenBLAS environments
  }
}
