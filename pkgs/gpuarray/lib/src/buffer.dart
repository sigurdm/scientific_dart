import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'package:resource_scope/resource_scope.dart';
import 'exceptions.dart';
import 'device.dart';

/// Usage flags for a [GpuBuffer], indicating allowed operations and access modes.
final class GpuBufferUsage {
  /// The underlying bitmask value.
  final int value;

  const GpuBufferUsage(this.value);

  /// Buffer can be used as a storage buffer in compute shaders.
  static const GpuBufferUsage storage = GpuBufferUsage(1 << 0);

  /// Buffer can be used as the source in a copy operation.
  static const GpuBufferUsage copySrc = GpuBufferUsage(1 << 1);

  /// Buffer can be used as the destination in a copy operation.
  static const GpuBufferUsage copyDst = GpuBufferUsage(1 << 2);

  /// Buffer can be used as a uniform buffer in compute shaders.
  static const GpuBufferUsage uniform = GpuBufferUsage(1 << 3);

  /// Combines two usage flags.
  GpuBufferUsage operator |(GpuBufferUsage other) =>
      GpuBufferUsage(value | other.value);

  /// Checks if this usage contains the given [flag].
  bool contains(GpuBufferUsage flag) => (value & flag.value) == flag.value;

  @override
  String toString() {
    final parts = <String>[];
    if (contains(storage)) parts.add('STORAGE');
    if (contains(copySrc)) parts.add('COPY_SRC');
    if (contains(copyDst)) parts.add('COPY_DST');
    if (contains(uniform)) parts.add('UNIFORM');
    return 'GpuBufferUsage(${parts.join('|')})';
  }
}

/// A block of memory allocated on a GPU device.
///
/// Implements [ScopedResource] for automatic memory management within [ResourceScope.scope].
final class GpuBuffer implements ffi.Finalizable, ScopedResource {
  static final _finalizer = ffi.NativeFinalizer(pkgFreeFunc);

  final ffi.Pointer<ffi.Void> _pointer;
  final int _sizeInBytes;
  final GpuBufferUsage _usage;
  final GpuDevice _device;
  int _refCount = 1;
  bool _isDisposed = false;

  GpuBuffer._(this._pointer, this._sizeInBytes, this._usage, this._device) {
    if (_pointer != ffi.nullptr) {
      _finalizer.attach(this, _pointer, detach: this);
    }
    ResourceScope.track(this);
    _device.registerBuffer(this);
  }

  /// Retains a reference to this buffer, incrementing its reference count.
  void retain() {
    if (_isDisposed) {
      throw GpuMemoryException('Cannot retain a disposed GpuBuffer.');
    }
    _refCount++;
  }

  /// Releases a reference to this buffer, decrementing its reference count.
  /// If the reference count drops to 0, frees the underlying native memory.
  void release() {
    if (_isDisposed) return;
    _refCount--;
    if (_refCount <= 0) {
      _disposeInternal();
    }
  }

  /// Reference count of active holders of this buffer.
  int get refCount => _refCount;

  void _disposeInternal() {
    if (_isDisposed) return;
    _isDisposed = true;
    ResourceScope.untrack(this);
    _device.unregisterBuffer(this);
    if (_pointer != ffi.nullptr) {
      _finalizer.detach(this);
      calloc.free(_pointer);
    }
  }

  /// Allocates a new [GpuBuffer] on the given [device].
  factory GpuBuffer.allocate({
    required int sizeInBytes,
    required GpuBufferUsage usage,
    required GpuDevice device,
  }) {
    if (sizeInBytes < 0) {
      throw ArgumentError.value(
        sizeInBytes,
        'sizeInBytes',
        'Buffer size cannot be negative.',
      );
    }
    if (device.isDisposed) {
      throw GpuDeviceDisposedException(
        'Cannot allocate buffer on disposed device.',
      );
    }

    final ptr = sizeInBytes == 0
        ? ffi.nullptr
        : calloc.allocate<ffi.Uint8>(sizeInBytes).cast<ffi.Void>();

    return GpuBuffer._(ptr, sizeInBytes, usage, device);
  }

  /// Pointer to the underlying memory block.
  ffi.Pointer<ffi.Void> get pointer {
    if (_isDisposed) {
      throw GpuMemoryException('Cannot access memory of a disposed GpuBuffer.');
    }
    return _pointer;
  }

  /// Size of this buffer in bytes.
  int get sizeInBytes => _sizeInBytes;

  /// Usage flags of this buffer.
  GpuBufferUsage get usage => _usage;

  /// The [GpuDevice] where this buffer is allocated.
  GpuDevice get device => _device;

  @override
  bool get isDisposed => _isDisposed;

  /// Copies [bytes] from a host memory [hostPtr] into this GPU buffer starting at [offset].
  void copyFromHost(
    ffi.Pointer<ffi.Void> hostPtr,
    int bytes, {
    int offset = 0,
  }) {
    if (_isDisposed) {
      throw GpuMemoryException('Cannot copy to a disposed GpuBuffer.');
    }
    if (offset < 0 || bytes < 0 || offset + bytes > _sizeInBytes) {
      throw RangeError(
        'Copy out of bounds: offset=$offset, bytes=$bytes, bufferSize=$_sizeInBytes',
      );
    }
    if (bytes == 0) return;

    final dst = (_pointer.cast<ffi.Uint8>() + offset).cast<ffi.Uint8>();
    final src = hostPtr.cast<ffi.Uint8>();
    dst.asTypedList(bytes).setAll(0, src.asTypedList(bytes));
  }

  /// Copies [bytes] from this GPU buffer starting at [offset] into a host memory [hostPtr].
  void copyToHost(ffi.Pointer<ffi.Void> hostPtr, int bytes, {int offset = 0}) {
    if (_isDisposed) {
      throw GpuMemoryException('Cannot copy from a disposed GpuBuffer.');
    }
    if (offset < 0 || bytes < 0 || offset + bytes > _sizeInBytes) {
      throw RangeError(
        'Copy out of bounds: offset=$offset, bytes=$bytes, bufferSize=$_sizeInBytes',
      );
    }
    if (bytes == 0) return;

    final src = (_pointer.cast<ffi.Uint8>() + offset).cast<ffi.Uint8>();
    final dst = hostPtr.cast<ffi.Uint8>();
    dst.asTypedList(bytes).setAll(0, src.asTypedList(bytes));
  }

  /// Copies [bytes] from this buffer into another [dst] buffer.
  void copyToBuffer(
    GpuBuffer dst,
    int bytes, {
    int srcOffset = 0,
    int dstOffset = 0,
  }) {
    if (_isDisposed || dst._isDisposed) {
      throw GpuMemoryException('Cannot copy using disposed GpuBuffers.');
    }
    if (srcOffset < 0 || bytes < 0 || srcOffset + bytes > _sizeInBytes) {
      throw RangeError(
        'Source copy out of bounds: $srcOffset + $bytes > $_sizeInBytes',
      );
    }
    if (dstOffset < 0 || dstOffset + bytes > dst._sizeInBytes) {
      throw RangeError(
        'Destination copy out of bounds: $dstOffset + $bytes > ${dst._sizeInBytes}',
      );
    }
    if (bytes == 0) return;

    final srcPtr = (_pointer.cast<ffi.Uint8>() + srcOffset).cast<ffi.Uint8>();
    final dstPtr = (dst._pointer.cast<ffi.Uint8>() + dstOffset)
        .cast<ffi.Uint8>();
    dstPtr.asTypedList(bytes).setAll(0, srcPtr.asTypedList(bytes));
  }

  @override
  void dispose() {
    release();
  }

  @override
  ScopedResource detachFromScope() {
    ResourceScope.untrack(this);
    return this;
  }

  @override
  ScopedResource detachToParentScope() {
    ResourceScope.promoteToParent(this);
    return this;
  }
}

/// Helper native free function pointer for NativeFinalizer.
final ffi.Pointer<ffi.NativeFunction<ffi.Void Function(ffi.Pointer<ffi.Void>)>>
pkgFreeFunc = ffi.Pointer.fromAddress(calloc.nativeFree.address);
