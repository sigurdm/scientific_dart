import 'dart:ffi' as ffi;
import 'package:resource_scope/resource_scope.dart';
import 'exceptions.dart';
import 'buffer.dart';

/// The type of GPU hardware or execution environment.
enum GpuDeviceType {
  /// Dedicated discrete GPU (e.g. NVIDIA, AMD Radeon).
  discrete,

  /// Integrated GPU sharing system memory (e.g. Intel Iris, Apple Silicon).
  integrated,

  /// Virtualized or software-rasterized compute device.
  virtual,

  /// CPU fallback compute device.
  cpu,
}

/// Represents a GPU compute device capable of allocating memory and executing compute kernels.
final class GpuDevice implements ScopedResource {
  static GpuDevice? _defaultDevice;

  /// The default compute device for the environment.
  static GpuDevice get defaultDevice {
    if (_defaultDevice != null && !_defaultDevice!._isDisposed) {
      return _defaultDevice!;
    }
    final dev = ResourceScope.unmanaged(
      () => GpuDevice._(
        name: 'Default GPU Device',
        type: GpuDeviceType.virtual,
        trackInScope: false,
      ),
    );
    _defaultDevice = dev;
    return dev;
  }

  /// Name identifier of the GPU device.
  final String name;

  /// Hardware category of this device.
  final GpuDeviceType type;

  final Set<WeakReference<GpuBuffer>> _activeBuffers = {};
  bool _isDisposed = false;

  GpuDevice._({
    required this.name,
    required this.type,
    bool trackInScope = true,
  }) {
    if (trackInScope) {
      ResourceScope.track(this);
    }
  }

  /// Creates a new custom [GpuDevice] instance.
  factory GpuDevice.create({
    String name = 'Custom GPU Device',
    GpuDeviceType type = GpuDeviceType.virtual,
  }) {
    return GpuDevice._(name: name, type: type, trackInScope: true);
  }

  @override
  bool get isDisposed => _isDisposed;

  void _pruneDeadBuffers() {
    _activeBuffers.removeWhere((ref) => ref.target == null);
  }

  /// Total number of active buffers currently allocated on this device.
  int get activeBufferCount {
    _pruneDeadBuffers();
    return _activeBuffers.length;
  }

  /// Total bytes of memory currently allocated on this device.
  int get allocatedMemoryBytes {
    _pruneDeadBuffers();
    var sum = 0;
    for (final ref in _activeBuffers) {
      final buf = ref.target;
      if (buf != null && !buf.isDisposed) {
        sum += buf.sizeInBytes;
      }
    }
    return sum;
  }

  /// Allocates a new [GpuBuffer] on this device.
  GpuBuffer createBuffer({
    required int sizeInBytes,
    required GpuBufferUsage usage,
  }) {
    if (_isDisposed) {
      throw GpuDeviceDisposedException(
        'Cannot allocate on a disposed GpuDevice.',
      );
    }
    _pruneDeadBuffers();
    return GpuBuffer.allocate(
      sizeInBytes: sizeInBytes,
      usage: usage,
      device: this,
    );
  }

  /// Allocates a [GpuBuffer] on this device and initializes it with data from [pointer].
  GpuBuffer createBufferWithData(
    ffi.Pointer<ffi.Void> pointer,
    int sizeInBytes,
    GpuBufferUsage usage,
  ) {
    final buffer = createBuffer(
      sizeInBytes: sizeInBytes,
      usage: usage | GpuBufferUsage.copyDst,
    );
    buffer.copyFromHost(pointer, sizeInBytes);
    return buffer;
  }

  /// Reads [sizeInBytes] from [buffer] into the host [pointer].
  void readBufferIntoPointer(
    GpuBuffer buffer,
    ffi.Pointer<ffi.Void> pointer,
    int sizeInBytes,
  ) {
    if (_isDisposed) {
      throw GpuDeviceDisposedException(
        'Cannot read from a disposed GpuDevice.',
      );
    }
    buffer.copyToHost(pointer, sizeInBytes);
  }

  /// Waits for all pending compute tasks on this device to complete.
  void synchronize() {
    if (_isDisposed) {
      throw GpuDeviceDisposedException(
        'Cannot synchronize a disposed GpuDevice.',
      );
    }
    // Execution is currently synchronous; hook for async execution queue drains.
  }

  /// Registers an active buffer with this device.
  void registerBuffer(GpuBuffer buffer) {
    _pruneDeadBuffers();
    _activeBuffers.add(WeakReference(buffer));
  }

  /// Unregisters a buffer from this device when it is disposed.
  void unregisterBuffer(GpuBuffer buffer) {
    _activeBuffers.removeWhere((ref) {
      final target = ref.target;
      return target == null || identical(target, buffer);
    });
  }

  @override
  void dispose() {
    if (_isDisposed) return;
    _isDisposed = true;
    ResourceScope.untrack(this);
    final buffers = _activeBuffers
        .map((ref) => ref.target)
        .whereType<GpuBuffer>()
        .toList();
    for (final buf in buffers) {
      if (!buf.isDisposed) {
        buf.dispose();
      }
    }
    _activeBuffers.clear();
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
