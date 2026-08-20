import 'dart:ffi' as ffi;
import 'package:test/test.dart';
import 'package:resource_scope/resource_scope.dart';
import 'package:gpuarray/src/backend/backend.dart';
import 'package:gpuarray/src/backend/memory_pool.dart';
import 'package:gpuarray/gpuarray.dart';

class TrackingBackend extends GpuBackend {
  final CpuVectorBackend _inner = const CpuVectorBackend();
  int allocCount = 0;
  int freeCount = 0;

  @override
  GpuDeviceType get deviceType => GpuDeviceType.cpu;

  @override
  ffi.Pointer<ffi.Uint8> allocateBuffer(int sizeInBytes) {
    allocCount++;
    return _inner.allocateBuffer(sizeInBytes);
  }

  @override
  void freeBuffer(ffi.Pointer<ffi.Uint8> pointer, int sizeInBytes) {
    freeCount++;
    _inner.freeBuffer(pointer, sizeInBytes);
  }
}

void main() {
  group('VRAM Block Memory Pool & Backend Drivers', () {
    test('Power-of-2 bucket sizing calculation', () {
      expect(GpuMemoryPool.computeBucketSize(10), equals(64));
      expect(GpuMemoryPool.computeBucketSize(64), equals(64));
      expect(GpuMemoryPool.computeBucketSize(65), equals(128));
      expect(GpuMemoryPool.computeBucketSize(1000), equals(1024));
      expect(GpuMemoryPool.computeBucketSize(1024), equals(1024));
      expect(GpuMemoryPool.computeBucketSize(1025), equals(2048));
    });

    test('GpuMemoryPool caches and reuses buffers in O(1)', () {
      final backend = TrackingBackend();
      final device = GpuDevice.create(
        name: 'Pooled Device',
        backend: backend,
        enableMemoryPool: true,
      );

      expect(device.memoryPool.hits, equals(0));
      expect(device.memoryPool.misses, equals(0));

      // 1st allocation: miss -> allocates 128 bytes bucket
      final buf1 = device.createBuffer(
        sizeInBytes: 100,
        usage: GpuBufferUsage.storage,
      );
      expect(device.memoryPool.misses, equals(1));
      expect(device.memoryPool.hits, equals(0));
      expect(backend.allocCount, equals(1));

      // Dispose buf1 -> recycled into pool
      buf1.dispose();
      expect(device.memoryPool.cachedBytes, equals(128));

      // 2nd allocation of same size bucket -> cache hit!
      final buf2 = device.createBuffer(
        sizeInBytes: 120,
        usage: GpuBufferUsage.storage,
      );
      expect(device.memoryPool.hits, equals(1));
      expect(device.memoryPool.misses, equals(1));
      expect(backend.allocCount, equals(1)); // No new native allocation!

      buf2.dispose();
      device.dispose();
    });

    test('GpuMemoryPool trim purges cached buffers to OS', () {
      final backend = TrackingBackend();
      final device = GpuDevice.create(
        backend: backend,
        enableMemoryPool: true,
      );

      final buf = device.createBuffer(sizeInBytes: 500, usage: GpuBufferUsage.storage);
      buf.dispose();
      expect(device.memoryPool.cachedBytes, equals(512));

      device.memoryPool.trim();
      expect(device.memoryPool.cachedBytes, equals(0));
      expect(backend.freeCount, equals(1));

      device.dispose();
    });

    test('ResourceScope seamlessly integrates with GpuMemoryPool', () {
      final device = GpuDevice.create(enableMemoryPool: true);

      final result = ResourceScope.returning(() {
        final a = GpuArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64, device: device);
        final b = GpuArray.fromList([10.0, 20.0, 30.0, 40.0], [4], DType.float64, device: device);
        final c = a + b;
        return c;
      });

      expect(result.toList(), equals([11.0, 22.0, 33.0, 44.0]));
      result.dispose();
      device.dispose();
    });
  });
}
