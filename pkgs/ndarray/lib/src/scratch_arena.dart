import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';

/// An Isolate-local scratch memory arena for high-performance transient FFI allocations.
///
/// Bypasses malloc/free boundary overhead by maintaining pre-allocated persistent C heap memory.
final class ScratchArena {
  ScratchArena._();

  static ffi.Pointer<ffi.Uint8>? _arena;
  static int _capacity =
      4096; // 4KB page is plenty for rank pointers, temp arrays etc.
  static int _offset = 0;

  static ffi.Pointer<ffi.Int>? _stridedBuffer;
  static int _stridedCapacity = 0;

  static void _init() {
    _arena ??= malloc<ffi.Uint8>(_capacity);
  }

  /// Allocates [bytes] of memory from the arena stack, aligned to 8 bytes.
  ///
  /// To free allocations, record [marker] before calling this and reset the stack
  /// back to the marker using [reset] when done.
  static ffi.Pointer<T> allocate<T extends ffi.NativeType>(int bytes) {
    _init();

    // Ensure 8-byte alignment for native FFI alignment requirements
    final alignedBytes = (bytes + 7) & ~7;

    if (_offset + alignedBytes > _capacity) {
      final newCapacity = _capacity * 2 + alignedBytes;
      final newArena = malloc<ffi.Uint8>(newCapacity);
      if (_offset > 0) {
        newArena
            .asTypedList(newCapacity)
            .setRange(
              0,
              _offset,
              _arena!.asTypedList(_capacity).sublist(0, _offset),
            );
      }
      malloc.free(_arena!);
      _arena = newArena;
      _capacity = newCapacity;
    }

    final ptr = ffi.Pointer<T>.fromAddress(_arena!.address + _offset);
    _offset += alignedBytes;
    return ptr;
  }

  /// Gets the current stack marker (offset) in the arena.
  static int get marker => _offset;

  /// Resets the arena stack back to the given [marker].
  static void reset(int marker) {
    assert(marker <= _offset);
    _offset = marker;
  }

  /// Gets or grows a persistent thread-local static buffer for leaf strided operations.
  ///
  /// Designed for synchronous leaf operations that do not make nested calls.
  static ffi.Pointer<ffi.Int> getStridedBuffer(int ndim) {
    final requiredSize = ndim * 4;
    if (_stridedBuffer == null || _stridedCapacity < requiredSize) {
      if (_stridedBuffer != null) {
        malloc.free(_stridedBuffer!);
      }
      _stridedBuffer = malloc<ffi.Int>(requiredSize);
      _stridedCapacity = requiredSize;
    }
    return _stridedBuffer!;
  }
}
