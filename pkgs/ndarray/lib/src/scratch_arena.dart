import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';

/// An Isolate-local scratch memory arena for high-performance transient FFI allocations.
///
/// Bypasses malloc/free boundary overhead by maintaining pre-allocated persistent C heap memory.
final class ScratchArena {
  ScratchArena._();

  static final List<ffi.Pointer<ffi.Uint8>> _pages = [];
  static const int _capacity = 2 * 1024 * 1024; // 2MB per page
  static int _currentPageIndex = 0;
  static int _offset = 0;

  static ffi.Pointer<ffi.Int>? _stridedBuffer;
  static int _stridedCapacity = 0;

  static void _init() {
    if (_pages.isEmpty) {
      _pages.add(malloc<ffi.Uint8>(_capacity));
    }
  }

  /// Allocates [bytes] of memory from the arena stack, aligned to 8 bytes.
  ///
  /// To free allocations, record [marker] before calling this and reset the stack
  /// back to the marker using [reset] when done.
  static ffi.Pointer<T> allocate<T extends ffi.NativeType>(int bytes) {
    _init();

    // Ensure 8-byte alignment for native FFI alignment requirements
    final alignedBytes = (bytes + 7) & ~7;

    // If single allocation exceeds standard 2MB page capacity, throw
    if (alignedBytes > _capacity) {
      throw StateError(
        'Single allocation size ($bytes bytes) exceeds ScratchArena page capacity ($_capacity bytes).',
      );
    }

    // If current page doesn't have enough space, switch to next page!
    if (_offset + alignedBytes > _capacity) {
      _currentPageIndex++;
      _offset = 0;
      if (_currentPageIndex >= _pages.length) {
        _pages.add(malloc<ffi.Uint8>(_capacity));
      }
    }

    final currentArena = _pages[_currentPageIndex];
    final ptr = ffi.Pointer<T>.fromAddress(currentArena.address + _offset);
    _offset += alignedBytes;
    return ptr;
  }

  /// Gets the current stack marker (packed page index and offset) in the arena.
  static int get marker => (_currentPageIndex << 32) | _offset;

  /// Resets the arena stack back to the given [marker].
  static void reset(int marker) {
    final pageIndex = marker >> 32;
    final offset = marker & 0xFFFFFFFF;

    assert(pageIndex <= _currentPageIndex);
    if (pageIndex == _currentPageIndex) {
      assert(offset <= _offset);
    }

    _currentPageIndex = pageIndex;
    _offset = offset;
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
