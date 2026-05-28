import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';

/// An Isolate-local scratch memory arena for high-performance transient FFI allocations.
///
/// Bypasses malloc/free boundary overhead by maintaining pre-allocated persistent C heap memory.
final class ScratchArena {
  ScratchArena._();

  static final List<ffi.Pointer<ffi.Uint8>> _pages = [];
  static final List<int> _pageCapacities = [];
  static const int _baseCapacity = 256 * 1024; // 256KB base capacity for page 0
  static int _currentPageIndex = 0;
  static int _offset = 0;

  static ffi.Pointer<ffi.Int>? _stridedBuffer;
  static int _stridedCapacity = 0;

  static void _init() {
    if (_pages.isEmpty) {
      _pages.add(malloc<ffi.Uint8>(_baseCapacity));
      _pageCapacities.add(_baseCapacity);
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

    final currentCapacity = _pageCapacities[_currentPageIndex];

    // If current page doesn't have enough space, switch to next page!
    if (_offset + alignedBytes > currentCapacity) {
      _currentPageIndex++;
      _offset = 0;

      if (_currentPageIndex >= _pages.length) {
        // Geometrically scale capacities: Page 0 (256KB), Page 1 (512KB), Page 2 (1MB) etc.
        final scaledCap = _baseCapacity << _currentPageIndex;
        final targetCap = alignedBytes > scaledCap ? alignedBytes : scaledCap;
        _pages.add(malloc<ffi.Uint8>(targetCap));
        _pageCapacities.add(targetCap);
      } else {
        // If reusing a cached page but its capacity is too small for this allocation:
        if (_pageCapacities[_currentPageIndex] < alignedBytes) {
          // Instead of freeing the small page, move it to the end of the list
          // so it can be reused later for standard page requests.
          final smallPage = _pages.removeAt(_currentPageIndex);
          final smallCap = _pageCapacities.removeAt(_currentPageIndex);
          _pages.add(smallPage);
          _pageCapacities.add(smallCap);

          // Insert the new custom-sized page at the current index.
          _pages.insert(_currentPageIndex, malloc<ffi.Uint8>(alignedBytes));
          _pageCapacities.insert(_currentPageIndex, alignedBytes);
        }
      }
    }

    final currentArena = _pages[_currentPageIndex];
    final ptr = ffi.Pointer<T>.fromAddress(currentArena.address + _offset);
    _offset += alignedBytes;
    return ptr;
  }

  /// Gets the current stack marker in the arena.
  static ScratchMarker get marker =>
      ScratchMarker._(_currentPageIndex, _offset);

  /// Resets the arena stack back to the given [marker].
  static void reset(ScratchMarker marker) {
    final pageIndex = marker.pageIndex;
    final offset = marker.offset;

    assert(pageIndex <= _currentPageIndex);
    if (pageIndex == _currentPageIndex) {
      assert(offset <= _offset);
    }

    // Selective Pruning: Deallocate any excess pages located beyond the
    // target marker to prevent native memory leaks and bloating.
    // We keep up to 2 standard pages (Page 0 and 1, totaling 768KB) persistently.
    if (_pages.length > 2 && _pages.length > pageIndex + 1) {
      final preserveCount = pageIndex + 1 > 2 ? pageIndex + 1 : 2;
      while (_pages.length > preserveCount) {
        final excessPage = _pages.removeLast();
        _pageCapacities.removeLast();
        malloc.free(excessPage);
      }
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

/// Represents a stable checkpoint marker for the [ScratchArena] memory stack.
final class ScratchMarker {
  /// The page index inside the ScratchArena page pool when the marker was recorded.
  final int pageIndex;

  /// The byte offset inside the page when the marker was recorded.
  final int offset;

  const ScratchMarker._(this.pageIndex, this.offset);
}
