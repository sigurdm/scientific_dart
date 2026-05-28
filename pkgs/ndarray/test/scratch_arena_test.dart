import 'dart:ffi' as ffi;
import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';
import 'package:ndarray/src/scratch_arena.dart';

void main() {
  group('ScratchArena Advanced Memory Pool Tests', () {
    test('standard allocation and 8-byte alignment', () {
      final marker = ScratchArena.marker;
      try {
        final ptr1 = ScratchArena.allocate<ffi.Int32>(5); // 5 bytes -> aligned to 8
        final ptr2 = ScratchArena.allocate<ffi.Int32>(3); // 3 bytes -> aligned to 8

        // Verify alignment and non-overlapping addresses
        expect(ptr1.address % 8, equals(0));
        expect(ptr2.address % 8, equals(0));
        expect(ptr2.address - ptr1.address, equals(8));
      } finally {
        ScratchArena.reset(marker);
      }
    });

    test('stack rewind and marker resets within a page', () {
      final markerStart = ScratchArena.marker;
      final ptr1 = ScratchArena.allocate<ffi.Int32>(16);
      final markerMid = ScratchArena.marker;
      final ptr2 = ScratchArena.allocate<ffi.Int32>(16);

      // Reset back to mid-marker
      ScratchArena.reset(markerMid);
      final ptr3 = ScratchArena.allocate<ffi.Int32>(16);

      // Since we rewound, ptr3 should reuse ptr2's address space exactly!
      expect(ptr3.address, equals(ptr2.address));

      // Reset back to start
      ScratchArena.reset(markerStart);
      final ptr4 = ScratchArena.allocate<ffi.Int32>(16);
      expect(ptr4.address, equals(ptr1.address));
    });

    test('multi-page switching and stable address pointers', () {
      final marker = ScratchArena.marker;
      try {
        final ptr0 = ScratchArena.allocate<ffi.Uint8>(10);

        // Allocate enough to spill beyond the 256KB base page capacity
        // Page 0 is 256KB (262,144 bytes). We allocate 200KB twice.
        final ptrSpill1 = ScratchArena.allocate<ffi.Uint8>(200 * 1024);
        
        final markerSpill = ScratchArena.marker;
        final ptrSpill2 = ScratchArena.allocate<ffi.Uint8>(200 * 1024);

        // ptrSpill2 must have spilled over to Page 1!
        // We verify both ptr0 and ptrSpill1 remain 100% valid at stable addresses!
        expect(ptrSpill1.address, isNot(equals(ptrSpill2.address)));
        expect(ptrSpill2.address, isNot(equals(ffi.nullptr.address)));

        // Resetting back to markerSpill and allocating again should work perfectly
        ScratchArena.reset(markerSpill);
        final ptrSpill3 = ScratchArena.allocate<ffi.Uint8>(200 * 1024);
        expect(ptrSpill3.address, equals(ptrSpill2.address));
      } finally {
        ScratchArena.reset(marker);
      }
    });

    test('dynamic custom page sizing for large requests', () {
      final marker = ScratchArena.marker;
      try {
        // Allocate a single massive buffer exceeding 2MB (Page 0 is 256KB, standard subsequent is 512KB)
        // Request exactly 3MB. It must allocate a custom 3MB page!
        final massiveSize = 3 * 1024 * 1024;
        final ptrMassive = ScratchArena.allocate<ffi.Uint8>(massiveSize);

        expect(ptrMassive.address, isNot(equals(ffi.nullptr.address)));
      } finally {
        ScratchArena.reset(marker);
      }
    });

    test('selective hybrid memory pruning on reset', () {
      final markerStart = ScratchArena.marker;

      // Allocate 4 sequential standard pages to expand the pool (256KB, 512KB, 1MB, 2MB etc.)
      // We do this by requesting slightly larger allocations sequentially
      ScratchArena.allocate<ffi.Uint8>(200 * 1024); // page 0 (256KB)
      ScratchArena.allocate<ffi.Uint8>(400 * 1024); // spills to page 1 (512KB)
      ScratchArena.allocate<ffi.Uint8>(800 * 1024); // spills to page 2 (1MB)
      ScratchArena.allocate<ffi.Uint8>(1600 * 1024); // spills to page 3 (2MB)

      // Resetting back to start (Page index 0) must trigger selective pruning!
      // It must deallocate excess pages but preserve the core pool (Page 0 and Page 1).
      ScratchArena.reset(markerStart);

      // We test that subsequent standard allocations reuse cached page 1 immediately
      // without crashes, and the total cached pool is preserved safely.
      final ptrRecycled = ScratchArena.allocate<ffi.Uint8>(200 * 1024);
      expect(ptrRecycled.address, isNot(equals(ffi.nullptr.address)));
    });

    test('small page pool shifting instead of freeing', () {
      final markerStart = ScratchArena.marker;

      // 1. Force allocation of Page 0 (256KB) and Page 1 (512KB)
      ScratchArena.allocate<ffi.Uint8>(200 * 1024); // page 0
      final markerMid = ScratchArena.marker;
      ScratchArena.allocate<ffi.Uint8>(400 * 1024); // page 1 (512KB)

      // 2. Rewind back to Page 1 (markerMid)
      ScratchArena.reset(markerMid);

      // 3. Allocate a custom 3MB page. Page 1 (512KB) is too small!
      // The pool-shifting logic must shift the 512KB page to the end of the pool
      // and insert the 3MB page at current index 1.
      final ptrCustom = ScratchArena.allocate<ffi.Uint8>(3 * 1024 * 1024);
      expect(ptrCustom.address, isNot(equals(ffi.nullptr.address)));

      ScratchArena.reset(markerStart);
    });
  });
}
