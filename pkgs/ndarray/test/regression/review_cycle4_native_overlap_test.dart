import 'dart:ffi' as ffi;
import 'dart:math' as math;
import 'package:ndarray/ndarray.dart';
import 'package:ndarray/src/ndarray_bindings.dart' as bindings;
import 'package:ndarray/src/scratch_arena.dart';
import 'package:test/test.dart';

void main() {
  group('Review Cycle 4 Native Overlap Fixes', () {
    test('contiguous partial slice overlap in add', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 10.0, 100.0, 1000.0, 10000.0],
          [5],
          DType.float64,
        );
        final src1 = a.slice([const Slice(start: 0, stop: 4)]);
        final src2 = a.slice([const Slice(start: 0, stop: 4)]);
        final dst = a.slice([const Slice(start: 1, stop: 5)]);

        add(src1, src2, out: dst);

        expect(a.getCell([0]).value, closeTo(1.0, 1e-12));
        expect(a.getCell([1]).value, closeTo(2.0, 1e-12));
        expect(a.getCell([2]).value, closeTo(20.0, 1e-12));
        expect(a.getCell([3]).value, closeTo(200.0, 1e-12));
        expect(a.getCell([4]).value, closeTo(2000.0, 1e-12));
      });
    });

    test('contiguous partial slice overlap in sin', () {
      NDArray.scope(() {
        final orig = [0.5, 1.0, 1.5, 2.0, 2.5];
        final a = NDArray<Float64>.fromList(orig, [5], DType.float64);
        final src = a.slice([const Slice(start: 0, stop: 4)]);
        final dst = a.slice([const Slice(start: 1, stop: 5)]);

        sin(src, out: dst);

        expect(a.getCell([0]).value, closeTo(orig[0], 1e-12));
        expect(a.getCell([1]).value, closeTo(math.sin(orig[0]), 1e-12));
        expect(a.getCell([2]).value, closeTo(math.sin(orig[1]), 1e-12));
        expect(a.getCell([3]).value, closeTo(math.sin(orig[2]), 1e-12));
        expect(a.getCell([4]).value, closeTo(math.sin(orig[3]), 1e-12));
      });
    });

    test('where(cond, flip(a), b, out: a) handles strided overlap', () {
      NDArray.scope(() {
        final cond = NDArray<Boolean>.fromList(
          [true, true, false, true],
          [4],
          DType.boolean,
        );
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float64,
        );
        final b = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [4],
          DType.float64,
        );

        where<Float64>(cond, flip(a), b, a);

        expect(a.getCell([0]).value, closeTo(4.0, 1e-12));
        expect(a.getCell([1]).value, closeTo(3.0, 1e-12));
        expect(a.getCell([2]).value, closeTo(30.0, 1e-12));
        expect(a.getCell([3]).value, closeTo(1.0, 1e-12));
      });
    });

    test('native s_where_double handles strided overlap directly', () {
      NDArray.scope(() {
        final cond = NDArray<Boolean>.fromList(
          [true, true, false, true],
          [4],
          DType.boolean,
        );
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float64,
        );
        final b = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [4],
          DType.float64,
        );
        final aFlipped = flip(a);

        final marker = ScratchArena.marker;
        try {
          final cShape = ScratchArena.allocate<ffi.Int>(ffi.sizeOf<ffi.Int>());
          final cStridesCond = ScratchArena.allocate<ffi.Int>(
            ffi.sizeOf<ffi.Int>(),
          );
          final cStridesX = ScratchArena.allocate<ffi.Int>(
            ffi.sizeOf<ffi.Int>(),
          );
          final cStridesY = ScratchArena.allocate<ffi.Int>(
            ffi.sizeOf<ffi.Int>(),
          );
          final cStridesRes = ScratchArena.allocate<ffi.Int>(
            ffi.sizeOf<ffi.Int>(),
          );
          cShape[0] = 4;
          cStridesCond[0] = cond.strides[0];
          cStridesX[0] = aFlipped.strides[0];
          cStridesY[0] = b.strides[0];
          cStridesRes[0] = a.strides[0];

          bindings.s_where_double(
            cond.pointer.cast(),
            cStridesCond,
            aFlipped.pointer.cast(),
            cStridesX,
            b.pointer.cast(),
            cStridesY,
            a.pointer.cast(),
            cStridesRes,
            cShape,
            1,
          );
        } finally {
          ScratchArena.reset(marker);
        }

        expect(a.getCell([0]).value, closeTo(4.0, 1e-12));
        expect(a.getCell([1]).value, closeTo(3.0, 1e-12));
        expect(a.getCell([2]).value, closeTo(30.0, 1e-12));
        expect(a.getCell([3]).value, closeTo(1.0, 1e-12));
      });
    });

    test('unwrap(flip(a), out: a) handles strided overlap', () {
      NDArray.scope(() {
        final phase = [0.0, 0.5, 1.0, 1.0 + 2 * math.pi, 1.5 + 4 * math.pi];
        final a = NDArray<Float64>.fromList(phase, [5], DType.float64);
        final flippedCopy = flip(a).copy();
        final expected = unwrap<Float64>(flippedCopy);

        unwrap<Float64>(flip(a), out: a);

        for (var i = 0; i < 5; i++) {
          expect(
            a.getCell([i]).value,
            closeTo(expected.getCell([i]).value, 1e-12),
          );
        }
      });
    });

    test('native s_unwrap_double handles strided overlap directly', () {
      NDArray.scope(() {
        final phase = [0.0, 0.5, 1.0, 1.0 + 2 * math.pi, 1.5 + 4 * math.pi];
        final a = NDArray<Float64>.fromList(phase, [5], DType.float64);
        final flippedCopy = flip(a).copy();
        final expected = unwrap<Float64>(flippedCopy);
        final aFlipped = flip(a);

        final marker = ScratchArena.marker;
        try {
          final cShape = ScratchArena.allocate<ffi.Int>(ffi.sizeOf<ffi.Int>());
          final cStridesSrc = ScratchArena.allocate<ffi.Int>(
            ffi.sizeOf<ffi.Int>(),
          );
          final cStridesRes = ScratchArena.allocate<ffi.Int>(
            ffi.sizeOf<ffi.Int>(),
          );
          cShape[0] = 5;
          cStridesSrc[0] = aFlipped.strides[0];
          cStridesRes[0] = a.strides[0];

          bindings.s_unwrap_double(
            aFlipped.pointer.cast(),
            cStridesSrc,
            a.pointer.cast(),
            cStridesRes,
            cShape,
            1,
            0,
            math.pi,
          );
        } finally {
          ScratchArena.reset(marker);
        }

        for (var i = 0; i < 5; i++) {
          expect(
            a.getCell([i]).value,
            closeTo(expected.getCell([i]).value, 1e-12),
          );
        }
      });
    });
  });
}
