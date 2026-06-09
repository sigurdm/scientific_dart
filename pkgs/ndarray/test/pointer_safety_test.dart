import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';

void main() {
  group('Pointer Safety & View Correctness (Phase 2)', () {
    test('Negative strides pointer offsetting', () {
      final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
      // Slice with negative stride: [4.0, 3.0, 2.0, 1.0]
      final view = a.slice([Slice(step: -1)]);

      expect(view.toList(), [4.0, 3.0, 2.0, 1.0]);
      // Local offset model: view.data starts at physical 0 (since minPhysicalOffset is 0).
      // Logical start is physical 3.
      // So offset elements relative to data start is 3.
      expect(view.offsetElements, 3);

      // The logical start is at index 3.
      // So view.pointer should be parent.pointer + 3 * 8 bytes.
      final expectedAddress = a.pointer.address + 3 * 8;
      expect(view.pointer.address, expectedAddress);
    });

    test('Nested view offset accumulation', () {
      final a = NDArray.fromList(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [8],
        DType.float64,
      );
      // View 1: slice [2:7] -> [2.0, 3.0, 4.0, 5.0, 6.0]
      final v1 = a.slice([Slice(start: 2, stop: 7)]);
      expect(v1.pointer.address, a.pointer.address + 2 * 8);
      expect(v1.offsetElements, 0); // starts at data start (physical 2)

      // View 2: slice of v1 with negative stride [::-1] -> [6.0, 5.0, 4.0, 3.0, 2.0]
      // Logical start of v2 is physical index 6 of root.
      // Physical range is [2, 6], so v2.data starts at physical 2.
      // Logical start (6) is index 4 relative to v2.data start (2).
      final v2 = v1.slice([Slice(step: -1)]);
      expect(v2.offsetElements, 4);
      expect(v2.pointer.address, a.pointer.address + 6 * 8);

      // View 3: slice of v2 with positive stride [1:4] -> [5.0, 4.0, 3.0]
      // v2 indices: 1, 2, 3 -> physical indices: 5, 4, 3.
      // Logical start of v3 is physical 5.
      // Physical range is [3, 5], so v3.data starts at physical 3.
      // Logical start (5) is index 2 relative to v3.data start (3).
      final v3 = v2.slice([Slice(start: 1, stop: 4)]);
      expect(v3.offsetElements, 2);
      expect(v3.pointer.address, a.pointer.address + 5 * 8);
      expect(v3.toList(), [5.0, 4.0, 3.0]);
    });

    test('min() along axis on negative-stride view fallback path', () {
      final a = NDArray.fromList(
        [10.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [2, 3],
        DType.float64,
      );

      // Slice to get a negative stride view along axis 1:
      // [3.0, 2.0, 10.0]
      // [6.0, 5.0,  4.0]
      final view = a.slice([Slice(), Slice(step: -1)]);
      expect(view.toList(), [3.0, 2.0, 10.0, 6.0, 5.0, 4.0]);

      // Now call min along axis 0.
      // Expected result: [min(3,6), min(2,5), min(10,4)] = [3.0, 2.0, 4.0]
      final res = min(view, axis: 0);
      expect(res.toList(), [3.0, 2.0, 4.0]);
    });

    test('cumprod() fallback on negative-stride view', () {
      // cumprod uses cumOpFFI.
      // For uint8, it uses fallback.
      final a = NDArray.fromList([1, 2, 3, 4], [4], DType.uint8);
      final view = a.slice([Slice(step: -1)]); // [4, 3, 2, 1]
      expect(view.toList(), [4, 3, 2, 1]);

      // cumprod along axis 0
      final res = cumprod(view, axis: 0);
      expect(res.toList(), [4, 12, 24, 24]);
    });

    test('sqrt() fallback on negative-stride view', () {
      // sqrt fallback for non-contiguous uses copy, which we optimized.
      final a = NDArray.fromList([4, 9, 16, 25], [4], DType.int32);
      final view = a.slice([Slice(step: -1)]); // [25, 16, 9, 4]
      expect(view.toList(), [25, 16, 9, 4]);

      final res = sqrt(view);
      expect(res.toList(), [5.0, 4.0, 3.0, 2.0]);
    });

    test('out buffer contiguity assertion', () {
      final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
      final b = NDArray.fromList([3.0, 4.0], [2], DType.float64);

      // Create a non-contiguous out buffer
      final parent = NDArray.create([4], DType.float64);
      final nonContigOut = parent.slice([
        Slice(start: 0, stop: 4, step: 2),
      ]); // shape [2], non-contiguous

      expect(() => matmul(a, b, out: nonContigOut), throwsArgumentError);
    });
  });
}
