import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Vectorized Bitwise Operations Tests', () {
    test(
      'bitwise_and, bitwise_or, bitwise_xor basic behavior (int32 and int64)',
      () {
        NDArray.scope(() {
          final a32 = NDArray.fromList([5, 12, 3], [3], DType.int32);
          final b32 = NDArray.fromList([3, 4, 5], [3], DType.int32);

          final a64 = NDArray.fromList([5, 12, 3], [3], DType.int64);
          final b64 = NDArray.fromList([3, 4, 5], [3], DType.int64);

          // bitwise_and: 5 & 3 = 1, 12 & 4 = 4, 3 & 5 = 1
          expect(bitwise_and(a32, b32).toList(), [1, 4, 1]);
          expect(bitwise_and(a64, b64).toList(), [1, 4, 1]);

          // bitwise_or: 5 | 3 = 7, 12 | 4 = 12, 3 | 5 = 7
          expect(bitwise_or(a32, b32).toList(), [7, 12, 7]);
          expect(bitwise_or(a64, b64).toList(), [7, 12, 7]);

          // bitwise_xor: 5 ^ 3 = 6, 12 ^ 4 = 8, 3 ^ 5 = 6
          expect(bitwise_xor(a32, b32).toList(), [6, 8, 6]);
          expect(bitwise_xor(a64, b64).toList(), [6, 8, 6]);
        });
      },
    );

    test('bitwise operations with uint8 and int16 dtypes', () {
      NDArray.scope(() {
        final a8 = NDArray.fromList([15, 240], [2], DType.uint8);
        final b8 = NDArray.fromList([240, 15], [2], DType.uint8);

        final a16 = NDArray.fromList([15, 240], [2], DType.int16);
        final b16 = NDArray.fromList([240, 15], [2], DType.int16);

        expect(bitwise_and(a8, b8).toList(), [0, 0]);
        expect(bitwise_and(a16, b16).toList(), [0, 0]);

        expect(bitwise_or(a8, b8).toList(), [255, 255]);
        expect(bitwise_or(a16, b16).toList(), [255, 255]);

        expect(bitwise_xor(a8, b8).toList(), [255, 255]);
        expect(bitwise_xor(a16, b16).toList(), [255, 255]);
      });
    });

    test('left_shift and right_shift basic and edge case safe shifting', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1, 2, 4, 8], [4], DType.int32);
        final shift = NDArray.fromList([1, 2, 3, 4], [4], DType.int32);

        // left_shift: 1<<1=2, 2<<2=8, 4<<3=32, 8<<4=128
        expect(left_shift(a, shift).toList(), [2, 8, 32, 128]);

        // right_shift: 2>>1=1, 8>>2=2, 32>>3=4, 128>>4=8
        final shifted = left_shift(a, shift);
        expect(right_shift(shifted, shift).toList(), [1, 2, 4, 8]);

        // Safe shifting behavior (prevents undefined C behavior)
        final badShift = NDArray.fromList([-1, 32, 100, 0], [4], DType.int32);
        expect(left_shift(a, badShift).toList(), [0, 0, 0, 8]);
        expect(right_shift(a, badShift).toList(), [0, 0, 0, 8]);
      });
    });

    test('invert (bitwise NOT) for all four integer dtypes', () {
      NDArray.scope(() {
        final a32 = NDArray.fromList([0, -1, 5], [3], DType.int32);
        final a64 = NDArray.fromList([0, -1, 5], [3], DType.int64);
        final a8 = NDArray.fromList([0, 255, 5], [3], DType.uint8);
        final a16 = NDArray.fromList([0, -1, 5], [3], DType.int16);

        expect(invert(a32).toList(), [-1, 0, -6]);
        expect(invert(a64).toList(), [-1, 0, -6]);
        expect(invert(a8).toList(), [255, 0, 250]);
        expect(invert(a16).toList(), [-1, 0, -6]);
      });
    });

    test('broadcasting with bitwise operations (contiguous & strided)', () {
      NDArray.scope(() {
        // Broadcast scalar-like array [2] (shape [1]) against [3, 4, 5] (shape [3])
        final a = NDArray.fromList([3, 4, 5], [3], DType.int32);
        final b = NDArray.fromList([2], [1], DType.int32);

        expect(bitwise_and(a, b).toList(), [2, 0, 0]); // 3&2=2, 4&2=0, 5&2=0
        expect(bitwise_or(a, b).toList(), [3, 6, 7]); // 3|2=3, 4|2=6, 5|2=7

        // Strided sliced broadcasting test
        final stridedA = a.slice([Slice(start: 0, stop: 3, step: 2)]); // [3, 5]
        expect(bitwise_or(stridedA, b).toList(), [3, 7]);
      });
    });

    test('mixed integer type upcasting/promotion', () {
      NDArray.scope(() {
        final a32 = NDArray.fromList([5, 12], [2], DType.int32);
        final b64 = NDArray.fromList([3, 4], [2], DType.int64);

        // bitwise_and should resolve to DType.int64
        final res = bitwise_and(a32, b64);
        expect(res.dtype, DType.int64);
        expect(res.toList(), [1, 4]);
      });
    });

    test('named recycler out parameter buffer reuse', () {
      NDArray.scope(() {
        final a = NDArray.fromList([5, 12], [2], DType.int32);
        final b = NDArray.fromList([3, 4], [2], DType.int32);
        final out = NDArray<int>.create([2], DType.int32);

        final res = bitwise_and(a, b, out: out);
        expect(identical(res, out), true);
        expect(out.toList(), [1, 4]);
      });
    });

    test('non-integer input throws ArgumentError', () {
      NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final b = NDArray.fromList([3, 4], [2], DType.int32);

        expect(() => bitwise_and(a, b), throwsArgumentError);
        expect(() => invert(a), throwsArgumentError);
      });
    });

    test('recycler out buffer shape incompatibility throws ArgumentError', () {
      NDArray.scope(() {
        final a = NDArray.fromList([5, 12], [2], DType.int32);
        final b = NDArray.fromList([3, 4], [2], DType.int32);
        final wrongOut = NDArray<int>.create([3], DType.int32);

        expect(() => bitwise_and(a, b, out: wrongOut), throwsArgumentError);
      });
    });
  });
}
