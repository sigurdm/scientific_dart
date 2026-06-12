import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';
import 'dart:typed_data';

void main() {
  group("Strided out and offset tests (Merged)", () {
    // --- From Worker 1 ---
    test("add with strided out and offsets", () {
      final baseA = NDArray.fromList(
        Float64List.fromList([0.0, 1.0, 2.0, 0.0]),
        [4],
        DType.float64,
      );
      final baseB = NDArray.fromList(
        Float64List.fromList([0.0, 3.0, 4.0, 0.0]),
        [4],
        DType.float64,
      );

      final a = baseA.slice([Slice(start: 1, stop: 3)]);
      final b = baseB.slice([Slice(start: 1, stop: 3)]);

      final baseOut = NDArray.zeros([4], DType.float64);
      final out = baseOut.slice([Slice(start: 0, stop: 4, step: 2)]);

      add(a, b, out: out);

      expect(baseOut.data[0], 4.0);
      expect(baseOut.data[1], 0.0);
      expect(baseOut.data[2], 6.0);
      expect(baseOut.data[3], 0.0);
    });

    test("abs with strided out and offsets (real)", () {
      final baseA = NDArray.fromList(
        Float64List.fromList([0.0, -1.5, -2.0, 0.0]),
        [4],
        DType.float64,
      );
      final a = baseA.slice([Slice(start: 1, stop: 3)]);

      final baseOut = NDArray.zeros([4], DType.float64);
      final out = baseOut.slice([Slice(start: 0, stop: 4, step: 2)]);

      abs(a, out: out);

      expect(baseOut.data[0], 1.5);
      expect(baseOut.data[1], 0.0);
      expect(baseOut.data[2], 2.0);
      expect(baseOut.data[3], 0.0);
    });

    test("abs with strided out (complex -> real)", () {
      final baseA = NDArray<Complex>.fromList(
        [Complex(0, 0), Complex(-3, 4), Complex(5, -12), Complex(0, 0)],
        [4],
        DType.complex128,
      );
      final a = baseA.slice([Slice(start: 1, stop: 3)]);

      final baseOut = NDArray.zeros([4], DType.float64);
      final out = baseOut.slice([Slice(start: 0, stop: 4, step: 2)]);

      abs(a, out: out);

      expect(baseOut.data[0], 5.0);
      expect(baseOut.data[1], 0.0);
      expect(baseOut.data[2], 13.0);
      expect(baseOut.data[3], 0.0);
    });

    test("conj with strided out (real)", () {
      final baseA = NDArray.fromList(
        Float64List.fromList([0.0, 1.0, 2.0, 0.0]),
        [4],
        DType.float64,
      );
      final a = baseA.slice([Slice(start: 1, stop: 3)]);

      final baseOut = NDArray.zeros([4], DType.float64);
      final out = baseOut.slice([Slice(start: 0, stop: 4, step: 2)]);

      conj(a, out: out);

      expect(baseOut.data[0], 1.0);
      expect(baseOut.data[1], 0.0);
      expect(baseOut.data[2], 2.0);
      expect(baseOut.data[3], 0.0);
    });

    test("conj with strided out (complex)", () {
      final baseA = NDArray<Complex>.fromList(
        [Complex(0, 0), Complex(1, -2), Complex(3, -4), Complex(0, 0)],
        [4],
        DType.complex128,
      );
      final a = baseA.slice([Slice(start: 1, stop: 3)]);

      final baseOut = NDArray<Complex>.fromList(
        [Complex(0, 0), Complex(0, 0), Complex(0, 0), Complex(0, 0)],
        [4],
        DType.complex128,
      );
      final out = baseOut.slice([Slice(start: 0, stop: 4, step: 2)]);

      conj(a, out: out);

      expect(baseOut.data[0], Complex(1, 2));
      expect(baseOut.data[1], Complex(0, 0));
      expect(baseOut.data[2], Complex(3, 4));
      expect(baseOut.data[3], Complex(0, 0));
    });

    test("hypot with strided out and offsets", () {
      final baseA = NDArray.fromList(
        Float64List.fromList([0.0, 3.0, 5.0, 0.0]),
        [4],
        DType.float64,
      );
      final baseB = NDArray.fromList(
        Float64List.fromList([0.0, 4.0, 12.0, 0.0]),
        [4],
        DType.float64,
      );
      final a = baseA.slice([Slice(start: 1, stop: 3)]);
      final b = baseB.slice([Slice(start: 1, stop: 3)]);

      final baseOut = NDArray.zeros([4], DType.float64);
      final out = baseOut.slice([Slice(start: 0, stop: 4, step: 2)]);

      hypot(a, b, out: out);

      expect(baseOut.data[0], 5.0);
      expect(baseOut.data[1], 0.0);
      expect(baseOut.data[2], 13.0);
      expect(baseOut.data[3], 0.0);
    });

    // --- From Worker 2 ---
    test("Unary op (sin) with strided out and non-zero offset input", () {
      final aBacking = Int32List.fromList([99, 0, 99, 1]);
      final aParent = NDArray.fromList(aBacking, [4], DType.int32);
      final a = aParent.slice([
        Slice(start: 1, stop: 4, step: 2),
      ]); // shape [2], strides [2], offset 1

      final outBacking = Float64List.fromList([99.0, 99.0, 99.0, 99.0, 99.0]);
      final outParent = NDArray.fromList(outBacking, [5], DType.float64);
      final out = outParent.slice([
        Slice(start: 1, stop: 5, step: 2),
      ]); // shape [2], strides [2], offset 1

      sin(a, out: out);

      expect(outParent.data[0], 99.0);
      expect(outParent.data[1], closeTo(0.0, 1e-7));
      expect(outParent.data[2], 99.0);
      expect(outParent.data[3], closeTo(0.84147098, 1e-7));
      expect(outParent.data[4], 99.0);
    });

    test("Binary op (add) with strided out and non-zero offset inputs", () {
      final aBacking = Float64List.fromList([99.0, 1.0, 99.0, 2.0]);
      final a = NDArray.fromList(
        aBacking,
        [4],
        DType.float64,
      ).slice([Slice(start: 1, stop: 4, step: 2)]); // [1.0, 2.0]

      final bBacking = Float64List.fromList([99.0, 10.0, 99.0, 20.0]);
      final b = NDArray.fromList(
        bBacking,
        [4],
        DType.float64,
      ).slice([Slice(start: 1, stop: 4, step: 2)]); // [10.0, 20.0]

      final outBacking = Float64List.fromList([99.0, 99.0, 99.0, 99.0, 99.0]);
      final outParent = NDArray.fromList(outBacking, [5], DType.float64);
      final out = outParent.slice([Slice(start: 1, stop: 5, step: 2)]);

      add(a, b, out: out);

      expect(outParent.data[0], 99.0);
      expect(outParent.data[1], 11.0);
      expect(outParent.data[2], 99.0);
      expect(outParent.data[3], 22.0);
      expect(outParent.data[4], 99.0);
    });

    test("Real-number conj with strided out (Worker 2 version)", () {
      final aBacking = Float64List.fromList([1.0, 2.0]);
      final a = NDArray.fromList(aBacking, [2], DType.float64);

      final outBacking = Float64List.fromList([99.0, 99.0, 99.0, 99.0, 99.0]);
      final outParent = NDArray.fromList(outBacking, [5], DType.float64);
      final out = outParent.slice([Slice(start: 1, stop: 5, step: 2)]);

      conj(a, out: out);

      expect(outParent.data[0], 99.0);
      expect(outParent.data[1], 1.0);
      expect(outParent.data[2], 99.0);
      expect(outParent.data[3], 2.0);
      expect(outParent.data[4], 99.0);
    });

    test("Optimized abs with contiguous input/output", () {
      final a = NDArray.fromList(Float64List.fromList([-1.0, -2.0]), [
        2,
      ], DType.float64);
      final out = NDArray<double>.create([2], DType.float64);
      abs(a, out: out);
      expect(out.data[0], 1.0);
      expect(out.data[1], 2.0);
    });

    test(
      "Optimized abs with contiguous sliced view (length != data.length)",
      () {
        final base = NDArray.fromList(
          Float64List.fromList([-9.0, -1.0, -2.0, -9.0]),
          [4],
          DType.float64,
        );
        final a = base.slice([
          Slice(start: 1, stop: 3),
        ]); // shape [2], contiguous, offset 1, data.length is 4
        final outBase = NDArray.zeros([4], DType.float64);
        final out = outBase.slice([
          Slice(start: 1, stop: 3),
        ]); // shape [2], contiguous

        abs(a, out: out);

        expect(outBase.data[0], 0.0);
        expect(outBase.data[1], 1.0);
        expect(outBase.data[2], 2.0);
        expect(outBase.data[3], 0.0);
      },
    );
  });
}
