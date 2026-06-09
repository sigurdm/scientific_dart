import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';
import 'dart:typed_data';

void main() {
  group('Out Parameter Tests', () {
    test(
      'det with out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0, 4.0]), [
          2,
          2,
        ], DType.float64);
        final outBuffer = NDArray<Float64>.zeros([], DType.float64);
        final d = det(a, out: outBuffer);
        expect(identical(d, outBuffer), true);
        expect(outBuffer.data[0], closeTo(-2.0, 1e-9));
      }),
    );

    test(
      'min with out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList([4.0, 2.0, 5.0, 1.0], [2, 2], DType.float64);
        final outBuffer = NDArray<Float64>.zeros([2], DType.float64);
        final res = min(a, axis: 1, out: outBuffer);
        expect(identical(res, outBuffer), true);
        expect(outBuffer.toList(), [2.0, 1.0]);
      }),
    );

    test(
      'max with out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList([4.0, 2.0, 5.0, 1.0], [2, 2], DType.float64);
        final outBuffer = NDArray<Float64>.zeros([2], DType.float64);
        final res = max(a, axis: 1, out: outBuffer);
        expect(identical(res, outBuffer), true);
        expect(outBuffer.toList(), [4.0, 5.0]);
      }),
    );

    test(
      'nanmin with out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [4.0, double.nan, 5.0, 1.0],
          [2, 2],
          DType.float64,
        );
        final outBuffer = NDArray<Float64>.zeros([2], DType.float64);
        final res = nanmin(a, axis: 1, out: outBuffer);
        expect(identical(res, outBuffer), true);
        expect(outBuffer.toList(), [4.0, 1.0]);
      }),
    );

    test(
      'nanmax with out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [4.0, double.nan, 5.0, 1.0],
          [2, 2],
          DType.float64,
        );
        final outBuffer = NDArray<Float64>.zeros([2], DType.float64);
        final res = nanmax(a, axis: 1, out: outBuffer);
        expect(identical(res, outBuffer), true);
        expect(outBuffer.toList(), [4.0, 5.0]);
      }),
    );

    test(
      'fft with out parameter (no transpose)',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
        final outBuffer = NDArray<Complex>.zeros([4], DType.complex128);
        final res = fft(a, out: outBuffer);
        expect(identical(res, outBuffer), true);
        // Expected FFT result: [10, -2+2i, -2, -2-2i]
        expect(outBuffer.data[0].real, closeTo(10.0, 1e-9));
        expect(outBuffer.data[0].imag, closeTo(0.0, 1e-9));
        expect(outBuffer.data[1].real, closeTo(-2.0, 1e-9));
        expect(outBuffer.data[1].imag, closeTo(2.0, 1e-9));
      }),
    );

    test(
      'fft with out parameter (with transpose)',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final outBuffer = NDArray<Complex>.zeros([2, 2], DType.complex128);
        // FFT along axis 0
        final res = fft(a, axis: 0, out: outBuffer);
        expect(identical(res, outBuffer), true);
        // a is:
        // [[1, 2],
        //  [3, 4]]
        // Along axis 0:
        // Col 0: [1, 3] -> FFT: [4, -2]
        // Col 1: [2, 4] -> FFT: [6, -2]
        // Result:
        // [[4, 6],
        //  [-2, -2]]
        expect(outBuffer.data[0].real, closeTo(4.0, 1e-9));
        expect(outBuffer.data[1].real, closeTo(6.0, 1e-9));
        expect(outBuffer.data[2].real, closeTo(-2.0, 1e-9));
        expect(outBuffer.data[3].real, closeTo(-2.0, 1e-9));
      }),
    );

    test(
      'ifft with out parameter',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [
            Complex(10.0, 0.0),
            Complex(-2.0, 2.0),
            Complex(-2.0, 0.0),
            Complex(-2.0, -2.0),
          ],
          [4],
          DType.complex128,
        );
        final outBuffer = NDArray<Complex>.zeros([4], DType.complex128);
        final res = ifft(a, out: outBuffer);
        expect(identical(res, outBuffer), true);
        expect(outBuffer.data[0].real, closeTo(1.0, 1e-9));
        expect(outBuffer.data[1].real, closeTo(2.0, 1e-9));
        expect(outBuffer.data[2].real, closeTo(3.0, 1e-9));
        expect(outBuffer.data[3].real, closeTo(4.0, 1e-9));
      }),
    );

    test(
      'gradientArray with out parameter',
      () => NDArray.scope(() {
        final f = NDArray.fromList([1.0, 2.0, 4.0, 8.0], [2, 2], DType.float64);
        final out1 = NDArray<Float64>.zeros([2, 2], DType.float64);
        final out2 = NDArray<Float64>.zeros([2, 2], DType.float64);
        final res = gradientArray(f, out: [out1, out2]);
        expect(identical(res[0], out1), true);
        expect(identical(res[1], out2), true);
        // f is:
        // [[1, 2],
        //  [4, 8]]
        // Grad axis 0 (rows):
        // Col 0: [1, 4] -> [3, 3]
        // Col 1: [2, 8] -> [6, 6]
        // Grad 0:
        // [[3, 6],
        //  [3, 6]]
        // Grad axis 1 (cols):
        // Row 0: [1, 2] -> [1, 1]
        // Row 1: [4, 8] -> [4, 4]
        // Grad 1:
        // [[1, 1],
        //  [4, 4]]
        expect(out1.toList(), [3.0, 6.0, 3.0, 6.0]);
        expect(out2.toList(), [1.0, 1.0, 4.0, 4.0]);
      }),
    );

    test(
      'svd with out parameter (m >= n)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );
        final outU = NDArray<Float64>.zeros([3, 3], DType.float64);
        final outS = NDArray<Float64>.zeros([2], DType.float64);
        final outVh = NDArray<Float64>.zeros([2, 2], DType.float64);

        final res = svd(a, out: (U: outU, S: outS, Vh: outVh));
        expect(identical(res.U, outU), true);
        expect(identical(res.S, outS), true);
        expect(identical(res.Vh, outVh), true);

        // SVD values should be populated
        expect(outS.data[0], greaterThan(0.0));
        expect(outS.data[1], greaterThan(0.0));
      }),
    );

    test(
      'svd with out parameter (m < n)',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );
        final outU = NDArray<Float64>.zeros([2, 2], DType.float64);
        final outS = NDArray<Float64>.zeros([2], DType.float64);
        final outVh = NDArray<Float64>.zeros([3, 3], DType.float64);

        final res = svd(a, out: (U: outU, S: outS, Vh: outVh));
        expect(identical(res.U, outU), true);
        expect(identical(res.S, outS), true);
        expect(identical(res.Vh, outVh), true);

        expect(outS.data[0], greaterThan(0.0));
        expect(outS.data[1], greaterThan(0.0));
      }),
    );
  });
}
