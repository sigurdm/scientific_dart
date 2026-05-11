import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';
import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as math;

void main() {
  group('Quality Enhancements & Safety Gates Tests', () {
    test(
      'arange() zero and negative step validation',
      () => NDArray.scope(() {
        expect(() => NDArray.arange(0.0, 5.0, step: 0.0), throwsArgumentError);
        expect(() => NDArray.arange(0.0, 5.0, step: -1.0), throwsArgumentError);
        expect(() => NDArray.arange(5.0, 0.0, step: 1.0), throwsArgumentError);

        // Valid arange with negative step
        final a = NDArray<double>.arange(
          5.0,
          0.0,
          step: -1.0,
          dtype: DType.float64,
        );
        expect(a.toList(), [5.0, 4.0, 3.0, 2.0, 1.0]);
      }),
    );

    test(
      'reshape() non-contiguous view copies memory first',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
          [2, 2],
          DType.float64,
        );

        // Transposed view is non-contiguous (strides: [1, 2])
        final transposed = parent.transposed;
        expect(transposed.isContiguous, false);
        expect(transposed.toList(), [1.0, 3.0, 2.0, 4.0]);

        // Reshaping the transposed view: it should automatically copy first
        final reshaped = transposed.reshape([4]);
        expect(reshaped.isContiguous, true);
        expect(reshaped.toList(), [1.0, 3.0, 2.0, 4.0]);

        // Verify memory decoupling (modifying reshaped does NOT affect parent)
        reshaped.data[0] = Float64(99.0);
        expect(parent.data[0], 1.0);
      }),
    );

    test(
      'count_nonzero() fallback path on non-contiguous views',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          Float64List.fromList([1.0, 0.0, 3.0, 0.0, 5.0, 6.0]),
          [2, 3],
          DType.float64,
        );

        // Non-contiguous column slice: select column 0 (elements [1.0, 0.0] -> wait, strides are [3, 1], column 0 is data[0], data[3])
        // Logical: row 0: [1.0, 0.0, 3.0], row 1: [0.0, 5.0, 6.0]. Column 0 is: [1.0, 0.0].
        // Column 2 is: [3.0, 6.0] (nonzero count is 2).
        final view = a.slice([Slice.all(), Index(2)]);
        expect(view.isContiguous, false);
        expect(view.toList(), [3.0, 6.0]);

        // In the old code, count_nonzero(view) would incorrectly return 0 due to discarded recursion yields!
        expect(count_nonzero(view), 2);
      }),
    );

    test(
      'NDArray Structural Value Equality operator == and hashCode',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final b = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final c = NDArray.fromList([1.0, 2.0, 9.0], [3], DType.float64);
        final d = NDArray.fromList([1.0, 2.0, 3.0], [1, 3], DType.float64);

        expect(a == b, true);
        expect(a == c, false);
        expect(a == d, false); // different shape
        expect(a.hashCode == b.hashCode, true);

        // Non-contiguous view comparisons tests (triggering recursive walkers)
        final parent1 = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final parent2 = NDArray.fromList(
          [1.0, 3.0, 2.0, 4.0],
          [2, 2],
          DType.float64,
        );

        final viewT1 =
            parent1.transposed; // non-contiguous: [[1.0, 3.0], [2.0, 4.0]]
        expect(viewT1.isContiguous, false);

        // 1. Non-contiguous == contiguous
        expect(viewT1 == parent2, true);
        expect(viewT1 == parent1, false);

        // 2. Non-contiguous hashCode
        expect(viewT1.hashCode == parent2.hashCode, true);
      }),
    );

    test(
      'View Invalidation Safety (throws StateError on disposed views)',
      () => NDArray.scope(() {
        final parent = NDArray.ones([2, 2], DType.float64);
        final view = parent.slice([Index(0)]); // Select first row

        expect(parent.isDisposed, false);
        expect(view.isDisposed, false);

        parent.dispose();

        expect(parent.isDisposed, true);
        expect(view.isDisposed, true);

        // Accessing public APIs should throw StateError
        expect(() => view.toList(), throwsStateError);
        expect(() => view.pointer, throwsStateError);
        expect(() => view[0], throwsStateError);
        expect(() => view.reshape([2]), throwsStateError);
      }),
    );

    test(
      'NPY double quotes / mixed quotes header parsing compatibility',
      () => NDArray.scope(() {
        // Construct a fake .npy buffer using double quotes in header dictionary
        final headerStr =
            '{"descr": "<f8", "fortran_order": False, "shape": (2, 2)}';

        final prefixLen = 6 + 2 + 2;
        var paddedHeaderLen =
            ((prefixLen + headerStr.length + 1) + 63) ~/ 64 * 64 - prefixLen;
        final padCount = paddedHeaderLen - headerStr.length - 1;
        final paddedHeader = "$headerStr${' ' * padCount}\n";

        final headerBytes = Uint8List.fromList(paddedHeader.codeUnits);
        final lenBytes = Uint8List(2);
        ByteData.view(
          lenBytes.buffer,
        ).setUint16(0, headerBytes.length, Endian.little);

        final rawData = Float64List.fromList([1.0, 2.0, 3.0, 4.0]);
        final rawDataBytes = Uint8List.view(rawData.buffer);

        final fullBuffer = Uint8List(
          6 + 2 + 2 + headerBytes.length + rawDataBytes.length,
        );
        var offset = 0;

        fullBuffer.setRange(offset, offset + 6, const [
          0x93,
          0x4e,
          0x55,
          0x4d,
          0x50,
          0x59,
        ]);
        offset += 6;
        fullBuffer.setRange(offset, offset + 2, const [0x01, 0x00]);
        offset += 2;
        fullBuffer.setRange(offset, offset + 2, lenBytes);
        offset += 2;
        fullBuffer.setRange(offset, offset + headerBytes.length, headerBytes);
        offset += headerBytes.length;
        fullBuffer.setRange(offset, offset + rawDataBytes.length, rawDataBytes);

        const path = 'scratch/double_quotes_simulated.npy';
        File(path).writeAsBytesSync(fullBuffer, flush: true);

        // Load should parse successfully now
        final loaded = load(path);
        expect(loaded.shape, [2, 2]);
        expect(loaded.dtype, DType.float64);
        expect(loaded.toList(), [1.0, 2.0, 3.0, 4.0]);
      }),
    );

    test(
      'randint() supporting wide ranges (> 2^32)',
      () => NDArray.scope(() {
        // Generate a 64-bit integer range that exceeds 2^32
        final a = randint(
          [10],
          low: 100000000000,
          high: 200000000000,
          dtype: DType.int64,
        );
        expect(a.shape, [10]);
        expect(a.dtype, DType.int64);
        for (final val in a.data) {
          expect(val, greaterThanOrEqualTo(100000000000));
          expect(val, lessThan(200000000000));
        }
      }),
    );

    test(
      'transpose() negative axis support',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
          [2, 3],
          DType.float64,
        );
        // Reverse axes order via negative indices: [-1, -2] is equivalent to [1, 0]
        final b = a.transpose([-1, -2]);
        expect(b.shape, [3, 2]);
        expect(b.toList(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
      }),
    );

    test(
      'exponential() supporting lam (rate) parameter alias',
      () => NDArray.scope(() {
        final a = exponential([10], lam: 2.0); // lam = 2.0 -> scale = 0.5
        expect(a.shape, [10]);
        expect(a.dtype, DType.float64);
        for (final val in a.data) {
          expect(val, greaterThanOrEqualTo(0.0));
        }

        expect(() => exponential([10], lam: -1.0), throwsArgumentError);
      }),
    );

    test(
      'DType.complex64 array creation, viewing and operations coverage',
      () => NDArray.scope(() {
        final a = NDArray<Complex>.create([2, 2], DType.complex64);
        expect(a.shape, [2, 2]);
        expect(a.dtype, DType.complex64);

        a.data[0] = Complex(1.0, 2.0);
        a.data[1] = Complex(3.0, 4.0);
        a.data[2] = Complex(5.0, 6.0);
        a.data[3] = Complex(7.0, 8.0);

        expect(a.data[0], Complex(1.0, 2.0));
        expect(a.data[3], Complex(7.0, 8.0));

        // View creation coverage (parent.dtype == DType.complex64)
        final view = a.slice([Index(0)]); // Select first row
        expect(view.shape, [2]);
        expect(view.dtype, DType.complex64);
        expect(view.data[0], Complex(1.0, 2.0));
        expect(view.data[1], Complex(3.0, 4.0));

        // Operation coverage (add complex64 arrays)
        final b = NDArray<Complex>.create([2, 2], DType.complex64);
        b.data.fillRange(0, 4, Complex(10.0, 10.0));

        final c = add(a, b);
        expect(c.dtype, DType.complex64);
        expect(c.data[0], Complex(11.0, 12.0));
        expect(c.data[3], Complex(17.0, 18.0));
      }),
    );

    test(
      'add() contiguous float32 SIMD fast paths and remainders coverage',
      () {
        final a8 = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [8],
          DType.float32,
        );
        final b8 = NDArray.fromList(
          [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
          [8],
          DType.float32,
        );

        final res8 = add(a8, b8);
        expect(res8.dtype, DType.float32);
        expect(res8.toList(), [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]);

        final a6 = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [6],
          DType.float32,
        );
        final b6 = NDArray.fromList(
          [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
          [6],
          DType.float32,
        );

        final res6 = add(a6, b6);
        expect(res6.dtype, DType.float32);
        expect(res6.toList(), [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
      },
    );

    test(
      'subtract() contiguous float64 and float32 FFI fast paths coverage',
      () {
        final a64 = NDArray.fromList([20.0, 30.0], [2], DType.float64);
        final b64 = NDArray.fromList([5.0, 10.0], [2], DType.float64);

        final res64 = subtract(a64, b64);
        expect(res64.dtype, DType.float64);
        expect(res64.toList(), [15.0, 20.0]);

        final a32 = NDArray.fromList([20.0, 30.0], [2], DType.float32);
        final b32 = NDArray.fromList([5.0, 10.0], [2], DType.float32);

        final res32 = subtract(a32, b32);
        expect(res32.dtype, DType.float32);
        expect(res32.toList(), [15.0, 20.0]);
      },
    );

    test(
      'linspace() with num == 1 coverage',
      () => NDArray.scope(() {
        final a = NDArray<double>.linspace(5.0, 10.0, 1, dtype: DType.float64);
        expect(a.shape, [1]);
        expect(a.toList(), [5.0]);
      }),
    );

    test(
      'reshape() size mismatch throws ArgumentError coverage',
      () => NDArray.scope(() {
        final a = NDArray.ones([2, 2], DType.float64);
        expect(() => a.reshape([3]), throwsArgumentError);
      }),
    );

    test(
      'transpose() invalid axes length throws ArgumentError coverage',
      () => NDArray.scope(() {
        final a = NDArray.ones([2, 2], DType.float64);
        expect(() => a.transpose([0]), throwsArgumentError);
      }),
    );

    test(
      'Complex array reductions (sum, prod, mean) and stacking coverage',
      () {
        final a = NDArray<Complex>.fromList(
          [
            Complex(1.0, 1.0),
            Complex(2.0, 0.0),
            Complex(3.0, 0.0),
            Complex(4.0, 0.0),
          ],
          [2, 2],
          DType.complex128,
        );

        // Test sum()
        final totalSum = sum(a);
        expect(totalSum.scalar, Complex(10.0, 1.0));

        // Test mean()
        final totalMean = mean(a);
        expect(totalMean.scalar, Complex(2.5, 0.25));

        // Test prod()
        final totalProd = prod(a);
        expect(totalProd.scalar, Complex(24.0, 24.0)); // (1+i)*2*3*4 = 24 + 24i

        // Test concatenate() and hstack()
        final b = NDArray<Complex>.fromList(
          [Complex(10.0, 0.0), Complex(10.0, 0.0)],
          [1, 2],
          DType.complex128,
        );

        final row0 = a.slice([Index(0)]).reshape([1, 2]);
        final combined = concatenate([row0, b], axis: 0);
        expect(combined.shape, [2, 2]);
        expect(combined.dtype, DType.complex128);
        expect(combined.data[0], Complex(1.0, 1.0));
        expect(combined.data[2], Complex(10.0, 0.0));
      },
    );

    test(
      'qsort floating-point NaN sorting safety and stability',
      () => NDArray.scope(() {
        final a = NDArray.fromList(
          [3.0, double.nan, 1.0, double.nan, 2.0],
          [5],
          DType.float64,
        );

        final b = sort(a);

        // The 3 non-NaN elements must be sorted ascending: 1.0, 2.0, 3.0
        expect(b.data[0], 1.0);
        expect(b.data[1], 2.0);
        expect(b.data[2], 3.0);

        // The NaNs must be pushed consistently to the end of the array
        expect(b.data[3].isNaN, true);
        expect(b.data[4].isNaN, true);
      }),
    );

    test(
      'matmul() high-speed 1D vector dot product gate',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final b = NDArray.fromList([4.0, 5.0, 6.0], [3], DType.float64);

        final c = matmul(a, b);

        expect(c.shape, []); // Must be a 0D scalar array
        expect(c.dtype, DType.float64);
        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        expect(c.data[0], 32.0);
      }),
    );

    test(
      'fft() and ifft() empty/0D shape validation checks',
      () => NDArray.scope(() {
        final emptyArr = NDArray.zeros([], DType.float64);

        expect(() => fft(emptyArr), throwsArgumentError);
        expect(() => ifft(emptyArr), throwsArgumentError);
      }),
    );

    test(
      'fft() and ifft() invalid transform length n <= 0 checks',
      () => NDArray.scope(() {
        final signal = NDArray.zeros([5], DType.float64);

        expect(() => fft(signal, n: 0), throwsArgumentError);
        expect(() => fft(signal, n: -5), throwsArgumentError);
        expect(() => ifft(signal, n: 0), throwsArgumentError);
        expect(() => ifft(signal, n: -1), throwsArgumentError);
      }),
    );

    test(
      'fft() and ifft() processing complex inputs directly',
      () => NDArray.scope(() {
        final complexSignal = NDArray<Complex>.fromList(
          [Complex(1.0, 1.0), Complex(2.0, 2.0)],
          [2],
          DType.complex128,
        );

        final freq = fft(complexSignal);
        expect(freq.dtype, DType.complex128);
        expect(freq.shape, [2]);

        final time = ifft(freq);
        expect(time.dtype, DType.complex128);
        expect(time.shape, [2]);

        // The inverse transform must recover the original complex signal values (within floating precision)
        expect(time.data[0].real, closeTo(1.0, 1e-9));
        expect(time.data[0].imag, closeTo(1.0, 1e-9));
        expect(time.data[1].real, closeTo(2.0, 1e-9));
        expect(time.data[1].imag, closeTo(2.0, 1e-9));
      }),
    );

    test(
      'fft() and ifft() with real inputs and zero padding padding checks',
      () {
        final realSignal = NDArray<double>.fromList(
          [1.0, 2.0],
          [2],
          DType.float64,
        );

        // 1. fft with zero-padding (n = 4)
        final freqPadded = fft(realSignal, n: 4);
        expect(freqPadded.shape, [4]);

        // 2. ifft with real input and zero padding (n = 4)
        final timePadded = ifft(realSignal, n: 4);
        expect(timePadded.shape, [4]);
      },
    );

    test(
      'ifft() with non-contiguous strided view input',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          [
            Complex(1.0, 1.0),
            Complex(2.0, 2.0),
            Complex(3.0, 3.0),
            Complex(4.0, 4.0),
          ],
          [2, 2],
          DType.complex128,
        );

        // Transposed view is non-contiguous
        final transposed = parent.transposed;
        expect(transposed.isContiguous, false);

        // Should automatically copy transposed inside ifft
        final result = ifft(transposed);

        expect(result.shape, [2, 2]);
        expect(result.dtype, DType.complex128);
        expect(result.data[0].real, closeTo(2.0, 1e-9));
        expect(result.data[0].imag, closeTo(2.0, 1e-9));
        expect(result.data[1].real, closeTo(-1.0, 1e-9));
        expect(result.data[1].imag, closeTo(-1.0, 1e-9));
        expect(result.data[2].real, closeTo(3.0, 1e-9));
        expect(result.data[2].imag, closeTo(3.0, 1e-9));
        expect(result.data[3].real, closeTo(-1.0, 1e-9));
        expect(result.data[3].imag, closeTo(-1.0, 1e-9));
      }),
    );

    test(
      'uniform() validation checks coverage',
      () => NDArray.scope(() {
        expect(
          () => uniform([5], dtype: DType.int64 as dynamic),
          throwsArgumentError,
        );
      }),
    );

    test(
      'randint() validation checks coverage',
      () => NDArray.scope(() {
        expect(
          () => randint([5], low: 0, high: 10, dtype: DType.float64 as dynamic),
          throwsArgumentError,
        );
        expect(() => randint([5], low: 10, high: 5), throwsArgumentError);
      }),
    );

    test(
      'normal() validation and zero-avoidance checks coverage',
      () => NDArray.scope(() {
        expect(() => normal([5], scale: 0.0), throwsArgumentError);

        final a = normal([2], random: ZeroThenDoubleRandom());
        expect(a.shape, [2]);
        expect(a.dtype, DType.float64);
      }),
    );

    test(
      'exponential() validation checks coverage',
      () => NDArray.scope(() {
        expect(
          () => exponential([5], dtype: DType.int32 as dynamic),
          throwsArgumentError,
        );
      }),
    );

    test(
      'poisson() validation, large lambda, and zero-avoidance checks',
      () => NDArray.scope(() {
        expect(
          () => poisson([5], dtype: DType.float64 as dynamic),
          throwsArgumentError,
        );
        expect(() => poisson([5], lam: 0.0), throwsArgumentError);

        final a = poisson([2], lam: 40.0, random: ZeroThenDoubleRandom());
        expect(a.shape, [2]);
        expect(a.dtype, DType.int64);
      }),
    );

    test(
      'binomial() validation, zero trials, zero stddev, and zero-avoidance checks',
      () {
        expect(
          () => binomial([5], n: 10, p: 0.5, dtype: DType.float64 as dynamic),
          throwsArgumentError,
        );
        expect(() => binomial([5], n: -1, p: 0.5), throwsArgumentError);
        expect(() => binomial([5], n: 10, p: 1.5), throwsArgumentError);

        final zeroTrials = binomial([5], n: 0, p: 0.5);
        expect(zeroTrials.toList(), [0, 0, 0, 0, 0]);

        final zeroStddev = binomial([5], n: 100, p: 1.0);
        expect(zeroStddev.toList(), [100, 100, 100, 100, 100]);

        final a = binomial([2], n: 100, p: 0.5, random: ZeroThenDoubleRandom());
        expect(a.shape, [2]);
        expect(a.dtype, DType.int64);

        final oddLength = binomial([5], n: 100, p: 0.5);
        expect(oddLength.shape, [5]);
        expect(oddLength.dtype, DType.int64);
      },
    );
    test(
      'save() and load() non-contiguous strided views of all dtypes',
      () => NDArray.scope(() {
        // 1. float32
        final f32 = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float32,
        );
        final f32View = f32.transposed; // non-contiguous!
        save('scratch/f32_view.npy', f32View);
        final f32Loaded = load('scratch/f32_view.npy');
        expect(f32Loaded.shape, [2, 2]);
        expect(f32Loaded.dtype, DType.float32);
        expect(f32Loaded.toList(), [1.0, 3.0, 2.0, 4.0]);

        // 2. int32
        final i32 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        final i32View = i32.transposed;
        save('scratch/i32_view.npy', i32View);
        final i32Loaded = load('scratch/i32_view.npy');
        expect(i32Loaded.toList(), [1, 3, 2, 4]);

        // 3. int64
        final i64 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int64);
        final i64View = i64.transposed;
        save('scratch/i64_view.npy', i64View);
        final i64Loaded = load('scratch/i64_view.npy');
        expect(i64Loaded.toList(), [1, 3, 2, 4]);

        // 4. boolean
        final b = NDArray.fromList(
          [true, false, true, false],
          [2, 2],
          DType.boolean,
        );
        final bView = b.transposed;
        save('scratch/b_view.npy', bView);
        final bLoaded = load('scratch/b_view.npy');
        expect(bLoaded.toList(), [true, true, false, false]);

        // 5. complex128
        final c128 = NDArray<Complex>.fromList(
          [
            Complex(1.0, 1.0),
            Complex(2.0, 2.0),
            Complex(3.0, 3.0),
            Complex(4.0, 4.0),
          ],
          [2, 2],
          DType.complex128,
        );
        final c128View = c128.transposed;
        save('scratch/c128_view.npy', c128View);
        final c128Loaded = load('scratch/c128_view.npy');
        expect(c128Loaded.toList(), [
          Complex(1.0, 1.0),
          Complex(3.0, 3.0),
          Complex(2.0, 2.0),
          Complex(4.0, 4.0),
        ]);

        // 6. complex64
        final c64 = NDArray<Complex>.fromList(
          [
            Complex(1.0, 1.0),
            Complex(2.0, 2.0),
            Complex(3.0, 3.0),
            Complex(4.0, 4.0),
          ],
          [2, 2],
          DType.complex64,
        );
        final c64View = c64.transposed;
        save('scratch/c64_view.npy', c64View);
        final c64Loaded = load('scratch/c64_view.npy');
        expect(c64Loaded.toList(), [
          Complex(1.0, 1.0),
          Complex(3.0, 3.0),
          Complex(2.0, 2.0),
          Complex(4.0, 4.0),
        ]);
      }),
    );

    test(
      'save() and load() Uint8 and Int16 arrays coverage',
      () => NDArray.scope(() {
        // 1. Uint8
        final u8 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.uint8);
        save('scratch/u8_array.npy', u8);
        final u8Loaded = load('scratch/u8_array.npy');
        expect(u8Loaded.toList(), [1, 2, 3, 4]);
        expect(u8Loaded.dtype, DType.uint8);

        // 2. Int16
        final i16 = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.int16);
        save('scratch/i16_array.npy', i16);
        final i16Loaded = load('scratch/i16_array.npy');
        expect(i16Loaded.toList(), [10, 20, 30, 40]);
        expect(i16Loaded.dtype, DType.int16);
      }),
    );

    test(
      'broadcastShapes() incompatible shapes throws ArgumentError',
      () => NDArray.scope(() {
        final a = NDArray.ones([2], DType.float64);
        final b = NDArray.ones([3], DType.float64);

        expect(() => add(a, b), throwsArgumentError);
      }),
    );

    test(
      'DType properties getters isComplex, isFloating, isInteger coverage',
      () {
        expect(DType.complex128.isComplex, true);
        expect(DType.float64.isComplex, false);
        expect(DType.float64.isFloating, true);
        expect(DType.int64.isFloating, false);
        expect(DType.int64.isInteger, true);
        expect(DType.float64.isInteger, false);
      },
    );

    test(
      'NDArray.eye() factory with integer dtype coverage',
      () => NDArray.scope(() {
        final a = NDArray.eye(2, DType.int32);
        expect(a.toList(), [1, 0, 0, 1]);
        expect(a.dtype, DType.int32);
      }),
    );

    test(
      'NDArray.view() FFI constructors with float32 and int64 coverage',
      () => NDArray.scope(() {
        final pF32 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float32);
        final vF32 = NDArray.view(
          pF32,
          shape: [2],
          strides: [1],
          offsetElements: 1,
        );
        expect(vF32.toList(), [2.0, 3.0]);
        expect(vF32.dtype, DType.float32);

        final pI64 = NDArray.fromList([10, 20, 30], [3], DType.int64);
        final vI64 = NDArray.view(
          pI64,
          shape: [2],
          strides: [1],
          offsetElements: 1,
        );
        expect(vI64.toList(), [20, 30]);
        expect(vI64.dtype, DType.int64);
      }),
    );

    test(
      '_resolveDType() complex64 and float64 cross-promotion coverage',
      () => NDArray.scope(() {
        final a = NDArray.fromList([Complex(1.0, 1.0)], [1], DType.complex64);
        final b = NDArray.fromList([2.0], [1], DType.float64);

        final c = add(a, b);
        expect(c.dtype, DType.complex128);
        expect(c.data[0].real, 3.0);
        expect(c.data[0].imag, 1.0);
      }),
    );

    test(
      'matmul() copy-free 100% transposed and sliced views multi-dimensional multiplication',
      () {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList([5.0, 6.0, 7.0, 8.0], [2, 2], DType.float64);

        final aT = a.transposed;
        final bT = b.transposed;

        final result = matmul(aT, bT);

        expect(result.shape, [2, 2]);
        expect(result.toList(), [23.0, 31.0, 34.0, 46.0]);
      },
    );

    test(
      'ufuncs in-place out buffer shape and dtype validation checks',
      () => NDArray.scope(() {
        final a = NDArray<double>.ones([3], DType.float64);
        final b = NDArray<double>.ones([3], DType.float64);
        final incompatibleOut = NDArray<double>.ones([4], DType.float64);
        final incompatibleDTypeOut = NDArray<int>.ones([3], DType.int32);

        // 1. add() contiguous shape mismatch
        expect(() => add(a, b, out: incompatibleOut), throwsArgumentError);
        expect(() => add(a, b, out: incompatibleDTypeOut), throwsArgumentError);

        // 2. add() broadcast shape mismatch
        final broadcastA = NDArray.ones([1, 3], DType.float64);
        expect(
          () => add(broadcastA, b, out: incompatibleOut),
          throwsArgumentError,
        );

        // 3. sqrt() shape mismatch
        expect(() => sqrt(a, out: incompatibleOut), throwsArgumentError);

        // 4. sin() shape mismatch
        expect(() => sin(a, out: incompatibleOut), throwsArgumentError);
      }),
    );

    test(
      'NDArray.ones() factory with complex dtypes coverage',
      () => NDArray.scope(() {
        final a = NDArray.ones([2], DType.complex128);
        expect(a.toList(), [Complex(1.0, 0.0), Complex(1.0, 0.0)]);
        expect(a.dtype, DType.complex128);

        final b = NDArray.ones([2], DType.complex64);
        expect(b.toList(), [Complex(1.0, 0.0), Complex(1.0, 0.0)]);
        expect(b.dtype, DType.complex64);
      }),
    );

    test(
      'add() cross-type complex/int and int/complex additions coverage',
      () => NDArray.scope(() {
        final c = NDArray<Complex>.fromList(
          [Complex(1.0, 1.0)],
          [1],
          DType.complex128,
        );
        final i = NDArray<int>.fromList([2], [1], DType.int64);
        final d = NDArray<double>.fromList([3.0], [1], DType.float64);

        // 1. Complex + int
        final res1 = add(c, i);
        expect(res1.dtype, DType.complex128);
        expect(res1.data[0].real, 3.0);
        expect(res1.data[0].imag, 1.0);

        // 2. double + Complex
        final res2 = add(d, c);
        expect(res2.dtype, DType.complex128);
        expect(res2.data[0].real, 4.0);
        expect(res2.data[0].imag, 1.0);

        // 3. int + Complex
        final res3 = add(i, c);
        expect(res3.dtype, DType.complex128);
        expect(res3.data[0].real, 3.0);
      }),
    );

    test(
      'inv() in-place out buffer validations and solvers coverage',
      () => NDArray.scope(() {
        final a = NDArray<double>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final out = NDArray<double>.zeros([2, 2], DType.float64);
        final incompatibleOut = NDArray<double>.ones([3, 3], DType.float64);

        // 1. Incompatible out shape throws ArgumentError
        expect(() => inv(a, out: incompatibleOut), throwsArgumentError);

        // 2. Valid in-place solving
        final res = inv(a, out: out);
        expect(identical(res, out), true);
        expect(res.data[0], closeTo(-2.0, 1e-9));
        expect(res.data[1], closeTo(1.0, 1e-9));
        expect(res.data[2], closeTo(1.5, 1e-9));
        expect(res.data[3], closeTo(-0.5, 1e-9));
      }),
    );

    test(
      '_resolveDType cross-promotion additions coverage',
      () => NDArray.scope(() {
        final f64 = NDArray<double>.fromList([1.0], [1], DType.float64);
        final f32 = NDArray<double>.fromList([2.0], [1], DType.float32);
        final i64 = NDArray<int>.fromList([3], [1], DType.int64);
        final i32 = NDArray<int>.fromList([4], [1], DType.int32);

        // 1. float64 + float32 -> float64
        final r1 = add(f64, f32);
        expect(r1.dtype, DType.float64);

        // 2. float32 + int64 -> float32
        final r2 = add(f32, i64);
        expect(r2.dtype, DType.float32);

        // 3. int64 + int32 -> int64
        final r3 = add(i64, i32);
        expect(r3.dtype, DType.int64);

        // 4. int32 + float64 -> float64
        final r4 = add(i32, f64);
        expect(r4.dtype, DType.float64);
        expect(r4.toList(), [5.0]);
      }),
    );

    test(
      'NDArray.eye() complex identity matrix type safety validations',
      () => NDArray.scope(() {
        final eye128 = NDArray<Complex>.eye(3, DType.complex128);
        expect(eye128.dtype, DType.complex128);
        expect(eye128.data[0], Complex(1.0, 0.0));
        expect(eye128.data[1], Complex(0.0, 0.0));
        expect(eye128.data[4], Complex(1.0, 0.0));

        final eye64 = NDArray<Complex>.eye(3, DType.complex64);
        expect(eye64.dtype, DType.complex64);
        expect(eye64.data[0], Complex(1.0, 0.0));
        expect(eye64.data[4], Complex(1.0, 0.0));
      }),
    );

    test(
      'prod() contiguous FFI leaf paths coverage',
      () => NDArray.scope(() {
        final f64 = NDArray<double>.fromList(
          [2.0, 3.0, 4.0],
          [3],
          DType.float64,
        );
        final f32 = NDArray<double>.fromList(
          [5.0, 2.0, 3.0],
          [3],
          DType.float32,
        );

        // 1. float64 contiguous FFI prod()
        final r1 = prod(f64);
        expect(r1.scalar, closeTo(24.0, 1e-9));

        // 2. float32 contiguous FFI prod()
        final r2 = prod(f32);
        expect(r2.scalar, closeTo(30.0, 1e-9));
      }),
    );

    test(
      'prod() and sum() on non-contiguous strided views along axes',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          [3, 2],
          DType.float64,
        );
        final viewT = parent.transposed; // non-contiguous view of shape [2, 3]
        expect(viewT.isContiguous, false);

        // Global reduction
        expect(prod(viewT).scalar, 720.0); // 1*2*3*4*5*6 = 720
        expect(sum(viewT).scalar, 21.0);

        // Reduction along axes (triggers _reduceRecursive fallback paths)
        final p0 = prod(viewT, axis: 0); // Product along axis 0 -> shape [3]
        expect(p0.shape, [3]);
        expect(p0.toList(), [
          2.0,
          12.0,
          30.0,
        ]); // col 0: 1*2=2, col 1: 3*4=12, col 2: 5*6=30

        final p1 = prod(viewT, axis: 1); // Product along axis 1 -> shape [2]
        expect(p1.shape, [2]);
        expect(p1.toList(), [15.0, 48.0]); // row 0: 1*3*5=15, row 1: 2*4*6=48
      }),
    );

    test(
      'variance() and std() on non-contiguous view calculation',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
          [3, 2],
          DType.float64,
        );

        final viewT = parent.transposed;

        expect(variance(viewT).scalar, closeTo(17.5 / 6.0, 1e-9));
        expect(std(viewT).scalar, closeTo(math.sqrt(17.5 / 6.0), 1e-9));
      }),
    );

    test(
      'variance() and std() on sliced view calculation',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
          [3, 2],
          DType.float64,
        );

        final view = parent.slice([Slice(start: 0, stop: 2), Slice.all()]);
        expect(variance(view).scalar, closeTo(1.25, 1e-9));
        expect(std(view).scalar, closeTo(math.sqrt(1.25), 1e-9));
      }),
    );

    test(
      'clip() with named out parameter recycler and sliced contiguous view',
      () {
        final parent = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
          [3, 2],
          DType.float64,
        );

        final view = parent.slice([Slice(start: 0, stop: 2), Slice.all()]);
        final out = NDArray<double>.zeros([2, 2], DType.float64);

        final res = clip(view, min: 2.0, max: 3.0, out: out);
        expect(identical(res, out), true);
        expect(out.toList(), [2.0, 2.0, 3.0, 3.0]);

        expect(parent.toList(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
      },
    );

    group('NaN-Ignoring Statistical Reductions tests', () {
      test(
        'Verify nansum() treats NaNs as zeros',
        () => NDArray.scope(() {
          final a = NDArray<double>.fromList(
            Float64List.fromList([1.0, double.nan, 3.0, double.nan]),
            [2, 2],
            DType.float64,
          );

          expect(nansum(a).scalar, closeTo(4.0, 1e-9));

          final s0 = nansum(a, axis: 0);
          expect(s0.shape, [2]);
          expect(s0.toList(), [4.0, 0.0]);
        }),
      );

      test(
        'Verify nanmean() ignores NaNs',
        () => NDArray.scope(() {
          final a = NDArray<double>.fromList(
            Float64List.fromList([1.0, double.nan, 3.0, 4.0]),
            [2, 2],
            DType.float64,
          );

          expect(nanmean(a).scalar, closeTo(8.0 / 3.0, 1e-9));

          final m0 = nanmean(a, axis: 0);
          expect(m0.shape, [2]);
          expect(m0.toList(), [2.0, 4.0]);
        }),
      );

      test(
        'Verify nanvar() and nanstd() ignore NaNs',
        () => NDArray.scope(() {
          final a = NDArray<double>.fromList(
            Float64List.fromList([1.0, double.nan, 2.0, 3.0]),
            [2, 2],
            DType.float64,
          );

          // mean = (1 + 2 + 3) / 3 = 2.0
          // var = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
          expect(nanvar(a).scalar, closeTo(2.0 / 3.0, 1e-9));
          expect(nanstd(a).scalar, closeTo(math.sqrt(2.0 / 3.0), 1e-9));

          final v0 = nanvar(a, axis: 0);
          final s0 = nanstd(a, axis: 0);

          expect(v0.shape, [2]);
          expect(s0.shape, [2]);
        }),
      );
    });

    test(
      'arange() and linspace() type safety with complex dtypes',
      () => NDArray.scope(() {
        // 1. arange with complex128
        final a128 = NDArray<Complex>.arange(0, 3, dtype: DType.complex128);
        expect(a128.shape, [3]);
        expect(a128.dtype, DType.complex128);
        expect(a128.toList(), [
          Complex(0.0, 0.0),
          Complex(1.0, 0.0),
          Complex(2.0, 0.0),
        ]);

        // 2. arange with complex64
        final a64 = NDArray<Complex>.arange(1, 3, dtype: DType.complex64);
        expect(a64.shape, [2]);
        expect(a64.dtype, DType.complex64);
        expect(a64.toList(), [Complex(1.0, 0.0), Complex(2.0, 0.0)]);

        // 3. linspace with complex128
        final l128 = NDArray<Complex>.linspace(
          1.0,
          2.0,
          3,
          dtype: DType.complex128,
        );
        expect(l128.shape, [3]);
        expect(l128.dtype, DType.complex128);
        expect(l128.toList(), [
          Complex(1.0, 0.0),
          Complex(1.5, 0.0),
          Complex(2.0, 0.0),
        ]);

        // 4. linspace with single-element num == 1 and complex64
        final l64 = NDArray<Complex>.linspace(
          5.0,
          10.0,
          1,
          dtype: DType.complex64,
        );
        expect(l64.shape, [1]);
        expect(l64.dtype, DType.complex64);
        expect(l64.toList(), [Complex(5.0, 0.0)]);
      }),
    );

    test(
      'where() with out parameter recycler and shape validations',
      () => NDArray.scope(() {
        final cond = NDArray.fromList(
          [true, false, false, true],
          [2, 2],
          DType.boolean,
        );
        final x = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final y = NDArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float64,
        );
        final out = NDArray<double>.zeros([2, 2], DType.float64);
        final incompatibleOut = NDArray<double>.zeros([3], DType.float64);

        // 1. Incompatible shape throws ArgumentError
        expect(() => where(cond, x, y, incompatibleOut), throwsArgumentError);

        // 2. Valid in-place recycling
        final res = where(cond, x, y, out);
        expect(identical(res, out), true);
        expect(out.toList(), [1.0, 20.0, 30.0, 4.0]);
      }),
    );

    test(
      'uniform() and normal() Float32, and randint() Int32 coverage',
      () => NDArray.scope(() {
        // 1. uniform with float32
        final u = uniform([10], dtype: DType.float32);
        expect(u.shape, [10]);
        expect(u.dtype, DType.float32);
        for (final val in u.data) {
          expect(val, greaterThanOrEqualTo(0.0));
          expect(val, lessThan(1.0));
        }

        // 2. randint with int32
        final ri = randint([10], low: 0, high: 5, dtype: DType.int32);
        expect(ri.shape, [10]);
        expect(ri.dtype, DType.int32);
        for (final val in ri.data) {
          expect(val, greaterThanOrEqualTo(0));
          expect(val, lessThan(5));
        }

        // 3. normal with float32
        final n = normal([10], dtype: DType.float32);
        expect(n.shape, [10]);
        expect(n.dtype, DType.float32);
      }),
    );

    test(
      'setByMask() with NDArray values and capacity validations',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final mask = NDArray.fromList(
          [true, false, false, true],
          [2, 2],
          DType.boolean,
        );
        final values = NDArray.fromList([99.0, 100.0], [2], DType.float64);
        final insufficientValues = NDArray.fromList([99.0], [1], DType.float64);

        // 1. Incompatible capacity throws ArgumentError
        expect(
          () => parent.setByMask(mask, insufficientValues),
          throwsArgumentError,
        );

        // 2. Valid array value mask mutation
        parent.setByMask(mask, values);
        expect(parent.toList(), [99.0, 2.0, 3.0, 100.0]);
      }),
    );

    test(
      'setIndices() RangeError and ArgumentError boundaries coverage',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final indices = NDArray.fromList([0, 1], [2], DType.int32);
        final values = NDArray.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float64,
        );
        final insufficientValues = NDArray.fromList([10.0], [1], DType.float64);

        // 1. Invalid axis throws RangeError
        expect(
          () => parent.setIndices(indices, values, axis: -1),
          throwsRangeError,
        );
        expect(
          () => parent.setIndices(indices, values, axis: 2),
          throwsRangeError,
        );

        // 2. Insufficient values capacity throws ArgumentError
        expect(
          () => parent.setIndices(indices, insufficientValues, axis: 0),
          throwsArgumentError,
        );
      }),
    );

    test(
      'Contiguous sub-slice flatten() optimization correctness',
      () => NDArray.scope(() {
        final parent = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float64,
        );
        // Slice representing the first 2 elements (strides: [1], contiguous: true, totalSize < data.length!)
        final view = parent.slice([Slice(start: 0, stop: 2)]);

        expect(view.isContiguous, true);
        expect(view.shape, [2]);

        final flat = view.flatten();
        expect(flat.shape, [2]);
        expect(flat.isContiguous, true);
        expect(flat.toList(), [1.0, 2.0]);
      }),
    );

    test(
      'NDArray.fill() ufunc correctness and performance speedups verification',
      () {
        // 1. Contiguous Double Precision fill
        final a = NDArray<double>.zeros([5], DType.float64);
        a.fill(42.5);
        expect(a.toList(), [42.5, 42.5, 42.5, 42.5, 42.5]);

        // 2. Contiguous Int32 Precision fill
        final b = NDArray<int>.zeros([5], DType.int32);
        b.fill(99);
        expect(b.toList(), [99, 99, 99, 99, 99]);

        // 3. Strided view fallback JIT fill
        final parent = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float64,
        );
        final view = parent.slice([
          Slice(start: 0, stop: 4, step: 2),
        ]); // indices: 0, 2

        expect(view.shape, [2]);
        expect(view.isContiguous, false);

        view.fill(77.0);
        expect(parent.toList(), [77.0, 2.0, 77.0, 4.0]);
      },
    );

    test(
      'diag() diagonal matrix ufunc correctness and zero-copy view validations',
      () {
        // 1. Main diagonal extraction (k = 0)
        final mat = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
          [3, 3],
          DType.float64,
        );

        final mainDiag = diag(mat);

        expect(mainDiag.shape, [3]);
        expect(mainDiag.isContiguous, false); // spaced view
        expect(mainDiag.toList(), [1.0, 5.0, 9.0]);

        // 2. Strided views offset extractions (k = 1 and k = -1)
        final upperDiag = diag(mat, k: 1);
        final lowerDiag = diag(mat, k: -1);

        expect(upperDiag.shape, [2]);
        expect(upperDiag.toList(), [2.0, 6.0]);
        expect(lowerDiag.shape, [2]);
        expect(lowerDiag.toList(), [4.0, 8.0]);

        // 3. Construct diagonal 2D matrix from 1D vector
        final vec = NDArray.fromList([10.0, 20.0], [2], DType.float64);
        final dMat = diag(vec);
        final uDiagMat = diag(vec, k: 1);

        expect(dMat.shape, [2, 2]);
        expect(dMat.toList(), [10.0, 0.0, 0.0, 20.0]);

        expect(uDiagMat.shape, [3, 3]);
        expect(uDiagMat.toList(), [
          0.0,
          10.0,
          0.0,
          0.0,
          0.0,
          20.0,
          0.0,
          0.0,
          0.0,
        ]);

        // 4. Rank out of bounds throws ArgumentError
        final tensor3d = NDArray.zeros([2, 2, 2], DType.float64);
        expect(() => diag(tensor3d), throwsArgumentError);
      },
    );

    test(
      'isclose() and allclose() approximate equality ufunc correctness',
      () => NDArray.scope(() {
        final a = NDArray.fromList([1.0, 1.00001, 2.0], [3], DType.float64);
        final b = NDArray.fromList([1.0, 1.00003, 2.0], [3], DType.float64);

        // 1. Default tolerances: rtol = 1e-05, atol = 1e-08
        final closeDefault = isclose(a, b);
        expect(closeDefault.toList(), [true, false, true]);

        // 2. Stretched tolerances: rtol = 1e-04
        final closeStretched = isclose(a, b, rtol: 1e-04);
        expect(closeStretched.toList(), [true, true, true]);

        // 3. allclose logic
        expect(allclose(a, b), false);
        expect(allclose(a, b, rtol: 1e-04), true);

        // 4. Infinite matching values
        final infA = NDArray.fromList(
          [double.infinity, double.negativeInfinity],
          [2],
          DType.float64,
        );
        final infB = NDArray.fromList(
          [double.infinity, double.negativeInfinity],
          [2],
          DType.float64,
        );
        final infC = NDArray.fromList(
          [double.negativeInfinity, double.infinity],
          [2],
          DType.float64,
        );

        final closeInf = isclose(infA, infB);
        final closeInfMismatch = isclose(infA, infC);

        expect(closeInf.toList(), [true, true]);
        expect(closeInfMismatch.toList(), [false, false]);

        // 5. NaN value equalNan checks
        final nanA = NDArray.fromList([double.nan], [1], DType.float64);
        final nanB = NDArray.fromList([double.nan], [1], DType.float64);

        final closeNanDefault = isclose(nanA, nanB);
        final closeNanEqual = isclose(nanA, nanB, equalNan: true);

        expect(closeNanDefault.toList(), [false]);
        expect(closeNanEqual.toList(), [true]);
      }),
    );

    test(
      'nan_to_num() dataset cleaning ufunc correctness',
      () => NDArray.scope(() {
        // 1. Default Float64 cleaning
        final a = NDArray.fromList(
          [1.0, double.nan, double.infinity, double.negativeInfinity],
          [4],
          DType.float64,
        );

        final cleanDefault = nan_to_num(a);

        expect(cleanDefault.toList()[0], 1.0);
        expect(cleanDefault.toList()[1], 0.0);
        expect(cleanDefault.toList()[2], double.maxFinite);
        expect(cleanDefault.toList()[3], -double.maxFinite);

        // 2. Custom parameters cleaning
        final cleanCustom = nan_to_num(
          a,
          nan: 99.0,
          posinf: 500.0,
          neginf: -500.0,
        );

        expect(cleanCustom.toList(), [1.0, 99.0, 500.0, -500.0]);

        // 3. View-safe in-place recycling
        final parent = NDArray.fromList(
          [double.nan, 2.0, double.nan, 4.0],
          [4],
          DType.float64,
        );
        final view = parent.slice([
          Slice(start: 0, stop: 4, step: 2),
        ]); // indices: 0, 2

        expect(view.isContiguous, false);
        expect(view.isContiguous, false);
        nan_to_num(view, nan: 100.0, out: view);

        expect(parent.toList(), [100.0, 2.0, 100.0, 4.0]);
      }),
    );

    test(
      'Linear Algebra solvers det() and solve() singular and preconditions exceptions',
      () {
        // 1. det() singular matrix returns 0.0
        final singularMat = NDArray.fromList(
          [
            1.0, 2.0,
            2.0, 4.0, // linearly dependent rows!
          ],
          [2, 2],
          DType.float64,
        );
        expect(det(singularMat), 0.0);

        // 2. solve() non-square matrix throws ArgumentError
        final nonSquareA = NDArray.fromList(
          [1.0, 2.0, 3.0],
          [1, 3],
          DType.float64,
        );
        final b = NDArray.fromList([1.0], [1], DType.float64);
        expect(() => solve(nonSquareA, b), throwsArgumentError);

        // 3. solve() incompatible RHS shape throws ArgumentError
        final squareA = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final incompatibleB = NDArray.fromList(
          [1.0, 2.0, 3.0],
          [3],
          DType.float64,
        );
        expect(() => solve(squareA, incompatibleB), throwsArgumentError);

        // 4. solve() singular Float64 matrix throws singular ArgumentError
        final singularFloat64A = NDArray.fromList(
          [1.0, 2.0, 2.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final validB = NDArray.fromList([5.0, 6.0], [2], DType.float64);
        expect(() => solve(singularFloat64A, validB), throwsArgumentError);

        // 5. solve() singular Float32 matrix throws singular ArgumentError
        final singularFloat32A = NDArray.fromList(
          [1.0, 2.0, 2.0, 4.0],
          [2, 2],
          DType.float32,
        );
        final validFloat32B = NDArray.fromList([5.0, 6.0], [2], DType.float32);
        expect(
          () => solve(singularFloat32A, validFloat32B),
          throwsArgumentError,
        );
      },
    );

    test(
      'expand_dims() and squeeze() shape view manipulations ufunc correctness',
      () {
        // 1. expand_dims() verification
        final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);

        final aExp0 = expand_dims(a, 0);
        final aExp1 = expand_dims(a, 1);
        final aExpNeg = expand_dims(a, -1); // normalized to axis 1

        expect(aExp0.shape, [1, 3]);
        expect(aExp0.strides, [1, 1]);

        expect(aExp1.shape, [3, 1]);
        expect(aExp1.strides, [1, 1]);

        expect(aExpNeg.shape, [3, 1]);

        // expand_dims out of bounds exception
        expect(() => expand_dims(a, 3), throwsArgumentError);

        // 2. squeeze() verification
        final b = NDArray.zeros([1, 3, 1], DType.float64);

        final bSqueezedAll = squeeze(b);
        final bSqueezed0 = squeeze(b, axis: [0]);

        expect(bSqueezedAll.shape, [3]);
        expect(bSqueezedAll.strides, [1]);

        expect(bSqueezed0.shape, [3, 1]);

        // Squeeze non-unit dimension axis throws ArgumentError
        expect(() => squeeze(b, axis: [1]), throwsArgumentError);
      },
    );

    test(
      'Contiguous sub-slice view sum() and prod() FFI reductions correctness',
      () {
        final parent = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [4],
          DType.float64,
        );
        final view = parent.slice([
          Slice(start: 0, stop: 2),
        ]); // elements: 1.0, 2.0

        expect(view.isContiguous, true);
        expect(view.shape, [2]);

        // 1. sum() verification
        final s = sum(view);
        expect(s.scalar, 3.0);

        // 2. prod() verification
        final p = prod(view);
        expect(p.scalar, 2.0);
      },
    );

    test(
      'concatenate() validation errors throws exceptions',
      () => NDArray.scope(() {
        expect(() => concatenate(<NDArray<Object>>[]), throwsArgumentError);

        final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final wrongShape = NDArray.fromList(
          [1.0, 2.0, 3.0],
          [3],
          DType.float64,
        );
        final wrongRank = NDArray.fromList([1.0, 2.0], [1, 2], DType.float64);
        final wrongDType = NDArray.fromList([1, 2], [2], DType.int32);

        expect(() => concatenate([a, a], axis: 5), throwsRangeError);
        expect(() => concatenate([a, a], axis: -5), throwsRangeError);
        expect(
          () => concatenate(<NDArray<Object>>[a, wrongDType]),
          throwsArgumentError,
        );

        expect(
          () => concatenate(<NDArray<Object>>[a, wrongRank]),
          throwsArgumentError,
        );

        final mat1 = NDArray.fromList(
          [1.0, 1.0, 1.0, 1.0],
          [2, 2],
          DType.float64,
        );
        final mat2 = NDArray.fromList(
          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
          [2, 3],
          DType.float64,
        );
        expect(
          () => concatenate(<NDArray<double>>[mat1, mat2], axis: 0),
          throwsArgumentError,
        );
      }),
    );

    test(
      'Cross-type arithmetic coverage for subtract, multiply, and divide',
      () {
        final c = NDArray<Complex>.fromList(
          [Complex(10.0, 10.0)],
          [1],
          DType.complex128,
        );
        final i = NDArray<int>.fromList([2], [1], DType.int64);
        final d = NDArray<double>.fromList([4.0], [1], DType.float64);

        // --- subtract() Gaps ---
        // 1. Complex - int
        final s1 = subtract(c, i);
        expect(s1.dtype, DType.complex128);
        expect(s1.data[0], Complex(8.0, 10.0));

        // 2. int - Complex
        final s2 = subtract(i, c);
        expect(s2.dtype, DType.complex128);
        expect(s2.data[0], Complex(-8.0, -10.0));

        // 3. double - int
        final s3 = subtract(d, i);
        expect(s3.dtype, DType.float64);
        expect(s3.data[0], 2.0);

        // 4. int - double
        final s4 = subtract(i, d);
        expect(s4.dtype, DType.float64);
        expect(s4.data[0], -2.0);

        // 5. int - int
        final s5 = subtract(i, i);
        expect(s5.dtype, DType.int64);
        expect(s5.data[0], 0);

        // --- multiply() Gaps ---
        // 1. Complex * Complex
        final m1 = multiply(c, c);
        expect(m1.dtype, DType.complex128);
        expect(m1.data[0], Complex(0.0, 200.0)); // (10+10i)^2 = 0 + 200i

        // 2. Complex * double
        final m2 = multiply(c, d);
        expect(m2.dtype, DType.complex128);
        expect(m2.data[0], Complex(40.0, 40.0));

        // 3. Complex * int
        final m3 = multiply(c, i);
        expect(m3.dtype, DType.complex128);
        expect(m3.data[0], Complex(20.0, 20.0));

        // 4. double * Complex
        final m4 = multiply(d, c);
        expect(m4.dtype, DType.complex128);
        expect(m4.data[0], Complex(40.0, 40.0));

        // 5. int * Complex
        final m5 = multiply(i, c);
        expect(m5.dtype, DType.complex128);
        expect(m5.data[0], Complex(20.0, 20.0));

        // 6. double * int
        final m6 = multiply(d, i);
        expect(m6.dtype, DType.float64);
        expect(m6.data[0], 8.0);

        // 7. int * double
        final m7 = multiply(i, d);
        expect(m7.dtype, DType.float64);
        expect(m7.data[0], 8.0);

        // 8. int * int
        final m8 = multiply(i, i);
        expect(m8.dtype, DType.int64);
        expect(m8.data[0], 4);

        // --- divide() Gaps ---
        // 1. Complex / Complex
        final div1 = divide(c, c);
        expect(div1.dtype, DType.complex128);
        expect(div1.data[0], Complex(1.0, 0.0));

        // 2. Complex / double
        final div2 = divide(c, d);
        expect(div2.dtype, DType.complex128);
        expect(div2.data[0], Complex(2.5, 2.5));

        // 3. Complex / int
        final div3 = divide(c, i);
        expect(div3.dtype, DType.complex128);
        expect(div3.data[0], Complex(5.0, 5.0));

        // 4. double / Complex
        final div4 = divide(d, c);
        expect(div4.dtype, DType.complex128);
        expect(div4.data[0], Complex(0.2, -0.2)); // 4 / (10+10i) = 0.2 - 0.2i

        // 5. int / Complex
        final div5 = divide(i, c);
        expect(div5.data[0], Complex(0.1, -0.1)); // 2 / (10+10i) = 0.1 - 0.1i
      },
    );

    test(
      'nansum() flat sum type safety on Int32/Int64 and Complex128/Complex64 NDArrays',
      () {
        // 1. Int32 Summation (should return int scalar without double-cast crash)
        final aInt32 = NDArray.fromList([10, 20, 30, 40], [4], DType.int32);
        final sumI32 = nansum(aInt32);
        expect(sumI32.scalar, 100);
        expect(sumI32.scalar, isA<int>());

        // 2. Int64 Summation
        final aInt64 = NDArray.fromList([100, 200], [2], DType.int64);
        final sumI64 = nansum(aInt64);
        expect(sumI64.scalar, 300);
        expect(sumI64.scalar, isA<int>());

        // 3. Complex128 Summation (should return Complex scalar)
        final aC128 = NDArray<Complex>.create([2], DType.complex128);
        aC128.data[0] = Complex(1.0, 2.0);
        aC128.data[1] = Complex(3.0, 4.0);
        final sumC128 = nansum(aC128);
        expect(sumC128.scalar, Complex(4.0, 6.0));
        expect(sumC128.scalar, isA<Complex>());
      },
    );

    test(
      'solve() optimized contiguous block copy and non-contiguous view solvers correctness',
      () {
        // 1. Contiguous Float64 matrix solve
        final a = NDArray.fromList([3.0, 1.0, 1.0, 2.0], [2, 2], DType.float64);
        final b = NDArray.fromList([9.0, 8.0], [2], DType.float64);

        final x = solve(a, b);
        expect(x.toList(), [2.0, 3.0]);

        // 2. Non-contiguous transposed Float64 matrix solve
        final aParent = NDArray.fromList(
          [3.0, 1.0, 1.0, 2.0],
          [2, 2],
          DType.float64,
        );
        final aTransposed = aParent.transposed; // non-contiguous!
        // aTransposed is: [[3.0, 1.0], [1.0, 2.0]] which is symmetric, so solve is same
        final bParent = NDArray.fromList([9.0, 8.0], [2], DType.float64);

        final x2 = solve(aTransposed, bParent);
        expect(x2.toList(), [2.0, 3.0]);
      },
    );

    test(
      'Zero-copy FFT and IFFT contiguous Float64 complex128 correctness',
      () {
        // 1. 1D Complex Vector Zero-Copy FFT and IFFT
        final a = NDArray<Complex>.create([4], DType.complex128);
        a.data[0] = Complex(1.0, 0.0);
        a.data[1] = Complex(2.0, 0.0);
        a.data[2] = Complex(3.0, 0.0);
        a.data[3] = Complex(4.0, 0.0);

        final resFFT = fft(a);
        expect(resFFT.dtype, DType.complex128);
        expect(resFFT.shape, [4]);

        // Mathematical FFT outputs for [1, 2, 3, 4]:
        // F(0) = 10 + 0i
        // F(1) = -2 + 2i
        // F(2) = -2 + 0i
        // F(3) = -2 - 2i
        expect(resFFT.data[0], Complex(10.0, 0.0));
        expect(resFFT.data[1].real, closeTo(-2.0, 1e-10));
        expect(resFFT.data[1].imag, closeTo(2.0, 1e-10));
        expect(resFFT.data[2].real, closeTo(-2.0, 1e-10));
        expect(resFFT.data[2].imag, closeTo(0.0, 1e-10));
        expect(resFFT.data[3].real, closeTo(-2.0, 1e-10));
        expect(resFFT.data[3].imag, closeTo(-2.0, 1e-10));

        // Round-trip back with zero-copy IFFT
        final resIFFT = ifft(resFFT);
        expect(resIFFT.dtype, DType.complex128);
        expect(resIFFT.shape, [4]);
        expect(resIFFT.data[0].real, closeTo(1.0, 1e-10));
        expect(resIFFT.data[0].imag, closeTo(0.0, 1e-10));
        expect(resIFFT.data[1].real, closeTo(2.0, 1e-10));
        expect(resIFFT.data[1].imag, closeTo(0.0, 1e-10));
        expect(resIFFT.data[2].real, closeTo(3.0, 1e-10));
        expect(resIFFT.data[2].imag, closeTo(0.0, 1e-10));
        expect(resIFFT.data[3].real, closeTo(4.0, 1e-10));
        expect(resIFFT.data[3].imag, closeTo(0.0, 1e-10));

        // 2. High-Dimensional Stacked 2D Complex Matrix Zero-Copy FFT and IFFT
        final mat = NDArray<Complex>.create([2, 2], DType.complex128);
        mat.data[0] = Complex(1.0, 0.0);
        mat.data[1] = Complex(2.0, 0.0);
        mat.data[2] = Complex(3.0, 0.0);
        mat.data[3] = Complex(4.0, 0.0);

        final matFFT = fft(mat);
        expect(matFFT.shape, [2, 2]);
        expect(matFFT.dtype, DType.complex128);

        // Row 0: [1, 2] -> [3, -1]
        // Row 1: [3, 4] -> [7, -1]
        expect(matFFT.data[0].real, closeTo(3.0, 1e-10));
        expect(matFFT.data[1].real, closeTo(-1.0, 1e-10));
        expect(matFFT.data[2].real, closeTo(7.0, 1e-10));
        expect(matFFT.data[3].real, closeTo(-1.0, 1e-10));

        final matIFFT = ifft(matFFT);
        expect(matIFFT.shape, [2, 2]);
        expect(matIFFT.data[0].real, closeTo(1.0, 1e-10));
        expect(matIFFT.data[1].real, closeTo(2.0, 1e-10));
        expect(matIFFT.data[2].real, closeTo(3.0, 1e-10));
        expect(matIFFT.data[3].real, closeTo(4.0, 1e-10));
      },
    );

    test(
      'Non-contiguous/strided integer ufuncs fallback walks (tan, abs, ceil, floor, round)',
      () {
        final i = NDArray.fromList([-1, -2, -3, -4], [2, 2], DType.int64);
        final iT = i.transposed;

        final rAbs = abs(iT);
        expect(rAbs.toList(), [1, 3, 2, 4]);

        final rTan = tan(iT);
        expect(rTan.data[0], closeTo(math.tan(-1.0), 1e-9));

        final rCeil = ceil(iT);
        expect(rCeil.toList(), [-1, -3, -2, -4]);

        final rFloor = floor(iT);
        expect(rFloor.toList(), [-1, -3, -2, -4]);

        final rRound = round(iT);
        expect(rRound.toList(), [-1, -3, -2, -4]);
      },
    );

    test(
      'Contiguous and non-contiguous clip() precision ufuncs coverage',
      () => NDArray.scope(() {
        // 1. Contiguous Float32 clip
        final f32 = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float32);
        final resF32 = clip(f32, min: 2.0, max: 3.0);
        expect(resF32.dtype, DType.float32);
        expect(resF32.toList(), [2.0, 2.0, 3.0, 3.0]);

        // 2. Non-contiguous integer clip
        final i32 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        final i32T = i32.transposed;
        final resI32 = clip(i32T, min: 2, max: 3);
        expect(resI32.toList(), [
          2,
          3,
          2,
          3,
        ]); // transposed: 1, 3, 2, 4 -> clipped: 2, 3, 2, 3

        // 3. Non-contiguous double clip
        final f64 = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final f64T = f64.transposed;
        final resF64 = clip(f64T, min: 2.0, max: 3.0);
        expect(resF64.toList(), [2.0, 3.0, 2.0, 3.0]);
      }),
    );

    test(
      'Multivariate Normal Distribution multivariateNormal() correctness',
      () => NDArray.scope(() {
        final mean = NDArray.fromList([10.0, 20.0], [2], DType.float64);
        // Identity covariance Sigma = I
        final cov = NDArray.fromList(
          [1.0, 0.0, 0.0, 1.0],
          [2, 2],
          DType.float64,
        );

        // Draw 500 samples (returns shape [500, 2])
        final samples = multivariateNormal(mean, cov, size: [500]);

        expect(samples.shape, [500, 2]);
        expect(samples.dtype, DType.float64);

        // Check statistical means of the drawn samples to verify standard convergence!
        var sumX = 0.0;
        var sumY = 0.0;
        for (var i = 0; i < 500; i++) {
          sumX += samples.getCell([i, 0]);
          sumY += samples.getCell([i, 1]);
        }
        final meanX = sumX / 500;
        final meanY = sumY / 500;

        // Means should converge closely to [10, 20] under standard deviation 1
        expect(meanX, closeTo(10.0, 0.5));
        expect(meanY, closeTo(20.0, 0.5));

        // Verify ArgumentError throwing exceptions
        final badMean = NDArray.fromList([10.0, 20.0], [1, 2], DType.float64);
        expect(() => multivariateNormal(badMean, cov), throwsArgumentError);

        final mismatchedMean = NDArray.fromList([10.0], [1], DType.float64);
        expect(
          () => multivariateNormal(mismatchedMean, cov),
          throwsArgumentError,
        );
      }),
    );

    test(
      'Sliding Window Views slidingWindowView() zero-copy view correctness',
      () {
        // 1D array sliding window of size 3
        final a = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [5],
          DType.float64,
        );

        final view = slidingWindowView(a, [3]);
        expect(view.shape, [3, 3]);
        expect(view.dtype, DType.float64);

        // Output values should exactly match the sliding windows:
        // Row 0: [1.0, 2.0, 3.0]
        // Row 1: [2.0, 3.0, 4.0]
        // Row 2: [3.0, 4.0, 5.0]
        expect(view.getCell([0, 0]), 1.0);
        expect(view.getCell([0, 1]), 2.0);
        expect(view.getCell([0, 2]), 3.0);

        expect(view.getCell([1, 0]), 2.0);
        expect(view.getCell([1, 1]), 3.0);
        expect(view.getCell([1, 2]), 4.0);

        expect(view.getCell([2, 0]), 3.0);
        expect(view.getCell([2, 1]), 4.0);
        expect(view.getCell([2, 2]), 5.0);

        // 2D array sliding window along specified axis
        final mat = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
          [2, 4],
          DType.float64,
        );

        // Slide window of size 2 along axis 1 (columns)
        final win2d = slidingWindowView(mat, [2], axis: [1]);
        expect(win2d.shape, [
          2,
          3,
          2,
        ]); // original [2, 4] -> columns: 4 - 2 + 1 = 3 -> [2, 3, 2]

        // Row 0 sliding windows:
        // Win 0: [1.0, 2.0]
        // Win 1: [2.0, 3.0]
        // Win 2: [3.0, 4.0]
        expect(win2d.getCell([0, 0, 0]), 1.0);
        expect(win2d.getCell([0, 0, 1]), 2.0);
        expect(win2d.getCell([0, 1, 0]), 2.0);
        expect(win2d.getCell([0, 1, 1]), 3.0);

        // Verify ArgumentError validations
        expect(
          () => slidingWindowView(a, [6]),
          throwsArgumentError,
        ); // window exceeds size
        expect(
          () => slidingWindowView(a, [0]),
          throwsArgumentError,
        ); // invalid positive size
        expect(
          () => slidingWindowView(a, [2, 2]),
          throwsArgumentError,
        ); // axes count mismatch
      },
    );

    test(
      'Advanced Vector Selection select() correctness and broadcasting',
      () => NDArray.scope(() {
        final cond1 = NDArray.fromList(
          [true, false, false],
          [3],
          DType.boolean,
        );
        final cond2 = NDArray.fromList(
          [false, true, false],
          [3],
          DType.boolean,
        );

        final choice1 = NDArray.fromList([10, 20, 30], [3], DType.int32);
        final choice2 = NDArray.fromList([100, 200, 300], [3], DType.int32);

        // 1. Standard select
        final res = select(
          [cond1, cond2],
          [choice1, choice2],
          defaultValue: 999,
        );

        expect(res.shape, [3]);
        expect(res.dtype, DType.int32);
        expect(
          res.toList(),
          [10, 200, 999],
        ); // cond1 true at 0 -> 10; cond2 true at 1 -> 200; default at 2 -> 999

        // 2. Broadcasting select (scalar condition, vector choices)
        final condScalar = NDArray.fromList([true], [1], DType.boolean);
        final resB = select([condScalar], [choice1], defaultValue: 999);

        expect(resB.shape, [3]); // broadcasted
        expect(resB.toList(), [10, 20, 30]);

        // 3. Verify ArgumentError throwing exceptions
        expect(() => select([], [choice1]), throwsArgumentError); // empty list
        expect(() => select([cond1], []), throwsArgumentError);
        expect(
          () => select([cond1, cond2], [choice1]),
          throwsArgumentError,
        ); // length mismatch

        final badShapeChoice = NDArray.fromList([1, 2], [2], DType.int32);
        expect(
          () => select([cond1], [badShapeChoice]),
          throwsArgumentError,
        ); // shape mismatch
      }),
    );

    test('Multinomial Distribution multinomial() trial simulation correctness', () {
      final pvals = NDArray.fromList([0.2, 0.5, 0.3], [3], DType.float64);

      // Draw 1000 samples of 10 trials (shape [1000, 3])
      final samples = multinomial(10, pvals, size: [1000]);

      expect(samples.shape, [1000, 3]);
      expect(samples.dtype, DType.int32);

      // Test multinomial with pvals requiring normalization (does not sum to 1.0)
      final nonNormalizedPvals = NDArray.fromList(
        [0.2, 0.6, 0.3],
        [3],
        DType.float64,
      );
      final samplesNonNorm = multinomial(10, nonNormalizedPvals, size: [5]);
      expect(samplesNonNorm.shape, [5, 3]);

      // For every sample, the sum of category counts must exactly equal the trials 'n' (10)!
      for (var i = 0; i < 1000; i++) {
        final sum =
            samples.getCell([i, 0]) +
            samples.getCell([i, 1]) +
            samples.getCell([i, 2]);
        expect(sum, 10);
      }

      // Statistical ratios should converge close to [0.2, 0.5, 0.3]
      var count0 = 0.0;
      var count1 = 0.0;
      var count2 = 0.0;
      for (var i = 0; i < 1000; i++) {
        count0 += samples.getCell([i, 0]);
        count1 += samples.getCell([i, 1]);
        count2 += samples.getCell([i, 2]);
      }
      final r0 = count0 / 10000.0;
      final r1 = count1 / 10000.0;
      final r2 = count2 / 10000.0;

      expect(r0, closeTo(0.2, 0.05));
      expect(r1, closeTo(0.5, 0.05));
      expect(r2, closeTo(0.3, 0.05));

      // Verify exceptions throwing
      expect(
        () => multinomial(-5, pvals),
        throwsArgumentError,
      ); // negative trials

      final badShapePvals = NDArray.fromList([0.5, 0.5], [1, 2], DType.float64);
      expect(
        () => multinomial(10, badShapePvals),
        throwsArgumentError,
      ); // bad shape pvals

      final negativePvals = NDArray.fromList([-0.2, 1.2], [2], DType.float64);
      expect(
        () => multinomial(10, negativePvals),
        throwsArgumentError,
      ); // negative probability
    });

    test(
      'Type-preserving reductions min(), max(), nanmin(), nanmax() DType parity',
      () {
        // 1. Integer min() / max() DType preservation
        final aInt32 = NDArray.fromList(
          [10, 2, 30, 4, 50, 6],
          [3, 2],
          DType.int32,
        );

        final minI32 = min(aInt32, axis: 0);
        expect(minI32.shape, [2]);
        expect(minI32.dtype, DType.int32); // Preserves Int32!
        expect(minI32.toList(), [10, 2]);

        final maxI32 = max(aInt32, axis: 0);
        expect(maxI32.shape, [2]);
        expect(maxI32.dtype, DType.int32); // Preserves Int32!
        expect(maxI32.toList(), [50, 6]);

        // 2. Float32 min() / max() DType preservation
        final aFloat32 = NDArray.fromList(
          [10.0, 2.0, 30.0, 4.0, 50.0, 6.0],
          [3, 2],
          DType.float32,
        );

        final minF32 = min(aFloat32, axis: 0);
        expect(minF32.dtype, DType.float32); // Preserves Float32!

        // 3. nanmin() / nanmax() DType preservation
        final nanF64 = NDArray.fromList(
          [1.0, double.nan, 3.0, 4.0, double.nan, 6.0],
          [3, 2],
          DType.float64,
        );

        final nanMinF64 = nanmin(nanF64, axis: 0);
        expect(nanMinF64.shape, [2]);
        expect(nanMinF64.dtype, DType.float64);
        expect(nanMinF64.getCell([0]), 1.0);

        final nanMaxF64 = nanmax(nanF64, axis: 0);
        expect(nanMaxF64.shape, [2]);
        expect(nanMaxF64.dtype, DType.float64);
        expect(nanMaxF64.getCell([0]), 3.0);
        expect(nanMaxF64.getCell([1]), 6.0);
      },
    );

    test(
      'FFT and IFFT zero-padding with ComplexList inputs and outputs',
      () => NDArray.scope(() {
        final a = NDArray<Complex>.create([4], DType.complex128);
        a.data[0] = Complex(1.0, 0.0);
        a.data[1] = Complex(2.0, 0.0);
        a.data[2] = Complex(3.0, 0.0);
        a.data[3] = Complex(4.0, 0.0);

        // Trigger zero-padding block for ComplexList input
        final res = fft(a, n: 8);
        expect(res.shape, [8]);
        expect(res.dtype, DType.complex128);

        // Trigger IFFT zero-padding block for ComplexList input
        final invRes = ifft(res, n: 12);
        expect(invRes.shape, [12]);
        expect(invRes.dtype, DType.complex128);
      }),
    );

    test(
      'NDArray cross-type comparison operators coverage',
      () => NDArray.scope(() {
        final comp = NDArray<Complex>.create([2], DType.complex128);
        comp.data[0] = Complex(1.0, 0.0);
        comp.data[1] = Complex(3.0, 0.0);

        final dbl = NDArray.fromList([2.0, 2.0], [2], DType.float64);

        final integer = NDArray.fromList([2, 2], [2], DType.int32);

        // 1. Complex with double
        final cDbl = comp.eq(dbl);
        expect(cDbl.toList(), [false, false]); // 1 != 2, 3 != 2

        // 2. Complex with int
        final cInt = comp.eq(integer);
        expect(cInt.toList(), [false, false]);

        // 3. double with Complex
        final dblC = dbl.eq(comp);
        expect(dblC.toList(), [false, false]);

        // 4. double with int
        final dblInt = dbl.eq(integer);
        expect(dblInt.toList(), [true, true]);

        // 5. int with Complex
        final intC = integer.eq(comp);
        expect(intC.toList(), [false, false]);

        // 6. int with double
        final intDbl = integer.eq(dbl);
        expect(intDbl.toList(), [true, true]);
      }),
    );

    test(
      'FFI native C flatten() and hashCode() correctness and invariants across all DTypes',
      () => NDArray.scope(() {
        // 1. Float64
        final f64 = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final f64T = f64.transposed;
        final f64Flat = f64T.flatten();
        expect(f64Flat.isContiguous, true);
        expect(f64Flat.toList(), [1.0, 3.0, 2.0, 4.0]);
        final f64Contig = NDArray.fromList(
          [1.0, 3.0, 2.0, 4.0],
          [2, 2],
          DType.float64,
        );
        expect(f64T == f64Contig, true);
        expect(f64T.hashCode == f64Contig.hashCode, true);

        // 2. Float32
        final f32 = NDArray.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float32,
        );
        final f32T = f32.transposed;
        final f32Flat = f32T.flatten();
        expect(f32Flat.isContiguous, true);
        expect(f32Flat.toList(), [1.0, 3.0, 2.0, 4.0]);
        final f32Contig = NDArray.fromList(
          [1.0, 3.0, 2.0, 4.0],
          [2, 2],
          DType.float32,
        );
        expect(f32T == f32Contig, true);
        expect(f32T.hashCode == f32Contig.hashCode, true);

        // 3. Int64
        final i64 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int64);
        final i64T = i64.transposed;
        final i64Flat = i64T.flatten();
        expect(i64Flat.isContiguous, true);
        expect(i64Flat.toList(), [1, 3, 2, 4]);
        final i64Contig = NDArray.fromList([1, 3, 2, 4], [2, 2], DType.int64);
        expect(i64T == i64Contig, true);
        expect(i64T.hashCode == i64Contig.hashCode, true);

        // 4. Int32
        final i32 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        final i32T = i32.transposed;
        final i32Flat = i32T.flatten();
        expect(i32Flat.isContiguous, true);
        expect(i32Flat.toList(), [1, 3, 2, 4]);
        final i32Contig = NDArray.fromList([1, 3, 2, 4], [2, 2], DType.int32);
        expect(i32T == i32Contig, true);
        expect(i32T.hashCode == i32Contig.hashCode, true);

        // 5. Complex128
        final c128 = NDArray<Complex>.fromList(
          [
            Complex(1.0, 1.0),
            Complex(2.0, 2.0),
            Complex(3.0, 3.0),
            Complex(4.0, 4.0),
          ],
          [2, 2],
          DType.complex128,
        );
        final c128T = c128.transposed;
        final c128Flat = c128T.flatten();
        expect(c128Flat.isContiguous, true);
        expect(c128Flat.toList(), [
          Complex(1.0, 1.0),
          Complex(3.0, 3.0),
          Complex(2.0, 2.0),
          Complex(4.0, 4.0),
        ]);
        final c128Contig = NDArray<Complex>.fromList(
          [
            Complex(1.0, 1.0),
            Complex(3.0, 3.0),
            Complex(2.0, 2.0),
            Complex(4.0, 4.0),
          ],
          [2, 2],
          DType.complex128,
        );
        expect(c128T == c128Contig, true);
        expect(c128T.hashCode == c128Contig.hashCode, true);

        // 6. Complex64
        final c64 = NDArray<Complex>.fromList(
          [
            Complex(1.0, 1.0),
            Complex(2.0, 2.0),
            Complex(3.0, 3.0),
            Complex(4.0, 4.0),
          ],
          [2, 2],
          DType.complex64,
        );
        final c64T = c64.transposed;
        final c64Flat = c64T.flatten();
        expect(c64Flat.isContiguous, true);
        expect(c64Flat.toList(), [
          Complex(1.0, 1.0),
          Complex(3.0, 3.0),
          Complex(2.0, 2.0),
          Complex(4.0, 4.0),
        ]);
        final c64Contig = NDArray<Complex>.fromList(
          [
            Complex(1.0, 1.0),
            Complex(3.0, 3.0),
            Complex(2.0, 2.0),
            Complex(4.0, 4.0),
          ],
          [2, 2],
          DType.complex64,
        );
        expect(c64T == c64Contig, true);
        expect(c64T.hashCode == c64Contig.hashCode, true);

        // 7. Boolean
        final b = NDArray.fromList(
          [true, false, true, false],
          [2, 2],
          DType.boolean,
        );
        final bT = b.transposed;
        final bFlat = bT.flatten();
        expect(bFlat.isContiguous, true);
        expect(bFlat.toList(), [true, true, false, false]);
        final bContig = NDArray.fromList(
          [true, true, false, false],
          [2, 2],
          DType.boolean,
        );
        expect(bT == bContig, true);
        expect(bT.hashCode == bContig.hashCode, true);
      }),
    );
    test(
      'Random generators support into parameter recycler buffer reuse',
      () => NDArray.scope(() {
        final buffer = NDArray<Float64>.zeros([10], DType.float64);

        // 1. uniform into
        final u = uniform([10], into: buffer);
        expect(identical(u, buffer), true);

        // 2. normal into
        final n = normal([10], into: buffer);
        expect(identical(n, buffer), true);

        // 3. exponential into
        final e = exponential([10], into: buffer);
        expect(identical(e, buffer), true);

        // 4. randint into
        final bufferInt = NDArray<Int64>.zeros([10], DType.int64);
        final r = randint([10], low: 0, high: 10, into: bufferInt);
        expect(identical(r, bufferInt), true);
      }),
    );
  });
}

final class ZeroThenDoubleRandom implements math.Random {
  var count = 0;

  @override
  double nextDouble() {
    count++;
    if (count == 1) {
      return 0.0; // Return 0.0 on first draw to force zero-avoidance loop path
    }
    return 0.5;
  }

  @override
  bool nextBool() => false;

  @override
  int nextInt(int max) => 0;
}
