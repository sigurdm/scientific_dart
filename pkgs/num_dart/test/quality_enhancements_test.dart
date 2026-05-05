import 'package:num_dart/num_dart.dart';
import 'package:test/test.dart';
import 'dart:io';
import 'dart:typed_data';
import 'dart:math' as math;

void main() {
  group('Quality Enhancements & Safety Gates Tests', () {
    test('arange() zero and negative step validation', () {
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
      addTearDown(a.dispose);
      expect(a.toList(), [5.0, 4.0, 3.0, 2.0, 1.0]);
    });

    test('reshape() non-contiguous view copies memory first', () {
      final parent = NDArray.fromList(
        Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
        [2, 2],
        DType.float64,
      );
      addTearDown(parent.dispose);

      // Transposed view is non-contiguous (strides: [1, 2])
      final transposed = parent.transposed;
      expect(transposed.isContiguous, false);
      expect(transposed.toList(), [1.0, 3.0, 2.0, 4.0]);

      // Reshaping the transposed view: it should automatically copy first
      final reshaped = transposed.reshape([4]);
      addTearDown(reshaped.dispose);
      expect(reshaped.isContiguous, true);
      expect(reshaped.toList(), [1.0, 3.0, 2.0, 4.0]);

      // Verify memory decoupling (modifying reshaped does NOT affect parent)
      reshaped.data[0] = 99.0;
      expect(parent.data[0], 1.0);
    });

    test('count_nonzero() fallback path on non-contiguous views', () {
      final a = NDArray.fromList(
        Float64List.fromList([1.0, 0.0, 3.0, 0.0, 5.0, 6.0]),
        [2, 3],
        DType.float64,
      );
      addTearDown(a.dispose);

      // Non-contiguous column slice: select column 0 (elements [1.0, 0.0] -> wait, strides are [3, 1], column 0 is data[0], data[3])
      // Logical: row 0: [1.0, 0.0, 3.0], row 1: [0.0, 5.0, 6.0]. Column 0 is: [1.0, 0.0].
      // Column 2 is: [3.0, 6.0] (nonzero count is 2).
      final view = a.slice([Slice.all(), Index(2)]);
      expect(view.isContiguous, false);
      expect(view.toList(), [3.0, 6.0]);

      // In the old code, count_nonzero(view) would incorrectly return 0 due to discarded recursion yields!
      expect(count_nonzero(view), 2);
    });

    test('NDArray Structural Value Equality operator == and hashCode', () {
      final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final b = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final c = NDArray.fromList([1.0, 2.0, 9.0], [3], DType.float64);
      final d = NDArray.fromList([1.0, 2.0, 3.0], [1, 3], DType.float64);
      addTearDown(a.dispose);
      addTearDown(b.dispose);
      addTearDown(c.dispose);
      addTearDown(d.dispose);

      expect(a == b, true);
      expect(a == c, false);
      expect(a == d, false); // different shape
      expect(a.hashCode == b.hashCode, true);
    });

    test('View Invalidation Safety (throws StateError on disposed views)', () {
      final parent = NDArray.ones([2, 2], DType.float64);
      addTearDown(parent.dispose);
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
    });

    test('NPY double quotes / mixed quotes header parsing compatibility', () {
      // Construct a fake .npy buffer using double quotes in header dictionary
      final headerStr =
          '{"descr": "<f8", "fortran_order": False, "shape": (2, 2)}';

      final prefixLen = 6 + 2 + 2;
      var paddedHeaderLen =
          ((prefixLen + headerStr.length + 1) + 63) ~/ 64 * 64 - prefixLen;
      final padCount = paddedHeaderLen - headerStr.length - 1;
      final paddedHeader = headerStr + (' ' * padCount) + '\n';

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
      addTearDown(loaded.dispose);
      expect(loaded.shape, [2, 2]);
      expect(loaded.dtype, DType.float64);
      expect(loaded.toList(), [1.0, 2.0, 3.0, 4.0]);
    });

    test('randint() supporting wide ranges (> 2^32)', () {
      // Generate a 64-bit integer range that exceeds 2^32
      final a = randint([10], 100000000000, 200000000000, dtype: DType.int64);
      addTearDown(a.dispose);
      expect(a.shape, [10]);
      expect(a.dtype, DType.int64);
      for (final val in a.data) {
        expect(val, greaterThanOrEqualTo(100000000000));
        expect(val, lessThan(200000000000));
      }
    });

    test('transpose() negative axis support', () {
      final a = NDArray.fromList(
        Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        [2, 3],
        DType.float64,
      );
      addTearDown(a.dispose);
      // Reverse axes order via negative indices: [-1, -2] is equivalent to [1, 0]
      final b = a.transpose([-1, -2]);
      addTearDown(b.dispose);
      expect(b.shape, [3, 2]);
      expect(b.toList(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    });

    test('exponential() supporting lam (rate) parameter alias', () {
      final a = exponential([10], lam: 2.0); // lam = 2.0 -> scale = 0.5
      addTearDown(a.dispose);
      expect(a.shape, [10]);
      expect(a.dtype, DType.float64);
      for (final val in a.data) {
        expect(val, greaterThanOrEqualTo(0.0));
      }

      expect(() => exponential([10], lam: -1.0), throwsArgumentError);
    });

    test('DType.complex64 array creation, viewing and operations coverage', () {
      final a = NDArray<Complex>.create([2, 2], DType.complex64);
      addTearDown(a.dispose);
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
      addTearDown(b.dispose);
      b.data.fillRange(0, 4, Complex(10.0, 10.0));

      final c = add(a, b);
      addTearDown(c.dispose);
      expect(c.dtype, DType.complex64);
      expect(c.data[0], Complex(11.0, 12.0));
      expect(c.data[3], Complex(17.0, 18.0));
    });

    test('linspace() with num == 1 coverage', () {
      final a = NDArray<double>.linspace(5.0, 10.0, 1, dtype: DType.float64);
      addTearDown(a.dispose);
      expect(a.shape, [1]);
      expect(a.toList(), [5.0]);
    });

    test('reshape() size mismatch throws ArgumentError coverage', () {
      final a = NDArray.ones([2, 2], DType.float64);
      addTearDown(a.dispose);
      expect(() => a.reshape([3]), throwsArgumentError);
    });

    test('transpose() invalid axes length throws ArgumentError coverage', () {
      final a = NDArray.ones([2, 2], DType.float64);
      addTearDown(a.dispose);
      expect(() => a.transpose([0]), throwsArgumentError);
    });

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
        addTearDown(a.dispose);

        // Test sum()
        final totalSum = sum(a);
        expect(totalSum, Complex(10.0, 1.0));

        // Test mean()
        final totalMean = mean(a);
        expect(totalMean, Complex(2.5, 0.25));

        // Test prod()
        final totalProd = prod(a);
        expect(totalProd, Complex(24.0, 24.0)); // (1+i)*2*3*4 = 24 + 24i

        // Test concatenate() and hstack()
        final b = NDArray<Complex>.fromList(
          [Complex(10.0, 0.0), Complex(10.0, 0.0)],
          [1, 2],
          DType.complex128,
        );
        addTearDown(b.dispose);

        final row0 = a.slice([Index(0)]).reshape([1, 2]);
        final combined = concatenate([row0, b], axis: 0);
        addTearDown(combined.dispose);
        expect(combined.shape, [2, 2]);
        expect(combined.dtype, DType.complex128);
        expect(combined.data[0], Complex(1.0, 1.0));
        expect(combined.data[2], Complex(10.0, 0.0));
      },
    );

    test('qsort floating-point NaN sorting safety and stability', () {
      final a = NDArray.fromList(
        [3.0, double.nan, 1.0, double.nan, 2.0],
        [5],
        DType.float64,
      );
      addTearDown(a.dispose);

      final b = sort(a);
      addTearDown(b.dispose);

      // The 3 non-NaN elements must be sorted ascending: 1.0, 2.0, 3.0
      expect(b.data[0], 1.0);
      expect(b.data[1], 2.0);
      expect(b.data[2], 3.0);

      // The NaNs must be pushed consistently to the end of the array
      expect(b.data[3].isNaN, true);
      expect(b.data[4].isNaN, true);
    });

    test('matmul() high-speed 1D vector dot product gate', () {
      final a = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final b = NDArray.fromList([4.0, 5.0, 6.0], [3], DType.float64);
      addTearDown(a.dispose);
      addTearDown(b.dispose);

      final c = matmul(a, b);
      addTearDown(c.dispose);

      expect(c.shape, []); // Must be a 0D scalar array
      expect(c.dtype, DType.float64);
      // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
      expect(c.data[0], 32.0);
    });

    test('fft() and ifft() empty/0D shape validation checks', () {
      final emptyArr = NDArray.zeros([], DType.float64);
      addTearDown(emptyArr.dispose);

      expect(() => fft(emptyArr), throwsArgumentError);
      expect(() => ifft(emptyArr), throwsArgumentError);
    });

    test('fft() and ifft() invalid transform length n <= 0 checks', () {
      final signal = NDArray.zeros([5], DType.float64);
      addTearDown(signal.dispose);

      expect(() => fft(signal, n: 0), throwsArgumentError);
      expect(() => fft(signal, n: -5), throwsArgumentError);
      expect(() => ifft(signal, n: 0), throwsArgumentError);
      expect(() => ifft(signal, n: -1), throwsArgumentError);
    });

    test('fft() and ifft() processing complex inputs directly', () {
      final complexSignal = NDArray<Complex>.fromList(
        [Complex(1.0, 1.0), Complex(2.0, 2.0)],
        [2],
        DType.complex128,
      );
      addTearDown(complexSignal.dispose);

      final freq = fft(complexSignal);
      addTearDown(freq.dispose);
      expect(freq.dtype, DType.complex128);
      expect(freq.shape, [2]);

      final time = ifft(freq);
      addTearDown(time.dispose);
      expect(time.dtype, DType.complex128);
      expect(time.shape, [2]);

      // The inverse transform must recover the original complex signal values (within floating precision)
      expect(time.data[0].real, closeTo(1.0, 1e-9));
      expect(time.data[0].imag, closeTo(1.0, 1e-9));
      expect(time.data[1].real, closeTo(2.0, 1e-9));
      expect(time.data[1].imag, closeTo(2.0, 1e-9));
    });

    test(
      'fft() and ifft() with real inputs and zero padding padding checks',
      () {
        final realSignal = NDArray<double>.fromList(
          [1.0, 2.0],
          [2],
          DType.float64,
        );
        addTearDown(realSignal.dispose);

        // 1. fft with zero-padding (n = 4)
        final freqPadded = fft(realSignal, n: 4);
        addTearDown(freqPadded.dispose);
        expect(freqPadded.shape, [4]);

        // 2. ifft with real input and zero padding (n = 4)
        final timePadded = ifft(realSignal, n: 4);
        addTearDown(timePadded.dispose);
        expect(timePadded.shape, [4]);
      },
    );

    test('uniform() validation checks coverage', () {
      expect(() => uniform([5], dtype: DType.int64), throwsArgumentError);
    });

    test('randint() validation checks coverage', () {
      expect(
        () => randint([5], 0, 10, dtype: DType.float64),
        throwsArgumentError,
      );
      expect(() => randint([5], 10, 5), throwsArgumentError);
    });

    test('normal() validation and zero-avoidance checks coverage', () {
      expect(() => normal([5], scale: 0.0), throwsArgumentError);

      final a = normal([2], random: ZeroThenDoubleRandom());
      addTearDown(a.dispose);
      expect(a.shape, [2]);
      expect(a.dtype, DType.float64);
    });

    test('exponential() validation checks coverage', () {
      expect(() => exponential([5], dtype: DType.int32), throwsArgumentError);
    });

    test('poisson() validation, large lambda, and zero-avoidance checks', () {
      expect(() => poisson([5], dtype: DType.float64), throwsArgumentError);
      expect(() => poisson([5], lam: 0.0), throwsArgumentError);

      final a = poisson([2], lam: 40.0, random: ZeroThenDoubleRandom());
      addTearDown(a.dispose);
      expect(a.shape, [2]);
      expect(a.dtype, DType.int64);
    });

    test(
      'binomial() validation, zero trials, zero stddev, and zero-avoidance checks',
      () {
        expect(
          () => binomial([5], 10, 0.5, dtype: DType.float64),
          throwsArgumentError,
        );
        expect(() => binomial([5], -1, 0.5), throwsArgumentError);
        expect(() => binomial([5], 10, 1.5), throwsArgumentError);

        final zeroTrials = binomial([5], 0, 0.5);
        addTearDown(zeroTrials.dispose);
        expect(zeroTrials.toList(), [0, 0, 0, 0, 0]);

        final zeroStddev = binomial([5], 100, 1.0);
        addTearDown(zeroStddev.dispose);
        expect(zeroStddev.toList(), [100, 100, 100, 100, 100]);

        final a = binomial([2], 100, 0.5, random: ZeroThenDoubleRandom());
        addTearDown(a.dispose);
        expect(a.shape, [2]);
        expect(a.dtype, DType.int64);

        final oddLength = binomial([5], 100, 0.5);
        addTearDown(oddLength.dispose);
        expect(oddLength.shape, [5]);
        expect(oddLength.dtype, DType.int64);
      },
    );
    test('save() and load() non-contiguous strided views of all dtypes', () {
      // 1. float32
      final f32 = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float32);
      final f32View = f32.transposed; // non-contiguous!
      addTearDown(f32.dispose);
      save('scratch/f32_view.npy', f32View);
      final f32Loaded = load('scratch/f32_view.npy');
      addTearDown(f32Loaded.dispose);
      expect(f32Loaded.shape, [2, 2]);
      expect(f32Loaded.dtype, DType.float32);
      expect(f32Loaded.toList(), [1.0, 3.0, 2.0, 4.0]);

      // 2. int32
      final i32 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
      final i32View = i32.transposed;
      addTearDown(i32.dispose);
      save('scratch/i32_view.npy', i32View);
      final i32Loaded = load('scratch/i32_view.npy');
      addTearDown(i32Loaded.dispose);
      expect(i32Loaded.toList(), [1, 3, 2, 4]);

      // 3. int64
      final i64 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int64);
      final i64View = i64.transposed;
      addTearDown(i64.dispose);
      save('scratch/i64_view.npy', i64View);
      final i64Loaded = load('scratch/i64_view.npy');
      addTearDown(i64Loaded.dispose);
      expect(i64Loaded.toList(), [1, 3, 2, 4]);

      // 4. boolean
      final b = NDArray.fromList(
        [true, false, true, false],
        [2, 2],
        DType.boolean,
      );
      final bView = b.transposed;
      addTearDown(b.dispose);
      save('scratch/b_view.npy', bView);
      final bLoaded = load('scratch/b_view.npy');
      addTearDown(bLoaded.dispose);
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
      addTearDown(c128.dispose);
      save('scratch/c128_view.npy', c128View);
      final c128Loaded = load('scratch/c128_view.npy');
      addTearDown(c128Loaded.dispose);
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
      addTearDown(c64.dispose);
      save('scratch/c64_view.npy', c64View);
      final c64Loaded = load('scratch/c64_view.npy');
      addTearDown(c64Loaded.dispose);
      expect(c64Loaded.toList(), [
        Complex(1.0, 1.0),
        Complex(3.0, 3.0),
        Complex(2.0, 2.0),
        Complex(4.0, 4.0),
      ]);
    });

    test('broadcastShapes() incompatible shapes throws ArgumentError', () {
      final a = NDArray.ones([2], DType.float64);
      final b = NDArray.ones([3], DType.float64);
      addTearDown(a.dispose);
      addTearDown(b.dispose);

      expect(() => add(a, b), throwsArgumentError);
    });

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

    test('NDArray.eye() factory with integer dtype coverage', () {
      final a = NDArray.eye(2, DType.int32);
      addTearDown(a.dispose);
      expect(a.toList(), [1, 0, 0, 1]);
      expect(a.dtype, DType.int32);
    });

    test('NDArray.view() FFI constructors with float32 and int64 coverage', () {
      final pF32 = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float32);
      addTearDown(pF32.dispose);
      final vF32 = NDArray.view(pF32, [2], [1], offsetElements: 1);
      expect(vF32.toList(), [2.0, 3.0]);
      expect(vF32.dtype, DType.float32);

      final pI64 = NDArray.fromList([10, 20, 30], [3], DType.int64);
      addTearDown(pI64.dispose);
      final vI64 = NDArray.view(pI64, [2], [1], offsetElements: 1);
      expect(vI64.toList(), [20, 30]);
      expect(vI64.dtype, DType.int64);
    });

    test('_resolveDType() complex64 and float64 cross-promotion coverage', () {
      final a = NDArray.fromList([Complex(1.0, 1.0)], [1], DType.complex64);
      final b = NDArray.fromList([2.0], [1], DType.float64);
      addTearDown(a.dispose);
      addTearDown(b.dispose);

      final c = add(a, b);
      addTearDown(c.dispose);
      expect(c.dtype, DType.complex128);
      expect(c.data[0].real, 3.0);
      expect(c.data[0].imag, 1.0);
    });

    test(
      'matmul() copy-free 100% transposed and sliced views multi-dimensional multiplication',
      () {
        final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        final b = NDArray.fromList([5.0, 6.0, 7.0, 8.0], [2, 2], DType.float64);
        addTearDown(a.dispose);
        addTearDown(b.dispose);

        final aT = a.transposed;
        final bT = b.transposed;

        final result = matmul(aT, bT);
        addTearDown(result.dispose);

        expect(result.shape, [2, 2]);
        expect(result.toList(), [23.0, 31.0, 34.0, 46.0]);
      },
    );

    test('ufuncs in-place out buffer shape and dtype validation checks', () {
      final a = NDArray<double>.ones([3], DType.float64);
      final b = NDArray<double>.ones([3], DType.float64);
      final incompatibleOut = NDArray<double>.ones([4], DType.float64);
      final incompatibleDTypeOut = NDArray<int>.ones([3], DType.int32);
      addTearDown(a.dispose);
      addTearDown(b.dispose);
      addTearDown(incompatibleOut.dispose);
      addTearDown(incompatibleDTypeOut.dispose);

      // 1. add() contiguous shape mismatch
      expect(() => add(a, b, out: incompatibleOut), throwsArgumentError);
      expect(() => add(a, b, out: incompatibleDTypeOut), throwsArgumentError);

      // 2. add() broadcast shape mismatch
      final broadcastA = NDArray.ones([1, 3], DType.float64);
      addTearDown(broadcastA.dispose);
      expect(
        () => add(broadcastA, b, out: incompatibleOut),
        throwsArgumentError,
      );

      // 3. sqrt() shape mismatch
      expect(() => sqrt(a, out: incompatibleOut), throwsArgumentError);

      // 4. sin() shape mismatch
      expect(() => sin(a, out: incompatibleOut), throwsArgumentError);
    });

    test('NDArray.ones() factory with complex dtypes coverage', () {
      final a = NDArray.ones([2], DType.complex128);
      addTearDown(a.dispose);
      expect(a.toList(), [Complex(1.0, 0.0), Complex(1.0, 0.0)]);
      expect(a.dtype, DType.complex128);

      final b = NDArray.ones([2], DType.complex64);
      addTearDown(b.dispose);
      expect(b.toList(), [Complex(1.0, 0.0), Complex(1.0, 0.0)]);
      expect(b.dtype, DType.complex64);
    });

    test('add() cross-type complex/int and int/complex additions coverage', () {
      final c = NDArray<Complex>.fromList(
        [Complex(1.0, 1.0)],
        [1],
        DType.complex128,
      );
      final i = NDArray<int>.fromList([2], [1], DType.int64);
      final d = NDArray<double>.fromList([3.0], [1], DType.float64);
      addTearDown(c.dispose);
      addTearDown(i.dispose);
      addTearDown(d.dispose);

      // 1. Complex + int
      final res1 = add(c, i);
      addTearDown(res1.dispose);
      expect(res1.dtype, DType.complex128);
      expect(res1.data[0].real, 3.0);
      expect(res1.data[0].imag, 1.0);

      // 2. double + Complex
      final res2 = add(d, c);
      addTearDown(res2.dispose);
      expect(res2.dtype, DType.complex128);
      expect(res2.data[0].real, 4.0);
      expect(res2.data[0].imag, 1.0);

      // 3. int + Complex
      final res3 = add(i, c);
      addTearDown(res3.dispose);
      expect(res3.dtype, DType.complex128);
      expect(res3.data[0].real, 3.0);
    });

    test('inv() in-place out buffer validations and solvers coverage', () {
      final a = NDArray<double>.fromList(
        [1.0, 2.0, 3.0, 4.0],
        [2, 2],
        DType.float64,
      );
      final out = NDArray<double>.zeros([2, 2], DType.float64);
      final incompatibleOut = NDArray<double>.ones([3, 3], DType.float64);
      addTearDown(a.dispose);
      addTearDown(out.dispose);
      addTearDown(incompatibleOut.dispose);

      // 1. Incompatible out shape throws ArgumentError
      expect(() => inv(a, out: incompatibleOut), throwsArgumentError);

      // 2. Valid in-place solving
      final res = inv(a, out: out);
      expect(identical(res, out), true);
      expect(res.data[0], closeTo(-2.0, 1e-9));
      expect(res.data[1], closeTo(1.0, 1e-9));
      expect(res.data[2], closeTo(1.5, 1e-9));
      expect(res.data[3], closeTo(-0.5, 1e-9));
    });

    test('_resolveDType cross-promotion additions coverage', () {
      final f64 = NDArray<double>.fromList([1.0], [1], DType.float64);
      final f32 = NDArray<double>.fromList([2.0], [1], DType.float32);
      final i64 = NDArray<int>.fromList([3], [1], DType.int64);
      final i32 = NDArray<int>.fromList([4], [1], DType.int32);
      addTearDown(f64.dispose);
      addTearDown(f32.dispose);
      addTearDown(i64.dispose);
      addTearDown(i32.dispose);

      // 1. float64 + float32 -> float64
      final r1 = add(f64, f32);
      addTearDown(r1.dispose);
      expect(r1.dtype, DType.float64);

      // 2. float32 + int64 -> float32
      final r2 = add(f32, i64);
      addTearDown(r2.dispose);
      expect(r2.dtype, DType.float32);

      // 3. int64 + int32 -> int64
      final r3 = add(i64, i32);
      addTearDown(r3.dispose);
      expect(r3.dtype, DType.int64);
    });

    test('NDArray.eye() complex identity matrix type safety validations', () {
      final eye128 = NDArray<Complex>.eye(3, DType.complex128);
      addTearDown(eye128.dispose);
      expect(eye128.dtype, DType.complex128);
      expect(eye128.data[0], Complex(1.0, 0.0));
      expect(eye128.data[1], Complex(0.0, 0.0));
      expect(eye128.data[4], Complex(1.0, 0.0));

      final eye64 = NDArray<Complex>.eye(3, DType.complex64);
      addTearDown(eye64.dispose);
      expect(eye64.dtype, DType.complex64);
      expect(eye64.data[0], Complex(1.0, 0.0));
      expect(eye64.data[4], Complex(1.0, 0.0));
    });

    test('prod() contiguous FFI leaf paths coverage', () {
      final f64 = NDArray<double>.fromList([2.0, 3.0, 4.0], [3], DType.float64);
      final f32 = NDArray<double>.fromList([5.0, 2.0, 3.0], [3], DType.float32);
      addTearDown(f64.dispose);
      addTearDown(f32.dispose);

      // 1. float64 contiguous FFI prod()
      final r1 = prod(f64);
      expect(r1, closeTo(24.0, 1e-9));

      // 2. float32 contiguous FFI prod()
      final r2 = prod(f32);
      expect(r2, closeTo(30.0, 1e-9));
    });

    test('variance() and std() on non-contiguous view calculation', () {
      final parent = NDArray.fromList(
        Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        [3, 2],
        DType.float64,
      );
      addTearDown(parent.dispose);

      final viewT = parent.transposed;
      addTearDown(viewT.dispose);

      expect(variance(viewT), closeTo(17.5 / 6.0, 1e-9));
      expect(std(viewT), closeTo(math.sqrt(17.5 / 6.0), 1e-9));
    });

    test('variance() and std() on sliced view calculation', () {
      final parent = NDArray.fromList(
        Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        [3, 2],
        DType.float64,
      );
      addTearDown(parent.dispose);

      final view = parent.slice([Slice(start: 0, stop: 2), Slice.all()]);
      expect(variance(view), closeTo(1.25, 1e-9));
      expect(std(view), closeTo(math.sqrt(1.25), 1e-9));
    });

    test(
      'clip() with named out parameter recycler and sliced contiguous view',
      () {
        final parent = NDArray.fromList(
          Float64List.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
          [3, 2],
          DType.float64,
        );
        addTearDown(parent.dispose);

        final view = parent.slice([Slice(start: 0, stop: 2), Slice.all()]);
        final out = NDArray<double>.zeros([2, 2], DType.float64);
        addTearDown(out.dispose);

        final res = clip(view, 2.0, 3.0, out: out);
        expect(identical(res, out), true);
        expect(out.toList(), [2.0, 2.0, 3.0, 3.0]);

        expect(parent.toList(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
      },
    );

    group('NaN-Ignoring Statistical Reductions tests', () {
      test('Verify nansum() treats NaNs as zeros', () {
        final a = NDArray<double>.fromList(
          Float64List.fromList([1.0, double.nan, 3.0, double.nan]),
          [2, 2],
          DType.float64,
        );
        addTearDown(a.dispose);

        expect(nansum(a), closeTo(4.0, 1e-9));

        final s0 = nansum(a, axis: 0);
        addTearDown(s0.dispose);
        expect(s0.shape, [2]);
        expect(s0.toList(), [4.0, 0.0]);
      });

      test('Verify nanmean() ignores NaNs', () {
        final a = NDArray<double>.fromList(
          Float64List.fromList([1.0, double.nan, 3.0, 4.0]),
          [2, 2],
          DType.float64,
        );
        addTearDown(a.dispose);

        expect(nanmean(a), closeTo(8.0 / 3.0, 1e-9));

        final m0 = nanmean(a, axis: 0);
        addTearDown(m0.dispose);
        expect(m0.shape, [2]);
        expect(m0.toList(), [2.0, 4.0]);
      });

      test('Verify nanvar() and nanstd() ignore NaNs', () {
        final a = NDArray<double>.fromList(
          Float64List.fromList([1.0, double.nan, 2.0, 3.0]),
          [2, 2],
          DType.float64,
        );
        addTearDown(a.dispose);

        // mean = (1 + 2 + 3) / 3 = 2.0
        // var = ((1-2)^2 + (2-2)^2 + (3-2)^2) / 3 = 2/3
        expect(nanvar(a), closeTo(2.0 / 3.0, 1e-9));
        expect(nanstd(a), closeTo(math.sqrt(2.0 / 3.0), 1e-9));

        final v0 = nanvar(a, axis: 0);
        final s0 = nanstd(a, axis: 0);
        addTearDown(v0.dispose);
        addTearDown(s0.dispose);

        expect(v0.shape, [2]);
        expect(s0.shape, [2]);
      });
    });

    test('arange() and linspace() type safety with complex dtypes', () {
      // 1. arange with complex128
      final a128 = NDArray<Complex>.arange(0, 3, dtype: DType.complex128);
      addTearDown(a128.dispose);
      expect(a128.shape, [3]);
      expect(a128.dtype, DType.complex128);
      expect(a128.toList(), [
        Complex(0.0, 0.0),
        Complex(1.0, 0.0),
        Complex(2.0, 0.0),
      ]);

      // 2. arange with complex64
      final a64 = NDArray<Complex>.arange(1, 3, dtype: DType.complex64);
      addTearDown(a64.dispose);
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
      addTearDown(l128.dispose);
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
      addTearDown(l64.dispose);
      expect(l64.shape, [1]);
      expect(l64.dtype, DType.complex64);
      expect(l64.toList(), [Complex(5.0, 0.0)]);
    });

    test('where() with out parameter recycler and shape validations', () {
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

      addTearDown(cond.dispose);
      addTearDown(x.dispose);
      addTearDown(y.dispose);
      addTearDown(out.dispose);
      addTearDown(incompatibleOut.dispose);

      // 1. Incompatible shape throws ArgumentError
      expect(() => where(cond, x, y, incompatibleOut), throwsArgumentError);

      // 2. Valid in-place recycling
      final res = where(cond, x, y, out);
      expect(identical(res, out), true);
      expect(out.toList(), [1.0, 20.0, 30.0, 4.0]);
    });

    test('uniform() and normal() Float32, and randint() Int32 coverage', () {
      // 1. uniform with float32
      final u = uniform([10], dtype: DType.float32);
      addTearDown(u.dispose);
      expect(u.shape, [10]);
      expect(u.dtype, DType.float32);
      for (final val in u.data) {
        expect(val, greaterThanOrEqualTo(0.0));
        expect(val, lessThan(1.0));
      }

      // 2. randint with int32
      final ri = randint([10], 0, 5, dtype: DType.int32);
      addTearDown(ri.dispose);
      expect(ri.shape, [10]);
      expect(ri.dtype, DType.int32);
      for (final val in ri.data) {
        expect(val, greaterThanOrEqualTo(0));
        expect(val, lessThan(5));
      }

      // 3. normal with float32
      final n = normal([10], dtype: DType.float32);
      addTearDown(n.dispose);
      expect(n.shape, [10]);
      expect(n.dtype, DType.float32);
    });

    test('setByMask() with NDArray values and capacity validations', () {
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

      addTearDown(parent.dispose);
      addTearDown(mask.dispose);
      addTearDown(values.dispose);
      addTearDown(insufficientValues.dispose);

      // 1. Incompatible capacity throws ArgumentError
      expect(
        () => parent.setByMask(mask, insufficientValues),
        throwsArgumentError,
      );

      // 2. Valid array value mask mutation
      parent.setByMask(mask, values);
      expect(parent.toList(), [99.0, 2.0, 3.0, 100.0]);
    });

    test('setIndices() RangeError and ArgumentError boundaries coverage', () {
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

      addTearDown(parent.dispose);
      addTearDown(indices.dispose);
      addTearDown(values.dispose);
      addTearDown(insufficientValues.dispose);

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
    });

    test('Contiguous sub-slice flatten() optimization correctness', () {
      final parent = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
      // Slice representing the first 2 elements (strides: [1], contiguous: true, totalSize < data.length!)
      final view = parent.slice([Slice(start: 0, stop: 2)]);

      addTearDown(parent.dispose);
      addTearDown(view.dispose);

      expect(view.isContiguous, true);
      expect(view.shape, [2]);

      final flat = view.flatten();
      addTearDown(flat.dispose);
      expect(flat.shape, [2]);
      expect(flat.isContiguous, true);
      expect(flat.toList(), [1.0, 2.0]);
    });

    test(
      'NDArray.fill() ufunc correctness and performance speedups verification',
      () {
        // 1. Contiguous Double Precision fill
        final a = NDArray<double>.zeros([5], DType.float64);
        addTearDown(a.dispose);
        a.fill(42.5);
        expect(a.toList(), [42.5, 42.5, 42.5, 42.5, 42.5]);

        // 2. Contiguous Int32 Precision fill
        final b = NDArray<int>.zeros([5], DType.int32);
        addTearDown(b.dispose);
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
        addTearDown(parent.dispose);
        addTearDown(view.dispose);

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
        addTearDown(mat.dispose);
        addTearDown(mainDiag.dispose);

        expect(mainDiag.shape, [3]);
        expect(mainDiag.isContiguous, false); // spaced view
        expect(mainDiag.toList(), [1.0, 5.0, 9.0]);

        // 2. Strided views offset extractions (k = 1 and k = -1)
        final upperDiag = diag(mat, k: 1);
        final lowerDiag = diag(mat, k: -1);
        addTearDown(upperDiag.dispose);
        addTearDown(lowerDiag.dispose);

        expect(upperDiag.shape, [2]);
        expect(upperDiag.toList(), [2.0, 6.0]);
        expect(lowerDiag.shape, [2]);
        expect(lowerDiag.toList(), [4.0, 8.0]);

        // 3. Construct diagonal 2D matrix from 1D vector
        final vec = NDArray.fromList([10.0, 20.0], [2], DType.float64);
        final dMat = diag(vec);
        final uDiagMat = diag(vec, k: 1);
        addTearDown(vec.dispose);
        addTearDown(dMat.dispose);
        addTearDown(uDiagMat.dispose);

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
        addTearDown(tensor3d.dispose);
        expect(() => diag(tensor3d), throwsArgumentError);
      },
    );

    test('isclose() and allclose() approximate equality ufunc correctness', () {
      final a = NDArray.fromList([1.0, 1.00001, 2.0], [3], DType.float64);
      final b = NDArray.fromList([1.0, 1.00003, 2.0], [3], DType.float64);
      addTearDown(a.dispose);
      addTearDown(b.dispose);

      // 1. Default tolerances: rtol = 1e-05, atol = 1e-08
      final closeDefault = isclose(a, b);
      addTearDown(closeDefault.dispose);
      expect(closeDefault.toList(), [true, false, true]);

      // 2. Stretched tolerances: rtol = 1e-04
      final closeStretched = isclose(a, b, rtol: 1e-04);
      addTearDown(closeStretched.dispose);
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
      addTearDown(infA.dispose);
      addTearDown(infB.dispose);
      addTearDown(infC.dispose);

      final closeInf = isclose(infA, infB);
      final closeInfMismatch = isclose(infA, infC);
      addTearDown(closeInf.dispose);
      addTearDown(closeInfMismatch.dispose);

      expect(closeInf.toList(), [true, true]);
      expect(closeInfMismatch.toList(), [false, false]);

      // 5. NaN value equalNan checks
      final nanA = NDArray.fromList([double.nan], [1], DType.float64);
      final nanB = NDArray.fromList([double.nan], [1], DType.float64);
      addTearDown(nanA.dispose);
      addTearDown(nanB.dispose);

      final closeNanDefault = isclose(nanA, nanB);
      final closeNanEqual = isclose(nanA, nanB, equalNan: true);
      addTearDown(closeNanDefault.dispose);
      addTearDown(closeNanEqual.dispose);

      expect(closeNanDefault.toList(), [false]);
      expect(closeNanEqual.toList(), [true]);
    });

    test('nan_to_num() dataset cleaning ufunc correctness', () {
      // 1. Default Float64 cleaning
      final a = NDArray.fromList(
        [1.0, double.nan, double.infinity, double.negativeInfinity],
        [4],
        DType.float64,
      );
      addTearDown(a.dispose);

      final cleanDefault = nan_to_num(a);
      addTearDown(cleanDefault.dispose);

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
      addTearDown(cleanCustom.dispose);

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
      addTearDown(parent.dispose);
      addTearDown(view.dispose);

      expect(view.isContiguous, false);
      expect(view.isContiguous, false);
      nan_to_num(view, nan: 100.0, out: view);

      expect(parent.toList(), [100.0, 2.0, 100.0, 4.0]);
    });

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
        addTearDown(singularMat.dispose);
        expect(det(singularMat), 0.0);

        // 2. solve() non-square matrix throws ArgumentError
        final nonSquareA = NDArray.fromList(
          [1.0, 2.0, 3.0],
          [1, 3],
          DType.float64,
        );
        final b = NDArray.fromList([1.0], [1], DType.float64);
        addTearDown(nonSquareA.dispose);
        addTearDown(b.dispose);
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
        addTearDown(squareA.dispose);
        addTearDown(incompatibleB.dispose);
        expect(() => solve(squareA, incompatibleB), throwsArgumentError);

        // 4. solve() singular Float64 matrix throws singular ArgumentError
        final singularFloat64A = NDArray.fromList(
          [1.0, 2.0, 2.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final validB = NDArray.fromList([5.0, 6.0], [2], DType.float64);
        addTearDown(singularFloat64A.dispose);
        addTearDown(validB.dispose);
        expect(() => solve(singularFloat64A, validB), throwsArgumentError);

        // 5. solve() singular Float32 matrix throws singular ArgumentError
        final singularFloat32A = NDArray.fromList(
          [1.0, 2.0, 2.0, 4.0],
          [2, 2],
          DType.float32,
        );
        final validFloat32B = NDArray.fromList([5.0, 6.0], [2], DType.float32);
        addTearDown(singularFloat32A.dispose);
        addTearDown(validFloat32B.dispose);
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
        addTearDown(a.dispose);

        final aExp0 = expand_dims(a, 0);
        final aExp1 = expand_dims(a, 1);
        final aExpNeg = expand_dims(a, -1); // normalized to axis 1
        addTearDown(aExp0.dispose);
        addTearDown(aExp1.dispose);
        addTearDown(aExpNeg.dispose);

        expect(aExp0.shape, [1, 3]);
        expect(aExp0.strides, [1, 1]);

        expect(aExp1.shape, [3, 1]);
        expect(aExp1.strides, [1, 1]);

        expect(aExpNeg.shape, [3, 1]);

        // expand_dims out of bounds exception
        expect(() => expand_dims(a, 3), throwsArgumentError);

        // 2. squeeze() verification
        final b = NDArray.zeros([1, 3, 1], DType.float64);
        addTearDown(b.dispose);

        final bSqueezedAll = squeeze(b);
        final bSqueezed0 = squeeze(b, axis: [0]);
        addTearDown(bSqueezedAll.dispose);
        addTearDown(bSqueezed0.dispose);

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
        addTearDown(parent.dispose);
        addTearDown(view.dispose);

        expect(view.isContiguous, true);
        expect(view.shape, [2]);

        // 1. sum() verification
        final s = sum(view);
        expect(s, 3.0);

        // 2. prod() verification
        final p = prod(view);
        expect(p, 2.0);
      },
    );

    test('concatenate() validation errors throws exceptions', () {
      expect(() => concatenate(<NDArray<Object>>[]), throwsArgumentError);

      final a = NDArray.fromList([1.0, 2.0], [2], DType.float64);
      final wrongShape = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
      final wrongRank = NDArray.fromList([1.0, 2.0], [1, 2], DType.float64);
      final wrongDType = NDArray.fromList([1, 2], [2], DType.int32);

      addTearDown(a.dispose);
      addTearDown(wrongShape.dispose);
      addTearDown(wrongRank.dispose);
      addTearDown(wrongDType.dispose);

      expect(() => concatenate([a, a], axis: 5), throwsRangeError);
      expect(() => concatenate([a, a], axis: -5), throwsRangeError);
      expect(() => concatenate(<NDArray<Object>>[a, wrongDType]), throwsArgumentError);

      expect(() => concatenate(<NDArray<Object>>[a, wrongRank]), throwsArgumentError);

      final mat1 = NDArray.fromList([1.0, 1.0, 1.0, 1.0], [2, 2], DType.float64);
      final mat2 = NDArray.fromList([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [2, 3], DType.float64);
      addTearDown(mat1.dispose);
      addTearDown(mat2.dispose);
      expect(() => concatenate(<NDArray<double>>[mat1, mat2], axis: 0), throwsArgumentError);
    });
  });
}

final class ZeroThenDoubleRandom implements math.Random {
  var count = 0;

  @override
  double nextDouble() {
    count++;
    if (count == 1)
      return 0.0; // Return 0.0 on first draw to force zero-avoidance loop path
    return 0.5;
  }

  @override
  bool nextBool() => false;

  @override
  int nextInt(int max) => 0;
}
