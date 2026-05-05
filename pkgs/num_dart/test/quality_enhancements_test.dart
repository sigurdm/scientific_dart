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
      expect(res3.data[0].imag, 1.0);
    });
  });
}

class ZeroThenDoubleRandom implements math.Random {
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
