import 'dart:io';
import 'dart:typed_data';
import 'package:archive/archive.dart';
import 'package:ndarray/ndarray.dart';
import 'dart:async' show Zone;
import 'package:test/test.dart';

void main() {
  group('Workstream 4 - DX & Core APIs (M2)', () {
    test('M2: astype() converts dtypes and respects copy parameter', () {
      final a = NDArray.fromList([1, 2, 3], [3], DType.int32);

      // copy: false with matching dtype returns the identical instance
      final same = a.astype(DType.int32, copy: false);
      expect(identical(same, a), isTrue);

      // copy: true with matching dtype returns a new copy
      final copied = a.astype(DType.int32, copy: true);
      expect(identical(copied, a), isFalse);
      expect(copied.dtype, DType.int32);
      expect(copied.toList(), [1, 2, 3]);

      // Convert Int32 -> Float64
      final asFloat = a.astype(DType.float64);
      expect(asFloat.dtype, DType.float64);
      expect(asFloat.toList(), [1.0, 2.0, 3.0]);

      // Convert Float64 -> Int32 (truncation)
      final f = NDArray.fromList([1.7, 2.4, -3.9], [3], DType.float64);
      final truncated = f.astype(DType.int32);
      expect(truncated.dtype, DType.int32);
      expect(truncated.toList(), [1, 2, -3]);

      // Convert Int64 -> Boolean (non-zero -> true)
      final ints = NDArray.fromList([0, 5, -1, 0], [4], DType.int64);
      final bools = ints.astype(DType.boolean);
      expect(bools.dtype, DType.boolean);
      expect(bools.toList(), [false, true, true, false]);

      // Convert Boolean -> Int32 (true -> 1, false -> 0)
      final boolInts = bools.astype(DType.int32);
      expect(boolInts.dtype, DType.int32);
      expect(boolInts.toList(), [0, 1, 1, 0]);

      // Top-level astype function
      final topLevel = astype(a, DType.float32);
      expect(topLevel.dtype, DType.float32);
      expect(topLevel.toList(), [1.0, 2.0, 3.0]);

      // Disposed array throws StateError
      copied.dispose();
      expect(() => copied.astype(DType.float64), throwsStateError);
      expect(() => astype(copied, DType.float64), throwsStateError);
    });

    test('M2: toString() formats 0D, 1D, 2D, and large truncated arrays', () {
      // 0-D scalars
      final sInt = NDArray.scalar(42, dtype: DType.int32);
      expect(sInt.toString(), '42, dtype=int32');

      final sFloat = NDArray.scalar(3.14, dtype: DType.float64);
      expect(sFloat.toString(), '3.14, dtype=float64');

      final sBool = NDArray.scalar(true, dtype: DType.boolean);
      expect(sBool.toString(), 'true, dtype=boolean');

      // 1-D small float64 (omits dtype=float64 per NumPy conventions)
      final aF64 = NDArray.fromList([1.0, 2.0, 3.5], [3], DType.float64);
      expect(aF64.toString(), '[1., 2., 3.5]');

      // 1-D small integer (includes dtype)
      final aI32 = NDArray.fromList([1, 2, 3], [3], DType.int32);
      expect(aI32.toString(), '[1, 2, 3], dtype=int32');

      // 1-D large array truncation (threshold > 6 elements)
      final large1D = NDArray.arange(0.0, 10.0);
      expect(large1D.toString(), '[0., 1., 2., ..., 7., 8., 9.]');

      // 2-D matrix formatting with column alignment
      final mat = NDArray.arange(0.0, 12.0).reshape([3, 4]);
      final matStr = mat.toString();
      expect(matStr, contains('[[0., 1.,  2.,  3.]'));
      expect(matStr, contains('[4., 5.,  6.,  7.]'));
      expect(matStr, contains('[8., 9., 10., 11.]]'));

      // 2-D large matrix truncation
      final large2D = NDArray.arange(0.0, 100.0).reshape([10, 10]);
      final large2DStr = large2D.toString();
      expect(large2DStr, contains('...'));
      expect(large2DStr, contains('[[ 0.,  1.,  2., ...,  7.,  8.,  9.]'));
      expect(large2DStr, contains('[90., 91., 92., ..., 97., 98., 99.]]'));

      // Empty arrays
      final empty1D = NDArray<Float64>.create([0], DType.float64);
      expect(empty1D.toString(), '[]');

      final empty2D = NDArray<Int32>.create([0, 3], DType.int32);
      expect(empty2D.toString(), '[], shape=[0, 3], dtype=int32');

      // Disposed array
      empty1D.dispose();
      expect(empty1D.toString(), '<disposed NDArray<float64>>');
    });
  });

  group('Memory, Scope & Stride Safety Regressions', () {
    test(
      'C1: Strided out view in sum does not crash and computes correct values',
      () {
        final a = NDArray.fromList([1, 2, 3, 4, 5, 6], [2, 3], DType.int8);
        final buf = NDArray<Int8>.create([6], DType.int8);
        final out = buf.slice([Slice(step: 2)]); // shape [3], strides [2]
        sum(a, axis: 0, out: out);
        expect(out.toList(), [5, 7, 9]);
        buf.dispose();
        a.dispose();
      },
    );

    test(
      'C4: mean(a, axis:) on empty axis returns NaN (not uninitialized heap)',
      () {
        final a = NDArray<Float64>.create([0, 3], DType.float64);
        final m = mean(a, axis: 0);
        expect(m.shape, [3]);
        for (var i = 0; i < 3; i++) {
          expect(m.getCell([i]).isNaN, isTrue);
        }
        m.dispose();
        a.dispose();
      },
    );

    test(
      'H2: matmul(x, x, out: x) in-place aliasing produces correct result',
      () {
        final x = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
        // [[1, 2], [3, 4]] @ [[1, 2], [3, 4]] = [[7, 10], [15, 22]]
        matmul(x, x, out: x);
        expect(x.toList(), [7.0, 10.0, 15.0, 22.0]);
        x.dispose();
      },
    );

    test(
      'H6: sort and argsort with non-contiguous out view works correctly',
      () {
        final a = NDArray.fromList([3.0, 1.0, 2.0, 0.0], [2, 2], DType.float64);
        final buf = NDArray.zeros([2, 2], DType.float64);
        final out = buf.transposed; // non-contiguous view
        sort(a, out: out);
        expect(out.getCell([0, 0]), 1.0);
        expect(out.getCell([0, 1]), 3.0);
        expect(out.getCell([1, 0]), 0.0);
        expect(out.getCell([1, 1]), 2.0);
        buf.dispose();
        a.dispose();
      },
    );

    test('H7: put_along_axis does not destroy caller array across scopes', () {
      final owned = NDArray.fromList([10, 20, 30], [3], DType.int64);
      final indices = NDArray.fromList([0], [1], DType.int64);
      final values = NDArray.fromList([99], [1], DType.int64);

      NDArray.scope(() {
        put_along_axis(owned, indices, values, 0);
      });

      expect(
        owned.isDisposed,
        isFalse,
        reason: 'Caller array must not be disposed by child scope',
      );
      expect(owned.getCell([0]), 99);
      owned.dispose();
      indices.dispose();
      values.dispose();
    });

    test('H8: nanvar and nanstd can be explicitly disposed without leak', () {
      final a = NDArray.fromList(
        [1.0, 2.0, double.nan, 4.0],
        [4],
        DType.float64,
      );
      final v = nanvar(a);
      expect(v.isDisposed, isFalse);
      v.dispose();
      expect(v.isDisposed, isTrue);

      final s = nanstd(a);
      expect(s.isDisposed, isFalse);
      s.dispose();
      expect(s.isDisposed, isTrue);
      a.dispose();
    });

    test('H14: Allocating inside a completed scope zone throws StateError', () {
      late Zone deadZone;
      NDArray.scope(() {
        deadZone = Zone.current;
      });
      expect(
        () => deadZone.run(() => NDArray.create([2], DType.float64)),
        throwsStateError,
      );
    });
  });

  group('Numerical Correctness & Input Validation Regressions', () {
    test('C2: Array with size > 2^31 elements throws UnsupportedError', () {
      expect(
        () => NDArray<Uint8>.create([2147483648], DType.uint8),
        throwsA(isA<UnsupportedError>()),
      );
    });

    test(
      'C3: Malformed .npz where dataLen does not match shape/dtype throws FormatException',
      () {
        final tempDir = Directory.systemTemp.createTempSync('npz_test_');
        final npzPath = '${tempDir.path}/malformed.npz';
        try {
          final archive = Archive();
          final header =
              "{'descr': '<f8', 'fortran_order': False, 'shape': (1,)}";
          final padCount = 64 - 10 - header.length - 1;
          final padded = "$header${' ' * padCount}\n";
          final bytes = BytesBuilder();
          bytes.add([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 0x01, 0x00]);
          final lenBytes = Uint8List(2);
          ByteData.view(
            lenBytes.buffer,
          ).setUint16(0, padded.length, Endian.little);
          bytes.add(lenBytes);
          bytes.add(padded.codeUnits);
          bytes.add(Uint8List(200)); // 200 bytes payload instead of 8 bytes!

          final fileBytes = bytes.toBytes();
          archive.addFile(ArchiveFile('test.npy', fileBytes.length, fileBytes));
          final zipData = ZipEncoder().encode(archive)!;
          File(npzPath).writeAsBytesSync(zipData);

          expect(() => loadz(npzPath), throwsA(isA<FormatException>()));
        } finally {
          tempDir.deleteSync(recursive: true);
        }
      },
    );

    test('H3: _wrapScalar preserves target dtype for float32 and int8', () {
      final f32 = NDArray.fromList([1.0, 2.0], [2], DType.float32);
      final resF32 = f32 + 1.0;
      expect(resF32.dtype, DType.float32);

      final i8 = NDArray.fromList([10, 20], [2], DType.int8);
      final resI8 = i8 + 1;
      expect(resI8.dtype, DType.int8);

      final i32 = NDArray.fromList([5, 10], [2], DType.int32);
      final resI32 = i32 * 2;
      expect(resI32.dtype, DType.int32);
    });

    test('H4: min, max, argmax, and median propagate NaN correctly', () {
      final a = NDArray.fromList([1.0, double.nan, 3.0], [3], DType.float64);
      expect(min(a).scalar.isNaN, isTrue, reason: 'min must propagate NaN');
      expect(max(a).scalar.isNaN, isTrue, reason: 'max must propagate NaN');
      expect(
        argmax(a),
        1,
        reason: 'argmax must return first NaN index per NumPy',
      );
      expect(
        median(a).scalar.isNaN,
        isTrue,
        reason: 'median must propagate NaN',
      );
      a.dispose();
    });

    test('H5: sum of boolean returns int64 count of trues', () {
      final b = NDArray.fromList([true, true, false, true], [4], DType.boolean);
      final s = sum(b);
      expect(s.dtype, DType.int64);
      expect(s.scalar, 3);
      b.dispose();
    });

    test(
      'H10: searchsorted rejects out-of-bounds sorter indices and wrong dtype',
      () {
        final arr = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
        final v = NDArray.fromList([1.5], [1], DType.float64);

        // Wrong sorter dtype (float64 instead of int32)
        final wrongDType = NDArray.fromList(
          [0.0, 1.0, 2.0],
          [3],
          DType.float64,
        );
        expect(
          () => searchsorted(arr, v, sorter: wrongDType as dynamic),
          throwsArgumentError,
        );

        // Out of bounds index
        final oobSorter = NDArray.fromList([0, 1, 999], [3], DType.int32);
        expect(() => searchsorted(arr, v, sorter: oobSorter), throwsRangeError);

        // Negative index
        final negSorter = NDArray.fromList([-1, 0, 1], [3], DType.int32);
        expect(() => searchsorted(arr, v, sorter: negSorter), throwsRangeError);

        arr.dispose();
        v.dispose();
        wrongDType.dispose();
        oobSorter.dispose();
        negSorter.dispose();
      },
    );
  });
}
