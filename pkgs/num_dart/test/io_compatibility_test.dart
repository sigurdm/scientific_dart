import 'package:num_dart/num_dart.dart';
import 'package:test/test.dart';
import 'dart:io';
import 'dart:typed_data';

void main() {
  group('NDArray NumPy Binary Interoperability & I/O Tests', () {
    setUpAll(() {
      // Ensure scratch directory exists for test file writes
      final dir = Directory('scratch');
      if (!dir.existsSync()) {
        dir.createSync();
      }
    });

    group('.npy Round-Trip Tests', () {
      test('Float64 2D Matrix round-trip', () {
        final a = NDArray.fromList(
          Float64List.fromList([1.5, 2.5, 3.5, 4.5, 5.5, 6.5]),
          [2, 3],
          DType.float64,
        );

        const path = 'scratch/test_f64.npy';
        save(path, a);

        final loaded = load(path);
        expect(loaded.shape, [2, 3]);
        expect(loaded.dtype, DType.float64);
        expect(loaded.toList(), [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]);
      });

      test('Float32 1D Vector round-trip', () {
        final a = NDArray.fromList(
          Float32List.fromList([-1.0, 0.0, 1.0, 10.5]),
          [4],
          DType.float32,
        );
        const path = 'scratch/test_f32.npy';
        save(path, a);

        final loaded = load(path);
        expect(loaded.shape, [4]);
        expect(loaded.dtype, DType.float32);
        expect(loaded.toList(), [-1.0, 0.0, 1.0, 10.5]);
      });

      test('Int32 matrix round-trip', () {
        final a = NDArray.fromList(Int32List.fromList([10, 20, 30, 40]), [
          2,
          2,
        ], DType.int32);
        const path = 'scratch/test_i32.npy';
        save(path, a);

        final loaded = load(path);
        expect(loaded.dtype, DType.int32);
        expect(loaded.toList(), [10, 20, 30, 40]);
      });

      test('Int64 vector round-trip', () {
        final a = NDArray.fromList(Int64List.fromList([999999999, 111111111]), [
          2,
        ], DType.int64);
        const path = 'scratch/test_i64.npy';
        save(path, a);

        final loaded = load(path);
        expect(loaded.dtype, DType.int64);
        expect(loaded.toList(), [999999999, 111111111]);
      });

      test('Complex128 array round-trip', () {
        final a = NDArray<Complex>.create([2], DType.complex128);
        a.data[0] = Complex(1.0, -2.0);
        a.data[1] = Complex(0.0, 3.5);

        const path = 'scratch/test_c16.npy';
        save(path, a);

        final loaded = load(path);
        expect(loaded.shape, [2]);
        expect(loaded.dtype, DType.complex128);
        expect(loaded.data[0], Complex(1.0, -2.0));
        expect(loaded.data[1], Complex(0.0, 3.5));
      });

      test(
        'Non-contiguous view save creates contiguous file copy seamlessly',
        () {
          final parent = NDArray.fromList(
            Float64List.fromList([1.0, 2.0, 3.0, 4.0]),
            [2, 2],
            DType.float64,
          );

          // Transposed view is non-contiguous (strides: [1, 2])
          final view = parent.transpose();
          expect(view.isContiguous, false);

          const path = 'scratch/test_view.npy';
          save(path, view); // should make contiguous copy in-flight

          final loaded = load(path);
          expect(loaded.shape, [2, 2]);
          // Transposed data logic: 1, 3, 2, 4
          expect(loaded.toList(), [1.0, 3.0, 2.0, 4.0]);
        },
      );
    });

    group('.npz Multi-Array Archive Tests', () {
      test('Save and Load uncompressed .npz archive map', () {
        final arr1 = NDArray.fromList(Float64List.fromList([1.0, 2.0]), [
          2,
        ], DType.float64);
        final arr2 = NDArray.fromList(Int32List.fromList([5, 6, 7, 8]), [
          2,
          2,
        ], DType.int32);

        final map = {'array_one': arr1, 'array_two': arr2};

        const path = 'scratch/archive.npz';
        savez(path, map, compressed: false);

        final loaded = loadz(path);
        expect(loaded.containsKey('array_one'), true);
        expect(loaded.containsKey('array_two'), true);

        expect(loaded['array_one']!.toList(), [1.0, 2.0]);
        expect(loaded['array_two']!.toList(), [5, 6, 7, 8]);
        expect(loaded['array_two']!.shape, [2, 2]);
      });

      test('Save and Load compressed .npz archive map', () {
        final arr1 = NDArray.fromList(Float32List.fromList([0.5, 1.5]), [
          2,
        ], DType.float32);
        const path = 'scratch/archive_comp.npz';
        savez(path, {'x': arr1}, compressed: true);

        final loaded = loadz(path);
        expect(loaded['x']!.toList(), [0.5, 1.5]);
        expect(loaded['x']!.dtype, DType.float32);
      });
    });

    group('Fortran Column-Major Layout Simulation Tests', () {
      test('Simulate and parse a Python-generated fortran_order=True header file', () {
        // We will build a fake .npy in-memory byte buffer that flags 'fortran_order': True.
        // For a 2x3 matrix [[1, 2, 3], [4, 5, 6]], the Column-Major flat ordering in memory
        // is: [1, 4, 2, 5, 3, 6]!
        final descr = _dtypeToDescr(DType.float64);
        final headerStr =
            "{'descr': '$descr', 'fortran_order': True, 'shape': (2, 3)}";

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

        // 6 doubles for data [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]
        final rawData = Float64List.fromList([1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
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

        // Write this fake file to disk
        const path = 'scratch/fortran_simulated.npy';
        File(path).writeAsBytesSync(fullBuffer, flush: true);

        // Load it via num_dart load()!
        final loaded = load(path);

        expect(loaded.shape, [2, 3]);
        // The zero-copy Fortran strides must be exactly: [1, 2]!
        expect(loaded.strides, [1, 2]);

        // Under stride-view indexing, loaded[i, j] translates to data[i*strides[0] + j*strides[1]].
        // loaded[0, 0] -> data[0] = 1.0
        // loaded[0, 1] -> data[2] = 2.0
        // loaded[0, 2] -> data[4] = 3.0
        // loaded[1, 0] -> data[1] = 4.0
        // loaded[1, 1] -> data[3] = 5.0
        // loaded[1, 2] -> data[5] = 6.0
        // So calling toList() (which loops row-major logically) should yield exactly [1, 2, 3, 4, 5, 6]!
        expect(loaded.toList(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        // Success! Stride reindexing mapping loaded column-major binary files with absolute zero data copies!
      });
    });
  });
}

// Simple helper function to map DType to descriptor string for test creation
String _dtypeToDescr(DType dtype) {
  switch (dtype) {
    case DType.float64:
      return '<f8';
    case DType.float32:
      return '<f4';
    case DType.int64:
      return '<i8';
    case DType.int32:
      return '<i4';
    case DType.complex128:
      return '<c16';
    case DType.complex64:
      return '<c8';
  }
}
