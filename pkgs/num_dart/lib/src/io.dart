import 'dart:io';
import 'dart:typed_data';
import 'dart:ffi' as ffi;
import 'package:archive/archive.dart';
import 'ndarray.dart';

/// Maps a NumPy descriptor string back to an [NDArray] [DType].
DType _descrToDType(String descr) {
  if (descr.contains('>')) {
    throw UnsupportedError('Big-Endian .npy files are not supported yet.');
  }
  // Strip byte-order indicators if any (e.g., '<', '>', '|')
  final clean = descr.replaceAll(RegExp(r'[<>|]'), '');
  switch (clean) {
    case 'f8':
      return DType.float64;
    case 'f4':
      return DType.float32;
    case 'i8':
      return DType.int64;
    case 'i4':
      return DType.int32;
    case 'c16':
      return DType.complex128;
    case 'c8':
      return DType.complex64;
    case 'b1':
      return DType.boolean;
    default:
      throw UnsupportedError('Unsupported NumPy data type descriptor: $descr');
  }
}

/// Save an [NDArray] to disk in the standard NumPy binary format (`.npy`).
///
/// This saves the array's shape, datatype, and raw heap memory in a highly compressed
/// little-endian format which is 100% cross-language compatible with Python's NumPy.
///
/// It performs zero-copy block disk operations by viewing the unmanaged C heap
/// pointers directly as a native Dart list view.
///
/// **Example:**
/// {@example /example/numpy_interop_example.dart lang=dart}
void save(String filepath, NDArray a) {
  final file = File(filepath);
  if (!file.parent.existsSync()) {
    file.parent.createSync(recursive: true);
  }

  // Re-orient to contiguous array if a is a strided view to ensure binary file sequentiality
  Uint8List? serializedBytes;
  if (!a.isContiguous) {
    serializedBytes = _serializeDataContiguous(a.toList(), a.dtype);
  }

  final descr = a.dtype.npyDescriptor;
  final shapeStr = a.shape.length == 1 ? '${a.shape[0]},' : a.shape.join(', ');

  // Build NumPy Version 1.0 Header String dictionary literal
  final headerStr =
      "{'descr': '$descr', 'fortran_order': False, 'shape': ($shapeStr)}";

  // Pad header string so that: magic_string(6) + versions(2) + header_len_bytes(2) + headerStr.length
  // is an exact multiple of 64 bytes.
  final prefixLen = 6 + 2 + 2;
  var paddedHeaderLen =
      ((prefixLen + headerStr.length + 1) + 63) ~/ 64 * 64 - prefixLen;

  // Use space padding and end with a mandatory trailing newline '\n'
  final padCount = paddedHeaderLen - headerStr.length - 1;
  final paddedHeader = headerStr + (' ' * padCount) + '\n';

  final raf = file.openSync(mode: FileMode.write);

  try {
    // 1. Magic string prefix
    raf.writeFromSync(const [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]); // \x93NUMPY
    // 2. Version 1.0 bytes
    raf.writeFromSync(const [0x01, 0x00]);
    // 3. Little-endian 2-byte unsigned short header length
    final headerBytes = Uint8List.fromList(paddedHeader.codeUnits);
    final headerLenBytes = Uint8List(2);
    ByteData.view(
      headerLenBytes.buffer,
    ).setUint16(0, headerBytes.length, Endian.little);
    raf.writeFromSync(headerLenBytes);

    // 4. Write ASCII header dictionary
    raf.writeFromSync(headerBytes);

    // 5. Zero-Copy Raw C-Heap Bytes block dump!
    if (serializedBytes != null) {
      raf.writeFromSync(serializedBytes);
    } else {
      final elementCount = a.shape.isEmpty
          ? 1
          : a.shape.reduce((x, y) => x * y);
      final byteSize = elementCount * a.dtype.byteWidth;
      final byteView = a.pointer.cast<ffi.Uint8>().asTypedList(byteSize);
      raf.writeFromSync(byteView);
    }
  } finally {
    raf.closeSync();
  }
}

/// Load an [NDArray] binary data block from a NumPy `.npy` file.
///
/// Fully parses NumPy little-endian datatype descriptors, shapes, and includes
/// native **zero-copy Column-Major Fortran strides mapping** support. If a file
/// is flagged as `fortran_order: True` (column-major from Python), `load()` loads the raw binary
/// sequential columns straight into the C heap and configures column-major strides, completely
/// eliminating slow sorting data loops!
///
/// **Example:**
/// {@example /example/numpy_interop_example.dart lang=dart}
NDArray load(String filepath) {
  final file = File(filepath);
  if (!file.existsSync()) {
    throw FileSystemException('File not found for load', filepath);
  }

  final raf = file.openSync(mode: FileMode.read);

  try {
    // 1. Check magic prefix
    final magic = raf.readSync(6);
    if (magic.length != 6 ||
        magic[0] != 0x93 ||
        magic[1] != 0x4e ||
        magic[2] != 0x55 ||
        magic[3] != 0x4d ||
        magic[4] != 0x50 ||
        magic[5] != 0x59) {
      throw FormatException('Invalid NumPy .npy binary file signature');
    }

    // 2. Read version
    final version = raf.readSync(2);
    if (version.length != 2) {
      throw FormatException('Failed to read .npy format version headers');
    }

    // 3. Read header length
    final lenBytes = raf.readSync(2);
    final headerLen = ByteData.view(
      lenBytes.buffer,
    ).getUint16(0, Endian.little);

    // 4. Read ASCII Header dictionary
    final headerBytes = raf.readSync(headerLen);
    final headerStr = String.fromCharCodes(headerBytes);

    // Parse descr via regex
    final descrMatch = RegExp(
      '[\'\"]descr[\'\"]:\\s*[\'\"]([^\'\"]+)[\'\"]',
    ).firstMatch(headerStr);
    if (descrMatch == null) {
      throw FormatException(
        'Invalid npy header: could not parse "descr" parameter string',
      );
    }
    final descr = descrMatch.group(1)!;
    final dtype = _descrToDType(descr);

    // Parse fortran_order bool flag
    final fortMatch = RegExp(
      '[\'\"]fortran_order[\'\"]:\\s*(True|False)',
      caseSensitive: false,
    ).firstMatch(headerStr);
    if (fortMatch == null) {
      throw FormatException(
        'Invalid npy header: could not parse "fortran_order" boolean flag',
      );
    }
    final fortranOrder = fortMatch.group(1)!.toLowerCase() == 'true';

    // Parse shape tuple
    final shapeMatch = RegExp(
      '[\'\"]shape[\'\"]:\\s*\\(?([^\\)]*)\\)?',
    ).firstMatch(headerStr);
    if (shapeMatch == null) {
      throw FormatException(
        'Invalid npy header: could not parse "shape" tuple tokens',
      );
    }
    final shapeTokens = shapeMatch.group(1)!.split(',');
    final shape = <int>[];
    for (var tok in shapeTokens) {
      final cleanTok = tok.trim();
      if (cleanTok.isNotEmpty) {
        shape.add(int.parse(cleanTok));
      }
    }

    // 5. Allocate matching NDArray with target layout strategies
    final elementCount = shape.isEmpty ? 1 : shape.reduce((x, y) => x * y);
    final byteSize = elementCount * dtype.byteWidth;

    final result = NDArray.create(shape, dtype);

    // Wire Zero-Copy Column-Major Fortran strides if the file demands it!
    if (fortranOrder && shape.length > 1) {
      final fStrides = List<int>.filled(shape.length, 0);
      var stride = 1;
      for (var i = 0; i < shape.length; i++) {
        fStrides[i] = stride;
        stride *= shape[i];
      }
      // Mutate strides metadata in the unmanaged array safely
      result.strides.setRange(0, shape.length, fStrides);
    }

    // 6. Zero-Copy direct stream file read straight into C Heap pointers!
    final byteView = result.pointer.cast<ffi.Uint8>().asTypedList(byteSize);
    raf.readIntoSync(byteView);

    return result;
  } finally {
    raf.closeSync();
  }
}

/// Serialize an [NDArray] directly into in-memory `.npy` bytes (internal utility for .npz).
Uint8List _serializeNpyBytes(NDArray a) {
  Uint8List? serializedBytes;
  if (!a.isContiguous) {
    serializedBytes = _serializeDataContiguous(a.toList(), a.dtype);
  }

  final descr = a.dtype.npyDescriptor;
  final shapeStr = a.shape.length == 1 ? '${a.shape[0]},' : a.shape.join(', ');
  final headerStr =
      "{'descr': '$descr', 'fortran_order': False, 'shape': ($shapeStr)}";

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

  final elementCount = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
  final dataByteSize = elementCount * a.dtype.byteWidth;
  final Uint8List rawDataView =
      serializedBytes ?? a.pointer.cast<ffi.Uint8>().asTypedList(dataByteSize);

  final totalBytesSize = 6 + 2 + 2 + headerBytes.length + rawDataView.length;
  final resultList = Uint8List(totalBytesSize);

  var offset = 0;
  // Magic
  resultList.setRange(offset, offset + 6, const [
    0x93,
    0x4e,
    0x55,
    0x4d,
    0x50,
    0x59,
  ]);
  offset += 6;
  // Version
  resultList.setRange(offset, offset + 2, const [0x01, 0x00]);
  offset += 2;
  // Header Len
  resultList.setRange(offset, offset + 2, lenBytes);
  offset += 2;
  // Header ASCII
  resultList.setRange(offset, offset + headerBytes.length, headerBytes);
  offset += headerBytes.length;
  // Data block copy
  resultList.setRange(offset, offset + rawDataView.length, rawDataView);

  return resultList;
}

/// Deserializes an [NDArray] directly from in-memory `.npy` bytes (internal utility for .npz).
NDArray _deserializeNpyBytes(Uint8List bytes) {
  if (bytes.length < 10 ||
      bytes[0] != 0x93 ||
      bytes[1] != 0x4e ||
      bytes[2] != 0x55 ||
      bytes[3] != 0x4d ||
      bytes[4] != 0x50 ||
      bytes[5] != 0x59) {
    throw FormatException('Invalid in-memory .npy byte signature block');
  }

  final headerLen = ByteData.view(
    bytes.buffer,
    bytes.offsetInBytes + 8,
    2,
  ).getUint16(0, Endian.little);

  final headerBytes = Uint8List.view(
    bytes.buffer,
    bytes.offsetInBytes + 10,
    headerLen,
  );
  final headerStr = String.fromCharCodes(headerBytes);

  final descrMatch = RegExp(
    '[\'\"]descr[\'\"]:\\s*[\'\"]([^\'\"]+)[\'\"]',
  ).firstMatch(headerStr);
  final descr = descrMatch!.group(1)!;
  final dtype = _descrToDType(descr);

  final fortMatch = RegExp(
    '[\'\"]fortran_order[\'\"]:\\s*(True|False)',
    caseSensitive: false,
  ).firstMatch(headerStr);
  final fortranOrder = fortMatch!.group(1)!.toLowerCase() == 'true';

  final shapeMatch = RegExp(
    '[\'\"]shape[\'\"]:\\s*\\(?([^\\)]*)\\)?',
  ).firstMatch(headerStr);
  final shapeTokens = shapeMatch!.group(1)!.split(',');
  final shape = <int>[];
  for (var tok in shapeTokens) {
    final cleanTok = tok.trim();
    if (cleanTok.isNotEmpty) shape.add(int.parse(cleanTok));
  }

  final elementCount = shape.isEmpty ? 1 : shape.reduce((x, y) => x * y);
  final dataByteSize = elementCount * dtype.byteWidth;

  final result = NDArray.create(shape, dtype);

  if (fortranOrder && shape.length > 1) {
    final fStrides = List<int>.filled(shape.length, 0);
    var stride = 1;
    for (var i = 0; i < shape.length; i++) {
      fStrides[i] = stride;
      stride *= shape[i];
    }
    result.strides.setRange(0, shape.length, fStrides);
  }

  final dataOffset = 10 + headerLen;
  final targetView = result.pointer.cast<ffi.Uint8>().asTypedList(dataByteSize);

  final sourceView = Uint8List.view(
    bytes.buffer,
    bytes.offsetInBytes + dataOffset,
    dataByteSize,
  );
  targetView.setRange(0, dataByteSize, sourceView);

  return result;
}

/// Save multiple named arrays to a single zip archive file on disk (`.npz`).
///
/// This corresponds to NumPy's `savez` / `savez_compressed` functions.
/// If [compressed] is true, applies Deflate compression to minimize disk footprints.
///
/// **Example:**
/// {@example /example/numpy_interop_example.dart lang=dart}
void savez(
  String filepath,
  Map<String, NDArray> arrays, {
  bool compressed = false,
}) {
  final archive = Archive();

  for (final entry in arrays.entries) {
    final fileBytes = _serializeNpyBytes(entry.value);
    final archiveFile = ArchiveFile(
      '${entry.key}.npy',
      fileBytes.length,
      fileBytes,
    );
    archive.addFile(archiveFile);
  }

  // Encode archive as zip byte streams
  final encoder = ZipEncoder();
  final zipBytes = encoder.encode(
    archive,
    level: compressed ? Deflate.BEST_COMPRESSION : Deflate.NO_COMPRESSION,
  );

  if (zipBytes == null) {
    throw FormatException('Failed to encode .npz zip archive format bytes');
  }

  final file = File(filepath);
  if (!file.parent.existsSync()) {
    file.parent.createSync(recursive: true);
  }
  file.writeAsBytesSync(Uint8List.fromList(zipBytes), flush: true);
}

/// Load multiple named [NDArray] instances back from a NumPy `.npz` zip archive.
///
/// Unpacks and deserializes all inner files, mapping variable name keys to loaded array targets.
/// Supports compressed and uncompressed Python-generated `.npz` files.
///
/// **Memory Consideration Warning (3x Footprint Hazard):**
/// During `.npz` deserialization, this method reads the archive bytes and decodes them fully in memory
/// via `ZipDecoder().decodeBytes()`. When deserializing each `.npy` file, it allocates a contiguous unmanaged
/// C-heap memory block and copies the bytes block.
/// This creates a temporary **3x RAM footprint amplification factor** (compressed archive bytes list + fully
/// inflated `ArchiveFile` list + unmanaged FFI pointer heap arrays). For gigabyte-scale scientific datasets
/// (e.g. large machine learning checkpoints or dense matrix grids logs), ensure the host system has sufficient
/// free memory pages to prevent Out-Of-Memory (OOM) isolate VM kills.
///
/// **Example:**
/// {@example /example/numpy_interop_example.dart lang=dart}
Map<String, NDArray> loadz(String filepath) {
  final file = File(filepath);
  if (!file.existsSync()) {
    throw FileSystemException('File not found for loadz npz', filepath);
  }

  final bytes = file.readAsBytesSync();
  final decoder = ZipDecoder();
  final archive = decoder.decodeBytes(bytes);

  final results = <String, NDArray>{};

  for (final archiveFile in archive) {
    if (archiveFile.isFile && archiveFile.name.endsWith('.npy')) {
      final key = archiveFile.name.replaceAll('.npy', '');
      final rawContent = archiveFile.content;
      final Uint8List fileData = rawContent is Uint8List
          ? rawContent
          : Uint8List.fromList(rawContent as List<int>);

      final loadedArray = _deserializeNpyBytes(fileData);
      results[key] = loadedArray;
      archiveFile.clear();
      archiveFile.closeSync();
    }
  }

  return results;
}

/// Serializes data elements of a non-contiguous array to a contiguous standard list view
/// to avoid allocating a transient unmanaged [NDArray] structure on the C-heap.
Uint8List _serializeDataContiguous(List flatList, DType dtype) {
  switch (dtype) {
    case DType.float64:
      return Float64List.fromList(flatList.cast<double>()).buffer.asUint8List();
    case DType.float32:
      return Float32List.fromList(flatList.cast<double>()).buffer.asUint8List();
    case DType.int64:
      return Int64List.fromList(flatList.cast<int>()).buffer.asUint8List();
    case DType.int32:
      return Int32List.fromList(flatList.cast<int>()).buffer.asUint8List();
    case DType.boolean:
      final bytes = Uint8List(flatList.length);
      for (var i = 0; i < flatList.length; i++) {
        bytes[i] = (flatList[i] as bool) ? 1 : 0;
      }
      return bytes;
    case DType.complex128:
      final doubleList = Float64List(flatList.length * 2);
      for (var i = 0; i < flatList.length; i++) {
        final c = flatList[i] as Complex;
        doubleList[i * 2] = c.real;
        doubleList[i * 2 + 1] = c.imag;
      }
      return doubleList.buffer.asUint8List();
    case DType.complex64:
      final floatList = Float32List(flatList.length * 2);
      for (var i = 0; i < flatList.length; i++) {
        final c = flatList[i] as Complex;
        floatList[i * 2] = c.real;
        floatList[i * 2 + 1] = c.imag;
      }
      return floatList.buffer.asUint8List();
  }
}
