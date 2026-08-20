import 'dart:convert';
import 'dart:ffi' as ffi;
import 'dart:io';
import 'dart:typed_data';
import '../dtype.dart';
import '../gpu_array.dart';
import '../device.dart';

/// DType name mapping for SafeTensors specification.
String _dtypeToSafetensors(DType dtype) {
  switch (dtype) {
    case DType.float64:
      return 'F64';
    case DType.float32:
      return 'F32';
    case DType.float16:
      return 'F16';
    case DType.bfloat16:
      return 'BF16';
    case DType.int64:
      return 'I64';
    case DType.int32:
      return 'I32';
    case DType.int16:
      return 'I16';
    case DType.int8:
      return 'I8';
    case DType.uint64:
      return 'U64';
    case DType.uint32:
      return 'U32';
    case DType.uint16:
      return 'U16';
    case DType.uint8:
      return 'U8';
    case DType.boolean:
      return 'BOOL';
    case DType.complex64:
      return 'C64';
    case DType.complex128:
      return 'C128';
  }
}

DType _safetensorsToDtype(String st) {
  switch (st) {
    case 'F64':
      return DType.float64;
    case 'F32':
      return DType.float32;
    case 'F16':
      return DType.float16;
    case 'BF16':
      return DType.bfloat16;
    case 'I64':
      return DType.int64;
    case 'I32':
      return DType.int32;
    case 'I16':
      return DType.int16;
    case 'I8':
      return DType.int8;
    case 'U64':
      return DType.uint64;
    case 'U32':
      return DType.uint32;
    case 'U16':
      return DType.uint16;
    case 'U8':
      return DType.uint8;
    case 'BOOL':
      return DType.boolean;
    case 'C64':
      return DType.complex64;
    case 'C128':
      return DType.complex128;
    default:
      throw ArgumentError('Unsupported SafeTensors dtype: $st');
  }
}

/// Serializes [tensors] to binary SafeTensors format.
Uint8List saveSafetensors(
  Map<String, GpuArray> tensors, {
  Map<String, String>? metadata,
}) {
  final headerMap = <String, dynamic>{};
  if (metadata != null) {
    headerMap['__metadata__'] = metadata;
  }

  var currentOffset = 0;
  final tensorBytesList = <Uint8List>[];

  for (final entry in tensors.entries) {
    final name = entry.key;
    final tensor = entry.value;
    final contiguousTensor = tensor.isContiguous ? tensor : tensor.copy();
    final byteLen = contiguousTensor.byteSize;

    headerMap[name] = {
      'dtype': _dtypeToSafetensors(contiguousTensor.dtype),
      'shape': contiguousTensor.shape,
      'data_offsets': [currentOffset, currentOffset + byteLen],
    };

    currentOffset += byteLen;

    final u8List = Uint8List(byteLen);
    final ptr = (contiguousTensor.buffer.pointer.cast<ffi.Uint8>() +
            contiguousTensor.offsetElements * contiguousTensor.dtype.byteWidth)
        .cast<ffi.Uint8>();
    for (var i = 0; i < byteLen; i++) {
      u8List[i] = ptr[i];
    }
    tensorBytesList.add(u8List);

    if (!identical(contiguousTensor, tensor)) {
      contiguousTensor.dispose();
    }
  }

  final headerJson = jsonEncode(headerMap);
  final headerBytes = utf8.encode(headerJson);
  final headerLen = headerBytes.length;

  final totalSize = 8 + headerLen + currentOffset;
  final result = Uint8List(totalSize);
  final bdata = ByteData.sublistView(result);

  bdata.setUint64(0, headerLen, Endian.little);
  result.setRange(8, 8 + headerLen, headerBytes);

  var offset = 8 + headerLen;
  for (final tBytes in tensorBytesList) {
    result.setRange(offset, offset + tBytes.length, tBytes);
    offset += tBytes.length;
  }

  return result;
}

/// Deserializes a binary SafeTensors byte buffer into a dictionary of [GpuArray] tensors.
Map<String, GpuArray> loadSafetensors(
  Uint8List bytes, {
  GpuDevice? device,
}) {
  final dev = device ?? GpuDevice.defaultDevice;
  final bdata = ByteData.sublistView(bytes);
  final headerLen = bdata.getUint64(0, Endian.little);

  final headerBytes = bytes.sublist(8, 8 + headerLen);
  final headerJson = utf8.decode(headerBytes);
  final headerMap = jsonDecode(headerJson) as Map<String, dynamic>;

  final dataStartOffset = 8 + headerLen;
  final result = <String, GpuArray>{};

  for (final entry in headerMap.entries) {
    if (entry.key == '__metadata__') continue;

    final info = entry.value as Map<String, dynamic>;
    final dtypeStr = info['dtype'] as String;
    final dtype = _safetensorsToDtype(dtypeStr);
    final shape = (info['shape'] as List).cast<int>();
    final offsets = (info['data_offsets'] as List).cast<int>();

    final start = dataStartOffset + offsets[0];
    final end = dataStartOffset + offsets[1];
    final rawTensorBytes = bytes.sublist(start, end);

    final tensor = GpuArray.empty(shape, dtype, device: dev);
    final ptr = tensor.buffer.pointer.cast<ffi.Uint8>();
    for (var i = 0; i < rawTensorBytes.length; i++) {
      ptr[i] = rawTensorBytes[i];
    }

    result[entry.key] = tensor;
  }

  return result;
}

/// Saves [tensors] to a `.safetensors` file at [filePath].
void saveSafetensorsFile(
  String filePath,
  Map<String, GpuArray> tensors, {
  Map<String, String>? metadata,
}) {
  final bytes = saveSafetensors(tensors, metadata: metadata);
  File(filePath).writeAsBytesSync(bytes);
}

/// Loads tensors from a `.safetensors` file at [filePath].
Map<String, GpuArray> loadSafetensorsFile(
  String filePath, {
  GpuDevice? device,
}) {
  final bytes = File(filePath).readAsBytesSync();
  return loadSafetensors(bytes, device: device);
}
