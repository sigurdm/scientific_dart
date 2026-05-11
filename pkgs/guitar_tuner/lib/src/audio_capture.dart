@ffi.DefaultAsset('package:guitar_tuner/tuner_bridge')
library;

import 'dart:ffi' as ffi;
import 'package:ndarray/ndarray.dart';

@ffi.Native<ffi.Pointer<ffi.Void> Function(ffi.Int32, ffi.Int32)>()
external ffi.Pointer<ffi.Void> tuner_init(int sampleRate, int ringBufferSize);

@ffi.Native<
  ffi.Void Function(ffi.Pointer<ffi.Void>, ffi.Pointer<ffi.Float>, ffi.Int32)
>()
external void tuner_get_samples(
  ffi.Pointer<ffi.Void> context,
  ffi.Pointer<ffi.Float> output,
  int count,
);

@ffi.Native<ffi.Void Function(ffi.Pointer<ffi.Void>)>()
external void tuner_close(ffi.Pointer<ffi.Void> context);

final class AudioCapture {
  final int sampleRate;
  final int bufferSize;

  ffi.Pointer<ffi.Void> _context = ffi.nullptr;

  AudioCapture({this.sampleRate = 44100, this.bufferSize = 4096});

  void open() {
    // Initialize with a ring buffer 4x the size of the request buffer
    _context = tuner_init(sampleRate, bufferSize * 4);
    if (_context == ffi.nullptr) {
      throw Exception('Failed to initialize audio capture via miniaudio');
    }
  }

  /// Reads the current audio samples into the provided [buffer].
  ///
  /// The [buffer] must have [DType.float32] and its first dimension must be at least [bufferSize].
  void read(NDArray<double> buffer) {
    if (_context == ffi.nullptr) return;

    if (buffer.dtype != DType.float32) {
      throw ArgumentError('Buffer must be DType.float32');
    }
    if (buffer.shape[0] < bufferSize) {
      throw ArgumentError(
        'Buffer is too small (expected at least $bufferSize)',
      );
    }

    tuner_get_samples(_context, buffer.pointer.cast(), bufferSize);
  }

  void close() {
    if (_context != ffi.nullptr) {
      tuner_close(_context);
      _context = ffi.nullptr;
    }
  }
}
