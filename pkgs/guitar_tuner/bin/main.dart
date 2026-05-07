import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'package:ndarray/ndarray.dart';
import '../lib/src/audio_capture.dart';
import '../lib/src/tuner_logic.dart';

void main() async {
  const sampleRate = 44100;
  const bufferSize = 8192; // Higher resolution for tuning

  print('Guitar Tuner (Cross-platform)');
  print('-----------------------------');
  print('Initializing audio capture...');

  final capture = AudioCapture(sampleRate: sampleRate, bufferSize: bufferSize);
  final logic = TunerLogic(sampleRate: sampleRate, bufferSize: bufferSize);

  // Pre-allocate capture buffer to avoid garbage collection pressure
  final NDArray<double> captureBuffer = NDArray<double>.create([
    bufferSize,
  ], DType.float32);

  try {
    capture.open();
    print('Listening... Press Ctrl+C to stop.\n');

    // Run the loop
    while (true) {
      // Read directly into the pre-allocated NDArray buffer (Zero-copy)
      capture.read(captureBuffer);

      // Check if we have enough signal (RMS threshold) via num_dart
      final (double currentRms, TunerResult? result) = NDArray.scope(() {
        final sqSamples = multiply(captureBuffer, captureBuffer);
        final rmsVal = math.sqrt(mean(sqSamples as NDArray<double>) as double);

        TunerResult? currentResult;
        if (rmsVal > 0.01) {
          currentResult = logic.process(captureBuffer);
        }
        return (rmsVal, currentResult);
      });

      if (result != null) {
        // Clear line and print result
        stdout.write('\r\x1b[K$result (RMS: ${currentRms.toStringAsFixed(4)})');
      } else {
        stdout.write(
          '\r\x1b[KWaiting for signal... (RMS: ${currentRms.toStringAsFixed(4)})',
        );
      }

      // Small sleep to not hog the CPU too much
      await Future.delayed(const Duration(milliseconds: 100));
    }
  } catch (e) {
    print('\nError: $e');
  } finally {
    capture.close();
    captureBuffer.dispose();
    logic.dispose();
    print('\nClosed.');
  }
}
