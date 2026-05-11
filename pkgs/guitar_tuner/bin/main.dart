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

  final List<String> peakLog = [];
  final List<String> waterfall = [];
  const waterfallHeight = 5;

  try {
    capture.open();
    print('Listening... Press Ctrl+C to stop.');
    // Reserve space for waterfall and status
    for (var i = 0; i < waterfallHeight + 1; i++) print('');

    // Run the loop
    while (true) {
      // Read directly into the pre-allocated NDArray buffer (Zero-copy)
      capture.read(captureBuffer);

      // Check if we have enough signal (RMS threshold) via num_dart
      final (double currentRms, TunerResult result) = NDArray.scope(() {
        final sqSamples = multiply(captureBuffer, captureBuffer);
        final rmsVal = math.sqrt(mean(sqSamples).scalar);

        final currentResult = logic.process(captureBuffer, rmsVal);
        return (rmsVal, currentResult);
      });

      // Move cursor up to start of our display area
      stdout.write('\x1b[${waterfallHeight + 1}A');

      // 1. Update waterfall
      waterfall.add(result.spectrumLine);
      if (waterfall.length > waterfallHeight) waterfall.removeAt(0);

      // 2. Log peaks (only if signal is significant)
      if (currentRms > 0.005 && result.note != '--') {
        if (peakLog.isEmpty || peakLog.last.split(' ')[0] != result.note) {
          peakLog.add(
            '${result.note} @ ${result.frequency.toStringAsFixed(1)} Hz',
          );
          if (peakLog.length > 5) peakLog.removeAt(0);
        }
      }

      // 3. Print waterfall
      for (var i = 0; i < waterfallHeight; i++) {
        final line = i < waterfall.length ? waterfall[i] : ' ' * 80;
        stdout.writeln('\x1b[K$line');
      }

      // 4. Print status line
      stdout.write('\r\x1b[K$result');
      if (peakLog.isNotEmpty) {
        stdout.write(' | Peaks: ${peakLog.join(", ")}');
      }
      stdout.write('\n'); // Ensure we are on the status line

      // Small sleep for responsiveness
      await Future.delayed(const Duration(milliseconds: 50));
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
