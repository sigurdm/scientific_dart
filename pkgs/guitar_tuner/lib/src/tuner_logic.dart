import 'dart:math' as math;
import 'package:ndarray/ndarray.dart';

final class TunerResult {
  final String note;
  final double frequency;
  final double targetFrequency;
  final double cents;
  final double rms;

  TunerResult({
    required this.note,
    required this.frequency,
    required this.targetFrequency,
    required this.cents,
    required this.rms,
  });

  @override
  String toString() {
    final status =
        cents.abs() < 5 ? 'IN TUNE' : (cents > 0 ? 'TOO HIGH' : 'TOO LOW');
    final visual = _generateVisual();
    final vol = _generateVolumeBar();
    return 'Vol: $vol | $note |$visual| (${frequency.toStringAsFixed(1)} Hz) - $status (${cents.toStringAsFixed(1)} cents)';
  }

  String _generateVolumeBar() {
    const width = 10;
    // Map RMS (approx 0 to 0.5 for loud signal) to bar width
    int level = (rms * 20).round().clamp(0, width);
    return '[' + '#' * level + ' ' * (width - level) + ']';
  }

  String _generateVisual() {
    const width = 21; // odd number
    const center = width ~/ 2;
    final chars = List.filled(width, '-');
    chars[center] = '|';

    // Map cents (-50 to 50) to index (0 to 20)
    int needlePos = (center + (cents / 50.0 * center)).round();
    needlePos = needlePos.clamp(0, width - 1);

    if (cents.abs() < 5) {
      chars[needlePos] = '★'; // Special char for "in tune"
    } else {
      chars[needlePos] = '▲';
    }
    return chars.join();
  }
}

final class TunerLogic {
  final int sampleRate;
  final int bufferSize;
  final NDArray<double> _window;

  static const Map<String, double> guitarNotes = {
    'E2': 82.41,
    'A2': 110.00,
    'D3': 146.83,
    'G3': 196.00,
    'B3': 246.94,
    'E4': 329.63,
  };

  TunerLogic({required this.sampleRate, required this.bufferSize})
      : _window = hanning(bufferSize, dtype: DType.float32);

  TunerResult process(NDArray<double> input, double rms) {
    return NDArray.scope(() {
      // 2. Apply Hanning window to reduce spectral leakage
      final windowedInput = multiply(input, _window) as NDArray<double>;

      // 3. Perform FFT
      final spectrum = fft(windowedInput);

      // 4. Calculate magnitude spectrum
      final magnitudes = abs(spectrum);

      // 5. Find peak in the spectrum (excluding DC and higher half due to symmetry)
      final minIdx = (50 * bufferSize / sampleRate).floor();
      final maxIdx = (1000 * bufferSize / sampleRate).ceil();

      final searchRange = magnitudes.slice([Slice(start: minIdx, stop: maxIdx)]);
      final peakIdxInSlice = argmax(searchRange) as int;
      final peakIdx = minIdx + peakIdxInSlice;

      // 6. Parabolic interpolation for more accurate peak frequency
      double refinedPeakIdx = peakIdx.toDouble();
      if (peakIdx > 0 && peakIdx < magnitudes.shape[0] - 1) {
        final magsList = magnitudes.toList();
        final y1 = magsList[peakIdx - 1];
        final y2 = magsList[peakIdx];
        final y3 = magsList[peakIdx + 1];

        final denom = y1 - 2 * y2 + y3;
        if (denom != 0) {
          refinedPeakIdx = peakIdx + 0.5 * (y1 - y3) / denom;
        }
      }

      final freq = refinedPeakIdx * sampleRate / bufferSize;

      // 4. Find closest note
      String closestNote = 'Unknown';
      double minDiff = double.infinity;
      double targetFreq = 0.0;

      guitarNotes.forEach((note, noteFreq) {
        final diff = (freq - noteFreq).abs();
        if (diff < minDiff) {
          minDiff = diff;
          closestNote = note;
          targetFreq = noteFreq;
        }
      });

      // 5. Calculate cents difference
      final cents = 1200 * math.log(freq / targetFreq) / math.log(2);

      return TunerResult(
        note: closestNote,
        frequency: freq,
        targetFrequency: targetFreq,
        cents: cents,
        rms: rms,
      );
    });
  }

  void dispose() {
    _window.dispose();
  }
}
