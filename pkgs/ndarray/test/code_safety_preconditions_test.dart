import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';

void main() {
  group('Code Safety & Parameter Preconditions Tests', () {
    test('spacers numSamples strictly positive check', () {
      NDArray.scope(() {
        // logspace and geomspace must throw ArgumentError for non-positive numSamples
        expect(() => logspace(0.0, 3.0, 0), throwsArgumentError);
        expect(() => logspace(0.0, 3.0, -5), throwsArgumentError);
        expect(() => geomspace(1.0, 100.0, 0), throwsArgumentError);
        expect(() => geomspace(1.0, 100.0, -3), throwsArgumentError);
      });
    });

    test('GridRange constructor parameter validation check', () {
      // GridRange must throw ArgumentError for zero step or non-positive numPoints
      expect(() => GridRange(0.0, 10.0, step: 0.0), throwsArgumentError);
      expect(() => GridRange(0.0, 10.0, numPoints: 0), throwsArgumentError);
      expect(() => GridRange(0.0, 10.0, numPoints: -5), throwsArgumentError);
    });

    test('FFT functions input disposed array state check', () {
      final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float64);
      a.dispose();
      
      // fft and ifft must throw StateError if input is disposed
      expect(() => fft(a), throwsStateError);
      expect(() => ifft(a), throwsStateError);
    });
  });
}
