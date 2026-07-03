import 'package:code_editor/viewport.dart';
import 'package:test/test.dart';

void main() {
  group('LineWrappingEngine', () {
    final metrics = FontMetrics(characterWidth: 10.0);
    final measurer = TextMeasurer(metrics);

    test('computes word-boundary soft wraps correctly', () {
      const line = 'hello world flutter dart';
      final wrap = LineWrappingEngine.computeWrap(line, 120.0, measurer, wordWrap: true);

      expect(wrap.breakOffsets, isNotEmpty);
      expect(wrap.subLineCount, greaterThan(1));
    });

    test('maps document column to display sub-line position and back', () {
      const line = 'abcdefghijklmnopqrstuvwxyz';
      final wrap = LineWrappingEngine.computeWrap(line, 100.0, measurer, wordWrap: false);

      expect(wrap.breakOffsets, equals([10, 20]));
      expect(wrap.subLineCount, equals(3));

      expect(wrap.documentColumnToDisplay(5), equals(const DisplayPosition(0, 5)));
      expect(wrap.documentColumnToDisplay(15), equals(const DisplayPosition(1, 5)));
      expect(wrap.documentColumnToDisplay(25), equals(const DisplayPosition(2, 5)));

      expect(wrap.displayColumnToDocument(1, 5), equals(15));
    });
  });
}
