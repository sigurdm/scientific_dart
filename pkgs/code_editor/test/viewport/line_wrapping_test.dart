import 'package:code_editor/viewport.dart';
import 'package:test/test.dart';

void main() {
  group('LineWrappingEngine', () {
    test('computes word-boundary soft wraps correctly', () {
      const engine = LineWrappingEngine(maxColumns: 10, enabled: true);
      final slices = engine.wrapLine('hello world flutter dart', 0);

      expect(slices.length, greaterThan(1));
      expect(slices.first.content, equals('hello '));
    });

    test('wraps long lines into multiple slices', () {
      const engine = LineWrappingEngine(maxColumns: 10, enabled: true);
      final slices = engine.wrapLine('abcdefghijklmnopqrstuvwxyz', 0);

      expect(slices.length, equals(3));
      expect(slices[0].content, equals('abcdefghij'));
      expect(slices[1].content, equals('klmnopqrst'));
      expect(slices[2].content, equals('uvwxyz'));
    });
  });
}
