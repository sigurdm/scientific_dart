import 'package:code_editor/code_editor.dart';
import 'package:test/test.dart';

void main() {
  group('FindReplaceController Tests', () {
    test('finds plain text matches case-insensitively and navigates', () {
      final buffer = PieceTreeTextBuffer('hello world Hello Dart hello');
      final controller = FindReplaceController();

      controller.find(
        buffer,
        const SearchOptions(query: 'hello', matchCase: false),
      );
      expect(controller.matchCount, 3);
      expect(controller.activeMatchIndex, 0);

      final next = controller.findNext();
      expect(next, isNotNull);
      expect(controller.activeMatchIndex, 1);
      expect(next!.text, 'Hello');

      final prev = controller.findPrevious();
      expect(prev, isNotNull);
      expect(controller.activeMatchIndex, 0);
    });

    test('respects case-sensitivity and whole word search', () {
      final buffer = PieceTreeTextBuffer('foo fooBar FOO foo');
      final controller = FindReplaceController();

      controller.find(
        buffer,
        const SearchOptions(
          query: 'foo',
          matchCase: true,
          matchWholeWord: true,
        ),
      );
      expect(controller.matchCount, 2);
    });

    test('finds regex patterns correctly', () {
      final buffer = PieceTreeTextBuffer('val1: 100, val2: 200, val3: 300');
      final controller = FindReplaceController();

      controller.find(
        buffer,
        const SearchOptions(query: r'val\d+:\s+\d+', isRegex: true),
      );
      expect(controller.matchCount, 3);
    });
  });
}
