import 'package:code_editor/code_editor.dart';
import 'package:test/test.dart';

void main() {
  group('SnippetEngine & Controller Snippet Tests', () {
    test('expands built-in for loop snippet with tab stops', () {
      final engine = SnippetEngine();
      final snippet = engine.findSnippet('for');
      expect(snippet, isNotNull);

      final buffer = PieceTreeTextBuffer('');
      final res = engine.expandSnippet(
        body: snippet!.body,
        insertOffset: 0,
        buffer: buffer,
      );

      expect(res.insertedText, 'for (var  = 0;  < ; ++) {\n  \n}');
      expect(res.tabStops.length, 5); // $1, $1, $2, $1, $0
      expect(res.tabStops.first.index, 1);
      expect(res.tabStops.last.index, 0);
    });

    test('controller Tab expands snippet prefix at cursor', () {
      final controller = CodeEditorController(initialText: 'for');
      controller.selection = const TextSelection.collapsed(TextPosition(0, 3));

      controller.tabPressed();
      expect(controller.text.startsWith('for (var '), isTrue);
    });
  });
}
