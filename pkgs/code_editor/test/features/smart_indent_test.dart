import 'package:code_editor/code_editor.dart';
import 'package:test/test.dart';

void main() {
  group('SmartIndentEngine Tests', () {
    const engine = SmartIndentEngine(tabSize: 2, indentString: '  ');

    test('preserves existing indentation on Enter', () {
      final buffer = PieceTreeTextBuffer('  final a = 1;');
      final res = engine.calculateEnter(
        buffer: buffer,
        position: const TextPosition(0, 14),
      );
      expect(res.textToInsert, '\n  ');
      expect(res.relativeCaretOffset, 3);
    });

    test('increases indentation after opening brace or colon', () {
      final buffer = PieceTreeTextBuffer('  void run() {');
      final res = engine.calculateEnter(
        buffer: buffer,
        position: const TextPosition(0, 14),
      );
      expect(res.textToInsert, '\n    ');
      expect(res.relativeCaretOffset, 5);
    });

    test('expands block when Enter is pressed between braces', () {
      final buffer = PieceTreeTextBuffer('  void run() {}');
      // Cursor between { and } (column 14)
      final res = engine.calculateEnter(
        buffer: buffer,
        position: const TextPosition(0, 14),
      );
      // Format: \n    \n
      expect(res.textToInsert, '\n    \n  ');
      expect(res.relativeCaretOffset, 5); // Cursor placed on indented line
    });

    test('indents and outdents line blocks correctly', () {
      final lines = ['var a = 1;', 'var b = 2;'];
      final indented = engine.indentLines(lines);
      expect(indented, ['  var a = 1;', '  var b = 2;']);

      final outdented = engine.outdentLines(indented);
      expect(outdented, ['var a = 1;', 'var b = 2;']);
    });
  });
}
