import 'package:code_editor/code_editor.dart';
import 'package:test/test.dart';

void main() {
  group('AutoCloseEngine Tests', () {
    const engine = AutoCloseEngine();

    test(
      'returns InsertPairAction for opening bracket on collapsed cursor',
      () {
        final buffer = PieceTreeTextBuffer('final x = ');
        final action = engine.handleType(
          typedChar: '(',
          buffer: buffer,
          selection: const TextSelection.collapsed(TextPosition(0, 10)),
        );
        expect(action, isA<InsertPairAction>());
        final insert = action as InsertPairAction;
        expect(insert.open, '(');
        expect(insert.close, ')');
      },
    );

    test('returns WrapSelectionAction when text is selected', () {
      final buffer = PieceTreeTextBuffer('final x = 123;');
      final action = engine.handleType(
        typedChar: '"',
        buffer: buffer,
        selection: const TextSelection(
          base: TextPosition(0, 10),
          extent: TextPosition(0, 13),
        ),
      );
      expect(action, isA<WrapSelectionAction>());
      final wrap = action as WrapSelectionAction;
      expect(wrap.open, '"');
      expect(wrap.close, '"');
    });

    test(
      'returns SkipCloseAction when typing over existing closing bracket',
      () {
        final buffer = PieceTreeTextBuffer('final x = ()');
        final action = engine.handleType(
          typedChar: ')',
          buffer: buffer,
          selection: const TextSelection.collapsed(TextPosition(0, 11)),
        );
        expect(action, isA<SkipCloseAction>());
      },
    );

    test('deletes both opening and closing bracket on backspace', () {
      final buffer = PieceTreeTextBuffer('final x = ()');
      // Cursor between ( and ) at column 11
      final count = engine.checkBackspacePairDeletion(
        buffer: buffer,
        position: const TextPosition(0, 11),
      );
      expect(count, 2);
    });
  });
}
