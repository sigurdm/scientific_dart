import 'package:code_editor/core.dart';
import 'package:test/test.dart';

void main() {
  group('Selection & SelectionModel', () {
    test('selection reversed and bounds properties', () {
      const forward = Selection(TextPosition(0, 2), TextPosition(0, 5));
      expect(forward.isReversed, isFalse);
      expect(forward.start, equals(const TextPosition(0, 2)));
      expect(forward.end, equals(const TextPosition(0, 5)));

      const reversed = Selection(TextPosition(0, 5), TextPosition(0, 2));
      expect(reversed.isReversed, isTrue);
      expect(reversed.start, equals(const TextPosition(0, 2)));
      expect(reversed.end, equals(const TextPosition(0, 5)));
    });

    test('multi-cursor merging', () {
      final model = SelectionModel();
      model.setSelections([
        const Selection(TextPosition(0, 0), TextPosition(0, 5)),
        const Selection(TextPosition(0, 3), TextPosition(0, 8)),
        const Selection(TextPosition(1, 0), TextPosition(1, 4)),
      ]);

      expect(model.selections.length, equals(2));
      expect(model.selections.first, equals(const Selection(TextPosition(0, 0), TextPosition(0, 8))));
      expect(model.selections.last, equals(const Selection(TextPosition(1, 0), TextPosition(1, 4))));
    });

    test('word selection bounds helper', () {
      final buffer = PieceTreeTextBuffer('hello_world foo_bar 123');
      final model = SelectionModel();

      final sel1 = model.selectWordAt(buffer, const TextPosition(0, 3));
      expect(sel1, equals(const Selection(TextPosition(0, 0), TextPosition(0, 11)))); // hello_world

      final sel2 = model.selectWordAt(buffer, const TextPosition(0, 14));
      expect(sel2, equals(const Selection(TextPosition(0, 12), TextPosition(0, 19)))); // foo_bar
    });

    test('selection shift on insert', () {
      final buffer = PieceTreeTextBuffer('Hello World');
      final model = SelectionModel();
      model.setSingleCursor(const TextPosition(0, 6)); // At 'W'

      // Transform selection before/during insert of "Beautiful " at offset 6
      model.transformOnInsert(buffer, 6, 'Beautiful ');
      buffer.insert(6, 'Beautiful ');

      expect(model.primary.position, equals(const TextPosition(0, 16)));
    });

    test('selection shift on delete', () {
      final buffer = PieceTreeTextBuffer('Hello Beautiful World');
      final model = SelectionModel();
      model.setSingleCursor(const TextPosition(0, 16)); // At 'W'

      // Transform selection before delete of "Beautiful " (offset 6, len 10)
      model.transformOnDelete(buffer, 6, 10);
      buffer.delete(6, 10);

      expect(model.primary.position, equals(const TextPosition(0, 6)));
    });
  });
}
