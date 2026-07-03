import 'package:code_editor/core.dart';
import 'package:test/test.dart';

void main() {
  group('UndoManager', () {
    late PieceTreeTextBuffer buffer;
    late SelectionModel selectionModel;
    late UndoManager undoManager;

    setUp(() {
      buffer = PieceTreeTextBuffer('Hello World');
      selectionModel = SelectionModel();
      undoManager = UndoManager(
        maxStackSize: 5,
        typingWindow: const Duration(milliseconds: 100),
      );
    });

    test('single insert transaction undo and redo', () {
      final op = InsertOperation(11, '!');
      final tx = EditorTransaction(
        operations: [op],
        selectionsBefore: selectionModel.selections,
        selectionsAfter: [const Selection.collapsed(TextPosition(0, 12))],
      );

      op.apply(buffer);
      undoManager.commitTransaction(tx);

      expect(buffer.text, equals('Hello World!'));
      expect(undoManager.canUndo, isTrue);

      undoManager.undo(buffer, selectionModel);
      expect(buffer.text, equals('Hello World'));
      expect(undoManager.canRedo, isTrue);

      undoManager.redo(buffer, selectionModel);
      expect(buffer.text, equals('Hello World!'));
    });

    test('stack limit enforcement', () {
      for (var i = 0; i < 10; i++) {
        final op = InsertOperation(buffer.length, ' $i');
        final tx = EditorTransaction(
          operations: [op],
          selectionsBefore: selectionModel.selections,
          selectionsAfter: selectionModel.selections,
          timestamp: DateTime.now().add(Duration(seconds: i + 1)), // Avoid window coalescing
        );
        op.apply(buffer);
        undoManager.commitTransaction(tx);
      }

      // Max stack size is 5, so we can only undo 5 times
      var undoCount = 0;
      while (undoManager.canUndo) {
        undoManager.undo(buffer, selectionModel);
        undoCount++;
      }
      expect(undoCount, equals(5));
    });

    test('document versioning and dirty state', () {
      expect(undoManager.isDirty, isFalse);
      undoManager.markSaved();

      final op = InsertOperation(0, 'A');
      final tx = EditorTransaction(
        operations: [op],
        selectionsBefore: selectionModel.selections,
        selectionsAfter: selectionModel.selections,
      );
      op.apply(buffer);
      undoManager.commitTransaction(tx);

      expect(undoManager.isDirty, isTrue);
      expect(undoManager.version, equals(1));

      undoManager.undo(buffer, selectionModel);
      expect(undoManager.isDirty, isFalse);
    });
  });
}
