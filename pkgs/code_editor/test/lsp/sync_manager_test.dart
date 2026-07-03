import 'package:code_editor/core.dart';
import 'package:code_editor/lsp.dart';
import 'package:test/test.dart';

void main() {
  group('LspDocumentSyncManager', () {
    test('incremental change events generation for insertion and deletion', () {
      final buffer = PieceTreeTextBuffer('hello world');
      final syncManager = LspDocumentSyncManager(uri: 'file:///test.dart', version: 1);

      final insertOp = InsertOperation(5, ' my');
      final tx1 = EditorTransaction(
        operations: [insertOp],
        selectionsBefore: const [Selection.collapsed(TextPosition(0, 5))],
        selectionsAfter: const [Selection.collapsed(TextPosition(0, 8))],
      );

      final events1 = syncManager.createChangeEvents(buffer, tx1);
      expect(events1.length, equals(1));
      expect(events1.first.text, equals(' my'));
      expect(events1.first.range, equals(const LspRange(LspPosition(0, 5), LspPosition(0, 5))));
      expect(syncManager.version, equals(2));

      insertOp.apply(buffer); // Buffer is now "hello my world"

      final deleteOp = DeleteOperation(5, 3, deletedText: ' my');
      final tx2 = EditorTransaction(
        operations: [deleteOp],
        selectionsBefore: const [Selection.collapsed(TextPosition(0, 8))],
        selectionsAfter: const [Selection.collapsed(TextPosition(0, 5))],
      );

      final events2 = syncManager.createChangeEvents(buffer, tx2);
      expect(events2.length, equals(1));
      expect(events2.first.text, equals(''));
      expect(events2.first.range, equals(const LspRange(LspPosition(0, 5), LspPosition(0, 8))));
      expect(syncManager.version, equals(3));
    });
  });
}
