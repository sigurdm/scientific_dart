import '../core/buffer/text_buffer.dart';
import '../core/history/edit_operation.dart';
import '../core/history/editor_transaction.dart';
import 'lsp_primitives.dart';

/// Manages incremental LSP 3.17 document synchronization (`didChange` notifications).
class LspDocumentSyncManager {
  final String uri;
  int version;

  LspDocumentSyncManager({
    required this.uri,
    this.version = 1,
  });

  /// Transforms an [EditorTransaction] into a list of incremental [LspTextDocumentContentChangeEvent]s.
  List<LspTextDocumentContentChangeEvent> createChangeEvents(
    TextBuffer bufferBeforeTx,
    EditorTransaction tx,
  ) {
    version++;
    final events = <LspTextDocumentContentChangeEvent>[];

    for (final op in tx.operations) {
      if (op is InsertOperation) {
        final (line, col) = bufferBeforeTx.getLineAndColumnAt(op.offset);
        final pos = LspPosition(line, col);
        events.add(LspTextDocumentContentChangeEvent(
          range: LspRange(pos, pos),
          rangeLength: 0,
          text: op.text,
        ));
      } else if (op is DeleteOperation) {
        final (startLine, startCol) = bufferBeforeTx.getLineAndColumnAt(op.offset);
        final (endLine, endCol) = bufferBeforeTx.getLineAndColumnAt(op.offset + op.length);
        final startPos = LspPosition(startLine, startCol);
        final endPos = LspPosition(endLine, endCol);

        events.add(LspTextDocumentContentChangeEvent(
          range: LspRange(startPos, endPos),
          rangeLength: op.length,
          text: '',
        ));
      }
    }

    return events;
  }
}
