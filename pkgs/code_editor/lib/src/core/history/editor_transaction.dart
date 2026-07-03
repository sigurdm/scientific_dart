import '../selection/selection.dart';
import 'edit_operation.dart';

/// Represents an atomic batch of edit operations on a document text buffer.
class EditorTransaction {
  final List<EditOperation> operations;
  final List<Selection> selectionsBefore;
  final List<Selection> selectionsAfter;
  final DateTime timestamp;

  EditorTransaction({
    required this.operations,
    required this.selectionsBefore,
    required this.selectionsAfter,
    DateTime? timestamp,
  }) : timestamp = timestamp ?? DateTime.now();

  /// Inverts the transaction for undo execution.
  EditorTransaction invert() {
    final invertedOps = operations.reversed.map((op) => op.invert()).toList();
    return EditorTransaction(
      operations: invertedOps,
      selectionsBefore: selectionsAfter,
      selectionsAfter: selectionsBefore,
      timestamp: DateTime.now(),
    );
  }
}
