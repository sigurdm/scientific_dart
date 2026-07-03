import '../buffer/text_buffer.dart';

/// Abstract invertible edit operation.
abstract class EditOperation {
  int get offset;

  /// Applies the operation to [buffer].
  void apply(TextBuffer buffer);

  /// Returns the inverse operation that undoes this change.
  EditOperation invert();
}

/// Insertion edit operation.
class InsertOperation implements EditOperation {
  @override
  final int offset;
  final String text;

  InsertOperation(this.offset, this.text);

  @override
  void apply(TextBuffer buffer) {
    buffer.insert(offset, text);
  }

  @override
  EditOperation invert() {
    return DeleteOperation(offset, text.length, deletedText: text);
  }
}

/// Deletion edit operation.
class DeleteOperation implements EditOperation {
  @override
  final int offset;
  final int length;
  final String deletedText;

  DeleteOperation(this.offset, this.length, {required this.deletedText});

  @override
  void apply(TextBuffer buffer) {
    buffer.delete(offset, length);
  }

  @override
  EditOperation invert() {
    return InsertOperation(offset, deletedText);
  }
}
