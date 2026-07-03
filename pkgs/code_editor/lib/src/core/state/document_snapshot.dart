import '../selection/selection.dart';

/// Immutable snapshot of editor document state at a specific version.
class DocumentSnapshot {
  final String content;
  final int version;
  final int lineCount;
  final List<Selection> selections;
  final DateTime timestamp;

  DocumentSnapshot({
    required this.content,
    required this.version,
    required this.lineCount,
    required List<Selection> selections,
    DateTime? timestamp,
  })  : selections = List.unmodifiable(selections),
        timestamp = timestamp ?? DateTime.now();

  @override
  String toString() =>
      'DocumentSnapshot(v$version, lines: $lineCount, length: ${content.length})';
}
