enum TextAffinity { upstream, downstream }

/// Represents an immutable 0-indexed position within a text document.
class TextPosition implements Comparable<TextPosition> {
  final int line;
  final int column;
  final TextAffinity affinity;

  const TextPosition(
    this.line,
    this.column, {
    this.affinity = TextAffinity.downstream,
  });

  @override
  int compareTo(TextPosition other) {
    if (line != other.line) {
      return line.compareTo(other.line);
    }
    return column.compareTo(other.column);
  }

  /// Returns `true` if this position comes before [other].
  bool operator <(TextPosition other) => compareTo(other) < 0;

  /// Returns `true` if this position comes before or at [other].
  bool operator <=(TextPosition other) => compareTo(other) <= 0;

  /// Returns `true` if this position comes after [other].
  bool operator >(TextPosition other) => compareTo(other) > 0;

  /// Returns `true` if this position comes after or at [other].
  bool operator >=(TextPosition other) => compareTo(other) >= 0;

  TextPosition copyWith({int? line, int? column, TextAffinity? affinity}) {
    return TextPosition(
      line ?? this.line,
      column ?? this.column,
      affinity: affinity ?? this.affinity,
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is TextPosition &&
          runtimeType == other.runtimeType &&
          line == other.line &&
          column == other.column &&
          affinity == other.affinity;

  @override
  int get hashCode => Object.hash(line, column, affinity);

  @override
  String toString() => 'TextPosition($line, $column, affinity: $affinity)';
}
