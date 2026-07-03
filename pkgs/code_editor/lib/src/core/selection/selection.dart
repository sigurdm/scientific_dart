import 'text_position.dart';

/// Immutable representation of a text selection or cursor position.
class Selection {
  final TextPosition anchor;
  final TextPosition position;

  const Selection(this.anchor, this.position);

  const Selection.collapsed(TextPosition pos)
      : anchor = pos,
        position = pos;

  bool get isCollapsed => anchor == position;

  bool get isReversed => position.compareTo(anchor) < 0;

  TextPosition get start => isReversed ? position : anchor;

  TextPosition get end => isReversed ? anchor : position;

  Selection copyWith({
    TextPosition? anchor,
    TextPosition? position,
  }) {
    return Selection(
      anchor ?? this.anchor,
      position ?? this.position,
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Selection &&
          runtimeType == other.runtimeType &&
          anchor == other.anchor &&
          position == other.position;

  @override
  int get hashCode => Object.hash(anchor, position);

  @override
  String toString() => 'Selection(anchor: $anchor, position: $position)';
}
