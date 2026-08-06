import 'text_position.dart';

/// Immutable representation of a text selection or cursor position.
class Selection {
  /// The position at which the selection originated (anchor / base).
  final TextPosition anchor;

  /// The position at which the selection currently terminates (position / extent).
  final TextPosition position;

  /// The affinity of the selection.
  final TextAffinity affinity;

  /// Creates a [Selection] with positional [anchor], optional [position], and optional [affinity].
  const Selection(
    this.anchor, [
    TextPosition? position,
    this.affinity = TextAffinity.downstream,
  ]) : position = position ?? anchor;

  /// Creates a [Selection] with named [base] and [extent] positions.
  const Selection.range({
    required TextPosition base,
    required TextPosition extent,
    this.affinity = TextAffinity.downstream,
  }) : anchor = base,
       position = extent;

  /// Creates a collapsed selection (caret position) at [pos].
  const Selection.collapsed(
    TextPosition pos, {
    this.affinity = TextAffinity.downstream,
  }) : anchor = pos,
       position = pos;

  /// Base position getter.
  TextPosition get base => anchor;

  /// Extent position getter.
  TextPosition get extent => position;

  /// Whether the selection is collapsed to a single caret position.
  bool get isCollapsed => anchor == position;

  /// Whether the selection is reversed (position comes before anchor).
  bool get isReversed => position.compareTo(anchor) < 0;

  /// The starting position of the selection (inclusive).
  TextPosition get start => isReversed ? position : anchor;

  /// The ending position of the selection (exclusive).
  TextPosition get end => isReversed ? anchor : position;

  /// Returns a new [Selection] with updated fields.
  Selection copyWith({
    TextPosition? anchor,
    TextPosition? position,
    TextPosition? base,
    TextPosition? extent,
    TextAffinity? affinity,
  }) {
    return Selection.range(
      base: base ?? anchor ?? this.anchor,
      extent: extent ?? position ?? this.position,
      affinity: affinity ?? this.affinity,
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is Selection &&
          runtimeType == other.runtimeType &&
          anchor == other.anchor &&
          position == other.position &&
          affinity == other.affinity;

  @override
  int get hashCode => Object.hash(anchor, position, affinity);

  @override
  String toString() => 'Selection(base: $anchor, extent: $position)';
}

/// Representation of a text selection using named parameters.
class TextSelection extends Selection {
  /// Creates a [TextSelection] with named [base] and [extent] positions.
  const TextSelection({
    required TextPosition base,
    required TextPosition extent,
    TextAffinity affinity = TextAffinity.downstream,
  }) : super.range(base: base, extent: extent, affinity: affinity);

  /// Creates a collapsed [TextSelection] at [position].
  const TextSelection.collapsed(
    TextPosition position, {
    TextAffinity affinity = TextAffinity.downstream,
  }) : super.collapsed(position, affinity: affinity);

  @override
  TextSelection copyWith({
    TextPosition? anchor,
    TextPosition? position,
    TextPosition? base,
    TextPosition? extent,
    TextAffinity? affinity,
  }) {
    return TextSelection(
      base: base ?? anchor ?? this.anchor,
      extent: extent ?? position ?? this.position,
      affinity: affinity ?? this.affinity,
    );
  }
}
