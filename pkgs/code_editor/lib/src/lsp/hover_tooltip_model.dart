/// Hover Tooltip Data Model for LSP hover tooltips UI overlay.
library;

/// Represents an LSP markdown hover tooltip overlay anchored at text coordinates.
final class HoverTooltipModel {
  /// The X position (column index or pixel offset) of the hover target.
  final double x;

  /// The Y position (line index or pixel offset) of the hover target.
  final double y;

  /// Raw markdown string content of the tooltip.
  final String markdownContent;

  /// Optional signature or code header block (e.g. function signature or type signature).
  final String? signature;

  /// Whether the tooltip is currently visible.
  final bool isVisible;

  /// Creates a new [HoverTooltipModel].
  const HoverTooltipModel({
    required this.x,
    required this.y,
    required this.markdownContent,
    this.signature,
    this.isVisible = true,
  });

  /// Creates a copy of this model with modified parameters.
  HoverTooltipModel copyWith({
    double? x,
    double? y,
    String? markdownContent,
    String? signature,
    bool? isVisible,
  }) {
    return HoverTooltipModel(
      x: x ?? this.x,
      y: y ?? this.y,
      markdownContent: markdownContent ?? this.markdownContent,
      signature: signature ?? this.signature,
      isVisible: isVisible ?? this.isVisible,
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is HoverTooltipModel &&
          runtimeType == other.runtimeType &&
          x == other.x &&
          y == other.y &&
          markdownContent == other.markdownContent &&
          signature == other.signature &&
          isVisible == other.isVisible;

  @override
  int get hashCode => Object.hash(x, y, markdownContent, signature, isVisible);
}
