import '../selection/selection_model.dart';

/// Handler for mouse selection events (click, drag, double-click, triple-click).
final class MouseSelectionHandler {
  /// Handles single click event to position cursor at [targetPosition].
  static TextSelection handleSingleClick(TextPosition targetPosition) {
    return TextSelection.collapsed(targetPosition);
  }

  /// Handles mouse drag update event, keeping [currentSelection.base] fixed and setting extent to [targetPosition].
  static TextSelection handleDragUpdate(
    TextSelection currentSelection,
    TextPosition targetPosition,
  ) {
    return currentSelection.copyWith(extent: targetPosition);
  }

  /// Handles double click event, selecting the word surrounding [targetPosition].
  static TextSelection handleDoubleClick(
    List<String> lines,
    TextPosition targetPosition,
  ) {
    return SelectionModel.getWordBoundary(lines, targetPosition);
  }

  /// Handles triple click event, selecting the entire line surrounding [targetPosition].
  static TextSelection handleTripleClick(
    List<String> lines,
    TextPosition targetPosition,
  ) {
    if (lines.isEmpty) {
      return TextSelection.collapsed(const TextPosition(0, 0));
    }

    final lineIndex = targetPosition.line.clamp(0, lines.length - 1);
    final isLastLine = lineIndex == lines.length - 1;

    final basePos = TextPosition(lineIndex, 0);
    final extentPos = isLastLine
        ? TextPosition(lineIndex, lines[lineIndex].length)
        : TextPosition(lineIndex + 1, 0);

    return TextSelection(base: basePos, extent: extentPos);
  }
}
