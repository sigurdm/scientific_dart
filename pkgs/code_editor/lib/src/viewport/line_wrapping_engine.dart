import 'font_metrics.dart';

/// Represents a 2D coordinate on a wrapped line.
class DisplayPosition {
  final int subLineIndex;
  final int subLineColumn;

  const DisplayPosition(this.subLineIndex, this.subLineColumn);

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is DisplayPosition &&
          runtimeType == other.runtimeType &&
          subLineIndex == other.subLineIndex &&
          subLineColumn == other.subLineColumn;

  @override
  int get hashCode => Object.hash(subLineIndex, subLineColumn);

  @override
  String toString() => 'DisplayPosition(subLine: $subLineIndex, col: $subLineColumn)';
}

/// Result of line wrapping computation.
class LineWrapResult {
  /// Offsets in character string where soft wraps occur.
  final List<int> breakOffsets;

  const LineWrapResult(this.breakOffsets);

  int get subLineCount => breakOffsets.length + 1;

  /// Maps a document column (character offset in line) to (subLineIndex, subLineColumn).
  DisplayPosition documentColumnToDisplay(int docColumn) {
    if (breakOffsets.isEmpty || docColumn <= 0) {
      return DisplayPosition(0, docColumn);
    }

    int subLineIdx = 0;
    int prevOffset = 0;

    for (int i = 0; i < breakOffsets.length; i++) {
      final brk = breakOffsets[i];
      if (docColumn < brk) {
        return DisplayPosition(subLineIdx, docColumn - prevOffset);
      }
      prevOffset = brk;
      subLineIdx++;
    }

    return DisplayPosition(subLineIdx, docColumn - prevOffset);
  }

  /// Maps a (subLineIndex, subLineColumn) to document column offset.
  int displayColumnToDocument(int subLineIndex, int subLineColumn) {
    if (breakOffsets.isEmpty || subLineIndex <= 0) {
      return subLineColumn;
    }

    if (subLineIndex > breakOffsets.length) {
      subLineIndex = breakOffsets.length;
    }

    int startOffset = subLineIndex == 0 ? 0 : breakOffsets[subLineIndex - 1];
    return startOffset + subLineColumn;
  }
}

class LineWrappingEngine {
  /// Computes line wrap breaks for [text] fitting within [viewportWidth].
  static LineWrapResult computeWrap(
    String text,
    double viewportWidth,
    TextMeasurer measurer, {
    bool wordWrap = true,
  }) {
    if (text.isEmpty || viewportWidth <= 0.0) {
      return const LineWrapResult([]);
    }

    final totalWidth = measurer.measureWidth(text);
    if (totalWidth <= viewportWidth) {
      return const LineWrapResult([]);
    }

    final breakOffsets = <int>[];
    int lineStart = 0;
    int lastWordBoundary = -1;
    int idx = 0;

    while (idx < text.length) {
      final substr = text.substring(lineStart, idx + 1);
      final width = measurer.measureWidth(substr);

      final char = text[idx];
      if (wordWrap && (char == ' ' || char == '\t' || char == '-' || char == '_' || char == '.' || char == '/')) {
        lastWordBoundary = idx + 1;
      }

      if (width > viewportWidth) {
        int breakAt;
        if (wordWrap && lastWordBoundary > lineStart && lastWordBoundary <= idx) {
          breakAt = lastWordBoundary;
        } else {
          breakAt = idx > lineStart ? idx : idx + 1;
        }

        breakOffsets.add(breakAt);
        lineStart = breakAt;
        idx = breakAt;
        lastWordBoundary = -1;
      } else {
        idx++;
      }
    }

    return LineWrapResult(breakOffsets);
  }
}
