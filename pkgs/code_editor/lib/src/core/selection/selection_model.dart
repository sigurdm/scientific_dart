import '../buffer/text_buffer.dart';
import 'selection.dart';
import 'text_position.dart';

/// Multi-cursor selection manager with selection transformations, merging,
/// word & line boundary helpers.
class SelectionModel {
  List<Selection> _selections = [
    const Selection.collapsed(TextPosition(0, 0)),
  ];

  List<Selection> get selections => List.unmodifiable(_selections);

  Selection get primary => _selections.first;

  void setSelections(List<Selection> newSelections) {
    if (newSelections.isEmpty) return;
    _selections = List.from(newSelections);
    mergeOverlapping();
  }

  void setSingleCursor(TextPosition position) {
    _selections = [Selection.collapsed(position)];
  }

  void addCursor(TextPosition position) {
    _selections.add(Selection.collapsed(position));
    mergeOverlapping();
  }

  /// Sorts selections and merges overlapping ranges into single unified selections.
  void mergeOverlapping() {
    if (_selections.length <= 1) return;

    // Sort by start position
    _selections.sort((a, b) => a.start.compareTo(b.start));

    final merged = <Selection>[];
    var current = _selections.first;

    for (var i = 1; i < _selections.length; i++) {
      final next = _selections[i];
      if (next.start.compareTo(current.end) <= 0) {
        // Overlaps or touches
        final newEnd = (current.end.compareTo(next.end) >= 0) ? current.end : next.end;
        if (current.isReversed) {
          current = Selection(newEnd, current.start);
        } else {
          current = Selection(current.start, newEnd);
        }
      } else {
        merged.add(current);
        current = next;
      }
    }
    merged.add(current);
    _selections = merged;
  }

  /// Selects the word surrounding [pos] in [buffer].
  Selection selectWordAt(TextBuffer buffer, TextPosition pos) {
    if (pos.line >= buffer.lineCount) return Selection.collapsed(pos);
    final lineText = buffer.getLine(pos.line);

    var startCol = pos.column.clamp(0, lineText.length);
    var endCol = pos.column.clamp(0, lineText.length);

    bool isWordChar(String ch) {
      if (ch.isEmpty) return false;
      final code = ch.codeUnitAt(0);
      return (code >= 65 && code <= 90) ||
          (code >= 97 && code <= 122) ||
          (code >= 48 && code <= 57) ||
          code == 95; // A-Z, a-z, 0-9, _
    }

    while (startCol > 0 && isWordChar(lineText[startCol - 1])) {
      startCol--;
    }
    while (endCol < lineText.length && isWordChar(lineText[endCol])) {
      endCol++;
    }

    return Selection(
      TextPosition(pos.line, startCol),
      TextPosition(pos.line, endCol),
    );
  }

  /// Selects the entire line at [lineIndex].
  Selection selectLineAt(TextBuffer buffer, int lineIndex) {
    final line = lineIndex.clamp(0, buffer.lineCount - 1);
    final lineLen = buffer.getLineLength(line);
    return Selection(
      TextPosition(line, 0),
      TextPosition(line, lineLen),
    );
  }

  /// Adjusts selections after an insertion at [insertOffset] of [newText].
  void transformOnInsert(TextBuffer bufferBeforeInsert, int insertOffset, String newText) {
    final (insLine, insCol) = bufferBeforeInsert.getLineAndColumnAt(insertOffset);

    int insEndLine = insLine;
    int insEndCol = insCol;
    final lines = newText.split('\n');
    if (lines.length == 1) {
      insEndCol += lines.first.length;
    } else {
      insEndLine += lines.length - 1;
      insEndCol = lines.last.length;
    }

    final newSelections = <Selection>[];

    for (final sel in _selections) {
      final anchor = _shiftPositionOnInsert(sel.anchor, insLine, insCol, insEndLine, insEndCol, newText.length);
      final pos = _shiftPositionOnInsert(sel.position, insLine, insCol, insEndLine, insEndCol, newText.length);
      newSelections.add(Selection(anchor, pos));
    }

    _selections = newSelections;
    mergeOverlapping();
  }

  TextPosition _shiftPositionOnInsert(
    TextPosition p,
    int insLine,
    int insCol,
    int insEndLine,
    int insEndCol,
    int insertLen,
  ) {
    if (p.line < insLine) return p;
    if (p.line == insLine) {
      if (p.column < insCol) return p;
      // Affected by insert on same line
      final lineDiff = insEndLine - insLine;
      if (lineDiff == 0) {
        return TextPosition(p.line, p.column + insertLen);
      } else {
        return TextPosition(insEndLine, insEndCol + (p.column - insCol));
      }
    } else {
      // Below insertion
      final lineDiff = insEndLine - insLine;
      return TextPosition(p.line + lineDiff, p.column);
    }
  }

  /// Adjusts selections after deletion at [deleteOffset] of [deleteLen] code units.
  void transformOnDelete(TextBuffer bufferBeforeDelete, int deleteOffset, int deleteLen) {
    final (delStartLine, delStartCol) = bufferBeforeDelete.getLineAndColumnAt(deleteOffset);
    final (delEndLine, delEndCol) = bufferBeforeDelete.getLineAndColumnAt(deleteOffset + deleteLen);

    final newSelections = <Selection>[];

    for (final sel in _selections) {
      final anchor = _shiftPositionOnDelete(sel.anchor, delStartLine, delStartCol, delEndLine, delEndCol);
      final pos = _shiftPositionOnDelete(sel.position, delStartLine, delStartCol, delEndLine, delEndCol);
      newSelections.add(Selection(anchor, pos));
    }

    _selections = newSelections;
    mergeOverlapping();
  }

  TextPosition _shiftPositionOnDelete(
    TextPosition p,
    int delStartLine,
    int delStartCol,
    int delEndLine,
    int delEndCol,
  ) {
    if (p.line < delStartLine) return p;
    if (p.line == delStartLine && p.column <= delStartCol) return p;

    if (p.line > delStartLine && p.line < delEndLine) {
      // Inside deleted line range
      return TextPosition(delStartLine, delStartCol);
    }

    if (p.line == delEndLine) {
      if (p.column <= delEndCol) {
        return TextPosition(delStartLine, delStartCol);
      } else {
        return TextPosition(delStartLine, delStartCol + (p.column - delEndCol));
      }
    }

    // Below deleted range
    final lineDiff = delEndLine - delStartLine;
    return TextPosition(p.line - lineDiff, p.column);
  }
}
