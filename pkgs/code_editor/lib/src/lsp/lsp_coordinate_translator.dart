import '../core/buffer/text_buffer.dart';
import '../core/selection/selection.dart';
import '../core/selection/text_position.dart';
import 'lsp_primitives.dart';

/// Bidirectional coordinate translator between Core Editor positions/selections
/// and LSP 3.17 positions/ranges (UTF-16 code units).
class LspCoordinateTranslator {
  /// Converts an editor [TextPosition] to an [LspPosition].
  static LspPosition toLspPosition(TextBuffer buffer, TextPosition pos) {
    final line = pos.line.clamp(0, buffer.lineCount - 1);
    final lineLen = buffer.getLineLength(line);
    final col = pos.column.clamp(0, lineLen);
    return LspPosition(line, col);
  }

  /// Converts an [LspPosition] to an editor [TextPosition].
  static TextPosition toTextPosition(TextBuffer buffer, LspPosition lspPos) {
    final line = lspPos.line.clamp(0, buffer.lineCount - 1);
    final lineLen = buffer.getLineLength(line);
    final col = lspPos.character.clamp(0, lineLen);
    return TextPosition(line, col);
  }

  /// Converts an editor [Selection] to an [LspRange].
  static LspRange toLspRange(TextBuffer buffer, Selection selection) {
    final startLsp = toLspPosition(buffer, selection.start);
    final endLsp = toLspPosition(buffer, selection.end);
    return LspRange(startLsp, endLsp);
  }

  /// Converts an [LspRange] to an editor [Selection].
  static Selection toSelection(TextBuffer buffer, LspRange range) {
    final startPos = toTextPosition(buffer, range.start);
    final endPos = toTextPosition(buffer, range.end);
    return Selection(startPos, endPos);
  }

  /// Converts a rune offset (Unicode code points) to UTF-16 code unit offset in [text].
  static int runeOffsetToCodeUnitOffset(String text, int runeOffset) {
    var currentRune = 0;
    var currentCodeUnit = 0;
    for (final char in text.runes) {
      if (currentRune >= runeOffset) break;
      currentCodeUnit += (char > 0xFFFF) ? 2 : 1;
      currentRune++;
    }
    return currentCodeUnit;
  }

  /// Converts a UTF-16 code unit offset to rune offset (Unicode code points) in [text].
  static int codeUnitOffsetToRuneOffset(String text, int codeUnitOffset) {
    var currentCodeUnit = 0;
    var currentRune = 0;
    for (final char in text.runes) {
      if (currentCodeUnit >= codeUnitOffset) break;
      currentCodeUnit += (char > 0xFFFF) ? 2 : 1;
      currentRune++;
    }
    return currentRune;
  }
}
