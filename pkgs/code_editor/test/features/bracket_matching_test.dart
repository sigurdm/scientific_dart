import 'package:code_editor/code_editor.dart';
import 'package:test/test.dart';

void main() {
  group('BracketMatcher Tests', () {
    const matcher = BracketMatcher();

    test('finds matching parentheses on same line', () {
      final buffer = PieceTreeTextBuffer('final x = (1 + 2);');
      // Position at opening paren '(' (column 10)
      final res = matcher.findMatchingBracket(
        buffer,
        const TextPosition(0, 10),
      );
      expect(res, isNotNull);
      expect(res!.isOpening, isTrue);
      expect(res.match, const TextPosition(0, 16));
    });

    test('finds matching closing parenthesis backward', () {
      final buffer = PieceTreeTextBuffer('final x = (1 + 2);');
      // Position at closing paren ')' (column 16)
      final res = matcher.findMatchingBracket(
        buffer,
        const TextPosition(0, 16),
      );
      expect(res, isNotNull);
      expect(res!.isOpening, isFalse);
      expect(res.match, const TextPosition(0, 10));
    });

    test('handles multiline nested braces correctly', () {
      final code = '''
void main() {
  if (true) {
    print('hello');
  }
}''';
      final buffer = PieceTreeTextBuffer(code);
      // Main opening brace at line 0, column 12
      final res = matcher.findMatchingBracket(
        buffer,
        const TextPosition(0, 12),
      );
      expect(res, isNotNull);
      expect(res!.isOpening, isTrue);
      expect(res.match.line, 4);
      expect(res.match.column, 0);

      // Inner if opening brace at line 1, column 12
      final innerRes = matcher.findMatchingBracket(
        buffer,
        const TextPosition(1, 12),
      );
      expect(innerRes, isNotNull);
      expect(innerRes!.match.line, 3);
      expect(innerRes.match.column, 2);
    });

    test('returns null for unclosed brackets or non-bracket characters', () {
      final buffer = PieceTreeTextBuffer('void foo() { var x = 1;');
      // Opening brace at col 11 without closing brace
      final res = matcher.findMatchingBracket(
        buffer,
        const TextPosition(0, 11),
      );
      expect(res, isNull);

      // Character 'v' at col 0
      final nonBracket = matcher.findMatchingBracket(
        buffer,
        const TextPosition(0, 0),
      );
      expect(nonBracket, isNull);
    });
  });
}
