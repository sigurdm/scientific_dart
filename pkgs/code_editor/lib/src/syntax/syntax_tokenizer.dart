import 'line_state.dart';
import 'syntax_token.dart';

/// Result of tokenizing a single line of code.
class LineTokenizationResult {
  final List<SyntaxToken> tokens;
  final LineState endState;

  const LineTokenizationResult({
    required this.tokens,
    required this.endState,
  });
}

/// Abstract line tokenization interface.
abstract class SyntaxTokenizer {
  /// Tokenizes a single line of text starting with [previousState].
  LineTokenizationResult tokenizeLine(String lineText, LineState previousState);
}
