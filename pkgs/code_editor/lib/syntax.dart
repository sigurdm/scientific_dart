/// Syntax highlighting and tokenization engine.
///
/// Features incremental line-state tokenization, TextMate lexing state machine,
/// hierarchical scope matching, and VS Code theme color resolution.
library code_editor.syntax;

export 'src/syntax/color_theme.dart';
export 'src/syntax/incremental_tokenizer.dart';
export 'src/syntax/line_state.dart';
export 'src/syntax/scope_matcher.dart';
export 'src/syntax/syntax_token.dart';
export 'src/syntax/syntax_tokenizer.dart';
export 'src/syntax/textmate_lexer.dart';
