/// ANSI VT100 Terminal renderer backend.
///
/// Features double-buffered ANSI 24-bit TrueColor cell matrix rendering,
/// minimum stdout delta updates, and raw mode stdin escape sequence parser.
library code_editor.backends.terminal;

export '../src/backends/terminal/vt100_encoder.dart';
export '../src/backends/terminal/terminal_input_parser.dart';
export '../src/backends/terminal/terminal_renderer.dart';
