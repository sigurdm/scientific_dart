/// Core library for the multi-backend code editor package.
///
/// Exports the core buffer engine, LSP 3.17 protocol adapters, syntax
/// tokenization engine, virtual viewport layout calculator, and abstract render model.
///
/// Platform-specific renderers (HTML, Flutter, ANSI Terminal) are exported
/// via `package:code_editor/backends/html.dart`, `package:code_editor/backends/flutter.dart`,
/// and `package:code_editor/backends/terminal.dart` respectively.
library code_editor;

export 'core.dart';
export 'lsp.dart';
export 'render.dart' hide DiagnosticSeverity;
export 'syntax.dart';
export 'viewport.dart';
