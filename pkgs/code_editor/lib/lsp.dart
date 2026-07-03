/// Language Server Protocol (LSP) 3.17 integration adapters.
///
/// Provides UTF-16 coordinate translation, incremental document sync manager,
/// and feature adapters for completions, diagnostics, hover tooltips, symbols, and code actions.
library code_editor.lsp;

export 'src/lsp/lsp_code_action_adapter.dart';
export 'src/lsp/lsp_completion_adapter.dart';
export 'src/lsp/lsp_coordinate_translator.dart';
export 'src/lsp/lsp_diagnostic_adapter.dart';
export 'src/lsp/lsp_hover_adapter.dart';
export 'src/lsp/lsp_primitives.dart';
export 'src/lsp/lsp_symbol_adapter.dart';
export 'src/lsp/lsp_sync_manager.dart';
