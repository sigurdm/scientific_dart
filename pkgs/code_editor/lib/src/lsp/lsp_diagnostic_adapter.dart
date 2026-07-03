import 'lsp_primitives.dart';

class DiagnosticSquiggle {
  final LspRange range;
  final String message;
  final int severity; // 1: Error, 2: Warning, 3: Info, 4: Hint
  final String? source;

  DiagnosticSquiggle({
    required this.range,
    required this.message,
    required this.severity,
    this.source,
  });
}

/// Store for LSP diagnostic squiggles and gutter indicators.
class LspDiagnosticAdapter {
  final List<LspDiagnostic> _diagnostics = [];

  List<LspDiagnostic> get diagnostics => List.unmodifiable(_diagnostics);

  void updateDiagnostics(List<LspDiagnostic> diagnostics) {
    _diagnostics.clear();
    _diagnostics.addAll(diagnostics);
  }

  void clear() {
    _diagnostics.clear();
  }

  /// Returns diagnostics that overlap with [lineIndex].
  List<LspDiagnostic> getDiagnosticsForLine(int lineIndex) {
    return _diagnostics.where((d) {
      return lineIndex >= d.range.start.line && lineIndex <= d.range.end.line;
    }).toList();
  }

  /// Returns diagnostic squiggles formatted for renderer consumption.
  List<DiagnosticSquiggle> getSquigglesForLine(int lineIndex) {
    return getDiagnosticsForLine(lineIndex).map((d) {
      return DiagnosticSquiggle(
        range: d.range,
        message: d.message,
        severity: d.severity ?? 1,
        source: d.source,
      );
    }).toList();
  }
}
