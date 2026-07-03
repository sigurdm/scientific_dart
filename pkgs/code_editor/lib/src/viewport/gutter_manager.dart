import 'folding_manager.dart';
import 'font_metrics.dart';

enum GitDiffState { none, added, modified, deleted }

enum DiagnosticSeverity { error, warning, info, hint }

enum FoldControlState { none, expanded, collapsed }

class GutterLineMetadata {
  final int lineIndex;
  final int lineNumber;
  final FoldControlState foldState;
  final GitDiffState gitDiff;
  final List<DiagnosticSeverity> diagnostics;

  const GutterLineMetadata({
    required this.lineIndex,
    required this.lineNumber,
    this.foldState = FoldControlState.none,
    this.gitDiff = GitDiffState.none,
    this.diagnostics = const [],
  });
}

class GutterManager {
  final Map<int, GitDiffState> _gitDiffs = {};
  final Map<int, List<DiagnosticSeverity>> _diagnostics = {};

  bool showLineNumbers;
  bool showFoldControls;
  bool showGitDiff;
  bool showDiagnostics;
  double extraPadding;

  GutterManager({
    this.showLineNumbers = true,
    this.showFoldControls = true,
    this.showGitDiff = true,
    this.showDiagnostics = true,
    this.extraPadding = 12.0,
  });

  void setGitDiff(int lineIndex, GitDiffState state) {
    if (state == GitDiffState.none) {
      _gitDiffs.remove(lineIndex);
    } else {
      _gitDiffs[lineIndex] = state;
    }
  }

  void setDiagnostics(int lineIndex, List<DiagnosticSeverity> list) {
    if (list.isEmpty) {
      _diagnostics.remove(lineIndex);
    } else {
      _diagnostics[lineIndex] = list;
    }
  }

  void clearGitDiffs() => _gitDiffs.clear();
  void clearDiagnostics() => _diagnostics.clear();

  double computeGutterWidth(int totalLines, FontMetrics fontMetrics) {
    double width = extraPadding;

    if (showLineNumbers) {
      final digits = totalLines.toString().length;
      width += digits * fontMetrics.characterWidth;
    }

    if (showFoldControls) {
      width += 16.0;
    }

    if (showGitDiff) {
      width += 4.0;
    }

    if (showDiagnostics) {
      width += 16.0;
    }

    return width;
  }

  GutterLineMetadata getGutterMetadata(int lineIndex, FoldingManager foldingManager) {
    FoldControlState foldState = FoldControlState.none;
    if (showFoldControls) {
      final region = foldingManager.getRegionAt(lineIndex);
      if (region != null) {
        foldState = region.isCollapsed
            ? FoldControlState.collapsed
            : FoldControlState.expanded;
      }
    }

    return GutterLineMetadata(
      lineIndex: lineIndex,
      lineNumber: lineIndex + 1,
      foldState: foldState,
      gitDiff: _gitDiffs[lineIndex] ?? GitDiffState.none,
      diagnostics: _diagnostics[lineIndex] ?? const [],
    );
  }
}
