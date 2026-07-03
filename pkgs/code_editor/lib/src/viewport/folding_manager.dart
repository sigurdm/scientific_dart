import 'package:meta/meta.dart';

@immutable
class FoldRegion {
  final int startLine;
  final int endLine;
  final bool isCollapsed;

  const FoldRegion({
    required this.startLine,
    required this.endLine,
    this.isCollapsed = true,
  });

  FoldRegion copyWith({bool? isCollapsed}) {
    return FoldRegion(
      startLine: startLine,
      endLine: endLine,
      isCollapsed: isCollapsed ?? this.isCollapsed,
    );
  }

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is FoldRegion &&
          runtimeType == other.runtimeType &&
          startLine == other.startLine &&
          endLine == other.endLine &&
          isCollapsed == other.isCollapsed;

  @override
  int get hashCode => Object.hash(startLine, endLine, isCollapsed);

  @override
  String toString() => 'FoldRegion($startLine..$endLine, collapsed: $isCollapsed)';
}

class FoldingManager {
  final List<FoldRegion> _regions = [];

  List<FoldRegion> get regions => List.unmodifiable(_regions);

  void addRegion(int startLine, int endLine) {
    if (startLine >= endLine) return;

    final idx = _regions.indexWhere((r) => r.startLine == startLine);
    if (idx != -1) {
      _regions[idx] = FoldRegion(startLine: startLine, endLine: endLine, isCollapsed: _regions[idx].isCollapsed);
    } else {
      _regions.add(FoldRegion(startLine: startLine, endLine: endLine, isCollapsed: false));
      _regions.sort((a, b) => a.startLine.compareTo(b.startLine));
    }
  }

  void removeRegion(int startLine) {
    _regions.removeWhere((r) => r.startLine == startLine);
  }

  void collapse(int startLine) {
    _setCollapsed(startLine, true);
  }

  void expand(int startLine) {
    _setCollapsed(startLine, false);
  }

  void toggleFold(int startLine) {
    final r = getRegionAt(startLine);
    if (r != null) {
      _setCollapsed(startLine, !r.isCollapsed);
    }
  }

  void _setCollapsed(int startLine, bool collapsed) {
    final idx = _regions.indexWhere((r) => r.startLine == startLine);
    if (idx != -1) {
      _regions[idx] = _regions[idx].copyWith(isCollapsed: collapsed);
    }
  }

  FoldRegion? getRegionAt(int startLine) {
    for (final r in _regions) {
      if (r.startLine == startLine) return r;
    }
    return null;
  }

  bool isFoldHeader(int lineIndex) {
    return _regions.any((r) => r.startLine == lineIndex);
  }

  bool isLineHidden(int lineIndex) {
    for (final r in _regions) {
      if (r.isCollapsed && lineIndex > r.startLine && lineIndex <= r.endLine) {
        return true;
      }
    }
    return false;
  }

  int documentToVisualLine(int docLine, int totalLines) {
    if (docLine <= 0) return 0;
    int visual = 0;
    for (int i = 0; i < docLine && i < totalLines; i++) {
      if (!isLineHidden(i)) {
        visual++;
      }
    }
    return visual;
  }

  int visualToDocumentLine(int visualLine, int totalLines) {
    if (visualLine <= 0) return 0;
    int currentVisual = 0;
    for (int i = 0; i < totalLines; i++) {
      if (!isLineHidden(i)) {
        if (currentVisual == visualLine) return i;
        currentVisual++;
      }
    }
    return totalLines - 1;
  }

  int getVisibleLineCount(int totalLines) {
    int count = 0;
    for (int i = 0; i < totalLines; i++) {
      if (!isLineHidden(i)) count++;
    }
    return count;
  }
}
