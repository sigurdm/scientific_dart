import 'folding_manager.dart';
import 'line_height_tree.dart';

class VirtualLayoutSlice {
  final int firstVisibleLine;
  final int lastVisibleLine;
  final int firstRenderedLine;
  final int lastRenderedLine;
  final double topOffset;
  final double totalHeight;
  final List<int> visibleLineIndices;

  const VirtualLayoutSlice({
    required this.firstVisibleLine,
    required this.lastVisibleLine,
    required this.firstRenderedLine,
    required this.lastRenderedLine,
    required this.topOffset,
    required this.totalHeight,
    required this.visibleLineIndices,
  });

  int get renderedLineCount => visibleLineIndices.length;
}

class VirtualLayoutCalculator {
  static VirtualLayoutSlice computeLayout({
    required double scrollTop,
    required double viewportHeight,
    required LineHeightTree lineHeightTree,
    FoldingManager? foldingManager,
    int overscan = 3,
  }) {
    if (lineHeightTree.length == 0) {
      return const VirtualLayoutSlice(
        firstVisibleLine: 0,
        lastVisibleLine: 0,
        firstRenderedLine: 0,
        lastRenderedLine: 0,
        topOffset: 0.0,
        totalHeight: 0.0,
        visibleLineIndices: [],
      );
    }

    final totalHeight = lineHeightTree.totalHeight;
    final clampedScrollTop = scrollTop.clamp(0.0, totalHeight > 0 ? totalHeight : 0.0);
    final bottomY = clampedScrollTop + viewportHeight;

    int firstVis = lineHeightTree.lineAtHeight(clampedScrollTop);
    int lastVis = lineHeightTree.lineAtHeight(bottomY);

    if (firstVis < 0) firstVis = 0;
    if (lastVis >= lineHeightTree.length) lastVis = lineHeightTree.length - 1;
    if (lastVis < firstVis) lastVis = firstVis;

    int firstRendered = (firstVis - overscan).clamp(0, lineHeightTree.length - 1);
    int lastRendered = (lastVis + overscan).clamp(0, lineHeightTree.length - 1);

    final visibleIndices = <int>[];
    for (int i = firstRendered; i <= lastRendered; i++) {
      if (foldingManager == null || !foldingManager.isLineHidden(i)) {
        visibleIndices.add(i);
      }
    }

    final topOffset = lineHeightTree.getLineTop(firstRendered);

    return VirtualLayoutSlice(
      firstVisibleLine: firstVis,
      lastVisibleLine: lastVis,
      firstRenderedLine: firstRendered,
      lastRenderedLine: lastRendered,
      topOffset: topOffset,
      totalHeight: totalHeight,
      visibleLineIndices: visibleIndices,
    );
  }
}
