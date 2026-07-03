import 'package:code_editor/viewport.dart';
import 'package:test/test.dart';

void main() {
  group('VirtualLayoutCalculator', () {
    test('computes visible viewport slice and overscan range', () {
      final tree = LineHeightTree(100, defaultHeight: 20.0);

      final slice = VirtualLayoutCalculator.computeLayout(
        scrollTop: 200.0,
        viewportHeight: 100.0,
        lineHeightTree: tree,
        overscan: 3,
      );

      expect(slice.firstVisibleLine, equals(10));
      expect(slice.lastVisibleLine, equals(15));
      expect(slice.firstRenderedLine, equals(7));
      expect(slice.lastRenderedLine, equals(18));
      expect(slice.topOffset, equals(140.0));
      expect(slice.totalHeight, equals(2000.0));
      expect(slice.renderedLineCount, equals(12));
    });

    test('respects FoldingManager hidden lines in virtual slice', () {
      final tree = LineHeightTree(20, defaultHeight: 20.0);
      final folding = FoldingManager();
      folding.addRegion(2, 5);
      folding.collapse(2);

      final slice = VirtualLayoutCalculator.computeLayout(
        scrollTop: 0.0,
        viewportHeight: 200.0,
        lineHeightTree: tree,
        foldingManager: folding,
        overscan: 0,
      );

      expect(slice.visibleLineIndices, isNot(contains(3)));
      expect(slice.visibleLineIndices, isNot(contains(4)));
      expect(slice.visibleLineIndices, isNot(contains(5)));
      expect(slice.visibleLineIndices, contains(2));
      expect(slice.visibleLineIndices, contains(6));
    });
  });
}
