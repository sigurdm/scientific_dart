import 'package:code_editor/code_editor.dart';
import 'package:test/test.dart';

void main() {
  group('FoldingManager Tests', () {
    test('scanCodeBlocks detects brace blocks and toggles fold state', () {
      final codeLines = [
        'void main() {',
        '  print(1);',
        '  print(2);',
        '}',
        'void other() {',
        '  print(3);',
        '}',
      ];

      final folding = FoldingManager();
      folding.scanCodeBlocks(codeLines);

      expect(folding.regions.length, 2);
      expect(folding.isFoldHeader(0), isTrue);
      expect(folding.isFoldHeader(4), isTrue);
      expect(folding.isLineHidden(1), isFalse);

      // Collapse region at line 0
      folding.toggleFold(0);
      expect(folding.getRegionAt(0)?.isCollapsed, isTrue);
      expect(folding.isLineHidden(1), isTrue);
      expect(folding.isLineHidden(2), isTrue);
      expect(folding.isLineHidden(3), isFalse); // closing brace line is visible
      expect(
        folding.getVisibleLineCount(codeLines.length),
        5,
      ); // 7 - 2 hidden lines

      // Toggle back to expand
      folding.toggleFold(0);
      expect(folding.getRegionAt(0)?.isCollapsed, isFalse);
      expect(folding.isLineHidden(1), isFalse);
      expect(folding.getVisibleLineCount(codeLines.length), 7);
    });
  });
}
