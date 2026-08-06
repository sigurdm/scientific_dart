import 'package:code_editor/core.dart';
import 'package:code_editor/viewport.dart';
import 'package:test/test.dart';

void main() {
  group('VirtualLayoutCalculator', () {
    test('computes line wrapping virtual rows and character mappings', () {
      const engine = LineWrappingEngine(maxColumns: 10, enabled: true);
      final calc = VirtualLayoutCalculator(lineWrappingEngine: engine);

      final lines = ['short', 'a very long line of text that wraps'];

      calc.computeLayout(lines);

      expect(calc.totalVirtualRows, greaterThan(2));

      // Test line 0 mapping
      const pos0 = TextPosition(0, 3);
      final vPos0 = calc.documentToVirtualPosition(pos0);
      expect(vPos0.virtualRow, equals(0));
      expect(vPos0.virtualColumn, equals(3));

      final docPos0 = calc.virtualToDocumentPosition(vPos0);
      expect(docPos0, equals(pos0));

      // Test line 1 wrapped slice mapping
      final rowInfo1 = calc.getVirtualRowInfo(1);
      expect(rowInfo1.lineIndex, equals(1));
    });
  });
}
