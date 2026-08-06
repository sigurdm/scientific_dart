import 'package:code_editor/syntax.dart';
import 'package:test/test.dart';

void main() {
  group('IncrementalTokenizer Tests', () {
    test('Tokenizes line tokens with scope rules', () {
      final theme = ColorTheme.darkPlus();
      final cache = StyleCache(theme);

      final scopes = [StyleScope('keyword.control.dart')];
      final style = cache.getStyle(scopes);
      expect(style.foreground, equals(0xFF569CD6));
    });
  });
}
