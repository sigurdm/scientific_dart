import 'package:code_editor/syntax.dart';
import 'package:test/test.dart';

void main() {
  group('IncrementalTokenizer', () {
    late TextMateLexer lexer;
    late IncrementalTokenizer incrementalTokenizer;

    setUp(() {
      final rules = [
        TextMateRule(
          id: 'comment_block',
          name: 'comment.block.dart',
          type: TokenType.comment,
          begin: RegExp(r'/\*'),
          end: RegExp(r'\*/'),
        ),
        TextMateRule(
          id: 'keyword_class',
          name: 'keyword.declaration.dart',
          type: TokenType.keyword,
          match: RegExp(r'\b(class|var|void|int)\b'),
        ),
      ];
      lexer = TextMateLexer(rootRules: rules);
      incrementalTokenizer = IncrementalTokenizer(tokenizer: lexer);
    });

    test('initializes document and tokenizes all lines', () {
      final doc = [
        'class Foo {',
        '  /* start block',
        '     inside block */',
        '  var x = 1;',
        '}',
      ];

      incrementalTokenizer.setDocument(doc);
      expect(incrementalTokenizer.cachedLineStates.length, equals(5));
      expect(incrementalTokenizer.lastTokenizedLineCount, equals(5));

      expect(incrementalTokenizer.getLineState(0), isA<EmptyLineState>());
      expect(incrementalTokenizer.getLineState(1), isA<StackLineState>());
      final s1 = incrementalTokenizer.getLineState(1) as StackLineState;
      expect(s1.stack, equals(['comment_block']));

      expect(incrementalTokenizer.getLineState(2), isA<EmptyLineState>());
    });

    test('early state convergence halts re-tokenization', () {
      final doc = List<String>.generate(100, (i) => 'var x = $i;');
      doc[5] = '/* comment start';
      doc[6] = 'comment body';
      doc[7] = 'comment end */';

      incrementalTokenizer.setDocument(doc);
      expect(incrementalTokenizer.lastTokenizedLineCount, equals(100));

      doc[50] = 'var x = 999;';
      incrementalTokenizer.updateDocument(doc, startLineIndex: 50);

      expect(incrementalTokenizer.lastTokenizedLineCount, lessThanOrEqualTo(2));
      expect(incrementalTokenizer.getTokensForLine(50).first.text, contains('var'));
    });

    test('propagates state changes downstream when convergence does not immediately match', () {
      final doc = [
        'var a = 1;',
        'var b = 2;',
        'var c = 3;',
        'var d = 4;',
      ];

      incrementalTokenizer.setDocument(doc);

      doc[0] = '/* comment start';
      incrementalTokenizer.updateDocument(doc, startLineIndex: 0);

      expect(incrementalTokenizer.getLineState(0), isA<StackLineState>());
      expect(incrementalTokenizer.getLineState(1), isA<StackLineState>());
    });
  });
}
