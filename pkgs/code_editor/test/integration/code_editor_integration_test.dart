import 'package:code_editor/backends/flutter.dart';
import 'package:code_editor/backends/html.dart';
import 'package:code_editor/backends/terminal.dart';
import 'package:code_editor/core.dart';
import 'package:code_editor/lsp.dart';
import 'package:code_editor/render.dart';
import 'package:code_editor/syntax.dart';
import 'package:code_editor/viewport.dart';
import 'package:test/test.dart';

void main() {
  group('Code Editor End-to-End Integration Tests', () {
    late PieceTreeTextBuffer buffer;
    late TextMateLexer lexer;
    late LineHeightTree heightTree;

    setUp(() {
      buffer = PieceTreeTextBuffer('void main() {\n  int count = 42;\n}');
      lexer = TextMateLexer(
        rootRules: [
          TextMateRule(
            id: 'keyword',
            name: 'keyword.control',
            type: TokenType.keyword,
            match: RegExp(r'\b(void|int|return)\b'),
          ),
          TextMateRule(
            id: 'identifier',
            name: 'entity.name.function',
            type: TokenType.identifier,
            match: RegExp(r'\b(main|print)\b'),
          ),
        ],
      );
      heightTree = LineHeightTree(3, defaultHeight: 20.0);
    });

    test('Full pipeline: Buffer -> Edits -> Syntax -> LSP -> Layout -> Renderers', () {
      // 1. Verify Initial Text Buffer
      expect(buffer.lineCount, equals(3));
      expect(buffer.getLine(1).trimRight(), equals('  int count = 42;'));

      // 2. Apply Invertible Operations & Undo History
      final insertOp = InsertOperation(17, '\n  print("Count: \$count");');
      buffer.insert(17, '\n  print("Count: \$count");');
      expect(buffer.lineCount, equals(4));

      final inverseOp = insertOp.invert();
      expect(inverseOp, isA<DeleteOperation>());

      // 3. Syntax Tokenization & Scope Matching
      final line0Result = lexer.tokenizeLine(buffer.getLine(0), const EmptyLineState());
      expect(
        line0Result.tokens.any(
          (SyntaxToken t) => t.type == TokenType.keyword && t.offset == 0,
        ),
        isTrue,
      );

      final theme = ColorTheme(
        name: 'Dark+',
        rules: [
          const ThemeRule(scopes: ['keyword.control'], bold: true, foreground: 0xFF569CD6),
        ],
      );
      final style = theme.resolveStyle([StyleScope('keyword.control')]);
      expect(style.bold, isTrue);

      // 4. LSP Coordinate Translation & Sync
      const editorPos = TextPosition(0, 4);
      final lspPos = LspCoordinateTranslator.toLspPosition(buffer, editorPos);
      expect(lspPos.line, equals(0));
      expect(lspPos.character, equals(4));

      final convertedBack = LspCoordinateTranslator.toTextPosition(buffer, lspPos);
      expect(convertedBack.line, equals(0));
      expect(convertedBack.column, equals(4));

      final syncManager = LspDocumentSyncManager(uri: 'file:///main.dart');
      final snapshotBuffer = PieceTreeTextBuffer('void main() {\n  int count = 42;\n}');
      final tx = EditorTransaction(
        operations: [insertOp],
        selectionsBefore: const <Selection>[],
        selectionsAfter: const <Selection>[],
      );
      final changes = syncManager.createChangeEvents(snapshotBuffer, tx);
      expect(changes.length, equals(1));

      // 5. Calculate Virtual Layout & Viewport Payload
      heightTree.resize(buffer.lineCount, defaultHeight: 20.0);
      final slice = VirtualLayoutCalculator.computeLayout(
        scrollTop: 0,
        viewportHeight: 600,
        lineHeightTree: heightTree,
      );

      expect(slice.firstVisibleLine, equals(0));
      expect(slice.lastVisibleLine, equals(3));

      const renderViewport = RenderViewport(
        width: 800,
        height: 600,
        scrollX: 0,
        scrollY: 0,
        firstVisibleLine: 0,
        lastVisibleLine: 3,
        lines: [
          RenderLine(
            lineIndex: 0,
            text: 'void main() {',
            tokens: [
              RenderToken(
                text: 'void',
                startColumn: 0,
                endColumn: 4,
                style: RenderTokenStyle(foreground: '#569CD6'),
              ),
              RenderToken(
                text: ' main() {',
                startColumn: 4,
                endColumn: 13,
                style: RenderTokenStyle(foreground: '#DCDCAA'),
              ),
            ],
            top: 0,
            height: 20,
          ),
        ],
        selections: [SelectionRect(left: 0, top: 0, width: 32, height: 20)],
        carets: [CaretPosition(x: 32, y: 0, height: 20, line: 0, column: 4)],
        gutter: RenderGutter(
          width: 40,
          items: [GutterItem(lineIndex: 0, lineNumberText: '1')],
        ),
        lineHeight: 20,
        charWidth: 8,
      );

      // 6. Test Multi-Backend UI Renderers

      // 6a. HTML DOM Web Renderer
      final htmlRenderer = HtmlRenderer();
      htmlRenderer.attach('web_container');
      htmlRenderer.render(renderViewport);
      expect(htmlRenderer.isAttached, isTrue);

      // 6b. Flutter Canvas Renderer
      final flutterRenderer = FlutterRenderer();
      flutterRenderer.attach('flutter_canvas');
      flutterRenderer.render(renderViewport);
      expect(flutterRenderer.isAttached, isTrue);

      // 6c. ANSI Terminal Renderer
      final terminalRenderer = TerminalRenderer(rows: 10, cols: 40);
      terminalRenderer.attach('terminal_stdout');
      terminalRenderer.render(renderViewport);
      final ansiOutput = terminalRenderer.renderDeltaToString(renderViewport);
      expect(ansiOutput, contains(Vt100Encoder.reset));

      // 7. Keybinding Dispatcher Integration
      final keybindings = KeybindingRegistry();
      var saveTriggered = false;
      keybindings.register(
        KeyCombination.parse('Ctrl+S'),
        'editor.action.save',
        () {
          saveTriggered = true;
        },
      );

      final handled = keybindings.dispatch(
        const KeyCombination('S', ctrl: true),
      );
      expect(handled, isTrue);
      expect(saveTriggered, isTrue);
    });
  });
}
