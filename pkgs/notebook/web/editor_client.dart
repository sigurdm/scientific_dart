import 'dart:js_interop';
import 'package:web/web.dart' as web;
import 'package:code_editor/code_editor.dart';
import 'package:code_editor/backends/html.dart';

@JS('window.DartEditor')
external set _dartEditorConstructor(JSFunction fn);

@JS('runCell')
external void _runCellJS(JSString cellId);

@JS('window.requestCompletionCm')
external void _requestCompletion(JSObject cm, String cellId);

void main() {
  final lexer = TextMateLexer(
    rootRules: [
      TextMateRule(
        id: 'keyword',
        match: RegExp(
          r'\b(var|final|const|import|class|void|return|if|for|in|while|async|await|try|catch|finally|throw|rethrow|enum|extension|mixin|with|implements|extends|is|as)\b',
        ),
        type: TokenType.keyword,
      ),
      TextMateRule(
        id: 'type',
        match: RegExp(
          r'\b(int|double|String|bool|List|Map|Set|NDArray|Float64|Float32|Int32|Int64|Object|dynamic|Future|Stream|Iterable|ScratchArena)\b',
        ),
        type: TokenType.identifier,
      ),
      TextMateRule(
        id: 'number',
        match: RegExp(r'\b\d+(\.\d+)?\b'),
        type: TokenType.number,
      ),
      TextMateRule(
        id: 'string',
        match: RegExp(
          r"'[^']*'|"
          r'"[^"]*"',
        ),
        type: TokenType.string,
      ),
      TextMateRule(
        id: 'comment',
        match: RegExp(r'//.*$'),
        type: TokenType.comment,
      ),
      TextMateRule(
        id: 'operator',
        match: RegExp(r'[+\-*/%=<>!&|^~]+'),
        type: TokenType.operator,
      ),
      TextMateRule(
        id: 'punctuation',
        match: RegExp(r'[\(\)\{\}\[\];,.]'),
        type: TokenType.punctuation,
      ),
    ],
  );

  _dartEditorConstructor =
      ((web.HTMLTextAreaElement textarea, JSObject? options) {
        final rawId = textarea.id;
        final cellId = rawId.startsWith('editor-')
            ? rawId.substring('editor-'.length)
            : rawId;
        final initialCode = textarea.value;
        final parent = textarea.parentElement!;

        // Container for editor
        final container = web.document.createElement('div') as web.HTMLElement;
        container.className = 'dart-code-editor-container';
        container.style.position = 'relative';
        container.style.width = '100%';
        container.style.minHeight = '100px';
        container.style.backgroundColor = '#1e1e2e';
        container.style.borderRadius = '8px';
        container.style.border = '1px solid #313244';
        container.style.overflow = 'hidden';
        container.style.fontFamily =
            "'Fira Code', 'JetBrains Mono', 'Consolas', monospace";
        container.style.fontSize = '14px';
        container.style.lineHeight = '24px';
        container.style.color = '#cdd6f4';

        // Highlight display layer
        final displayLayer =
            web.document.createElement('div') as web.HTMLElement;
        displayLayer.className = 'editor-display-layer';
        displayLayer.style.position = 'absolute';
        displayLayer.style.left = '0px';
        displayLayer.style.top = '0px';
        displayLayer.style.width = '100%';
        displayLayer.style.height = '100%';
        displayLayer.style.pointerEvents = 'none';

        // Style original textarea to overlay transparently over display layer
        textarea.style.position = 'relative';
        textarea.style.zIndex = '2';
        textarea.style.width = '100%';
        textarea.style.minHeight = '100px';
        textarea.style.paddingLeft = '52px';
        textarea.style.paddingTop = '6px';
        textarea.style.paddingBottom = '6px';
        textarea.style.backgroundColor = 'transparent';
        textarea.style.color = 'transparent';
        textarea.style.caretColor = '#f5e0dc';
        textarea.style.border = 'none';
        textarea.style.outline = 'none';
        textarea.style.resize = 'vertical';
        textarea.style.fontFamily =
            "'Fira Code', 'JetBrains Mono', 'Consolas', monospace";
        textarea.style.fontSize = '14px';
        textarea.style.lineHeight = '24px';

        parent.appendChild(container);
        container.appendChild(displayLayer);
        container.appendChild(textarea);

        // Core Engine setup from package:code_editor
        final buffer = PieceTreeTextBuffer(initialCode);
        final tokenizer = IncrementalTokenizer(tokenizer: lexer);
        final renderer = HtmlRenderer();
        renderer.attach(displayLayer);

        void updateRender() {
          final currentText = textarea.value;
          buffer.delete(0, buffer.length);
          buffer.insert(0, currentText);

          final lines = currentText.split('\n');
          tokenizer.setDocument(lines);
          final tokensPerLine = tokenizer.cachedLineTokens;

          final lineCount = buffer.lineCount;
          final renderLines = <RenderLine>[];
          for (var i = 0; i < lineCount; i++) {
            final start = buffer.getLineOffset(i);
            final len = buffer.getLineLength(i);
            final lineStr = buffer.getTextInRange(start, len);
            final tokens = (i < tokensPerLine.length)
                ? tokensPerLine[i]
                : <SyntaxToken>[];
            var colOffset = 0;
            final renderTokens = <RenderToken>[];
            for (final t in tokens) {
              final endCol = colOffset + t.text.length;
              renderTokens.add(
                RenderToken(
                  text: t.text,
                  startColumn: colOffset,
                  endColumn: endCol,
                  style: RenderTokenStyle(
                    foreground: _getScopeColor(t.type, t.text),
                    bold: t.type == TokenType.keyword,
                  ),
                ),
              );
              colOffset = endCol;
            }

            renderLines.add(
              RenderLine(
                lineIndex: i,
                top: i * 24.0 + 6.0,
                height: 24.0,
                text: lineStr,
                tokens: renderTokens,
              ),
            );
          }

          final viewport = RenderViewport(
            firstVisibleLine: 0,
            lastVisibleLine: lineCount > 0 ? lineCount - 1 : 0,
            width: container.clientWidth > 0
                ? container.clientWidth.toDouble()
                : 800.0,
            height: (lineCount * 24.0 + 20.0).clamp(100.0, 600.0),
            scrollX: 0.0,
            scrollY: 0.0,
            lineHeight: 24.0,
            charWidth: 8.5,
            lines: renderLines,
            gutter: RenderGutter(
              width: 44.0,
              items: List.generate(
                lineCount,
                (i) => GutterItem(lineIndex: i, lineNumberText: '${i + 1}'),
              ),
            ),
            selections: const [],
            carets: const [],
          );

          final htmlStr = renderer.renderToHtmlString(viewport);
          displayLayer.innerHTML = htmlStr.toJS;
        }

        updateRender();

        final bridge = _DartEditorBridge(
          getValue: (() => textarea.value.toJS).toJS,
          setValue: ((String newCode) {
            textarea.value = newCode;
            updateRender();
          }.toJS),
        );

        // Attach Keydown Listener for Shift+Enter & Ctrl+Space
        textarea.addEventListener(
          'keydown',
          ((web.KeyboardEvent e) {
            if (e.shiftKey && e.key == 'Enter') {
              e.preventDefault();
              try {
                _runCellJS(cellId.toJS);
              } catch (err) {
                print('Shift+Enter execution failed: $err');
              }
            } else if ((e.ctrlKey || e.metaKey) && e.key == ' ') {
              e.preventDefault();
              try {
                _requestCompletion(bridge, cellId);
              } catch (err) {
                print('Completion request failed: $err');
              }
            }
          }.toJS),
        );

        textarea.addEventListener(
          'input',
          ((web.Event e) {
            updateRender();
          }.toJS),
        );

        return bridge;
      }.toJS);
}

@JS()
@anonymous
extension type _DartEditorBridge._(JSObject _) implements JSObject {
  external factory _DartEditorBridge({
    JSFunction getValue,
    JSFunction setValue,
  });
}

String _getScopeColor(TokenType type, String text) {
  if (type == TokenType.keyword) return '#cba6f7';
  if (type == TokenType.identifier) {
    if (RegExp(r'^[A-Z]').hasMatch(text)) return '#f9e2af';
    return '#89b4fa';
  }
  if (type == TokenType.number) return '#fab387';
  if (type == TokenType.string) return '#a6e3a1';
  if (type == TokenType.comment) return '#6c7086';
  if (type == TokenType.operator) return '#89dceb';
  if (type == TokenType.punctuation) return '#9399b2';
  return '#cdd6f4';
}
