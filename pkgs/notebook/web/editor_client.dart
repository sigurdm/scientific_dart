import 'dart:js_interop';
import 'package:web/web.dart' as web;
import 'package:code_editor/code_editor.dart';
import 'package:code_editor/backends/html.dart';

@JS('globalThis')
external JSObject get globalThis;

@JS('runCell')
external void _runCellJS(JSString cellId);

@JS('requestCompletionCm')
external void _requestCompletion(JSObject cm, JSString cellId);

extension type _GlobalScope(JSObject _) implements JSObject {
  external set DartEditor(JSFunction fn);
}

void main() {
  _GlobalScope(
    globalThis,
  ).DartEditor = ((web.HTMLTextAreaElement textarea, JSObject? options) {
    final rawId = textarea.id;
    final cellId = rawId.startsWith('editor-')
        ? rawId.substring('editor-'.length)
        : rawId;
    final initialCode = textarea.value;
    final parent = textarea.parentElement!;

    // Hide original textarea
    textarea.style.display = 'none';

    // Mount container
    final mountContainer = web.document.createElement('div') as web.HTMLElement;
    mountContainer.className = 'dart-editor-mount-container';
    mountContainer.style.width = '100%';
    mountContainer.style.margin = '4px 0';
    parent.appendChild(mountContainer);

    final controller = CodeEditorController(
      initialText: initialCode,
      options: const EditorOptions(
        fontFamily: "'Fira Code', 'JetBrains Mono', 'Consolas', monospace",
        fontSize: 14.0,
        lineHeight: 24.0,
        tabSize: 2,
        autoClosingBrackets: true,
        smartIndent: true,
        matchBrackets: true,
        highlightActiveLine: true,
      ),
    );

    late final _DartEditorBridge bridge;

    final editor = HtmlCodeEditor(
      hostElement: mountContainer,
      controller: controller,
      onExecute: () {
        _runCellJS(cellId.toJS);
      },
      onCompletionRequested: (offset, x, y) {
        _requestCompletion(bridge, cellId.toJS);
      },
    );

    controller.addListener(() {
      textarea.value = controller.text;
    });

    JSString getValue() => controller.text.toJS;
    void setValue(JSString newCode) {
      controller.text = newCode.toDart;
    }

    void focus() {
      editor.focus();
    }

    void formatCode() {
      controller.formatDocument();
    }

    void setDiagnostic(
      JSNumber line,
      JSNumber startColumn,
      JSNumber endColumn,
      JSString message,
    ) {
      controller.setDiagnostics([
        DiagnosticSquiggle(
          line: line.toDartInt,
          startColumn: startColumn.toDartInt,
          endColumn: endColumn.toDartInt,
          severity: DiagnosticSeverity.error,
          message: message.toDart,
        ),
      ]);
    }

    void clearDiagnostics() {
      controller.clearDiagnostics();
    }

    bridge = _DartEditorBridge._(JSObject());
    bridge.getValue = getValue.toJS;
    bridge.setValue = setValue.toJS;
    bridge.focus = focus.toJS;
    bridge.formatCode = formatCode.toJS;
    bridge.setDiagnostic = setDiagnostic.toJS;
    bridge.clearDiagnostics = clearDiagnostics.toJS;

    return bridge;
  }.toJS);
}

@JS()
@anonymous
extension type _DartEditorBridge._(JSObject _) implements JSObject {
  external set getValue(JSFunction fn);
  external set setValue(JSFunction fn);
  external set focus(JSFunction fn);
  external set formatCode(JSFunction fn);
  external set setDiagnostic(JSFunction fn);
  external set clearDiagnostics(JSFunction fn);
}
