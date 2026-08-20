import '../../editor/code_editor_controller.dart';

/// Stub implementation for non-web environments.
final class HtmlCodeEditor {
  final Object hostElement;
  final CodeEditorController controller;

  void Function()? onExecute;
  void Function(int offset, int clientX, int clientY)? onCompletionRequested;
  void Function(int offset, int clientX, int clientY)? onHoverRequested;

  HtmlCodeEditor({
    required this.hostElement,
    required this.controller,
    this.onExecute,
    this.onCompletionRequested,
    this.onHoverRequested,
  });

  Object get rootElement => Object();

  void focus() {}
  void render() {}
}
