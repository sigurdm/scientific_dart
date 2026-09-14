import 'package:code_editor/code_editor.dart';

void main() {
  print('=== CodeEditorController Example ===');

  // 1. Initialize controller with sample Dart code
  final controller = CodeEditorController(
    initialText: 'void main() {\n  print("Hello, scientific Dart!");\n}',
  );

  print('Lines count: ${controller.lineCount}');
  print('Content:\n${controller.text}\n');

  // 2. Inspect syntax tokens for line 0
  final tokens = controller.lineTokens.first;
  print('Tokens in line 0:');
  for (final token in tokens) {
    print('  ${token.type}: "${token.text}" (offset ${token.offset}..${token.end})');
  }

  // 3. Perform text manipulation
  controller.selection = const TextSelection.collapsed(TextPosition(1, 2));
  controller.insertText('// Added by controller\n  ');
  print('\nUpdated Content:\n${controller.text}');
}
