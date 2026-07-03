import 'lsp_primitives.dart';

class CompletionDropdownItem {
  final String label;
  final String insertText;
  final String? detail;
  final String? documentation;
  final int? kind;

  CompletionDropdownItem({
    required this.label,
    required this.insertText,
    this.detail,
    this.documentation,
    this.kind,
  });
}

/// Adapter transforming LSP completion items into completion dropdown view models.
class LspCompletionAdapter {
  static List<CompletionDropdownItem> adaptCompletions(List<LspCompletionItem> items) {
    return items.map((item) {
      return CompletionDropdownItem(
        label: item.label,
        insertText: item.insertText ?? item.label,
        detail: item.detail,
        documentation: item.documentation,
        kind: item.kind,
      );
    }).toList();
  }
}
