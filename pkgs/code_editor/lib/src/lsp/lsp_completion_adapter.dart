import 'completion_popup_model.dart';
import 'lsp_primitives.dart';

/// Adapter transforming LSP completion items into completion dropdown view models.
final class LspCompletionAdapter {
  static List<CompletionDropdownItem> adaptCompletions(
    List<LspCompletionItem> items,
  ) {
    return items.map((item) {
      return CompletionDropdownItem(
        label: item.label,
        insertText: item.insertText ?? item.label,
        detail: item.detail,
        documentation: item.documentation,
        kind: item.kind?.toString(),
      );
    }).toList();
  }
}
