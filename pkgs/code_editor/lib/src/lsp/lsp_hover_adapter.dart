import 'lsp_primitives.dart';

class HoverTooltipViewModel {
  final String markdownContent;
  final LspRange? range;

  HoverTooltipViewModel({required this.markdownContent, this.range});
}

/// Adapter presenting LSP hover information as Markdown tooltip view models.
class LspHoverAdapter {
  static HoverTooltipViewModel? adaptHover(LspHover? hover) {
    if (hover == null || hover.contents.isEmpty) return null;
    return HoverTooltipViewModel(
      markdownContent: hover.contents,
      range: hover.range,
    );
  }
}
