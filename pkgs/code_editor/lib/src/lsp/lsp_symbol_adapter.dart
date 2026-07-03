import 'lsp_primitives.dart';

class OutlineNode {
  final String name;
  final int kind;
  final LspRange range;
  final LspRange selectionRange;
  final List<OutlineNode> children;

  OutlineNode({
    required this.name,
    required this.kind,
    required this.range,
    required this.selectionRange,
    this.children = const [],
  });
}

/// Adapter building document outline trees and breadcrumbs from LSP symbols.
class LspSymbolAdapter {
  static List<OutlineNode> adaptDocumentSymbols(List<LspDocumentSymbol> symbols) {
    return symbols.map(_convertSymbol).toList();
  }

  static OutlineNode _convertSymbol(LspDocumentSymbol symbol) {
    return OutlineNode(
      name: symbol.name,
      kind: symbol.kind,
      range: symbol.range,
      selectionRange: symbol.selectionRange,
      children: symbol.children.map(_convertSymbol).toList(),
    );
  }

  /// Calculates the active breadcrumb trail for a given position [line].
  static List<String> buildBreadcrumbs(List<OutlineNode> nodes, int line) {
    final trail = <String>[];

    void findTrail(List<OutlineNode> currentNodes) {
      for (final node in currentNodes) {
        if (line >= node.range.start.line && line <= node.range.end.line) {
          trail.add(node.name);
          findTrail(node.children);
          break;
        }
      }
    }

    findTrail(nodes);
    return trail;
  }
}
