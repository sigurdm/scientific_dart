import 'syntax_token.dart';

/// Scope matching utilities for TextMate scope selectors.
class ScopeMatcher {
  /// Computes a specificity score for how well [selector] matches the given [scopes].
  /// Returns 0 if [selector] does not match. Higher score indicates a more specific match.
  static int matchScore(List<StyleScope> scopes, String selector) {
    if (scopes.isEmpty || selector.trim().isEmpty) return 0;

    final parts = selector.trim().split(RegExp(r'\s+'));
    if (parts.isEmpty) return 0;

    int scopeIdx = scopes.length - 1;
    int partIdx = parts.length - 1;
    int totalScore = 0;

    // Rightmost part must match the innermost scope (or an ancestor scope)
    final targetPart = parts[partIdx];
    int singleScore = 0;
    bool found = false;
    for (int i = scopes.length - 1; i >= 0; i--) {
      singleScore = _matchSingleScope(scopes[i], targetPart);
      if (singleScore > 0) {
        scopeIdx = i;
        found = true;
        break;
      }
    }
    if (!found) return 0;

    totalScore += singleScore * 10;
    partIdx--;
    scopeIdx--;

    // Match ancestor parts from right to left
    while (partIdx >= 0 && scopeIdx >= 0) {
      final part = parts[partIdx];
      int score = 0;
      while (scopeIdx >= 0) {
        score = _matchSingleScope(scopes[scopeIdx], part);
        scopeIdx--;
        if (score > 0) break;
      }
      if (score == 0) return 0;
      totalScore += score * 2;
      partIdx--;
    }

    if (partIdx >= 0) return 0;

    return totalScore + parts.length;
  }

  /// Matches a single dot-separated selector against a single StyleScope.
  static int _matchSingleScope(StyleScope scope, String selectorPart) {
    final selectorSegments = selectorPart.split('.');
    final scopeSegments = scope.segments;

    if (selectorSegments.length > scopeSegments.length) return 0;

    for (int i = 0; i < selectorSegments.length; i++) {
      if (selectorSegments[i] != scopeSegments[i]) return 0;
    }

    return selectorSegments.length;
  }
}
