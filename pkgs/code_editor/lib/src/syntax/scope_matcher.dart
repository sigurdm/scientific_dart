import 'syntax_token.dart';

/// Scope matching utilities for TextMate scope selectors with compound backtracking support.
final class ScopeMatcher {
  ScopeMatcher._();

  /// Computes a specificity score for how well [selector] matches the given [scopes].
  /// Returns 0 if [selector] does not match. Higher score indicates a more specific match.
  static int matchScore(List<StyleScope> scopes, String selector) {
    if (scopes.isEmpty || selector.trim().isEmpty) return 0;

    // Handle comma-separated selectors (e.g. "comment, string.quoted")
    if (selector.contains(',')) {
      final subSelectors = selector.split(',');
      int maxScore = 0;
      for (final sub in subSelectors) {
        final score = matchScore(scopes, sub);
        if (score > maxScore) maxScore = score;
      }
      return maxScore;
    }

    final parts = selector.trim().split(RegExp(r'\s+'));
    if (parts.isEmpty) return 0;

    int maxScore = 0;
    final targetPart = parts.last;

    // Backtracking search: rightmost selector segment can match any scope from right to left
    for (int i = scopes.length - 1; i >= 0; i--) {
      final singleScore = _matchSingleScope(scopes[i], targetPart);
      if (singleScore > 0) {
        final score = _matchAncestorParts(scopes, parts, i, singleScore);
        if (score > maxScore) maxScore = score;
      }
    }

    return maxScore;
  }

  static int _matchAncestorParts(
    List<StyleScope> scopes,
    List<String> parts,
    int targetScopeIdx,
    int targetScore,
  ) {
    int partIdx = parts.length - 2;
    int scopeIdx = targetScopeIdx - 1;
    int totalScore = targetScore * 10;

    while (partIdx >= 0 && scopeIdx >= 0) {
      final part = parts[partIdx];
      int score = 0;
      while (scopeIdx >= 0) {
        score = _matchSingleScope(scopes[scopeIdx], part);
        if (score > 0) {
          scopeIdx--;
          break;
        }
        scopeIdx--;
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
