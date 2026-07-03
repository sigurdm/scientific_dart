import 'dart:convert';
import 'scope_matcher.dart';
import 'syntax_token.dart';

class ThemeRule {
  final List<String> scopes;
  final int? foreground;
  final int? background;
  final bool? bold;
  final bool? italic;
  final bool? underline;
  final bool? strikethrough;

  const ThemeRule({
    required this.scopes,
    this.foreground,
    this.background,
    this.bold,
    this.italic,
    this.underline,
    this.strikethrough,
  });

  factory ThemeRule.fromJson(Map<String, dynamic> json) {
    List<String> scopesList = [];
    final scopeVal = json['scope'];
    if (scopeVal is String) {
      scopesList = [scopeVal];
    } else if (scopeVal is List) {
      scopesList = scopeVal.map((e) => e.toString()).toList();
    }

    final settings = json['settings'] as Map<String, dynamic>? ?? {};
    final fgStr = settings['foreground'] as String?;
    final bgStr = settings['background'] as String?;
    final fontStyle = settings['fontStyle'] as String? ?? '';

    final isBold = fontStyle.contains('bold');
    final isItalic = fontStyle.contains('italic');
    final isUnderline = fontStyle.contains('underline');
    final isStrikethrough = fontStyle.contains('strikethrough');

    return ThemeRule(
      scopes: scopesList,
      foreground: fgStr != null ? parseHexColor(fgStr) : null,
      background: bgStr != null ? parseHexColor(bgStr) : null,
      bold: isBold ? true : null,
      italic: isItalic ? true : null,
      underline: isUnderline ? true : null,
      strikethrough: isStrikethrough ? true : null,
    );
  }
}

int? parseHexColor(String hex) {
  var clean = hex.trim().replaceAll('#', '');
  if (clean.length == 3) {
    final r = clean[0];
    final g = clean[1];
    final b = clean[2];
    clean = '$r$r$g$g$b$b';
  }
  if (clean.length == 6) {
    clean = 'FF$clean';
  } else if (clean.length == 8) {
    final rrggbb = clean.substring(0, 6);
    final aa = clean.substring(6, 8);
    clean = '$aa$rrggbb';
  }
  final value = int.tryParse(clean, radix: 16);
  return value;
}

class ColorTheme {
  final String name;
  final List<ThemeRule> rules;
  final ResolvedTokenStyle defaultStyle;

  ColorTheme({
    required this.name,
    required this.rules,
    this.defaultStyle = const ResolvedTokenStyle(foreground: 0xFFD4D4D4, background: 0xFF1E1E1E),
  });

  factory ColorTheme.fromJson(Map<String, dynamic> json) {
    final name = json['name'] as String? ?? 'Unnamed Theme';
    final tokenColors = json['tokenColors'] as List? ?? [];

    final rules = <ThemeRule>[];
    for (final item in tokenColors) {
      if (item is Map<String, dynamic>) {
        rules.add(ThemeRule.fromJson(item));
      }
    }

    return ColorTheme(name: name, rules: rules);
  }

  factory ColorTheme.fromJsonString(String jsonStr) {
    return ColorTheme.fromJson(jsonDecode(jsonStr) as Map<String, dynamic>);
  }

  ResolvedTokenStyle resolveStyle(List<StyleScope> scopes) {
    if (scopes.isEmpty) return defaultStyle;

    int? fg = defaultStyle.foreground;
    int? bg = defaultStyle.background;
    bool bold = defaultStyle.bold;
    bool italic = defaultStyle.italic;
    bool underline = defaultStyle.underline;
    bool strikethrough = defaultStyle.strikethrough;

    int bestFgScore = -1;
    int bestBgScore = -1;
    int bestBoldScore = -1;
    int bestItalicScore = -1;
    int bestUnderlineScore = -1;
    int bestStrikethroughScore = -1;

    for (final rule in rules) {
      for (final selector in rule.scopes) {
        final score = ScopeMatcher.matchScore(scopes, selector);
        if (score <= 0) continue;

        if (rule.foreground != null && score > bestFgScore) {
          bestFgScore = score;
          fg = rule.foreground;
        }
        if (rule.background != null && score > bestBgScore) {
          bestBgScore = score;
          bg = rule.background;
        }
        if (rule.bold != null && score > bestBoldScore) {
          bestBoldScore = score;
          bold = rule.bold!;
        }
        if (rule.italic != null && score > bestItalicScore) {
          bestItalicScore = score;
          italic = rule.italic!;
        }
        if (rule.underline != null && score > bestUnderlineScore) {
          bestUnderlineScore = score;
          underline = rule.underline!;
        }
        if (rule.strikethrough != null && score > bestStrikethroughScore) {
          bestStrikethroughScore = score;
          strikethrough = rule.strikethrough!;
        }
      }
    }

    return ResolvedTokenStyle(
      foreground: fg,
      background: bg,
      bold: bold,
      italic: italic,
      underline: underline,
      strikethrough: strikethrough,
    );
  }
}

/// Caches resolved styles for token scope hierarchies.
class StyleCache {
  final ColorTheme theme;
  final Map<String, ResolvedTokenStyle> _cache = {};

  StyleCache(this.theme);

  ResolvedTokenStyle getStyle(List<StyleScope> scopes) {
    if (scopes.isEmpty) return theme.defaultStyle;
    final key = scopes.map((s) => s.raw).join('|');
    return _cache.putIfAbsent(key, () => theme.resolveStyle(scopes));
  }

  void clear() {
    _cache.clear();
  }
}
