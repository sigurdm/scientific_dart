enum BufferType { original, add }

/// Buffer storage for original text content.
class OriginalBuffer {
  final String content;
  OriginalBuffer(this.content);

  int get length => content.length;
}

/// Append-only buffer storage for added text insertions.
class AddBuffer {
  final StringBuffer _buffer = StringBuffer();
  String _cachedText = '';

  int get length => _cachedText.length;

  /// Appends [text] to the add buffer and returns the start offset in add buffer.
  int append(String text) {
    final startOffset = _cachedText.length;
    _buffer.write(text);
    _cachedText += text;
    return startOffset;
  }

  /// Returns the text content starting at [start] of [length].
  String getText(int start, int length) {
    return _cachedText.substring(start, start + length);
  }

  /// Returns full text content.
  String get content => _cachedText;
}
