/// Base exception class for all ndarray-related errors.
final class NdArrayException implements Exception {
  /// The error message.
  final String message;

  /// Creates a new [NdArrayException] with the given [message].
  const NdArrayException(this.message);

  @override
  String toString() => 'NdArrayException: $message';
}

/// Exception thrown when a linear algebra operation fails.
final class LinAlgException extends NdArrayException {
  /// Creates a new [LinAlgException] with the given [message].
  const LinAlgException(super.message);

  @override
  String toString() => 'LinAlgException: $message';
}

/// Exception thrown when a matrix is singular and cannot be inverted or solved.
final class SingularMatrixException extends LinAlgException {
  /// Creates a new [SingularMatrixException] with the given [message].
  const SingularMatrixException(super.message);

  @override
  String toString() => 'SingularMatrixException: $message';
}
