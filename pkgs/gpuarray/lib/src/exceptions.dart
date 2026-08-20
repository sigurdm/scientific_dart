/// Exception thrown when a GPU array or device operation fails.
class GpuException implements Exception {
  /// The error message.
  final String message;

  /// Creates a new [GpuException] with the given [message].
  const GpuException(this.message);

  @override
  String toString() => 'GpuException: $message';
}

/// Exception thrown when an invalid buffer or out-of-memory error occurs.
class GpuMemoryException extends GpuException {
  /// Creates a new [GpuMemoryException] with the given [message].
  const GpuMemoryException(super.message);

  @override
  String toString() => 'GpuMemoryException: $message';
}

/// Exception thrown when accessing a disposed GPU resource or device.
class GpuDeviceDisposedException extends GpuException {
  /// Creates a new [GpuDeviceDisposedException] with the given [message].
  const GpuDeviceDisposedException([
    super.message = 'Attempted to access a disposed GPU resource or device.',
  ]);

  @override
  String toString() => 'GpuDeviceDisposedException: $message';
}

/// Exception thrown when tensor shapes are incompatible for an operation or broadcasting.
class GpuShapeMismatchException extends GpuException {
  /// The shape of the first operand.
  final List<int> shapeA;

  /// The shape of the second operand.
  final List<int> shapeB;

  /// The operation name.
  final String operation;

  /// Creates a new [GpuShapeMismatchException].
  GpuShapeMismatchException(this.operation, this.shapeA, this.shapeB)
    : super(
        'Cannot perform $operation on incompatible shapes: $shapeA and $shapeB',
      );

  @override
  String toString() => 'GpuShapeMismatchException: $message';
}

/// Exception thrown when an axis index is out of bounds for a tensor.
class GpuAxisOutOfBoundsException extends GpuException {
  /// The requested axis.
  final int axis;

  /// The tensor rank.
  final int rank;

  /// Creates a new [GpuAxisOutOfBoundsException].
  GpuAxisOutOfBoundsException(this.axis, this.rank)
    : super('Axis $axis is out of bounds for tensor of rank $rank');

  @override
  String toString() => 'GpuAxisOutOfBoundsException: $message';
}
