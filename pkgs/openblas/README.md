# openblas

## Thread Safety

This package is designed to be safe for use across multiple Dart isolates. The build hook explicitly enables threading support when compiling OpenBLAS, allowing concurrent calls from different isolates to share the underlying native library safely.
