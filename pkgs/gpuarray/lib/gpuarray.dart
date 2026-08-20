/// GPU-accelerated N-dimensional array computing for Dart.
///
/// This package provides high-performance tensor computing on GPU devices
/// (`GpuArray`), device and buffer memory management (`GpuDevice`, `GpuBuffer`),
/// and seamless, zero-intermediate-copy interoperability with `package:ndarray`.
library;

export 'src/exceptions.dart';
export 'src/dtype.dart';
export 'src/buffer.dart';
export 'src/device.dart';
export 'src/gpu_array.dart';
export 'src/slice.dart';
export 'src/operations/indexing.dart';
export 'src/operations/manipulation.dart';
export 'src/linalg/linalg.dart';
export 'src/fft/fft.dart';
export 'src/random/random.dart';
export 'src/autograd/autograd.dart';
export 'src/nn/nn.dart';
export 'src/serialization/safetensors.dart';
export 'src/interop.dart';
export 'src/backend/backend.dart' show GpuBackend, GpuDeviceType, CpuVectorBackend;
export 'src/backend/webgpu_backend.dart';
export 'wgsl.dart';
export 'jit.dart';
export 'serialization.dart';
