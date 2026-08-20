import 'package:ndarray/ndarray.dart' as nd;
import 'gpu_array.dart';
import 'device.dart';

/// Extension on [nd.NDArray] to easily transfer host C-memory arrays to GPU VRAM.
///
/// This extension lives purely in the `gpuarray` package, keeping `package:ndarray`
/// completely decoupled and oblivious of GPU concepts.
extension NDArrayGpuExtension<T> on nd.NDArray<T> {
  /// Transfers this host [nd.NDArray] to a [GpuArray] on the given [device].
  ///
  /// The transfer is performed directly between host C memory and GPU buffer
  /// with zero intermediate Dart heap allocations.
  ///
  /// Example:
  /// ```dart
  /// import 'package:ndarray/ndarray.dart';
  /// import 'package:gpuarray/gpuarray.dart';
  ///
  /// final cpuArr = NDArray.fromList([1.0, 2.0, 3.0], [3], DType.float64);
  /// final gpuArr = cpuArr.toGpu();
  /// ```
  GpuArray<T> toGpu({GpuDevice? device}) {
    return GpuArray<T>.fromNDArray(this, device: device);
  }
}
