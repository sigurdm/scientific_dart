# gpuarray

GPU-accelerated N-dimensional array computing for Dart with seamless `package:ndarray` interoperability.

`gpuarray` provides high-performance tensor computing on GPU devices (`GpuArray`), device and buffer memory management (`GpuDevice`, `GpuBuffer`), and zero-intermediate-copy memory transfers to and from host `NDArray`s.

## Key Design Principles

1. **Oblivious Core**: `package:ndarray` remains 100% pure CPU, lightweight, and oblivious to GPU concepts. All GPU functionality and interop extensions live exclusively in `gpuarray`.
2. **Explicit Host $\longleftrightarrow$ Device Transfers**: Keeps data on the GPU across multi-step compute pipelines, avoiding silent PCIe transfer bottlenecks.
3. **Deterministic Memory Management**: Implements `ScopedResource` for automatic VRAM management within `ResourceScope.scope()`.
4. **Broadcast & Strided Arithmetic**: Full N-dimensional broadcasting and strided views support for all binary ufuncs and reductions.

---

## Getting Started

Add `gpuarray` to your `pubspec.yaml`:

```yaml
dependencies:
  gpuarray: ^0.0.1
  ndarray: ^0.0.2
  resource_scope: ^0.0.1
```

---

## Usage

### 1. Creating GPU Arrays & Arithmetic

```dart
import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';

void main() {
  ResourceScope.scope(() {
    final a = GpuArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
    final b = GpuArray.fromList([10.0, 20.0, 30.0, 40.0], [2, 2], DType.float64);

    // Chained elementwise ufuncs executed in VRAM
    final c = (a * 2.0) + b;
    print(c.toList()); // [[12.0, 24.0], [36.0, 48.0]]

    // Unary math functions
    final s = c.sin();
    print(s.toList());
  });
}
```

### 2. Matrix Multiplication & Reductions

```dart
import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';

void main() {
  ResourceScope.scope(() {
    final m1 = GpuArray.fromList([
      [1.0, 2.0],
      [3.0, 4.0],
    ], [2, 2], DType.float64);

    final m2 = GpuArray.fromList([
      [5.0, 6.0],
      [7.0, 8.0],
    ], [2, 2], DType.float64);

    // Matrix multiplication
    final product = m1.matmul(m2);
    print(product.toList()); // [[19.0, 22.0], [43.0, 50.0]]

    // Reductions along axes
    final colSum = product.sum(axis: 0);
    print(colSum.toList()); // [62.0, 72.0]

    final totalMean = product.mean();
    print(totalMean.scalar); // 33.5
  });
}
```

### 3. Interoperability with Host `NDArray`

Convert between CPU host `NDArray` and `GpuArray` directly between C-heap memory and GPU buffers:

```dart
import 'package:ndarray/ndarray.dart';
import 'package:gpuarray/gpuarray.dart';
import 'package:resource_scope/resource_scope.dart';

void main() {
  ResourceScope.scope(() {
    // 1. Host array
    final hostArr = NDArray.fromList([1.0, 4.0, 9.0, 16.0], [4], DType.float64);

    // 2. Upload to GPU
    final gpuArr = hostArr.toGpu();

    // 3. Compute on GPU
    final gpuSqrt = gpuArr.sqrt();

    // 4. Download result back to NDArray
    final resultND = gpuSqrt.toNDArray();
    print(resultND.toList()); // [1.0, 2.0, 3.0, 4.0]
  });
}
```

---

## Benchmarks

To run the performance benchmark suite:

```bash
dart benchmark/gpu_array_benchmark.dart
```

---

## Roadmap & Parity Goals

See **[ROADMAP.md](file:///usr/local/google/home/sigurdm/projects/math/pkgs/gpuarray/ROADMAP.md)** for our phased milestones toward full **CuPy** (scientific computing) and **PyTorch** (deep learning / autodiff) feature parity.

---

## License

This package is licensed under the **[Apache License, Version 2.0](file:///usr/local/google/home/sigurdm/projects/math/pkgs/gpuarray/LICENSE)**.
