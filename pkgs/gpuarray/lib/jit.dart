/// Dynamic JIT kernel compilation and operator fusion engine for GPU computing.
///
/// Enables fusing complex sequences of elementwise operations (e.g. $y = \text{silu}(a \cdot x + b)$)
/// into a single, high-performance GPU compute shader pass without intermediate VRAM allocations.
library;

export 'src/backend/wgsl/kernel_fusion.dart';
export 'src/backend/wgsl/jit_compiler.dart';
