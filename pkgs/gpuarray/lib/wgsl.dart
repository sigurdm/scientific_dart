/// WebGPU Shading Language (WGSL) compute shader generators and templates.
///
/// This library provides hardware compute shader templates for elementwise arithmetic,
/// fast parallel tree reductions in workgroup shared memory, tiled matrix multiplications (GEMM),
/// 2D convolutions, and numerical normalizations (Softmax, RMSNorm).
library;

export 'src/backend/wgsl/wgsl_types.dart';
export 'src/backend/wgsl/wgsl_templates.dart';
