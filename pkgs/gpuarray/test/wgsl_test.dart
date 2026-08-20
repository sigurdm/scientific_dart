import 'package:test/test.dart';
import 'package:gpuarray/wgsl.dart';
import 'package:gpuarray/jit.dart';

void main() {
  group('WGSL Types and Binding Layouts', () {
    test('WgslDType converts from DType correctly', () {
      expect(WgslDType.float32.wgslType, equals('f32'));
      expect(WgslDType.float32.byteSize, equals(4));
      expect(WgslDType.float16.wgslType, equals('f16'));
      expect(WgslDType.int32.wgslType, equals('i32'));
      expect(WgslDType.uint32.wgslType, equals('u32'));
    });

    test('WgslBinding generates valid WGSL variable declarations', () {
      const storageRead = WgslBinding(
        group: 0,
        binding: 0,
        name: 'src_a',
        dtype: WgslDType.float32,
        access: WgslBufferAccess.read,
      );
      expect(
        storageRead.toWgslDeclaration(),
        equals('@group(0) @binding(0) var<storage, read> src_a: array<f32>;'),
      );

      const storageWrite = WgslBinding(
        group: 0,
        binding: 1,
        name: 'dst',
        dtype: WgslDType.float32,
        access: WgslBufferAccess.readWrite,
      );
      expect(
        storageWrite.toWgslDeclaration(),
        equals('@group(0) @binding(1) var<storage, read_write> dst: array<f32>;'),
      );

      const uniformBinding = WgslBinding(
        group: 0,
        binding: 2,
        name: 'uniforms',
        isUniform: true,
        customTypeName: 'Uniforms',
      );
      expect(
        uniformBinding.toWgslDeclaration(),
        equals('@group(0) @binding(2) var<uniform> uniforms: Uniforms;'),
      );
    });

    test('WgslWorkgroupSize generates correct attributes', () {
      expect(const WgslWorkgroupSize(256, 1, 1).toAttribute(), equals('@workgroup_size(256)'));
      expect(const WgslWorkgroupSize(16, 16, 1).toAttribute(), equals('@workgroup_size(16, 16)'));
      expect(const WgslWorkgroupSize(8, 8, 4).toAttribute(), equals('@workgroup_size(8, 8, 4)'));
    });

    test('WgslShaderModule calculates 1D, 2D, and 3D dispatches accurately', () {
      final module1D = WgslShaderModule(
        name: 'test_1d',
        code: '@compute @workgroup_size(256) fn main() {}',
        workgroupSize: const WgslWorkgroupSize(256, 1, 1),
      );
      final d1 = module1D.calculateDispatch1D(1000);
      expect(d1.workgroupsX, equals(4)); // ceil(1000 / 256) = 4
      expect(d1.workgroupsY, equals(1));
      expect(d1.workgroupsZ, equals(1));

      final module2D = WgslShaderModule(
        name: 'test_2d',
        code: '@compute @workgroup_size(16, 16) fn main() {}',
        workgroupSize: const WgslWorkgroupSize(16, 16, 1),
      );
      final d2 = module2D.calculateDispatch2D(100, 50);
      expect(d2.workgroupsX, equals(7)); // ceil(100 / 16) = 7
      expect(d2.workgroupsY, equals(4)); // ceil(50 / 16) = 4

      final d3 = module2D.calculateDispatch3D(32, 32, 8);
      expect(d3.workgroupsX, equals(2));
      expect(d3.workgroupsY, equals(2));
      expect(d3.workgroupsZ, equals(8));
    });
  });

  group('WGSL Elementwise Shader Templates', () {
    test('Contiguous binary compute shaders generate valid WGSL', () {
      final ops = ['add', 'sub', 'mul', 'div', 'pow', 'max', 'min', 'eq', 'gt'];
      for (final op in ops) {
        final shader = WgslTemplates.elementwiseBinary(op: op, strided: false);
        expect(shader.name, contains('elementwise_binary_$op'));
        expect(shader.bindings.length, equals(4));
        expect(shader.code, contains('@compute'));
        expect(shader.code, contains('@workgroup_size(256)'));
        expect(shader.code, contains('dst[idx] = '));

        final validation = WgslSyntaxValidator.validate(shader.code);
        expect(validation.isValid, isTrue, reason: 'Failed validation for op $op: ${validation.errors}');
      }
    });

    test('Strided binary compute shaders include multi-index routines', () {
      final shader = WgslTemplates.elementwiseBinary(op: 'add', strided: true);
      expect(shader.code, contains('struct StridedMetadata'));
      expect(shader.code, contains('flat_to_strided_offsets'));
      expect(shader.code, contains('src_a[off_a]'));
      expect(shader.code, contains('src_b[off_b]'));
      expect(shader.code, contains('dst[off_dst] = '));

      final validation = WgslSyntaxValidator.validate(shader.code);
      expect(validation.isValid, isTrue, reason: validation.errors.toString());
    });

    test('Contiguous and strided unary shaders generate valid WGSL', () {
      final unaryOps = [
        'relu',
        'silu',
        'gelu',
        'sigmoid',
        'tanh',
        'exp',
        'log',
        'sqrt',
        'rsqrt',
        'abs',
        'negate',
        'sin',
        'cos',
        'hardswish',
        'mish',
      ];

      for (final op in unaryOps) {
        final shader = WgslTemplates.elementwiseUnary(op: op, strided: false);
        expect(shader.bindings.length, equals(3));
        expect(shader.code, contains('@compute'));
        final val = WgslSyntaxValidator.validate(shader.code);
        expect(val.isValid, isTrue, reason: 'Op $op validation error: ${val.errors}');

        final stridedShader = WgslTemplates.elementwiseUnary(op: op, strided: true);
        final stridedVal = WgslSyntaxValidator.validate(stridedShader.code);
        expect(stridedVal.isValid, isTrue, reason: 'Strided op $op error: ${stridedVal.errors}');
      }
    });
  });

  group('WGSL Reduction Shaders', () {
    test('Tree reduction compute shader allocates workgroup shared memory', () {
      final reductionOps = ['sum', 'prod', 'min', 'max', 'sum_sq'];
      for (final op in reductionOps) {
        final shader = WgslTemplates.treeReduction(op: op, workgroupSize: 256);
        expect(shader.code, contains('var<workgroup> sdata: array<f32, 256>;'));
        expect(shader.code, contains('workgroupBarrier()'));
        expect(shader.code, contains('dst[workgroup_id.x] = sdata[0];'));

        final validation = WgslSyntaxValidator.validate(shader.code);
        expect(validation.isValid, isTrue, reason: 'Reduction $op failed: ${validation.errors}');
      }
    });
  });

  group('WGSL Matrix Multiplication & Convolution Shaders', () {
    test('Tiled GEMM shader allocates 16x16 shared memory tiles', () {
      final shader = WgslTemplates.tiledMatmul(tileSize: 16, hasBias: false);
      expect(shader.code, contains('var<workgroup> tile_a: array<array<f32, 16>, 16>;'));
      expect(shader.code, contains('var<workgroup> tile_b: array<array<f32, 16>, 16>;'));
      expect(shader.code, contains('workgroupBarrier()'));
      expect(shader.code, contains('uniforms.alpha * acc'));

      final validation = WgslSyntaxValidator.validate(shader.code);
      expect(validation.isValid, isTrue, reason: validation.errors.toString());
    });

    test('Tiled GEMM with bias incorporates bias addition', () {
      final shader = WgslTemplates.tiledMatmul(tileSize: 16, hasBias: true);
      expect(shader.bindings.length, equals(5));
      expect(shader.code, contains('bias[col]'));

      final validation = WgslSyntaxValidator.validate(shader.code);
      expect(validation.isValid, isTrue, reason: validation.errors.toString());
    });

    test('Batched GEMM computes batched matrix products', () {
      final shader = WgslTemplates.batchedMatmul(tileSize: 16);
      expect(shader.code, contains('batch_idx = global_id.z'));
      expect(shader.code, contains('batch_stride_a'));

      final validation = WgslSyntaxValidator.validate(shader.code);
      expect(validation.isValid, isTrue, reason: validation.errors.toString());
    });

    test('2D Convolution shader handles NCHW tensor layout with padding and stride', () {
      final shader = WgslTemplates.conv2d(hasBias: true);
      expect(shader.code, contains('struct Conv2dUniforms'));
      expect(shader.code, contains('dilation_h'));
      expect(shader.code, contains('pad_h'));
      expect(shader.code, contains('bias[oc]'));

      final validation = WgslSyntaxValidator.validate(shader.code);
      expect(validation.isValid, isTrue, reason: validation.errors.toString());
    });

    test('2D Transpose uses padded shared memory tile to avoid bank conflicts', () {
      final shader = WgslTemplates.transpose(tileSize: 16);
      expect(shader.code, contains('array<array<f32, 17>, 16>'));
      expect(shader.code, contains('workgroupBarrier()'));

      final validation = WgslSyntaxValidator.validate(shader.code);
      expect(validation.isValid, isTrue, reason: validation.errors.toString());
    });

    test('Softmax and RMSNorm shaders generate valid WGSL', () {
      final softmaxShader = WgslTemplates.softmax(workgroupSize: 256);
      expect(softmaxShader.code, contains('s_max'));
      expect(softmaxShader.code, contains('s_sum'));
      final vSoftmax = WgslSyntaxValidator.validate(softmaxShader.code);
      expect(vSoftmax.isValid, isTrue, reason: vSoftmax.errors.toString());

      final rmsNormShader = WgslTemplates.rmsNorm(workgroupSize: 256);
      expect(rmsNormShader.code, contains('s_sq'));
      expect(rmsNormShader.code, contains('inverseSqrt'));
      final vRms = WgslSyntaxValidator.validate(rmsNormShader.code);
      expect(vRms.isValid, isTrue, reason: vRms.errors.toString());
    });
  });

  group('Dynamic JIT Kernel Fusion & AST Generator', () {
    test('Builds simple fused expression AST', () {
      final x = Expr.variable('x', bindingIndex: 0);
      final y = Expr.variable('y', bindingIndex: 1);
      final expr = (x * 2.0 + y).silu();

      expect(expr.depth, equals(4));
      expect(expr.variables.map((v) => v.name).toSet(), equals({'x', 'y'}));
      expect(expr.toFingerprint(), contains('silu(add(mul('));
    });

    test('Compiles fused elementwise shader y = silu(a * x + b)', () {
      final a = Expr.variable('a', bindingIndex: 0);
      final x = Expr.variable('x', bindingIndex: 1);
      final b = Expr.variable('b', bindingIndex: 2);
      final expr = (a * x + b).silu();

      final compiler = WgslJitCompiler();
      final module = compiler.compile(expr, kernelName: 'fused_linear_silu');

      expect(module.name, equals('fused_linear_silu'));
      expect(module.bindings.length, equals(5)); // a, x, b, dst, uniforms
      expect(module.code, contains('let a_val = a[idx];'));
      expect(module.code, contains('let x_val = x[idx];'));
      expect(module.code, contains('let b_val = b[idx];'));
      expect(module.code, contains('silu('));

      final val = WgslSyntaxValidator.validate(module.code);
      expect(val.isValid, isTrue, reason: val.errors.toString());
    });

    test('JIT Compiler caches compiled shader modules by fingerprint', () {
      final compiler = WgslJitCompiler();
      compiler.clearCache();
      expect(compiler.cachedCount, equals(0));

      final x = Expr.variable('x', bindingIndex: 0);
      final expr1 = (x * 3.0 + 1.0).relu();
      final expr2 = (x * 3.0 + 1.0).relu(); // Identical structure

      final m1 = compiler.compile(expr1);
      expect(compiler.cacheMisses, equals(1));
      expect(compiler.cacheHits, equals(0));
      expect(compiler.cachedCount, equals(1));

      final m2 = compiler.compile(expr2);
      expect(compiler.cacheHits, equals(1));
      expect(compiler.cacheMisses, equals(1));
      expect(identical(m1, m2), isTrue);

      final diffExpr = (x * 5.0).gelu();
      compiler.compile(diffExpr);
      expect(compiler.cacheMisses, equals(2));
      expect(compiler.cachedCount, equals(2));
    });

    test('Fused kernel with clamp and where / select constructs', () {
      final x = Expr.variable('x', bindingIndex: 0);
      final expr = x.where(x.exp(), x.log()).clamp(0.0, 10.0);

      final compiler = WgslJitCompiler();
      final module = compiler.compile(expr);
      expect(module.code, contains('select('));
      expect(module.code, contains('clamp('));

      final val = WgslSyntaxValidator.validate(module.code);
      expect(val.isValid, isTrue, reason: val.errors.toString());
    });

    test('Strided fused kernel generates multidimensional offset translation', () {
      final x = Expr.variable('x', bindingIndex: 0);
      final y = Expr.variable('y', bindingIndex: 1);
      final expr = (x + y).gelu();

      final compiler = WgslJitCompiler();
      final module = compiler.compile(expr, strided: true);

      expect(module.code, contains('flat_to_strided_offsets'));
      expect(module.code, contains('struct StridedMetadata'));

      final val = WgslSyntaxValidator.validate(module.code);
      expect(val.isValid, isTrue, reason: val.errors.toString());
    });
  });

  group('WGSL Syntax Validator Error Detection', () {
    test('Catches unmatched braces and parentheses', () {
      const badBrace = '@compute @workgroup_size(256) fn main() { dst[0] = 1.0;';
      final v1 = WgslSyntaxValidator.validate(badBrace);
      expect(v1.isValid, isFalse);
      expect(v1.errors.any((e) => e.contains('Unclosed brace')), isTrue);

      const badParen = '@compute @workgroup_size(256 fn main() {}';
      final v2 = WgslSyntaxValidator.validate(badParen);
      expect(v2.isValid, isFalse);
    });

    test('Catches duplicate resource bindings', () {
      const dupBinding = '''
@group(0) @binding(0) var<storage, read> src_a: array<f32>;
@group(0) @binding(0) var<storage, read> src_b: array<f32>;
@compute @workgroup_size(256) fn main() {}
''';
      final v = WgslSyntaxValidator.validate(dupBinding);
      expect(v.isValid, isFalse);
      expect(v.errors.any((e) => e.contains('Duplicate resource binding')), isTrue);
    });

    test('Catches invalid storage access qualifier', () {
      const badAccess = '''
@group(0) @binding(0) var<storage, write_only> src_a: array<f32>;
@compute @workgroup_size(256) fn main() {}
''';
      final v = WgslSyntaxValidator.validate(badAccess);
      expect(v.isValid, isFalse);
      expect(v.errors.any((e) => e.contains('Invalid storage buffer access qualifier')), isTrue);
    });
  });
}
