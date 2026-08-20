import 'dart:io';
import 'package:path/path.dart' as p;
import 'package:test/test.dart';
import 'package:notebook/notebook.dart';

void main() {
  late NotebookKernel kernel;

  setUp(() async {
    final workspaceDir = Directory.current.path;
    final sdkPath =
        Platform.environment['DART_SDK'] ??
        p.dirname(p.dirname(Platform.resolvedExecutable));

    kernel = NotebookKernel(workspaceDir: workspaceDir, dartSdkPath: sdkPath);
    await kernel.start();
  });

  tearDown(() async {
    await kernel.stop();
  });

  test('evaluates variable declaration without semicolon', () async {
    final code = '''
// Welcome to Dart NDArray Notebook!
// Type 'Ctrl+Space' or '.' to test autocompletion.
var a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64)
''';

    final result = await kernel.execute(code);
    expect(result, contains('Declared variable a'));

    final evalResult = await kernel.execute('a * 2');
    expect(evalResult, contains('NDArray'));
  });

  test('captures prints', () async {
    final result = await kernel.execute("print('hi')");
    expect(result, contains('"hi"'));
  });

  test(
    'evaluates multi-statement input cell with loops and declarations',
    () async {
      final code = '''
var pixels = <double>[];
for (var y = 0; y < 10; y++) {
  for (var x = 0; x < 10; x++) {
    pixels.add(x / 10.0);
    pixels.add(y / 10.0);
    pixels.add(0.5);
  }
}
var gradientArray = NDArray.fromList(pixels, [10, 10, 3], DType.float64);
Image(gradientArray)
''';

      final result = await kernel.execute(code);
      expect(result, contains('data:image/bmp;base64,'));

      final evalResult = await kernel.execute('gradientArray.shape');
      expect(evalResult, contains('[10, 10, 3]'));
    },
  );

  test('evaluates pure import statement', () async {
    final result = await kernel.execute(
      "import 'package:path/path.dart' as p;",
    );
    expect(result, contains('Imported path'));
  });

  test(
    'evaluates multi-line cell containing import statements and statements',
    () async {
      final code = '''
import 'package:gpuarray/gpuarray.dart';

final dev = GpuDevice.cpu();
final arr = GpuArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float32, device: dev);
arr.sum().scalar
''';

      final result = await kernel.execute(code);
      expect(result, contains('10.0'));
    },
  );

  test('evaluates multi-line cell with WebGPU widget display', () async {
    final code = '''
import 'package:gpuarray/gpuarray.dart';

final dev = GpuDevice.cpu();
final arr = GpuArray.fromList([1.0, 2.0, 3.0], [3], DType.float32, device: dev);
display(arr.toWebGpuWidget());
''';

    final result = await kernel.execute(code);
    expect(result, contains('text/html'));
    expect(result, contains('webgpu-notebook-widget'));
  });

  test('evaluates fused kernel interactive WebGPU browser widget', () async {
    final code = '''
import 'package:gpuarray/gpuarray.dart' as gpuarray;
import 'package:notebook/notebook.dart';

final x = gpuarray.GpuArray.fromList([1.0, 2.0, 3.0, 4.0], [4], DType.float32);
final xVar = gpuarray.Expr.variable('x', bindingIndex: 0);
final fused = (xVar * 2.5 + 1.2).silu();
final descriptor = gpuarray.FusedKernelDescriptor(
  name: 'interactive_silu',
  expression: fused,
);
display(descriptor.createBrowserWidget(
  inputArrays: [x],
  outputShape: [4],
  title: 'Interactive SiLU (Browser WebGPU)',
  sliders: [
    WebGpuSlider(
      name: 'multiplier',
      label: 'Input Multiplier',
      min: 0.1,
      max: 10.0,
      initialValue: 2.5,
      uniformWordIndex: 1,
    ),
  ],
));
''';

    final result = await kernel.execute(code);
    expect(result, contains('text/html'));
    expect(result, contains('webgpu-notebook-widget'));
    expect(result, contains('Interactive SiLU (Browser WebGPU)'));
  });

  test('evaluates 2D graphical fused radial ripple widget', () async {
    final code = '''
final size = 64;
final xGrid = NDArray.zeros([size, size], DType.float32);
final yGrid = NDArray.zeros([size, size], DType.float32);

for (var r = 0; r < size; r++) {
  for (var c = 0; c < size; c++) {
    xGrid[[r, c]] = (c / size - 0.5) * 10.0;
    yGrid[[r, c]] = (r / size - 0.5) * 10.0;
  }
}

final x = GpuArray.fromNDArray(xGrid);
final y = GpuArray.fromNDArray(yGrid);

final xIn = GpuExpr.variable('x_grid');
final yIn = GpuExpr.variable('y_grid');
final ripple = GpuExpr.scalar('ripple', defaultValue: 3.5);
final phase = GpuExpr.scalar('phase', defaultValue: 0.0);
final freq = GpuExpr.scalar('freq', defaultValue: 2.0);

final dist = (xIn * xIn + yIn * yIn).sqrt();
final zExpr = ((dist * ripple - phase).sin() * (xIn * freq).cos() + 1.0) / 2.0;

final descriptor = FusedKernelDescriptor(
  name: 'fused_radial_ripple',
  outputExpr: zExpr,
);

display(descriptor.createBrowserWidget(
  inputArrays: [x, y],
  outputShape: [size, size],
  sliders: [
    const WebGpuSlider(
      name: 'ripple',
      label: 'Ripple Frequency',
      min: 0.5,
      max: 10.0,
      initialValue: 3.5,
      step: 0.1,
    ),
    const WebGpuSlider(
      name: 'phase',
      label: 'Phase / Time',
      min: 0.0,
      max: 6.28,
      initialValue: 0.0,
      step: 0.05,
    ),
    const WebGpuSlider(
      name: 'freq',
      label: 'Harmonic Modulation',
      min: 0.0,
      max: 8.0,
      initialValue: 2.0,
      step: 0.1,
    ),
  ],
  renderToCanvas: true,
  canvasWidth: size,
  canvasHeight: size,
  colorMap: 'turbo',
  title: '🌊 2D GPUArray Fused Radial Ripple (WebGPU)',
));
''';

    final result = await kernel.execute(code);
    expect(result, contains('text/html'));
    expect(result, contains('<canvas'));
    expect(result, contains('fused_radial_ripple'));
  });
}
