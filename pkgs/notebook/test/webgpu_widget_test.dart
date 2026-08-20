import 'package:test/test.dart';
import 'package:notebook/notebook.dart';

void main() {
  group('Notebook WebGPU Widget Integration', () {
    test(
      'WebGpuShaderWidget formats into Displayable HTML with WebGPU client bootstrap',
      () {
        final widget = WebGpuShaderWidget(
          title: 'Mandelbrot Fractal Compute',
          wgsl: '@compute @workgroup_size(16, 16, 1) fn main() {}',
          entryPoint: 'main',
          workgroups: [32, 32, 1],
          uniforms: [512, 512, 100, 0],
        );

        expect(widget.mimeType, equals('text/html'));
        final html = widget.toHtml();
        expect(html, contains('Mandelbrot Fractal Compute'));
        expect(html, contains('navigator.gpu'));
        expect(html, contains('createShaderModule'));
        expect(html, contains('createComputePipeline'));
      },
    );

    test('display() and prettyFormat() capture objects providing toHtml()', () {
      clearCapturedOutput();
      final widget = WebGpuShaderWidget(
        title: 'Neural Activation Visualizer',
        wgsl: '@compute @workgroup_size(256, 1, 1) fn main() {}',
      );

      display(widget);
      final jsonOutput = getCapturedOutputsJson();
      expect(jsonOutput, contains('text/html'));
      expect(jsonOutput, contains('Neural Activation Visualizer'));

      final formatted = prettyFormat(widget);
      expect(formatted, contains('Neural Activation Visualizer'));
      expect(formatted, contains('navigator.gpu'));
    });
  });
}
