import 'dart:convert';
import 'dart:math' as math;
import '../dtype.dart';

/// Configuration for an interactive UI slider in a browser WebGPU widget.
final class WebGpuSlider {
  /// Name of the parameter in the uniform buffer.
  final String name;

  /// Human-readable label displayed in the UI.
  final String label;

  /// Minimum selectable value.
  final double min;

  /// Maximum selectable value.
  final double max;

  /// Step increment for the slider.
  final double step;

  /// Initial default value.
  final double initialValue;

  /// Whether this parameter represents an integer (u32/i32) rather than a float (f32).
  final bool isInteger;

  /// The 0-based word index in the uniform buffer where this value is stored.
  final int uniformWordIndex;

  const WebGpuSlider({
    required this.name,
    required this.label,
    required this.min,
    required this.max,
    this.step = 0.01,
    required this.initialValue,
    this.isInteger = false,
    this.uniformWordIndex = 0,
  });

  Map<String, dynamic> toJson() => {
    'name': name,
    'label': label,
    'min': min,
    'max': max,
    'step': step,
    'initialValue': initialValue,
    'isInteger': isInteger,
    'uniformWordIndex': uniformWordIndex,
  };

  factory WebGpuSlider.fromJson(Map<String, dynamic> json) => WebGpuSlider(
    name: json['name'] as String,
    label: json['label'] as String,
    min: (json['min'] as num).toDouble(),
    max: (json['max'] as num).toDouble(),
    step: (json['step'] as num?)?.toDouble() ?? 0.01,
    initialValue: (json['initialValue'] as num).toDouble(),
    isInteger: json['isInteger'] as bool? ?? false,
    uniformWordIndex: (json['uniformWordIndex'] as num).toInt(),
  );
}

/// Serialized payload for an input or output tensor buffer in a WebGPU compute pipeline.
final class GpuBufferPayload {
  /// Bind group binding index (e.g. 0 for @binding(0)).
  final int bindingIndex;

  /// Variable name in the shader.
  final String name;

  /// Tensor data type.
  final DType dtype;

  /// Multidimensional tensor shape.
  final List<int> shape;

  /// Whether this buffer is an output destination.
  final bool isOutput;

  /// Base64 encoded binary payload for input tensors.
  final String? base64Data;

  /// Total buffer size in bytes.
  final int sizeInBytes;

  const GpuBufferPayload({
    required this.bindingIndex,
    required this.name,
    required this.dtype,
    required this.shape,
    this.isOutput = false,
    this.base64Data,
    required this.sizeInBytes,
  });

  Map<String, dynamic> toJson() => {
    'bindingIndex': bindingIndex,
    'name': name,
    'dtype': dtype.name,
    'shape': shape,
    'isOutput': isOutput,
    if (base64Data != null) 'base64Data': base64Data,
    'sizeInBytes': sizeInBytes,
  };

  factory GpuBufferPayload.fromJson(Map<String, dynamic> json) =>
      GpuBufferPayload(
        bindingIndex: (json['bindingIndex'] as num).toInt(),
        name: json['name'] as String,
        dtype: DType.values.byName(json['dtype'] as String),
        shape: List<int>.from(json['shape'] as List),
        isOutput: json['isOutput'] as bool? ?? false,
        base64Data: json['base64Data'] as String?,
        sizeInBytes: (json['sizeInBytes'] as num).toInt(),
      );
}

/// Standalone, fully serializable package representing a WebGPU compute pipeline.
final class GpuComputePipelinePackage {
  /// Name of the compute kernel.
  final String name;

  /// Raw WGSL compute shader source code.
  final String wgslCode;

  /// Compute shader entry point function name (e.g. 'main').
  final String entryPoint;

  /// Workgroup invocation layout `[workgroupSizeX, workgroupSizeY, workgroupSizeZ]`.
  final List<int> workgroupSize;

  /// Descriptors for all input buffers.
  final List<GpuBufferPayload> inputs;

  /// Descriptor for the primary output destination buffer.
  final GpuBufferPayload output;

  /// Uniform struct scalar values encoded as 32-bit words (u32/f32).
  final List<int> uniforms;

  /// Optional interactive UI sliders for real-time browser parameter exploration.
  final List<WebGpuSlider> sliders;

  /// Whether the output buffer represents visual data to render directly to an HTML5 canvas.
  final bool renderToCanvas;

  /// Canvas width in pixels when [renderToCanvas] is true.
  final int canvasWidth;

  /// Canvas height in pixels when [renderToCanvas] is true.
  final int canvasHeight;

  /// Visual color map for canvas rendering ('grayscale', 'viridis', 'plasma', 'heatmap').
  final String colorMap;

  /// Arbitrary user or compiler metadata.
  final Map<String, dynamic> metadata;

  const GpuComputePipelinePackage({
    required this.name,
    required this.wgslCode,
    this.entryPoint = 'main',
    this.workgroupSize = const [256, 1, 1],
    required this.inputs,
    required this.output,
    this.uniforms = const [],
    this.sliders = const [],
    this.renderToCanvas = false,
    this.canvasWidth = 512,
    this.canvasHeight = 512,
    this.colorMap = 'viridis',
    this.metadata = const {},
  });

  /// Serializes this pipeline package to a JSON map.
  Map<String, dynamic> toJson() => {
    'name': name,
    'wgslCode': wgslCode,
    'entryPoint': entryPoint,
    'workgroupSize': workgroupSize,
    'inputs': inputs.map((e) => e.toJson()).toList(),
    'output': output.toJson(),
    'uniforms': uniforms,
    'sliders': sliders.map((e) => e.toJson()).toList(),
    'renderToCanvas': renderToCanvas,
    'canvasWidth': canvasWidth,
    'canvasHeight': canvasHeight,
    'colorMap': colorMap,
    'metadata': metadata,
  };

  /// Constructs a [GpuComputePipelinePackage] from a JSON map.
  factory GpuComputePipelinePackage.fromJson(Map<String, dynamic> json) =>
      GpuComputePipelinePackage(
        name: json['name'] as String,
        wgslCode: json['wgslCode'] as String,
        entryPoint: json['entryPoint'] as String? ?? 'main',
        workgroupSize: List<int>.from(
          json['workgroupSize'] as List? ?? [256, 1, 1],
        ),
        inputs: (json['inputs'] as List)
            .map((e) => GpuBufferPayload.fromJson(e as Map<String, dynamic>))
            .toList(),
        output: GpuBufferPayload.fromJson(
          json['output'] as Map<String, dynamic>,
        ),
        uniforms: List<int>.from(json['uniforms'] as List? ?? []),
        sliders: (json['sliders'] as List? ?? [])
            .map((e) => WebGpuSlider.fromJson(e as Map<String, dynamic>))
            .toList(),
        renderToCanvas: json['renderToCanvas'] as bool? ?? false,
        canvasWidth: (json['canvasWidth'] as num?)?.toInt() ?? 512,
        canvasHeight: (json['canvasHeight'] as num?)?.toInt() ?? 512,
        colorMap: json['colorMap'] as String? ?? 'viridis',
        metadata: Map<String, dynamic>.from(json['metadata'] as Map? ?? {}),
      );

  /// Generates self-contained, interactive HTML & WebGPU JavaScript code that executes
  /// this compute shader directly on the client's GPU in any WebGPU-capable browser.
  String toHtml({String? containerId, String? title}) {
    final uid =
        containerId ??
        'webgpu_widget_${name}_${math.Random().nextInt(1000000)}';
    final headerTitle = title ?? '⚡ $name (Client-Side WebGPU Compute)';
    final payloadJson = jsonEncode(toJson());

    final sliderHtml = StringBuffer();
    if (sliders.isNotEmpty) {
      sliderHtml.writeln(
        '<div style="margin-top:12px; padding:12px; background:#11111b; border-radius:6px; border:1px solid #313244;">',
      );
      sliderHtml.writeln(
        '  <div style="font-weight:bold; color:#cdd6f4; margin-bottom:8px; font-size:13px;">⚙️ Interactive GPU Parameters (60 FPS Local Dispatch)</div>',
      );
      for (var i = 0; i < sliders.length; i++) {
        final s = sliders[i];
        final sliderId = '${uid}_slider_$i';
        final valId = '${uid}_val_$i';
        sliderHtml.writeln('''
  <div style="display:flex; align-items:center; margin-bottom:6px; font-size:12px; color:#bac2de;">
    <span style="width:140px; font-family:monospace;">${s.label}:</span>
    <input type="range" id="$sliderId" min="${s.min}" max="${s.max}" step="${s.step}" value="${s.initialValue}" style="flex:1; margin:0 10px; cursor:pointer;" />
    <span id="$valId" style="width:50px; font-family:monospace; color:#89b4fa; text-align:right;">${s.initialValue}</span>
  </div>''');
      }
      sliderHtml.writeln('</div>');
    }

    final canvasDisplay = renderToCanvas ? 'block' : 'none';

    return '''
<div id="$uid" class="webgpu-notebook-widget" style="background:#181825; border:1px solid #45475a; border-radius:8px; padding:16px; margin:10px 0; font-family:system-ui, -apple-system, sans-serif; color:#cdd6f4;">
  <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
    <div style="font-weight:bold; color:#89b4fa; font-size:14px;">$headerTitle</div>
    <div id="${uid}_badge" style="background:#313244; color:#a6adc8; font-size:11px; padding:2px 8px; border-radius:4px;">WebGPU</div>
  </div>

  <div id="${uid}_status" style="color:#a6adc8; font-size:13px; margin-bottom:8px;">Initializing WebGPU device on client...</div>

  ${sliderHtml.toString()}

  <canvas id="${uid}_canvas" width="$canvasWidth" height="$canvasHeight" style="display:$canvasDisplay; margin-top:12px; border-radius:6px; border:1px solid #313244; max-width:100%; box-shadow:0 4px 12px rgba(0,0,0,0.3);"></canvas>

  <div id="${uid}_details" style="margin-top:10px; font-size:12px; color:#a6adc8; display:none;">
    <span id="${uid}_perf" style="color:#a6e3a1; font-weight:bold;"></span>
    <pre id="${uid}_output" style="background:#11111b; color:#cdd6f4; padding:10px; border-radius:6px; border:1px solid #313244; margin-top:6px; max-height:160px; overflow-y:auto; font-family:monospace; font-size:11px;"></pre>
  </div>

  <script>
    (async () => {
      const pkg = $payloadJson;
      const statusEl = document.getElementById('${uid}_status');
      const badgeEl = document.getElementById('${uid}_badge');
      const detailsEl = document.getElementById('${uid}_details');
      const perfEl = document.getElementById('${uid}_perf');
      const outputEl = document.getElementById('${uid}_output');
      const canvasEl = document.getElementById('${uid}_canvas');

      if (!navigator.gpu) {
        statusEl.innerHTML = '<span style="color:#f38ba8;">⚠️ WebGPU is not supported in this browser. Please enable WebGPU in Chrome/Edge (chrome://flags/#enable-unsafe-webgpu) or Safari.</span>';
        badgeEl.style.background = '#f38ba8';
        badgeEl.style.color = '#11111b';
        badgeEl.textContent = 'WebGPU Unavailable';
        return;
      }

      try {
        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
          statusEl.innerHTML = '<span style="color:#f38ba8;">Failed to acquire WebGPU GPUAdapter.</span>';
          return;
        }
        const device = await adapter.requestDevice();
        const arch = adapter.info?.architecture || adapter.info?.vendor || 'Physical GPU';
        badgeEl.style.background = '#a6e3a1';
        badgeEl.style.color = '#11111b';
        badgeEl.textContent = 'Client GPU: ' + arch;

        // 1. Compile WGSL Compute Shader
        const shaderModule = device.createShaderModule({ code: pkg.wgslCode });
        const pipeline = device.createComputePipeline({
          layout: 'auto',
          compute: { module: shaderModule, entryPoint: pkg.entryPoint }
        });

        // 2. Upload Input Buffers
        const gpuInputBuffers = [];
        const bindGroupEntries = [];
        let curBinding = 0;

        for (const inp of pkg.inputs) {
          let byteArr;
          if (inp.base64Data) {
            const raw = atob(inp.base64Data);
            byteArr = new Uint8Array(raw.length);
            for (let i = 0; i < raw.length; i++) byteArr[i] = raw.charCodeAt(i);
          } else {
            byteArr = new Uint8Array(inp.sizeInBytes);
          }
          const buf = device.createBuffer({
            size: Math.max(16, (byteArr.byteLength + 3) & ~3),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
          });
          device.queue.writeBuffer(buf, 0, byteArr);
          gpuInputBuffers.push(buf);
          bindGroupEntries.push({ binding: curBinding++, resource: { buffer: buf } });
        }

        // 3. Allocate Destination Output Buffer
        const outSize = Math.max(16, (pkg.output.sizeInBytes + 3) & ~3);
        const outBuffer = device.createBuffer({
          size: outSize,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        bindGroupEntries.push({ binding: curBinding++, resource: { buffer: outBuffer } });

        const stagingBuffer = device.createBuffer({
          size: outSize,
          usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        // 4. Allocate Uniforms Buffer
        let uniformBuffer = null;
        let uniformData = new Uint32Array(pkg.uniforms.length > 0 ? pkg.uniforms : [1, 0, 0, 0]);

        // Pre-populate slider initial values into uniformData
        if (pkg.sliders && pkg.sliders.length > 0) {
          for (let i = 0; i < pkg.sliders.length; i++) {
            const s = pkg.sliders[i];
            const val = parseFloat(s.initialValue);
            if (s.isInteger) {
              uniformData[s.uniformWordIndex] = Math.round(val);
            } else {
              const f32View = new Float32Array(uniformData.buffer);
              f32View[s.uniformWordIndex] = val;
            }
          }
        }

        if (pkg.uniforms && pkg.uniforms.length > 0) {
          const uniSize = Math.max(16, ((uniformData.byteLength + 15) & ~15));
          uniformBuffer = device.createBuffer({
            size: uniSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
          });
          device.queue.writeBuffer(uniformBuffer, 0, uniformData);
          bindGroupEntries.push({ binding: curBinding++, resource: { buffer: uniformBuffer } });
        }

        const bindGroup = device.createBindGroup({
          layout: pipeline.getBindGroupLayout(0),
          entries: bindGroupEntries,
        });

        // 5. Execution Routine
        async function runCompute() {
          const t0 = performance.now();

          if (uniformBuffer && uniformData) {
            device.queue.writeBuffer(uniformBuffer, 0, uniformData);
          }

          const encoder = device.createCommandEncoder();
          const pass = encoder.beginComputePass();
          pass.setPipeline(pipeline);
          pass.setBindGroup(0, bindGroup);

          const totalElements = pkg.output.shape.reduce((a, b) => a * b, 1);
          const wgSizeX = pkg.workgroupSize[0] || 256;
          const wgSizeY = pkg.workgroupSize[1] || 1;
          const wgSizeZ = pkg.workgroupSize[2] || 1;

          let wgCountX, wgCountY, wgCountZ;
          if (pkg.renderToCanvas && pkg.canvasWidth && pkg.canvasHeight && wgSizeY > 1) {
            wgCountX = Math.ceil(pkg.canvasWidth / wgSizeX);
            wgCountY = Math.ceil(pkg.canvasHeight / wgSizeY);
            wgCountZ = 1;
          } else {
            wgCountX = Math.min(65535, Math.ceil(totalElements / wgSizeX));
            wgCountY = 1;
            wgCountZ = 1;
          }
          pass.dispatchWorkgroups(wgCountX, wgCountY, wgCountZ);
          pass.end();

          encoder.copyBufferToBuffer(outBuffer, 0, stagingBuffer, 0, outSize);
          device.queue.submit([encoder.finish()]);

          await stagingBuffer.mapAsync(GPUMapMode.READ);
          const mapped = stagingBuffer.getMappedRange(0, pkg.output.sizeInBytes);
          const floatArr = new Float32Array(mapped.slice(0));
          stagingBuffer.unmap();

          const elapsed = (performance.now() - t0).toFixed(2);
          perfEl.textContent = '⚡ Latency: ' + elapsed + ' ms | Total Elements: ' + totalElements.toLocaleString();
          detailsEl.style.display = 'block';
          statusEl.innerHTML = '<span style="color:#a6e3a1;">Compute kernel active on client GPU (' + arch + ').</span>';

          // Update preview text
          const previewCount = Math.min(16, floatArr.length);
          let previewStr = '[';
          for (let i = 0; i < previewCount; i++) {
            previewStr += (i > 0 ? ', ' : '') + floatArr[i].toFixed(4);
          }
          if (floatArr.length > previewCount) previewStr += ', ... (' + (floatArr.length - previewCount) + ' more)';
          previewStr += ']';
          outputEl.textContent = 'Output Tensor (' + pkg.output.dtype + ' ' + JSON.stringify(pkg.output.shape) + '):\\n' + previewStr;

          // Render to Canvas if requested
          if (pkg.renderToCanvas && canvasEl) {
            const ctx = canvasEl.getContext('2d');
            const imgData = ctx.createImageData(canvasEl.width, canvasEl.height);
            const pixels = imgData.data;
            const count = Math.min(floatArr.length, canvasEl.width * canvasEl.height);

            for (let i = 0; i < count; i++) {
              const val = floatArr[i];
              const pIdx = i * 4;
              const norm = Math.max(0, Math.min(1, isNaN(val) ? 0 : val));

              if (pkg.colorMap === 'viridis') {
                pixels[pIdx] = Math.floor(Math.sin(norm * Math.PI * 0.9) * 200 + 30);
                pixels[pIdx + 1] = Math.floor(Math.pow(norm, 0.7) * 240);
                pixels[pIdx + 2] = Math.floor((1 - norm) * 180 + Math.sin(norm * Math.PI) * 75);
                pixels[pIdx + 3] = 255;
              } else if (pkg.colorMap === 'magma' || pkg.colorMap === 'inferno') {
                pixels[pIdx] = Math.floor(Math.pow(norm, 0.6) * 255);
                pixels[pIdx + 1] = Math.floor(Math.pow(norm, 1.8) * 220);
                pixels[pIdx + 2] = Math.floor(Math.sin(norm * Math.PI * 0.8) * 160 + (norm < 0.5 ? norm * 200 : 0));
                pixels[pIdx + 3] = 255;
              } else if (pkg.colorMap === 'plasma') {
                pixels[pIdx] = Math.floor(Math.sin(norm * Math.PI * 0.7) * 230 + 20);
                pixels[pIdx + 1] = Math.floor(Math.pow(norm, 1.5) * 220);
                pixels[pIdx + 2] = Math.floor((1 - norm) * 210 + 40);
                pixels[pIdx + 3] = 255;
              } else if (pkg.colorMap === 'turbo' || pkg.colorMap === 'rainbow') {
                const h = (1 - norm) * 240;
                const s = 1.0, l = 0.5;
                const c = (1 - Math.abs(2 * l - 1)) * s;
                const x = c * (1 - Math.abs((h / 60) % 2 - 1));
                const m = l - c / 2;
                let r = 0, g = 0, b = 0;
                if (h >= 0 && h < 60) { r = c; g = x; b = 0; }
                else if (h >= 60 && h < 120) { r = x; g = c; b = 0; }
                else if (h >= 120 && h < 180) { r = 0; g = c; b = x; }
                else if (h >= 180 && h < 240) { r = 0; g = x; b = c; }
                pixels[pIdx] = Math.floor((r + m) * 255);
                pixels[pIdx + 1] = Math.floor((g + m) * 255);
                pixels[pIdx + 2] = Math.floor((b + m) * 255);
                pixels[pIdx + 3] = 255;
              } else {
                const v = Math.floor(norm * 255);
                pixels[pIdx] = v;
                pixels[pIdx + 1] = v;
                pixels[pIdx + 2] = v;
                pixels[pIdx + 3] = 255;
              }
            }
            ctx.putImageData(imgData, 0, 0);
          }
        }

        // Initial Run
        await runCompute();

        // 6. Connect Interactive Sliders
        for (let i = 0; i < pkg.sliders.length; i++) {
          const s = pkg.sliders[i];
          const sliderEl = document.getElementById('${uid}_slider_' + i);
          const valEl = document.getElementById('${uid}_val_' + i);
          if (sliderEl && valEl) {
            sliderEl.addEventListener('input', (e) => {
              const val = parseFloat(e.target.value);
              valEl.textContent = s.isInteger ? Math.round(val) : val.toFixed(2);
              if (s.isInteger) {
                uniformData[s.uniformWordIndex] = Math.round(val);
              } else {
                const f32View = new Float32Array(uniformData.buffer);
                f32View[s.uniformWordIndex] = val;
              }
              runCompute();
            });
          }
        }

      } catch (err) {
        statusEl.innerHTML = '<span style="color:#f38ba8;">WebGPU Runtime Error: ' + err.message + '</span>';
        console.error('WebGPU Execution Exception:', err);
      }
    })();
  </script>
</div>
''';
  }
}

/// Rich displayable widget wrapping a [GpuComputePipelinePackage] for browser notebooks.
final class WebGpuWidget {
  /// The underlying serialized compute pipeline package.
  final GpuComputePipelinePackage pipeline;

  /// Custom title displayed at the top of the widget.
  final String title;

  const WebGpuWidget(this.pipeline, {this.title = 'WebGPU Compute Widget'});

  /// Creates a real-time interactive Mandelbrot fractal WebGPU compute widget.
  factory WebGpuWidget.fractal({
    int width = 512,
    int height = 512,
    String colorMap = 'turbo',
    String? title,
  }) {
    const wgsl = '''
struct Uniforms {
  width: u32,
  height: u32,
  max_iter: u32,
  pad0: u32,
  center_x: f32,
  center_y: f32,
  zoom: f32,
  color_shift: f32,
}

@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= uniforms.width || id.y >= uniforms.height) { return; }
  
  let aspect = f32(uniforms.width) / f32(uniforms.height);
  let uv = vec2<f32>(
    ((f32(id.x) / f32(uniforms.width) - 0.5) * aspect) * uniforms.zoom + uniforms.center_x,
    (f32(id.y) / f32(uniforms.height) - 0.5) * uniforms.zoom + uniforms.center_y
  );
  
  var z = uv;
  var c = uv;
  var iter = 0u;
  while (iter < uniforms.max_iter && (z.x * z.x + z.y * z.y) < 4.0) {
    let x_new = z.x * z.x - z.y * z.y + c.x;
    let y_new = 2.0 * z.x * z.y + c.y;
    z = vec2<f32>(x_new, y_new);
    iter += 1u;
  }
  
  let idx = id.y * uniforms.width + id.x;
  if (iter >= uniforms.max_iter) {
    dst[idx] = 0.0;
  } else {
    let mag = sqrt(z.x * z.x + z.y * z.y);
    let log_zn = log(mag * mag) / 2.0;
    let nu = log(log_zn / log(2.0)) / log(2.0);
    let iter_smooth = f32(iter) + 1.0 - nu;
    let val = (iter_smooth / f32(uniforms.max_iter) + uniforms.color_shift) % 1.0;
    dst[idx] = max(0.01, val);
  }
}
''';

    final pkg = GpuComputePipelinePackage(
      name: 'mandelbrot_fractal',
      wgslCode: wgsl,
      workgroupSize: [16, 16, 1],
      inputs: [],
      output: GpuBufferPayload(
        bindingIndex: 0,
        name: 'dst',
        dtype: DType.float32,
        shape: [height, width],
        isOutput: true,
        sizeInBytes: width * height * 4,
      ),
      uniforms: [width, height, 80, 0],
      sliders: [
        const WebGpuSlider(
          name: 'zoom',
          label: 'Zoom Factor',
          min: 0.001,
          max: 3.5,
          initialValue: 2.2,
          step: 0.005,
          uniformWordIndex: 6,
        ),
        const WebGpuSlider(
          name: 'center_x',
          label: 'Center X',
          min: -2.0,
          max: 1.5,
          initialValue: -0.7,
          step: 0.005,
          uniformWordIndex: 4,
        ),
        const WebGpuSlider(
          name: 'center_y',
          label: 'Center Y',
          min: -1.5,
          max: 1.5,
          initialValue: 0.0,
          step: 0.005,
          uniformWordIndex: 5,
        ),
        const WebGpuSlider(
          name: 'max_iter',
          label: 'Max Iterations',
          min: 10,
          max: 200,
          initialValue: 80,
          step: 1,
          isInteger: true,
          uniformWordIndex: 2,
        ),
        const WebGpuSlider(
          name: 'color_shift',
          label: 'Color Palette Shift',
          min: 0.0,
          max: 1.0,
          initialValue: 0.0,
          step: 0.01,
          uniformWordIndex: 7,
        ),
      ],
      renderToCanvas: true,
      canvasWidth: width,
      canvasHeight: height,
      colorMap: colorMap,
    );

    return WebGpuWidget(
      pkg,
      title: title ?? '🌀 Real-Time WebGPU Mandelbrot Fractal (60 FPS)',
    );
  }

  /// Creates a real-time interactive 2D Wave Interference WebGPU compute widget.
  factory WebGpuWidget.waveInterference({
    int width = 512,
    int height = 512,
    String colorMap = 'plasma',
    String? title,
  }) {
    const wgsl = '''
struct Uniforms {
  width: u32,
  height: u32,
  pad0: u32,
  pad1: u32,
  freq: f32,
  phase: f32,
  separation: f32,
  amplitude: f32,
}

@group(0) @binding(0) var<storage, read_write> dst: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= uniforms.width || id.y >= uniforms.height) { return; }
  
  let aspect = f32(uniforms.width) / f32(uniforms.height);
  let uv = vec2<f32>(
    (f32(id.x) / f32(uniforms.width) - 0.5) * aspect * 2.0,
    (f32(id.y) / f32(uniforms.height) - 0.5) * 2.0
  );
  
  let p1 = vec2<f32>(-uniforms.separation, 0.0);
  let p2 = vec2<f32>(uniforms.separation, 0.0);
  let p3 = vec2<f32>(0.0, uniforms.separation * 0.8);
  
  let d1 = distance(uv, p1);
  let d2 = distance(uv, p2);
  let d3 = distance(uv, p3);
  
  let w1 = sin(d1 * uniforms.freq - uniforms.phase);
  let w2 = sin(d2 * uniforms.freq - uniforms.phase);
  let w3 = sin(d3 * uniforms.freq - uniforms.phase * 1.4);
  
  let total = (w1 + w2 + w3) / 3.0;
  let normalized = (total * uniforms.amplitude + 1.0) * 0.5;
  
  let idx = id.y * uniforms.width + id.x;
  dst[idx] = clamp(normalized, 0.0, 1.0);
}
''';

    final pkg = GpuComputePipelinePackage(
      name: 'wave_interference',
      wgslCode: wgsl,
      workgroupSize: [16, 16, 1],
      inputs: [],
      output: GpuBufferPayload(
        bindingIndex: 0,
        name: 'dst',
        dtype: DType.float32,
        shape: [height, width],
        isOutput: true,
        sizeInBytes: width * height * 4,
      ),
      uniforms: [width, height, 0, 0],
      sliders: [
        const WebGpuSlider(
          name: 'freq',
          label: 'Wave Frequency',
          min: 5.0,
          max: 60.0,
          initialValue: 24.0,
          step: 0.5,
          uniformWordIndex: 4,
        ),
        const WebGpuSlider(
          name: 'phase',
          label: 'Phase / Time',
          min: 0.0,
          max: 12.56,
          initialValue: 0.0,
          step: 0.05,
          uniformWordIndex: 5,
        ),
        const WebGpuSlider(
          name: 'separation',
          label: 'Source Separation',
          min: 0.05,
          max: 1.0,
          initialValue: 0.45,
          step: 0.01,
          uniformWordIndex: 6,
        ),
        const WebGpuSlider(
          name: 'amplitude',
          label: 'Wave Amplitude',
          min: 0.2,
          max: 3.0,
          initialValue: 1.2,
          step: 0.05,
          uniformWordIndex: 7,
        ),
      ],
      renderToCanvas: true,
      canvasWidth: width,
      canvasHeight: height,
      colorMap: colorMap,
    );

    return WebGpuWidget(
      pkg,
      title: title ?? '🌊 2D Wave Interference Simulation (WebGPU)',
    );
  }

  /// Standard MIME type for notebook display engines.
  String get mimeType => 'text/html';

  /// Renders self-contained HTML and JavaScript for browser execution.
  String toHtml() => pipeline.toHtml(title: title);

  @override
  String toString() => toHtml();
}
