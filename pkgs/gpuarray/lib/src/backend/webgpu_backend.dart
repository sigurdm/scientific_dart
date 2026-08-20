/// WebGPU compute hardware acceleration backend driver.
///
/// Automatically routes between [BrowserWebGpuBackend] (on Web with package:web / dart:js_interop)
/// and [WgpuNativeBackend] (on native desktop/server with wgpu-native C-FFI).
library;

export 'native/wgpu_native_backend.dart'
    if (dart.library.js_interop) 'web/webgpu_backend.dart';
