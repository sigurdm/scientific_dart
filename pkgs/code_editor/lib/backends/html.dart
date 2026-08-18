/// Web HTML/DOM virtualized renderer backend.
///
/// Provides HTML virtual list rendering with node recycling, XSS attribute escaping,
/// CSS selection overlays, and DOM event bindings.
library code_editor.backends.html;

export '../src/backends/html/dom_node_pool.dart';
export '../src/backends/html/html_code_editor_stub.dart'
    if (dart.library.js_interop) '../src/backends/html/html_code_editor.dart';
export '../src/backends/html/html_renderer.dart';
