/// Unified rendering abstractions and event bindings.
///
/// Defines the abstract [EditorRenderer] interface, immutable [RenderViewport]
/// snapshot model, and platform-agnostic keybinding registry.
library code_editor.render;

export 'src/render/render_viewport.dart';
export 'src/render/editor_renderer.dart';
export 'src/events/keybinding_registry.dart';
