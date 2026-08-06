import 'render_viewport.dart';

/// Callback signature for renderer user interaction events (clicks, keypresses, scroll).
typedef RenderEventListener =
    void Function(String eventType, Map<String, dynamic> data);

/// Abstract contract for editor presentation backends.
abstract class EditorRenderer {
  /// Whether this renderer is attached to a visual output surface.
  bool get isAttached;

  /// The most recent viewport snapshot rendered by this backend.
  RenderViewport? get lastViewport;

  /// Attach the renderer to a target platform node or surface container.
  void attach(Object target);

  /// Detach the renderer from its target surface.
  void detach();

  /// Render the given viewport snapshot onto the target output surface.
  void render(RenderViewport viewport);

  /// Register an event listener for user inputs captured by the renderer backend.
  void addEventListener(RenderEventListener listener);

  /// Remove a registered event listener.
  void removeEventListener(RenderEventListener listener);

  /// Dispose resources used by this renderer.
  void dispose();
}
