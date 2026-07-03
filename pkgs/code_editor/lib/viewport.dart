/// Virtual viewport spatial indexing and layout engine.
///
/// Includes O(log N) line height tree, soft line wrapping engine, code folding,
/// gutter metadata manager, and virtual slice window layout calculator.
library code_editor.viewport;

export 'src/viewport/folding_manager.dart';
export 'src/viewport/font_metrics.dart';
export 'src/viewport/gutter_manager.dart';
export 'src/viewport/line_height_tree.dart';
export 'src/viewport/line_wrapping_engine.dart';
export 'src/viewport/virtual_layout_calculator.dart';
