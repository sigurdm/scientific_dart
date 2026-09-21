import 'package:notebook/notebook.dart';

void main() {
  print('=== Jupyter Notebook (ipynb) Creation Example ===');

  // 1. Create notebook cells
  final markdownCell = IpynbCell(
    id: 'intro-cell',
    cellType: IpynbCellType.markdown,
    source:
        '# Scientific Dart Notebook\nDemonstrating notebook file generation.',
  );

  final codeCell = IpynbCell(
    id: 'code-cell-1',
    cellType: IpynbCellType.code,
    source: 'var a = 21;\nprint("The answer is \${a * 2}");',
    outputs: [IpynbOutput.stream(text: 'The answer is 42\n')],
    executionCount: 1,
  );

  // 2. Build the notebook document
  final notebook = IpynbNotebook(cells: [markdownCell, codeCell]);

  print('Notebook created with ${notebook.cells.length} cells.');
  print('\nJSON representation:\n${notebook.toJsonString(pretty: true)}');
}
