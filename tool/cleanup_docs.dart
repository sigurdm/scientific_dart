import 'dart:io';

void main() {
  final files = [
    'pkgs/ndarray/lib/src/ndarray.dart',
    'pkgs/ndarray/lib/src/operations.dart',
    'pkgs/ndarray/lib/src/broadcasting.dart',
    'pkgs/ndarray/lib/src/random.dart',
    'pkgs/ndarray/lib/src/fft.dart',
  ];

  for (final filePath in files) {
    final file = File(filePath);
    final lines = file.readAsLinesSync();
    final newLines = <String>[];

    for (var line in lines) {
      if (line.trim().startsWith('///')) {
        // Remove double backticks
        line = line.replaceAll('``', '`');

        // Remove backticks from things like `NDArray<double>` or `DType.float64`
        // but only if they look like code, not things like `.npy`
        
        // Simple heuristic: remove backticks if they wrap something starting with Uppercase 
        // or containing a dot followed by lowercase.
        
        line = line.replaceAllMapped(RegExp(r'`([A-Z][A-Za-z0-9<>]+)`'), (m) => m[1]!);
        line = line.replaceAllMapped(RegExp(r'`(DType\.[a-z0-9]+)`'), (m) => m[1]!);
        
        // Fix the specific cases found: `DType.float64);` -> DType.float64);
        line = line.replaceAll('`DType.float64);`', 'DType.float64);');
        line = line.replaceAll('`DType.float32);`', 'DType.float32);');
        line = line.replaceAll('`DType.int32);`', 'DType.int32);');
        line = line.replaceAll('`DType.int64);`', 'DType.int64);');
        line = line.replaceAll('`DType.boolean);`', 'DType.boolean);');
        line = line.replaceAll('`DType.float64,`', 'DType.float64,');
      }
      newLines.add(line);
    }
    file.writeAsStringSync(newLines.join('\n') + '\n');
  }
}
