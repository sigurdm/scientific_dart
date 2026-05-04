import 'dart:io';
import 'dart:convert';

void main() async {
  print('============================================================================');
  print('         num_dart AUTOMATED TEST COVERAGE INFRASTRUCTURE TOOL');
  print('============================================================================');

  final rawCoverageDir = Directory('coverage/raw');
  if (rawCoverageDir.existsSync()) {
    print('Clearing old raw coverage trace data maps...');
    rawCoverageDir.deleteSync(recursive: true);
  }

  print('\nStep 1: Executing full unit test suite and collecting raw JSON V8 traces...');
  final testProcess = await Process.run('dart', [
    'test',
    '--coverage=coverage/raw',
  ]);

  if (testProcess.exitCode != 0) {
    stderr.writeln('Error: dart test coverage execution failed!\n${testProcess.stderr}');
    exit(testProcess.exitCode);
  }
  print('Unit tests suite executed perfectly with exit code 0.');

  print('\nStep 2: Formatting raw JSON coverage maps into standardized LCOV format...');
  final formatProcess = await Process.run('dart', [
    'run',
    'coverage:format_coverage',
    '--in=coverage/raw',
    '--out=coverage/lcov.info',
    '--report-on=lib',
    '--lcov',
  ]);

  if (formatProcess.exitCode != 0) {
    stderr.writeln('Error: format_coverage execution failed!\n${formatProcess.stderr}');
    exit(formatProcess.exitCode);
  }
  print('Formatted LCOV report file successfully written to: coverage/lcov.info');

  print('\nStep 3: Parsing LCOV report stream data and building coverage dashboard...');
  final lcovFile = File('coverage/lcov.info');
  if (!lcovFile.existsSync()) {
    stderr.writeln('Error: coverage/lcov.info file not found after formatting pass!');
    exit(1);
  }

  final lines = await lcovFile.readAsLines();
  
  var totalLinesFound = 0;
  var totalLinesHit = 0;
  
  String? currentFile;
  var currentFileLinesFound = 0;
  var currentFileLinesHit = 0;

  print('\n----------------------------------------------------------------------------');
  print(' FILE-BY-FILE COVERAGE METRICS');
  print('----------------------------------------------------------------------------');
  print('${"FILE NAME".padRight(35)} | ${"COVERAGE %".padRight(12)} | ${"EXECUTED / TOTAL LINES"}');
  print('----------------------------------------------------------------------------');

  for (final line in lines) {
    if (line.startsWith('SF:')) {
      // Source File header line
      final fullPath = line.substring(3);
      // Extract file basename/relative path for premium readability
      currentFile = fullPath.contains('lib/src/') 
          ? fullPath.substring(fullPath.indexOf('lib/src/'))
          : fullPath.substring(fullPath.lastIndexOf('/') + 1);
      currentFileLinesFound = 0;
      currentFileLinesHit = 0;
    } else if (line.startsWith('LF:')) {
      // Lines Found count
      currentFileLinesFound = int.parse(line.substring(3));
      totalLinesFound += currentFileLinesFound;
    } else if (line.startsWith('LH:')) {
      // Lines Hit count
      currentFileLinesHit = int.parse(line.substring(3));
      totalLinesHit += currentFileLinesHit;
      
      // Conclude file record block and print out row!
      if (currentFile != null && currentFileLinesFound > 0) {
        final pct = (currentFileLinesHit / currentFileLinesFound) * 100.0;
        final pctStr = '${pct.toStringAsFixed(1)}%';
        print('${currentFile.padRight(35)} | ${pctStr.padRight(12)} | $currentFileLinesHit / $currentFileLinesFound');
      }
    }
  }

  print('----------------------------------------------------------------------------');
  if (totalLinesFound == 0) {
    print('Error: No executable code lines discovered in lib/ directory reports!');
  } else {
    final globalPct = (totalLinesHit / totalLinesFound) * 100.0;
    print('🏆 GLOBAL WORKSPACE LINE COVERAGE SUMMARY:');
    print('   GLOBAL COVERAGE : ${globalPct.toStringAsFixed(2)}%');
    print('   LINES EXECUTED  : $totalLinesHit');
    print('   TOTAL LINES     : $totalLinesFound');
  }
  print('============================================================================\n');
}
