import 'dart:io';

import 'package:criterion/criterion.dart';
import 'package:ndarray/ndarray.dart';

void main() async {
  const size = 80; // 80 x 80 matrix
  const reportDir = 'benchmark/report/nonzero_coords';

  try {
    await NDArray.scope(() async {
      await criterion(
        'NDArray Non-Zero Coordinates Search Benchmark',
        (c) {
          final a = NDArray.zeros([size, size], DType.float64);
          for (var i = 0; i < size; i += 5) {
            for (var j = 0; j < size; j += 3) {
              a.setCell([i, j], 9.9);
            }
          }

          c.group('nonzero() Search Strategy', () {
            c.bench('1. Optimized Flat Offset nonzero()', () {
              NDArray.scope(() {
                final res = nonzero(a);
                blackhole(res.length);
              });
            }, throughput: Throughput.elements(size * size));

            c.bench('2. Slow Bracket Selector Sweep Fallback (getCell)', () {
              NDArray.scope(() {
                final coordinateLists = List.generate(2, (_) => <int>[]);
                for (var i = 0; i < size; i++) {
                  for (var j = 0; j < size; j++) {
                    final val = a.getCell([i, j]);
                    if (val != 0.0) {
                      coordinateLists[0].add(i);
                      coordinateLists[1].add(j);
                    }
                  }
                }
                final res = coordinateLists.map((list) {
                  return NDArray<DTypeTag>.fromList(list, [
                    list.length,
                  ], DType.int32);
                }).toList();
                blackhole(res.length);
              });
            }, throughput: Throughput.elements(size * size));
          });
        },
        config: CriterionConfig(
          generateHtmlReport: false,
          exportJson: false,
          reportDir: reportDir,
        ),
      );
    });
  } finally {
    final dir = Directory('benchmark/report');
    if (dir.existsSync()) {
      dir.deleteSync(recursive: true);
    }
  }
}
