import 'dart:ffi' as ffi;
import 'package:criterion/criterion.dart';
import 'package:ffi/ffi.dart';

void main() async {
  const size = 4; // Simulating 4D shape/strides

  await criterion(
    'Allocation vs Buffer Reuse Overhead',
    (c) {
      final buffer = malloc<ffi.Int>(size);

      c.group('FFI Native Buffer Allocation', () {
        c.bench('Malloc + Free per iteration', () {
          final ptr = malloc<ffi.Int>(size);
          ptr[0] = 1;
          ptr[1] = 2;
          ptr[2] = 3;
          ptr[3] = 4;
          blackhole(ptr[0]);
          malloc.free(ptr);
        });

        c.bench('Buffer Reuse (preallocated)', () {
          final ptr = buffer;
          ptr[0] = 1;
          ptr[1] = 2;
          ptr[2] = 3;
          ptr[3] = 4;
          blackhole(ptr[0]);
        });
      });
    },
    config: CriterionConfig(
      generateHtmlReport: true,
      exportJson: true,
      reportDir: 'benchmark/report/allocation',
    ),
  );
}
