import 'package:ndarray/ndarray.dart';
import 'package:ndarray/src/ndarray_bindings.dart';

void main() {
  NDArray.scope(() {
    final size = 100000;
    final x = NDArray.ones([size], DType.float64);
    final out = NDArray.create([size], DType.float64);

    // Warm up
    for (var i = 0; i < 100; i++) {
      v_sin_double(x.pointer.cast(), out.pointer.cast(), size);
      sin(x, out: out);
    }

    // 1. Benchmark raw FFI call only
    final swFFI = Stopwatch()..start();
    for (var i = 0; i < 1000; i++) {
      v_sin_double(x.pointer.cast(), out.pointer.cast(), size);
    }
    swFFI.stop();
    print('Raw FFI v_sin_double (1000 runs): ${swFFI.elapsedMilliseconds} ms');

    // 2. Benchmark full wrapper sin()
    final swWrapper = Stopwatch()..start();
    for (var i = 0; i < 1000; i++) {
      sin(x, out: out);
    }
    swWrapper.stop();
    print('Wrapper sin() (1000 runs): ${swWrapper.elapsedMilliseconds} ms');
  });
}
