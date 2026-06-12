import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('NDArray.fromPointer Tests', () {
    test('Wrap raw pointer and access data (Externally Managed)', () {
      final pointer = malloc<ffi.Double>(4);
      for (var i = 0; i < 4; i++) {
        pointer[i] = (i + 1) * 10.0;
      }

      final arr = NDArray.fromPointer(pointer.cast(), [2, 2], DType.float64);

      expect(arr.shape, [2, 2]);
      expect(arr.data, [10.0, 20.0, 30.0, 40.0]);

      // Modify through NDArray
      arr.data[1] = 99.0;
      expect(pointer[1], 99.0);

      // Slicing
      final slice = arr.slice([Slice.all(), Slice(start: 1)]);
      expect(slice.shape, [2, 1]);
      expect(slice.toList(), [99.0, 40.0]);

      expect(arr.isDisposed, false);
      arr.dispose();
      expect(arr.isDisposed, true);
      expect(slice.isDisposed, true);

      // Ensure calling dispose on externally managed pointer array does not free the raw pointer
      expect(pointer[0], 10.0); // Still accessible since it wasn't freed!

      malloc.free(pointer);
    });

    test('Wrap with Custom Native Finalizer and manually dispose', () {
      final pointer = malloc<ffi.Double>(4);
      for (var i = 0; i < 4; i++) {
        pointer[i] = (i + 1) * 2.0;
      }

      final arr = NDArray.fromPointer(
        pointer.cast(),
        [4],
        DType.float64,
        nativeFinalizer: malloc.nativeFree.cast(),
      );

      expect(arr.data, [2.0, 4.0, 6.0, 8.0]);
      expect(arr.isDisposed, false);

      // Manual dispose should invoke custom finalizer and free memory
      arr.dispose();
      expect(arr.isDisposed, true);

      // Note: The backing memory pointer is now freed, accessing it is undefined behavior.
    });

    test('Integration with NDArray.scope', () {
      final pointer = malloc<ffi.Double>(2);
      pointer[0] = 1.0;
      pointer[1] = 2.0;

      NDArray<double, Float64Marker>? arrRef;

      NDArray.scope(() {
        final arr = NDArray.fromPointer(pointer.cast(), [2], DType.float64);
        arrRef = arr;
        expect(arr.isDisposed, false);
      });

      // Once scope exits, the pointer array is marked as disposed to protect against invalid access
      expect(arrRef!.isDisposed, true);
      expect(() => arrRef!.data, throwsStateError);

      // Since it was externally managed, the raw pointer was not freed by the scope
      expect(pointer[0], 1.0);
      malloc.free(pointer);
    });
  });
}
