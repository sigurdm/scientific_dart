import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'package:ndarray/ndarray.dart';

void main() {
  print('=== NDArray User-Allocated Memory (External Pointer) Examples ===\n');

  runExternallyManagedExample();
  runCustomFinalizerExample();
}

void runExternallyManagedExample() {
  print('--- 1. Externally Managed Memory (No Custom Finalizer) ---');

  // 1. User allocates raw C memory manually
  final int elementCount = 4;
  final ffi.Pointer<ffi.Double> pointer = malloc<ffi.Double>(elementCount);

  // Initialize memory
  for (var i = 0; i < elementCount; i++) {
    pointer[i] = (i + 1) * 10.0;
  }

  // 2. Wrap raw C memory in NDArray.fromPointer
  // Since we don't pass a nativeFinalizer, this is externally managed.
  final arr = NDArray<double>.fromPointer(pointer.cast(), [
    2,
    2,
  ], DType.float64);

  print('Wrapped array data: ${arr.data}'); // [10.0, 20.0, 30.0, 40.0]

  // 3. Perform operations/slicing
  final sliced = arr.slice([
    Slice.all(),
    Slice(start: 1),
  ]); // Slice second column
  print('Sliced view data: $sliced');

  // 4. Dispose NDArray (invalidates views but does not free pointer)
  arr.dispose();
  print('Array successfully disposed.');

  // 5. User manually frees the raw memory they allocated
  malloc.free(pointer);
  print('Raw memory manually freed by user.\n');
}

void runCustomFinalizerExample() {
  print('--- 2. Custom Native Finalizer (Self-Managed External Memory) ---');

  // We can use malloc.nativeFree as our custom deallocator function pointer!
  final int elementCount = 4;
  final ffi.Pointer<ffi.Double> pointer = malloc<ffi.Double>(elementCount);

  for (var i = 0; i < elementCount; i++) {
    pointer[i] = (i + 1) * 1.5;
  }

  // Wrap with custom nativeFinalizer.
  // The array now owns the deallocation lifecycle via the custom finalizer.
  final arr = NDArray<double>.fromPointer(
    pointer.cast(),
    [4],
    DType.float64,
    nativeFinalizer: malloc.nativeFree.cast(),
  );

  print('Custom-finalized array: ${arr.data}');

  // Calling dispose() will manually trigger the custom finalizer and free the pointer!
  arr.dispose();
  print('Array and its custom pointer successfully disposed together! 🏆\n');
}
