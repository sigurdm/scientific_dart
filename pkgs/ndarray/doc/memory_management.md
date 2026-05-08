# Memory Management with NDArray Scopes

The `ndarray` package uses unmanaged C-heap memory to achieve peak performance and zero-copy interoperability with native libraries like OpenBLAS and PocketFFT. While this provides massive speed advantages, it means memory must be explicitly freed.

To make this safe and ergonomic, `ndarray` provides an **Automatic Disposal Scope** mechanism.

---

## The Problem: Manual Disposal

Without scopes, you must manually track and dispose of every array you create:

```dart
void calculate() {
  final a = NDArray.zeros([1000], DType.float64);
  try {
    final b = NDArray.ones([1000], DType.float64);
    try {
      final c = add(a, b);
      try {
        print(mean(c));
      } finally {
        c.dispose();
      }
    } finally {
      b.dispose();
    }
  } finally {
    a.dispose();
  }
}
```

This is verbose, prone to leaks if you forget a `dispose()` call, and especially difficult when dealing with intermediate results (like the output of `add(a, b)`).

---

## The Solution: `NDArray.scope`

`NDArray.scope` creates a "safe zone" where every array created is automatically tracked and deterministically freed when the scope finishes.

### Basic Usage

Simply wrap your logic in `NDArray.scope`:

```dart
void calculate() {
  NDArray.scope(() {
    final a = NDArray.zeros([1000], DType.float64);
    final b = NDArray.ones([1000], DType.float64);
    
    // Intermediate results are also tracked!
    final c = a + b; 
    
    print(mean(c));
  }); // All arrays (a, b, and c) are freed here!
}
```

### Returning Values with `detachFromScope()`

If you need a specific array to survive beyond the scope (e.g., as a return value), use `detachFromScope()`. This "unregisters" the array from the current scope.

```dart
NDArray computeResult() {
  return NDArray.scope(() {
    final a = NDArray.arange(0, 10);
    final b = a * 2;
    
    // 'a' will be disposed, but 'b' will be returned to the caller.
    return b.detachFromScope();
  });
}
```

> **Note**: You are now responsible for calling `.dispose()` on the returned array when you are finished with it.


## Asynchronous Scopes

The scope mechanism fully supports `async/await`. If your callback returns a `Future`, the scope will wait for that future to complete before disposing of the tracked arrays.

```dart
Future<void> processDataAsync() async {
  await NDArray.scope(() async {
    final data = await loadLargeArray();
    final processed = performComplexMath(data);
    await uploadResults(processed);
  }); // Cleanup happens only after all awaits finish.
}
```

### Avoiding repeated allocations with `into:`

While `NDArray.scope` cleans up intermediate allocations automatically, allocating new arrays inside high-frequency hot loops still incurs heap allocation and garbage collection overhead. 

Instead it is often better to pre-allocate a fixed destination array once and pass it to the `into:` named parameter in subsequent operations. This completely avoids repeated allocations by writing the result directly into the pre-allocated memory:

```dart
void processInLoop(NDArray<Float64> input) {
  // 1. Pre-allocate fixed output buffer once!
  final outputBuffer = NDArray<Float64>.zeros(input.shape, DType.float64);

  for (var i = 0; i < 10000; i++) {
    // 2. Zero new allocations! The ufunc writes directly into the buffer.
    input.multiply(Float64(2.0), into: outputBuffer);
    
    // Use the outputBuffer results...
  }
  
  // 3. Free the fixed buffer when completely done
  outputBuffer.dispose();
}
```

To create a buffer of the right shape for the result of a broadcasted operation, you can use [NDArray.broadcastShapes].
