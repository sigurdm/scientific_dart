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

---

## How It Works: Implicit Tracking

`NDArray.scope` uses **Dart Zones** to maintain an implicit context. Every time you call a constructor or a factory (like `zeros()`, `fromList()`, etc.), the `NDArray` instance checks if it's being created inside a scope. If it is, it adds itself to that scope's disposal list.

This means that even third-party utility functions that return `NDArray`s will automatically support scopes if they use the standard `ndarray` factories.

---

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

---

## Best Practices

1.  **Scope for "Units of Work"**: Wrap logical operations (like a single frame of audio processing or a single training step) in a scope.
2.  **Avoid Global Arrays**: Try not to keep `NDArray`s in long-lived variables. If you must, ensure they are either `detachFromScope()`'d or created outside any scope and manually disposed.
3.  **Nested Scopes**: Scopes can be nested. Inner scopes will only dispose of arrays created within them, while the outer scope handles the rest.
4.  **Error Handling**: Scopes use `try-finally` internally. Even if your code throws an exception, all native memory allocated within that block is guaranteed to be freed.

---

## Comparison at a Glance

| Feature | Manual (`.dispose()`) | `NDArray.scope` |
| :--- | :--- | :--- |
| **Risk of Leaks** | High (easy to forget) | Zero (within scope) |
| **Ergonomics** | Verbose `try-finally` | Clean, NumPy-like |
| **Intermediates** | Must be manually saved & freed | Automatically caught |
| **Async Safety** | Manual tracking required | Automatic completion wait |
