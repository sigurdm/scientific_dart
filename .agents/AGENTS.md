Remember when adding new operations:
* always allow for an out: argument whereever it makes sense
* always allow for complex float and integer inputs where it makes sense.
* Always use strong typing and generics for all NDArray arguments.
* always use ffi with C implementation. Use intrinsics to provide optimized flat contiguous version for speed. Try to be smart about this.
* always allow for strided version of the operation
* always use a switch to dispatch to the correct implementation based on DType rather than a if-else if chain.
* Documentation should be rich and detailed, in the same style as numpy.
* openblas and lapack bindings belong in the openblas package.
* Always use ScratchArena for temporary allocations.

When running dart commands use the sdk specified in .vscode/settings.json.