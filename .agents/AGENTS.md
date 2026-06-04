Remember when adding new operations, fixing old ones:
* always allow for an out: argument whereever it makes sense
* always allow for complex float and integer inputs where it makes sense.
* Always use strong typing and generics for all NDArray arguments.
* always use ffi with C implementation. Use intrinsics to provide optimized flat contiguous version for speed. Try to be smart about this.
* always allow for strided version of the operation
* always use a switch to dispatch to the correct implementation based on DType rather than a if-else if chain.
* Documentation should be rich and detailed, in the same style as numpy.
* openblas and lapack bindings belong in the openblas package.
* Always use ScratchArena for temporary allocations.
* Avoid using setRange and toList handling ndarrays in dart space. Rather make views and use NDArray.copy.
* When returning multiple values use records with named fields instead of HashMaps.
* It is usally ok for the result of an operation to have same dtype as the input. We like the conversions to be explicit.
 

When running dart commands use the sdk specified in .vscode/settings.json.