# Implementation Plan - sinc universal function

This plan describes the steps to implement the sinc universal function using a unified synthesis approach.

## Steps

1. **C++ Implementation** (pkgs/ndarray/hook/custom_ufuncs.cpp & custom_ufuncs.h):
   - Add template helper real_sinc<T> for float and double.
   - Add template helper complex_sinc_impl<T> for standard complex types (std::complex<T>).
   - Implement cpx_sinc and cpx_sinc_f wrappers.
   - Define contiguous vector kernels:
     - v_sinc_double
     - v_sinc_float
     - v_sinc_complex128
     - v_sinc_complex64
   - Define strided vector kernels using macros:
     - s_sinc_double
     - s_sinc_float
     - s_sinc_complex128
     - s_sinc_complex64
   - Declare all FFI endpoints in custom_ufuncs.h.

2. **FFI Bindings**:
   - Run dart run ffigen to update the FFI bindings in pkgs/ndarray/lib/src/ndarray_bindings.dart.

3. **Dart Integration** (pkgs/ndarray/lib/src/operations/math.dart):
   - Expose the public sinc function.
   - For integer inputs, cast-promote them to float64 array using promoteToDouble and recursively call sinc.
   - Dispatch on contiguous vs strided, switching on a.dtype.
   - Ensure generic casting bug fix: NDArray<R>.create(a.shape, targetDType as DType<R>).

4. **Testing** (pkgs/ndarray/test/math/sinc_test.dart):
   - Combine tests from both prior attempts.
   - Verify contiguous, strided, real, complex, boundary cases (x=0, |x| < 10^-4 Taylor expansion, large values), out recycler, and integer fallback.

5. **Verification**:
   - Run the newly created test file sinc_test.dart.
   - Run the entire math test suite (test/math).
