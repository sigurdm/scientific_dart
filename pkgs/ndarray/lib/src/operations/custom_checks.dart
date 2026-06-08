// ignore_for_file: non_constant_identifier_names
@ffi.DefaultAsset('package:ndarray/ndarray')
library;

import 'dart:ffi' as ffi;

@ffi.Native<ffi.Uint8 Function(ffi.Pointer<ffi.Int32>, ffi.Int)>()
external int v_any_less_than_zero_int32(ffi.Pointer<ffi.Int32> arr, int size);

@ffi.Native<ffi.Uint8 Function(ffi.Pointer<ffi.Int64>, ffi.Int)>()
external int v_any_less_than_zero_int64(ffi.Pointer<ffi.Int64> arr, int size);

@ffi.Native<ffi.Uint8 Function(ffi.Pointer<ffi.Int32>, ffi.Int)>()
external int v_any_equal_to_zero_int32(ffi.Pointer<ffi.Int32> arr, int size);

@ffi.Native<ffi.Uint8 Function(ffi.Pointer<ffi.Int64>, ffi.Int)>()
external int v_any_equal_to_zero_int64(ffi.Pointer<ffi.Int64> arr, int size);

@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Int>,
    ffi.Int32,
    ffi.Pointer<ffi.Void>,
    ffi.Int32,
    ffi.Pointer<ffi.Int>,
    ffi.Int32,
  )
>()
external void s_cast_generic(
  ffi.Pointer<ffi.Void> src_ptr,
  ffi.Pointer<ffi.Int> stridesSrc,
  int dtypeSrc,
  ffi.Pointer<ffi.Void> dest_ptr,
  int dtypeDst,
  ffi.Pointer<ffi.Int> shape,
  int rank,
);

@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<ffi.Void>,
    ffi.Pointer<ffi.Void>,
    ffi.Int,
    ffi.Int,
    ffi.Int32,
  )
>()
external void v_extract_upper_triangular(
  ffi.Pointer<ffi.Void> src_ptr,
  ffi.Pointer<ffi.Void> dest_ptr,
  int k,
  int n,
  int dtype,
);

@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<ffi.Void>,
    ffi.Int,
    ffi.Int32,
  )
>()
external void v_zero_upper_triangular(
  ffi.Pointer<ffi.Void> ptr,
  int n,
  int dtype,
);
