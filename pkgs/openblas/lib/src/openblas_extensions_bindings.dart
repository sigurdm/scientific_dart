@ffi.DefaultAsset('package:openblas/openblas_extensions')
library;

import 'dart:ffi' as ffi;

@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Int>,
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Int>,
    ffi.Pointer<ffi.Int>,
    ffi.Int,
  )
>()
external void s_det_double(
  ffi.Pointer<ffi.Double> a,
  ffi.Pointer<ffi.Int> stridesA,
  ffi.Pointer<ffi.Double> res,
  ffi.Pointer<ffi.Int> stridesRes,
  ffi.Pointer<ffi.Int> shape,
  int rank,
);
