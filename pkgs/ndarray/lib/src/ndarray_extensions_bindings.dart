// ignore_for_file: non_constant_identifier_names
@ffi.DefaultAsset('package:ndarray/ndarray')
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
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Int>,
    ffi.Pointer<ffi.Void>,
  )
>()
external void s_det_double(
  ffi.Pointer<ffi.Double> a,
  ffi.Pointer<ffi.Int> stridesA,
  ffi.Pointer<ffi.Double> res,
  ffi.Pointer<ffi.Int> stridesRes,
  ffi.Pointer<ffi.Int> shape,
  int rank,
  ffi.Pointer<ffi.Double> aCopy,
  ffi.Pointer<ffi.Int> ipiv,
  ffi.Pointer<ffi.Void> lapack_getrf,
);

@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Int>,
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Int>,
    ffi.Pointer<ffi.Int>,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Int>,
    ffi.Pointer<ffi.Void>,
  )
>()
external void s_det_float(
  ffi.Pointer<ffi.Float> a,
  ffi.Pointer<ffi.Int> stridesA,
  ffi.Pointer<ffi.Float> res,
  ffi.Pointer<ffi.Int> stridesRes,
  ffi.Pointer<ffi.Int> shape,
  int rank,
  ffi.Pointer<ffi.Float> aCopy,
  ffi.Pointer<ffi.Int> ipiv,
  ffi.Pointer<ffi.Void> lapack_getrf,
);

@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Int>,
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Int>,
    ffi.Pointer<ffi.Int>,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Int>,
    ffi.Pointer<ffi.Void>,
  )
>()
external void s_det_complex_double(
  ffi.Pointer<ffi.Double> a,
  ffi.Pointer<ffi.Int> stridesA,
  ffi.Pointer<ffi.Double> res,
  ffi.Pointer<ffi.Int> stridesRes,
  ffi.Pointer<ffi.Int> shape,
  int rank,
  ffi.Pointer<ffi.Double> aCopy,
  ffi.Pointer<ffi.Int> ipiv,
  ffi.Pointer<ffi.Void> lapack_getrf,
);

@ffi.Native<
  ffi.Void Function(
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Int>,
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Int>,
    ffi.Pointer<ffi.Int>,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Int>,
    ffi.Pointer<ffi.Void>,
  )
>()
external void s_det_complex_float(
  ffi.Pointer<ffi.Float> a,
  ffi.Pointer<ffi.Int> stridesA,
  ffi.Pointer<ffi.Float> res,
  ffi.Pointer<ffi.Int> stridesRes,
  ffi.Pointer<ffi.Int> shape,
  int rank,
  ffi.Pointer<ffi.Float> aCopy,
  ffi.Pointer<ffi.Int> ipiv,
  ffi.Pointer<ffi.Void> lapack_getrf,
);
