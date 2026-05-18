@ffi.DefaultAsset('package:openblas/openblas')
library;

import 'dart:ffi' as ffi;
import 'openblas_bindings.dart'; // For blasint!

@ffi.Native<
  ffi.Void Function(
    ffi.Int,
    ffi.Int,
    ffi.Int,
    blasint,
    blasint,
    blasint,
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Float>,
    blasint,
    ffi.Pointer<ffi.Float>,
    blasint,
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Float>,
    blasint,
  )
>()
external void cblas_cgemm(
  int order,
  int transA,
  int transB,
  int m,
  int n,
  int k,
  ffi.Pointer<ffi.Float> alpha,
  ffi.Pointer<ffi.Float> a,
  int lda,
  ffi.Pointer<ffi.Float> b,
  int ldb,
  ffi.Pointer<ffi.Float> beta,
  ffi.Pointer<ffi.Float> c,
  int ldc,
);

@ffi.Native<
  ffi.Void Function(
    ffi.Int,
    ffi.Int,
    ffi.Int,
    blasint,
    blasint,
    blasint,
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Double>,
    blasint,
    ffi.Pointer<ffi.Double>,
    blasint,
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Double>,
    blasint,
  )
>()
external void cblas_zgemm(
  int order,
  int transA,
  int transB,
  int m,
  int n,
  int k,
  ffi.Pointer<ffi.Double> alpha,
  ffi.Pointer<ffi.Double> a,
  int lda,
  ffi.Pointer<ffi.Double> b,
  int ldb,
  ffi.Pointer<ffi.Double> beta,
  ffi.Pointer<ffi.Double> c,
  int ldc,
);
