// ignore_for_file: non_constant_identifier_names
@ffi.DefaultAsset('package:openblas/openblas_extensions')
library;

import 'dart:ffi' as ffi;

@ffi.Native<ffi.Pointer<ffi.Void> Function()>()
external ffi.Pointer<ffi.Void> get_dgetrf_ptr();

@ffi.Native<ffi.Pointer<ffi.Void> Function()>()
external ffi.Pointer<ffi.Void> get_sgetrf_ptr();

@ffi.Native<ffi.Pointer<ffi.Void> Function()>()
external ffi.Pointer<ffi.Void> get_zgetrf_ptr();

@ffi.Native<ffi.Pointer<ffi.Void> Function()>()
external ffi.Pointer<ffi.Void> get_cgetrf_ptr();
