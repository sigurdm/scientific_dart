@ffi.DefaultAsset('package:openblas/openblas')
library operations;

import 'dart:typed_data';
import 'dart:math' as math;
import 'ndarray.dart';
import 'broadcasting.dart';
import 'package:openblas/openblas.dart';
import 'dart:ffi' as ffi;
import 'package:ffi/ffi.dart';
import 'numdart_bindings.dart';

// LAPACK Extension Bindings linking explicitly to package:openblas asset via DefaultAsset

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Uint8,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
    ffi.Int,
  )
>()
external int LAPACKE_dpotrf(
  int matrix_layout,
  int uplo,
  int n,
  ffi.Pointer<ffi.Double> a,
  int lda,
);

@ffi.Native<
  ffi.Int Function(ffi.Int, ffi.Uint8, ffi.Int, ffi.Pointer<ffi.Float>, ffi.Int)
>()
external int LAPACKE_spotrf(
  int matrix_layout,
  int uplo,
  int n,
  ffi.Pointer<ffi.Float> a,
  int lda,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
  )
>()
external int LAPACKE_dgeqrf(
  int matrix_layout,
  int m,
  int n,
  ffi.Pointer<ffi.Double> a,
  int lda,
  ffi.Pointer<ffi.Double> tau,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
  )
>()
external int LAPACKE_sgeqrf(
  int matrix_layout,
  int m,
  int n,
  ffi.Pointer<ffi.Float> a,
  int lda,
  ffi.Pointer<ffi.Float> tau,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
  )
>()
external int LAPACKE_dorgqr(
  int matrix_layout,
  int m,
  int n,
  int k,
  ffi.Pointer<ffi.Double> a,
  int lda,
  ffi.Pointer<ffi.Double> tau,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
  )
>()
external int LAPACKE_sorgqr(
  int matrix_layout,
  int m,
  int n,
  int k,
  ffi.Pointer<ffi.Float> a,
  int lda,
  ffi.Pointer<ffi.Float> tau,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Uint8,
    ffi.Uint8,
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Double>,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
  )
>()
external int LAPACKE_dgesvd(
  int matrix_layout,
  int jobu,
  int jobvt,
  int m,
  int n,
  ffi.Pointer<ffi.Double> a,
  int lda,
  ffi.Pointer<ffi.Double> s,
  ffi.Pointer<ffi.Double> u,
  int ldu,
  ffi.Pointer<ffi.Double> vt,
  int ldvt,
  ffi.Pointer<ffi.Double> superb,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Uint8,
    ffi.Uint8,
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
  )
>()
external int LAPACKE_sgesvd(
  int matrix_layout,
  int jobu,
  int jobvt,
  int m,
  int n,
  ffi.Pointer<ffi.Float> a,
  int lda,
  ffi.Pointer<ffi.Float> s,
  ffi.Pointer<ffi.Float> u,
  int ldu,
  ffi.Pointer<ffi.Float> vt,
  int ldvt,
  ffi.Pointer<ffi.Float> superb,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
    ffi.Pointer<ffi.Int>,
  )
>()
external int LAPACKE_sgetrf(
  int matrix_layout,
  int m,
  int n,
  ffi.Pointer<ffi.Float> a,
  int lda,
  ffi.Pointer<ffi.Int> ipiv,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
    ffi.Pointer<ffi.Int>,
  )
>()
external int LAPACKE_sgetri(
  int matrix_layout,
  int n,
  ffi.Pointer<ffi.Float> a,
  int lda,
  ffi.Pointer<ffi.Int> ipiv,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Uint8,
    ffi.Uint8,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Double>,
    ffi.Pointer<ffi.Double>,
    ffi.Int,
    ffi.Pointer<ffi.Double>,
    ffi.Int,
  )
>()
external int LAPACKE_dgeev(
  int matrix_layout,
  int jobvl,
  int jobvr,
  int n,
  ffi.Pointer<ffi.Double> a,
  int lda,
  ffi.Pointer<ffi.Double> wr,
  ffi.Pointer<ffi.Double> wi,
  ffi.Pointer<ffi.Double> vl,
  int ldvl,
  ffi.Pointer<ffi.Double> vr,
  int ldvr,
);

@ffi.Native<
  ffi.Int Function(
    ffi.Int,
    ffi.Uint8,
    ffi.Uint8,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Float>,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
    ffi.Pointer<ffi.Float>,
    ffi.Int,
  )
>()
external int LAPACKE_sgeev(
  int matrix_layout,
  int jobvl,
  int jobvr,
  int n,
  ffi.Pointer<ffi.Float> a,
  int lda,
  ffi.Pointer<ffi.Float> wr,
  ffi.Pointer<ffi.Float> wi,
  ffi.Pointer<ffi.Float> vl,
  int ldvl,
  ffi.Pointer<ffi.Float> vr,
  int ldvr,
);

/// Operations that can call out to FFI.
enum Operation { add, matmul }

/// Global configuration for FFI breakover thresholds.
/// Keys are operation names, values are array size thresholds.
final Map<Operation, int> ffiThresholds = {Operation.matmul: 1};

bool listEquals(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

DType _resolveDType(DType a, DType b) {
  if (a == b) return a;
  if (a == DType.complex128 || b == DType.complex128) return DType.complex128;
  if (a == DType.complex64 || b == DType.complex64) {
    if (a == DType.float64 || b == DType.float64) return DType.complex128;
    return DType.complex64;
  }
  if (a == DType.float64 || b == DType.float64) return DType.float64;
  if (a == DType.float32 || b == DType.float32) return DType.float32;
  if (a == DType.int64 || b == DType.int64) return DType.int64;
  return DType.int32;
}

/// Element-wise addition with broadcasting and dtype upcasting support.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray add(NDArray a, NDArray b, {NDArray? out}) {
  final targetDType = _resolveDType(a.dtype, b.dtype);

  // 0. Native C Vector Extension Fast-Path Gate for Contiguous Same-Shape arrays
  if (a.isContiguous && b.isContiguous && listEquals(a.shape, b.shape)) {
    final result = out ?? NDArray.create(a.shape, targetDType);
    if (out != null) {
      if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
        throw ArgumentError(
          'Provided out buffer has incompatible shape or dtype.',
        );
      }
    }
    if (a.dtype == DType.float64 && b.dtype == DType.float64) {
      v_add_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.data.length,
      );
      return result;
    } else if (a.dtype == DType.float32 && b.dtype == DType.float32) {
      v_add_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.data.length,
      );
      return result;
    }
  }

  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final result = out ?? NDArray.create(commonShape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for broadcasting.',
      );
    }
  }
  final resultStrides = NDArray.computeCStrides(commonShape);

  // 0B. Flat Contiguous Complex128 Track Add
  if (a.isContiguous && b.isContiguous && listEquals(a.shape, b.shape)) {
    if (targetDType == DType.complex128 &&
        a.dtype == DType.complex128 &&
        b.dtype == DType.complex128) {
      v_add_complex(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.data.length,
      );
      return result;
    }
  }

  // 0C. General Multidimensional Strided Broadcasting Engine in C (Rank <= 8)
  if (commonShape.length <= 8) {
    final cShape = malloc<ffi.Int>(commonShape.length);
    final cStridesA = malloc<ffi.Int>(stridesA.length);
    final cStridesB = malloc<ffi.Int>(stridesB.length);
    final cStridesRes = malloc<ffi.Int>(resultStrides.length);

    for (var i = 0; i < commonShape.length; i++) {
      cShape[i] = commonShape[i];
      cStridesA[i] = stridesA[i];
      cStridesB[i] = stridesB[i];
      cStridesRes[i] = resultStrides[i];
    }

    try {
      if (targetDType == DType.float64 &&
          a.dtype == DType.float64 &&
          b.dtype == DType.float64) {
        s_add_double(
          a.pointer.cast(),
          cStridesA,
          b.pointer.cast(),
          cStridesB,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
        );
        return result;
      } else if (targetDType == DType.complex128 &&
          a.dtype == DType.complex128 &&
          b.dtype == DType.complex128) {
        s_add_complex(
          a.pointer.cast(),
          cStridesA,
          b.pointer.cast(),
          cStridesB,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
        );
        return result;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesA);
      malloc.free(cStridesB);
      malloc.free(cStridesRes);
    }
  }

  // Fast SIMD path for identical, contiguous Float32 arrays
  if (a.dtype == DType.float32 &&
      b.dtype == DType.float32 &&
      listEquals(a.shape, b.shape) &&
      listEquals(a.strides, NDArray.computeCStrides(a.shape)) &&
      listEquals(b.strides, NDArray.computeCStrides(b.shape))) {
    final aData = a.data as Float32List;
    final bData = b.data as Float32List;
    final resultData = result.data as Float32List;

    final vaList = Float32x4List.view(aData.buffer);
    final vbList = Float32x4List.view(bData.buffer);
    final vrList = Float32x4List.view(resultData.buffer);

    final simdLen = vaList.length;
    for (var i = 0; i < simdLen; i++) {
      vrList[i] = vaList[i] + vbList[i];
    }

    final remainderStart = simdLen * 4;
    for (var i = remainderStart; i < aData.length; i++) {
      resultData[i] = aData[i] + bData[i];
    }
    return result;
  }

  // Statically-generic type dispatch branches
  if (targetDType == DType.complex128 || targetDType == DType.complex64) {
    final rData = result.data as List<Complex>;
    if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
      final aData = a.data as List<Complex>;
      if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
        _elementWiseOp<Complex, Complex, Complex>(
          rData,
          aData,
          b.data as List<Complex>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x + y,
        );
      } else if (b.dtype == DType.float64 || b.dtype == DType.float32) {
        _elementWiseOp<Complex, double, Complex>(
          rData,
          aData,
          b.data as List<double>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x + y,
        );
      } else {
        _elementWiseOp<Complex, int, Complex>(
          rData,
          aData,
          b.data as List<int>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x + y,
        );
      }
    } else if (a.dtype == DType.float64 || a.dtype == DType.float32) {
      final aData = a.data as List<double>;
      if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
        _elementWiseOp<double, Complex, Complex>(
          rData,
          aData,
          b.data as List<Complex>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => y + x,
        );
      }
    } else if (a.dtype == DType.int32 || a.dtype == DType.int64) {
      final aData = a.data as List<int>;
      if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
        _elementWiseOp<int, Complex, Complex>(
          rData,
          aData,
          b.data as List<Complex>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => y + x,
        );
      }
    }
  } else if (targetDType == DType.float64 || targetDType == DType.float32) {
    final rData = result.data as List<double>;
    if (a.dtype == DType.float64 || a.dtype == DType.float32) {
      final aData = a.data as List<double>;
      if (b.dtype == DType.float64 || b.dtype == DType.float32) {
        _elementWiseOp<double, double, double>(
          rData,
          aData,
          b.data as List<double>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x + y,
        );
      } else {
        _elementWiseOp<double, int, double>(
          rData,
          aData,
          b.data as List<int>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x + y,
        );
      }
    } else if (a.dtype == DType.int32 || a.dtype == DType.int64) {
      final aData = a.data as List<int>;
      if (b.dtype == DType.float64 || b.dtype == DType.float32) {
        _elementWiseOp<int, double, double>(
          rData,
          aData,
          b.data as List<double>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x + y,
        );
      }
    }
  } else {
    _elementWiseOp<int, int, int>(
      result.data as List<int>,
      a.data as List<int>,
      b.data as List<int>,
      commonShape,
      stridesA,
      stridesB,
      resultStrides,
      0,
      0,
      0,
      0,
      (x, y) => x + y,
    );
  }

  return result;
}

void _elementWiseOp<Ta, Tb, Tr>(
  List<Tr> result,
  List<Ta> a,
  List<Tb> b,
  List<int> shape,
  List<int> stridesA,
  List<int> stridesB,
  List<int> stridesResult,
  int dim,
  int offsetA,
  int offsetB,
  int offsetResult,
  Tr Function(Ta, Tb) op,
) {
  if (dim == shape.length) {
    result[offsetResult] = op(a[offsetA], b[offsetB]);
    return;
  }

  for (var i = 0; i < shape[dim]; i++) {
    _elementWiseOp<Ta, Tb, Tr>(
      result,
      a,
      b,
      shape,
      stridesA,
      stridesB,
      stridesResult,
      dim + 1,
      offsetA + i * stridesA[dim],
      offsetB + i * stridesB[dim],
      offsetResult + i * stridesResult[dim],
      op,
    );
  }
}

/// Element-wise subtraction with broadcasting and dtype upcasting support.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray subtract(NDArray a, NDArray b) {
  // 0. Native C Vector Extension Fast-Path Gate for Contiguous Same-Shape arrays
  if (a.isContiguous && b.isContiguous && listEquals(a.shape, b.shape)) {
    if (a.dtype == DType.float64 && b.dtype == DType.float64) {
      final result = NDArray.create(a.shape, DType.float64);
      v_sub_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.data.length,
      );
      return result;
    } else if (a.dtype == DType.float32 && b.dtype == DType.float32) {
      final result = NDArray.create(a.shape, DType.float32);
      v_sub_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.data.length,
      );
      return result;
    }
  }

  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final targetDType = _resolveDType(a.dtype, b.dtype);
  final result = NDArray.create(commonShape, targetDType);
  final resultStrides = NDArray.computeCStrides(commonShape);

  if (targetDType == DType.complex128 || targetDType == DType.complex64) {
    final rData = result.data as List<Complex>;
    if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
      final aData = a.data as List<Complex>;
      if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
        _elementWiseOp<Complex, Complex, Complex>(
          rData,
          aData,
          b.data as List<Complex>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x - y,
        );
      } else if (b.dtype == DType.float64 || b.dtype == DType.float32) {
        _elementWiseOp<Complex, double, Complex>(
          rData,
          aData,
          b.data as List<double>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x - y,
        );
      } else {
        _elementWiseOp<Complex, int, Complex>(
          rData,
          aData,
          b.data as List<int>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x - y,
        );
      }
    } else if (a.dtype == DType.float64 || a.dtype == DType.float32) {
      final aData = a.data as List<double>;
      if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
        _elementWiseOp<double, Complex, Complex>(
          rData,
          aData,
          b.data as List<Complex>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => Complex(x - y.real, -y.imag),
        );
      }
    } else if (a.dtype == DType.int32 || a.dtype == DType.int64) {
      final aData = a.data as List<int>;
      if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
        _elementWiseOp<int, Complex, Complex>(
          rData,
          aData,
          b.data as List<Complex>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => Complex(x - y.real, -y.imag),
        );
      }
    }
  } else if (targetDType == DType.float64 || targetDType == DType.float32) {
    final rData = result.data as List<double>;
    if (a.dtype == DType.float64 || a.dtype == DType.float32) {
      final aData = a.data as List<double>;
      if (b.dtype == DType.float64 || b.dtype == DType.float32) {
        _elementWiseOp<double, double, double>(
          rData,
          aData,
          b.data as List<double>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x - y,
        );
      } else {
        _elementWiseOp<double, int, double>(
          rData,
          aData,
          b.data as List<int>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x - y,
        );
      }
    } else if (a.dtype == DType.int32 || a.dtype == DType.int64) {
      final aData = a.data as List<int>;
      if (b.dtype == DType.float64 || b.dtype == DType.float32) {
        _elementWiseOp<int, double, double>(
          rData,
          aData,
          b.data as List<double>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x - y,
        );
      }
    }
  } else {
    _elementWiseOp<int, int, int>(
      result.data as List<int>,
      a.data as List<int>,
      b.data as List<int>,
      commonShape,
      stridesA,
      stridesB,
      resultStrides,
      0,
      0,
      0,
      0,
      (x, y) => x - y,
    );
  }

  return result;
}

/// Element-wise multiplication with broadcasting and dtype upcasting support.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray multiply(NDArray a, NDArray b) {
  // 0. Native C Vector Extension Fast-Path Gate for Contiguous Same-Shape arrays
  if (a.isContiguous && b.isContiguous && listEquals(a.shape, b.shape)) {
    if (a.dtype == DType.float64 && b.dtype == DType.float64) {
      final result = NDArray.create(a.shape, DType.float64);
      v_mul_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.data.length,
      );
      return result;
    } else if (a.dtype == DType.float32 && b.dtype == DType.float32) {
      final result = NDArray.create(a.shape, DType.float32);
      v_mul_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.data.length,
      );
      return result;
    }
  }

  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  final targetDType = _resolveDType(a.dtype, b.dtype);
  final result = NDArray.create(commonShape, targetDType);
  final resultStrides = NDArray.computeCStrides(commonShape);

  if (targetDType == DType.complex128 || targetDType == DType.complex64) {
    final rData = result.data as List<Complex>;
    if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
      final aData = a.data as List<Complex>;
      if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
        _elementWiseOp<Complex, Complex, Complex>(
          rData,
          aData,
          b.data as List<Complex>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x * y,
        );
      } else if (b.dtype == DType.float64 || b.dtype == DType.float32) {
        _elementWiseOp<Complex, double, Complex>(
          rData,
          aData,
          b.data as List<double>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x * y,
        );
      } else {
        _elementWiseOp<Complex, int, Complex>(
          rData,
          aData,
          b.data as List<int>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x * y,
        );
      }
    } else if (a.dtype == DType.float64 || a.dtype == DType.float32) {
      final aData = a.data as List<double>;
      if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
        _elementWiseOp<double, Complex, Complex>(
          rData,
          aData,
          b.data as List<Complex>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => y * x,
        );
      }
    } else if (a.dtype == DType.int32 || a.dtype == DType.int64) {
      final aData = a.data as List<int>;
      if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
        _elementWiseOp<int, Complex, Complex>(
          rData,
          aData,
          b.data as List<Complex>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => y * x,
        );
      }
    }
  } else if (targetDType == DType.float64 || targetDType == DType.float32) {
    final rData = result.data as List<double>;
    if (a.dtype == DType.float64 || a.dtype == DType.float32) {
      final aData = a.data as List<double>;
      if (b.dtype == DType.float64 || b.dtype == DType.float32) {
        _elementWiseOp<double, double, double>(
          rData,
          aData,
          b.data as List<double>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x * y,
        );
      } else {
        _elementWiseOp<double, int, double>(
          rData,
          aData,
          b.data as List<int>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x * y,
        );
      }
    } else if (a.dtype == DType.int32 || a.dtype == DType.int64) {
      final aData = a.data as List<int>;
      if (b.dtype == DType.float64 || b.dtype == DType.float32) {
        _elementWiseOp<int, double, double>(
          rData,
          aData,
          b.data as List<double>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x * y,
        );
      }
    }
  } else {
    _elementWiseOp<int, int, int>(
      result.data as List<int>,
      a.data as List<int>,
      b.data as List<int>,
      commonShape,
      stridesA,
      stridesB,
      resultStrides,
      0,
      0,
      0,
      0,
      (x, y) => x * y,
    );
  }

  return result;
}

/// Element-wise division with broadcasting and dtype upcasting support.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray divide(NDArray a, NDArray b) {
  // 0. Native C Vector Extension Fast-Path Gate for Contiguous Same-Shape arrays
  if (a.isContiguous && b.isContiguous && listEquals(a.shape, b.shape)) {
    if (a.dtype == DType.float64 && b.dtype == DType.float64) {
      final result = NDArray.create(a.shape, DType.float64);
      v_div_double(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.data.length,
      );
      return result;
    } else if (a.dtype == DType.float32 && b.dtype == DType.float32) {
      final result = NDArray.create(a.shape, DType.float32);
      v_div_float(
        a.pointer.cast(),
        b.pointer.cast(),
        result.pointer.cast(),
        a.data.length,
      );
      return result;
    }
  }

  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;
  final stridesA = broadcastResult.stridesA;
  final stridesB = broadcastResult.stridesB;

  // True division always upcasts to a floating or complex type!
  var targetDType = _resolveDType(a.dtype, b.dtype);
  if (targetDType == DType.int32 || targetDType == DType.int64) {
    targetDType = DType.float64;
  }

  final result = NDArray.create(commonShape, targetDType);
  final resultStrides = NDArray.computeCStrides(commonShape);

  if (targetDType == DType.complex128 || targetDType == DType.complex64) {
    final rData = result.data as List<Complex>;
    if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
      final aData = a.data as List<Complex>;
      if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
        _elementWiseOp<Complex, Complex, Complex>(
          rData,
          aData,
          b.data as List<Complex>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x / y,
        );
      } else if (b.dtype == DType.float64 || b.dtype == DType.float32) {
        _elementWiseOp<Complex, double, Complex>(
          rData,
          aData,
          b.data as List<double>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x / y,
        );
      } else {
        _elementWiseOp<Complex, int, Complex>(
          rData,
          aData,
          b.data as List<int>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x / y,
        );
      }
    } else if (a.dtype == DType.float64 || a.dtype == DType.float32) {
      final aData = a.data as List<double>;
      if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
        // double / Complex:
        _elementWiseOp<double, Complex, Complex>(
          rData,
          aData,
          b.data as List<Complex>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => Complex(x, 0) / y,
        );
      }
    } else if (a.dtype == DType.int32 || a.dtype == DType.int64) {
      final aData = a.data as List<int>;
      if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
        _elementWiseOp<int, Complex, Complex>(
          rData,
          aData,
          b.data as List<Complex>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => Complex(x.toDouble(), 0) / y,
        );
      }
    }
  } else {
    // Floating point results
    final rData = result.data as List<double>;
    if (a.dtype == DType.float64 || a.dtype == DType.float32) {
      final aData = a.data as List<double>;
      if (b.dtype == DType.float64 || b.dtype == DType.float32) {
        _elementWiseOp<double, double, double>(
          rData,
          aData,
          b.data as List<double>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x / y,
        );
      } else {
        _elementWiseOp<double, int, double>(
          rData,
          aData,
          b.data as List<int>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x / y,
        );
      }
    } else if (a.dtype == DType.int32 || a.dtype == DType.int64) {
      final aData = a.data as List<int>;
      if (b.dtype == DType.float64 || b.dtype == DType.float32) {
        _elementWiseOp<int, double, double>(
          rData,
          aData,
          b.data as List<double>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x / y,
        );
      } else {
        // int / int -> double
        _elementWiseOp<int, int, double>(
          rData,
          aData,
          b.data as List<int>,
          commonShape,
          stridesA,
          stridesB,
          resultStrides,
          0,
          0,
          0,
          0,
          (x, y) => x / y,
        );
      }
    }
  }

  return result;
}

/// Matrix multiplication for Float64 arrays using OpenBLAS.
List<int> _broadcastStackShapes(List<int> sA, List<int> sB) {
  final result = <int>[];
  final lenA = sA.length;
  final lenB = sB.length;
  final maxLen = math.max(lenA, lenB);

  for (var i = 0; i < maxLen; i++) {
    final dimA = (lenA - 1 - i >= 0) ? sA[lenA - 1 - i] : 1;
    final dimB = (lenB - 1 - i >= 0) ? sB[lenB - 1 - i] : 1;

    if (dimA == dimB) {
      result.insert(0, dimA);
    } else if (dimA == 1) {
      result.insert(0, dimB);
    } else if (dimB == 1) {
      result.insert(0, dimA);
    } else {
      throw ArgumentError(
        'Incompatible stack shapes for broadcasting in matmul: $sA and $sB',
      );
    }
  }
  return result;
}

/// Matrix multiplication for Float64 and Float32 arrays using OpenBLAS, supporting high-dimensional stack broadcasting and 1D vector promotions.
NDArray<double> matmul(NDArray<double> a, NDArray<double> b) {
  final targetDType = (a.dtype == DType.float32 || b.dtype == DType.float32)
      ? DType.float32
      : DType.float64;

  if (a.shape.length == 1 && b.shape.length == 1) {
    final n = a.shape[0];
    if (n != b.shape[0]) {
      throw ArgumentError(
        'Incompatible vector dimensions for 1D dot product in matmul: ${a.shape} and ${b.shape}',
      );
    }
    if (targetDType == DType.float64) {
      final scalarRes = cblas_ddot(
        n,
        a.pointer.cast<ffi.Double>(),
        1,
        b.pointer.cast<ffi.Double>(),
        1,
      );
      return NDArray.fromList([scalarRes], [], DType.float64);
    } else {
      final scalarRes = cblas_sdot(
        n,
        a.pointer.cast<ffi.Float>(),
        1,
        b.pointer.cast<ffi.Float>(),
        1,
      );
      return NDArray.fromList([scalarRes], [], DType.float32);
    }
  }

  // Copy upfront ONLY if neither inner strides is 1 (very rare custom sliced strides)
  if (a.shape.length >= 2) {
    final r = a.shape.length;
    if (a.strides[r - 1] != 1 && a.strides[r - 2] != 1) {
      a = NDArray.fromList(a.toList(), a.shape, a.dtype);
    }
  }
  if (b.shape.length >= 2) {
    final r = b.shape.length;
    if (b.strides[r - 1] != 1 && b.strides[r - 2] != 1) {
      b = NDArray.fromList(b.toList(), b.shape, b.dtype);
    }
  }

  var aPromoted = false;
  var bPromoted = false;

  NDArray<double> aView = a;
  if (a.shape.length == 1) {
    aView = NDArray.view(
      a,
      [1, a.shape[0]],
      [0, a.strides[0]],
      offsetElements: 0,
    );
    aPromoted = true;
  }

  NDArray<double> bView = b;
  if (b.shape.length == 1) {
    bView = NDArray.view(
      b,
      [b.shape[0], 1],
      [b.strides[0], 0],
      offsetElements: 0,
    );
    bPromoted = true;
  }

  final rankA = aView.shape.length;
  final rankB = bView.shape.length;

  final m = aView.shape[rankA - 2];
  final kA = aView.shape[rankA - 1];
  final kB = bView.shape[rankB - 2];
  final n = bView.shape[rankB - 1];

  if (kA != kB) {
    throw ArgumentError(
      'Incompatible inner matrix dimensions for matmul: kA($kA) != kB($kB). Shapes: ${a.shape} and ${b.shape}',
    );
  }

  final stackA = aView.shape.sublist(0, rankA - 2);
  final stackB = bView.shape.sublist(0, rankB - 2);
  final broadcastStack = _broadcastStackShapes(stackA, stackB);

  final resShape = [...broadcastStack, m, n];
  final result = NDArray<double>.zeros(resShape, targetDType);

  // Stride resolution logic for 100% copy-free BLAS matrix multiplication
  var transA = 111; // CblasNoTrans
  var lda = kA;
  if (!aPromoted) {
    if (aView.strides[rankA - 1] == 1) {
      transA = 111;
      lda = aView.strides[rankA - 2];
    } else if (aView.strides[rankA - 2] == 1) {
      transA = 112; // CblasTrans
      lda = aView.strides[rankA - 1];
    }
  }

  var transB = 111; // CblasNoTrans
  var ldb = n;
  if (!bPromoted) {
    if (bView.strides[rankB - 1] == 1) {
      transB = 111;
      ldb = bView.strides[rankB - 2];
    } else if (bView.strides[rankB - 2] == 1) {
      transB = 112; // CblasTrans
      ldb = bView.strides[rankB - 1];
    }
  }

  final lenA = stackA.length;
  final lenB = stackB.length;
  final lenResult = broadcastStack.length;

  final walkStridesA = List<int>.filled(lenResult, 0);
  final walkStridesB = List<int>.filled(lenResult, 0);

  for (var i = 0; i < lenResult; i++) {
    final resAxis = lenResult - 1 - i;
    final axisA = lenA - 1 - i;
    final axisB = lenB - 1 - i;

    if (axisA >= 0) {
      walkStridesA[resAxis] = (stackA[axisA] == broadcastStack[resAxis])
          ? aView.strides[axisA]
          : 0;
    } else {
      walkStridesA[resAxis] = 0;
    }

    if (axisB >= 0) {
      walkStridesB[resAxis] = (stackB[axisB] == broadcastStack[resAxis])
          ? bView.strides[axisB]
          : 0;
    } else {
      walkStridesB[resAxis] = 0;
    }
  }

  final walkStridesRes = List<int>.filled(lenResult, 0);
  var resStride = m * n;
  for (var i = lenResult - 1; i >= 0; i--) {
    walkStridesRes[i] = resStride;
    resStride *= broadcastStack[i];
  }

  void walk(int dim, int offsetA, int offsetB, int offsetRes) {
    if (dim == lenResult) {
      if (targetDType == DType.float64) {
        cblas_dgemm(
          101, // CblasRowMajor
          transA,
          transB,
          m,
          n,
          kA,
          1.0,
          aView.pointer.cast<ffi.Double>().elementAt(offsetA),
          lda,
          bView.pointer.cast<ffi.Double>().elementAt(offsetB),
          ldb,
          0.0,
          result.pointer.cast<ffi.Double>().elementAt(offsetRes),
          n, // ldc (result is always contiguous row-major)
        );
      } else {
        cblas_sgemm(
          101, // CblasRowMajor
          transA,
          transB,
          m,
          n,
          kA,
          1.0,
          aView.pointer.cast<ffi.Float>().elementAt(offsetA),
          lda,
          bView.pointer.cast<ffi.Float>().elementAt(offsetB),
          ldb,
          0.0,
          result.pointer.cast<ffi.Float>().elementAt(offsetRes),
          n, // ldc (result is always contiguous row-major)
        );
      }
      return;
    }

    final size = broadcastStack[dim];
    final strideA = walkStridesA[dim];
    final strideB = walkStridesB[dim];
    final strideRes = walkStridesRes[dim];

    for (var i = 0; i < size; i++) {
      walk(
        dim + 1,
        offsetA + i * strideA,
        offsetB + i * strideB,
        offsetRes + i * strideRes,
      );
    }
  }

  walk(0, 0, 0, 0);

  // Post-calculation 1D dummy dimensions demotions
  if (aPromoted && bPromoted) {
    return result.reshape([]); // 0D scalar array for pure vector dot products
  } else if (aPromoted) {
    final newShape = List<int>.from(result.shape)
      ..removeAt(result.shape.length - 2);
    return result.reshape(newShape);
  } else if (bPromoted) {
    final newShape = List<int>.from(result.shape)
      ..removeAt(result.shape.length - 1);
    return result.reshape(newShape);
  }

  return result;
}

/// Compute the sum of elements in the array.
///
/// If [axis] is provided, sums along that axis and returns a new array.
/// Otherwise, sums all elements and returns a scalar value of type [T].
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final s0 = sum(a, axis: 0); // Sum along rows
/// print(s0.data); // [4.0, 6.0]
/// ```
dynamic sum<T extends Object>(NDArray<T> a, {int? axis}) {
  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    if (a.isContiguous) {
      if (a.dtype == DType.float64) {
        return r_sum_double(a.pointer.cast(), size) as T;
      } else if (a.dtype == DType.float32) {
        return r_sum_float(a.pointer.cast(), size) as T;
      }
    }
    final List<T> elements = size == a.data.length ? a.data : a.toList();
    var acc = elements.first as dynamic;
    for (var i = 1; i < elements.length; i++) {
      acc += elements[i];
    }
    return acc as T;
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final result = NDArray<T>.zeros(newShape, a.dtype);

  _reduceRecursive(
    a,
    result,
    List<int>.filled(a.shape.length, 0),
    List<int>.filled(newShape.length, 0),
    axis,
    0,
    (current, val) => ((current as dynamic) + val) as T,
  );
  return result;
}

/// Compute the product of elements in the array.
///
/// If [axis] is provided, multiplies along that axis and returns a new array.
/// Otherwise, multiplies all elements and returns a scalar value of type [T].
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final p0 = prod(a, axis: 0); // Product along rows
/// print(p0.data); // [3.0, 8.0]
/// ```
dynamic prod<T extends Object>(NDArray<T> a, {int? axis}) {
  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    if (a.isContiguous) {
      if (a.dtype == DType.float64) {
        return r_prod_double(a.pointer.cast(), size) as T;
      } else if (a.dtype == DType.float32) {
        return r_prod_float(a.pointer.cast(), size) as T;
      }
    }
    final List<T> elements = size == a.data.length ? a.data : a.toList();
    var acc = elements.first as dynamic;
    for (var i = 1; i < elements.length; i++) {
      acc *= elements[i];
    }
    return acc as T;
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final result = NDArray<T>.ones(newShape, a.dtype);

  _reduceRecursive(
    a,
    result,
    List<int>.filled(a.shape.length, 0),
    List<int>.filled(newShape.length, 0),
    axis,
    0,
    (current, val) => ((current as dynamic) * val) as T,
  );
  return result;
}

/// Returns true if all elements along a given [axis] evaluate to True.
///
/// If [axis] is omitted/null, performs a global reduction and returns a single Dart [bool].
///
/// **Preconditions:**
/// - The array [a] must not be disposed.
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([true, true, false], [3], DType.boolean);
/// final res = all(a); // false
/// ```
dynamic all(NDArray a, {int? axis}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute all() on a disposed array.');
  }

  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final List elements = size == a.data.length ? a.data : a.toList();
    for (var i = 0; i < elements.length; i++) {
      if (!_isTrue(elements[i])) {
        return false;
      }
    }
    return true;
  }

  var targetAxis = axis;
  if (targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }
  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(targetAxis);
  final result = NDArray<bool>.create(newShape, DType.boolean);
  result.fill(true); // Initialize to true everywhere

  _reduceRecursive<Object, bool>(
    a as NDArray<Object>,
    result,
    List<int>.filled(a.shape.length, 0),
    List<int>.filled(newShape.length, 0),
    targetAxis,
    0,
    (current, val) => current && _isTrue(val),
  );

  return result;
}

/// Returns true if any element along a given [axis] evaluates to True.
///
/// If [axis] is omitted/null, performs a global reduction and returns a single Dart [bool].
///
/// **Preconditions:**
/// - The array [a] must not be disposed.
/// - If provided, [axis] must be within bounds `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([true, false, false], [3], DType.boolean);
/// final res = any(a); // true
/// ```
dynamic any(NDArray a, {int? axis}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute any() on a disposed array.');
  }

  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final List elements = size == a.data.length ? a.data : a.toList();
    for (var i = 0; i < elements.length; i++) {
      if (_isTrue(elements[i])) {
        return true;
      }
    }
    return false;
  }

  var targetAxis = axis;
  if (targetAxis < 0) {
    targetAxis = a.shape.length + targetAxis;
  }
  if (targetAxis < 0 || targetAxis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(targetAxis);
  final result = NDArray<bool>.zeros(newShape, DType.boolean); // Pre-initialized to false

  _reduceRecursive<Object, bool>(
    a as NDArray<Object>,
    result,
    List<int>.filled(a.shape.length, 0),
    List<int>.filled(newShape.length, 0),
    targetAxis,
    0,
    (current, val) => current || _isTrue(val),
  );

  return result;
}

/// Compute the element-wise square root of the array.
///
/// Returns a new array with the results.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 4.0, 9.0], [3], DType.float64);
/// final b = sqrt(a);
/// print(b.data); // [1.0, 2.0, 3.0]
/// ```
///
/// **Gotchas:**
/// - Negative values will result in [double.nan].
NDArray<double> sqrt<T extends num>(NDArray<T> a, {NDArray? out}) {
  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  final result =
      out as NDArray<double>? ?? NDArray<double>.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for sqrt.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_sqrt_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_sqrt_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  }

  for (var i = 0; i < a.data.length; i++) {
    result.data[i] = math.sqrt(a.data[i].toDouble());
  }
  return result;
}

/// Compute the element-wise sine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous array layouts, offloads the loop directly to high-speed native C
///   vector math kernels (`v_sin_double`/`v_sin_float`), bypassing all Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/transcendental_example.dart lang=dart}
///
/// Reference: [Trigonometric Sine Function](https://en.wikipedia.org/wiki/Sine_and_cosine)
NDArray<double> sin<T extends num>(NDArray<T> a, {NDArray? out}) {
  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  final result =
      out as NDArray<double>? ?? NDArray<double>.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for sin.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_sin_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_sin_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  }

  for (var i = 0; i < a.data.length; i++) {
    result.data[i] = math.sin(a.data[i].toDouble());
  }
  return result;
}

/// Compute the element-wise cosine of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous array layouts, offloads the loop directly to high-speed native C
///   vector math kernels (`v_cos_double`/`v_cos_float`), bypassing all Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/transcendental_example.dart lang=dart}
///
/// Reference: [Trigonometric Cosine Function](https://en.wikipedia.org/wiki/Sine_and_cosine)
NDArray<double> cos<T extends num>(NDArray<T> a, {NDArray? out}) {
  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  final result =
      out as NDArray<double>? ?? NDArray<double>.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for cos.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_cos_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_cos_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  }

  for (var i = 0; i < a.data.length; i++) {
    result.data[i] = math.cos(a.data[i].toDouble());
  }
  return result;
}

/// Compute the element-wise exponential of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous array layouts, offloads the loop directly to high-speed native C
///   vector math kernels (`v_exp_double`/`v_exp_float`), bypassing all Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/transcendental_example.dart lang=dart}
///
/// Reference: [Exponential Function](https://en.wikipedia.org/wiki/Exponential_function)
NDArray<double> exp<T extends num>(NDArray<T> a, {NDArray? out}) {
  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  final result =
      out as NDArray<double>? ?? NDArray<double>.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for exp.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_exp_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_exp_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  }

  for (var i = 0; i < a.data.length; i++) {
    result.data[i] = math.exp(a.data[i].toDouble());
  }
  return result;
}

/// Compute the element-wise natural logarithm of the array.
///
/// **Preconditions:**
/// - Input array [a] elements must be numeric (`T extends num`).
/// - If provided, the [out] recycler array must exactly match the shape and compatible dtype of [a].
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total number of elements.
/// - For C-contiguous array layouts, offloads the loop directly to high-speed native C
///   vector math kernels (`v_log_double`/`v_log_float`), bypassing all Dart VM loop overhead.
///
/// **Example:**
/// {@example /example/transcendental_example.dart lang=dart}
///
/// Reference: [Natural Logarithm](https://en.wikipedia.org/wiki/Natural_logarithm)
NDArray<double> log<T extends num>(NDArray<T> a, {NDArray? out}) {
  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  final result =
      out as NDArray<double>? ?? NDArray<double>.create(a.shape, targetDType);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for log.',
      );
    }
  }

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_log_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_log_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  }

  for (var i = 0; i < a.data.length; i++) {
    result.data[i] = math.log(a.data[i].toDouble());
  }
  return result;
}

/// Compute the mean of elements in the array.
///
/// **Gotchas:**
/// - Returns a scalar if [axis] is null, or a new [NDArray] if [axis] is provided.
dynamic mean<T extends Object>(NDArray<T> a, {int? axis}) {
  final s = sum(a, axis: axis);
  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    return s / size;
  } else {
    final sizeAxis = a.shape[axis];
    for (var i = 0; i < s.data.length; i++) {
      s.data[i] = (s.data[i] / sizeAxis) as T;
    }
    return s;
  }
}

/// Compute the variance along the specified axis.
///
/// Returns the variance of the array elements, a measure of the spread of a
/// distribution. The variance is computed for the flattened array by default,
/// otherwise over the specified axis.
///
/// **Gotchas:**
/// - Returns a scalar if [axis] is null, or a new [NDArray] if [axis] is provided.
/// - Currently does not support `ddof` (Delta Degrees of Freedom).
dynamic variance<T extends num>(NDArray<T> a, {int? axis}) {
  final m = mean(a, axis: axis);

  if (axis == null) {
    var sumSqDiff = 0.0;
    final meanVal = m as double;

    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final List<num> elements = size == a.data.length
        ? a.data as List<num>
        : a.toList() as List<num>;

    for (var i = 0; i < elements.length; i++) {
      final diff = elements[i].toDouble() - meanVal;
      sumSqDiff += diff * diff;
    }
    return sumSqDiff / elements.length;
  } else {
    // Reshape m to keep dimensions for broadcasting
    final targetShape = List<int>.from(a.shape);
    targetShape[axis] = 1;
    final reshapedM = (m as NDArray<T>).reshape(targetShape);

    final diff = subtract(a, reshapedM);
    final sqDiff = multiply(diff, diff);

    // Convert to NDArray<double> to avoid truncation in mean
    final sqDiffDouble = NDArray<double>.create(sqDiff.shape, DType.float64);
    for (var i = 0; i < sqDiff.data.length; i++) {
      sqDiffDouble.data[i] = sqDiff.data[i].toDouble();
    }

    return mean(sqDiffDouble, axis: axis);
  }
}

/// Compute the standard deviation along the specified axis.
///
/// Returns the standard deviation, a measure of the spread of a distribution,
/// of the array elements. The standard deviation is computed for the
/// flattened array by default, otherwise over the specified axis.
dynamic std<T extends num>(NDArray<T> a, {int? axis}) {
  final v = variance(a, axis: axis);
  if (axis == null) {
    return math.sqrt(v as double);
  } else {
    return sqrt(v as NDArray<double>);
  }
}

/// Compute the sum of array elements along a specified axis, treating NaNs as zeros.
///
/// **Preconditions:**
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total elements count.
///
/// **Example:**
/// ```dart
/// final a = NDArray<double>.fromList([1.0, double.nan, 3.0, double.nan], [2, 2], DType.float64);
/// final s = nansum(a); // returns 4.0
/// ```
dynamic nansum<T extends Object>(NDArray<T> a, {int? axis}) {
  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final List<T> elements = size == a.data.length ? a.data : a.toList();
    var acc = 0.0;
    for (var i = 0; i < elements.length; i++) {
      final val = elements[i];
      if (val is double && val.isNaN) continue;
      acc += (val as num).toDouble();
    }
    return acc as T;
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final result = NDArray<T>.zeros(newShape, a.dtype);

  _reduceRecursive(
    a,
    result,
    List<int>.filled(a.shape.length, 0),
    List<int>.filled(newShape.length, 0),
    axis,
    0,
    (current, val) {
      if (val is double && val.isNaN) return current;
      return ((current as dynamic) + val) as T;
    },
  );
  return result;
}

/// Compute the arithmetic mean along a specified axis, ignoring NaNs.
///
/// **Preconditions:**
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total elements count, walking
///   coordinate strides and tracking counts dynamically.
///
/// **Example:**
/// ```dart
/// final a = NDArray<double>.fromList([1.0, double.nan, 3.0, 4.0], [2, 2], DType.float64);
/// final m = nanmean(a); // returns 2.6666666666666665
/// ```
dynamic nanmean<T extends Object>(NDArray<T> a, {int? axis}) {
  if (axis == null) {
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final List<T> elements = size == a.data.length ? a.data : a.toList();
    var sumVal = 0.0;
    var count = 0;
    for (var i = 0; i < elements.length; i++) {
      final val = elements[i];
      if (val is double && val.isNaN) continue;
      sumVal += (val as num).toDouble();
      count++;
    }
    if (count == 0) return double.nan;
    return sumVal / count;
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final result = NDArray<T>.zeros(newShape, a.dtype);
  final counts = NDArray<int>.zeros(newShape, DType.int32);

  _nanReduceRecursive(
    a,
    result,
    counts,
    List<int>.filled(a.shape.length, 0),
    List<int>.filled(newShape.length, 0),
    axis,
    0,
  );

  for (var i = 0; i < result.data.length; i++) {
    final c = counts.data[i];
    if (c == 0) {
      result.data[i] = double.nan as T;
    } else {
      result.data[i] = ((result.data[i] as dynamic) / c) as T;
    }
  }
  return result;
}

/// Recursive helper to accumulate sum and count of non-NaN elements along an axis.
void _nanReduceRecursive<T extends Object>(
  NDArray<T> a,
  NDArray<T> result,
  NDArray<int> counts,
  List<int> coordA,
  List<int> coordRes,
  int axis,
  int dim,
) {
  if (dim == a.shape.length) {
    final val = a.getCell(coordA);
    if (val is double && val.isNaN) return;
    final current = result.getCell(coordRes);
    result.setCell(coordRes, ((current as dynamic) + val) as T);
    counts.setCell(coordRes, counts.getCell(coordRes) + 1);
    return;
  }
  if (dim == axis) {
    for (var i = 0; i < a.shape[axis]; i++) {
      coordA[dim] = i;
      _nanReduceRecursive(a, result, counts, coordA, coordRes, axis, dim + 1);
    }
  } else {
    final resDim = dim < axis ? dim : dim - 1;
    for (var i = 0; i < a.shape[dim]; i++) {
      coordA[dim] = i;
      coordRes[resDim] = i;
      _nanReduceRecursive(a, result, counts, coordA, coordRes, axis, dim + 1);
    }
  }
}

/// Compute the variance along the specified axis, ignoring NaNs.
///
/// **Preconditions:**
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total elements count.
///
/// **Example:**
/// ```dart
/// final a = NDArray<double>.fromList([1.0, double.nan, 2.0, 3.0], [2, 2], DType.float64);
/// final v = nanvar(a); // returns 0.6666666666666666
/// ```
dynamic nanvar<T extends num>(NDArray<T> a, {int? axis}) {
  final m = nanmean(a, axis: axis);

  if (axis == null) {
    var sumSqDiff = 0.0;
    final meanVal = m as double;
    if (meanVal.isNaN) return double.nan;

    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    final List<num> elements = size == a.data.length
        ? a.data as List<num>
        : a.toList() as List<num>;

    var count = 0;
    for (var i = 0; i < elements.length; i++) {
      final val = elements[i].toDouble();
      if (val.isNaN) continue;
      final diff = val - meanVal;
      sumSqDiff += diff * diff;
      count++;
    }
    if (count == 0) return double.nan;
    return sumSqDiff / count;
  } else {
    // Reshape m to keep dimensions for broadcasting
    final targetShape = List<int>.from(a.shape);
    targetShape[axis] = 1;
    final reshapedM = (m as NDArray<T>).reshape(targetShape);

    final diff = subtract(a, reshapedM);
    final sqDiff = multiply(diff, diff);

    // Convert to NDArray<double> to avoid truncation in nanmean
    final sqDiffDouble = NDArray<double>.create(sqDiff.shape, DType.float64);
    for (var i = 0; i < sqDiff.data.length; i++) {
      sqDiffDouble.data[i] = sqDiff.data[i].toDouble();
    }

    return nanmean(sqDiffDouble, axis: axis);
  }
}

/// Compute the standard deviation along the specified axis, ignoring NaNs.
///
/// **Preconditions:**
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N)$ where $N$ is the total elements count.
///
/// **Example:**
/// ```dart
/// final a = NDArray<double>.fromList([1.0, double.nan, 2.0, 3.0], [2, 2], DType.float64);
/// final s = nanstd(a); // returns sqrt(0.6666666666666666)
/// ```
dynamic nanstd<T extends num>(NDArray<T> a, {int? axis}) {
  final v = nanvar(a, axis: axis);
  if (axis == null) {
    return math.sqrt(v as double);
  } else {
    return sqrt(v as NDArray<double>);
  }
}

/// Compute the minimum of elements in the array.
///
/// **Gotchas:**
/// - Returns a scalar if [axis] is null, or a new [NDArray] if [axis] is provided.
/// - When [axis] is provided, the result is currently always a [DType.float64] array.
dynamic min<T extends num>(NDArray<T> a, {int? axis}) {
  if (axis == null) {
    return a.data.reduce((value, element) => math.min(value, element));
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final result = NDArray<double>.fromList(
    List<double>.filled(
      newShape.isEmpty ? 1 : newShape.reduce((x, y) => x * y),
      double.infinity,
    ),
    newShape,
    DType.float64,
  );

  _reduceRecursive(
    a,
    result,
    List<int>.filled(a.shape.length, 0),
    List<int>.filled(newShape.length, 0),
    axis,
    0,
    (current, val) => math.min(current, val.toDouble()),
  );
  return result;
}

/// Compute the maximum of elements in the array.
///
/// **Gotchas:**
/// - Returns a scalar if [axis] is null, or a new [NDArray] if [axis] is provided.
/// - When [axis] is provided, the result is currently always a [DType.float64] array.
dynamic max<T extends num>(NDArray<T> a, {int? axis}) {
  if (axis == null) {
    return a.data.reduce((value, element) => math.max(value, element));
  }

  if (axis < 0 || axis >= a.shape.length) {
    throw ArgumentError('axis $axis out of bounds for shape ${a.shape}');
  }

  final newShape = List<int>.from(a.shape)..removeAt(axis);
  final result = NDArray<double>.fromList(
    List<double>.filled(
      newShape.isEmpty ? 1 : newShape.reduce((x, y) => x * y),
      -double.infinity,
    ),
    newShape,
    DType.float64,
  );

  _reduceRecursive(
    a,
    result,
    List<int>.filled(a.shape.length, 0),
    List<int>.filled(newShape.length, 0),
    axis,
    0,
    (current, val) => math.max(current, val.toDouble()),
  );
  return result;
}

/// Compute the inverse of a square 2D matrix using Gauss-Jordan elimination.
///
/// Returns a new array with the inverse matrix.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([4.0, 7.0, 2.0, 6.0], [2, 2], DType.float64);
/// final b = inv(a);
/// print(b.data); // [0.6, -0.7, -0.2, 0.4]
/// ```
///
/// **Gotchas:**
/// - Throws [ArgumentError] if the matrix is not square or not 2D.
/// - Throws [ArgumentError] if the matrix is singular (non-invertible).
/// - Pure Dart implementation; might be slow for very large matrices.
NDArray<double> inv(NDArray a, {NDArray<double>? out}) {
  if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
    throw ArgumentError('Matrix must be square and 2D (was ${a.shape})');
  }
  final n = a.shape[0];
  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;

  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for matrix inversion.',
      );
    }
  }

  NDArray src = a;
  if (!a.isContiguous) {
    src = NDArray.fromList(a.toList(), a.shape, a.dtype);
  }

  final ipiv = malloc<ffi.Int>(n);

  try {
    if (targetDType == DType.float32) {
      final NDArray<double> result;
      if (out != null) {
        result = out;
        if (src.dtype == DType.float32) {
          (result.data as Float32List).setRange(
            0,
            src.data.length,
            src.data as Float32List,
          );
        } else {
          for (var i = 0; i < src.data.length; i++) {
            result.data[i] = (src.data[i] as num).toDouble();
          }
        }
      } else if (!identical(a, src)) {
        result = src as NDArray<double>;
      } else {
        result = NDArray<double>.create(src.shape, DType.float32);
        if (src.dtype == DType.float32) {
          (result.data as Float32List).setRange(
            0,
            src.data.length,
            src.data as Float32List,
          );
        } else {
          for (var i = 0; i < src.data.length; i++) {
            result.data[i] = (src.data[i] as num).toDouble();
          }
        }
      }

      final info = LAPACKE_sgetrf(
        101,
        n,
        n,
        result.pointer.cast<ffi.Float>(),
        n,
        ipiv,
      );
      if (info < 0)
        throw ArgumentError('Illegal value in call to LAPACKE_sgetrf: $info');
      if (info > 0)
        throw ArgumentError('Matrix is singular and cannot be inverted');

      final infoTri = LAPACKE_sgetri(
        101,
        n,
        result.pointer.cast<ffi.Float>(),
        n,
        ipiv,
      );
      if (infoTri < 0)
        throw ArgumentError(
          'Illegal value in call to LAPACKE_sgetri: $infoTri',
        );
      return result;
    } else {
      final NDArray<double> result;
      if (out != null) {
        result = out;
        if (src.dtype == DType.float64) {
          (result.data as Float64List).setRange(
            0,
            src.data.length,
            src.data as Float64List,
          );
        } else {
          for (var i = 0; i < src.data.length; i++) {
            result.data[i] = (src.data[i] as num).toDouble();
          }
        }
      } else if (!identical(a, src)) {
        result = src as NDArray<double>;
      } else {
        result = NDArray<double>.create(src.shape, DType.float64);
        if (src.dtype == DType.float64) {
          (result.data as Float64List).setRange(
            0,
            src.data.length,
            src.data as Float64List,
          );
        } else {
          for (var i = 0; i < src.data.length; i++) {
            result.data[i] = (src.data[i] as num).toDouble();
          }
        }
      }

      final info = LAPACKE_dgetrf(
        101,
        n,
        n,
        result.pointer.cast<ffi.Double>(),
        n,
        ipiv,
      );
      if (info < 0)
        throw ArgumentError('Illegal value in call to LAPACKE_dgetrf: $info');
      if (info > 0)
        throw ArgumentError('Matrix is singular and cannot be inverted');

      final infoTri = LAPACKE_dgetri(
        101,
        n,
        result.pointer.cast<ffi.Double>(),
        n,
        ipiv,
      );
      if (infoTri < 0)
        throw ArgumentError(
          'Illegal value in call to LAPACKE_dgetri: $infoTri',
        );
      return result;
    }
  } finally {
    malloc.free(ipiv);
  }
}

/// Compute the determinant of a square 2D matrix using OpenBLAS.
///
/// Returns the determinant as a double.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, 2.0, 3.0, 4.0], [2, 2], DType.float64);
/// final d = det(a);
/// print(d); // -2.0
/// ```
///
/// **Gotchas:**
/// - Throws [ArgumentError] if the matrix is not square or not 2D.
double det(NDArray<double> a) {
  if (a.dtype != DType.float64 && a.dtype != DType.float32) {
    throw ArgumentError('det only supports Float64 and Float32 dtypes');
  }
  if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
    throw ArgumentError('Matrix must be square and 2D (was ${a.shape})');
  }
  final n = a.shape[0];

  if (a.dtype == DType.float64) {
    // Create a copy of the matrix because dgetrf overwrites it
    final aCopy = NDArray<double>.fromList(
      List<double>.from(a.data),
      a.shape,
      DType.float64,
    );

    final ipiv = malloc<ffi.Int>(n);

    try {
      final info = LAPACKE_dgetrf(
        101, // LAPACK_ROW_MAJOR
        n,
        n,
        aCopy.pointer.cast<ffi.Double>(),
        n,
        ipiv,
      );

      if (info < 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_dgetrf: $info');
      }

      // If info > 0, U(i,i) is exactly zero. The factorization has been completed,
      // but the factor U is exactly singular.
      // In this case, the determinant is 0.
      if (info > 0) {
        return 0.0;
      }

      var detValue = 1.0;
      var swaps = 0;

      final aCopyData = aCopy.data;
      for (var i = 0; i < n; i++) {
        detValue *= aCopyData[i * n + i];
        if (ipiv[i] != i + 1) {
          swaps++;
        }
      }

      if (swaps % 2 != 0) {
        detValue = -detValue;
      }

      return detValue;
    } finally {
      malloc.free(ipiv);
      aCopy.dispose();
    }
  } else {
    // Create a copy of the matrix because sgetrf overwrites it
    final aCopy = NDArray<double>.fromList(
      List<double>.from(a.data),
      a.shape,
      DType.float32,
    );

    final ipiv = malloc<ffi.Int>(n);

    try {
      final info = LAPACKE_sgetrf(
        101, // LAPACK_ROW_MAJOR
        n,
        n,
        aCopy.pointer.cast<ffi.Float>(),
        n,
        ipiv,
      );

      if (info < 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_sgetrf: $info');
      }

      if (info > 0) {
        return 0.0;
      }

      var detValue = 1.0;
      var swaps = 0;

      final aCopyData = aCopy.data;
      for (var i = 0; i < n; i++) {
        detValue *= aCopyData[i * n + i];
        if (ipiv[i] != i + 1) {
          swaps++;
        }
      }

      if (swaps % 2 != 0) {
        detValue = -detValue;
      }

      return detValue;
    } finally {
      malloc.free(ipiv);
      aCopy.dispose();
    }
  }
}

/// Solve a linear matrix equation, or system of linear scalar equations.
///
/// Computes the "exact" solution, `x`, of the linear equation `a * x = b`.
/// Natively offloads to LAPACK solvers (`dgesv`, `sgesv`, `zgesv`, `cgesv`) depending on precision.
///
/// **Preconditions:**
/// - Matrix [a] must be square (size $N \times N$) and 2-dimensional.
/// - Array [b] first dimension must exactly equal the first dimension of [a] ($N$).
/// - Matrix [a] must be non-singular (invertible).
///
/// **Throws:**
/// - [ArgumentError] if [a] is not square or not 2D.
/// - [ArgumentError] if [b] dimensions do not match [a].
/// - [ArgumentError] if [a] is singular and cannot be solved.
///
/// **Performance considerations:**
/// - Algorithmic complexity is $O(N^3)$ executed in native C-compiled space.
///
/// **Example:**
/// ```dart
/// final a = NDArray<double>.fromList([3.0, 1.0, 1.0, 2.0], [2, 2], DType.float64);
/// final b = NDArray<double>.fromList([9.0, 8.0], [2], DType.float64);
/// final x = solve(a, b);
/// print(x.toList()); // [2.0, 3.0]
/// ```
NDArray solve(NDArray a, NDArray b) {
  if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
    throw ArgumentError('Matrix a must be square and 2D (was ${a.shape})');
  }
  final n = a.shape[0];

  if (b.shape.isEmpty || b.shape[0] != n) {
    throw ArgumentError(
      'Dimensions of b must match dimensions of a (expected first dimension $n, was ${b.shape.isEmpty ? 0 : b.shape[0]})',
    );
  }

  final nrhs = b.shape.length > 1 ? b.shape[1] : 1;
  final ipiv = malloc<ffi.Int>(n);

  try {
    if (a.dtype == DType.float64 && b.dtype == DType.float64) {
      final aCopy = NDArray<double>.fromList(
        List<double>.from(a.data as List<double>),
        a.shape,
        DType.float64,
      );
      final bCopy = NDArray<double>.fromList(
        List<double>.from(b.data as List<double>),
        b.shape,
        DType.float64,
      );

      final info = LAPACKE_dgesv(
        101,
        n,
        nrhs,
        aCopy.pointer.cast<ffi.Double>(),
        n,
        ipiv,
        bCopy.pointer.cast<ffi.Double>(),
        nrhs,
      );
      if (info < 0)
        throw ArgumentError('Illegal value in call to LAPACKE_dgesv: $info');
      if (info > 0)
        throw ArgumentError('Matrix is singular and cannot be solved');
      aCopy.dispose();
      return bCopy;
    } else if (a.dtype == DType.float32 && b.dtype == DType.float32) {
      final aCopy = NDArray<double>.fromList(
        List<double>.from(a.data as List<double>),
        a.shape,
        DType.float32,
      );
      final bCopy = NDArray<double>.fromList(
        List<double>.from(b.data as List<double>),
        b.shape,
        DType.float32,
      );

      final info = LAPACKE_sgesv(
        101,
        n,
        nrhs,
        aCopy.pointer.cast<ffi.Float>(),
        n,
        ipiv,
        bCopy.pointer.cast<ffi.Float>(),
        nrhs,
      );
      if (info < 0)
        throw ArgumentError('Illegal value in call to LAPACKE_sgesv: $info');
      if (info > 0)
        throw ArgumentError('Matrix is singular and cannot be solved');
      aCopy.dispose();
      return bCopy;
    } else if (a.dtype == DType.complex128 && b.dtype == DType.complex128) {
      final aList = (a.data as ComplexList).backingList;
      final bList = (b.data as ComplexList).backingList;

      final aCopy = NDArray<Complex>.create(a.shape, DType.complex128);
      (aCopy.data as ComplexList).backingList.setRange(0, aList.length, aList);

      final bCopy = NDArray<Complex>.create(b.shape, DType.complex128);
      (bCopy.data as ComplexList).backingList.setRange(0, bList.length, bList);

      final info = LAPACKE_zgesv(
        101,
        n,
        nrhs,
        aCopy.pointer.cast<ffi.Double>(),
        n,
        ipiv,
        bCopy.pointer.cast<ffi.Double>(),
        nrhs,
      );
      if (info < 0)
        throw ArgumentError('Illegal value in call to LAPACKE_zgesv: $info');
      if (info > 0)
        throw ArgumentError('Matrix is singular and cannot be solved');
      aCopy.dispose();
      return bCopy;
    } else if (a.dtype == DType.complex64 && b.dtype == DType.complex64) {
      final aList = (a.data as ComplexList).backingList;
      final bList = (b.data as ComplexList).backingList;

      final aCopy = NDArray<Complex>.create(a.shape, DType.complex64);
      (aCopy.data as ComplexList).backingList.setRange(0, aList.length, aList);

      final bCopy = NDArray<Complex>.create(b.shape, DType.complex64);
      (bCopy.data as ComplexList).backingList.setRange(0, bList.length, bList);

      final info = LAPACKE_cgesv(
        101,
        n,
        nrhs,
        aCopy.pointer.cast<ffi.Float>(),
        n,
        ipiv,
        bCopy.pointer.cast<ffi.Float>(),
        nrhs,
      );
      if (info < 0)
        throw ArgumentError('Illegal value in call to LAPACKE_cgesv: $info');
      if (info > 0)
        throw ArgumentError('Matrix is singular and cannot be solved');
      aCopy.dispose();
      return bCopy;
    } else {
      // Fallback: convert to Float64
      final aDouble = NDArray<double>.create(a.shape, DType.float64);
      for (var i = 0; i < a.data.length; i++) {
        aDouble.data[i] = (a.data[i] as num).toDouble();
      }

      final bDouble = NDArray<double>.create(b.shape, DType.float64);
      for (var i = 0; i < b.data.length; i++) {
        bDouble.data[i] = (b.data[i] as num).toDouble();
      }

      final info = LAPACKE_dgesv(
        101,
        n,
        nrhs,
        aDouble.pointer.cast<ffi.Double>(),
        n,
        ipiv,
        bDouble.pointer.cast<ffi.Double>(),
        nrhs,
      );
      if (info < 0)
        throw ArgumentError('Illegal value in call to LAPACKE_dgesv: $info');
      if (info > 0)
        throw ArgumentError('Matrix is singular and cannot be solved');

      aDouble.dispose();
      return bDouble;
    }
  } finally {
    malloc.free(ipiv);
  }
}

/// Compute the eigenvalues and right eigenvectors of a square array.
///
/// Returns a Map with keys 'eigenvalues' and 'eigenvectors'.
/// Both are returned as NDArray<Complex> because they can be complex
/// even for real matrices.
Map<String, NDArray<Complex>> eig(NDArray a) {
  if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
    throw ArgumentError('Matrix must be square and 2D (was ${a.shape})');
  }
  final n = a.shape[0];

  final jobvl = 'N'.codeUnitAt(0);
  final jobvr = 'V'.codeUnitAt(0);

  if (a.dtype == DType.complex128) {
    final aComplex = NDArray<Complex>.create(a.shape, DType.complex128);
    final backing = (a.data as ComplexList).backingList;
    (aComplex.data as ComplexList).backingList.setRange(
      0,
      backing.length,
      backing,
    );

    final w = NDArray<Complex>.create([n], DType.complex128);
    final vr = NDArray<Complex>.create([n, n], DType.complex128);

    final info = LAPACKE_zgeev(
      101, // LAPACK_ROW_MAJOR
      jobvl,
      jobvr,
      n,
      aComplex.pointer.cast<ffi.Double>(),
      n,
      w.pointer.cast<ffi.Double>(),
      ffi.nullptr.cast<ffi.Double>(),
      n,
      vr.pointer.cast<ffi.Double>(),
      n,
    );

    if (info < 0)
      throw ArgumentError('Illegal value in call to LAPACKE_zgeev: $info');
    if (info > 0)
      throw ArgumentError(
        'The QR algorithm failed to compute all eigenvalues.',
      );

    aComplex.dispose();
    return {'eigenvalues': w, 'eigenvectors': vr};
  } else if (a.dtype == DType.complex64) {
    final aComplex = NDArray<Complex>.create(a.shape, DType.complex64);
    final backing = (a.data as ComplexList).backingList;
    (aComplex.data as ComplexList).backingList.setRange(
      0,
      backing.length,
      backing,
    );

    final w = NDArray<Complex>.create([n], DType.complex64);
    final vr = NDArray<Complex>.create([n, n], DType.complex64);

    final info = LAPACKE_cgeev(
      101, // LAPACK_ROW_MAJOR
      jobvl,
      jobvr,
      n,
      aComplex.pointer.cast<ffi.Float>(),
      n,
      w.pointer.cast<ffi.Float>(),
      ffi.nullptr.cast<ffi.Float>(),
      n,
      vr.pointer.cast<ffi.Float>(),
      n,
    );

    if (info < 0)
      throw ArgumentError('Illegal value in call to LAPACKE_cgeev: $info');
    if (info > 0)
      throw ArgumentError(
        'The QR algorithm failed to compute all eigenvalues.',
      );

    aComplex.dispose();
    return {'eigenvalues': w, 'eigenvectors': vr};
  } else if (a.dtype == DType.float64 ||
      a.dtype == DType.int32 ||
      a.dtype == DType.int64) {
    final aReal = NDArray<double>.create(a.shape, DType.float64);
    for (var i = 0; i < a.data.length; i++) {
      aReal.data[i] = (a.data[i] as num).toDouble();
    }

    final wr = NDArray<double>.zeros([n], DType.float64);
    final wi = NDArray<double>.zeros([n], DType.float64);
    final vrReal = NDArray<double>.create([n, n], DType.float64);

    final info = LAPACKE_dgeev(
      101,
      jobvl,
      jobvr,
      n,
      aReal.pointer.cast<ffi.Double>(),
      n,
      wr.pointer.cast<ffi.Double>(),
      wi.pointer.cast<ffi.Double>(),
      ffi.nullptr.cast<ffi.Double>(),
      n,
      vrReal.pointer.cast<ffi.Double>(),
      n,
    );

    if (info < 0)
      throw ArgumentError('Illegal value in call to LAPACKE_dgeev: $info');
    if (info > 0)
      throw ArgumentError(
        'The QR algorithm failed to compute all eigenvalues.',
      );

    final w = NDArray<Complex>.create([n], DType.complex128);
    final vr = NDArray<Complex>.create([n, n], DType.complex128);

    for (var j = 0; j < n; j++) {
      w.data[j] = Complex(wr.data[j], wi.data[j]);
    }

    var j = 0;
    while (j < n) {
      if (wi.data[j] == 0.0) {
        for (var r = 0; r < n; r++) {
          vr.data[r * n + j] = Complex(vrReal.data[r * n + j], 0.0);
        }
        j++;
      } else {
        for (var r = 0; r < n; r++) {
          final realPart = vrReal.data[r * n + j];
          final imagPart = vrReal.data[r * n + j + 1];
          vr.data[r * n + j] = Complex(realPart, imagPart);
          vr.data[r * n + j + 1] = Complex(realPart, -imagPart);
        }
        j += 2;
      }
    }

    aReal.dispose();
    wr.dispose();
    wi.dispose();
    vrReal.dispose();
    return {'eigenvalues': w, 'eigenvectors': vr};
  } else if (a.dtype == DType.float32) {
    final aReal = NDArray<double>.create(a.shape, DType.float32);
    for (var i = 0; i < a.data.length; i++) {
      aReal.data[i] = (a.data[i] as num).toDouble();
    }

    final wr = NDArray<double>.zeros([n], DType.float32);
    final wi = NDArray<double>.zeros([n], DType.float32);
    final vrReal = NDArray<double>.create([n, n], DType.float32);

    final info = LAPACKE_sgeev(
      101,
      jobvl,
      jobvr,
      n,
      aReal.pointer.cast<ffi.Float>(),
      n,
      wr.pointer.cast<ffi.Float>(),
      wi.pointer.cast<ffi.Float>(),
      ffi.nullptr.cast<ffi.Float>(),
      n,
      vrReal.pointer.cast<ffi.Float>(),
      n,
    );

    if (info < 0)
      throw ArgumentError('Illegal value in call to LAPACKE_sgeev: $info');
    if (info > 0)
      throw ArgumentError(
        'The QR algorithm failed to compute all eigenvalues.',
      );

    final w = NDArray<Complex>.create([n], DType.complex64);
    final vr = NDArray<Complex>.create([n, n], DType.complex64);

    for (var j = 0; j < n; j++) {
      w.data[j] = Complex(wr.data[j], wi.data[j]);
    }

    var j = 0;
    while (j < n) {
      if (wi.data[j] == 0.0) {
        for (var r = 0; r < n; r++) {
          vr.data[r * n + j] = Complex(vrReal.data[r * n + j], 0.0);
        }
        j++;
      } else {
        for (var r = 0; r < n; r++) {
          final realPart = vrReal.data[r * n + j];
          final imagPart = vrReal.data[r * n + j + 1];
          vr.data[r * n + j] = Complex(realPart, imagPart);
          vr.data[r * n + j + 1] = Complex(realPart, -imagPart);
        }
        j += 2;
      }
    }

    aReal.dispose();
    wr.dispose();
    wi.dispose();
    vrReal.dispose();
    return {'eigenvalues': w, 'eigenvectors': vr};
  } else {
    throw UnimplementedError('Type ${a.dtype} not supported for eig');
  }
}

/// Recursive helper to traverse and reduce an array along an axis.
void _reduceRecursive<S extends Object, D extends Object>(
  NDArray<S> src,
  NDArray<D> dest,
  List<int> currentPos,
  List<int> destPos,
  int targetAxis,
  int currentDim,
  D Function(D current, S value) op,
) {
  if (currentDim == src.shape.length) {
    // Calculate flat index for src
    var srcOffset = 0;
    for (var i = 0; i < src.shape.length; i++) {
      srcOffset += currentPos[i] * src.strides[i];
    }

    // Calculate flat index for dest
    var destOffset = 0;
    for (var i = 0; i < dest.shape.length; i++) {
      destOffset += destPos[i] * dest.strides[i];
    }

    dest.data[destOffset] = op(dest.data[destOffset], src.data[srcOffset]);
    return;
  }

  for (var i = 0; i < src.shape[currentDim]; i++) {
    currentPos[currentDim] = i;
    if (currentDim < targetAxis) {
      destPos[currentDim] = i;
    } else if (currentDim > targetAxis) {
      destPos[currentDim - 1] = i;
    }
    _reduceRecursive(
      src,
      dest,
      currentPos,
      destPos,
      targetAxis,
      currentDim + 1,
      op,
    );
  }
}

/// Concatenates a list of arrays along a specified axis.
NDArray<T> concatenate<T extends Object>(
  List<NDArray<T>> arrays, {
  int axis = 0,
}) {
  if (arrays.isEmpty) {
    throw ArgumentError('List of arrays must not be empty');
  }

  final first = arrays.first;
  final rank = first.shape.length;
  final dtype = first.dtype;

  if (axis < 0 || axis >= rank) {
    throw RangeError.index(axis, first.shape, 'axis out of range');
  }

  for (var i = 1; i < arrays.length; i++) {
    final arr = arrays[i];
    if (arr.dtype != dtype) {
      throw ArgumentError('All arrays must have the same DType');
    }
    if (arr.shape.length != rank) {
      throw ArgumentError('All arrays must have the same rank');
    }
    for (var j = 0; j < rank; j++) {
      if (j != axis && arr.shape[j] != first.shape[j]) {
        throw ArgumentError('Shapes must match except in dimension $axis');
      }
    }
  }

  final targetShape = List<int>.from(first.shape);
  var totalAxisSize = 0;
  for (final arr in arrays) {
    totalAxisSize += arr.shape[axis];
  }
  targetShape[axis] = totalAxisSize;

  final result = NDArray<T>.create(targetShape, dtype);

  var allContiguous = true;
  for (final arr in arrays) {
    if (!arr.isContiguous) {
      allContiguous = false;
      break;
    }
  }

  if (allContiguous && axis == 0) {
    var destOffset = 0;
    for (final arr in arrays) {
      final size = arr.shape.isEmpty ? 1 : arr.shape.reduce((a, b) => a * b);
      _copyContiguousFlat(arr, result, destOffset, size);
      destOffset += size;
    }
    return result;
  }

  var axisOffset = 0;
  for (final arr in arrays) {
    _copyConcatenateRecursive(
      arr,
      result,
      axis,
      axisOffset,
      List<int>.filled(rank, 0),
      0,
    );
    axisOffset += arr.shape[axis];
  }

  return result;
}

void _copyContiguousFlat(NDArray src, NDArray dest, int destOffset, int size) {
  final dtype = src.dtype;
  if (dtype == DType.float64) {
    final srcList = src.pointer.cast<ffi.Double>().asTypedList(size);
    final destList = dest.pointer.cast<ffi.Double>().asTypedList(
      destOffset + size,
    );
    destList.setRange(destOffset, destOffset + size, srcList);
  } else if (dtype == DType.float32) {
    final srcList = src.pointer.cast<ffi.Float>().asTypedList(size);
    final destList = dest.pointer.cast<ffi.Float>().asTypedList(
      destOffset + size,
    );
    destList.setRange(destOffset, destOffset + size, srcList);
  } else if (dtype == DType.int32) {
    final srcList = src.pointer.cast<ffi.Int32>().asTypedList(size);
    final destList = dest.pointer.cast<ffi.Int32>().asTypedList(
      destOffset + size,
    );
    destList.setRange(destOffset, destOffset + size, srcList);
  } else if (dtype == DType.int64) {
    final srcList = src.pointer.cast<ffi.Int64>().asTypedList(size);
    final destList = dest.pointer.cast<ffi.Int64>().asTypedList(
      destOffset + size,
    );
    destList.setRange(destOffset, destOffset + size, srcList);
  } else if (dtype == DType.complex128) {
    final srcList = src.pointer.cast<ffi.Double>().asTypedList(size * 2);
    final destList = dest.pointer.cast<ffi.Double>().asTypedList(
      (destOffset + size) * 2,
    );
    destList.setRange(destOffset * 2, (destOffset + size) * 2, srcList);
  } else if (dtype == DType.complex64) {
    final srcList = src.pointer.cast<ffi.Float>().asTypedList(size * 2);
    final destList = dest.pointer.cast<ffi.Float>().asTypedList(
      (destOffset + size) * 2,
    );
    destList.setRange(destOffset * 2, (destOffset + size) * 2, srcList);
  } else if (dtype == DType.boolean) {
    final srcList = src.pointer.cast<ffi.Uint8>().asTypedList(size);
    final destList = dest.pointer.cast<ffi.Uint8>().asTypedList(
      destOffset + size,
    );
    destList.setRange(destOffset, destOffset + size, srcList);
  } else {
    throw UnimplementedError('Type $dtype not supported for fast concatenate');
  }
}

void _copyConcatenateRecursive<T extends Object>(
  NDArray<T> src,
  NDArray<T> dest,
  int axis,
  int axisOffset,
  List<int> currentIndices,
  int currentDim,
) {
  if (currentDim == src.shape.length) {
    final destIndices = List<int>.from(currentIndices);
    destIndices[axis] += axisOffset;
    dest[destIndices] = src[currentIndices];
    return;
  }

  for (var i = 0; i < src.shape[currentDim]; i++) {
    currentIndices[currentDim] = i;
    _copyConcatenateRecursive(
      src,
      dest,
      axis,
      axisOffset,
      currentIndices,
      currentDim + 1,
    );
  }
}

/// Stacks arrays in sequence vertically (row wise).
NDArray<T> vstack<T extends Object>(List<NDArray<T>> arrays) {
  return concatenate(arrays, axis: 0);
}

/// Stacks arrays in sequence horizontally (column wise).
NDArray<T> hstack<T extends Object>(List<NDArray<T>> arrays) {
  return concatenate(arrays, axis: 1);
}

/// Constructs a new array by repeating [array] the number of times given by [reps].
///
/// This function corresponds to NumPy's `tile` function. It allocates new memory and
/// copies elements.
///
/// If [reps] has a length shorter than `array.shape.length`, `1`s are prepended to [reps].
/// If `array.shape.length` is shorter than [reps], `1`s are prepended to the array's shape
/// (effectively expanding it with front dimensions of size 1).
///
/// **Preconditions:**
/// - [reps] can be an [int] or a [List<int>]. If a list, all elements must be non-negative.
///
/// **Throws:**
/// - [ArgumentError] if [reps] contains negative integers or is an invalid type.
///
/// **Example:**
/// {@example /example/shape_examples.dart lang=dart}
NDArray<T> tile<T>(NDArray<T> array, dynamic reps) {
  List<int> repsList;
  if (reps is int) {
    repsList = [reps];
  } else if (reps is List<int>) {
    repsList = List<int>.from(reps);
  } else {
    throw ArgumentError('reps must be an int or a List<int>');
  }

  for (final rep in repsList) {
    if (rep < 0) {
      throw ArgumentError('reps values must be non-negative (was $rep)');
    }
  }

  final srcShape = array.shape;
  final maxLen = srcShape.length > repsList.length
      ? srcShape.length
      : repsList.length;

  final promotedReps = List<int>.filled(maxLen, 1);
  for (var i = 0; i < repsList.length; i++) {
    promotedReps[maxLen - 1 - i] = repsList[repsList.length - 1 - i];
  }

  final promotedSrcShape = List<int>.filled(maxLen, 1);
  for (var i = 0; i < srcShape.length; i++) {
    promotedSrcShape[maxLen - 1 - i] = srcShape[srcShape.length - 1 - i];
  }

  final targetShape = List<int>.filled(maxLen, 0);
  for (var i = 0; i < maxLen; i++) {
    targetShape[i] = promotedSrcShape[i] * promotedReps[i];
  }

  final result = NDArray<T>.create(targetShape, array.dtype);

  _tileCopyRecursive<T>(array, result, List<int>.filled(maxLen, 0), 0);

  return result;
}

void _tileCopyRecursive<T>(
  NDArray<T> src,
  NDArray<T> dest,
  List<int> destPos,
  int currentDim,
) {
  if (currentDim == dest.shape.length) {
    final diff = dest.shape.length - src.shape.length;
    final srcPos = List<int>.filled(src.shape.length, 0);
    for (var i = 0; i < src.shape.length; i++) {
      srcPos[i] = destPos[i + diff] % src.shape[i];
    }
    dest[destPos] = src[srcPos];
    return;
  }

  for (var i = 0; i < dest.shape[currentDim]; i++) {
    destPos[currentDim] = i;
    _tileCopyRecursive(src, dest, destPos, currentDim + 1);
  }
}

/// Repeats elements of [array] a specified number of times.
///
/// This function corresponds to NumPy's `repeat` function. It allocates new memory and
/// copies elements.
///
/// If [axis] is omitted/null, the array is flattened (via `ravel()`) to a 1D array first,
/// and the elements are repeated along that flat 1D array.
///
/// **Preconditions:**
/// - [repeats] can be an [int] or a [List<int>]. If an `int`, all elements along [axis]
///   are repeated that number of times. If a list, its length must exactly match the size
///   of the dimension along [axis].
/// - All values in [repeats] must be non-negative.
/// - If provided, [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [RangeError] if [axis] is out of range.
/// - [ArgumentError] if [repeats] has a mismatched size or contains negative integers.
///
/// **Example:**
/// {@example /example/shape_examples.dart lang=dart}
NDArray<T> repeat<T>(NDArray<T> array, dynamic repeats, {int? axis}) {
  NDArray<T> srcArray = array;
  int targetAxis;

  if (axis == null) {
    srcArray = array.ravel();
    targetAxis = 0;
  } else {
    final rank = array.shape.length;
    if (axis < -rank || axis >= rank) {
      throw RangeError.range(axis, -rank, rank - 1, 'axis');
    }
    targetAxis = axis < 0 ? rank + axis : axis;
  }

  List<int> repList;
  if (repeats is int) {
    if (repeats < 0)
      throw ArgumentError('repeats must be non-negative (was $repeats)');
    repList = List<int>.filled(srcArray.shape[targetAxis], repeats);
  } else if (repeats is List<int>) {
    for (final r in repeats) {
      if (r < 0)
        throw ArgumentError('repeats values must be non-negative (was $r)');
    }
    if (repeats.length != srcArray.shape[targetAxis]) {
      throw ArgumentError(
        'repeats list length (${repeats.length}) must match dimension along axis $targetAxis (${srcArray.shape[targetAxis]})',
      );
    }
    repList = List<int>.from(repeats);
  } else {
    throw ArgumentError('repeats must be an int or a List<int>');
  }

  final prefixSums = List<int>.filled(srcArray.shape[targetAxis] + 1, 0);
  for (var i = 0; i < repList.length; i++) {
    prefixSums[i + 1] = prefixSums[i] + repList[i];
  }

  final targetShape = List<int>.from(srcArray.shape);
  targetShape[targetAxis] = prefixSums.last;

  final result = NDArray<T>.create(targetShape, srcArray.dtype);

  _repeatCopyRecursive<T>(
    srcArray,
    result,
    List<int>.filled(srcArray.shape.length, 0),
    0,
    targetAxis,
    prefixSums,
  );

  return result;
}

void _repeatCopyRecursive<T>(
  NDArray<T> src,
  NDArray<T> dest,
  List<int> srcPos,
  int currentDim,
  int targetAxis,
  List<int> prefixSums,
) {
  if (currentDim == src.shape.length) {
    final startDestIdx = prefixSums[srcPos[targetAxis]];
    final endDestIdx = prefixSums[srcPos[targetAxis] + 1];

    if (startDestIdx == endDestIdx) return;

    final destPos = List<int>.from(srcPos);
    for (var dIdx = startDestIdx; dIdx < endDestIdx; dIdx++) {
      destPos[targetAxis] = dIdx;
      dest[destPos] = src[srcPos];
    }
    return;
  }

  for (var i = 0; i < src.shape[currentDim]; i++) {
    srcPos[currentDim] = i;
    _repeatCopyRecursive(
      src,
      dest,
      srcPos,
      currentDim + 1,
      targetAxis,
      prefixSums,
    );
  }
}

void _unaryOp<Ta, Tr>(
  List<Tr> result,
  List<Ta> a,
  List<int> shape,
  List<int> stridesA,
  List<int> stridesResult,
  int dim,
  int offsetA,
  int offsetResult,
  Tr Function(Ta) op,
) {
  if (dim == shape.length) {
    result[offsetResult] = op(a[offsetA]);
    return;
  }

  for (var i = 0; i < shape[dim]; i++) {
    _unaryOp<Ta, Tr>(
      result,
      a,
      shape,
      stridesA,
      stridesResult,
      dim + 1,
      offsetA + i * stridesA[dim],
      offsetResult + i * stridesResult[dim],
      op,
    );
  }
}

/// Compute the element-wise tangent of the array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray tan(NDArray a) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for tan');
  }
  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  final result = NDArray.create(a.shape, targetDType);

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_tan_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_tan_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  }
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.tan(x.toDouble()),
    );
  } else {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => math.tan(x),
    );
  }
  return result;
}

/// Compute the element-wise arc tangent of [y] / [x] with full broadcasting support.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray atan2(NDArray y, NDArray x) {
  if (y.dtype == DType.complex128 ||
      y.dtype == DType.complex64 ||
      x.dtype == DType.complex128 ||
      x.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for atan2');
  }
  final broadcastResult = broadcast(y, x);
  final shape = broadcastResult.shape;
  final targetDType = (y.dtype == DType.float32 && x.dtype == DType.float32)
      ? DType.float32
      : DType.float64;

  final result = NDArray.create(shape, targetDType);
  final resultStrides = NDArray.computeCStrides(shape);
  final rData = result.data as List<double>;

  if (y.dtype == DType.float64 || y.dtype == DType.float32) {
    final yData = y.data as List<double>;
    if (x.dtype == DType.float64 || x.dtype == DType.float32) {
      _elementWiseOp<double, double, double>(
        rData,
        yData,
        x.data as List<double>,
        shape,
        broadcastResult.stridesA,
        broadcastResult.stridesB,
        resultStrides,
        0,
        0,
        0,
        0,
        (a, b) => math.atan2(a, b),
      );
    } else {
      _elementWiseOp<double, int, double>(
        rData,
        yData,
        x.data as List<int>,
        shape,
        broadcastResult.stridesA,
        broadcastResult.stridesB,
        resultStrides,
        0,
        0,
        0,
        0,
        (a, b) => math.atan2(a, b.toDouble()),
      );
    }
  } else {
    final yData = y.data as List<int>;
    if (x.dtype == DType.float64 || x.dtype == DType.float32) {
      _elementWiseOp<int, double, double>(
        rData,
        yData,
        x.data as List<double>,
        shape,
        broadcastResult.stridesA,
        broadcastResult.stridesB,
        resultStrides,
        0,
        0,
        0,
        0,
        (a, b) => math.atan2(a.toDouble(), b),
      );
    } else {
      _elementWiseOp<int, int, double>(
        rData,
        yData,
        x.data as List<int>,
        shape,
        broadcastResult.stridesA,
        broadcastResult.stridesB,
        resultStrides,
        0,
        0,
        0,
        0,
        (a, b) => math.atan2(a.toDouble(), b.toDouble()),
      );
    }
  }
  return result;
}

/// Compute the element-wise hyperbolic sine of the array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray sinh(NDArray a) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for sinh');
  }
  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  final result = NDArray.create(a.shape, targetDType);
  final resultStrides = NDArray.computeCStrides(a.shape);

  final op = (double x) => (math.exp(x) - math.exp(-x)) / 2.0;
  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => op(x.toDouble()),
    );
  } else {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      op,
    );
  }
  return result;
}

/// Compute the element-wise hyperbolic cosine of the array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray cosh(NDArray a) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for cosh');
  }
  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  final result = NDArray.create(a.shape, targetDType);
  final resultStrides = NDArray.computeCStrides(a.shape);

  final op = (double x) => (math.exp(x) + math.exp(-x)) / 2.0;
  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => op(x.toDouble()),
    );
  } else {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      op,
    );
  }
  return result;
}

/// Compute the element-wise hyperbolic tangent of the array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray tanh(NDArray a) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for tanh');
  }
  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  final result = NDArray.create(a.shape, targetDType);
  final resultStrides = NDArray.computeCStrides(a.shape);

  final op = (double x) {
    final e2x = math.exp(2.0 * x);
    return (e2x - 1.0) / (e2x + 1.0);
  };
  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, double>(
      result.data as List<double>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => op(x.toDouble()),
    );
  } else {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      op,
    );
  }
  return result;
}

/// Compute the element-wise absolute value of the array.
///
/// For complex numbers, returns the magnitude as a real array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray abs(NDArray a) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    final targetDType = a.dtype == DType.complex64
        ? DType.float32
        : DType.float64;
    final result = NDArray.create(a.shape, targetDType);
    final resultStrides = NDArray.computeCStrides(a.shape);

    _unaryOp<Complex, double>(
      result.data as List<double>,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (c) => math.sqrt(c.real * c.real + c.imag * c.imag),
    );
    return result;
  }

  final result = NDArray.create(a.shape, a.dtype);
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, int>(
      result.data as List<int>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.abs(),
    );
  } else {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.abs(),
    );
  }
  return result;
}

/// Compute the element-wise ceiling of the array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray ceil(NDArray a) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for ceil');
  }
  final result = NDArray.create(a.shape, a.dtype);

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_ceil_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_ceil_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  }
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, int>(
      result.data as List<int>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x,
    );
  } else {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.ceilToDouble(),
    );
  }
  return result;
}

/// Compute the element-wise floor of the array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray floor(NDArray a) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for floor');
  }
  final result = NDArray.create(a.shape, a.dtype);

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_floor_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_floor_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  }
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, int>(
      result.data as List<int>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x,
    );
  } else {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.floorToDouble(),
    );
  }
  return result;
}

/// Compute the element-wise rounding of the array.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray round(NDArray a) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for round');
  }
  final result = NDArray.create(a.shape, a.dtype);

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_round_double(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    } else if (a.dtype == DType.float32) {
      v_round_float(a.pointer.cast(), result.pointer.cast(), a.data.length);
      return result;
    }
  }
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, int>(
      result.data as List<int>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x,
    );
  } else {
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.roundToDouble(),
    );
  }
  return result;
}

/// Returns the real part of a complex array element-wise.
///
/// If the input array [a] is already real (integer or float), returns a zero-copy
/// view of the array [a].
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
/// - If provided, the output recycler [out] must match the expected target shape and float DType.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [ArgumentError] if [out] is provided but has a shape or DType mismatch.
///
/// **Example:**
/// ```dart
/// final a = NDArray<Complex>.create([2], DType.complex128);
/// a.data[0] = Complex(3.0, 4.0);
/// a.data[1] = Complex(-1.0, 0.0);
/// final r = real(a); // [3.0, -1.0] (DType.float64)
/// ```
NDArray real(NDArray a, {NDArray? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute real() on a disposed array.');
  }

  final targetDType = a.dtype == DType.complex64 ? DType.float32 : DType.float64;

  if (a.dtype != DType.complex128 && a.dtype != DType.complex64) {
    if (out != null) {
      if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
        throw ArgumentError(
          'Recycler out must match shape ${a.shape} and DType ${a.dtype}',
        );
      }
      // Copy elements into out
      final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
      out.data.setRange(0, size, a.toList());
      return out;
    }
    return NDArray.view(a, a.shape, a.strides); // Zero-copy view for already real arrays!
  }

  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Recycler out must match shape ${a.shape} and DType $targetDType',
      );
    }
  }

  final result = out ?? NDArray<double>.zeros(a.shape, targetDType);
  final resultStrides = NDArray.computeCStrides(a.shape);

  _unaryOp<Complex, double>(
    result.data as List<double>,
    a.data as List<Complex>,
    a.shape,
    a.strides,
    resultStrides,
    0,
    0,
    0,
    (x) => x.real,
  );

  return result;
}

/// Returns the imaginary part of a complex array element-wise.
///
/// If the input array [a] is already real, returns a zero-filled array of matching shape
/// and target float DType.
///
/// **Preconditions:**
/// - The input array [a] must not be disposed.
/// - If provided, the output recycler [out] must match the expected target shape and float DType.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [ArgumentError] if [out] is provided but has a shape or DType mismatch.
///
/// **Example:**
/// ```dart
/// final a = NDArray<Complex>.create([2], DType.complex128);
/// a.data[0] = Complex(3.0, 4.0);
/// a.data[1] = Complex(-1.0, 0.0);
/// final im = imag(a); // [4.0, 0.0] (DType.float64)
/// ```
NDArray imag(NDArray a, {NDArray? out}) {
  if (a.isDisposed) {
    throw StateError('Cannot execute imag() on a disposed array.');
  }

  final targetDType = a.dtype == DType.complex64 ? DType.float32 : DType.float64;

  if (a.dtype != DType.complex128 && a.dtype != DType.complex64) {
    if (out != null) {
      if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
        throw ArgumentError(
          'Recycler out must match shape ${a.shape} and DType $targetDType',
        );
      }
      out.data.fillRange(0, out.data.length, 0.0);
      return out;
    }
    return NDArray<double>.zeros(a.shape, targetDType);
  }

  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Recycler out must match shape ${a.shape} and DType $targetDType',
      );
    }
  }

  final result = out ?? NDArray<double>.zeros(a.shape, targetDType);
  final resultStrides = NDArray.computeCStrides(a.shape);

  _unaryOp<Complex, double>(
    result.data as List<double>,
    a.data as List<Complex>,
    a.shape,
    a.strides,
    resultStrides,
    0,
    0,
    0,
    (x) => x.imag,
  );

  return result;
}

/// Converts angles from degrees to radians element-wise.
///
/// **Preconditions:**
/// - Input array [a] must not be disposed.
/// - Input array [a] must not contain complex numbers.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [UnsupportedError] if the array has a complex data type.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([180.0, 90.0, 45.0], [3], DType.float64);
/// final r = deg2rad(a); // [pi, pi / 2.0, pi / 4.0]
/// ```
NDArray deg2rad(NDArray a) {
  if (a.isDisposed) {
    throw StateError('Cannot execute deg2rad on a disposed array.');
  }
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for deg2rad');
  }
  // pi / 180.0 = 0.017453292519943295
  final factorDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  final factor = NDArray.fromList([0.017453292519943295], [1], factorDType);
  return multiply(a, factor);
}

/// Converts angles from radians to degrees element-wise.
///
/// **Preconditions:**
/// - Input array [a] must not be disposed.
/// - Input array [a] must not contain complex numbers.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
/// - [UnsupportedError] if the array has a complex data type.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([math.pi, math.pi / 2.0], [2], DType.float64);
/// final d = rad2deg(a); // [180.0, 90.0]
/// ```
NDArray rad2deg(NDArray a) {
  if (a.isDisposed) {
    throw StateError('Cannot execute rad2deg on a disposed array.');
  }
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for rad2deg');
  }
  // 180.0 / pi = 57.29577951308232
  final factorDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;
  final factor = NDArray.fromList([57.29577951308232], [1], factorDType);
  return multiply(a, factor);
}

/// Returns an element-wise boolean mask indicating which elements of the array are NaN (Not-a-Number).
///
/// **Preconditions:**
/// - The array must not be disposed.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.nan, 3.0], [3], DType.float64);
/// final mask = isnan(a); // [false, true, false]
/// ```
NDArray<bool> isnan(NDArray a) {
  if (a.isDisposed) {
    throw StateError('Cannot execute isnan on a disposed array.');
  }
  final result = NDArray<bool>.create(a.shape, DType.boolean);
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    _unaryOp<Complex, bool>(
      result.data,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.real.isNaN || x.imag.isNaN,
    );
  } else if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, bool>(
      result.data,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => false,
    );
  } else {
    _unaryOp<double, bool>(
      result.data,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.isNaN,
    );
  }
  return result;
}

/// Returns an element-wise boolean mask indicating which elements of the array are positive or negative infinity.
///
/// **Preconditions:**
/// - The array must not be disposed.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.infinity, 3.0], [3], DType.float64);
/// final mask = isinf(a); // [false, true, false]
/// ```
NDArray<bool> isinf(NDArray a) {
  if (a.isDisposed) {
    throw StateError('Cannot execute isinf on a disposed array.');
  }
  final result = NDArray<bool>.create(a.shape, DType.boolean);
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    _unaryOp<Complex, bool>(
      result.data,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.real.isInfinite || x.imag.isInfinite,
    );
  } else if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, bool>(
      result.data,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => false,
    );
  } else {
    _unaryOp<double, bool>(
      result.data,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.isInfinite,
    );
  }
  return result;
}

/// Returns an element-wise boolean mask indicating which elements of the array are finite (neither NaN nor infinite).
///
/// **Preconditions:**
/// - The array must not be disposed.
///
/// **Throws:**
/// - [StateError] if the array has been disposed.
///
/// **Example:**
/// ```dart
/// final a = NDArray.fromList([1.0, double.nan, double.infinity], [3], DType.float64);
/// final mask = isfinite(a); // [true, false, false]
/// ```
NDArray<bool> isfinite(NDArray a) {
  if (a.isDisposed) {
    throw StateError('Cannot execute isfinite on a disposed array.');
  }
  final result = NDArray<bool>.create(a.shape, DType.boolean);
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    _unaryOp<Complex, bool>(
      result.data,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.real.isFinite && x.imag.isFinite,
    );
  } else if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, bool>(
      result.data,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => true,
    );
  } else {
    _unaryOp<double, bool>(
      result.data,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.isFinite,
    );
  }
  return result;
}

/// Clip (limit) the values in an array.
///
/// Given an interval `[min, max]`, values outside the interval are clipped
/// to the interval edges.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray clip(NDArray a, num min, num max, {NDArray? out}) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for clip');
  }
  final result = out ?? NDArray.create(a.shape, a.dtype);
  if (out != null) {
    if (!listEquals(out.shape, a.shape)) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape for clip.',
      );
    }
    if (out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible DType for clip.',
      );
    }
  }

  final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);

  if (a.isContiguous) {
    if (a.dtype == DType.float64) {
      v_clip_double(
        a.pointer.cast(),
        result.pointer.cast(),
        min.toDouble(),
        max.toDouble(),
        size,
      );
      return result;
    } else if (a.dtype == DType.float32) {
      v_clip_float(
        a.pointer.cast(),
        result.pointer.cast(),
        min.toDouble(),
        max.toDouble(),
        size,
      );
      return result;
    }
  }
  final resultStrides = NDArray.computeCStrides(a.shape);

  if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    final mn = min.toInt();
    final mx = max.toInt();
    _unaryOp<int, int>(
      result.data as List<int>,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.clamp(mn, mx),
    );
  } else {
    final mn = min.toDouble();
    final mx = max.toDouble();
    _unaryOp<double, double>(
      result.data as List<double>,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => x.clamp(mn, mx),
    );
  }
  return result;
}

bool _isTrue(dynamic x) {
  if (x is bool) {
    return x;
  } else if (x is Complex) {
    return x.real != 0.0 || x.imag != 0.0;
  } else if (x is num) {
    return x != 0;
  }
  return false;
}

/// Compute the element-wise truth value of NOT [a].
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray logical_not(NDArray a) {
  final result = NDArray.create(a.shape, DType.int32);
  final resultStrides = NDArray.computeCStrides(a.shape);
  final rData = result.data as List<int>;

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    _unaryOp<Complex, int>(
      rData,
      a.data as List<Complex>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => _isTrue(x) ? 0 : 1,
    );
  } else if (a.dtype == DType.int32 || a.dtype == DType.int64) {
    _unaryOp<int, int>(
      rData,
      a.data as List<int>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => _isTrue(x) ? 0 : 1,
    );
  } else {
    _unaryOp<double, int>(
      rData,
      a.data as List<double>,
      a.shape,
      a.strides,
      resultStrides,
      0,
      0,
      0,
      (x) => _isTrue(x) ? 0 : 1,
    );
  }
  return result;
}

/// Compute the element-wise truth value of [a] AND [b] with broadcasting support.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray logical_and(NDArray a, NDArray b) {
  final broadcastResult = broadcast(a, b);
  final shape = broadcastResult.shape;
  final result = NDArray.create(shape, DType.int32);
  final resultStrides = NDArray.computeCStrides(shape);
  final rData = result.data as List<int>;

  _dispatchBinaryLogical(
    rData,
    a,
    b,
    shape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    (x, y) => (_isTrue(x) && _isTrue(y)) ? 1 : 0,
  );
  return result;
}

/// Compute the element-wise truth value of [a] OR [b] with broadcasting support.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray logical_or(NDArray a, NDArray b) {
  final broadcastResult = broadcast(a, b);
  final shape = broadcastResult.shape;
  final result = NDArray.create(shape, DType.int32);
  final resultStrides = NDArray.computeCStrides(shape);
  final rData = result.data as List<int>;

  _dispatchBinaryLogical(
    rData,
    a,
    b,
    shape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    (x, y) => (_isTrue(x) || _isTrue(y)) ? 1 : 0,
  );
  return result;
}

/// Compute the element-wise truth value of [a] XOR [b] with broadcasting support.
///
/// **Example:**
/// {@example /example/ufuncs_example.dart lang=dart}
NDArray logical_xor(NDArray a, NDArray b) {
  final broadcastResult = broadcast(a, b);
  final shape = broadcastResult.shape;
  final result = NDArray.create(shape, DType.int32);
  final resultStrides = NDArray.computeCStrides(shape);
  final rData = result.data as List<int>;

  _dispatchBinaryLogical(
    rData,
    a,
    b,
    shape,
    broadcastResult.stridesA,
    broadcastResult.stridesB,
    resultStrides,
    (x, y) => (_isTrue(x) != _isTrue(y)) ? 1 : 0,
  );
  return result;
}

void _dispatchBinaryLogical(
  List<int> rData,
  NDArray a,
  NDArray b,
  List<int> shape,
  List<int> sA,
  List<int> sB,
  List<int> sR,
  int Function(dynamic, dynamic) op,
) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    final aData = a.data as List<Complex>;
    if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
      _elementWiseOp<Complex, Complex, int>(
        rData,
        aData,
        b.data as List<Complex>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else if (b.dtype == DType.float64 || b.dtype == DType.float32) {
      _elementWiseOp<Complex, double, int>(
        rData,
        aData,
        b.data as List<double>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else if (b.dtype == DType.boolean) {
      _elementWiseOp<Complex, bool, int>(
        rData,
        aData,
        b.data as List<bool>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else {
      _elementWiseOp<Complex, int, int>(
        rData,
        aData,
        b.data as List<int>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    }
  } else if (a.dtype == DType.float64 || a.dtype == DType.float32) {
    final aData = a.data as List<double>;
    if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
      _elementWiseOp<double, Complex, int>(
        rData,
        aData,
        b.data as List<Complex>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else if (b.dtype == DType.float64 || b.dtype == DType.float32) {
      _elementWiseOp<double, double, int>(
        rData,
        aData,
        b.data as List<double>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else if (b.dtype == DType.boolean) {
      _elementWiseOp<double, bool, int>(
        rData,
        aData,
        b.data as List<bool>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else {
      _elementWiseOp<double, int, int>(
        rData,
        aData,
        b.data as List<int>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    }
  } else if (a.dtype == DType.boolean) {
    final aData = a.data as List<bool>;
    if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
      _elementWiseOp<bool, Complex, int>(
        rData,
        aData,
        b.data as List<Complex>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else if (b.dtype == DType.float64 || b.dtype == DType.float32) {
      _elementWiseOp<bool, double, int>(
        rData,
        aData,
        b.data as List<double>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else if (b.dtype == DType.boolean) {
      _elementWiseOp<bool, bool, int>(
        rData,
        aData,
        b.data as List<bool>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else {
      _elementWiseOp<bool, int, int>(
        rData,
        aData,
        b.data as List<int>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    }
  } else {
    final aData = a.data as List<int>;
    if (b.dtype == DType.complex128 || b.dtype == DType.complex64) {
      _elementWiseOp<int, Complex, int>(
        rData,
        aData,
        b.data as List<Complex>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else if (b.dtype == DType.float64 || b.dtype == DType.float32) {
      _elementWiseOp<int, double, int>(
        rData,
        aData,
        b.data as List<double>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else if (b.dtype == DType.boolean) {
      _elementWiseOp<int, bool, int>(
        rData,
        aData,
        b.data as List<bool>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    } else {
      _elementWiseOp<int, int, int>(
        rData,
        aData,
        b.data as List<int>,
        shape,
        sA,
        sB,
        sR,
        0,
        0,
        0,
        0,
        op,
      );
    }
  }
}

/// Returns a sorted copy of an array along a specified [axis].
///
/// This function corresponds to NumPy's `sort` function.
///
/// It uses native ANSI C `qsort` via FFI to perform zero-copy, high-speed
/// in-place sorting straight on the C heap for contiguous last-axis rows, completely
/// bypassing Dart memory marshalling.
///
/// Complex numbers are sorted lexicographically: by their real parts first,
/// and by their imaginary parts if the real parts are equal.
///
/// **Preconditions:**
/// - [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [RangeError] if [axis] is out of bounds.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
NDArray sort(NDArray a, {int axis = -1}) {
  final rank = a.shape.length;
  if (rank == 0) {
    return NDArray.fromList(List.from(a.data), [], a.dtype);
  }

  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(targetAxis, 0, rank - 1, 'axis');
  }

  if (targetAxis != rank - 1) {
    final swappedView = a.swapaxes(targetAxis, rank - 1);
    final sortedView = sort(swappedView, axis: rank - 1);
    return sortedView.swapaxes(targetAxis, rank - 1);
  }

  NDArray src = a;
  if (!a.isContiguous) {
    src = NDArray.fromList(a.toList(), a.shape, a.dtype);
  }

  final result = NDArray.create(src.shape, src.dtype);
  if (src.dtype == DType.float64) {
    (result.data as Float64List).setRange(
      0,
      src.data.length,
      src.data as Float64List,
    );
  } else if (src.dtype == DType.float32) {
    (result.data as Float32List).setRange(
      0,
      src.data.length,
      src.data as Float32List,
    );
  } else if (src.dtype == DType.int64) {
    (result.data as Int64List).setRange(
      0,
      src.data.length,
      src.data as Int64List,
    );
  } else if (src.dtype == DType.int32) {
    (result.data as Int32List).setRange(
      0,
      src.data.length,
      src.data as Int32List,
    );
  } else if (src.dtype == DType.complex128 || src.dtype == DType.complex64) {
    final srcBacking = (src.data as ComplexList).backingList;
    final resBacking = (result.data as ComplexList).backingList;
    if (srcBacking is Float64List && resBacking is Float64List) {
      resBacking.setRange(0, srcBacking.length, srcBacking);
    } else if (srcBacking is Float32List && resBacking is Float32List) {
      resBacking.setRange(0, srcBacking.length, srcBacking);
    }
  } else if (src.dtype == DType.boolean) {
    final srcBacking = (src.data as BoolList).backingList;
    final resBacking = (result.data as BoolList).backingList;
    resBacking.setRange(0, srcBacking.length, srcBacking);
  }

  final n = src.shape.last;
  final totalSize = src.shape.isEmpty ? 1 : src.shape.reduce((x, y) => x * y);
  final numRows = totalSize ~/ n;

  int elementSizeInBytes;
  if (src.dtype == DType.float64 || src.dtype == DType.int64) {
    elementSizeInBytes = 8;
  } else if (src.dtype == DType.float32 || src.dtype == DType.int32) {
    elementSizeInBytes = 4;
  } else if (src.dtype == DType.complex64) {
    elementSizeInBytes = 8;
  } else if (src.dtype == DType.complex128) {
    elementSizeInBytes = 16;
  } else {
    throw UnimplementedError('Unsupported dtype for sort: ${src.dtype}');
  }

  final baseCast = result.pointer.cast<ffi.Uint8>();
  final rowSizeInBytes = n * elementSizeInBytes;

  for (var r = 0; r < numRows; r++) {
    final rowPtr = baseCast + (r * rowSizeInBytes);

    // High-speed direct C sorters bypassing FFI context switches
    if (src.dtype == DType.float64) {
      native_sort_double(rowPtr.cast<ffi.Double>(), n);
    } else if (src.dtype == DType.float32) {
      native_sort_float(rowPtr.cast<ffi.Float>(), n);
    } else if (src.dtype == DType.int64) {
      native_sort_int64(rowPtr.cast<ffi.LongLong>(), n);
    } else if (src.dtype == DType.int32) {
      native_sort_int32(rowPtr.cast<ffi.Int>(), n);
    } else if (src.dtype == DType.complex128) {
      native_sort_complex128(rowPtr.cast<ffi.Double>(), n);
    } else if (src.dtype == DType.complex64) {
      native_sort_complex64(rowPtr.cast<ffi.Float>(), n);
    }
  }

  return result;
}

/// Returns the indices that would sort an array along a specified [axis].
///
/// This function corresponds to NumPy's `argsort` function.
///
/// It performs indirect index sorting, fully supporting complex numbers
/// lexicographical logic and internal axes reorientation via the axis-swapping pipeline.
///
/// **Preconditions:**
/// - [axis] must be within `[-rank, rank - 1]`.
///
/// **Throws:**
/// - [RangeError] if [axis] is out of bounds.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
NDArray<int> argsort(NDArray a, {int axis = -1}) {
  final rank = a.shape.length;
  if (rank == 0) {
    return NDArray.fromList(<int>[0], [], DType.int32);
  }

  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(targetAxis, 0, rank - 1, 'axis');
  }

  if (targetAxis != rank - 1) {
    final swappedView = a.swapaxes(targetAxis, rank - 1);
    final sortedIndicesView = argsort(swappedView, axis: rank - 1);
    return sortedIndicesView.swapaxes(targetAxis, rank - 1);
  }

  NDArray src = a;
  if (!a.isContiguous) {
    src = NDArray.fromList(a.toList(), a.shape, a.dtype);
  }

  final n = src.shape.last;
  final totalSize = src.shape.isEmpty ? 1 : src.shape.reduce((x, y) => x * y);
  final numRows = totalSize ~/ n;

  final result = NDArray<int>.create(src.shape, DType.int32);

  if (src.dtype == DType.float64) {
    final dataPtr = src.pointer.cast<ffi.Double>();
    final resPtr = result.pointer.cast<ffi.Int>();
    for (var r = 0; r < numRows; r++) {
      native_argsort_double(dataPtr + r * n, resPtr + r * n, n);
    }
    return result;
  } else if (src.dtype == DType.float32) {
    final dataPtr = src.pointer.cast<ffi.Float>();
    final resPtr = result.pointer.cast<ffi.Int>();
    for (var r = 0; r < numRows; r++) {
      native_argsort_float(dataPtr + r * n, resPtr + r * n, n);
    }
    return result;
  } else if (src.dtype == DType.int64) {
    final dataPtr = src.pointer.cast<ffi.LongLong>();
    final resPtr = result.pointer.cast<ffi.Int>();
    for (var r = 0; r < numRows; r++) {
      native_argsort_int64(dataPtr + r * n, resPtr + r * n, n);
    }
    return result;
  } else if (src.dtype == DType.int32) {
    final dataPtr = src.pointer.cast<ffi.Int>();
    final resPtr = result.pointer.cast<ffi.Int>();
    for (var r = 0; r < numRows; r++) {
      native_argsort_int32(dataPtr + r * n, resPtr + r * n, n);
    }
    return result;
  }

  for (var r = 0; r < numRows; r++) {
    final rowStart = r * n;
    final indices = List<int>.generate(n, (i) => i);

    if (src.dtype == DType.complex128 || src.dtype == DType.complex64) {
      final dataList = src.data as List<Complex>;
      indices.sort((i, j) {
        final cA = dataList[rowStart + i];
        final cB = dataList[rowStart + j];
        if (cA.real != cB.real) return cA.real.compareTo(cB.real);
        return cA.imag.compareTo(cB.imag);
      });
    } else {
      throw UnimplementedError('Unsupported dtype for argsort: ${src.dtype}');
    }

    for (var i = 0; i < n; i++) {
      result.data[rowStart + i] = indices[i];
    }
  }

  return result;
}

void _whereOpRec<Tc, Tx, Ty, Tr>(
  List<Tr> result,
  List<Tc> cond,
  List<Tx> x,
  List<Ty> y,
  List<int> shape,
  List<int> sCond,
  List<int> sX,
  List<int> sY,
  List<int> sResult,
  int dim,
  int oCond,
  int oX,
  int oY,
  int oResult,
) {
  if (dim == shape.length) {
    final cVal = cond[oCond];
    final isTrue = _isTrue(cVal);
    if (isTrue) {
      final xVal = x[oX];
      result[oResult] = (Tr == Complex && xVal is num)
          ? Complex(xVal.toDouble(), 0.0) as Tr
          : xVal as Tr;
    } else {
      final yVal = y[oY];
      result[oResult] = (Tr == Complex && yVal is num)
          ? Complex(yVal.toDouble(), 0.0) as Tr
          : yVal as Tr;
    }
    return;
  }

  for (var i = 0; i < shape[dim]; i++) {
    _whereOpRec<Tc, Tx, Ty, Tr>(
      result,
      cond,
      x,
      y,
      shape,
      sCond,
      sX,
      sY,
      sResult,
      dim + 1,
      oCond + i * sCond[dim],
      oX + i * sX[dim],
      oY + i * sY[dim],
      oResult + i * sResult[dim],
    );
  }
}

void _dispatchWhere(
  List rData,
  NDArray condition,
  NDArray x,
  NDArray y,
  List<int> shape,
  List<int> sCond,
  List<int> sX,
  List<int> sY,
  List<int> sResult,
) {
  final cData = condition.data;
  final xData = x.data;
  final yData = y.data;

  if (x.dtype == DType.complex128 ||
      x.dtype == DType.complex64 ||
      y.dtype == DType.complex128 ||
      y.dtype == DType.complex64) {
    final resList = rData as List<Complex>;
    _whereOpRec<dynamic, dynamic, dynamic, Complex>(
      resList,
      cData,
      xData,
      yData,
      shape,
      sCond,
      sX,
      sY,
      sResult,
      0,
      0,
      0,
      0,
      0,
    );
  } else if (x.dtype == DType.float64 ||
      x.dtype == DType.float32 ||
      y.dtype == DType.float64 ||
      y.dtype == DType.float32) {
    final resList = rData as List<double>;
    _whereOpRec<dynamic, dynamic, dynamic, double>(
      resList,
      cData,
      xData,
      yData,
      shape,
      sCond,
      sX,
      sY,
      sResult,
      0,
      0,
      0,
      0,
      0,
    );
  } else {
    final resList = rData as List<int>;
    _whereOpRec<dynamic, dynamic, dynamic, int>(
      resList,
      cData,
      xData,
      yData,
      shape,
      sCond,
      sX,
      sY,
      sResult,
      0,
      0,
      0,
      0,
      0,
    );
  }
}

/// Return elements chosen from [x] or [y] depending on [condition].
///
/// This function corresponds to NumPy's `where` function.
///
/// **Ternary Select and SIMD Pipelines**:
/// - If [x] and [y] are omitted, this functions acts as a dynamic alias returning [nonzero(condition)].
/// - If all three arrays share identical shapes, are of `float32`/`int32` types, and are
///   C-contiguous, an optimized SIMD hardware vector pipeline can be wired.
/// - Non-contiguous layouts or mixed types fall back to portably generic upcasting loops.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
List<int> _broadcastStrides(NDArray a, List<int> targetShape) {
  final strides = List<int>.filled(targetShape.length, 0);
  final offset = targetShape.length - a.shape.length;
  for (var i = 0; i < a.shape.length; i++) {
    final targetDim = targetShape[i + offset];
    final aDim = a.shape[i];
    if (aDim == targetDim) {
      strides[i + offset] = a.strides[i];
    } else if (aDim == 1) {
      strides[i + offset] = 0;
    } else {
      throw ArgumentError('Cannot broadcast shape ${a.shape} to $targetShape');
    }
  }
  return strides;
}

List<int> _broadcast3Shapes(List<int> s1, List<int> s2, List<int> s3) {
  final len = math.max(s1.length, math.max(s2.length, s3.length));
  final common = List<int>.filled(len, 1);
  for (var i = 0; i < len; i++) {
    final dim1 = s1.length - 1 - i >= 0 ? s1[s1.length - 1 - i] : 1;
    final dim2 = s2.length - 1 - i >= 0 ? s2[s2.length - 1 - i] : 1;
    final dim3 = s3.length - 1 - i >= 0 ? s3[s3.length - 1 - i] : 1;

    final target = math.max(dim1, math.max(dim2, dim3));
    if (dim1 != target && dim1 != 1) {
      throw ArgumentError('Incompatible shapes for broadcasting');
    }
    if (dim2 != target && dim2 != 1) {
      throw ArgumentError('Incompatible shapes for broadcasting');
    }
    if (dim3 != target && dim3 != 1) {
      throw ArgumentError('Incompatible shapes for broadcasting');
    }
    common[len - 1 - i] = target;
  }
  return common;
}

dynamic where(NDArray condition, [NDArray? x, NDArray? y, NDArray? out]) {
  if (x == null && y == null) {
    if (out != null) {
      throw ArgumentError(
        'out buffer cannot be provided when x and y are omitted.',
      );
    }
    return nonzero(condition);
  }

  if ((x == null && y != null) || (x != null && y == null)) {
    throw ArgumentError('Either both or neither of x and y must be given');
  }

  // Calculate target common shape via high-speed 3-way broadcast matching
  final commonShape = _broadcast3Shapes(condition.shape, x!.shape, y!.shape);

  final targetDType = _resolveDType(x.dtype, y.dtype);

  if (out != null) {
    if (!listEquals(out.shape, commonShape) || out.dtype != targetDType) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for where() result.',
      );
    }
  }

  // Compute precise broadcasted strides for each operand independently to commonShape
  final stridesCond = _broadcastStrides(condition, commonShape);
  final stridesX = _broadcastStrides(x, commonShape);
  final stridesY = _broadcastStrides(y, commonShape);

  final result = out ?? NDArray.create(commonShape, targetDType);
  final resultStrides = NDArray.computeCStrides(commonShape);

  // 0. Advanced ND Odometer Ternary Broadcasting Engine in C (Rank <= 8)
  if (commonShape.length <= 8 && condition.dtype == DType.boolean) {
    final cShape = malloc<ffi.Int>(commonShape.length);
    final cStridesCond = malloc<ffi.Int>(stridesCond.length);
    final cStridesX = malloc<ffi.Int>(stridesX.length);
    final cStridesY = malloc<ffi.Int>(stridesY.length);
    final cStridesRes = malloc<ffi.Int>(resultStrides.length);

    for (var i = 0; i < commonShape.length; i++) {
      cShape[i] = commonShape[i];
      cStridesCond[i] = stridesCond[i];
      cStridesX[i] = stridesX[i];
      cStridesY[i] = stridesY[i];
      cStridesRes[i] = resultStrides[i];
    }

    try {
      if (targetDType == DType.float64 &&
          x.dtype == DType.float64 &&
          y.dtype == DType.float64) {
        s_where_double(
          condition.pointer.cast(),
          cStridesCond,
          x.pointer.cast(),
          cStridesX,
          y.pointer.cast(),
          cStridesY,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
        );
        return result;
      } else if (targetDType == DType.float32 &&
          x.dtype == DType.float32 &&
          y.dtype == DType.float32) {
        s_where_float(
          condition.pointer.cast(),
          cStridesCond,
          x.pointer.cast(),
          cStridesX,
          y.pointer.cast(),
          cStridesY,
          result.pointer.cast(),
          cStridesRes,
          cShape,
          commonShape.length,
        );
        return result;
      }
    } finally {
      malloc.free(cShape);
      malloc.free(cStridesCond);
      malloc.free(cStridesX);
      malloc.free(cStridesY);
      malloc.free(cStridesRes);
    }
  }

  _dispatchWhere(
    result.data,
    condition,
    x,
    y,
    commonShape,
    stridesCond,
    stridesX,
    stridesY,
    resultStrides,
  );

  return result;
}

/// Returns the indices of the elements that are non-zero.
///
/// Returns a `List<NDArray<int>>` containing 1D integer arrays, one for each dimension
/// of [a], which give the coordinates of the non-zero elements along that dimension.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
List<NDArray<int>> nonzero(NDArray a) {
  final rank = a.shape.length;
  final coordinateLists = List.generate(rank, (_) => <int>[]);

  _nonzeroRecursive(a, List<int>.filled(rank, 0), 0, coordinateLists);

  return coordinateLists.map((list) {
    return NDArray<int>.fromList(list, [list.length], DType.int32);
  }).toList();
}

void _nonzeroRecursive(
  NDArray a,
  List<int> currentPos,
  int currentDim,
  List<List<int>> coordinateLists,
) {
  if (currentDim == a.shape.length) {
    final val = a[currentPos];
    if (_isTrue(val)) {
      for (var i = 0; i < currentPos.length; i++) {
        coordinateLists[i].add(currentPos[i]);
      }
    }
    return;
  }

  for (var i = 0; i < a.shape[currentDim]; i++) {
    currentPos[currentDim] = i;
    _nonzeroRecursive(a, currentPos, currentDim + 1, coordinateLists);
  }
}

/// Count the number of non-zero elements in the array [a].
///
/// If [axis] is provided, counts along that axis and returns a new array.
/// Otherwise, counts all elements globally and returns a single scalar integer.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
dynamic count_nonzero(NDArray a, {int? axis}) {
  if (axis == null) {
    var count = 0;
    if (a.isContiguous) {
      for (var i = 0; i < a.data.length; i++) {
        if (_isTrue(a.data[i])) count++;
      }
      return count;
    }
    final rank = a.shape.length;
    final pos = List<int>.filled(rank, 0);
    int countWalk(int dim) {
      if (dim == rank) return _isTrue(a[pos]) ? 1 : 0;
      var subCount = 0;
      for (var i = 0; i < a.shape[dim]; i++) {
        pos[dim] = i;
        subCount += countWalk(dim + 1);
      }
      return subCount;
    }

    return countWalk(0);
  }

  final rank = a.shape.length;
  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(targetAxis, 0, rank - 1, 'axis');
  }

  final targetShape = List<int>.from(a.shape)..removeAt(targetAxis);
  final result = NDArray<int>.zeros(targetShape, DType.int32);

  _countNonzeroRecursive(a, result, List<int>.filled(rank, 0), targetAxis, 0);

  return result;
}

void _countNonzeroRecursive(
  NDArray src,
  NDArray<int> dest,
  List<int> srcPos,
  int targetAxis,
  int currentDim,
) {
  if (currentDim == src.shape.length) {
    if (_isTrue(src[srcPos])) {
      final destPos = List<int>.from(srcPos)..removeAt(targetAxis);
      var destOffset = 0;
      for (var i = 0; i < dest.shape.length; i++) {
        destOffset += destPos[i] * dest.strides[i];
      }
      dest.data[destOffset] += 1;
    }
    return;
  }

  for (var i = 0; i < src.shape[currentDim]; i++) {
    srcPos[currentDim] = i;
    _countNonzeroRecursive(src, dest, srcPos, targetAxis, currentDim + 1);
  }
}

/// Returns the indices of the maximum values along an [axis].
///
/// If [axis] is null, flattens the array and returns a flat scalar integer index.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
dynamic argmax(NDArray a, {int? axis}) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for argmax');
  }

  if (axis == null) {
    NDArray src = a;
    if (!a.isContiguous) {
      src = NDArray.fromList(a.toList(), a.shape, a.dtype);
    }

    var maxIdx = 0;
    if (src.dtype == DType.int32 || src.dtype == DType.int64) {
      final dataList = src.data as List<int>;
      var maxVal = dataList[0];
      for (var i = 1; i < dataList.length; i++) {
        if (dataList[i] > maxVal) {
          maxVal = dataList[i];
          maxIdx = i;
        }
      }
    } else {
      final dataList = src.data as List<double>;
      var maxVal = dataList[0];
      for (var i = 1; i < dataList.length; i++) {
        if (dataList[i] > maxVal) {
          maxVal = dataList[i];
          maxIdx = i;
        }
      }
    }
    return maxIdx;
  }

  final rank = a.shape.length;
  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(targetAxis, 0, rank - 1, 'axis');
  }

  final targetShape = List<int>.from(a.shape)..removeAt(targetAxis);
  final result = NDArray<int>.create(targetShape, DType.int32);
  result.data.fillRange(0, result.data.length, 0);

  _argMinMaxRecursive(
    a,
    result,
    List<int>.filled(rank, 0),
    targetAxis,
    0,
    true,
  );
  return result;
}

/// Returns the indices of the minimum values along an [axis].
///
/// If [axis] is null, flattens the array and returns a flat scalar integer index.
///
/// **Example:**
/// {@example /example/sorting_searching_example.dart lang=dart}
dynamic argmin(NDArray a, {int? axis}) {
  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    throw UnsupportedError('Complex numbers are not supported for argmin');
  }

  if (axis == null) {
    NDArray src = a;
    if (!a.isContiguous) {
      src = NDArray.fromList(a.toList(), a.shape, a.dtype);
    }

    var minIdx = 0;
    if (src.dtype == DType.int32 || src.dtype == DType.int64) {
      final dataList = src.data as List<int>;
      var minVal = dataList[0];
      for (var i = 1; i < dataList.length; i++) {
        if (dataList[i] < minVal) {
          minVal = dataList[i];
          minIdx = i;
        }
      }
    } else {
      final dataList = src.data as List<double>;
      var minVal = dataList[0];
      for (var i = 1; i < dataList.length; i++) {
        if (dataList[i] < minVal) {
          minVal = dataList[i];
          minIdx = i;
        }
      }
    }
    return minIdx;
  }

  final rank = a.shape.length;
  final targetAxis = axis < 0 ? rank + axis : axis;
  if (targetAxis < 0 || targetAxis >= rank) {
    throw RangeError.range(targetAxis, 0, rank - 1, 'axis');
  }

  final targetShape = List<int>.from(a.shape)..removeAt(targetAxis);
  final result = NDArray<int>.create(targetShape, DType.int32);
  result.data.fillRange(0, result.data.length, 0);

  _argMinMaxRecursive(
    a,
    result,
    List<int>.filled(rank, 0),
    targetAxis,
    0,
    false,
  );
  return result;
}

void _argMinMaxRecursive(
  NDArray src,
  NDArray<int> dest,
  List<int> srcPos,
  int targetAxis,
  int currentDim,
  bool isMax,
) {
  if (currentDim == src.shape.length) {
    final destPos = List<int>.from(srcPos)..removeAt(targetAxis);
    var destOffset = 0;
    for (var i = 0; i < dest.shape.length; i++) {
      destOffset += destPos[i] * dest.strides[i];
    }

    final currentBestIndex = dest.data[destOffset];
    final currentBestPos = List<int>.from(srcPos);
    currentBestPos[targetAxis] = currentBestIndex;

    final currentVal = src[srcPos] as num;
    final bestVal = src[currentBestPos] as num;

    if (isMax) {
      if (srcPos[targetAxis] > currentBestIndex && currentVal > bestVal) {
        dest.data[destOffset] = srcPos[targetAxis];
      }
    } else {
      if (srcPos[targetAxis] > currentBestIndex && currentVal < bestVal) {
        dest.data[destOffset] = srcPos[targetAxis];
      }
    }
    return;
  }

  for (var i = 0; i < src.shape[currentDim]; i++) {
    srcPos[currentDim] = i;
    _argMinMaxRecursive(src, dest, srcPos, targetAxis, currentDim + 1, isMax);
  }
}

/// Compute the Cholesky decomposition of a square 2D matrix.
///
/// Factorizes a symmetric, positive-definite matrix [a] into `A = L * L^T`, where `L`
/// is a lower triangular matrix factor.
///
/// Returns a Map `{'L': lMatrix}` containing the lower factor.
///
/// **Example:**
/// {@example /example/linalg_example.dart lang=dart}
Map<String, NDArray> cholesky(NDArray a) {
  if (a.shape.length != 2 || a.shape[0] != a.shape[1]) {
    throw ArgumentError('Matrix must be square and 2D (was ${a.shape})');
  }
  final n = a.shape[0];
  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;

  // Create result matrix L and copy elements from a
  final lMat = NDArray<double>.zeros([n, n], targetDType);

  if (a.dtype == DType.float64) {
    (lMat.data as Float64List).setRange(0, n * n, a.data as Float64List);
  } else if (a.dtype == DType.float32) {
    (lMat.data as Float32List).setRange(0, n * n, a.data as Float32List);
  } else {
    for (var i = 0; i < n * n; i++) {
      lMat.data[i] = (a.data[i] as num).toDouble();
    }
  }

  // Char 'L' in ASCII is 76
  const uploL = 76;

  if (targetDType == DType.float64) {
    final info = LAPACKE_dpotrf(
      101, // ROW_MAJOR
      uploL,
      n,
      lMat.pointer.cast<ffi.Double>(),
      n,
    );
    if (info < 0) {
      throw ArgumentError('Illegal value in call to LAPACKE_dpotrf: $info');
    }
    if (info > 0) {
      throw ArgumentError(
        'Matrix must be symmetric positive-definite for Cholesky decomposition',
      );
    }

    // Zero-out strictly upper triangular part
    final lData = lMat.data as Float64List;
    for (var i = 0; i < n; i++) {
      for (var j = i + 1; j < n; j++) {
        lData[i * n + j] = 0.0;
      }
    }
  } else {
    final info = LAPACKE_spotrf(
      101, // ROW_MAJOR
      uploL,
      n,
      lMat.pointer.cast<ffi.Float>(),
      n,
    );
    if (info < 0) {
      throw ArgumentError('Illegal value in call to LAPACKE_spotrf: $info');
    }
    if (info > 0) {
      throw ArgumentError(
        'Matrix must be symmetric positive-definite for Cholesky decomposition',
      );
    }

    // Zero-out strictly upper triangular part
    final lData = lMat.data as Float32List;
    for (var i = 0; i < n; i++) {
      for (var j = i + 1; j < n; j++) {
        lData[i * n + j] = 0.0;
      }
    }
  }

  return {'L': lMat};
}

/// Computes the QR decomposition of a matrix $A = Q R$.
///
/// Decomposes a matrix [a] into an orthogonal matrix `Q` and an upper triangular matrix `R`
/// such that `a = Q * R`.
/// Natively offloads to LAPACK solvers (`dgeqrf` / `sgeqrf` and `dorgqr` / `sorgqr`) depending on precision.
///
/// **Preconditions:**
/// - Input matrix [a] must be 2-dimensional.
///
/// **Throws:**
/// - [ArgumentError] if [a] is not 2D.
/// - [StateError] if native FFI memory allocation or LAPACK solver initialization fails.
///
/// **Performance considerations:**
/// - Executes at high-speed natively in unmanaged C space.
///
/// **Example:**
/// ```dart
/// final a = NDArray<double>.fromList([12.0, -51.0, 4.0, 6.0, 167.0, -68.0, -4.0, 24.0, -41.0], [3, 3], DType.float64);
/// final res = qr(a);
/// final q = res['Q']!;
/// final r = res['R']!;
/// ```
Map<String, NDArray> qr(NDArray a) {
  if (a.shape.length != 2) {
    throw ArgumentError('Matrix must be 2D (was ${a.shape})');
  }
  final m = a.shape[0];
  final n = a.shape[1];
  final k = m < n ? m : n;
  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;

  final aCopy = NDArray.create([m, n], targetDType);
  if (a.isContiguous && a.dtype == targetDType) {
    if (targetDType == DType.float64) {
      aCopy.pointer
          .cast<ffi.Double>()
          .asTypedList(m * n)
          .setRange(0, m * n, a.data as List<double>);
    } else {
      aCopy.pointer
          .cast<ffi.Float>()
          .asTypedList(m * n)
          .setRange(0, m * n, a.data as List<double>);
    }
  } else {
    final flat = a.toList();
    if (targetDType == DType.float64) {
      aCopy.pointer
          .cast<ffi.Double>()
          .asTypedList(m * n)
          .setRange(0, m * n, flat.cast<double>());
    } else {
      aCopy.pointer
          .cast<ffi.Float>()
          .asTypedList(m * n)
          .setRange(0, m * n, flat.cast<double>());
    }
  }

  final rMat = NDArray<double>.zeros([k, n], targetDType);
  final qMat = NDArray<double>.zeros([m, k], targetDType);

  if (targetDType == DType.float64) {
    final tau = malloc<ffi.Double>(k);
    try {
      final info = LAPACKE_dgeqrf(
        101, // ROW_MAJOR
        m,
        n,
        aCopy.pointer.cast<ffi.Double>(),
        n,
        tau,
      );
      if (info != 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_dgeqrf: $info');
      }

      // Extract upper triangular matrix R from aCopy
      final rData = rMat.data as List<double>;
      final aCopyData = aCopy.data as List<double>;
      for (var i = 0; i < k; i++) {
        for (var j = i; j < n; j++) {
          rData[i * n + j] = aCopyData[i * n + j];
        }
      }

      // Copy reflectors to qMat (first k columns of aCopy)
      final qData = qMat.data as List<double>;
      for (var i = 0; i < m; i++) {
        for (var j = 0; j < k; j++) {
          qData[i * k + j] = aCopyData[i * n + j];
        }
      }

      // Call dorgqr to reconstruct the orthonormal columns of Q in-place in qMat
      final infoOrg = LAPACKE_dorgqr(
        101, // ROW_MAJOR
        m,
        k,
        k,
        qMat.pointer.cast<ffi.Double>(),
        k,
        tau,
      );
      if (infoOrg != 0) {
        throw ArgumentError(
          'Illegal value in call to LAPACKE_dorgqr: $infoOrg',
        );
      }
    } finally {
      malloc.free(tau);
      aCopy.dispose();
    }
  } else {
    final tau = malloc<ffi.Float>(k);
    try {
      final info = LAPACKE_sgeqrf(
        101, // ROW_MAJOR
        m,
        n,
        aCopy.pointer.cast<ffi.Float>(),
        n,
        tau,
      );
      if (info != 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_sgeqrf: $info');
      }

      // Extract upper triangular matrix R from aCopy
      final rData = rMat.data as List<double>;
      final aCopyData = aCopy.data as List<double>;
      for (var i = 0; i < k; i++) {
        for (var j = i; j < n; j++) {
          rData[i * n + j] = aCopyData[i * n + j];
        }
      }

      // Copy reflectors to qMat (first k columns of aCopy)
      final qData = qMat.data as List<double>;
      for (var i = 0; i < m; i++) {
        for (var j = 0; j < k; j++) {
          qData[i * k + j] = aCopyData[i * n + j];
        }
      }

      // Call sorgqr to reconstruct the orthonormal columns of Q in-place in qMat
      final infoOrg = LAPACKE_sorgqr(
        101, // ROW_MAJOR
        m,
        k,
        k,
        qMat.pointer.cast<ffi.Float>(),
        k,
        tau,
      );
      if (infoOrg != 0) {
        throw ArgumentError(
          'Illegal value in call to LAPACKE_sorgqr: $infoOrg',
        );
      }
    } finally {
      malloc.free(tau);
      aCopy.dispose();
    }
  }

  return {'Q': qMat, 'R': rMat};
}

/// Computes the Singular Value Decomposition (SVD) of a matrix $A = U S V^h$.
///
/// Decomposes a matrix [a] into left singular vectors `U`, singular values `S`,
/// and right singular vectors `Vh` such that `a = U * diag(S) * Vh`.
/// Natively offloads to LAPACK solvers (`dgesdd` / `sgesdd`) depending on precision.
///
/// **Preconditions:**
/// - Input matrix [a] must be 2-dimensional.
///
/// **Throws:**
/// - [ArgumentError] if [a] is not 2D.
/// - [StateError] if native FFI memory allocation or LAPACK solver initialization fails.
///
/// **Performance considerations:**
/// - Executes at high-speed natively in unmanaged C space.
///
/// **Example:**
/// ```dart
/// final a = NDArray<double>.fromList([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2], DType.float64);
/// final res = svd(a);
/// final u = res['U']!;
/// final s = res['S']!;
/// final vh = res['Vh']!;
/// ```
Map<String, NDArray> svd(NDArray a) {
  if (a.shape.length != 2) {
    throw ArgumentError('Matrix must be 2D (was ${a.shape})');
  }
  final m = a.shape[0];
  final n = a.shape[1];
  final targetDType = a.dtype == DType.float32 ? DType.float32 : DType.float64;

  if (m < n) {
    final aT = a.transpose();
    final resT = svd(aT);
    final uNew = resT['U']!;
    final sNew = resT['S']!;
    final vhNew = resT['Vh']!;

    final uResult = vhNew.transpose();
    final vhResult = uNew.transpose();

    return {'U': uResult, 'S': sNew, 'Vh': vhResult};
  }

  final aCopy = NDArray.create([m, n], targetDType);
  if (a.isContiguous && a.dtype == targetDType) {
    if (targetDType == DType.float64) {
      aCopy.pointer
          .cast<ffi.Double>()
          .asTypedList(m * n)
          .setRange(0, m * n, a.data as List<double>);
    } else {
      aCopy.pointer
          .cast<ffi.Float>()
          .asTypedList(m * n)
          .setRange(0, m * n, a.data as List<double>);
    }
  } else {
    final flat = a.toList();
    if (targetDType == DType.float64) {
      aCopy.pointer
          .cast<ffi.Double>()
          .asTypedList(m * n)
          .setRange(0, m * n, flat.cast<double>());
    } else {
      aCopy.pointer
          .cast<ffi.Float>()
          .asTypedList(m * n)
          .setRange(0, m * n, flat.cast<double>());
    }
  }

  final sMat = NDArray<double>.zeros([n], targetDType);
  final uMat = NDArray<double>.zeros([m, m], targetDType);
  final vtMat = NDArray<double>.zeros([n, n], targetDType);

  if (targetDType == DType.float64) {
    final superb = malloc<ffi.Double>(math.max(1, n - 1));
    try {
      final info = LAPACKE_dgesvd(
        101, // ROW_MAJOR
        65, // 'A'
        65, // 'A'
        m,
        n,
        aCopy.pointer.cast<ffi.Double>(),
        n,
        sMat.pointer.cast<ffi.Double>(),
        uMat.pointer.cast<ffi.Double>(),
        m,
        vtMat.pointer.cast<ffi.Double>(),
        n,
        superb,
      );
      if (info != 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_dgesvd: $info');
      }
    } finally {
      malloc.free(superb);
      aCopy.dispose();
    }
  } else {
    final superb = malloc<ffi.Float>(math.max(1, n - 1));
    try {
      final info = LAPACKE_sgesvd(
        101, // ROW_MAJOR
        65, // 'A'
        65, // 'A'
        m,
        n,
        aCopy.pointer.cast<ffi.Float>(),
        n,
        sMat.pointer.cast<ffi.Float>(),
        uMat.pointer.cast<ffi.Float>(),
        m,
        vtMat.pointer.cast<ffi.Float>(),
        n,
        superb,
      );
      if (info != 0) {
        throw ArgumentError('Illegal value in call to LAPACKE_sgesvd: $info');
      }
    } finally {
      malloc.free(superb);
      aCopy.dispose();
    }
  }

  return {'U': uMat, 'S': sMat, 'Vh': vtMat};
}

/// Extract a diagonal or construct a diagonal array.
///
/// If [v] is a 2D matrix, extracts the k-th diagonal elements vector as a zero-copy 1D view.
/// If [v] is a 1D vector, constructs a 2D square matrix with [v] as the k-th diagonal and zeros elsewhere.
///
/// **Preconditions:**
/// - Input [v] must be a 1D or 2D array.
///
/// **Throws:**
/// - [ArgumentError] if [v] rank is not 1 or 2.
///
/// **Example:**
/// {@example /example/diag_example.dart lang=dart}
///
/// Reference: [Diagonal Matrix](https://en.wikipedia.org/wiki/Diagonal_matrix)
NDArray diag(NDArray v, {int k = 0}) {
  if (v.shape.length == 2) {
    // 1. Extracting a diagonal from a 2D matrix as a copy-free 1D view!
    final m = v.shape[0];
    final n = v.shape[1];

    int startRow;
    int startCol;
    int len;

    if (k >= 0) {
      startRow = 0;
      startCol = k;
      if (startCol >= n) {
        return NDArray.create([0], v.dtype);
      }
      len = math.min(m, n - k);
    } else {
      startRow = -k;
      startCol = 0;
      if (startRow >= m) {
        return NDArray.create([0], v.dtype);
      }
      len = math.min(m + k, n);
    }

    if (len <= 0) {
      return NDArray.create([0], v.dtype);
    }

    // View offset in parent sequential coordinates
    final offsetElements = startRow * v.strides[0] + startCol * v.strides[1];
    // Diagonal spacing stride: strides[0] + strides[1]!
    final diagStride = v.strides[0] + v.strides[1];

    return NDArray.view(v, [len], [diagStride], offsetElements: offsetElements);
  } else if (v.shape.length == 1) {
    // 2. Constructing a 2D diagonal matrix from a 1D vector
    final n = v.shape[0];
    final size = n + k.abs();
    final result = NDArray.zeros([size, size], v.dtype);

    int startRow;
    int startCol;

    if (k >= 0) {
      startRow = 0;
      startCol = k;
    } else {
      startRow = -k;
      startCol = 0;
    }

    final vList = v.toList();
    final resData = result.data;
    final resStrides = result.strides;

    for (var i = 0; i < n; i++) {
      final targetIdx =
          (startRow + i) * resStrides[0] + (startCol + i) * resStrides[1];
      resData[targetIdx] = vList[i];
    }

    return result;
  } else {
    throw ArgumentError('Input array must be 1- or 2-dimensional.');
  }
}

/// Returns a boolean [NDArray] where two arrays are element-wise equal within a tolerance.
///
/// The tolerance relation is defined as:
/// `abs(a - b) <= (atol + rtol * abs(b))`
///
/// **Preconditions:**
/// - Input [a] and [b] must be numeric arrays.
/// - [a] and [b] must have compatible broadcast shapes.
///
/// **Example:**
/// {@example /example/isclose_example.dart lang=dart}
///
/// Reference: [Approximate Equality](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html)
NDArray<bool> isclose(
  NDArray a,
  NDArray b, {
  double rtol = 1e-05,
  double atol = 1e-08,
  bool equalNan = false,
}) {
  final broadcastResult = broadcast(a, b);
  final commonShape = broadcastResult.shape;

  final size = commonShape.isEmpty ? 1 : commonShape.reduce((x, y) => x * y);
  final result = NDArray<bool>.zeros(commonShape, DType.boolean);

  final aList = a.toList().cast<num>();
  final bList = b.toList().cast<num>();
  final resData = result.data;

  final broadcastResultRes = broadcast(a, b);
  final stridesA = broadcastResultRes.stridesA;
  final stridesB = broadcastResultRes.stridesB;
  final stridesRes = NDArray.computeCStrides(commonShape);

  final coord = List<int>.filled(commonShape.length, 0);

  for (var el = 0; el < size; el++) {
    var offsetA = 0;
    var offsetB = 0;
    var offsetRes = 0;
    for (var d = 0; d < commonShape.length; d++) {
      offsetA += coord[d] * stridesA[d];
      offsetB += coord[d] * stridesB[d];
      offsetRes += coord[d] * stridesRes[d];
    }

    final valA = aList[offsetA].toDouble();
    final valB = bList[offsetB].toDouble();

    var match = false;
    if (equalNan && valA.isNaN && valB.isNaN) {
      match = true;
    } else if (valA.isInfinite || valB.isInfinite) {
      match = valA == valB;
    } else {
      final diff = (valA - valB).abs();
      final limit = atol + rtol * valB.abs();
      match = diff <= limit;
    }

    resData[offsetRes] = match;

    // Advance coord odometer
    for (var d = commonShape.length - 1; d >= 0; d--) {
      coord[d]++;
      if (coord[d] < commonShape[d]) break;
      coord[d] = 0;
    }
  }

  return result;
}

/// Returns true if two arrays are element-wise equal within a tolerance.
///
/// The tolerance relation is defined as:
/// `abs(a - b) <= (atol + rtol * abs(b))`
///
/// **Preconditions:**
/// - Input [a] and [b] must be numeric arrays.
/// - [a] and [b] must have compatible broadcast shapes.
///
/// **Example:**
/// {@example /example/isclose_example.dart lang=dart}
///
/// Reference: [Approximate Equality](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html)
bool allclose(
  NDArray a,
  NDArray b, {
  double rtol = 1e-05,
  double atol = 1e-08,
  bool equalNan = false,
}) {
  final closeMask = isclose(a, b, rtol: rtol, atol: atol, equalNan: equalNan);
  final maskList = closeMask.toList();
  closeMask.dispose();
  for (final val in maskList) {
    if (!val) return false;
  }
  return true;
}

/// Replace NaN with zero and infinity with large finite numbers.
///
/// By default, maps NaN to [nan] (which defaults to 0.0), maps positive infinity
/// to [posinf] (or the maximum finite float value if null), and maps negative infinity
/// to [neginf] (or the minimum finite float value if null).
///
/// **Preconditions:**
/// - Input [a] must be a numeric array.
///
/// **Throws:**
/// - [ArgumentError] if the provided [out] buffer has an incompatible shape.
///
/// **Example:**
/// {@example /example/nan_to_num_example.dart lang=dart}
///
/// Reference: [Replace NaN and Infinities](https://numpy.org/doc/stable/reference/generated/numpy.nan_to_num.html)
NDArray nan_to_num(
  NDArray a, {
  double nan = 0.0,
  double? posinf,
  double? neginf,
  NDArray? out,
}) {
  if (out != null) {
    if (!listEquals(out.shape, a.shape) || out.dtype != a.dtype) {
      throw ArgumentError(
        'Provided out buffer has incompatible shape or dtype for nan_to_num.',
      );
    }
  }

  final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
  final aList = a.toList();
  final resultCopy = out ?? NDArray.create(a.shape, a.dtype);

  final maxLimit = a.dtype == DType.float32
      ? 3.4028234663852886e+38
      : double.maxFinite;
  final minLimit = -maxLimit;

  final targetPosInf = posinf ?? maxLimit;
  final targetNegInf = neginf ?? minLimit;

  final cleanList = <dynamic>[];

  if (a.dtype == DType.complex128 || a.dtype == DType.complex64) {
    final complexList = aList.cast<Complex>();
    for (var i = 0; i < size; i++) {
      var r = complexList[i].real;
      var img = complexList[i].imag;

      if (r.isNaN) r = nan;
      if (r == double.infinity) r = targetPosInf;
      if (r == double.negativeInfinity) r = targetNegInf;

      if (img.isNaN) img = nan;
      if (img == double.infinity) img = targetPosInf;
      if (img == double.negativeInfinity) img = targetNegInf;

      cleanList.add(Complex(r, img));
    }
  } else {
    final numList = aList.cast<num>();
    for (var i = 0; i < size; i++) {
      var val = numList[i].toDouble();

      if (val.isNaN) {
        val = nan;
      } else if (val == double.infinity) {
        val = targetPosInf;
      } else if (val == double.negativeInfinity) {
        val = targetNegInf;
      }

      cleanList.add(val);
    }
  }

  // View-Safe Strided Odometer Write Back!
  final resData = resultCopy.data;
  final resStrides = resultCopy.strides;
  final coord = List<int>.filled(a.shape.length, 0);

  for (var i = 0; i < size; i++) {
    var offsetRes = 0;
    for (var d = 0; d < a.shape.length; d++) {
      offsetRes += coord[d] * resStrides[d];
    }

    resData[offsetRes] = cleanList[i];

    // Advance odometer
    for (var d = a.shape.length - 1; d >= 0; d--) {
      coord[d]++;
      if (coord[d] < a.shape[d]) break;
      coord[d] = 0;
    }
  }

  return resultCopy;
}

/// Expand the shape of an array by inserting a new axis of size 1.
///
/// Inserts a new dimension of size 1 at the specified [axis] position.
/// If [axis] is negative, it is normalized relative to the rank of [a] plus 1.
///
/// **Preconditions:**
/// - [axis] normalized value must be between `0` and the rank of [a] inclusive.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds.
///
/// **Performance considerations:**
/// - This is a zero-copy view manipulation executing in absolute $O(1)$ time complexity.
///
/// **Example:**
/// {@example /example/shape_view_example.dart lang=dart}
///
/// Reference: [Expand Dimensions](https://numpy.org/doc/stable/reference/generated/numpy.expand_dims.html)
NDArray expand_dims(NDArray a, int axis) {
  final rank = a.shape.length;
  var targetAxis = axis < 0 ? rank + 1 + axis : axis;

  if (targetAxis < 0 || targetAxis > rank) {
    throw ArgumentError(
      'Axis $axis is out of bounds for array of rank $rank (valid bounds: [${-rank - 1}, $rank])',
    );
  }

  final newShape = List<int>.from(a.shape);
  final newStrides = List<int>.from(a.strides);

  newShape.insert(targetAxis, 1);
  // Insert stride mapping: copy sibling stride or default to 1 if rank is 0
  final siblingStride = targetAxis < rank ? a.strides[targetAxis] : 1;
  newStrides.insert(targetAxis, siblingStride);

  return NDArray.view(a, newShape, newStrides, offsetElements: 0);
}

/// Remove axes of size 1 from the shape of an array.
///
/// If [axis] is provided, only removes the specified dimensions (which must have size 1).
/// If [axis] is null, removes all dimensions of size 1.
///
/// **Preconditions:**
/// - Specified [axis] entries must indeed correspond to dimensions of size 1.
///
/// **Throws:**
/// - [ArgumentError] if [axis] is out of bounds or targets a dimension whose size is not 1.
///
/// **Performance considerations:**
/// - This is a zero-copy view manipulation executing in absolute $O(1)$ time complexity.
///
/// **Example:**
/// {@example /example/shape_view_example.dart lang=dart}
///
/// Reference: [Squeeze Dimensions](https://numpy.org/doc/stable/reference/generated/numpy.squeeze.html)
NDArray squeeze(NDArray a, {List<int>? axis}) {
  final rank = a.shape.length;
  final shape = a.shape;
  final strides = a.strides;

  final newShape = <int>[];
  final newStrides = <int>[];

  final squeezeAxes = <int>{};
  if (axis != null) {
    for (final ax in axis) {
      final targetAx = ax < 0 ? rank + ax : ax;
      if (targetAx < 0 || targetAx >= rank) {
        throw ArgumentError('Axis $ax is out of bounds for rank $rank');
      }
      if (shape[targetAx] != 1) {
        throw ArgumentError(
          'Cannot squeeze axis $ax because its dimension size is ${shape[targetAx]} (must be 1)',
        );
      }
      squeezeAxes.add(targetAx);
    }
  } else {
    for (var i = 0; i < rank; i++) {
      if (shape[i] == 1) {
        squeezeAxes.add(i);
      }
    }
  }

  for (var i = 0; i < rank; i++) {
    if (!squeezeAxes.contains(i)) {
      newShape.add(shape[i]);
      newStrides.add(strides[i]);
    }
  }

  // Squeezing all dimensions of a 1D unit tensor (e.g. shape [1]) yields a 0D scalar shape []
  return NDArray.view(a, newShape, newStrides, offsetElements: 0);
}
