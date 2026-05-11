import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('NDArray Extensions Test Suite', () {
    test(
      'Float64NDArrayOperations Contiguous & Strided ufuncs and Recycler',
      () {
        NDArray.scope(() {
          // Contiguous same-shape
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray.fromList(
            [10.0, 20.0, 30.0, 40.0],
            [2, 2],
            DType.float64,
          );

          final resAdd = a.add(b);
          expect(resAdd.toList(), [11.0, 22.0, 33.0, 44.0]);

          final resSub = a.subtract(b);
          expect(resSub.toList(), [-9.0, -18.0, -27.0, -36.0]);

          final resMul = a.multiply(b);
          expect(resMul.toList(), [10.0, 40.0, 90.0, 160.0]);

          final resDiv = b.divide(a);
          expect(resDiv.toList(), [10.0, 10.0, 10.0, 10.0]);

          // Strided non-contiguous views and broadcasting
          final aView = a.transpose();
          final bView = b.transpose();
          final resAddView = aView.add(bView);
          expect(resAddView.toList(), [11.0, 33.0, 22.0, 44.0]);

          // into Recycler parameter
          final intoBuf = NDArray<Float64>.create([2, 2], DType.float64);
          final resInto = a.add(b, into: intoBuf);
          expect(resInto, intoBuf);
          expect(resInto.toList(), [11.0, 22.0, 33.0, 44.0]);

          // into incompatible shape/dtype throws ArgumentError
          expect(
            () => a.add(b, into: NDArray<Float64>.create([3], DType.float64)),
            throwsArgumentError,
          );

          // matmul with recycler
          final intoMatmul = NDArray<Float64>.create([2, 2], DType.float64);
          final resMatmul = a.matmul(b, into: intoMatmul);
          expect(resMatmul, intoMatmul);
          expect(resMatmul.toList(), [70.0, 100.0, 150.0, 220.0]);

          // Mixed Float32
          final f32 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float32,
          );
          expect(a.addFloat32(f32).toList(), [2.0, 4.0, 6.0, 8.0]);
          expect(a.subtractFloat32(f32).toList(), [0.0, 0.0, 0.0, 0.0]);
          expect(a.multiplyFloat32(f32).toList(), [1.0, 4.0, 9.0, 16.0]);
          expect(a.divideFloat32(f32).toList(), [1.0, 1.0, 1.0, 1.0]);

          // Mixed Int64
          final i64 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int64);
          expect(a.addInt64(i64).toList(), [2.0, 4.0, 6.0, 8.0]);
          expect(a.subtractInt64(i64).toList(), [0.0, 0.0, 0.0, 0.0]);
          expect(a.multiplyInt64(i64).toList(), [1.0, 4.0, 9.0, 16.0]);
          expect(a.divideInt64(i64).toList(), [1.0, 1.0, 1.0, 1.0]);

          // Mixed Scalar
          expect(a.addScalar(10.0).toList(), [11.0, 12.0, 13.0, 14.0]);
          expect(a.subtractScalar(1.0).toList(), [0.0, 1.0, 2.0, 3.0]);
          expect(a.multiplyScalar(2.0).toList(), [2.0, 4.0, 6.0, 8.0]);
          expect(a.divideScalar(0.5).toList(), [2.0, 4.0, 6.0, 8.0]);
        });
      },
    );

    test(
      'Float32NDArrayOperations Contiguous & Strided ufuncs and Mixed Scalar',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float32,
          );
          final b = NDArray.fromList(
            [10.0, 20.0, 30.0, 40.0],
            [2, 2],
            DType.float32,
          );

          final resAdd = a.add(b);
          expect(resAdd.toList(), [11.0, 22.0, 33.0, 44.0]);

          final resSub = a.subtract(b);
          expect(resSub.toList(), [-9.0, -18.0, -27.0, -36.0]);

          final resMul = a.multiply(b);
          expect(resMul.toList(), [10.0, 40.0, 90.0, 160.0]);

          final resDiv = b.divide(a);
          expect(resDiv.toList(), [10.0, 10.0, 10.0, 10.0]);

          // Strided view
          final aView = a.transpose();
          final bView = b.transpose();
          expect(aView.add(bView).toList(), [11.0, 33.0, 22.0, 44.0]);

          // Recycler
          final intoBuf = NDArray<Float32>.create([2, 2], DType.float32);
          expect(a.add(b, into: intoBuf), intoBuf);

          // Incompatible recycler
          expect(
            () => a.add(b, into: NDArray<Float32>.create([3], DType.float32)),
            throwsArgumentError,
          );

          // Mixed Float64
          final f64 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          expect(a.addFloat64(f64).toList(), [2.0, 4.0, 6.0, 8.0]);
          expect(a.subtractFloat64(f64).toList(), [0.0, 0.0, 0.0, 0.0]);
          expect(a.multiplyFloat64(f64).toList(), [1.0, 4.0, 9.0, 16.0]);
          expect(a.divideFloat64(f64).toList(), [1.0, 1.0, 1.0, 1.0]);

          // Mixed Scalar
          expect(a.addScalar(10.0).toList(), [11.0, 12.0, 13.0, 14.0]);
          expect(a.subtractScalar(1.0).toList(), [0.0, 1.0, 2.0, 3.0]);
          expect(a.multiplyScalar(2.0).toList(), [2.0, 4.0, 6.0, 8.0]);
          expect(a.divideScalar(0.5).toList(), [2.0, 4.0, 6.0, 8.0]);
        });
      },
    );

    test(
      'Int64NDArrayOperations Contiguous & Strided ufuncs and Mixed Double',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int64);
          final b = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.int64);

          final resAdd = a.add(b);
          expect(resAdd.toList(), [11, 22, 33, 44]);

          final resSub = a.subtract(b);
          expect(resSub.toList(), [-9, -18, -27, -36]);

          final resMul = a.multiply(b);
          expect(resMul.toList(), [10, 40, 90, 160]);

          final resDiv = b.divide(a);
          expect(resDiv.toList(), [10.0, 10.0, 10.0, 10.0]);

          // Strided view
          final aView = a.transpose();
          final bView = b.transpose();
          expect(aView.add(bView).toList(), [11, 33, 22, 44]);

          // Recycler
          final intoBuf = NDArray<Int64>.create([2, 2], DType.int64);
          expect(a.add(b, into: intoBuf), intoBuf);

          final intoDoubleBuf = NDArray<Float64>.create([2, 2], DType.float64);
          expect(b.divide(a, into: intoDoubleBuf), intoDoubleBuf);

          // Incompatible recycler
          expect(
            () => a.add(b, into: NDArray<Int64>.create([3], DType.int64)),
            throwsArgumentError,
          );
          expect(
            () =>
                b.divide(a, into: NDArray<Float64>.create([3], DType.float64)),
            throwsArgumentError,
          );

          // Mixed Double
          final f64 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          expect(a.addDouble(f64).toList(), [2.0, 4.0, 6.0, 8.0]);
          expect(a.subtractDouble(f64).toList(), [0.0, 0.0, 0.0, 0.0]);
          expect(a.multiplyDouble(f64).toList(), [1.0, 4.0, 9.0, 16.0]);

          // Mixed Scalar
          expect(a.addScalar(10).toList(), [11, 12, 13, 14]);
          expect(a.subtractScalar(1).toList(), [0, 1, 2, 3]);
          expect(a.multiplyScalar(2).toList(), [2, 4, 6, 8]);
        });
      },
    );

    test(
      'Int32NDArrayOperations Contiguous & Strided ufuncs and Mixed Int64',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final b = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.int32);

          final resAdd = a.add(b);
          expect(resAdd.toList(), [11, 22, 33, 44]);

          final resSub = a.subtract(b);
          expect(resSub.toList(), [-9, -18, -27, -36]);

          final resMul = a.multiply(b);
          expect(resMul.toList(), [10, 40, 90, 160]);

          final resDiv = b.divide(a);
          expect(resDiv.toList(), [10.0, 10.0, 10.0, 10.0]);

          // Strided view
          final aView = a.transpose();
          final bView = b.transpose();
          expect(aView.add(bView).toList(), [11, 33, 22, 44]);

          // Recycler
          final intoBuf = NDArray<Int32>.create([2, 2], DType.int32);
          expect(a.add(b, into: intoBuf), intoBuf);

          final intoDoubleBuf = NDArray<Float64>.create([2, 2], DType.float64);
          expect(b.divide(a, into: intoDoubleBuf), intoDoubleBuf);

          // Incompatible recycler
          expect(
            () => a.add(b, into: NDArray<Int32>.create([3], DType.int32)),
            throwsArgumentError,
          );
          expect(
            () =>
                b.divide(a, into: NDArray<Float64>.create([3], DType.float64)),
            throwsArgumentError,
          );

          // Mixed Int64
          final i64 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int64);
          expect(a.addInt64(i64).toList(), [2, 4, 6, 8]);
          expect(a.subtractInt64(i64).toList(), [0, 0, 0, 0]);
          expect(a.multiplyInt64(i64).toList(), [1, 4, 9, 16]);

          // Mixed Scalar
          expect(a.addScalar(10).toList(), [11, 12, 13, 14]);
          expect(a.subtractScalar(1).toList(), [0, 1, 2, 3]);
          expect(a.multiplyScalar(2).toList(), [2, 4, 6, 8]);
        });
      },
    );

    test(
      'ComplexNDArrayOperations Contiguous & Strided ufuncs and Mixed Scalar',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [Complex(1, 2), Complex(3, 4)],
            [2],
            DType.complex128,
          );
          final b = NDArray.fromList(
            [Complex(10, 20), Complex(30, 40)],
            [2],
            DType.complex128,
          );

          final resAdd = a.add(b);
          expect(resAdd.toList()[0].real, 11.0);
          expect(resAdd.toList()[0].imag, 22.0);
          expect(resAdd.toList()[1].real, 33.0);
          expect(resAdd.toList()[1].imag, 44.0);

          final resSub = a.subtract(b);
          expect(resSub.toList()[0].real, -9.0);
          expect(resSub.toList()[0].imag, -18.0);

          final resMul = a.multiply(b);
          // (1+2i)*(10+20i) = 10 + 20i + 20i - 40 = -30 + 40i
          expect(resMul.toList()[0].real, -30.0);
          expect(resMul.toList()[0].imag, 40.0);

          final resDiv = b.divide(a);
          // (10+20i)/(1+2i) = 10*(1+2i)/(1+2i) = 10
          expect(resDiv.toList()[0].real, 10.0);
          expect(resDiv.toList()[0].imag, 0.0);

          // Strided view and recycler
          final aView = a.slice([const Slice(start: 0, stop: 2, step: 1)]);
          final bView = b.slice([const Slice(start: 0, stop: 2, step: 1)]);
          final intoBuf = NDArray<Complex>.create([2], DType.complex128);
          expect(aView.add(bView, into: intoBuf), intoBuf);

          // Incompatible recycler
          expect(
            () =>
                a.add(b, into: NDArray<Complex>.create([3], DType.complex128)),
            throwsArgumentError,
          );

          // Mixed Float64
          final f64 = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final resAddF64 = a.addFloat64(f64);
          expect(resAddF64.toList()[0].real, 2.0);
          expect(resAddF64.toList()[0].imag, 2.0);

          final resSubF64 = a.subtractFloat64(f64);
          expect(resSubF64.toList()[0].real, 0.0);
          expect(resSubF64.toList()[0].imag, 2.0);

          final resMulF64 = a.multiplyFloat64(f64);
          expect(resMulF64.toList()[0].real, 1.0);
          expect(resMulF64.toList()[0].imag, 2.0);

          final resDivF64 = a.divideFloat64(f64);
          expect(resDivF64.toList()[0].real, 1.0);
          expect(resDivF64.toList()[0].imag, 2.0);

          // Mixed Int64
          final i64 = NDArray.fromList([1, 2], [2], DType.int64);
          final resAddI64 = a.addInt64(i64);
          expect(resAddI64.toList()[0].real, 2.0);
          expect(resAddI64.toList()[0].imag, 2.0);

          final resSubI64 = a.subtractInt64(i64);
          expect(resSubI64.toList()[0].real, 0.0);

          final resMulI64 = a.multiplyInt64(i64);
          expect(resMulI64.toList()[0].real, 1.0);

          final resDivI64 = a.divideInt64(i64);
          expect(resDivI64.toList()[0].real, 1.0);

          // Mixed Scalar
          expect(a.addScalar(Complex(10, 10)).toList()[0].real, 11.0);
          expect(a.subtractScalar(Complex(1, 1)).toList()[0].real, 0.0);
          expect(a.multiplyScalar(2.0).toList()[0].real, 2.0);
          expect(a.divideScalar(0.5).toList()[0].real, 2.0);
        });
      },
    );
  });
}
