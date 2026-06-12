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

          final resAdd = add(a, b);
          expect(resAdd.toList(), [11.0, 22.0, 33.0, 44.0]);

          final resSub = subtract(a, b);
          expect(resSub.toList(), [-9.0, -18.0, -27.0, -36.0]);

          final resMul = multiply(a, b);
          expect(resMul.toList(), [10.0, 40.0, 90.0, 160.0]);

          final resDiv = divide(b, a);
          expect(resDiv.toList(), [10.0, 10.0, 10.0, 10.0]);

          // Strided non-contiguous views and broadcasting
          final aView = a.transpose();
          final bView = b.transpose();
          final resAddView = add(aView, bView);
          expect(resAddView.toList(), [11.0, 33.0, 22.0, 44.0]);

          // out Recycler parameter
          final intoBuf = NDArray.create([2, 2], DType.float64);
          final resInto = add(a, b, out: intoBuf);
          expect(resInto, intoBuf);
          expect(resInto.toList(), [11.0, 22.0, 33.0, 44.0]);

          // out incompatible shape/dtype throws ArgumentError
          expect(
            () => add(a, b, out: NDArray.create([3], DType.float64)),
            throwsArgumentError,
          );

          // matmul
          final resMatmul = matmul(a, b);
          expect(resMatmul.toList(), [70.0, 100.0, 150.0, 220.0]);

          // Mixed Float32
          final f32 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float32,
          );
          expect(add(a, f32).toList(), [2.0, 4.0, 6.0, 8.0]);
          expect(subtract(a, f32).toList(), [0.0, 0.0, 0.0, 0.0]);
          expect(multiply(a, f32).toList(), [1.0, 4.0, 9.0, 16.0]);
          expect(divide(a, f32).toList(), [1.0, 1.0, 1.0, 1.0]);

          // Mixed Int64
          final i64 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int64);
          expect(add(a, i64).toList(), [2.0, 4.0, 6.0, 8.0]);
          expect(subtract(a, i64).toList(), [0.0, 0.0, 0.0, 0.0]);
          expect(multiply(a, i64).toList(), [1.0, 4.0, 9.0, 16.0]);
          expect(divide(a, i64).toList(), [1.0, 1.0, 1.0, 1.0]);

          // Mixed Scalar
          expect(
            add(a, NDArray.fromList([10.0], [1], DType.float64)).toList(),
            [11.0, 12.0, 13.0, 14.0],
          );
          expect(
            subtract(a, NDArray.fromList([1.0], [1], DType.float64)).toList(),
            [0.0, 1.0, 2.0, 3.0],
          );
          expect(
            multiply(a, NDArray.fromList([2.0], [1], DType.float64)).toList(),
            [2.0, 4.0, 6.0, 8.0],
          );
          expect(
            divide(a, NDArray.fromList([0.5], [1], DType.float64)).toList(),
            [2.0, 4.0, 6.0, 8.0],
          );
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

          final resAdd = add(a, b);
          expect(resAdd.toList(), [11.0, 22.0, 33.0, 44.0]);

          final resSub = subtract(a, b);
          expect(resSub.toList(), [-9.0, -18.0, -27.0, -36.0]);

          final resMul = multiply(a, b);
          expect(resMul.toList(), [10.0, 40.0, 90.0, 160.0]);

          final resDiv = divide(b, a);
          expect(resDiv.toList(), [10.0, 10.0, 10.0, 10.0]);

          // Strided view
          final aView = a.transpose();
          final bView = b.transpose();
          expect(add(aView, bView).toList(), [11.0, 33.0, 22.0, 44.0]);

          // Recycler
          final intoBuf = NDArray.create([2, 2], DType.float32);
          expect(add(a, b, out: intoBuf), intoBuf);

          // Incompatible recycler
          expect(
            () => add(a, b, out: NDArray.create([3], DType.float32)),
            throwsArgumentError,
          );

          // Mixed Float64
          final f64 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          expect(add(a, f64).toList(), [2.0, 4.0, 6.0, 8.0]);
          expect(subtract(a, f64).toList(), [0.0, 0.0, 0.0, 0.0]);
          expect(multiply(a, f64).toList(), [1.0, 4.0, 9.0, 16.0]);
          expect(divide(a, f64).toList(), [1.0, 1.0, 1.0, 1.0]);

          // Mixed Scalar
          expect(
            add(a, NDArray.fromList([10.0], [1], DType.float32)).toList(),
            [11.0, 12.0, 13.0, 14.0],
          );
          expect(
            subtract(a, NDArray.fromList([1.0], [1], DType.float32)).toList(),
            [0.0, 1.0, 2.0, 3.0],
          );
          expect(
            multiply(a, NDArray.fromList([2.0], [1], DType.float32)).toList(),
            [2.0, 4.0, 6.0, 8.0],
          );
          expect(
            divide(a, NDArray.fromList([0.5], [1], DType.float32)).toList(),
            [2.0, 4.0, 6.0, 8.0],
          );
        });
      },
    );

    test(
      'Int64NDArrayOperations Contiguous & Strided ufuncs and Mixed Double',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int64);
          final b = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.int64);

          final resAdd = add(a, b);
          expect(resAdd.toList(), [11, 22, 33, 44]);

          final resSub = subtract(a, b);
          expect(resSub.toList(), [-9, -18, -27, -36]);

          final resMul = multiply(a, b);
          expect(resMul.toList(), [10, 40, 90, 160]);

          final resDiv = divide(b, a);
          expect(resDiv.toList(), [10.0, 10.0, 10.0, 10.0]);

          // Strided view
          final aView = a.transpose();
          final bView = b.transpose();
          expect(add(aView, bView).toList(), [11, 33, 22, 44]);

          // Recycler
          final intoBuf = NDArray.create([2, 2], DType.int64);
          expect(add(a, b, out: intoBuf), intoBuf);

          final intoDoubleBuf = NDArray.create([2, 2], DType.float64);
          expect(divide(b, a, out: intoDoubleBuf), intoDoubleBuf);

          // Incompatible recycler
          expect(
            () => add(a, b, out: NDArray.create([3], DType.int64)),
            throwsArgumentError,
          );
          expect(
            () => divide(b, a, out: NDArray.create([3], DType.float64)),
            throwsArgumentError,
          );

          // Mixed Double
          final f64 = NDArray.fromList(
            [1.0, 2.0, 3.0, 4.0],
            [2, 2],
            DType.float64,
          );
          expect(add(a, f64).toList(), [2.0, 4.0, 6.0, 8.0]);
          expect(subtract(a, f64).toList(), [0.0, 0.0, 0.0, 0.0]);
          expect(multiply(a, f64).toList(), [1.0, 4.0, 9.0, 16.0]);

          // Mixed Scalar
          expect(add(a, NDArray.fromList([10], [1], DType.int64)).toList(), [
            11,
            12,
            13,
            14,
          ]);
          expect(
            subtract(a, NDArray.fromList([1], [1], DType.int64)).toList(),
            [0, 1, 2, 3],
          );
          expect(
            multiply(a, NDArray.fromList([2], [1], DType.int64)).toList(),
            [2, 4, 6, 8],
          );
        });
      },
    );

    test(
      'Int32NDArrayOperations Contiguous & Strided ufuncs and Mixed Int64',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
          final b = NDArray.fromList([10, 20, 30, 40], [2, 2], DType.int32);

          final resAdd = add(a, b);
          expect(resAdd.toList(), [11, 22, 33, 44]);

          final resSub = subtract(a, b);
          expect(resSub.toList(), [-9, -18, -27, -36]);

          final resMul = multiply(a, b);
          expect(resMul.toList(), [10, 40, 90, 160]);

          final resDiv = divide(b, a);
          expect(resDiv.toList(), [10.0, 10.0, 10.0, 10.0]);

          // Strided view
          final aView = a.transpose();
          final bView = b.transpose();
          expect(add(aView, bView).toList(), [11, 33, 22, 44]);

          // Recycler
          final intoBuf = NDArray.create([2, 2], DType.int32);
          expect(add(a, b, out: intoBuf), intoBuf);

          final intoDoubleBuf = NDArray.create([2, 2], DType.float64);
          expect(divide(b, a, out: intoDoubleBuf), intoDoubleBuf);

          // Incompatible recycler
          expect(
            () => add(a, b, out: NDArray.create([3], DType.int32)),
            throwsArgumentError,
          );
          expect(
            () => divide(b, a, out: NDArray.create([3], DType.float64)),
            throwsArgumentError,
          );

          // Mixed Int64
          final i64 = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int64);
          expect(add(a, i64).toList(), [2, 4, 6, 8]);
          expect(subtract(a, i64).toList(), [0, 0, 0, 0]);
          expect(multiply(a, i64).toList(), [1, 4, 9, 16]);

          // Mixed Scalar
          expect(add(a, NDArray.fromList([10], [1], DType.int32)).toList(), [
            11,
            12,
            13,
            14,
          ]);
          expect(
            subtract(a, NDArray.fromList([1], [1], DType.int32)).toList(),
            [0, 1, 2, 3],
          );
          expect(
            multiply(a, NDArray.fromList([2], [1], DType.int32)).toList(),
            [2, 4, 6, 8],
          );
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

          final resAdd = add(a, b);
          expect(resAdd.toList()[0].real, 11.0);
          expect(resAdd.toList()[0].imag, 22.0);
          expect(resAdd.toList()[1].real, 33.0);
          expect(resAdd.toList()[1].imag, 44.0);

          final resSub = subtract(a, b);
          expect(resSub.toList()[0].real, -9.0);
          expect(resSub.toList()[0].imag, -18.0);

          final resMul = multiply(a, b);
          // (1+2i)*(10+20i) = 10 + 20i + 20i - 40 = -30 + 40i
          expect(resMul.toList()[0].real, -30.0);
          expect(resMul.toList()[0].imag, 40.0);

          final resDiv = divide(b, a);
          // (10+20i)/(1+2i) = 10*(1+2i)/(1+2i) = 10
          expect(resDiv.toList()[0].real, 10.0);
          expect(resDiv.toList()[0].imag, 0.0);

          // Strided view and recycler
          final aView = a.slice([const Slice(start: 0, stop: 2, step: 1)]);
          final bView = b.slice([const Slice(start: 0, stop: 2, step: 1)]);
          final intoBuf = NDArray.create([2], DType.complex128);
          expect(add(aView, bView, out: intoBuf), intoBuf);

          // Incompatible recycler
          expect(
            () => add(a, b, out: NDArray.create([3], DType.complex128)),
            throwsArgumentError,
          );

          // Mixed Float64
          final f64 = NDArray.fromList([1.0, 2.0], [2], DType.float64);
          final resAddF64 = add(a, f64);
          expect(resAddF64.toList()[0].real, 2.0);
          expect(resAddF64.toList()[0].imag, 2.0);

          final resSubF64 = subtract(a, f64);
          expect(resSubF64.toList()[0].real, 0.0);
          expect(resSubF64.toList()[0].imag, 2.0);

          final resMulF64 = multiply(a, f64);
          expect(resMulF64.toList()[0].real, 1.0);
          expect(resMulF64.toList()[0].imag, 2.0);

          final resDivF64 = divide(a, f64);
          expect(resDivF64.toList()[0].real, 1.0);
          expect(resDivF64.toList()[0].imag, 2.0);

          // Mixed Int64
          final i64 = NDArray.fromList([1, 2], [2], DType.int64);
          final resAddI64 = add(a, i64);
          expect(resAddI64.toList()[0].real, 2.0);
          expect(resAddI64.toList()[0].imag, 2.0);

          final resSubI64 = subtract(a, i64);
          expect(resSubI64.toList()[0].real, 0.0);

          final resMulI64 = multiply(a, i64);
          expect(resMulI64.toList()[0].real, 1.0);

          final resDivI64 = divide(a, i64);
          expect(resDivI64.toList()[0].real, 1.0);

          // Mixed Scalar
          expect(
            add(
              a,
              NDArray.fromList([Complex(10, 10)], [1], DType.complex128),
            ).toList()[0].real,
            11.0,
          );
          expect(
            subtract(
              a,
              NDArray.fromList([Complex(1, 1)], [1], DType.complex128),
            ).toList()[0].real,
            0.0,
          );
          expect(
            multiply(
              a,
              NDArray.fromList([Complex(2.0, 0.0)], [1], DType.complex128),
            ).toList()[0].real,
            2.0,
          );
          expect(
            divide(
              a,
              NDArray.fromList([Complex(0.5, 0.0)], [1], DType.complex128),
            ).toList()[0].real,
            2.0,
          );
        });
      },
    );
  });
}
