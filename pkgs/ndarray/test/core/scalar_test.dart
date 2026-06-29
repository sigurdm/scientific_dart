import "package:ndarray/ndarray.dart";
import "package:test/test.dart";

void main() {
  group("NDArray.scalar Constructor Tests", () {
    test("Float64 scalar", () {
      NDArray.scope(() {
        final a = NDArray.scalar(Float64(3.14159), DType.float64);
        expect(a.shape, <int>[]);
        expect(a.rank, 0);
        expect(a.size, 1);
        expect(a.dtype, DType.float64);
        expect(a.scalar, closeTo(3.14159, 1e-5));
      });
    });

    test("Float32 scalar", () {
      NDArray.scope(() {
        final a = NDArray.scalar(Float32(2.718), DType.float32);
        expect(a.shape, <int>[]);
        expect(a.rank, 0);
        expect(a.size, 1);
        expect(a.dtype, DType.float32);
        expect(a.scalar, closeTo(2.718, 1e-3));
      });
    });

    test("Int64 scalar", () {
      NDArray.scope(() {
        final a = NDArray.scalar(Int64(9223372036854775807), DType.int64);
        expect(a.shape, <int>[]);
        expect(a.rank, 0);
        expect(a.size, 1);
        expect(a.dtype, DType.int64);
        expect(a.scalar, 9223372036854775807);
      });
    });

    test("Int32 scalar", () {
      NDArray.scope(() {
        final a = NDArray.scalar(Int32(42), DType.int32);
        expect(a.shape, <int>[]);
        expect(a.rank, 0);
        expect(a.size, 1);
        expect(a.dtype, DType.int32);
        expect(a.scalar, 42);
      });
    });

    test("Int16 scalar", () {
      NDArray.scope(() {
        final a = NDArray.scalar(Int16(32767), DType.int16);
        expect(a.shape, <int>[]);
        expect(a.rank, 0);
        expect(a.size, 1);
        expect(a.dtype, DType.int16);
        expect(a.scalar, 32767);
      });
    });

    test("Uint8 scalar", () {
      NDArray.scope(() {
        final a = NDArray.scalar(Uint8(255), DType.uint8);
        expect(a.shape, <int>[]);
        expect(a.rank, 0);
        expect(a.size, 1);
        expect(a.dtype, DType.uint8);
        expect(a.scalar, 255);
      });
    });

    test("Boolean scalar", () {
      NDArray.scope(() {
        final a = NDArray.scalar(true, DType.boolean);
        expect(a.shape, <int>[]);
        expect(a.rank, 0);
        expect(a.size, 1);
        expect(a.dtype, DType.boolean);
        expect(a.scalar, true);

        final b = NDArray.scalar(false, DType.boolean);
        expect(b.scalar, false);
      });
    });

    test("Complex128 scalar", () {
      NDArray.scope(() {
        final val = Complex128(1.5, -2.5);
        final a = NDArray.scalar(val, DType.complex128);
        expect(a.shape, <int>[]);
        expect(a.rank, 0);
        expect(a.size, 1);
        expect(a.dtype, DType.complex128);
        expect(a.scalar.real, 1.5);
        expect(a.scalar.imag, -2.5);
      });
    });

    test("Complex64 scalar", () {
      NDArray.scope(() {
        final val = Complex64(0.5, 1.0);
        final a = NDArray.scalar(val, DType.complex64);
        expect(a.shape, <int>[]);
        expect(a.rank, 0);
        expect(a.size, 1);
        expect(a.dtype, DType.complex64);
        expect(a.scalar.real, closeTo(0.5, 1e-5));
        expect(a.scalar.imag, closeTo(1.0, 1e-5));
      });
    });
  });
}
