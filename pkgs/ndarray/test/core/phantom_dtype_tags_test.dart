import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Reified Phantom DTypeTag Hierarchy', () {
    test('runtime type checks distinguish all 15 dtypes and tag families', () {
      NDArray.scope(() {
        final f64 = NDArray.fromList([1.0, 2.0], [2], DType.float64);
        final f32 = NDArray.fromList([1.0, 2.0], [2], DType.float32);
        final f16 = NDArray.fromList([1.0, 2.0], [2], DType.float16);
        final i32 = NDArray.fromList([1, 2], [2], DType.int32);
        final i64 = NDArray.fromList([1, 2], [2], DType.int64);
        final c128 = NDArray.fromList(
          [Complex(1.0, 0.0)],
          [1],
          DType.complex128,
        );
        final b = NDArray.fromList([true, false], [2], DType.boolean);

        final Object erasedF64 = f64;
        expect(erasedF64 is NDArray<Float64>, isTrue);
        expect(erasedF64 is NDArray<Float32>, isFalse);
        expect(erasedF64 is NDArray<Float16>, isFalse);
        expect(erasedF64 is NDArray<AnyFloat>, isTrue);
        expect(erasedF64 is NDArray<AnyReal>, isTrue);
        expect(erasedF64 is NDArray<AnyInt>, isFalse);
        expect(erasedF64 is NDArray<AnyComplex>, isFalse);
        expect(erasedF64 is NDArray<AnyDType>, isTrue);

        expect((f32 as Object) is NDArray<Float32>, isTrue);
        expect((f32 as Object) is NDArray<Float64>, isFalse);
        expect((f16 as Object) is NDArray<Float16>, isTrue);
        expect((i32 as Object) is NDArray<Int32>, isTrue);
        expect((i32 as Object) is NDArray<Int64>, isFalse);
        expect((i64 as Object) is NDArray<Int64>, isTrue);
        expect((i64 as Object) is NDArray<AnyInt>, isTrue);
        expect((i64 as Object) is NDArray<AnyReal>, isTrue);
        expect((c128 as Object) is NDArray<Complex128>, isTrue);
        expect((c128 as Object) is NDArray<Complex64>, isFalse);
        expect((c128 as Object) is NDArray<AnyComplex>, isTrue);
        expect((c128 as Object) is NDArray<AnyReal>, isFalse);
        expect((b as Object) is NDArray<Boolean>, isTrue);
        expect((b as Object) is NDArray<AnyReal>, isFalse);
        expect((b as Object) is NDArray<AnyDType>, isTrue);
      });
    });

    test(
      'Bucket A: float16 linalg ops preserve Float16 runtime and static tag',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList(
            [4.0, 1.0, 1.0, 3.0],
            [2, 2],
            DType.float16,
          );
          final b = NDArray.fromList([1.0, 2.0], [2], DType.float16);

          final invA = inv(a);
          expect(invA.dtype, DType.float16);
          expect((invA as Object) is NDArray<Float16>, isTrue);

          final detA = det(a);
          expect(detA.dtype, DType.float16);
          expect((detA as Object) is NDArray<Float16>, isTrue);

          final slog = slogdet(a);
          expect(slog.sign.dtype, DType.float16);
          expect(slog.logabsdet.dtype, DType.float16);

          final x = solve(a, b);
          expect(x.dtype, DType.float16);
          expect((x as Object) is NDArray<Float16>, isTrue);

          final p = pinv(a);
          expect(p.dtype, DType.float16);
          expect((p as Object) is NDArray<Float16>, isTrue);

          final l = cholesky(a);
          expect(l.dtype, DType.float16);
          expect((l as Object) is NDArray<Float16>, isTrue);

          final qrRes = qr(a);
          expect(qrRes.q.dtype, DType.float16);
          expect(qrRes.r.dtype, DType.float16);

          final svdRes = svd(a);
          expect(svdRes.u.dtype, DType.float16);
          expect(svdRes.s.dtype, DType.float16);
          expect(svdRes.vh.dtype, DType.float16);
        });
      },
    );

    test(
      'Bucket B: index-returning ops accept Int64 out and preserve runtime tag',
      () {
        NDArray.scope(() {
          final a = NDArray.fromList([30.0, 10.0, 20.0], [3], DType.float64);
          final out64 = NDArray.zeros([3], DType.int64);

          final res = argsort(a, out: out64);
          expect(res.dtype, DType.int64);
          expect((res as Object) is NDArray<Int64>, isTrue);
          expect(res.toList(), [1, 2, 0]);
        });
      },
    );

    test(
      'Bucket C: mismatched output type annotation throws TypeError at runtime',
      () {
        NDArray.scope(() {
          final f64 = NDArray.fromList([0.0, 1.0], [2], DType.float64);
          expect(() {
            final NDArray<Float32> _ = sin<Float64, Float32>(f64);
          }, throwsA(isA<TypeError>()));
        });
      },
    );
  });
}
