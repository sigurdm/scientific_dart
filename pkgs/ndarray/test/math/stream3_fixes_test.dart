import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('Stream 3 Fixes', () {
    test('Task 1: einsum mixed dtypes and out buffer handling', () {
      NDArray.scope(() {
        // Mixed dtypes in einsum (int32 + float64)
        final aInt = NDArray<Int32>.fromList([1, 2, 3, 4], [2, 2], DType.int32);
        final bFloat = NDArray<Float64>.fromList(
          [0.5, 1.5, 2.5, 3.5],
          [2, 2],
          DType.float64,
        );
        final res = einsum<Object, Float64>(
          EinsumSubscripts.parse('ij,jk->ik'),
          [aInt, bFloat],
        );
        expect(res.dtype, equals(DType.float64));
        expect(res.shape, equals([2, 2]));
        // [1*0.5 + 2*2.5, 1*1.5 + 2*3.5] = [5.5, 8.5]
        // [3*0.5 + 4*2.5, 3*1.5 + 4*3.5] = [11.5, 15.5]
        expect(res[[0, 0]], closeTo(5.5, 1e-9));
        expect(res[[0, 1]], closeTo(8.5, 1e-9));
        expect(res[[1, 0]], closeTo(11.5, 1e-9));
        expect(res[[1, 1]], closeTo(18.5, 1e-9));

        // User-supplied out buffer in einsum
        final outBuf = NDArray<Float64>.zeros([2, 2], DType.float64);
        final outRes = einsum<Object, Float64>(
          EinsumSubscripts.parse('ij,jk->ik'),
          [aInt, bFloat],
          out: outBuf,
        );
        expect(identical(outRes, outBuf), isTrue);
        expect(outRes[[0, 0]], closeTo(5.5, 1e-9));
      });
    });

    test('Task 2: norm ord == 0 returns count of non-zero elements', () {
      NDArray.scope(() {
        final v = NDArray<Float64>.fromList(
          [0.0, 3.0, 0.0, -5.0, 2.0],
          [5],
          DType.float64,
        );
        final norm0 = norm(v, ord: 0).scalar;
        expect(norm0, equals(3.0));

        final vAllZero = NDArray<Float64>.zeros([4], DType.float64);
        expect(norm(vAllZero, ord: 0).scalar, equals(0.0));

        final vFloat32 = NDArray<Float32>.fromList(
          [1.0, 0.0, 2.0],
          [3],
          DType.float32,
        );
        expect(norm(vFloat32, ord: 0).scalar, equals(2.0));
      });
    });

    test(
      'Task 3: User-supplied out buffer detachment in multi_dot and convolve and correlate',
      () {
        // multi_dot with out in nested scope
        final m1 = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );
        final m2 = NDArray<Float64>.fromList(
          [2.0, 0.0, 1.0, 2.0],
          [2, 2],
          DType.float64,
        );
        final m3 = NDArray<Float64>.fromList(
          [1.0, 1.0, 0.0, 1.0],
          [2, 2],
          DType.float64,
        );

        final outDot = NDArray<Float64>.zeros([2, 2], DType.float64);
        NDArray.scope(() {
          multi_dot<Float64>([m1, m2, m3], out: outDot);
        });
        // outDot should still be valid (not disposed by inner scope)
        expect(outDot.isDisposed, isFalse);
        expect(outDot.shape, equals([2, 2]));

        // convolve with out in nested scope
        final sig = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0],
          [3],
          DType.float64,
        );
        final kernel = NDArray<Float64>.fromList(
          [0.5, 1.0],
          [2],
          DType.float64,
        );
        final outConv = NDArray<Float64>.zeros([4], DType.float64);
        NDArray.scope(() {
          convolve<Float64, Float64, Float64>(
            sig,
            kernel,
            mode: ConvMode.full,
            out: outConv,
          );
        });
        expect(outConv.isDisposed, isFalse);
        expect(outConv.shape, equals([4]));

        // correlate same with out in nested scope
        final outSame = NDArray<Float64>.zeros([3], DType.float64);
        NDArray.scope(() {
          correlate<Float64, Float64, Float64>(
            sig,
            kernel,
            mode: ConvMode.same,
            out: outSame,
          );
        });
        expect(outSame.isDisposed, isFalse);
        expect(outSame.shape, equals([3]));

        outDot.dispose();
        outConv.dispose();
        outSame.dispose();
        m1.dispose();
        m2.dispose();
        m3.dispose();
        sig.dispose();
        kernel.dispose();
      },
    );

    test('Task 4: lstsq basic computation and s result', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 1.0, 1.0, 2.0, 1.0, 3.0],
          [3, 2],
          DType.float64,
        );
        final b = NDArray<Float64>.fromList(
          [6.0, 5.0, 7.0],
          [3],
          DType.float64,
        );
        final res = lstsq<Float64, Float64, Float64>(a, b);
        expect(res.x.shape, equals([2]));
        expect(res.rank, equals(2));
        expect(res.s.shape, equals([2]));
        expect(res.s[[0]], greaterThan(0.0));
      });
    });

    test('Task 5 & 6: angle, unwrap, and correlate same mode', () {
      NDArray.scope(() {
        final sig = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 5.0],
          [5],
          DType.float64,
        );
        final kernel = NDArray<Float64>.fromList(
          [1.0, 1.0, 1.0],
          [3],
          DType.float64,
        );
        final corrSame = correlate<Float64, Float64, Float64>(
          sig,
          kernel,
          mode: ConvMode.same,
        );
        expect(corrSame.shape, equals([5]));
        expect(corrSame.isContiguous, isTrue);

        final unwrapped = unwrap(sig);
        expect(unwrapped.shape, equals([5]));
      });
    });

    test('Task 7: schur typed record with .t and .z', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [5.0, 7.0, -2.0, -4.0],
          [2, 2],
          DType.float64,
        );
        final ({NDArray<Float64> t, NDArray<Float64> z}) res =
            schur<Float64, Float64>(a, output: SchurForm.real);
        expect(res.t.shape, equals([2, 2]));
        expect(res.z.shape, equals([2, 2]));
        expect(res.t[[1, 0]], closeTo(0.0, 1e-10));
      });
    });

    test('Task 9: fftfreq and rfftfreq argument validation', () {
      expect(() => fftfreq(0), throwsArgumentError);
      expect(() => fftfreq(-1), throwsArgumentError);
      expect(() => rfftfreq(0), throwsArgumentError);
      expect(() => rfftfreq(-10), throwsArgumentError);

      NDArray.scope(() {
        final f = fftfreq(8, d: 0.1);
        expect(f.shape, equals([8]));
        final rf = rfftfreq(8, d: 0.1);
        expect(rf.shape, equals([5]));
      });
    });

    test('Task 10: batched qr decomposition', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 1.0, 3.0, 0.0, 1.0, 1.0, 0.0],
          [3, 2, 2],
          DType.float64,
        );

        final res = qr<Float64>(a);
        expect(res.Q.shape, equals([3, 2, 2]));
        expect(res.R.shape, equals([3, 2, 2]));

        // Check Q * R = A for each batch
        for (var b = 0; b < 3; b++) {
          final qSlice = res.Q
              .slice([
                Slice(start: b, stop: b + 1),
                const Slice(),
                const Slice(),
              ])
              .reshape([2, 2]);
          final rSlice = res.R
              .slice([
                Slice(start: b, stop: b + 1),
                const Slice(),
                const Slice(),
              ])
              .reshape([2, 2]);
          final aSlice = a
              .slice([
                Slice(start: b, stop: b + 1),
                const Slice(),
                const Slice(),
              ])
              .reshape([2, 2]);

          final recon = matmul(qSlice, rSlice);
          for (var r = 0; r < 2; r++) {
            for (var c = 0; c < 2; c++) {
              expect(recon[[r, c]], closeTo(aSlice[[r, c]], 1e-10));
            }
          }
        }
      });
    });
  });
}
