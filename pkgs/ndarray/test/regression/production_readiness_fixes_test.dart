import 'dart:async' show Zone;
import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';

void main() {
  group('Workstream 1: Memory, Scope, & Stride Safety Fixes', () {
    test(
      'C1: _s_stat_strided_fallback does not overflow buffer when out is strided',
      () {
        NDArray.scope(() {
          // Create an input array with float16 (which uses fallback path in stats.dart)
          final a = NDArray<Float16>.fromList(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2, 3],
            DType.float16,
          );

          // Create a larger backing array and take a non-contiguous slice as out
          final base = NDArray<Float16>.zeros([3, 3], DType.float16);
          // Base elements are 0. We'll use a strided slice: rows 0, 1, 2 at col 1 (strides = [3])
          final outView = NDArray<Float16>.view(
            base,
            shape: [3],
            strides: [3],
            offsetElements: 1,
          );
          expect(outView.isContiguous, isFalse);

          // Sum along axis 0 (reducing 2x3 -> 3) into outView
          final res = sum<Float16>(a, axis: 0, out: outView);
          expect(identical(res, outView), isTrue);

          // Column sums: [1+4, 2+5, 3+6] = [5.0, 7.0, 9.0]
          expect(outView.getCell([0]).toDouble(), equals(5.0));
          expect(outView.getCell([1]).toDouble(), equals(7.0));
          expect(outView.getCell([2]).toDouble(), equals(9.0));

          // Check that untouched base elements remain 0.0
          expect(base.getCell([0, 0]).toDouble(), equals(0.0));
          expect(base.getCell([0, 2]).toDouble(), equals(0.0));
          expect(base.getCell([1, 0]).toDouble(), equals(0.0));
          expect(base.getCell([1, 2]).toDouble(), equals(0.0));
          expect(base.getCell([2, 0]).toDouble(), equals(0.0));
          expect(base.getCell([2, 2]).toDouble(), equals(0.0));
        });
      },
    );

    test('H6: sort and argsort write correctly into non-contiguous out', () {
      NDArray.scope(() {
        final a = NDArray<Float64>.fromList(
          [4.0, 2.0, 5.0, 1.0, 3.0, 6.0],
          [2, 3],
          DType.float64,
        );

        // Backing array for out
        final baseSort = NDArray<Float64>.zeros([2, 5], DType.float64);
        final outSort = NDArray<Float64>.view(
          baseSort,
          shape: [2, 3],
          strides: [5, 1],
          offsetElements: 1,
        );
        expect(outSort.isContiguous, isFalse);

        sort(a, axis: -1, out: outSort);
        // Row 0 sorted: [2.0, 4.0, 5.0]
        // Row 1 sorted: [1.0, 3.0, 6.0]
        expect(outSort.getCell([0, 0]), equals(Float64(2.0)));
        expect(outSort.getCell([0, 1]), equals(Float64(4.0)));
        expect(outSort.getCell([0, 2]), equals(Float64(5.0)));
        expect(outSort.getCell([1, 0]), equals(Float64(1.0)));
        expect(outSort.getCell([1, 1]), equals(Float64(3.0)));
        expect(outSort.getCell([1, 2]), equals(Float64(6.0)));

        // argsort with non-contiguous out
        final baseArgsort = NDArray<int>.zeros([2, 5], DType.int32);
        final outArgsort = NDArray<int>.view(
          baseArgsort,
          shape: [2, 3],
          strides: [5, 1],
          offsetElements: 1,
        );
        expect(outArgsort.isContiguous, isFalse);

        argsort(a, axis: -1, out: outArgsort);
        expect(outArgsort.getCell([0, 0]), equals(1));
        expect(outArgsort.getCell([0, 1]), equals(0));
        expect(outArgsort.getCell([0, 2]), equals(2));
        expect(outArgsort.getCell([1, 0]), equals(0));
        expect(outArgsort.getCell([1, 1]), equals(1));
        expect(outArgsort.getCell([1, 2]), equals(2));
      });
    });

    test(
      'H6: partition and argpartition write correctly into non-contiguous out',
      () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [9.0, 1.0, 7.0, 3.0, 5.0, 2.0],
            [2, 3],
            DType.float64,
          );

          final basePart = NDArray<Float64>.zeros([2, 6], DType.float64);
          final outPart = NDArray<Float64>.view(
            basePart,
            shape: [2, 3],
            strides: [6, 1],
            offsetElements: 2,
          );
          expect(outPart.isContiguous, isFalse);

          partition(a, 1, axis: -1, out: outPart);
          // For row 0: elements are 9, 1, 7. kth=1 means index 1 should have 7.0, index 0 <= 7.0, index 2 >= 7.0
          expect(outPart.getCell([0, 1]), equals(Float64(7.0)));
          expect(outPart.getCell([0, 0]).toDouble(), lessThanOrEqualTo(7.0));
          expect(outPart.getCell([0, 2]).toDouble(), greaterThanOrEqualTo(7.0));

          final baseArgpart = NDArray<int>.zeros([2, 6], DType.int32);
          final outArgpart = NDArray<int>.view(
            baseArgpart,
            shape: [2, 3],
            strides: [6, 1],
            offsetElements: 2,
          );
          expect(outArgpart.isContiguous, isFalse);

          argpartition(a, 1, axis: -1, out: outArgpart);
          final kthIdx = outArgpart.getCell([0, 1]);
          expect(a.getCell([0, kthIdx]), equals(Float64(7.0)));
        });
      },
    );

    test(
      'C4: mean on empty axis returns NaN instead of uninitialized heap garbage',
      () {
        NDArray.scope(() {
          // Shape [0, 4], axis: 0 -> targetShape is [4]
          final emptyArr = NDArray<Float64>.zeros([0, 4], DType.float64);

          final m0 = mean(emptyArr, axis: 0);
          expect(m0.shape, equals([4]));
          for (var i = 0; i < 4; i++) {
            expect(m0.getCell([i]).toDouble().isNaN, isTrue);
          }

          // Test with explicit out buffer
          final outBuf = NDArray<Float64>.zeros([4], DType.float64);
          mean(emptyArr, axis: 0, out: outBuf);
          for (var i = 0; i < 4; i++) {
            expect(outBuf.getCell([i]).toDouble().isNaN, isTrue);
          }

          // Test std and var on empty axis
          final s0 = std(emptyArr, axis: 0);
          expect(s0.shape, equals([4]));
          for (var i = 0; i < 4; i++) {
            expect(s0.getCell([i]).toDouble().isNaN, isTrue);
          }

          final v0 = variance(emptyArr, axis: 0);
          expect(v0.shape, equals([4]));
          for (var i = 0; i < 4; i++) {
            expect(v0.getCell([i]).toDouble().isNaN, isTrue);
          }
        });
      },
    );

    test('H2: matmul(x, x, out: x) handles in-place aliasing safely', () {
      NDArray.scope(() {
        final x = NDArray<Float64>.fromList(
          [1.0, 2.0, 3.0, 4.0],
          [2, 2],
          DType.float64,
        );

        // x * x = [[7, 10], [15, 22]]
        final res = matmul(x, x, out: x);
        expect(identical(res, x), isTrue);

        expect(x.getCell([0, 0]), equals(Float64(7.0)));
        expect(x.getCell([0, 1]), equals(Float64(10.0)));
        expect(x.getCell([1, 0]), equals(Float64(15.0)));
        expect(x.getCell([1, 1]), equals(Float64(22.0)));
      });
    });

    test(
      'H2: matmul(a, b, out: a) handles in-place aliasing with first operand',
      () {
        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [1.0, 0.0, 0.0, 2.0],
            [2, 2],
            DType.float64,
          );
          final b = NDArray<Float64>.fromList(
            [3.0, 4.0, 5.0, 6.0],
            [2, 2],
            DType.float64,
          );

          // [[1, 0], [0, 2]] * [[3, 4], [5, 6]] = [[3, 4], [10, 12]]
          matmul(a, b, out: a);
          expect(a.getCell([0, 0]), equals(Float64(3.0)));
          expect(a.getCell([0, 1]), equals(Float64(4.0)));
          expect(a.getCell([1, 0]), equals(Float64(10.0)));
          expect(a.getCell([1, 1]), equals(Float64(12.0)));
        });
      },
    );

    test('H7: put_along_axis does not adopt caller array into local scope', () {
      late NDArray<Float64> callerArr;

      NDArray.scope(() {
        callerArr = NDArray<Float64>.fromList(
          [10.0, 20.0, 30.0, 40.0],
          [2, 2],
          DType.float64,
        );

        // Execute put_along_axis inside an inner scope
        NDArray.scope(() {
          final indices = NDArray<int>.fromList([1, 0], [2, 1], DType.int32);
          final values = NDArray<Float64>.fromList(
            [99.0, 88.0],
            [2, 1],
            DType.float64,
          );
          put_along_axis(callerArr, indices, values, 1);
        });

        // After inner scope exits, callerArr must NOT be disposed!
        expect(callerArr.isDisposed, isFalse);
        expect(callerArr.getCell([0, 1]), equals(Float64(99.0)));
        expect(callerArr.getCell([1, 0]), equals(Float64(88.0)));
      });

      // Now outer scope has exited, so callerArr is disposed
      expect(callerArr.isDisposed, isTrue);
    });

    test(
      'H7: take_along_axis does not dispose caller out array across scopes',
      () {
        late NDArray<Float64> callerOut;

        NDArray.scope(() {
          final a = NDArray<Float64>.fromList(
            [10.0, 20.0, 30.0, 40.0],
            [2, 2],
            DType.float64,
          );
          callerOut = NDArray<Float64>.zeros([2, 1], DType.float64);

          NDArray.scope(() {
            final indices = NDArray<int>.fromList([1, 0], [2, 1], DType.int32);
            take_along_axis(a, indices, 1, out: callerOut);
          });

          expect(callerOut.isDisposed, isFalse);
          expect(callerOut.getCell([0, 0]), equals(Float64(20.0)));
          expect(callerOut.getCell([1, 0]), equals(Float64(30.0)));
        });

        expect(callerOut.isDisposed, isTrue);
      },
    );

    test(
      'H8: nanvar and nanstd return owned arrays that can be explicitly disposed',
      () {
        final a = NDArray<Float64>.fromList(
          [1.0, double.nan, 3.0, 4.0, 5.0, 6.0],
          [2, 3],
          DType.float64,
        );

        final v = nanvar(a, axis: 0);
        expect(v.isDisposed, isFalse);
        expect(v.shape, equals([3]));
        v.dispose();
        expect(v.isDisposed, isTrue);

        final s = nanstd(a, axis: 0);
        expect(s.isDisposed, isFalse);
        expect(s.shape, equals([3]));
        s.dispose();
        expect(s.isDisposed, isTrue);

        a.dispose();
        expect(a.isDisposed, isTrue);
      },
    );

    test(
      'H14: Closed ResourceScope rejects late tracking and disposes resource immediately',
      () {
        late dynamic capturedZone;
        NDArray.scope(() {
          capturedZone = Zone.current;
        });

        NDArray? lateArr;
        expect(
          () {
            capturedZone.run(() {
              lateArr = NDArray<Float64>.zeros([2, 2], DType.float64);
            });
          },
          throwsA(
            isA<StateError>().having(
              (e) => e.message,
              'message',
              contains('closed scope'),
            ),
          ),
        );

        // Even though constructor threw, if lateArr was assigned or tracked, it was disposed
        expect(lateArr, isNull);
      },
    );
  });
}
