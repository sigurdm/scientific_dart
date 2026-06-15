import 'dart:isolate';
import 'package:test/test.dart';
import 'package:ndarray/ndarray.dart';

// Helper for isolate division test
void _divisionWorker(SendPort sendPort) {
  NDArray.scope(() {
    try {
      final a = NDArray.fromList([1], [], DType.int64);
      final b = NDArray.fromList([0], [], DType.int64);
      floor_divide(
        a,
        b,
      ); // Use floor_divide to trigger integer division by zero
      sendPort.send('ERROR: Did not throw');
    } catch (e) {
      if (e is UnsupportedError &&
          e.toString().contains('Integer division by zero')) {
        sendPort.send('OK');
      } else {
        sendPort.send('ERROR: Threw wrong exception: $e');
      }
    }
  });
}

void _normalWorker(SendPort sendPort) {
  NDArray.scope(() {
    try {
      final a = NDArray.fromList([4], [], DType.int64);
      final b = NDArray.fromList([2], [], DType.int64);
      final res = floor_divide(a, b);
      if (res.scalar == 2) {
        sendPort.send('OK');
      } else {
        sendPort.send('ERROR: Wrong result: ${res.scalar}');
      }
    } catch (e) {
      sendPort.send('ERROR: Threw exception: $e');
    }
  });
}

void main() {
  group('Regression Tests for Native Bugs', () {
    test(
      'Thread-safe division error tracking in concurrent isolates',
      () async {
        const numWorkers = 20;
        final receivePorts = List.generate(numWorkers, (_) => ReceivePort());
        final futures = <Future<dynamic>>[];

        for (var i = 0; i < numWorkers; i++) {
          futures.add(receivePorts[i].first);
          if (i % 2 == 0) {
            await Isolate.spawn(_divisionWorker, receivePorts[i].sendPort);
          } else {
            await Isolate.spawn(_normalWorker, receivePorts[i].sendPort);
          }
        }

        final results = await Future.wait(futures);
        for (var i = 0; i < numWorkers; i++) {
          expect(results[i], 'OK', reason: 'Worker $i failed');
          receivePorts[i].close();
        }
      },
    );

    test('Complex NaN sorting strict weak ordering', () {
      NDArray.scope(() {
        final nan = double.nan;
        final data = [
          Complex(nan, 1.0),
          Complex(2.0, nan),
          Complex(nan, nan),
          Complex(1.0, 2.0),
          Complex(3.0, 4.0),
          Complex(1.0, 1.0),
        ];
        final a = NDArray.fromList(data, [6], DType.complex128);

        // This should not crash or hang.
        final sorted = sort(a);

        expect(sorted.shape, [6]);
        // Lexicographical sort order:
        // 1. (1.0, 1.0)
        // 2. (1.0, 2.0)
        // 3. (2.0, nan)  -- 2.0 real, nan imag goes to end of 2.0s
        // 4. (3.0, 4.0)
        // 5. (nan, 1.0)  -- nan real goes to end
        // 6. (nan, nan)  -- nan real, nan imag goes to very end

        final expected = [
          Complex(1.0, 1.0),
          Complex(1.0, 2.0),
          Complex(2.0, nan),
          Complex(3.0, 4.0),
          Complex(nan, 1.0),
          Complex(nan, nan),
        ];

        final actual = sorted.toList();
        for (var i = 0; i < 6; i++) {
          final act = actual[i] as Complex;
          final exp = expected[i];
          if (exp.real.isNaN) {
            expect(
              act.real.isNaN,
              isTrue,
              reason: 'Element $i real should be NaN',
            );
          } else {
            expect(act.real, exp.real, reason: 'Element $i real mismatch');
          }
          if (exp.imag.isNaN) {
            expect(
              act.imag.isNaN,
              isTrue,
              reason: 'Element $i imag should be NaN',
            );
          } else {
            expect(act.imag, exp.imag, reason: 'Element $i imag mismatch');
          }
        }
      });
    });

    test('Stable NaN segregation for double (stable sort)', () {
      NDArray.scope(() {
        final nan = double.nan;
        final a = NDArray.fromList([nan, 0.0, -0.0], [3], DType.float64);

        final sorted = sort(a, kind: SortKind.mergesort);

        expect(sorted.toList()[0], 0.0);
        expect(sorted.toList()[1], -0.0);
        expect(sorted.toList()[2].isNaN, isTrue);

        expect(1.0 / (sorted.toList()[0] as double), double.infinity);
        expect(1.0 / (sorted.toList()[1] as double), double.negativeInfinity);
      });
    });

    test('Stable NaN segregation for float (stable sort)', () {
      NDArray.scope(() {
        final nan = double.nan;
        final a = NDArray.fromList([nan, 0.0, -0.0], [3], DType.float32);

        final sorted = sort(a, kind: SortKind.mergesort);

        expect(sorted.toList()[0], 0.0);
        expect(sorted.toList()[1], -0.0);
        expect(sorted.toList()[2].isNaN, isTrue);

        expect(1.0 / (sorted.toList()[0] as double), double.infinity);
        expect(1.0 / (sorted.toList()[1] as double), double.negativeInfinity);
      });
    });
  });
}
