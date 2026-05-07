import 'package:ndarray/ndarray.dart';
import 'package:test/test.dart';

void main() {
  group('NDArray.scope tests', () {
    test('Scope automatically disposes arrays', () {
      late NDArray a;
      late NDArray b;
      late NDArray c;

      NDArray.scope(() {
        a = NDArray.zeros([10], DType.float64);
        b = NDArray.ones([10], DType.float64);
        c = add(a, b);

        expect(a.isDisposed, isFalse);
        expect(b.isDisposed, isFalse);
        expect(c.isDisposed, isFalse);
      });

      expect(a.isDisposed, isTrue);
      expect(b.isDisposed, isTrue);
      expect(c.isDisposed, isTrue);
    });

    test('detachFromScope() preserves array after scope ends', () {
      late NDArray a;
      late NDArray result;

      result = NDArray.scope(() {
        a = NDArray.zeros([10], DType.float64);
        final b = NDArray.ones([10], DType.float64);
        final c = add(a, b);
        return c.detachFromScope();
      });

      expect(a.isDisposed, isTrue);
      expect(result.isDisposed, isFalse);

      // Clean up manually since it was detached
      result.dispose();
      expect(result.isDisposed, isTrue);
    });

    test('Nested scopes work correctly', () {
      late NDArray outer;
      late NDArray inner;

      NDArray.scope(() {
        outer = NDArray.zeros([10], DType.float64);

        NDArray.scope(() {
          inner = NDArray.ones([10], DType.float64);
          expect(inner.isDisposed, isFalse);
        });

        expect(inner.isDisposed, isTrue);
        expect(outer.isDisposed, isFalse);
      });

      expect(outer.isDisposed, isTrue);
    });

    test('Scope disposes on error', () {
      late NDArray a;

      try {
        NDArray.scope(() {
          a = NDArray.zeros([10], DType.float64);
          throw Exception('test error');
        });
      } catch (_) {
        // expected
      }

      expect(a.isDisposed, isTrue);
    });

    test('Nested scope detach completely removes array from all scopes', () {
      late NDArray outer;
      late NDArray inner;

      NDArray.scope(() {
        outer = NDArray.zeros([10], DType.float64);

        NDArray.scope(() {
          inner = NDArray.ones([10], DType.float64);
          inner.detachFromScope(); // Completely detached!
          expect(inner.isDisposed, isFalse);
        });

        // Inner scope ended, 'inner' remains alive
        expect(inner.isDisposed, isFalse);
        expect(outer.isDisposed, isFalse);
      });

      // Outer scope ended, 'outer' is disposed, but 'inner' is completely unmanaged, so it survives!
      expect(outer.isDisposed, isTrue);
      expect(inner.isDisposed, isFalse);

      // Clean up manually
      inner.dispose();
      expect(inner.isDisposed, isTrue);
    });

    test('Hybrid List-to-Set promotion triggers past 100 arrays', () {
      final trackedArrays = <NDArray>[];

      NDArray.scope(() {
        // Allocate 105 arrays to force the List-to-Set threshold promotion crossing!
        for (var i = 0; i < 105; i++) {
          final a = NDArray.zeros([1], DType.float64);
          trackedArrays.add(a);
        }

        // Verify all are still alive
        for (final a in trackedArrays) {
          expect(a.isDisposed, isFalse);
        }
      });

      // Scope ended, so all 105 arrays should be automatically disposed of by the promoted HashSet!
      for (final a in trackedArrays) {
        expect(a.isDisposed, isTrue);
      }
    });
  });
}
