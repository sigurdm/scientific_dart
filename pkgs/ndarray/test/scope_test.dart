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
  });
}
