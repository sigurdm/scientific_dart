import 'package:num_dart/num_dart.dart';
import 'package:test/test.dart';

void main() {
  group('NDArray Multidimensional Enumeration (ndenumerate) Tests', () {
    test('ndenumerate() basic 2D matrix walk checks', () {
      final a = NDArray.fromList([10, 20, 30, 40, 50, 60], [2, 3], DType.int32);
      final entries = ndenumerate(a).toList();

      expect(entries.length, 6);

      expect(entries[0].$1, [0, 0]);
      expect(entries[0].$2, 10);

      expect(entries[1].$1, [0, 1]);
      expect(entries[1].$2, 20);

      expect(entries[2].$1, [0, 2]);
      expect(entries[2].$2, 30);

      expect(entries[3].$1, [1, 0]);
      expect(entries[3].$2, 40);

      expect(entries[4].$1, [1, 1]);
      expect(entries[4].$2, 50);

      expect(entries[5].$1, [1, 2]);
      expect(entries[5].$2, 60);
    });

    test('ndenumerate() supports 1D arrays and 0D scalars', () {
      final a = NDArray.fromList([9.0, 8.0], [2], DType.float64);
      final entries1D = ndenumerate(a).toList();
      expect(entries1D.length, 2);
      expect(entries1D[0].$1, [0]);
      expect(entries1D[0].$2, 9.0);

      final scalar = NDArray.fromList([99], [], DType.int32);
      final entries0D = ndenumerate(scalar).toList();
      expect(entries0D.length, 1);
      expect(entries0D[0].$1, []);
      expect(entries0D[0].$2, 99);
    });

    test('ndenumerate() handles non-contiguous strided transposed views', () {
      final parent = NDArray.fromList([1, 2, 3, 4], [2, 2], DType.int32);
      final view = parent.transposed; // shape [2, 2], non-contiguous strides!
      expect(view.isContiguous, false);

      final entries = ndenumerate(view).toList();
      expect(entries.length, 4);

      // Transposed grid layout is:
      // [1, 3]
      // [2, 4]
      expect(entries[0].$1, [0, 0]);
      expect(entries[0].$2, 1);

      expect(entries[1].$1, [0, 1]);
      expect(entries[1].$2, 3);

      expect(entries[2].$1, [1, 0]);
      expect(entries[2].$2, 2);

      expect(entries[3].$1, [1, 1]);
      expect(entries[3].$2, 4);
    });

    test('disposed arrays throw StateError', () {
      final a = NDArray.fromList([1, 2], [2], DType.int32);
      a.dispose();
      expect(() => ndenumerate(a).toList(), throwsStateError);
    });
  });
}
