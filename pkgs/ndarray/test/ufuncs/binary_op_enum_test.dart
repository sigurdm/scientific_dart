import 'package:ndarray/src/ndarray_bindings.dart' as bindings;
import 'package:ndarray/src/operations/math/binary_op.dart';
import 'package:test/test.dart';

void main() {
  group('BinaryOp Enum & C++ Native Alignment Tests', () {
    test(
      'Verifies BinaryOp enum indices match native C++ BinaryOpCode enum values',
      () {
        final values = BinaryOp.values;
        expect(
          values.length,
          equals(35),
          reason: 'Expected 35 binary operations in BinaryOp enum.',
        );

        for (var i = 0; i < values.length; i++) {
          final op = values[i];
          expect(op.index, equals(i));

          // Call native C++ function get_binary_op_enum_val(i)
          final nativeVal = bindings.get_binary_op_enum_val(i);
          expect(
            nativeVal,
            equals(i),
            reason:
                'Mismatch between Dart BinaryOp.${op.name} (index $i) and native C++ BinaryOpCode enum value ($nativeVal).',
          );
        }
      },
    );

    test('Verifies out-of-bounds index query in native C++ returns -1', () {
      expect(bindings.get_binary_op_enum_val(-1), equals(-1));
      expect(
        bindings.get_binary_op_enum_val(BinaryOp.values.length),
        equals(-1),
      );
      expect(bindings.get_binary_op_enum_val(999), equals(-1));
    });
  });
}
