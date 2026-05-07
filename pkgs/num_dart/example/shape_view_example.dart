import 'dart:typed_data';
import 'package:num_dart/num_dart.dart';

void main() {
  print(
    '=== NDArray expand_dims() and squeeze() Shape View Manipulation ===\n',
  );

  // 1. Create a 1D vector of shape [3]
  final a = NDArray.fromList(Float64List.fromList([1.0, 2.0, 3.0]), [
    3,
  ], DType.float64);
  print('a: shape ${a.shape}, strides ${a.strides}, data ${a.toList()}');

  // 2. Expand dimensions at axis 0 -> shape [1, 3]
  final aExpand0 = expand_dims(a, 0);
  print(
    '\nexpand_dims(a, 0): shape ${aExpand0.shape}, strides ${aExpand0.strides}',
  );
  print('Is aExpand0 a zero-copy view? ${aExpand0.parent != null}'); // true

  // 3. Expand dimensions at axis 1 -> shape [3, 1]
  final aExpand1 = expand_dims(a, 1);
  print(
    'expand_dims(a, 1): shape ${aExpand1.shape}, strides ${aExpand1.strides}',
  );

  // 4. Squeeze dimensions
  // Create a 3D tensor of shape [1, 3, 1]
  final tensor = NDArray.fromList(Float64List.fromList([10.0, 20.0, 30.0]), [
    1,
    3,
    1,
  ], DType.float64);
  print('\nRaw tensor: shape ${tensor.shape}, strides ${tensor.strides}');

  // Squeeze all axes of size 1 -> shape [3]
  final squeezedAll = squeeze(tensor);
  print(
    'squeeze(tensor): shape ${squeezedAll.shape}, strides ${squeezedAll.strides}',
  );

  // Squeeze only axis 0 -> shape [3, 1]
  final squeezed0 = squeeze(tensor, axis: [0]);
  print(
    'squeeze(tensor, axis: [0]): shape ${squeezed0.shape}, strides ${squeezed0.strides}',
  );

  // Cleanup memory
  a.dispose();
  aExpand0.dispose();
  aExpand1.dispose();
  tensor.dispose();
  squeezedAll.dispose();
  squeezed0.dispose();
}
