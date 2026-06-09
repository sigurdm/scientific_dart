import 'package:ndarray/ndarray.dart';

void main() {
  NDArray.scope(() {
    final arr = NDArray<double>.fromList([1.0, 2.0, 3.0], [3], DType.float64);

    // Constant padding
    final constantPadded = pad(
      arr,
      PadWidth.all(2),
      mode: PaddingMode.constant,
      constantValues: PadValues.all(0.0),
    );
    print('Constant padded: ${constantPadded.toList()}');
    // Output: [0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0]

    // Edge padding
    final edgePadded = pad(arr, PadWidth.all(2), mode: PaddingMode.edge);
    print('Edge padded: ${edgePadded.toList()}');
    // Output: [1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0]

    // Reflect padding
    final reflectPadded = pad(arr, PadWidth.all(2), mode: PaddingMode.reflect);
    print('Reflect padded: ${reflectPadded.toList()}');
    // Output: [3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0]
  });
}
