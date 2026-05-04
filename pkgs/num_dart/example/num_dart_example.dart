import 'dart:typed_data';
import 'package:num_dart/num_dart.dart';

void main() {
  print('--- NDArray Creation ---');
  final a = NDArray.fromList(Float64List.fromList([1, 2, 3, 4]), [
    2,
    2,
  ], DType.float64);
  print('Array A:\nShape: ${a.shape}\nStrides: ${a.strides}\nData: ${a.data}');

  print('\n--- Broadcasting Addition ---');
  final b = NDArray.fromList(Float64List.fromList([10, 20]), [
    2,
    1,
  ], DType.float64);
  final c = NDArray.fromList(Float64List.fromList([1, 2, 3]), [
    1,
    3,
  ], DType.float64);
  print('Array B shape: ${b.shape}');
  print('Array C shape: ${c.shape}');

  final d = add(b, c);
  print('Result B + C shape: ${d.shape}');
  print('Result B + C data: ${d.data}');

  print('\n--- Matrix Multiplication (OpenBLAS) ---');
  final m1 = NDArray.fromList(Float64List.fromList([1, 2, 3, 4]), [
    2,
    2,
  ], DType.float64);
  final m2 = NDArray.fromList(Float64List.fromList([5, 6, 7, 8]), [
    2,
    2,
  ], DType.float64);

  final m3 = matmul(m1, m2);
  print('Result m1 * m2 shape: ${m3.shape}');
  print('Result m1 * m2 data: ${m3.data}');
}
