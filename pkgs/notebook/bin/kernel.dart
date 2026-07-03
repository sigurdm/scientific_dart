// ignore_for_file: unused_import
import 'dart:async';
import 'dart:io';
import 'dart:convert';
import 'dart:math' as math;
import 'package:ndarray/ndarray.dart';
import 'package:notebook/src/workspace.dart';
import 'package:notebook/src/kernel_helper.dart';

// Global variable registry for notebook variables
final Map<String, dynamic> _variables = {};

dynamic getVar(String name) => _variables[name];
void setVar(String name, dynamic value) {
  _variables[name] = value;
}

String prettyFormat(dynamic value) {
  if (value is NDArray) {
    return _formatNDArray(value);
  }
  return value.toString();
}

String _formatNDArray(NDArray arr) {
  final shape = arr.shape;
  // Make a contiguous copy so elements are sequential in flat memory
  final contiguous = arr.isContiguous ? arr : arr.copy();
  // ignore: invalid_use_of_internal_member
  final data = contiguous.data;
  final dtypeStr = arr.dtype.name;

  String formatNested(List<int> dims, int flatOffset) {
    if (dims.isEmpty) {
      return '';
    }
    if (dims.length == 1) {
      final len = dims[0];
      final elements = [];
      for (var i = 0; i < len; i++) {
        elements.add(data[flatOffset + i]);
      }
      return '[${elements.join(', ')}]';
    }

    final currentDim = dims[0];
    final subDims = dims.sublist(1);
    final subSize = subDims.reduce((a, b) => a * b);

    final chunks = [];
    for (var i = 0; i < currentDim; i++) {
      chunks.add(formatNested(subDims, flatOffset + i * subSize));
    }

    final indent = ' ' * (arr.rank - dims.length + 1);
    final separator = ',\n$indent';
    return '[${chunks.join(separator)}]';
  }

  final elementsStr = shape.isEmpty
      ? arr.scalar.toString()
      : formatNested(shape, 0);
  return 'NDArray<$dtypeStr>(shape: $shape, values:\n$elementsStr\n)';
}

void main() async {
  print("Kernel: active");
  stdin.listen((_) {});
}
