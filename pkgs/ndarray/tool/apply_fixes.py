import re
import os

workspace_root = os.path.dirname(os.path.abspath(__file__)) + "/.."

def fix_complex_dart():
    path = os.path.join(workspace_root, "lib/src/operations/math/complex.dart")
    with open(path, "r") as f:
        content = f.read()

    # Finding 3: real() and imag() memory corruption on sliced out
    # Replace result.data.setRange with a.copy(out: result)
    old_real = """  } else {
    // This path is taken if out != null and a is not complex.
    final size = a.shape.isEmpty ? 1 : a.shape.reduce((x, y) => x * y);
    result.data.setRange(0, size, a.toList() as List<R>);
    return result;
  }"""
    new_real = """  } else {
    // This path is taken if out != null and a is not complex.
    a.copy(out: result);
    return result;
  }"""
    assert old_real in content, "old_real not found"
    content = content.replace(old_real, new_real)

    old_imag = """  if (a.dtype != DType.complex128 && a.dtype != DType.complex64) {
    if (out != null) {
      result.data.fillRange(0, result.data.length, 0.0 as R);
      return result;
    }
    return NDArray.zeros(a.shape, targetDType) as NDArray<R>;
  }"""
    new_imag = """  if (a.dtype != DType.complex128 && a.dtype != DType.complex64) {
    if (out != null) {
      result.fill(0.0 as R);
      return result;
    }
    return NDArray.zeros(a.shape, targetDType) as NDArray<R>;
  }"""
    assert old_imag in content, "old_imag not found"
    content = content.replace(old_imag, new_imag)

    with open(path, "w") as f:
        f.write(content)
    print("Fixed complex.dart")

def fix_utility_dart():
    path = os.path.join(workspace_root, "lib/src/operations/math/utility.dart")
    with open(path, "r") as f:
        content = f.read()

    # Finding 4: ndenumerate and nan_to_num offsetElements bug
    old_ndenump1 = """  if (shape.isEmpty) {
    yield ([], a.data[0]);
    return;
  }

  final coord = List<int>.filled(shape.length, 0);
  int offset = 0;"""
    new_ndenump1 = """  if (shape.isEmpty) {
    yield ([], a.data[a.offsetElements]);
    return;
  }

  final coord = List<int>.filled(shape.length, 0);
  int offset = a.offsetElements;"""
    assert old_ndenump1 in content, "old_ndenump1 not found"
    content = content.replace(old_ndenump1, new_ndenump1)

    old_nan_offset = """  for (var i = 0; i < size; i++) {
    var offsetRes = 0;
    for (var d = 0; d < a.shape.length; d++) {
      offsetRes += coord[d] * resStrides[d];
    }"""
    new_nan_offset = """  for (var i = 0; i < size; i++) {
    var offsetRes = resultCopy.offsetElements;
    for (var d = 0; d < a.shape.length; d++) {
      offsetRes += coord[d] * resStrides[d];
    }"""
    assert old_nan_offset in content, "old_nan_offset not found"
    content = content.replace(old_nan_offset, new_nan_offset)

    # Finding 5: broadcastShapes zero-length arrays
    old_bcast = """List<int> broadcastShapes(List<int> s1, List<int> s2) {
  final len = math.max(s1.length, s2.length);
  final common = List<int>.filled(len, 1);
  for (var i = 0; i < len; i++) {
    final dim1 = s1.length - 1 - i >= 0 ? s1[s1.length - 1 - i] : 1;
    final dim2 = s2.length - 1 - i >= 0 ? s2[s2.length - 1 - i] : 1;

    final target = math.max(dim1, dim2);
    if (dim1 != target && dim1 != 1) {
      throw ArgumentError('Incompatible shapes for broadcasting');
    }
    if (dim2 != target && dim2 != 1) {
      throw ArgumentError('Incompatible shapes for broadcasting');
    }
    common[len - 1 - i] = target;
  }
  return common;
}"""
    new_bcast = """List<int> broadcastShapes(List<int> s1, List<int> s2) {
  final len = math.max(s1.length, s2.length);
  final common = List<int>.filled(len, 1);
  for (var i = 0; i < len; i++) {
    final dim1 = s1.length - 1 - i >= 0 ? s1[s1.length - 1 - i] : 1;
    final dim2 = s2.length - 1 - i >= 0 ? s2[s2.length - 1 - i] : 1;

    final int target;
    if (dim1 == dim2) {
      target = dim1;
    } else if (dim1 == 1) {
      target = dim2;
    } else if (dim2 == 1) {
      target = dim1;
    } else {
      throw ArgumentError('Incompatible shapes for broadcasting');
    }
    common[len - 1 - i] = target;
  }
  return common;
}"""
    assert old_bcast in content, "old_bcast not found"
    content = content.replace(old_bcast, new_bcast)

    with open(path, "w") as f:
        f.write(content)
    print("Fixed utility.dart")

def fix_helpers_dart():
    path = os.path.join(workspace_root, "lib/src/operations/helpers.dart")
    with open(path, "r") as f:
        content = f.read()

    # Finding 5: broadcast3Shapes zero-length arrays
    old_b3 = """List<int> broadcast3Shapes(List<int> s1, List<int> s2, List<int> s3) {
  final len = math.max(s1.length, math.max(s2.length, s3.length));
  final common = List<int>.filled(len, 1);
  for (var i = 0; i < len; i++) {
    final dim1 = s1.length - 1 - i >= 0 ? s1[s1.length - 1 - i] : 1;
    final dim2 = s2.length - 1 - i >= 0 ? s2[s2.length - 1 - i] : 1;
    final dim3 = s3.length - 1 - i >= 0 ? s3[s3.length - 1 - i] : 1;

    final target = math.max(dim1, math.max(dim2, dim3));
    if (dim1 != target && dim1 != 1) {
      throw ArgumentError('Incompatible shapes for broadcasting');
    }
    if (dim2 != target && dim2 != 1) {
      throw ArgumentError('Incompatible shapes for broadcasting');
    }
    if (dim3 != target && dim3 != 1) {
      throw ArgumentError('Incompatible shapes for broadcasting');
    }
    common[len - 1 - i] = target;
  }
  return common;
}"""
    new_b3 = """List<int> broadcast3Shapes(List<int> s1, List<int> s2, List<int> s3) {
  final len = math.max(s1.length, math.max(s2.length, s3.length));
  final common = List<int>.filled(len, 1);
  for (var i = 0; i < len; i++) {
    final dim1 = s1.length - 1 - i >= 0 ? s1[s1.length - 1 - i] : 1;
    final dim2 = s2.length - 1 - i >= 0 ? s2[s2.length - 1 - i] : 1;
    final dim3 = s3.length - 1 - i >= 0 ? s3[s3.length - 1 - i] : 1;

    var target = 1;
    for (final d in [dim1, dim2, dim3]) {
      if (d != 1) {
        if (target == 1) {
          target = d;
        } else if (target != d) {
          throw ArgumentError('Incompatible shapes for broadcasting');
        }
      }
    }
    common[len - 1 - i] = target;
  }
  return common;
}"""
    assert old_b3 in content, "old_b3 not found"
    content = content.replace(old_b3, new_b3)

    with open(path, "w") as f:
        f.write(content)
    print("Fixed helpers.dart")

def fix_bitwise_dart():
    path = os.path.join(workspace_root, "lib/src/operations/math/bitwise.dart")
    with open(path, "r") as f:
        content = f.read()

    # Finding 9: bitwise.dart toList() conversions
    old_cast = """  // Upcast inputs if they do not match the resolved target integer type
  NDArray aCast = a;
  if (a.dtype != targetDType) {
    aCast = NDArray.fromList(a.toList(), a.shape, targetDType);
  }

  NDArray bCast = b;
  if (b.dtype != targetDType) {
    bCast = NDArray.fromList(b.toList(), b.shape, targetDType);
  }"""
    new_cast = """  // Upcast inputs if they do not match the resolved target integer type
  final NDArray aCast = a.dtype != targetDType ? castNDArray(a, targetDType) : a;
  final NDArray bCast = b.dtype != targetDType ? castNDArray(b, targetDType) : b;"""
    assert old_cast in content, "old_cast not found"
    content = content.replace(old_cast, new_cast)

    with open(path, "w") as f:
        f.write(content)
    print("Fixed bitwise.dart Finding 9")

fix_complex_dart()
fix_utility_dart()
fix_helpers_dart()
fix_bitwise_dart()
